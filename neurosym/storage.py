"""Bounded resident reuse and accountable storage for regenerable intermediates."""
from collections import OrderedDict
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import time
import uuid

from .io import read_json, save_json


def require_free_space(path, write_bytes, reserve_bytes):
    if shutil.disk_usage(path).free < int(write_bytes) + int(reserve_bytes):
        raise RuntimeError('Insufficient filesystem free space for this output and configured reserve. Filesystem free space does not certify a project quota.')


def storage_report(root, policy, *, scan=False):
    """Read-only inventory; never infer a project quota from filesystem capacity."""
    root = Path(root).resolve()
    cache = root / 'data/processed/analysis-cache-v2'
    ledger = read_json(cache / '.budget.json') if (cache / '.budget.json').exists() else None
    result = {'cache': str(cache), 'policy': policy, 'ledger': ledger,
              'filesystem_free_bytes': shutil.disk_usage(root).free,
              'scope': 'Filesystem free space is not the PI/project quota. No files deleted or jobs inspected.'}
    if scan:
        result['directories'] = {}
        for name in ('data/processed/analysis-cache', 'data/processed/analysis-cache-v2', 'data/analysis',
                     'data/processed/semantics-reviewed/builds', 'artifacts'):
            directory = root / name
            files = [p.stat().st_size for p in directory.rglob('*') if p.is_file()] if directory.exists() else []
            result['directories'][name] = {'bytes': sum(files), 'files': len(files)}
    return result


def size_bytes(value):
    if isinstance(value, dict):
        return sum(size_bytes(k) + size_bytes(v) for k, v in value.items())
    if isinstance(value, str):
        return 4 * len(value)
    if isinstance(value, (int, float, bool)):
        return 32
    if isinstance(value, (tuple, list)):
        return sum(size_bytes(v) for v in value)
    if hasattr(value, 'numel'):
        return value.numel() * value.element_size()
    return getattr(value, 'nbytes', 0)


class ResidentCache:
    """Process-local LRU. Eviction removes references, never scientific files."""
    def __init__(self, max_bytes):
        self.max_bytes, self.bytes = int(max_bytes), 0
        self.items = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key, build):
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key][0]
        self.misses += 1
        value = build()
        size = size_bytes(value)
        if size <= self.max_bytes:
            while self.items and self.bytes + size > self.max_bytes:
                _, (_, removed) = self.items.popitem(last=False)
                self.bytes -= removed
            self.items[key] = value, size
            self.bytes += size
        return value

    def clear(self):
        self.items.clear()
        self.bytes = 0


@contextmanager
def ledger_lock(root):
    """OS-released lock, including process termination; never recover by PID guess."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.budget.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            deadline = time.monotonic() + 10
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Cache budget ledger is busy.')
                    time.sleep(.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


class DiskBudget:
    """Account published cache and reserve space for concurrent writers.

    A killed writer leaves an explicit reservation, not an unaccounted directory.
    Oversized entries are rejected at publication; this is not a filesystem quota.
    Nothing deletes source data, fitted outputs, active or prepared dependencies.
    """
    def __init__(self, root, limit_bytes, entry_bytes, free_bytes):
        self.root = Path(root).resolve()
        self.limit, self.entry, self.free = map(int, (limit_bytes, entry_bytes, free_bytes))
        if not 0 < self.entry <= self.limit or self.free < 0:
            raise ValueError('Invalid persistent cache budget.')

    def state(self):
        path = self.root / '.budget.json'
        if path.exists():
            return read_json(path)
        # Existing published entries are accounted for on first adoption.
        total, count = 0, 0
        for p in self.root.glob('*/*/receipt.json'):
            total += sum(f.stat().st_size for f in p.parent.iterdir() if f.is_file())
            count += 1
        if list(self.root.glob('*/.building-*')):
            raise RuntimeError('Untracked cache staging directories require inspection before adopting a budget.')
        return {'published_bytes': total, 'published_entries': count, 'reservations': {}}

    @contextmanager
    def reserve(self, stage):
        token = uuid.uuid4().hex
        with ledger_lock(self.root):
            state = self.state()
            allocated = state['published_bytes'] + sum(r['bytes'] for r in state['reservations'].values())
            if allocated + self.entry > self.limit or shutil.disk_usage(self.root).free < self.free + self.entry:
                raise RuntimeError('Analysis cache budget/free-space reserve exhausted. Inspect cache-status; do not delete inputs or active dependencies.')
            state['reservations'][token] = {'bytes': self.entry, 'stage': str(stage), 'pid': os.getpid(),
                                            'job': os.environ.get('SLURM_JOB_ID')}
            save_json(self.root / '.budget.json', state)
        try:
            yield token
        finally:
            # The caller has already removed its stage or published the result.
            if not Path(stage).exists():
                with ledger_lock(self.root):
                    state = self.state()
                    state['reservations'].pop(token, None)
                    save_json(self.root / '.budget.json', state)

    def publish(self, token, stage, destination, check):
        size = sum(p.stat().st_size for p in stage.iterdir() if p.is_file())
        if size > self.entry:
            raise RuntimeError('Cache entry exceeds its reserved staging budget; adjust execution storage policy explicitly.')
        with ledger_lock(self.root):
            state = self.state()
            if token not in state['reservations']:
                raise ValueError('Cache publication has no reservation.')
            if destination.exists():
                check()
            else:
                os.rename(stage, destination)
                state['published_bytes'] += size
                state['published_entries'] += 1
                state['reservations'].pop(token)
            save_json(self.root / '.budget.json', state)
