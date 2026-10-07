"""Cooperative walltime boundaries and process-owned analysis locks.

Signals request a yield; only an explicit safe point raises it. No scheduler
commands, lock deletion, or remote operations are performed by this module.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import time


class YieldRequested(Exception):
    """Exit 75 means saved work can be continued in another allocation."""


class Deadline:
    def __init__(self):
        self.end = None
        self.requested = False
        self.progress = 0

    def advance(self):
        self.progress += 1

    def install(self):
        limit = os.environ.get('NEUROSYM_WALL_SECONDS')
        end = os.environ.get('SLURM_JOB_END_TIME')
        reserve = int(os.environ.get('NEUROSYM_YIELD_RESERVE', '180'))
        if limit:
            self.end = time.monotonic() + max(0, float(limit) - reserve)
        if end and end.isdigit():
            actual = time.monotonic() + max(0, float(end) - time.time() - reserve)
            self.end = actual if self.end is None else min(self.end, actual)
        if os.environ.get('SLURM_JOB_ID'):
            for name in ('SIGUSR1', 'SIGTERM'):
                if hasattr(signal, name):
                    signal.signal(getattr(signal, name), self._signal)

    def _signal(self, *_):
        self.requested = True

    def due(self):
        return self.requested or (self.end is not None and time.monotonic() >= self.end)

    def check(self):
        if self.due():
            raise YieldRequested('Allocation boundary reached; resume saved work.')


deadline = Deadline()


@contextmanager
def process_lock(path, *, wait=True):
    """OS advisory lock released on exit/death; keep its inode permanently.

The filesystem must support cross-node advisory locking (verify on cluster).
Never unlink this file: another process may already have opened its inode.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        acquired = False
        try:
            while not acquired:
                deadline.check()
                try:
                    if os.name == 'nt':
                        import msvcrt
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except (BlockingIOError, PermissionError, OSError) as error:
                    import errno
                    if error.errno not in (errno.EAGAIN, errno.EACCES, errno.EDEADLK):
                        raise
                    if not wait:
                        raise RuntimeError('Another process owns this analysis operation: ' + str(path)) from error
                    time.sleep(.2)
            yield
        finally:
            if acquired:
                if os.name == 'nt':
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def analysis_lock(path):
    # Existing presence-only locks need explicit inspection, never auto-recovery.
    path = Path(path)
    if path.exists():
        raise RuntimeError('Legacy interrupted lock requires inspection: ' + str(path))
    with process_lock(path.with_name(path.name + '.oslock'), wait=False):
        yield
