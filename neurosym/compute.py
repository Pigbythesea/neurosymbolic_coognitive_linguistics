"""Execution utilities: immutable fold caches, explicit FP64 devices and timings.

No scientific hyperparameter, split or observation support is chosen here.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
from scipy import linalg

from .io import object_hash, read_json, save_json
from .storage import DiskBudget
from .runtime import deadline, process_lock


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_identity(data, model=None):
    identity = {
        'version': 1, 'semantics': data.semantics.build_hash,
        'contract': object_hash(data.reader.contract), 'spatial': object_hash(data.spatial.identity),
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'h5py', 'torch')},
        'code': {n: file_hash(Path(__file__).parent / n) for n in
                 ('compute.py', 'analysis_data.py', 'dataset.py', 'semantic_features.py',
                  'model_features.py', 'temporal.py', 'encoding_support.py', 'pca.py', 'runtime.py')},
    }
    if model:
        identity['model'] = {'id': model, 'alignment': data.model(model).aligned_run}
    return identity


class FoldCache:
    """Publish complete directories atomically; readers never see partial files.

    A per-entry OS lock coordinates first writers across workers. Payloads are
    numeric/JSON, never executable pickle objects. Locks survive as empty files;
    ownership is released by the OS, including after process death.
    """
    def __init__(self, root, *, require=False, policy=None):
        self.root = Path(root).resolve()
        self.require = require
        self.hits = self.misses = 0
        self.policy = policy
        self.budget = (DiskBudget(self.root, policy['cache_gib'] * 2**30,
                                 policy['cache_entry_gib'] * 2**30, policy['minimum_free_gib'] * 2**30)
                       if policy else None)

    def entry(self, kind, identity, build):
        if not kind.replace('-', '').isalnum():
            raise ValueError('Invalid cache namespace.')
        key = object_hash(identity)
        path = self.root / kind / key
        if path.exists():
            self.check(path, identity)
            self.hits += 1
            return path
        if self.require:
            raise FileNotFoundError('Preparation required for missing cache: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        # mkdir's normal inherited ACL matters on Windows: mkdtemp's private
        # ACL would make a published cache unreadable to a later agent/user.
        with process_lock(path.parent / ('.producer-' + key)):
            if path.exists():
                self.check(path, identity)
                self.hits += 1
                return path
            deadline.check()
            stage = path.parent / ('.building-' + uuid.uuid4().hex)
            reservation = self.budget.reserve(stage) if self.budget else nullcontext(None)
            with reservation as token:
                result = self._build(path, stage, identity, build, token)
            deadline.advance()
            deadline.check()  # published entries survive cooperative continuation
            return result

    def _build(self, path, stage, identity, build, token):
        stage.mkdir(mode=0o777)
        try:
            stage = stage.resolve()
            if not stage.is_relative_to(self.root):
                raise ValueError('Cache staging directory escaped its root.')
            build(stage)
            files = {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}
            if not files or any(p.is_dir() for p in stage.iterdir()):
                raise ValueError('Cache builders must write a nonempty flat file inventory.')
            save_json(stage / 'receipt.json', {'identity': identity, 'files': files})
            if self.budget:
                self.budget.publish(token, stage, path, lambda: self.check(path, identity))
            else:
                try:
                    os.rename(stage, path)
                except OSError:
                    if not path.exists():
                        raise
                    self.check(path, identity)
        finally:
            if stage.exists():
                if not stage.resolve().is_relative_to(self.root):
                    raise ValueError('Refusing cleanup outside the cache root.')
                for child in stage.iterdir():
                    child.unlink()
                stage.rmdir()
        self.misses += 1
        return path

    @staticmethod
    def check(path, identity):
        receipt = read_json(path / 'receipt.json')
        if receipt['identity'] != identity:
            raise ValueError('Cache identity mismatch: ' + str(path))
        for name, digest in receipt['files'].items():
            if Path(name).name != name or file_hash(path / name) != digest:
                raise ValueError('Cache payload changed: ' + str(path / name))

    def arrays(self, kind, identity, build):
        def write(path):
            np.savez(path / 'arrays.npz', **build())
        path = self.entry(kind, identity, write)
        with np.load(path / 'arrays.npz', allow_pickle=False) as file:
            return {k: file[k] for k in file.files}


class FP64:
    """NumPy/SciPy CPU or explicit torch CUDA; no reduced precision/fallback."""
    def __init__(self, device='cpu'):
        self.device = str(device)
        self.gpu = self.device.startswith('cuda')
        if self.device != 'cpu' and not self.gpu:
            raise ValueError('Encoding supports cpu or cuda[:index].')
        if self.gpu:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested for encoding but unavailable.')
            self.torch = torch

    def array(self, value):
        if self.gpu:
            return self.torch.as_tensor(value, dtype=self.torch.float64, device=self.device)
        return np.asarray(value, dtype=np.float64)

    def numpy(self, value):
        return value.detach().cpu().numpy() if self.gpu else np.asarray(value)

    def product(self, a, b):
        return self.array(a) @ self.array(b)

    def concatenate(self, values, axis=1):
        values = [self.array(v) for v in values]
        return self.torch.cat(values, dim=axis) if self.gpu else np.concatenate(values, axis=axis)

    def eigh(self, value):
        return self.torch.linalg.eigh(value) if self.gpu else linalg.eigh(value, check_finite=False)

    def factor(self, kernel, alpha):
        kernel = kernel.clone() if self.gpu else kernel.copy()
        if self.gpu:
            kernel.diagonal().add_(alpha)
            return self.torch.linalg.cholesky(kernel)
        kernel.flat[::len(kernel) + 1] += alpha
        return linalg.cho_factor(kernel, lower=True, check_finite=False)

    def solve(self, factor, target):
        target = self.array(target)
        return self.torch.cholesky_solve(target, factor) if self.gpu else linalg.cho_solve(factor, target, check_finite=False)

    def nonnegative(self, value):
        return value.clamp_min(0) if self.gpu else np.maximum(value, 0)


class Timings:
    def __init__(self, device='cpu'):
        self.device, self.seconds = str(device), {}
        self.started = time.perf_counter()

    def synchronize(self):
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name):
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - start

    def report(self):
        result = {'device': self.device, 'wall_seconds_this_process': time.perf_counter() - self.started,
                  'phase_seconds': self.seconds}
        if self.device.startswith('cuda'):
            import torch
            result.update(gpu=torch.cuda.get_device_name(self.device),
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        if os.name != 'nt':
            import resource
            result['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return result
