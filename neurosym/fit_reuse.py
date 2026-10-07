"""Explicit provenance-preserving reuse of qualified completed protocol-2 fits."""
from pathlib import Path

from .compute import file_hash
from .io import object_hash, read_json


SOURCE_MANIFEST = '0fdc388fef401ad513c9e2d853fc2601d4915711aaacf3b95534bfffcddde2c6'
KINDS = {'decoder', 'decoder-selection', 'encoding'}


def same_measurement(first, second):
    return {k: v for k, v in first.items() if k != 'code'} == {k: v for k, v in second.items() if k != 'code'}


def inside(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root / 'data' / 'analysis') or candidate == root / 'data' / 'analysis':
        raise ValueError('Fit reuse path escapes the analysis workspace.')
    return candidate


def reused_directory(root, identity):
    path = Path(root) / 'artifacts/fit-reuse.json'
    if identity['kind'] not in KINDS or not path.exists():
        return None
    registry = read_json(path)
    if registry['source_manifest'] != SOURCE_MANIFEST or registry['target_code'] != identity['code']:
        return None  # A stale registry never authorizes a new implementation.
    entry = registry['completed'].get(object_hash(identity))
    if entry is None:
        return None
    directory = inside(root, entry['path'])
    if file_hash(directory / 'complete.json') != entry['complete_sha256']:
        raise ValueError('Reused fit receipt changed.')
    old = read_json(directory / 'identity.json')
    complete = read_json(directory / 'complete.json')
    if (object_hash(old) != directory.name or complete['identity'] != old or
            not same_measurement(old, identity)):
        raise ValueError('Reused fit differs in data, scientific settings or dependencies.')
    # The original directory and identity are returned unchanged. New execution
    # receipts visibly refer to that original fitted parent, not a relabelled fit.
    return directory, old


def compatible_codes(root, first, second):
    """A comparison may cross code versions only through qualified fit reuse."""
    if first['code'] == second['code']:
        return True
    registry_path = Path(root) / 'artifacts/fit-reuse.json'
    if not registry_path.exists():
        return False
    registry = read_json(registry_path)
    from .analysis_runs import ANALYSIS_MODULES
    current = {name: file_hash(Path(__file__).parent / name) for name in ANALYSIS_MODULES}
    if registry['target_code'] != current:
        return False
    for identity in (first, second):
        if identity['code'] == current:
            continue
        if reused_directory(root, {**identity, 'code': current}) is None:
            return False
    return True


def prepared_cache_identity(cache_root, kind, identity):
    """Look up a qualified old numerical cache, never publish under its identity."""
    if kind not in {'projector', 'windows', 'pca-story-statistics'}:
        return None
    root = Path(cache_root).resolve().parents[2]
    path = root / 'artifacts/fit-reuse.json'
    if not path.exists():
        return None
    registry = read_json(path)
    from .analysis_runs import ANALYSIS_MODULES
    current = {name: file_hash(Path(__file__).parent / name) for name in ANALYSIS_MODULES}
    if registry.get('source_manifest') != SOURCE_MANIFEST or registry['target_code'] != current:
        return None
    import copy
    result = copy.deepcopy(identity)
    inputs = result.get('inputs', result.get('projector', {}).get('inputs'))
    if not isinstance(inputs, dict) or 'code' not in inputs:
        return None
    if any(current.get(name) != value for name, value in inputs['code'].items()):
        return None
    inputs['code'] = {name: registry['source_code'][name] for name in inputs['code']}
    return result
