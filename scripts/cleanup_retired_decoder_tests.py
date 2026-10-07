"""Remove only inventoried unfinished pre-protocol-2 decoder tests, after qualification."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RETIRED_EXPERIMENT = 'b90d89f9131e821e133ad3f43965a789008af1ee79eb1f1b27bb56f2ba068596'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURMD_NODENAME', '').startswith('login'):
        raise RuntimeError('Run filesystem inspection/cleanup on an allocated compute node.')
    from neurosym.io import read_json, object_hash, save_json
    from neurosym.execution import code_identity
    audit = args.audit or root / 'artifacts/trace-audit-1024054'
    inventory = read_json(audit / 'cleanup-inventory.json')
    protected = set()
    for path in (root / 'data/analysis/execution').glob('*/*.json'):
        protected.update(read_json(path).get('results', []))
    if (root / 'artifacts/fit-reuse.json').exists():
        protected.update(v['path'] for v in read_json(root / 'artifacts/fit-reuse.json')['completed'].values())
    candidates = []
    for item in inventory:
        if item['complete'] or item['current_manifest_result'] or item['experiment_definition_hash'] != RETIRED_EXPERIMENT:
            continue
        relative = Path(item['path'])
        if relative.parent.as_posix() != 'data/analysis/decoder' or len(relative.name) != 64:
            continue
        path = (root / relative).resolve()
        if (path.parent != (root / 'data/analysis/decoder').resolve() or
                (root / relative).is_symlink() or path.name != relative.name):
            raise ValueError('Cleanup path escapes the exact decoder workspace.')
        if not path.exists():
            continue
        identity = read_json(path / 'identity.json')
        if (identity.get('experiment_definition_hash') != RETIRED_EXPERIMENT or
                object_hash(identity) != path.name or identity.get('options') != item['options'] or
                (path / 'complete.json').exists() or relative.as_posix() in protected or
                any(p.is_symlink() for p in path.rglob('*'))):
            raise ValueError('An obsolete-test candidate changed or is protected: ' + str(relative))
        files = [p for p in path.rglob('*') if p.is_file()]
        candidates.append({'path': relative.as_posix(), 'bytes': sum(p.stat().st_size for p in files),
                           'files': len(files), 'identity': identity})
    print(json.dumps({'mode': 'apply' if args.apply else 'inspection', 'directories': len(candidates),
        'bytes': sum(v['bytes'] for v in candidates), 'paths': [v['path'] for v in candidates]}, indent=2), flush=True)
    if not args.apply:
        return
    evidence = read_json(root / 'artifacts/trace-verification-cuda.json')
    if (evidence.get('status') != 'verified' or not evidence.get('complete_samples') or
            not evidence.get('geometry_verified') or evidence.get('code') != code_identity()):
        raise ValueError('Current full fitted-reference CUDA/geometry verification must pass before cleanup.')
    rows = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%A|%j'], check=True,
        capture_output=True, text=True).stdout.splitlines()
    if any(row.split('|', 1)[0] != os.environ['SLURM_JOB_ID'] and
           row.split('|', 1)[1].startswith(('ns-', 'neurosym')) for row in rows):
        raise RuntimeError('Project jobs must be idle before cleanup.')
    journal = root / 'artifacts' / ('retired-tests-cleanup-' + str(time.time_ns()))
    journal.mkdir()
    save_json(journal / 'inventory.json', {'candidates': candidates, 'qualification': evidence, 'removed': []})
    # Retain small provenance/selection/history JSON, not obsolete tensor states.
    with zipfile.ZipFile(journal / 'metadata.zip', 'x', compression=zipfile.ZIP_DEFLATED) as file:
        for item in candidates:
            path = root / item['path']
            for metadata in path.rglob('*.json'):
                file.write(metadata, metadata.relative_to(root).as_posix())
    removed = []
    for item in candidates:
        path = (root / item['path']).resolve()
        if path.parent != (root / 'data/analysis/decoder').resolve() or (path / 'complete.json').exists():
            raise ValueError('Cleanup target changed after inspection.')
        shutil.rmtree(path)
        removed.append(item['path'])
        save_json(journal / 'inventory.json', {'candidates': candidates, 'qualification': evidence, 'removed': removed})
    print('Retired tests removed; current fits, all reference traces, encoding objects, inputs and caches retained.', flush=True)


if __name__ == '__main__':
    main()
