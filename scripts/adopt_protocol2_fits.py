"""Inspect/reuse completed protocol-2 fits; never relabel old checkpoints."""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--installed-code', action='store_true',
        help='Run this maintenance helper against the unchanged installed analysis code.')
    args = parser.parse_args()
    root = args.root.resolve()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURMD_NODENAME', '').startswith('login'):
        raise RuntimeError('Run inspection/adoption on an allocated compute node.')
    import json
    if args.installed_code:
        installed = json.loads((root / '.analysis-source.json').read_text())
        code_root = (root / installed['code_root']).resolve()
        if code_root.parent != root / 'analysis_code' or not (code_root / 'neurosym/execution.py').is_file():
            raise ValueError('Installed analysis code must be inside the project analysis_code directory.')
        sys.path.insert(0, str(code_root))
    import torch
    from neurosym.analysis_data import AnalysisData
    from neurosym.analysis_runs import run_identity, write_report
    from neurosym.compute import file_hash
    from neurosym.execution import code_identity
    from neurosym.fit_reuse import SOURCE_MANIFEST, KINDS, same_measurement, inside
    from neurosym.io import object_hash, read_json, immutable_json
    from neurosym.runtime import analysis_lock
    source = read_json(root / 'artifacts/execution' / SOURCE_MANIFEST / 'manifest.json')
    if object_hash({k: v for k, v in source.items() if k != 'content_hash'}) != SOURCE_MANIFEST:
        raise ValueError('Source protocol manifest changed.')
    data = AnalysisData(root)
    current = code_identity()
    changed = {'analysis_runs.py', 'decoder_fit.py', 'decoder_minibatch.py', 'compute.py', 'geometry.py', 'study_reports.py', 'encoding.py', 'execution.py'}
    for name, digest in source['code'].items():
        if name.endswith('.py') and not name.startswith('scripts/') and name not in changed and current.get(name) != digest:
            raise ValueError('An unqualified scientific implementation changed: ' + name)
    if args.apply:
        # All mutation is user-run, after real fitted-reference CUDA qualification.
        evidence = read_json(root / 'artifacts/trace-verification-cuda.json')
        if (evidence.get('status') != 'verified' or evidence.get('code') != current or
                not evidence.get('complete_samples') or not evidence.get('geometry_verified')):
            raise ValueError('Current real fitted-reference CUDA verification is required.')
        import subprocess
        rows = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%A|%j'], check=True, capture_output=True, text=True).stdout.splitlines()
        busy = [row for row in rows if row.split('|', 1)[0] != os.environ['SLURM_JOB_ID'] and
                row.split('|', 1)[1].startswith(('ns-', 'neurosym'))]
        if busy:
            raise RuntimeError('Project jobs must be idle for fit adoption: ' + repr(busy))
    completed, partial = {}, []
    target_code = None
    def import_fitted(directory, destination, old, new, pairing):
        d = read_json(directory / 'decoder.json')
        files = ['decoder.json', 'weights.pt', 'selection.json']
        if d['family'] != 'prior':
            files += ['projector/projector.json', 'projector/projector.npz']
        hashes = {name: file_hash(directory / name) for name in files}
        partial.append({'source': directory.relative_to(root).as_posix(),
            'destination': destination.relative_to(root).as_posix(), 'files': hashes})
        if args.apply:
            inside(root, destination.relative_to(root))
            with analysis_lock(destination / 'RUNNING.lock'):
                immutable_json(destination / 'identity.json', new)
                if (destination / 'fitted.json').exists():
                    return
                for name in files:
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(directory / name, target)
                write_report(destination / 'fitted.json', {'run_identity': object_hash(new), 'files': hashes,
                    'training_pairing': pairing,
                    'reused_from': {'identity': object_hash(old), 'path': directory.relative_to(root).as_posix(),
                                    'weights_sha256': hashes['weights.pt'], 'source_manifest': SOURCE_MANIFEST}})
    for kind in sorted(KINDS):
        for directory in sorted((root / data.config['output'] / kind).glob('*')):
            if not (directory / 'identity.json').is_file():
                continue
            old = read_json(directory / 'identity.json')
            if not old.get('code') or any(source['code'].get(k) != v for k, v in old['code'].items()):
                continue
            if directory.name != object_hash(old):
                raise ValueError('Original fitted identity changed: ' + str(directory))
            new = run_identity(data, kind, old['options'])
            target_code = new['code']
            if not same_measurement(old, new):
                raise ValueError('Data/configuration/package drift: ' + str(directory))
            destination = root / data.config['output'] / kind / object_hash(new)
            if (directory / 'complete.json').exists():
                receipt = read_json(directory / 'complete.json')
                if receipt['identity'] != old:
                    raise ValueError('Original completion receipt has another identity.')
                if kind == 'decoder' and receipt.get('traces_exported'):
                    # Re-export the measured context parents with identical
                    # weights, providing full production export/geometry timing.
                    # Retain their old trace files as immutable references.
                    import_fitted(directory, destination, old, new, receipt['training_pairing'])
                    continue
                completed[object_hash(new)] = {'path': directory.relative_to(root).as_posix(),
                    'complete_sha256': file_hash(directory / 'complete.json')}
            elif kind == 'decoder' and all((directory / n).is_file() for n in ['decoder.json', 'weights.pt', 'refit.pt', 'selection.json']):
                d, selection = read_json(directory / 'decoder.json'), read_json(directory / 'selection.json')
                checkpoint = torch.load(directory / 'refit.pt', map_location='cpu', weights_only=True)
                epochs = selection['refit_epochs']
                if checkpoint['completed'] != epochs or checkpoint['pending'] is not None or d['fit']['best_epoch'] != epochs:
                    continue  # Optimizer-in-progress states are not migrated.
                weights = torch.load(directory / 'weights.pt', map_location='cpu', weights_only=True)
                if set(weights) != set(checkpoint['model']) or any(not torch.equal(v, checkpoint['model'][k]) for k, v in weights.items()):
                    raise ValueError('Published weights differ from the completed refit checkpoint.')
                import_fitted(directory, destination, old, new, checkpoint['identity']['settings']['pairing'])
    if target_code is None:
        raise ValueError('No compatible completed protocol-2 fits found.')
    report = {'source_manifest': SOURCE_MANIFEST, 'source_code': source['code'], 'target_code': target_code, 'completed': completed,
              'finished_refits_imported': partial, 'applied': args.apply,
              'maintenance_helper': {'sha256': file_hash(Path(__file__)),
                  'installed_code': args.installed_code,
                  'qualification_code': current}}
    if args.apply:
        write_report(root / 'artifacts/fit-reuse.json', report)
    print(json.dumps(report, indent=2))
    print('FIT REUSE SUMMARY', json.dumps({'applied': args.apply,
        'completed_by_kind': {kind: sum(Path(v['path']).parent.name == kind for v in completed.values())
            for kind in sorted(KINDS)}, 'finished_refits': len(partial)}), flush=True)


if __name__ == '__main__':
    main()
