"""Collect real fitted evidence before optimizing trace execution/storage.

Run on an allocated compute node with the project idle. Reads the pinned
installation and prepared observation caches; writes only a new artifacts/
trace-audit-<job> directory and ZIP. Never fits, resumes or deletes anything.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import zipfile


ROOT = Path('/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics')
MANIFEST = '0fdc388fef401ad513c9e2d853fc2601d4915711aaacf3b95534bfffcddde2c6'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inside(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError('Path outside project: ' + str(path))
    return path


def read(path):
    return json.loads(inside(path).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def copy(source, target):
    source = inside(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def operator(example):
    ast = example['inputs']['ast']
    return ast['task'] + (':' + ','.join(s['op'] for s in ast['steps']) if ast['task'] == 'compose' else '')


def choose_sources(examples):
    """Whole real sources: operator coverage plus the largest candidate workload."""
    groups = defaultdict(list)
    for example in examples:
        groups[example['source_id']].append(example)
    coverage = {s: {operator(e) for e in rows} for s, rows in groups.items()}
    missing = set().union(*coverage.values())
    selected = []
    while missing:
        source = min(groups, key=lambda s: (-len(coverage[s] & missing), len(groups[s]), s))
        if not coverage[source] & missing:
            raise ValueError('Unable to cover observed operators.')
        selected.append(source)
        missing -= coverage[source]
    largest = max(groups, key=lambda s: (sum(len(e['inputs']['candidates']) for e in groups[s]), s))
    return sorted(set([*selected, largest])), groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or socket.gethostname().startswith('login') or os.name == 'nt':
        raise RuntimeError('Use the supplied compute-node batch job.')
    if ROOT.resolve(strict=True) != ROOT or Path.cwd().resolve() != ROOT:
        raise ValueError('Unexpected workspace.')
    if args.manifest != MANIFEST:
        raise ValueError('This collection is pinned to the measured protocol-2 manifest.')
    queue = subprocess.run(['squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%i|%j'],
                           check=True, capture_output=True, text=True).stdout
    active = [line for line in queue.splitlines() if '|' in line and
              line.split('|')[0].strip() != job and
              line.split('|')[1].strip().startswith(('ns-', 'neurosym-'))]
    if active:
        raise RuntimeError('Project workers must be idle while copying trace evidence: ' + repr(active))
    installed = read(ROOT / '.analysis-source.json')
    code = inside(ROOT / installed['code_root'])
    if not code.is_relative_to(ROOT / 'analysis_code'):
        raise ValueError('Expected an isolated installed snapshot.')
    sys.path.insert(0, str(code))
    import h5py
    import numpy as np
    import torch
    from neurosym.analysis_data import AnalysisData, SiteProjector
    from neurosym.decoder_fit import examples_and_windows, observation_options
    from neurosym.execution import code_identity
    from neurosym.io import object_hash

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.set_num_interop_threads(1)
    manifest_path = ROOT / 'artifacts/execution' / args.manifest / 'manifest.json'
    manifest = read(manifest_path)
    if object_hash({k: v for k, v in manifest.items() if k != 'content_hash'}) != args.manifest:
        raise ValueError('Manifest content changed.')
    if code_identity() != manifest['code']:
        raise ValueError('Installed code does not match the measured manifest.')
    data = AnalysisData(ROOT, build=inside(ROOT / manifest['build']), require_prepared=True)
    if data.config != manifest['analysis_config'] or data.compute_config != manifest['resources']:
        raise ValueError('Installed configuration differs from the measured run.')
    if (data.semantics.build_hash != manifest['semantic_build_hash'] or
            object_hash(data.reader.contract) != manifest['data_contract_hash'] or
            object_hash(data.spatial.identity) != manifest['spatial_hash']):
        raise ValueError('Data identity differs from the measured run.')
    output = ROOT / 'artifacts' / ('trace-audit-' + job)
    archive = output.with_suffix('.zip')
    if output.exists() or archive.exists():
        raise FileExistsError('Collection output already exists; retain it and use a new job.')
    output.mkdir()
    started = time.monotonic()
    report = {'manifest': args.manifest, 'job': job, 'host': socket.gethostname(),
              'mode': 'read existing fitted evidence; no fitting or deletion', 'fits': []}
    copy(Path(__file__), output / 'collector.py')
    execution = inside(ROOT / 'data/analysis/execution' / args.manifest)
    copy(manifest_path, output / 'manifest.json')
    copy(ROOT / '.analysis-source.json', output / 'analysis-source.json')
    for relative in manifest['code']:
        source = code / relative if relative.startswith('scripts/') else code / 'neurosym' / relative
        copy(source, output / 'reference-code' / source.relative_to(code))
    # Include the complete package: imported support modules are also needed
    # to inspect/replay the old implementation without modifying the live repo.
    for source in (code / 'neurosym').glob('*.py'):
        copy(source, output / 'reference-code/neurosym' / source.name)
    for source in execution.rglob('*.json'):
        copy(source, output / 'execution' / source.relative_to(execution))
    ledger_path = ROOT / 'artifacts/submissions' / args.manifest / 'dispatch.json'
    ledger = read(ledger_path)
    copy(ledger_path, output / 'dispatch.json')
    ids = sorted({a['task_id'].split('_')[0] for a in ledger['attempts'] if a.get('task_id')})
    accounting = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(ids),
        '--format=JobID%64,JobName%40,State,ElapsedRaw,TotalCPU,AllocCPUS,MaxRSS,ExitCode,NodeList'],
        check=True, capture_output=True, text=True)
    (output / 'accounting.txt').write_text(accounting.stdout, encoding='utf-8')
    for ident in ids:
        for source in (ROOT / 'logs').glob('analysis-' + ident + '_*.log'):
            copy(source, output / 'logs' / source.name)
    for pattern in ('p2-timing-controller-*.log', 'p2-derived-controller-*.log'):
        for source in (ROOT / 'logs').glob(pattern):
            copy(source, output / 'logs' / source.name)

    for worker in (102, 1220):
        entry = manifest['jobs'][manifest['workers'][worker]['indices'][0]]
        receipt = read(execution / (entry['id'] + '.json'))
        if (receipt.get('status') != 'complete' or receipt.get('job_id') != entry['id'] or
                receipt.get('manifest_hash') != args.manifest or len(receipt.get('results', [])) != 1):
            raise ValueError('A completed reference fit is required: ' + str(worker))
        run = inside(ROOT / receipt['results'][0])
        completed = read(run / 'complete.json')
        identity = completed['identity']
        if identity['options'] != entry['options'] or identity['semantic_build_hash'] != data.semantics.build_hash:
            raise ValueError('Completed fit identity mismatch.')
        destination = output / 'fits' / run.name
        destination.mkdir(parents=True)
        for name in ('identity.json', 'complete.json', 'decoder.json', 'selection.json', 'runtime.json',
                     'weights.pt', 'predictions.jsonl.gz', 'mismatched-predictions.jsonl.gz'):
            copy(run / name, destination / name)
        for name in ('projector.json', 'projector.npz'):
            copy(run / 'projector' / name, destination / 'projector' / name)
        print('COLLECT cached actual observations:', worker, flush=True)
        options, split = identity['options'], completed['partition']
        projector = data.projector(options['modality'], split['train'], **observation_options(options))
        fitted_projector = SiteProjector.load(run / 'projector')
        if (not np.array_equal(projector.assignment, fitted_projector.assignment) or
                projector.training_stories != fitted_projector.training_stories or
                projector.components != fitted_projector.components or
                len(projector.parameters) != len(fitted_projector.parameters) or
                any(not np.array_equal(a, b) for first, second in
                    zip(projector.parameters, fitted_projector.parameters, strict=True)
                    for a, b in zip(first, second, strict=True))):
            raise ValueError('Cached projector differs from the fitted parent.')
        examples, windows, _ = examples_and_windows(data, split['test'], projector, options)
        names = sorted(windows)
        shape = tuple(read(run / 'decoder.json')['shape'])
        if any(value.shape[1:] != shape or not np.isfinite(value).all() for value in windows.values()):
            raise ValueError('Prepared observations differ from fitted shape or contain nonfinite values.')
        # Keep all actual heldout windows and query IDs for full-fit checks.
        # Inputs/answers stay in the already-pinned local accepted corpus.
        np.savez_compressed(destination / 'windows.npz', **{str(i): windows[s] for i, s in enumerate(names)})
        save(destination / 'windows.json', names)
        save(destination / 'query-ids.json', [e['query_id'] for e in examples])
        selected, groups = choose_sources(examples)
        save(destination / 'sample-sources.json', selected)
        with gzip.open(destination / 'sample-examples.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as stream:
            for source in selected:
                for example in groups[source]:
                    stream.write(json.dumps(example, ensure_ascii=False) + '\n')
        trace_start = time.monotonic()
        sample = destination / 'traces-sample.h5'
        with h5py.File(run / 'traces.h5', 'r') as original, h5py.File(sample, 'x') as target:
            if not original.attrs.get('complete', False) or original.attrs.get('run_identity') != object_hash(identity):
                raise ValueError('Reference trace is incomplete or has a different identity.')
            if set(original.keys()) != {e['query_id'] for e in examples}:
                raise ValueError('Current accepted query support differs from the completed trace.')
            for key, value in original.attrs.items():
                target.attrs[key] = value
            target.attrs['audit_subset'] = True
            for source in selected:
                print('COLLECT trace source:', worker, source, flush=True)
                for example in groups[source]:
                    key = example['query_id']
                    if key not in original:
                        raise KeyError('Missing real trace query: ' + key)
                    for repeat in range(len(windows[source])):
                        group = original[key + '/' + str(repeat)]
                        if not group.attrs.get('prediction_complete', False) or group.attrs['source_id'] != source:
                            raise ValueError('Reference source/repeat is incomplete.')
                    original.copy(key, target, name=key)
            stats = Counter()
            def count_objects(name, obj):
                if isinstance(obj, h5py.Dataset):
                    stats['datasets'] += 1
                    stats['logical_array_bytes'] += obj.size * obj.dtype.itemsize
                    stats['stored_array_bytes'] += obj.id.get_storage_size()
                else:
                    stats['groups'] += 1
            target.visititems(count_objects)
        info = {'worker': worker, 'run': run.relative_to(ROOT).as_posix(), 'weights_sha256': digest(run / 'weights.pt'),
                'sources': len(names), 'queries': len(examples), 'sample_sources': selected,
                'sample_queries': sum(len(groups[s]) for s in selected), 'operators': sorted({operator(e) for e in examples}),
                'original_trace_bytes': (run / 'traces.h5').stat().st_size, 'sample_trace_bytes': sample.stat().st_size,
                'sample_copy_seconds': time.monotonic() - trace_start, 'sample_object_counts': dict(stats),
                'sample_note': 'HDF5 group copying can expand cross-query hard links. Counts describe this subset copy, not the original whole-file payload or export throughput.'}
        report['fits'].append(info)
        print('REFERENCE COLLECTED', json.dumps(info), flush=True)

    # Filesystem inventory only: no automatic deletion or inference of safety
    # merely from a file being old/unreferenced by this particular manifest.
    protected = set()
    for source in execution.glob('*.json'):
        value = read(source)
        protected.update(value.get('results', []))
    inventory = []
    analysis_root = inside(ROOT / 'data/analysis')
    measured_options = [manifest['jobs'][i]['options'] for w in (102, 1220)
                        for i in manifest['workers'][w]['indices']]
    for kind in sorted(analysis_root.iterdir()):
        if not kind.is_dir() or kind.is_symlink():
            continue
        for path in sorted(kind.iterdir()):
            if not path.is_dir() or path.is_symlink():
                continue
            files = [p for p in path.rglob('*') if p.is_file() and not p.is_symlink()]
            total = sum(inside(p).stat().st_size for p in files)
            relative = path.relative_to(ROOT).as_posix()
            identity = read(path / 'identity.json') if (path / 'identity.json').is_file() else {}
            current_modules = bool(identity.get('code')) and all(
                manifest['code'].get(key) == value for key, value in identity.get('code', {}).items())
            if (kind.name == 'decoder' and not (path / 'complete.json').exists() and current_modules and
                    identity.get('options') in measured_options and (path / 'refit.pt').is_file()):
                for name in ('identity.json', 'decoder.json', 'selection.json', 'weights.pt', 'refit.pt'):
                    if (path / name).is_file():
                        copy(path / name, output / 'partial-fits' / path.name / name)
            inventory.append({'path': relative, 'bytes': total, 'files': len(files),
                'complete': (path / 'complete.json').is_file(), 'has_weights': (path / 'weights.pt').is_file(),
                'has_checkpoint': (path / 'refit.pt').is_file(), 'current_manifest_result': relative in protected,
                'options': identity.get('options'), 'semantic_build_hash': identity.get('semantic_build_hash'),
                'experiment_definition_hash': identity.get('experiment_definition_hash'),
                'matches_current_modules': current_modules,
                'cleanup': 'retain pending provenance/dependency review'})
    save(output / 'cleanup-inventory.json', inventory)
    report['collection_seconds'] = time.monotonic() - started
    report['prepared_cache_hits'] = data.cache.hits
    report['prepared_cache_misses'] = data.cache.misses
    report['analysis_bytes'] = sum(v['bytes'] for v in inventory)
    save(output / 'report.json', report)
    checksums = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in output.rglob('*') if p.is_file()}
    save(output / 'checksums.json', checksums)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zipped:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                zipped.write(path, path.relative_to(output).as_posix())
    print('TRACE AUDIT READY', json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
          'sha256': digest(archive), 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
