"""Concrete preparation/fit jobs with pinned inputs and dependency receipts.

The manifest covers the expensive fitted parents for the declared program.
Geometry/comparison commands consume their recorded output paths; generating
this manifest neither submits jobs nor evaluates the final story.
"""
from pathlib import Path

from .analysis_runs import ANALYSIS_MODULES, partition, write_report
from .compute import Timings, file_hash
from .experiment_plan import experiment_plan
from .runtime import analysis_lock as exclusive_run, process_lock, deadline, YieldRequested
from .io import object_hash, read_json


def code_identity():
    result = {n: file_hash(Path(__file__).parent / n) for n in
              [*ANALYSIS_MODULES, 'experiment_plan.py']}
    result.update({n: file_hash(Path(__file__).resolve().parents[1] / n) for n in
                  ['scripts/run_analysis.py', 'scripts/verify_compute.py', 'scripts/submit_analysis.py',
                   'scripts/run_manifest_array.py', 'scripts/run_analysis_array.sbatch']})
    return result


def worker_inventory(jobs, limit):
    """Group related logical fits without removing any condition or seed."""
    if not isinstance(limit, int) or limit < 1:
        raise ValueError('Positive worker fit limit required.')
    groups = {}
    for index, job in enumerate(jobs):
        options = job['options']
        if job['kind'] == 'encoding':
            # Same predictors/selection grid across participants. Visit each
            # inner fold across this panel before moving to the next fold.
            group = {k: v for k, v in options.items() if k != 'subject'}
        elif job['kind'] == 'decoder':
            group = {k: v for k, v in options.items() if k not in {'seed', 'family', 'retrain_null'}}
        else:
            group = {'individual_job': job['id']}
        key = object_hash({'kind': job['kind'], 'resource': job['resource'], 'group': group,
                           'dependencies': job['dependencies']})
        groups.setdefault(key, []).append(index)
    workers = []
    for indices in groups.values():
        for offset in range(0, len(indices), limit):
            current = indices[offset:offset + limit]
            first = jobs[current[0]]
            workers.append({'resource': first['resource'], 'indices': current,
                            'dependencies': first['dependencies']})
    return workers


def execution_manifest(data, *, phase='development', encoding_device='cuda', decoder_device='cuda'):
    if phase not in {'development', 'final'} or {encoding_device, decoder_device} - {'cpu', 'cuda'}:
        raise ValueError('Unknown execution phase/device.')
    resolved = experiment_plan(data)
    plan = resolved['definitions']
    resources = read_json(data.root / 'configs/compute.json')
    if resources['format_version'] != 1:
        raise ValueError('Unsupported resource configuration.')
    main_folds = plan['development_folds'] if phase == 'development' else [plan['final_fold']]
    contexts = plan['geometry']['context_folds'] if phase == 'development' else []
    observations = [{'modality': 'brain', 'subject': s} for s in plan['subjects']]
    observations += [{'modality': 'model', 'model': m['id'], 'layer': layer}
                     for m in resolved['models'] for layer in m['descriptive_layers']]
    jobs, by_id = [], {}

    def add(kind, options, resource, dependencies=(), purpose=None):
        identity = {'kind': kind, 'options': options}
        key = object_hash(identity)
        if key in by_id:
            existing = jobs[by_id[key]]
            if existing['dependencies'] != list(dependencies) or existing['resource'] != resource:
                raise ValueError('Conflicting execution dependencies.')
            return key
        job = {'id': key, **identity, 'resource': resource, 'dependencies': list(dependencies), 'purpose': purpose}
        by_id[key] = len(jobs)
        jobs.append(job)
        return key

    # One CPU allocation per fold/seed processes all prior conditions. Its
    # identical fits are shared by the validated query-only fit cache.
    for fold in main_folds:
        for seed in plan['seeds']:
            conditions = [{**obs, 'fold': fold, 'seed': seed, 'family': 'prior', 'device': 'cpu'}
                          for obs in observations]
            add('prior-panel', {'conditions': conditions}, 'cpu-prior', purpose='all observation panels, shared query-only fits')

    for obs in observations:
        for fold in main_folds + contexts:
            prep_device = resources['preparation']['device']
            prep = add('prepare-decoder', {**obs, 'fold': fold, 'seed': plan['seeds'][0], 'device': prep_device},
                       'gpu-prepare' if prep_device == 'cuda' else 'prepare')
            families = [f for f in plan['decoder']['families'] if f != 'prior'] if fold in main_folds else [plan['geometry']['primary_decoder_parent']]
            for seed in plan['seeds']:
                for family in families:
                    base = {**obs, 'fold': fold, 'seed': seed, 'family': family,
                            'device': decoder_device, 'require_prepared': True}
                    add('decoder', base, ('gpu-' if decoder_device == 'cuda' else 'cpu-') + 'decoder', [prep])
                    if fold in main_folds and family in plan['decoder']['retrained_null_families']:
                        add('decoder', {**base, 'retrain_null': True},
                            ('gpu-' if decoder_device == 'cuda' else 'cpu-') + 'decoder', [prep])

    enc_resource = ('gpu-' if encoding_device == 'cuda' else 'cpu-') + 'encoding'
    for subject in plan['subjects']:
        for fold in main_folds + contexts:
            for seed in plan['seeds']:
                for contrast in plan['encoding_contrasts']:
                    if fold in contexts and contrast['comparison'] != 'matched-binding':
                        continue
                    for side in (['augmented'] if fold in contexts else ['baseline', 'augmented']):
                        add('encoding', {'subject': subject, 'fold': fold, 'seed': seed,
                            'groups': contrast[side], 'comparison': contrast['comparison'], 'device': encoding_device}, enc_resource)
                if fold in contexts:
                    continue
                for model in resolved['models']:
                    for layer in model['descriptive_layers']:
                        for groups in plan['model_encoding']['conditions']:
                            options = {'subject': subject, 'fold': fold, 'seed': seed, 'groups': groups,
                                'comparison': plan['model_encoding']['comparison'], 'device': encoding_device,
                                'mask_models': [model['id'] + ':' + str(layer)]}
                            if 'model' in groups:
                                options.update(model=model['id'], layer=layer)
                            add('encoding', options, enc_resource)
    counts = {}
    encoding_arrays_bytes, compact_bytes = 0, 0
    for job in jobs:
        counts[job['resource']] = counts.get(job['resource'], 0) + 1
        if job['kind'] == 'encoding':
            options = job['options']
            split = partition(data.semantics, options['fold'])
            contract = data.reader.contract['subjects'][options['subject']]
            # No masks/arrays are opened here. Full trimmed rows bound these
            # particular dense arrays before compression; this is not disk use.
            train_rows = sum(contract[s]['timepoints'] - 20 for s in split['train'])
            test_rows = sum(contract[s]['timepoints'] - 20 for s in split['test'])
            voxels = contract[split['train'][0]]['voxels']
            encoding_arrays_bytes += 4 * voxels * (train_rows + test_rows * (1 + len(options['groups'])))
            compact_bytes += 8 * train_rows * test_rows * len(options['groups']) + voxels * (8 + 32 * len(split['test']))
    workers = worker_inventory(jobs, resources['worker_fit_limit'])
    body = {'format_version': 2, 'phase': phase, 'semantic_build_hash': data.semantics.build_hash,
            'build': data.semantics.build.relative_to(data.root).as_posix(),
            'data_contract_hash': object_hash(data.reader.contract), 'spatial_hash': object_hash(data.spatial.identity),
            'analysis_config': data.config, 'definition_hash': resolved['definition_hash'],
            'model_lock_hash': resolved['model_lock_hash'], 'code': code_identity(),
            'resources': resources, 'jobs': jobs, 'counts': counts, 'workers': workers,
            'worker_counts': {name: sum(w['resource'] == name for w in workers) for name in counts},
            'storage_planning': {'encoding_dual_prediction_contribution_full_rows_bytes': encoding_arrays_bytes,
                'encoding_operator_and_metric_full_rows_bytes_before_sharing': compact_bytes,
                'cache_limit_bytes': resources['storage']['cache_gib'] * 2**30,
                'meaning': 'Full-row uncompressed array ceilings before comparison masks, compression and exact-fit sharing. Excludes decoder weights/traces, geometry and metadata; not an estimate of total disk use.'},
            'scope': 'Preparation and fitted parents, including declared descriptive layers and multistory geometry parents.',
            'seed_policy': 'All configured seeds retained for decoding and the randomized encoding search.',
            'final_release': 'Final jobs require a separate final manifest and explicit --allow-final.'}
    return {**body, 'content_hash': object_hash(body)}


def write_manifest(data, **options):
    report = execution_manifest(data, **options)
    path = data.root / 'artifacts/execution' / report['content_hash'] / 'manifest.json'
    write_report(path, report)
    write_report(data.root / 'artifacts/execution/latest.json', {'path': path.relative_to(data.root).as_posix(),
                                                               'content_hash': report['content_hash']})
    print('FIT JOB COUNTS:', report['counts'], flush=True)
    return path


def run_job(root, manifest_path, index, *, allow_final=False):
    return run_jobs(root, manifest_path, [index], allow_final=allow_final)[0]


def run_worker(root, manifest_path, worker_index, *, allow_final=False):
    manifest = read_json(Path(manifest_path))
    if manifest.get('format_version') != 2 or not 0 <= worker_index < len(manifest['workers']):
        raise ValueError('Unknown execution worker.')
    status = Path(root) / manifest['analysis_config']['output'] / 'execution' / manifest['content_hash'] / 'attempts' / (str(worker_index) + '.json')
    import os
    import time
    started = time.time()
    identity = {'worker': worker_index, 'job': os.environ.get('SLURM_JOB_ID'), 'started': started, 'attempt': os.environ.get('NEUROSYM_ATTEMPT_TOKEN')}
    try:
        result = run_jobs(root, manifest_path, manifest['workers'][worker_index]['indices'], allow_final=allow_final)
    except YieldRequested:
        write_report(status, {**identity, 'status': 'yielded', 'progress_steps': deadline.progress, 'elapsed_seconds': time.time() - started})
        raise
    except BaseException:
        write_report(status, {**identity, 'status': 'failed', 'elapsed_seconds': time.time() - started})
        raise
    write_report(status, {**identity, 'status': 'complete', 'elapsed_seconds': time.time() - started})
    return result


def ensure_device(data, manifest, device):
    """Verify whichever compatible GPU Slurm assigned; never require a model name."""
    import importlib.util
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Manifest requires CUDA; no CPU fallback.')
    gpu = torch.cuda.get_device_name(device)
    destination = data.root / 'artifacts/compute-devices' / (object_hash(gpu) + '.json')
    def valid():
        if not destination.exists():
            return False
        v = read_json(destination)
        return (v.get('status') == 'verified' and v.get('code') == manifest['code'] and
                v.get('semantic_build_hash') == data.semantics.build_hash and v.get('config_hash') == object_hash(data.config) and
                v.get('compute_config_hash') == object_hash(data.compute_config) and v.get('torch') == str(torch.__version__) and v.get('gpu') == gpu)
    if valid():
        return
    if not data.compute_config['execution']['auto_verify_device']:
        raise ValueError('Run compute verification on the allocated GPU and current code.')
    with process_lock(destination.with_suffix('.oslock')):
        if valid():
            return
        script = Path(__file__).resolve().parents[1] / 'scripts/verify_compute.py'
        spec = importlib.util.spec_from_file_location('allocated_device_verification', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        threads, rng, cuda_rng = torch.get_num_threads(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
        try:
            report = module.verify(data, device=device)
            write_report(destination, report)
        finally:
            torch.set_num_threads(threads)
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        deadline.check()


def run_jobs(root, manifest_path, indices, *, allow_final=False):
    from .analysis_data import AnalysisData
    from .decoder_fit import prepare_decoder, run_decoder
    from .encoding import prepare_encoding_panel, run_encoding
    manifest = read_json(Path(manifest_path))
    digest = manifest['content_hash']
    if object_hash({k: v for k, v in manifest.items() if k != 'content_hash'}) != digest:
        raise ValueError('Execution manifest changed.')
    if manifest['phase'] == 'final' and not allow_final:
        raise ValueError('Final evaluation has not been explicitly released.')
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= i < len(manifest['jobs']) for i in indices):
        raise ValueError('Manifest job indices outside inventory or duplicated.')
    jobs = [manifest['jobs'][i] for i in indices]
    if len({(j['kind'], j['resource']) for j in jobs}) != 1:
        raise ValueError('A worker must have one preparation/fit kind and resource.')
    worker_timings = Timings()
    data = AnalysisData(root, build=Path(root) / manifest['build'], require_prepared=jobs[0]['kind'] == 'decoder')
    plan = experiment_plan(data)
    if (data.config != manifest['analysis_config'] or code_identity() != manifest['code'] or
            data.compute_config != manifest['resources'] or
            object_hash(data.reader.contract) != manifest['data_contract_hash'] or
            object_hash(data.spatial.identity) != manifest['spatial_hash'] or
            data.semantics.build_hash != manifest['semantic_build_hash'] or
            plan['definition_hash'] != manifest['definition_hash'] or plan['model_lock_hash'] != manifest['model_lock_hash']):
        raise ValueError('Execution code, data or definitions changed; generate a new manifest.')
    for device in {j['options'].get('device', 'cpu') for j in jobs}:
        if device.startswith('cuda'):
            ensure_device(data, manifest, device)
    directory = Path(root) / data.config['output'] / 'execution' / digest
    directory.mkdir(parents=True, exist_ok=True)
    worker_timings.device = jobs[0]['options'].get('device', 'cpu')
    runtime_path = directory / 'workers' / (object_hash(indices) + '.json')

    def progress(completed, status):
        write_report(runtime_path, {'manifest_hash': digest, 'logical_indices': indices,
            'completed_logical_jobs': completed, 'status': status, **worker_timings.report()})

    progress(0, 'running')

    def receipt_path(identity):
        return directory / (identity + '.json')

    for job in jobs:
        for dependency in job['dependencies']:
            receipt = read_json(receipt_path(dependency))
            if receipt['manifest_hash'] != digest or receipt['job_id'] != dependency or receipt['status'] != 'complete':
                raise ValueError('Incomplete/mismatched preparation dependency: ' + dependency)
            for result in receipt['results']:
                if not (Path(root) / result / 'complete.json').is_file():
                    raise FileNotFoundError('Prepared dependency output missing: ' + result)
    if jobs[0]['kind'] == 'encoding':
        try:
            with worker_timings.phase('shared_encoding_selection'):
                prepare_encoding_panel(data, [j['options'] for j in jobs])
        except BaseException:
            progress(0, 'failed during shared selection')
            raise
        progress(0, 'selection complete; fits not yet complete')
    completed = []
    for job in jobs:
        destination = receipt_path(job['id'])
        deadline.check()
        with exclusive_run(destination.with_suffix('.lock')):
            if destination.exists():
                receipt = read_json(destination)
                if (receipt.get('manifest_hash') != digest or receipt.get('job_id') != job['id'] or
                        receipt.get('status') != 'complete' or
                        any(not (Path(root) / p / 'complete.json').is_file() for p in receipt['results'])):
                    raise ValueError('Existing execution receipt/output is incomplete or mismatched.')
                completed.append(destination)
                progress(len(completed), 'running')
                continue
            if job['kind'] == 'prepare-decoder':
                paths = [prepare_decoder(data, job['options'])]
            elif job['kind'] == 'encoding':
                paths = [run_encoding(data, job['options'])]
            elif job['kind'] == 'decoder':
                paths = [run_decoder(data, job['options'])]
            elif job['kind'] == 'prior-panel':
                paths = [run_decoder(data, options) for options in job['options']['conditions']]
            else:
                raise ValueError('Unknown manifest job kind.')
            write_report(destination, {'status': 'complete', 'manifest_hash': digest, 'job_id': job['id'],
                'results': [p.relative_to(root).as_posix() for p in paths]})
        completed.append(destination)
        progress(len(completed), 'running')
        print('LOGICAL JOB COMPLETE:', job['id'], flush=True)
    progress(len(completed), 'complete')
    return completed
