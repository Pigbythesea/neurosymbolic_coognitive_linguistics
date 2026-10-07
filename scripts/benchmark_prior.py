"""Time the installed prior on a complete real inner fold, outside production fits.

Each process starts from the same seed, trains three epochs through the existing
train_decoder function, and validates on the same development story. This is a
timing prefix, not a shortened scientific fit or a replacement for nested CV.
No production checkpoint is loaded, changed, or published by this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time


ROOT = Path('/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics')


def digest_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or socket.gethostname().split('.')[0].startswith('login')
            or ROOT.resolve() != ROOT or Path.cwd() != ROOT):
        raise RuntimeError('Benchmark requires an allocated compute node in the project workspace.')
    if not 1 <= args.threads <= int(os.environ['SLURM_CPUS_PER_TASK']):
        raise ValueError('Thread count exceeds the allocated CPUs.')
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT / 'artifacts/prior-benchmarks'):
        raise ValueError('Benchmark output must stay under artifacts/prior-benchmarks.')
    for name in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR'):
        if not Path(os.environ[name]).resolve().is_relative_to(ROOT):
            raise ValueError('Cache/temp path escapes workspace: ' + name)

    installed = json.loads((ROOT / '.analysis-source.json').read_text())
    code_relative = installed['code_root']
    code = (ROOT / code_relative).resolve()
    if not code.is_relative_to(ROOT / 'analysis_code'):
        raise ValueError('Installed code root escapes the isolated code directory.')
    for name, expected in installed['files'].items():
        if name.startswith(code_relative + '/') or name in ('configs/analysis.json', 'configs/compute.json'):
            path = ROOT / name
            if not path.resolve().is_relative_to(ROOT) or digest_file(path) != expected:
                raise ValueError('Installed source/configuration changed: ' + name)
    sys.path.insert(0, str(code))

    import resource
    import numpy as np
    import torch
    from neurosym.analysis_data import AnalysisData
    from neurosym.analysis_runs import partition, write_report
    from neurosym.decoder_fit import examples_and_windows, fit_identity, make_decoder, train_decoder
    from neurosym.io import object_hash
    from neurosym.runtime import deadline, YieldRequested

    deadline.install()
    # All cases share one allocation; do not restart its walltime budget in
    # each fresh Python process if a site omits SLURM_JOB_END_TIME.
    benchmark_end = float(os.environ['NEUROSYM_BENCH_END_EPOCH'])
    remaining = time.monotonic() + max(0., benchmark_end - time.time() - 120.)
    deadline.end = min(deadline.end, remaining) if deadline.end is not None else remaining
    if deadline.due():
        print('PRIOR BENCHMARK YIELDED before starting another case.', flush=True)
        return 75
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no device fallback.')
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = AnalysisData(ROOT)
    if data.semantics.build_hash != installed['semantic_build_hash']:
        raise ValueError('Semantic build differs from the installed bundle.')
    split = partition(data.semantics, '0')
    data.validate_partition(split['train'], test=split['test'])
    inner = split['inner'][0]
    options = {'modality': 'brain', 'subject': 'subject01', 'fold': '0',
               'seed': 11, 'family': 'prior', 'device': args.device}
    train, windows, _ = examples_and_windows(data, inner['train'], None, options)
    validation, val_windows, _ = examples_and_windows(data, inner['validation'], None, options)
    # make_decoder constructs its seeded weights on CPU before moving them to
    # the selected device, matching the production initialization procedure.
    model, _ = make_decoder(data, train, windows, None, options, args.device)
    cpu_description = next((line.split(':', 1)[1].strip() for line in
                            Path('/proc/cpuinfo').read_text().splitlines()
                            if line.startswith('model name')), 'unavailable')
    settings = {'learning_rate': data.config['decoder']['learning_rates'][0],
                'epochs': 3, 'seed': 11, 'pairing': None,
                'patience': data.config['decoder']['patience']}
    identity = fit_identity(data, model, train, windows, (validation, val_windows), **settings)
    neutral = {key: value for key, value in identity.items() if key != 'device'}
    initial = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial.update(name.encode())
        initial.update(value.detach().cpu().contiguous().numpy().tobytes())
    record = {
        'status': 'running', 'purpose': 'runtime measurement; no scientific fit receipt',
        'scientific_fits_executed': False, 'production_checkpoints_used': False,
        'job': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
        'cpu': cpu_description, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'device': args.device, 'threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(), 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
        'code_root': code_relative, 'benchmark_sha256': digest_file(Path(__file__)),
        'semantic_build_hash': data.semantics.build_hash,
        'workload_hash_without_device': object_hash(neutral),
        'initial_parameters_sha256': initial.hexdigest(),
        'train_stories': inner['train'], 'validation_stories': inner['validation'],
        'train_queries': len(train), 'validation_queries': len(validation),
        'train_sources': len({item['source_id'] for item in train}),
        'parameters': sum(value.numel() for value in model.parameters()),
        'settings': settings, 'configured_production_epoch_cap': data.config['decoder']['epochs'],
        'input_and_model_setup_seconds': time.perf_counter() - started,
    }
    write_report(destination / 'run.json', record)
    print('PRIOR BENCHMARK START:', args.device, 'threads=', args.threads,
          'train_sources=', record['train_sources'], 'train_queries=', len(train), flush=True)
    print('PRIOR BENCHMARK HARDWARE:', json.dumps({key: record[key] for key in (
        'host', 'cpu', 'cpu_affinity', 'threads', 'interop_threads', 'gpu',
        'parameters', 'input_and_model_setup_seconds')}), flush=True)
    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.perf_counter()
    try:
        result = train_decoder(data, model, train, windows,
                               validation=(validation, val_windows),
                               checkpoint=destination / 'checkpoint.pt',
                               reuse_prior=False, **settings)
    except YieldRequested:
        write_report(destination / 'result.json', {**record, 'status': 'yielded',
                     'elapsed_seconds': time.perf_counter() - wall_start})
        print('PRIOR BENCHMARK YIELDED; timing record is incomplete.', flush=True)
        return 75
    if args.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - wall_start
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    user_seconds = usage_end.ru_utime - usage_start.ru_utime
    system_seconds = usage_end.ru_stime - usage_start.ru_stime
    epoch_seconds = [item['wall_seconds'] for item in result['history']]
    if len(epoch_seconds) != settings['epochs']:
        raise ValueError('Benchmark did not complete the same three full epochs.')
    measured = {**record, 'status': 'complete', 'training_and_validation_seconds': elapsed,
                'user_cpu_seconds': user_seconds, 'system_cpu_seconds': system_seconds,
                'average_busy_cpu_cores': (user_seconds + system_seconds) / elapsed,
                'cold_epoch_seconds': epoch_seconds[0],
                'steady_epoch_seconds': epoch_seconds[1:],
                'median_steady_epoch_seconds': statistics.median(epoch_seconds[1:]),
                'peak_process_rss_bytes': usage_end.ru_maxrss * 1024,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0,
                'selected_epoch_within_timing_prefix': result['best_epoch'],
                'history': result['history']}
    # Last-epoch weights let us inspect CPU/GPU numerical drift independently
    # of which epoch has the best validation score. These are benchmark files.
    saved = torch.load(destination / 'checkpoint.pt', map_location='cpu', weights_only=True)
    np.savez(destination / 'last-epoch-parameters.npz',
             **{name: value.detach().cpu().numpy() for name, value in saved['model'].items()})
    write_report(destination / 'result.json', measured)
    print('PRIOR BENCHMARK RESULT:', json.dumps({key: measured[key] for key in (
        'device', 'threads', 'gpu', 'cpu', 'median_steady_epoch_seconds',
        'average_busy_cpu_cores', 'peak_process_rss_bytes', 'peak_cuda_allocated_bytes',
        'workload_hash_without_device', 'initial_parameters_sha256')}), flush=True)
    print('PRIOR BENCHMARK RECORD:', destination / 'result.json', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
