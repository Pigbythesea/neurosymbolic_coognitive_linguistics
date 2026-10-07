#!/bin/bash
#SBATCH --job-name=neurosym-prior-bench
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --comment=accept_cost
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/prior-benchmark-%j.log

set -euo pipefail
unset HISTFILE
umask 077
ulimit -c 0
source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEUROSYM_YIELD_RESERVE=120
export NEUROSYM_WALL_SECONDS=900
export NEUROSYM_BENCH_END_EPOCH=$(( $(date +%s) + 900 ))
NEURO_BENCH_DIR="$NEUROSYM_ROOT/artifacts/prior-benchmarks/job-$SLURM_JOB_ID"
NEURO_BENCH_PYTHON="$NEUROSYM_ROOT/.venv-analysis/bin/python"
printf 'PRIOR BENCHMARK job=%s host=%s suite=%s\n' "$SLURM_JOB_ID" "$(hostname -s)" "${1:-unset}"

run_case() {
    export OMP_NUM_THREADS="$2" OPENBLAS_NUM_THREADS="$2" MKL_NUM_THREADS="$2"
    "$NEURO_BENCH_PYTHON" -I -B -u artifacts/benchmark_prior.py \
        --device "$1" --threads "$2" --output "$NEURO_BENCH_DIR/$1-threads-$2"
}

case "${1:-}" in
    cpu)
        run_case cpu 1
        run_case cpu 4
        run_case cpu 9
        ;;
    gpu)
        if [[ -z "${2:-}" ]]; then
            printf '%s\n' 'GPU suite requires the completed CPU reference directory as its second argument.' >&2
            exit 1
        fi
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        run_case cuda 1
        ;;
    *)
        printf '%s\n' 'Specify cpu or gpu as the suite; GPU suite needs a GPU allocation.' >&2
        exit 1
        ;;
esac
"$NEURO_BENCH_PYTHON" -I -B -u - "$NEURO_BENCH_DIR" "${2:-$NEURO_BENCH_DIR/cpu-threads-1}" <<'PY'
from pathlib import Path
import json
import numpy as np
import sys

root = Path(sys.argv[1])
reference = Path(sys.argv[2]).resolve()
if not reference.is_relative_to(root.parent.resolve()):
    raise SystemExit('STOP: CPU reference must be under the project prior-benchmarks directory.')
baseline = json.loads((reference / 'result.json').read_text())
if baseline['status'] != 'complete' or baseline['device'] != 'cpu' or baseline['threads'] != 1:
    raise SystemExit('STOP: CPU reference must be a completed one-thread benchmark.')
comparisons = []
with np.load(reference / 'last-epoch-parameters.npz') as first:
    for path in sorted(root.glob('*/result.json')):
        report = json.loads(path.read_text())
        if any(report[key] != baseline[key] for key in
               ('status', 'workload_hash_without_device', 'initial_parameters_sha256')):
            raise SystemExit('STOP: benchmark cases differ in workload/initialization/completion.')
        error, difference_squared, reference_squared = 0., 0., 0.
        with np.load(path.parent / 'last-epoch-parameters.npz') as second:
            if set(first.files) != set(second.files):
                raise SystemExit('STOP: benchmark state inventories differ.')
            for key in first.files:
                a, b = first[key], second[key]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise SystemExit('STOP: benchmark state shapes/dtypes differ.')
                if not a.size:
                    continue
                difference = a.astype('float64') - b.astype('float64')
                error = max(error, float(np.max(np.abs(difference))))
                difference_squared += float(np.sum(difference ** 2))
                reference_squared += float(np.sum(a.astype('float64') ** 2))
        row = {'case': path.parent.name,
               'median_epoch_seconds': report['median_steady_epoch_seconds'],
               'speedup_vs_reference_cpu_1': baseline['median_steady_epoch_seconds'] / report['median_steady_epoch_seconds'],
               'same_host': baseline['host'] == report['host'],
               'reference_host': baseline['host'], 'case_host': report['host'],
               'reference_cpu': baseline['cpu'], 'case_cpu': report['cpu'],
               'last_parameter_max_abs_difference': error,
               'last_parameter_relative_l2_difference': (difference_squared / max(reference_squared, 1e-30)) ** .5,
               'max_validation_nll_difference': max(abs(a['validation_story_macro_nll'] - b['validation_story_macro_nll'])
                   for a, b in zip(baseline['history'], report['history'], strict=True)),
               'timing_prefix_selected_epoch': report['selected_epoch_within_timing_prefix']}
        comparisons.append(row)
        print('PRIOR BENCHMARK COMPARISON:', json.dumps(row), flush=True)
(root / 'comparison.json').write_text(json.dumps({
    'reference': str(reference), 'cases': comparisons,
    'scope': 'Performance and numerical diagnostics of a three-epoch timing prefix; not full-fit equivalence certification.',
    'hardware_scope': 'Different-host timings compare available allocations, not an isolated same-host CPU/CUDA speedup.',
}, indent=2), encoding='utf-8')
PY
printf 'PRIOR BENCHMARK COMPLETE: %s\n' "$NEURO_BENCH_DIR"
