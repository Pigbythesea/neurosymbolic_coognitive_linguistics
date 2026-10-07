#!/bin/bash
# One-time recovery for the inspected October 6 transfer; not a fit launcher.
#SBATCH --job-name=neurosym-finish-setup
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --comment=accept_cost
#SBATCH --dependency=singleton
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/finish-analysis-%j.log

set -euo pipefail
unset HISTFILE
umask 077
ulimit -c 0
trap 'result=$?; printf "FINISH SETUP EXIT status=%s job=%s\n" "$result" "${SLURM_JOB_ID:-unset}"' EXIT
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* ]]; then
    printf '%s\n' 'STOP: an allocated compute node is required.' >&2
    exit 1
fi
cd /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
printf 'FINISH SETUP job=%s host=%s start=%s\n' "$SLURM_JOB_ID" "$(hostname -s)" "$(date -u +%FT%TZ)"

/usr/bin/python3.11 -I -B -u - <<'PY'
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

root = Path('/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics')
if Path.cwd() != root or root.resolve() != root:
    raise SystemExit('STOP: workspace resolves elsewhere.')
expected = {
    'artifacts/analysis-source.zip': 'd8d203a58b79af3c856a26e96247be20b39004af187eb4acba5fdb93ca0a9a58',
    'artifacts/install_analysis_bundle.py': '6635def39212cc750ef62e1810e61bb7a50b8bdcd22b7c60a45165d4783d03e8',
    'artifacts/setup_analysis.sbatch': '68c725223f42a9ffa80354693b1bcb848b83172de010f3f8062e7c7d567c97d3',
    'scripts/cluster_env.sh': 'bfad6e21322e06b4e043e8851353656a420ef9a5a7ee64380705d84f10eb2245',
    'scripts/align_model_features.py': '5a4325399aa1b4f40add38294a8a4d63966078432deb05e60f1b27b6b2bd6a7e',
    'neurosym/model_features.py': 'af1a04bda705cb3c8d4419c2b71a7e0fa4ae8d89d6b347298cf7e5ea15a6ae22',
    'neurosym/temporal.py': 'ff084e2c347c7e0e0e536103c6e9e61bfc0386dc31ad14006b27b25569b3a219',
}
for name, digest in expected.items():
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise SystemExit('STOP: file resolves outside workspace: ' + name)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != digest:
        raise SystemExit('STOP: transferred/existing file differs: ' + name)
print('Uploaded bundle and existing alignment code: hashes verified.', flush=True)
for name in ('.tmp', '.cache', '.config', '.local', '.python', '.venv-analysis',
             'artifacts/lock-recovery', 'logs', 'analysis_code',
             'data/features/frozen/qwen38-27b/aligned'):
    if not (root / name).resolve().is_relative_to(root):
        raise SystemExit('STOP: output directory resolves outside workspace: ' + name)

lock = root / 'data/features/frozen/qwen38-27b/aligned/RUNNING.lock'
if lock.exists() or lock.is_symlink():
    if lock.is_symlink():
        raise SystemExit('STOP: alignment lock is a symlink.')
    before = lock.read_bytes()
    owner = {'pid': 3931069, 'job': '995154', 'started_utc': '2026-10-04T09:57:13.176008+00:00'}
    if json.loads(before) != owner:
        raise SystemExit('STOP: lock owner changed; inspect before recovery.')
    queue = subprocess.check_output(
        ['squeue', '-h', '-r', '-u', 'zzhan330', '-o', '%F|%i|%T'], text=True)
    if any(line.split('|')[0] == owner['job'] or
           line.split('|')[1].split('_')[0] == owner['job']
           for line in queue.splitlines() if line.strip()):
        raise SystemExit('STOP: recorded lock owner is still queued/running.')
    accounting = subprocess.check_output(
        ['sacct', '-X', '-n', '-P', '-j', owner['job'],
         '--format=JobID,State,ExitCode,Elapsed,NodeList'], text=True)
    rows = [line.strip().split('|') for line in accounting.splitlines() if line.strip()]
    if not any(row[0] == '995154_4' and row[1] == 'TIMEOUT' for row in rows):
        raise SystemExit('STOP: expected timeout is not confirmed by current accounting.')
    archive = root / 'artifacts/lock-recovery' / ('qwen38-27b-995154-' + os.environ['SLURM_JOB_ID'])
    if archive.exists():
        raise SystemExit('STOP: recovery archive already exists; inspect previous attempt.')
    archive.mkdir(parents=True)
    (archive / 'inspection.json').write_text(json.dumps({
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'owner': owner, 'queue': queue, 'accounting': accounting,
        'recovery_job': os.environ['SLURM_JOB_ID'],
    }, indent=2), encoding='utf-8')
    if lock.read_bytes() != before:
        raise SystemExit('STOP: lock changed during inspection.')
    lock.rename(archive / 'RUNNING.lock')
    print('Archived confirmed stale lock:', archive, flush=True)
else:
    print('No legacy alignment lock remains; existing outputs will be checked.', flush=True)
PY

source scripts/cluster_env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export PIP_CONFIG_FILE=/dev/null
/usr/bin/python3.11 -I -B -u - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['NEUROSYM_ROOT'])
names = ('TMPDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME',
         'UV_CACHE_DIR', 'UV_PYTHON_INSTALL_DIR', 'PIP_CACHE_DIR', 'HF_HOME', 'HF_HUB_CACHE',
         'HF_DATASETS_CACHE', 'HF_XET_CACHE', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR',
         'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR',
         'TORCHINDUCTOR_CACHE_DIR')
for name in names:
    path = Path(os.environ[name])
    if not path.is_absolute() or not path.resolve().is_relative_to(root):
        raise SystemExit('STOP: environment path escapes workspace: ' + name)
print('All configured cache/temp/config/data paths are inside:', root, flush=True)
PY

printf 'ALIGNMENT start=%s\n' "$(date -u +%FT%TZ)"
.venv-extraction/bin/python -I -B -u scripts/align_model_features.py --model qwen38-27b
printf 'ALIGNMENT end=%s\n' "$(date -u +%FT%TZ)"
# Running this script inside our allocation keeps its setup on this CPU node.
# Its embedded SBATCH defaults are comments here; the 30-minute allocation above applies.
bash artifacts/setup_analysis.sbatch
printf 'CLUSTER ANALYSIS SETUP COMPLETE end=%s\n' "$(date -u +%FT%TZ)"
