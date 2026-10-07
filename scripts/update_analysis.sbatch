#!/bin/bash
# Install the validated 2026-10-07 source-batching bundle. No dependency installs.
#SBATCH --job-name=ns-update-batch
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --comment=accept_cost
#SBATCH --dependency=singleton
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/update-analysis-%j.log

set -euo pipefail
unset HISTFILE
umask 077
ulimit -c 0
trap 'result=$?; printf "ANALYSIS UPDATE EXIT status=%s job=%s\n" "$result" "${SLURM_JOB_ID:-unset}"' EXIT
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* ]]; then
    printf '%s\n' 'STOP: an allocated compute node is required.' >&2
    exit 1
fi
cd /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
printf 'ANALYSIS UPDATE job=%s host=%s start=%s\n' "$SLURM_JOB_ID" "$(hostname -s)" "$(date -u +%FT%TZ)"

/usr/bin/python3.11 -I -B -u - <<'PY'
from pathlib import Path
import hashlib
import os
import re
import subprocess

root = Path('/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics')
if Path.cwd() != root or root.resolve() != root:
    raise SystemExit('STOP: workspace resolves elsewhere.')
expected = {
    'artifacts/analysis-source.zip': '887975589fdb38a2fd626b5556551319436504e1cc8cebdc7e58a358929cb0e7',
    'artifacts/install_analysis_bundle.py': '6635def39212cc750ef62e1810e61bb7a50b8bdcd22b7c60a45165d4783d03e8',
    'scripts/cluster_env.sh': 'bfad6e21322e06b4e043e8851353656a420ef9a5a7ee64380705d84f10eb2245',
}
for name, wanted in expected.items():
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise SystemExit('STOP: file escapes workspace: ' + name)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != wanted:
        raise SystemExit('STOP: uploaded bundle/helper differs: ' + name)
for name in ('.tmp', '.cache', '.config', '.local', '.python', '.venv-analysis',
             'artifacts/analysis-source-backups', 'logs', 'analysis_code'):
    if not (root / name).resolve().is_relative_to(root):
        raise SystemExit('STOP: output/environment directory escapes workspace: ' + name)
if not (root / '.venv-analysis/bin/python').is_file() or not (root / '.analysis-source.json').is_file():
    raise SystemExit('STOP: existing analysis environment/installation receipt is required.')
queue = subprocess.check_output(['squeue', '-h', '-r', '-u', 'zzhan330', '-o', '%i'], text=True)
active = []
for job in queue.split():
    if job == os.environ['SLURM_JOB_ID']:
        continue
    result = subprocess.run(['scontrol', 'show', 'job', '-o', job], capture_output=True, text=True)
    if result.returncode:
        if 'Invalid job id' in result.stderr:
            continue  # it ended after the queue snapshot
        raise SystemExit('STOP: unable to inspect active job ' + job + ': ' + result.stderr)
    state = re.search(r'\bJobState=(\S+)', result.stdout)
    if state and state[1] in {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}:
        continue
    workdir = re.search(r'\bWorkDir=(\S+)', result.stdout)
    output = re.search(r'\bStdOut=(\S+)', result.stdout)
    if any(match and Path(match[1]).resolve().is_relative_to(root) for match in (workdir, output)):
        active.append(job)
if active:
    raise SystemExit('STOP: project jobs are still active/queued; preserve/yield them before updating: ' + ', '.join(active))
print('Pinned upload hashes, project paths and absence of other project jobs verified.', flush=True)
PY

source scripts/cluster_env.sh
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
NEURO_UPDATE_PYTHON="$NEUROSYM_ROOT/.venv-analysis/bin/python"

"$NEURO_UPDATE_PYTHON" -I -B -u - <<'PY'
from pathlib import Path
import hashlib
from importlib.metadata import version
import json
import os
import torch

root = Path(os.environ['NEUROSYM_ROOT'])
for name in ('TMPDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME',
             'UV_CACHE_DIR', 'UV_PYTHON_INSTALL_DIR', 'PIP_CACHE_DIR', 'HF_HOME', 'HF_HUB_CACHE',
             'HF_DATASETS_CACHE', 'HF_XET_CACHE', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR',
             'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'TORCHINDUCTOR_CACHE_DIR'):
    if not Path(os.environ[name]).is_absolute() or not Path(os.environ[name]).resolve().is_relative_to(root):
        raise SystemExit('STOP: cache/config/temp path escapes workspace: ' + name)
expected = {'numpy': '2.3.3', 'h5py': '3.14.0', 'pydantic': '2.12.3', 'scipy': '1.16.3',
            'nibabel': '5.3.2', 'torch': '2.10.0'}
actual = {name: version(name) for name in expected}
if any(actual[name].split('+')[0] != wanted for name, wanted in expected.items()) or torch.version.cuda != '12.8':
    raise SystemExit('STOP: installed dependencies differ; no environment was modified: ' + json.dumps(actual))
print('Reusing existing environment:', json.dumps(actual), 'CUDA wheel:', torch.version.cuda, flush=True)
# Keep the old installation identity as well as its immutable code/checkpoints.
receipt = root / '.analysis-source.json'
if receipt.is_symlink():
    raise SystemExit('STOP: installation receipt is a symlink.')
before = receipt.read_bytes()
archive = root / 'artifacts/analysis-source-backups/receipts' / (hashlib.sha256(before).hexdigest() + '.json')
if archive.resolve() != archive:
    raise SystemExit('STOP: receipt archive resolves elsewhere.')
archive.parent.mkdir(parents=True, exist_ok=True)
if archive.exists() and archive.read_bytes() != before:
    raise SystemExit('STOP: existing receipt backup differs.')
if not archive.exists():
    archive.write_bytes(before)
PY

"$NEURO_UPDATE_PYTHON" -I -B -u artifacts/install_analysis_bundle.py artifacts/analysis-source.zip
NEURO_UPDATE_CODE=$("$NEURO_UPDATE_PYTHON" -I -B -c 'import json; print(json.load(open(".analysis-source.json"))["code_root"])')
"$NEURO_UPDATE_PYTHON" -I -B -u "$NEURO_UPDATE_CODE/scripts/run_analysis.py" preflight
printf 'ANALYSIS SOURCE UPDATE COMPLETE end=%s; environment reused; no scientific fits submitted.\n' "$(date -u +%FT%TZ)"
