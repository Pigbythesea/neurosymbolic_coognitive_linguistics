#!/bin/bash
# Administrative helper only: the qualified analysis installation stays immutable.
#SBATCH --job-name=ns-fit-maint
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --comment=accept_cost
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/fit-maint-%j.log

set -euo pipefail
trap 'status=$?; printf "FIT MAINTENANCE EXIT status=%s job=%s\n" "$status" "$SLURM_JOB_ID"' EXIT
source scripts/cluster_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NEURO_PYTHON="$NEUROSYM_ROOT/.venv-analysis/bin/python"
"$NEURO_PYTHON" -I -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ['NEUROSYM_ROOT'])
transfer = json.loads((root / 'artifacts/reuse-maintenance-transfer.json').read_text())
if set(transfer['files']) != {'artifacts/adopt_protocol2_fits.py', 'artifacts/maintain_qualified_fits.sbatch'}:
    raise SystemExit('STOP: unexpected maintenance transfer inventory.')
for name, wanted in transfer['files'].items():
    path = root / name
    if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
        raise SystemExit('STOP: maintenance helper hash/path mismatch: ' + name)
installed = json.loads((root / '.analysis-source.json').read_text())
if installed['code_root'] != transfer['qualified_code_root']:
    raise SystemExit('STOP: this transfer targets a different qualified installation.')
print('Maintenance transfer verified; analysis installation unchanged.', flush=True)
PY
NEURO_CODE=$("$NEURO_PYTHON" -I -B -c 'import json; print(json.load(open(".analysis-source.json"))["code_root"])')
NEURO_APPLY=()
case "${1:-inspect}" in
  inspect) ;;
  maintain) NEURO_APPLY=(--apply) ;;
  *) printf 'Use inspect or maintain.\n' >&2; exit 2 ;;
esac
"$NEURO_PYTHON" -I -B -u artifacts/adopt_protocol2_fits.py --root "$NEUROSYM_ROOT" --installed-code "${NEURO_APPLY[@]}"
"$NEURO_PYTHON" -I -B -u "$NEURO_CODE/scripts/cleanup_retired_decoder_tests.py" --root "$NEUROSYM_ROOT" "${NEURO_APPLY[@]}"
