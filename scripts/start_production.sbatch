#!/bin/bash
#SBATCH --job-name=ns-production
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --comment=accept_cost
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/production-%j.log

set -euo pipefail
source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NEURO_MODE=${1:-inspect}
[[ "$NEURO_MODE" == inspect || "$NEURO_MODE" == run ]] || exit 1
NEURO_PACKET=$("$NEUROSYM_ROOT/.venv-analysis/bin/python" -I -B artifacts/install_production_packet.py)
if [[ "$NEURO_MODE" == inspect ]]; then
  "$NEUROSYM_ROOT/.venv-analysis/bin/python" -I -B "$NEURO_PACKET/submit_production.py" --root "$NEUROSYM_ROOT"
  "$NEUROSYM_ROOT/.venv-analysis/bin/python" -I -B -u "$NEURO_PACKET/report_production.py" \
    "artifacts/execution/01b086932fdf03228e899b2e5c131e7514b46ba76d67b080245548837a2df61e/manifest.json" --root "$NEUROSYM_ROOT"
  "$NEUROSYM_ROOT/.venv-analysis/bin/python" -I -B - <<'PY'
from pathlib import Path
import zipfile
source = Path('artifacts/production-reports/01b086932fdf03228e899b2e5c131e7514b46ba76d67b080245548837a2df61e/latest')
temporary = Path('artifacts/production-evidence-latest.zip.tmp')
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in {'.json', '.md'}:
            archive.write(path, path.name)
temporary.replace('artifacts/production-evidence-latest.zip')
print((source / 'summary.md').read_text())
print('EVIDENCE ARCHIVE: artifacts/production-evidence-latest.zip')
PY
  printf '%s\n' 'PRODUCTION INSPECTION COMPLETE; no scientific fits submitted.'
else
  exec bash "$NEURO_PACKET/run_production.sbatch" "$NEURO_PACKET" "${2:-6}"
fi
