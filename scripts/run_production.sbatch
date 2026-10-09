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
NEURO_PACKET=$(realpath -e "${1:?Pass the versioned production packet directory}")
[[ "$NEURO_PACKET" == "$NEUROSYM_ROOT"/artifacts/production/* ]] || exit 1
exec "$NEUROSYM_ROOT/.venv-analysis/bin/python" -I -B -u \
  "$NEURO_PACKET/submit_production.py" --root "$NEUROSYM_ROOT" \
  --max-gpus "${2:-6}" --submit --continue-controller --controller-seconds 14400
