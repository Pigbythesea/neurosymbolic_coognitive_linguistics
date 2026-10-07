#!/bin/bash
#SBATCH --job-name=neurosym-analysis
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --comment=accept_cost
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/analysis-%j.log

set -euo pipefail
trap 'result=$?; printf "JOB EXIT status=%s job=%s\n" "$result" "${SLURM_JOB_ID:-unset}"' EXIT
source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="$NEUROSYM_ROOT/.venv-analysis/bin/python"
CODE_ROOT=$("$PYTHON" -I -B -c 'import json; print(json.load(open(".analysis-source.json"))["code_root"])')
exec "$PYTHON" -I -B -u "$CODE_ROOT/scripts/run_analysis.py" "$@"
