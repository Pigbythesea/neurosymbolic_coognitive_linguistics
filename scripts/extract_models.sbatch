#!/bin/bash
#SBATCH --job-name=neurosym-states
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%2
#SBATCH --comment=accept_cost
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/states-%A_%a.log

set -euo pipefail
completed=0
trap 'result=$?; if [[ "$result" -eq 0 && "$completed" -ne 1 ]]; then result=1; fi; printf "JOB EXIT status=%s job=%s task=%s\n" "$result" "${SLURM_JOB_ID:-unset}" "${SLURM_ARRAY_TASK_ID:-unset}"; exit "$result"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
printf 'EXTRACTION job=%s task=%s host=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname -s)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
PYTHON="$NEUROSYM_ROOT/.venv-extraction/bin/python"
"$PYTHON" -I -B -u scripts/extract_model_states.py --model-index "$SLURM_ARRAY_TASK_ID"
"$PYTHON" -I -B -u scripts/align_model_features.py --model-index "$SLURM_ARRAY_TASK_ID"
completed=1
printf '%s\n' 'MODEL STATES AND TEMPORAL FEATURES COMPLETE'
