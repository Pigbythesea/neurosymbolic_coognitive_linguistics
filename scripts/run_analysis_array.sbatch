#!/bin/bash
# Resources, account, queue and array inventory are supplied by submit_analysis.py.
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
PYTHON="$NEUROSYM_ROOT/.venv-analysis/bin/python"
# Slurm copies this script into its spool; BASH_SOURCE is not the bundle path.
CODE_ROOT="${NEUROSYM_ANALYSIS_CODE:?Missing pinned analysis bundle path}"
[[ "$CODE_ROOT" == "$NEUROSYM_ROOT"/analysis_code/* ]] || exit 1
exec "$PYTHON" -I -B -u "$CODE_ROOT/scripts/run_manifest_array.py" "$@"
