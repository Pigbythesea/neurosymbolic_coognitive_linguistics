#!/bin/bash
#SBATCH --job-name=ns-trace-qualify
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --comment=accept_cost
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/trace-qualify-%j.log

set -euo pipefail
trap 'status=$?; printf "TRACE QUALIFICATION EXIT status=%s job=%s\n" "$status" "$SLURM_JOB_ID"' EXIT
source scripts/cluster_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NEURO_PYTHON="$NEUROSYM_ROOT/.venv-analysis/bin/python"
NEURO_CODE=$("$NEURO_PYTHON" -I -B -c 'import json; print(json.load(open(".analysis-source.json"))["code_root"])')
case "${1:-verify}" in
  verify)
    "$NEURO_PYTHON" -I -B -u "$NEURO_CODE/scripts/verify_trace_tables.py" --root "$NEUROSYM_ROOT" --device cuda --geometry
    ;;
  inspect)
    "$NEURO_PYTHON" -I -B -u "$NEURO_CODE/scripts/adopt_protocol2_fits.py" --root "$NEUROSYM_ROOT"
    "$NEURO_PYTHON" -I -B -u "$NEURO_CODE/scripts/cleanup_retired_decoder_tests.py" --root "$NEUROSYM_ROOT"
    ;;
  maintain)
    "$NEURO_PYTHON" -I -B -u "$NEURO_CODE/scripts/adopt_protocol2_fits.py" --root "$NEUROSYM_ROOT" --apply
    "$NEURO_PYTHON" -I -B -u "$NEURO_CODE/scripts/cleanup_retired_decoder_tests.py" --root "$NEUROSYM_ROOT" --apply
    ;;
  *) printf 'Use verify, inspect or maintain.\n' >&2; exit 2 ;;
esac
