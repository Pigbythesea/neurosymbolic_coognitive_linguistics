#!/bin/bash
#SBATCH --job-name=neurosym-inspect
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --comment=accept_cost
#SBATCH --dependency=singleton
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/inspect-%j.log

set -euo pipefail
trap 'result=$?; printf "JOB EXIT status=%s job=%s\n" "$result" "${SLURM_JOB_ID:-unset}"' EXIT

source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PIP_CONFIG_FILE=/dev/null
printf 'Inspecting real reading data: job %s on %s\n' "$SLURM_JOB_ID" "$(hostname -s)"

if [[ -L "$NEUROSYM_ROOT/.venv" ]]; then
    printf '%s\n' 'STOP: .venv must be a directory inside this project, not a symlink.' >&2
    exit 1
fi
if [[ ! -e "$NEUROSYM_ROOT/.venv" ]]; then
    /usr/bin/python3.11 -I -B -m venv "$NEUROSYM_ROOT/.venv"
fi
if [[ ! -f "$NEUROSYM_ROOT/.venv/pyvenv.cfg" || ! -x "$NEUROSYM_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' 'STOP: the project virtual environment is incomplete.' >&2
    exit 1
fi
mkdir -p "$NEUROSYM_ROOT/artifacts"
"$NEUROSYM_ROOT/.venv/bin/python" -I -B -m pip --isolated --disable-pip-version-check install \
    --no-input --only-binary=:all: --require-virtualenv --no-compile \
    --cache-dir "$PIP_CACHE_DIR" --report artifacts/data-environment-install.json \
    -r requirements/data.txt
"$NEUROSYM_ROOT/.venv/bin/python" -I -B -u scripts/inspect_deniz.py \
    --manifest manifests/deniz-reading.json --data-root "$NEUROSYM_ROOT/data" \
    --output "$NEUROSYM_ROOT/artifacts/deniz-inspection" --scope all

printf '%s\n' 'CLUSTER DATA INSPECTION COMPLETE'
