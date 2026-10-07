#!/bin/bash
#SBATCH --job-name=neurosym-analysis-setup
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --comment=accept_cost
#SBATCH --dependency=singleton
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/analysis-setup-%j.log

set -euo pipefail
trap 'result=$?; printf "JOB EXIT status=%s job=%s\n" "$result" "${SLURM_JOB_ID:-unset}"' EXIT
source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PIP_CONFIG_FILE=/dev/null
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
/usr/bin/python3.11 -I -B artifacts/install_analysis_bundle.py artifacts/analysis-source.zip
if [[ -L .venv-analysis ]]; then
    printf '%s\n' 'STOP: analysis environment must not be a symlink.' >&2
    exit 1
fi
if [[ ! -e .venv-analysis ]]; then
    /usr/bin/python3.11 -I -B -m venv .venv-analysis
fi
PYTHON="$NEUROSYM_ROOT/.venv-analysis/bin/python"
"$PYTHON" -I -B -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: --require-virtualenv --no-compile \
    --cache-dir "$PIP_CACHE_DIR" --index-url https://pypi.org/simple --upgrade pip==26.2.1 typing_extensions==4.16.0
"$PYTHON" -I -B -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: --require-virtualenv --no-compile \
    --cache-dir "$PIP_CACHE_DIR" --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
"$PYTHON" -I -B -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: --require-virtualenv --no-compile \
    --cache-dir "$PIP_CACHE_DIR" --index-url https://pypi.org/simple -r requirements/analysis.txt --report artifacts/analysis-cluster-install.json
"$PYTHON" -I -B -m pip --isolated check
"$PYTHON" -I -B -m pip --isolated freeze > artifacts/analysis-cluster.freeze.txt
CODE_ROOT=$("$PYTHON" -I -B -c 'import json; print(json.load(open(".analysis-source.json"))["code_root"])')
"$PYTHON" -I -B -u "$CODE_ROOT/scripts/run_analysis.py" preflight
printf '%s\n' 'ANALYSIS ENVIRONMENT PREPARED; no scientific fits submitted.'
