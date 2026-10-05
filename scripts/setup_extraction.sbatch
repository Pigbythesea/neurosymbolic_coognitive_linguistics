#!/bin/bash
#SBATCH --job-name=neurosym-models
#SBATCH --account=tshu2_scai01
#SBATCH --qos=jhu
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --comment=accept_cost
#SBATCH --dependency=singleton
#SBATCH --chdir=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
#SBATCH --output=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/models-%j.log

set -euo pipefail
trap 'result=$?; printf "JOB EXIT status=%s job=%s\n" "$result" "${SLURM_JOB_ID:-unset}"' EXIT
source /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/cluster_env.sh
cd "$NEUROSYM_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PIP_CONFIG_FILE=/dev/null
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
printf 'SETUP job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname -s)"
/usr/bin/python3.11 -I -B artifacts/install_extraction_bundle.py artifacts/extraction-source.zip
if [[ "${1:-}" == "--source-only" ]]; then
    printf '%s\n' 'EXTRACTION SOURCE UPDATE COMPLETE'
    printf '%s\n' 'Ready for user submission: sbatch /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/extract_models.sbatch'
    exit 0
fi
if [[ -L .venv-extraction ]]; then
    printf '%s\n' 'STOP: extraction environment is a symlink.' >&2
    exit 1
fi
if [[ ! -e .venv-extraction ]]; then
    /usr/bin/python3.11 -I -B -m venv .venv-extraction
fi
PYTHON="$NEUROSYM_ROOT/.venv-extraction/bin/python"
# The system venv seed can reject normalized dependency names on the Torch index.
# Upgrade the installer first, then obtain typing_extensions directly from PyPI.
"$PYTHON" -I -B -m pip --version
"$PYTHON" -I -B -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: \
    --require-virtualenv --no-compile --cache-dir "$PIP_CACHE_DIR" \
    --report artifacts/extraction-pip-bootstrap.json --index-url https://pypi.org/simple \
    --upgrade pip==26.2.1
"$PYTHON" -I -B -m pip --version
"$PYTHON" -I -B -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: \
    --require-virtualenv --no-compile --cache-dir "$PIP_CACHE_DIR" \
    --report artifacts/extraction-typing-bootstrap.json --index-url https://pypi.org/simple \
    typing_extensions==4.16.0
"$PYTHON" -I -B -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: \
    --require-virtualenv --no-compile --cache-dir "$PIP_CACHE_DIR" \
    --report artifacts/extraction-torch-install.json \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -I -B -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: \
    --require-virtualenv --no-compile --cache-dir "$PIP_CACHE_DIR" \
    --report artifacts/extraction-environment-install.json --index-url https://pypi.org/simple \
    -r requirements/extraction.txt
"$PYTHON" -I -B -m pip --isolated check
"$PYTHON" -I -B -m pip --isolated freeze > artifacts/extraction-environment.freeze.txt
"$PYTHON" -I -B -u scripts/validate_extraction.py
"$PYTHON" -I -B -u scripts/download_extraction_models.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON" -I -B -u scripts/prepare_tokenization.py
printf '%s\n' 'MODEL DOWNLOAD, HASH VERIFICATION AND FULL-CORPUS TOKENIZATION COMPLETE'
printf '%s\n' 'Ready for user submission: sbatch /weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/scripts/extract_models.sbatch'
