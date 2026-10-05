#!/bin/bash
# Source from an allocated Skipjack compute node before running project tools.

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* ]]; then
    printf '%s\n' 'STOP: project setup and execution require an allocated compute node.' >&2
    return 1 2>/dev/null || exit 1
fi

export NEUROSYM_ROOT=/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics
if [[ ! -d "$NEUROSYM_ROOT" || "$(realpath -e "$NEUROSYM_ROOT")" != "$NEUROSYM_ROOT" ]]; then
    printf '%s\n' 'STOP: the expected project directory is missing or resolves elsewhere.' >&2
    return 1 2>/dev/null || exit 1
fi

# HOME remains untouched. Environments and caches live on project storage.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TMPDIR="$NEUROSYM_ROOT/.tmp"
export XDG_CACHE_HOME="$NEUROSYM_ROOT/.cache"
export XDG_CONFIG_HOME="$NEUROSYM_ROOT/.config"
export XDG_DATA_HOME="$NEUROSYM_ROOT/.local/share"
export XDG_STATE_HOME="$NEUROSYM_ROOT/.local/state"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$NEUROSYM_ROOT/.python"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"

mkdir -p "$TMPDIR"
