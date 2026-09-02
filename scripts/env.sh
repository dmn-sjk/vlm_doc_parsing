#!/usr/bin/env bash
# Shared environment for every entrypoint (login-node scripts and sbatch jobs alike).
# Source this; do not execute it.

PROJ_ROOT="${PROJ_ROOT:-$HOME/vlm_finetuning}"
PROJ_SCRATCH="${SCRATCH:?SCRATCH not set}/vlm_finetuning"

# --- caches: keep every large writer off $HOME (10 GiB quota, already ~95% full) ---
export HF_HOME="${SCRATCH}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# Xet buffers aggressively in RAM and gets OOM-killed by the login-node cgroup
# (observed: SIGKILL / exit 137 mid-download). Plain HTTP is slower but survives.
export HF_HUB_DISABLE_XET=1
export TRITON_CACHE_DIR="${PROJ_SCRATCH}/.triton"
export TORCHINDUCTOR_CACHE_DIR="${PROJ_SCRATCH}/.inductor"
export WANDB_DIR="${PROJ_SCRATCH}/wandb"
export WANDB_CACHE_DIR="${PROJ_SCRATCH}/.wandb-cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_DATASETS_CACHE}" "${WANDB_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# --- secrets: .env is gitignored; never echo these ---
if [ -f "${PROJ_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${PROJ_ROOT}/.env"
  set +a
fi

# --- venv ---
# Built by scripts/setup_env.sh from the committed pyproject.toml + uv.lock. Kept on
# $SCRATCH rather than the default .venv beside pyproject.toml, because this repo lives in
# $HOME (10 GiB quota) and the environment is ~7 GB.
export UV_PROJECT_ENVIRONMENT="${PROJ_SCRATCH}/venvs/vlm"
export UV_CACHE_DIR="${PROJ_SCRATCH}/.uv-cache"
VENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export DATASET_ID="${DATASET_ID:-naver-clova-ix/cord-v2}"
