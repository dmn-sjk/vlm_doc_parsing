#!/usr/bin/env bash
# Shared environment for every entrypoint (login-node scripts and sbatch jobs alike).
# Source this; do not execute it.
#
# Precedence: your environment > site profile > the defaults below. Nothing here overwrites
# a variable that is already set. Works with no profile and no $SCRATCH (plain workstation).

# Derived from this file's location rather than hardcoded. sbatch scripts set PROJ_ROOT
# themselves (from SLURM_SUBMIT_DIR) before sourcing; either way it is the repo root.
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- site profile: account, partitions, storage, NCCL interfaces (see configs/site/) ---
SITE_ENV="${SITE_ENV:-${PROJ_ROOT}/site.env}"
if [ -f "${SITE_ENV}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${SITE_ENV}"
  set +a
fi

# --- secrets: .env is gitignored; never echo these ---
if [ -f "${PROJ_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${PROJ_ROOT}/.env"
  set +a
fi

# --- caches: keep every large writer under the work dir ---
: "${SITE_WORK_DIR:=${PROJ_ROOT}/.work}"
: "${HF_HOME:=${SITE_WORK_DIR}/huggingface}"
: "${HF_DATASETS_CACHE:=${HF_HOME}/datasets}"
: "${TRITON_CACHE_DIR:=${SITE_WORK_DIR}/.triton}"
: "${TORCHINDUCTOR_CACHE_DIR:=${SITE_WORK_DIR}/.inductor}"
: "${WANDB_DIR:=${SITE_WORK_DIR}/wandb}"
: "${WANDB_CACHE_DIR:=${SITE_WORK_DIR}/.wandb-cache}"
export SITE_WORK_DIR HF_HOME HF_DATASETS_CACHE TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  WANDB_DIR WANDB_CACHE_DIR
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_DATASETS_CACHE}" "${WANDB_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# --- venv ---
# Built by scripts/setup_env.sh from the committed pyproject.toml + uv.lock. uv's own
# default location unless the site profile moves it (e.g. off a small $HOME).
: "${UV_PROJECT_ENVIRONMENT:=${PROJ_ROOT}/.venv}"
export UV_PROJECT_ENVIRONMENT
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export DATASET_ID="${DATASET_ID:-naver-clova-ix/cord-v2}"
