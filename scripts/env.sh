#!/usr/bin/env bash
# Shared environment for every entrypoint, login node and jobs alike. Source it.
#
# Precedence: your environment > site profile > the defaults below. An empty variable counts
# as unset, so `VAR= cmd` cannot clear a profile value.

(return 0 2>/dev/null) || { echo "error: source scripts/env.sh, do not run it" >&2; exit 1; }

# sbatch scripts set PROJ_ROOT from SLURM_SUBMIT_DIR before sourcing.
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Site profile (configs/site/). An explicit SITE_ENV must exist: falling back to defaults
# would put ~16 GB of caches under the repo. Relative paths are from the repo root.
if [ -n "${SITE_ENV:-}" ]; then
  case "${SITE_ENV}" in /*) ;; *) SITE_ENV="${PROJ_ROOT}/${SITE_ENV}" ;; esac
  if [ ! -f "${SITE_ENV}" ]; then
    echo "error: SITE_ENV=${SITE_ENV} does not exist" >&2
    return 1
  fi
else
  SITE_ENV="${PROJ_ROOT}/site.env"
fi
if [ -f "${SITE_ENV}" ]; then
  set -a
  # shellcheck disable=SC1090
  if ! . "${SITE_ENV}"; then
    set +a
    echo "error: site profile ${SITE_ENV} failed to load" >&2
    return 1
  fi
  set +a
fi

# Secrets: .env is gitignored; never echo these.
if [ -f "${PROJ_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${PROJ_ROOT}/.env"
  set +a
fi

# Caches: every large writer goes under the work dir.
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

# The venv built by scripts/setup_env.sh.
: "${UV_PROJECT_ENVIRONMENT:=${PROJ_ROOT}/.venv}"
export UV_PROJECT_ENVIRONMENT
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
