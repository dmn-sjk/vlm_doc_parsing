#!/usr/bin/env bash
# Reproduce the project environment from pyproject.toml + uv.lock.
#
# Run on a LOGIN NODE: compute nodes have no reliable outbound internet, and the venv is
# ~7 GB of wheels.
#
# The environment is deliberately NOT the default `.venv` next to pyproject.toml -- this
# repo lives in $HOME, which is a 10 GiB quota already ~95% full. UV_PROJECT_ENVIRONMENT
# redirects it to $SCRATCH. Note $SCRATCH is purged storage: if the venv disappears, this
# script rebuilds it byte-identically from uv.lock, which is the reason the lock is
# committed.
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJ_SCRATCH="${SCRATCH:?SCRATCH not set}/vlm_finetuning"

export UV_PROJECT_ENVIRONMENT="${PROJ_SCRATCH}/venvs/vlm"
export UV_CACHE_DIR="${PROJ_SCRATCH}/.uv-cache"
mkdir -p "${PROJ_SCRATCH}/venvs" "${UV_CACHE_DIR}"

command -v uv >/dev/null || { echo "uv not on PATH (expected ~/.local/bin/uv)"; exit 1; }

cd "${PROJ_ROOT}"

# --frozen: install exactly what uv.lock says and fail rather than silently re-resolving.
# Reproducibility is the whole point -- several behaviours this project depends on are
# version-specific (see the comment in pyproject.toml).
uv sync --frozen --no-dev "$@"

echo
echo "=== installed ==="
uv pip list --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
  | grep -iE "^(torch|torchvision|transformers|trl|peft|accelerate|datasets|numpy) "
echo
echo "venv: ${UV_PROJECT_ENVIRONMENT}"
echo "activate it for any command by sourcing scripts/env.sh"
