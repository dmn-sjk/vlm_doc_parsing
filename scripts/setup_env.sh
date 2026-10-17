#!/usr/bin/env bash
# Reproduce the project environment from pyproject.toml + uv.lock.
#
# Run on a LOGIN NODE: compute nodes have no reliable outbound internet, and the venv is
# ~7 GB of wheels.
#
# The venv goes wherever UV_PROJECT_ENVIRONMENT points: uv's default `.venv` next to
# pyproject.toml unless the site profile moves it (configs/site/cluster.env puts it on
# purged scratch storage). If it disappears, this script rebuilds it byte-identically from
# uv.lock, which is the reason the lock is committed.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
if [ -n "${UV_CACHE_DIR:-}" ]; then
  mkdir -p "${UV_CACHE_DIR}"
fi

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
