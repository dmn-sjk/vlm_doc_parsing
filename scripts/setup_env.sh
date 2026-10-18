#!/usr/bin/env bash
# Build the venv (at UV_PROJECT_ENVIRONMENT) exactly from uv.lock. Run on a login node:
# compute nodes may have no outbound internet, and the venv is ~7 GB of wheels.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
if [ -n "${UV_CACHE_DIR:-}" ]; then
  mkdir -p "${UV_CACHE_DIR}"
fi

command -v uv >/dev/null || { echo "uv not on PATH (expected ~/.local/bin/uv)"; exit 1; }

cd "${PROJ_ROOT}"

# --frozen: fail rather than re-resolve; training behaviour depends on the pinned versions.
uv sync --frozen --no-dev "$@"

echo
echo "=== installed ==="
uv pip list --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
  | grep -iE "^(torch|torchvision|transformers|trl|peft|accelerate|datasets|numpy) "
echo
echo "venv: ${UV_PROJECT_ENVIRONMENT}"
echo "activate it for any command by sourcing scripts/env.sh"
