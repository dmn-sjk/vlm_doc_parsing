#!/usr/bin/env bash
# Build the project venv on $SCRATCH (home is small and nearly full).
# Run on a LOGIN NODE -- compute nodes may have no outbound internet.
set -euo pipefail

PROJ_SCRATCH="${SCRATCH:?SCRATCH not set}/vlm_finetuning"
VENV="${PROJ_SCRATCH}/venvs/vlm"
export UV_CACHE_DIR="${PROJ_SCRATCH}/.uv-cache"

mkdir -p "${PROJ_SCRATCH}/venvs" "${UV_CACHE_DIR}"

command -v uv >/dev/null || { echo "uv not on PATH (expected ~/.local/bin/uv)"; exit 1; }

uv venv --python 3.12 "${VENV}"

VPY="${VENV}/bin/python"
uvpip() { uv pip install --python "${VPY}" "$@"; }

# torch first, from the cu128 index (A100 = sm80, cu128 wheels cover it)
uvpip torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128

uvpip \
  "transformers==5.16.1" \
  "trl==1.12.0" \
  "peft==0.20.0" \
  "accelerate==1.14.0" \
  "datasets==4.8.5" \
  "qwen-vl-utils==0.0.14" \
  "wandb" \
  "pillow" \
  "pyyaml" \
  "hf_transfer" \
  "huggingface_hub[cli]" \
  "python-dotenv" \
  "Levenshtein"

echo
echo "=== installed ==="
uv pip list --python "${VPY}" | grep -iE "^(torch|transformers|trl|peft|accelerate|datasets|qwen|wandb|numpy) "
echo
echo "venv: ${VENV}"
