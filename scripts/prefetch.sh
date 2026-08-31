#!/usr/bin/env bash
# Pre-download model + dataset into the shared HF cache on $SCRATCH.
# MUST run on a login node: compute nodes have no reliable outbound internet,
# so training runs with HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1.
set -euo pipefail
source "$(dirname "$0")/env.sh"

python - <<'PY'
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

model_id = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
print(f"--- snapshot_download {model_id} ---", flush=True)
p = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py"])
print("model at:", p, flush=True)

ds_id = os.environ.get("DATASET_ID", "naver-clova-ix/cord-v2")
print(f"--- load_dataset {ds_id} ---", flush=True)
d = load_dataset(ds_id)
print(d, flush=True)
PY
