#!/usr/bin/env bash
# Download the model and dataset into $HF_HOME.
set -euo pipefail
source "$(dirname "$0")/env.sh"

# The ids come from the Published run config.
PYTHONUNBUFFERED=1 python - "${PROJ_ROOT}/configs/sft_1node.yaml" <<'PY'
import sys
from huggingface_hub import snapshot_download
from datasets import load_dataset
from vlm_doc_parsing.config import load_config

cfg = load_config(sys.argv[1])
model_id = cfg.model.base_model
print(f"--- snapshot_download {model_id} ---")
p = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py"])
print("model at:", p)

ds_id = cfg.data.dataset_id
print(f"--- load_dataset {ds_id} ---")
d = load_dataset(ds_id)
print(d)
PY
