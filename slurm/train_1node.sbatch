#!/bin/bash
# Published run: 1 node x 8 A100 (40 GB) = 8 ranks, FSDP FULL_SHARD + LoRA on CORD-v2.
# Usage: scripts/submit.sh train1n [--set key.path=value ...]

#SBATCH --job-name=vlm-cord-1n8g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              # ONE task: accelerate spawns the 8 ranks
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=logs/train1n-%j.out
#SBATCH --error=logs/train1n-%j.err

set -euo pipefail

# $0 is a spool copy inside a job, so find the repo via the submit dir.
PROJ_ROOT="${PROJ_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$PROJ_ROOT"
[ -f pyproject.toml ] || { echo "error: ${PROJ_ROOT} is not the repo root; submit from the repository root" >&2; exit 1; }
# shellcheck disable=SC1091
source scripts/env.sh

# Prefetched on the login node; offline turns a hub hiccup into a fast error, not a stall.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=qwen25vl-cord-json   # set here, never in code (credentials: ~/.netrc)
export PYTHONUNBUFFERED=1

# NVLink only, so no IB tuning; a hung collective should still abort, not idle to the limit.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TRAIN_CFG="${TRAIN_CFG:-configs/sft_1node.yaml}"

echo "job=${SLURM_JOB_ID} host=$(hostname) cfg=${TRAIN_CFG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Ranks from the allocation, so `--gres=gpu:2` works unedited.
NPROC="${SLURM_GPUS_ON_NODE:-$( { nvidia-smi -L 2>/dev/null || true; } | wc -l)}"
if [ "${NPROC}" -lt 1 ]; then
  echo "error: cannot determine GPU count (SLURM_GPUS_ON_NODE unset, nvidia-smi saw none)" >&2
  exit 1
fi
echo "launching ${NPROC} ranks"

# Cores per rank (floor 1); SLURM_CPUS_PER_TASK is unset without -c (e.g. `srun --pty`).
OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} / NPROC ))
export OMP_NUM_THREADS=$(( OMP_NUM_THREADS > 0 ? OMP_NUM_THREADS : 1 ))

accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  --num_processes "${NPROC}" \
  --num_machines 1 \
  -m vlm_doc_parsing.train.train_sft --config "${TRAIN_CFG}" "$@"
