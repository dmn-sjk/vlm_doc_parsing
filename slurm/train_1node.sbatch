#!/bin/bash
# Production run: 1 node x 8 A100 (40 GB) = 8 ranks, FSDP FULL_SHARD + LoRA on CORD-v2.
#
# Why one node: CORD-v2's train split is 800 examples. Three epochs at global batch 32 is
# 75 optimizer steps -- roughly 20 minutes of compute. Spreading that over 16 GPUs halves
# an already-short run while doubling the queue wait and adding inter-node all-reduce
# latency. The multi-node path is proven separately by slurm/train_multinode.sbatch.
# Account/partition are deliberately not set here: scripts/submit.sh passes them from the
# site profile, or pass `sbatch -A <acct> -p <part>` / set SBATCH_ACCOUNT, SBATCH_PARTITION
# (precedence per `man sbatch`: CLI > SBATCH_* env > #SBATCH).
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

# Inside a job $0 is a copy in SLURM's spool dir, so the repo is found via the submit dir.
PROJ_ROOT="${PROJ_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$PROJ_ROOT"
[ -f pyproject.toml ] || { echo "error: ${PROJ_ROOT} is not the repo root; submit from the repository root" >&2; exit 1; }
# shellcheck disable=SC1091
source scripts/env.sh

# Everything is prefetched into $HF_HOME on the login node. Even where compute nodes do
# have outbound internet, pinning offline keeps the run reproducible and turns a hub
# hiccup into a fast error instead of a stall.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=qwen25vl-cord-json   # set here, never in code (credentials: ~/.netrc)

# Single node: NCCL stays on NVLink, so none of the IB tuning in train_multinode.sbatch
# applies. Async error handling still matters -- a hung collective should abort, not idle
# until the wall clock expires.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TRAIN_CFG="${TRAIN_CFG:-configs/sft_1node.yaml}"

echo "job=${SLURM_JOB_ID} host=$(hostname) cfg=${TRAIN_CFG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Derive the rank count from what SLURM actually granted rather than hardcoding 8, so the
# same script works for a reduced-GPU debug allocation (--gres=gpu:2) without editing it.
NPROC="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "launching ${NPROC} ranks"

# Split the task's cores across the ranks (floor 1).
OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NPROC ))
export OMP_NUM_THREADS=$(( OMP_NUM_THREADS > 0 ? OMP_NUM_THREADS : 1 ))

accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  --num_processes "${NPROC}" \
  --num_machines 1 \
  -m src.train.train_sft --config "${TRAIN_CFG}" "$@"

echo "train job ${SLURM_JOB_ID} finished with status $?"
