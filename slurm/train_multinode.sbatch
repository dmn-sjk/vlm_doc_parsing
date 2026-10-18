#!/bin/bash
# 2 nodes x 8 A100 = 16 ranks, FSDP FULL_SHARD + LoRA on CORD-v2.
# Usage: scripts/submit.sh train2n [--set key.path=value ...]
# Rendezvous and NCCL settings: docs/design-notes.md#multi-node-rendezvous.
#SBATCH --job-name=vlm-cord-2n8g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1              # ONE task per node: accelerate spawns the 8 ranks
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail

# $0 is a spool copy inside a job, so find the repo via the submit dir.
PROJ_ROOT="${PROJ_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$PROJ_ROOT"
[ -f pyproject.toml ] || { echo "error: ${PROJ_ROOT} is not the repo root; submit from the repository root" >&2; exit 1; }
# shellcheck disable=SC1091
source scripts/env.sh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=qwen25vl-cord-json   # set here, never in code (credentials: ~/.netrc)
export PYTHONUNBUFFERED=1

# Not every Slurm exports SLURM_GPUS_ON_NODE to the batch step; nodes are homogeneous.
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-$( { nvidia-smi -L 2>/dev/null || true; } | wc -l)}"
if [ "${GPUS_PER_NODE}" -lt 1 ]; then
  echo "error: cannot determine GPUs per node (SLURM_GPUS_ON_NODE unset, nvidia-smi saw none)" >&2
  exit 1
fi
# Cores per rank (floor 1); SLURM_CPUS_PER_TASK is unset without -c (e.g. `srun --pty`).
OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} / GPUS_PER_NODE ))
export OMP_NUM_THREADS=$(( OMP_NUM_THREADS > 0 ? OMP_NUM_THREADS : 1 ))

NNODES="${SLURM_NNODES}"
NPROC=$(( NNODES * GPUS_PER_NODE ))

# rdzv_backend=static needs an explicit master: the first host of the job.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# From the job id, so concurrent jobs on one node don't collide.
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export MASTER_ADDR MASTER_PORT

# NCCL_SOCKET_IFNAME / NCCL_IB_HCA come from the site profile.
export NCCL_IB_DISABLE=0        # IB verbs, not TCP over IPoIB
export NCCL_DEBUG=INFO          # the log shows "NET/IB" was chosen, not "NET/Socket"
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # a hung collective aborts, not idles to the limit
export TORCH_DISTRIBUTED_DEBUG=OFF        # DETAIL is very slow

echo "job=${SLURM_JOB_ID} nodes=${NNODES} nproc=${NPROC} master=${MASTER_ADDR}:${MASTER_PORT}"
scontrol show hostnames "$SLURM_JOB_NODELIST"

TRAIN_CFG="${TRAIN_CFG:-configs/sft_multinode.yaml}"
export NPROC NNODES TRAIN_CFG MASTER_ADDR MASTER_PORT

srun --ntasks="${NNODES}" --ntasks-per-node=1 slurm/launch_node.sh "$@"
