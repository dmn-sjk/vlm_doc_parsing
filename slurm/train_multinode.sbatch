#!/bin/bash
# Production run: 2 nodes x 8 A100 = 16 ranks, FSDP FULL_SHARD + LoRA on CORD-v2.
#SBATCH --job-name=vlm-cord-2n8g
#SBATCH --account=<account>
#SBATCH --partition=gpu-a100      # 2-day cap; the production partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1              # ONE task per node: accelerate spawns the 8 ranks
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/vlm_finetuning}"
cd "$PROJ_ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=qwen25vl-cord-json   # set here, never in code (credentials: ~/.netrc)
export OMP_NUM_THREADS=8                  # 64 cores / 8 ranks per node

GPUS_PER_NODE=8
NNODES="${SLURM_NNODES}"
NPROC=$(( NNODES * GPUS_PER_NODE ))

# --- rendezvous ---------------------------------------------------------------------
# accelerate_fsdp.yaml pins rdzv_backend=static, so torch.distributed needs an explicit
# master address/port rather than an elastic rendezvous. scontrol expands the compact
# nodelist ("t[0123-0124]") into real hostnames; the first one is the master.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# Derive the port from the job id so two concurrent jobs on the same node cannot collide,
# and keep it in the unprivileged ephemeral-safe range.
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export MASTER_ADDR MASTER_PORT

# --- NCCL on the cluster (8x A100 per node, single Mellanox HCA) --------------------------
# mlx5_0 / ib0 were read off the cluster, not assumed.
export NCCL_SOCKET_IFNAME=ib0   # bootstrap over IPoIB; without this NCCL can pick `ext`
                                # or `svc-eth` and the two nodes never find each other
export NCCL_IB_HCA=mlx5_0       # the node's only InfiniBand HCA
export NCCL_IB_DISABLE=0        # force the IB verbs transport (not TCP over IPoIB)
export NCCL_DEBUG=INFO          # so the log proves "NET/IB" was chosen, not "NET/Socket"
export NCCL_DEBUG_SUBSYS=INIT,NET   # ... without the per-collective firehose
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # a hung collective aborts instead of wedging
                                          # the job until the wall-clock limit
export TORCH_DISTRIBUTED_DEBUG=OFF        # DETAIL is very slow; turn on only to debug

echo "job=${SLURM_JOB_ID} nodes=${NNODES} nproc=${NPROC} master=${MASTER_ADDR}:${MASTER_PORT}"
scontrol show hostnames "$SLURM_JOB_NODELIST"

TRAIN_CFG="${TRAIN_CFG:-configs/sft_multinode.yaml}"

# `\$SLURM_PROCID` is escaped on purpose: it must be expanded inside each srun task
# (0 on the first node, 1 on the second), not by this batch shell.
LAUNCHER="accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  --num_processes ${NPROC} \
  --num_machines ${NNODES} \
  --machine_rank \$SLURM_PROCID \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT}"

# `bash -c` re-parses its argument, so a bare `$*` silently strips one layer of quoting:
# `--set train.eval_strategy="no"` arrives as `...=no`, YAML reads bare `no` as boolean
# false, and TrainingArguments dies with "False is not a valid IntervalStrategy"
# (observed in job 3108801). printf %q re-escapes each argument so it survives the extra
# parse intact. The single-node script passes "$@" straight to accelerate and needs none
# of this.
FWD_ARGS=$(printf '%q ' "$@")

srun --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -c "${LAUNCHER} -m src.train.train_sft --config ${TRAIN_CFG} ${FWD_ARGS}"

echo "train job ${SLURM_JOB_ID} finished with status $?"
