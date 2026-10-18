#!/bin/bash
# Production run: 2 nodes x 8 A100 = 16 ranks, FSDP FULL_SHARD + LoRA on CORD-v2.
# Account/partition are deliberately not set here: scripts/submit.sh passes them from the
# site profile, or pass `sbatch -A <acct> -p <part>` / set SBATCH_ACCOUNT, SBATCH_PARTITION
# (precedence per `man sbatch`: CLI > SBATCH_* env > #SBATCH).
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

# Inside a job $0 is a copy in SLURM's spool dir, so the repo is found via the submit dir.
PROJ_ROOT="${PROJ_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$PROJ_ROOT"
[ -f pyproject.toml ] || { echo "error: ${PROJ_ROOT} is not the repo root; submit from the repository root" >&2; exit 1; }
# shellcheck disable=SC1091
source scripts/env.sh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=qwen25vl-cord-json   # set here, never in code (credentials: ~/.netrc)

# Read off the allocation, not written down. SLURM_GPUS_ON_NODE is not exported to the
# batch step by every Slurm version/config, hence the fallback (nodes are homogeneous, so
# the batch host's count holds for all of them).
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-$( { nvidia-smi -L 2>/dev/null || true; } | wc -l)}"
if [ "${GPUS_PER_NODE}" -lt 1 ]; then
  echo "error: cannot determine GPUs per node (SLURM_GPUS_ON_NODE unset, nvidia-smi saw none)" >&2
  exit 1
fi
# Split the task's cores across the ranks accelerate spawns on each node (floor 1).
# SLURM_CPUS_PER_TASK is unset when run without -c (e.g. `srun --pty`), hence nproc.
OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} / GPUS_PER_NODE ))
export OMP_NUM_THREADS=$(( OMP_NUM_THREADS > 0 ? OMP_NUM_THREADS : 1 ))

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

# --- NCCL ---------------------------------------------------------------------------
# Interface/HCA selection (NCCL_SOCKET_IFNAME, NCCL_IB_HCA) is a site fact and comes from
# the site profile via env.sh; left unset, NCCL auto-detects. See configs/site/cluster.env.
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
# of this. With no args printf would still emit `''`, an empty argv element argparse
# rejects, hence the guard.
FWD_ARGS=""
[ $# -eq 0 ] || FWD_ARGS=$(printf '%q ' "$@")

srun --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -c "${LAUNCHER} -m vlm_doc_parsing.train.train_sft --config ${TRAIN_CFG} ${FWD_ARGS}"
