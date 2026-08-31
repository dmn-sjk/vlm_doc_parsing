#!/bin/bash
# Smoke test: 1 GPU, 5 steps, 16 samples. Catches data/trainer/PEFT/save bugs in minutes
# instead of 40 minutes into a 2-node queue wait (see CLAUDE.md "Smoke test before scale").
#SBATCH --job-name=vlm-smoke
#SBATCH --account=<account>
#SBATCH --partition=gpu-short          # 1 h cap, short queue -- smoke tests only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/vlm_finetuning}"
cd "$PROJ_ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

# Compute nodes are assumed to have no outbound internet; everything is prefetched into
# $HF_HOME on the login node. Offline mode turns a silent 60 s DNS hang into a fast error.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=qwen25vl-cord-json     # set here, never in code
export WANDB_MODE=disabled                  # the smoke config also sets report_to=none
export OMP_NUM_THREADS=8

echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# One-off check of the "compute nodes are offline" assumption (PROGRESS.md records it as
# assumed, not verified). Never fails the job.
echo "--- outbound internet probe ---"
curl -sS -m 8 -o /dev/null -w "huggingface.co -> HTTP %{http_code}\n" https://huggingface.co || \
  echo "huggingface.co -> unreachable (compute node is offline, as assumed)"

accelerate launch \
  --config_file configs/accelerate_single.yaml \
  -m src.train.train_sft \
  --config configs/sft_smoke.yaml \
  "$@"

echo "smoke job ${SLURM_JOB_ID} finished with status $?"
