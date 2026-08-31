#!/bin/bash
# Evaluation: 1 GPU, generate JSON for a held-out split and score it.
# Usage: sbatch slurm/eval.sbatch <adapter_dir> [extra args for src.eval.evaluate]
#SBATCH --job-name=vlm-eval
#SBATCH --account=<account>
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/vlm_finetuning}"
cd "$PROJ_ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=qwen25vl-cord-json
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8

# Pass the literal string "base" to score the un-finetuned checkpoint. A fine-tuned score
# in isolation is uninterpretable -- the baseline is what turns it into a claim.
ADAPTER="${1:?usage: sbatch slurm/eval.sbatch <adapter_dir|base> [args...]}"
shift || true

if [ "${ADAPTER}" = "base" ]; then
  ADAPTER_ARGS=()
  OUT="${PROJ_ROOT}/logs/eval_base.json"
else
  ADAPTER_ARGS=(--adapter "${ADAPTER}")
  OUT="${ADAPTER}/eval_results.json"
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} adapter=${ADAPTER}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Single process, no accelerate launcher: generation is sequential and the merged 7B fits
# on one A100. Sharding it would only add all-gather latency per token.
python -m src.eval.evaluate \
  --config "${EVAL_CFG:-configs/sft_1node.yaml}" \
  "${ADAPTER_ARGS[@]}" \
  --out "${OUT}" \
  "$@"

echo "eval job ${SLURM_JOB_ID} finished with status $?"
