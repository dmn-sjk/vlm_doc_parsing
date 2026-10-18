#!/bin/bash
# Evaluation: 1 GPU, generate JSON for a held-out split and score it.
# Usage: sbatch slurm/eval.sbatch <adapter_dir> [extra args for vlm_doc_parsing.eval.evaluate]
#SBATCH --job-name=vlm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -euo pipefail

# $0 is a spool copy inside a job, so find the repo via the submit dir.
PROJ_ROOT="${PROJ_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$PROJ_ROOT"
[ -f pyproject.toml ] || { echo "error: ${PROJ_ROOT} is not the repo root; submit from the repository root" >&2; exit 1; }
# shellcheck disable=SC1091
source scripts/env.sh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8         # one process; the published eval's value

# "base" scores the un-finetuned checkpoint, the baseline every adapter score needs.
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

# One process: the 7B fits on one A100, and sharding would add all-gather latency per token.
python -m vlm_doc_parsing.eval.evaluate \
  --config "${EVAL_CFG:-configs/sft_1node.yaml}" \
  "${ADAPTER_ARGS[@]}" \
  --out "${OUT}" \
  "$@"
