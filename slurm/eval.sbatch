#!/bin/bash
# Evaluation: 1 GPU, generate JSON for a held-out split and score it.
# Usage: sbatch slurm/eval.sbatch <adapter_dir> [extra args for vlm_doc_parsing.eval.evaluate]
# Account/partition are deliberately not set here: scripts/submit.sh passes them from the
# site profile, or pass `sbatch -A <acct> -p <part>` / set SBATCH_ACCOUNT, SBATCH_PARTITION
# (precedence per `man sbatch`: CLI > SBATCH_* env > #SBATCH).
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

# Inside a job $0 is a copy in SLURM's spool dir, so the repo is found via the submit dir.
PROJ_ROOT="${PROJ_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$PROJ_ROOT"
[ -f pyproject.toml ] || { echo "error: ${PROJ_ROOT} is not the repo root; submit from the repository root" >&2; exit 1; }
# shellcheck disable=SC1091
source scripts/env.sh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8         # one process; matches the value the published eval used

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
python -m vlm_doc_parsing.eval.evaluate \
  --config "${EVAL_CFG:-configs/sft_1node.yaml}" \
  "${ADAPTER_ARGS[@]}" \
  --out "${OUT}" \
  "$@"
