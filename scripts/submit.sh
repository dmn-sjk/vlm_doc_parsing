#!/usr/bin/env bash
# Submit one of the project's jobs with account/partition taken from the site profile.
#
# Usage: scripts/submit.sh [--sbatch-opt[=value] ...] <train1n|train2n|eval> [script args...]
#
#   scripts/submit.sh train1n
#   scripts/submit.sh --test-only train2n
#   scripts/submit.sh --time=00:25:00 train2n --set train.max_steps=5 --set train.eval_steps=5 \
#     --set 'train.output_dir=${SITE_WORK_DIR}/runs/short-2n' --set train.report_to=none
#   scripts/submit.sh eval <adapter_dir> --split test
#
# Options before the job name go to sbatch as bare flags or `--opt=value` (not
# `--opt value`). Everything after the job name goes to the sbatch script. A Short run sets
# its own output_dir and report_to=none so it never overwrites the published adapter.
# Plain `mkdir -p logs; sbatch -A <acct> -p <part> slurm/<file>.sbatch` works too.
set -euo pipefail

# readlink -f: run through a symlink, the repo is where the real file lives.
REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${REPO_ROOT}/scripts/submit.sh"
  exit "${1:-0}"
}

# sbatch records the cwd as SLURM_SUBMIT_DIR, which is how the jobs find the repo.
cd "${REPO_ROOT}"

sbatch_opts=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage ;;
    --*) sbatch_opts+=("$1"); shift ;;
    -*) echo "error: only long --opt[=value] sbatch options are supported: $1" >&2; usage 2 ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || usage 2
job="$1"; shift

case "${job}" in
  train1n) script=slurm/train_1node.sbatch ;;
  train2n) script=slurm/train_multinode.sbatch ;;
  eval)    script=slurm/eval.sbatch ;;
  *) echo "error: unknown job '${job}'" >&2; usage 2 ;;
esac

# Only account and partition, in a subshell: the job sources env.sh itself.
site=$(
  # shellcheck disable=SC1091
  source scripts/env.sh || exit 1
  printf '%s\n%s\n' "${SITE_ACCOUNT:-}" "${SITE_PARTITION:-}"
)
# `$(...)` strips trailing newlines, so an empty value can leave read at EOF (status 1).
{ read -r account || true; read -r partition || true; } <<< "${site}"

# Clusters with a default account/partition need neither.
cmd=(sbatch)
[ -n "${account}" ] && cmd+=(-A "${account}")
[ -n "${partition}" ] && cmd+=(-p "${partition}")
cmd+=("${sbatch_opts[@]}" "${script}" "$@")

# sbatch fails silently if the --output directory is missing.
mkdir -p logs

printf '+'; printf ' %q' "${cmd[@]}"; printf '\n'
exec "${cmd[@]}"
