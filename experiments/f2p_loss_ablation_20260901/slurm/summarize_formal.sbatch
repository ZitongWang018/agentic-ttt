#!/usr/bin/env bash
#SBATCH --job-name=f2p-summary
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/f2p_loss_ablation_20260901/logs/formal_summary_%j.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/f2p_loss_ablation_20260901"
MATRIX="${EXPERIMENT_ROOT}/configs/formal_matrix.tsv"

run_dirs=()
while IFS=$'\t' read -r name mode alpha beta; do
  if [[ "${name}" == "name" ]]; then
    continue
  fi
  run_dirs+=("${EXPERIMENT_ROOT}/formal/${name}")
done < "${MATRIX}"

if [[ "${#run_dirs[@]}" -ne 5 ]]; then
  echo "Expected 5 formal runs in ${MATRIX}, found ${#run_dirs[@]}" >&2
  exit 2
fi

"${REPO_ROOT}/.venv/bin/python" \
  "${EXPERIMENT_ROOT}/code/summarize_runs.py" \
  "${run_dirs[@]}" \
  --json "${EXPERIMENT_ROOT}/summary/formal.json" \
  --csv "${EXPERIMENT_ROOT}/summary/formal.csv"
