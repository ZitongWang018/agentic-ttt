#!/usr/bin/env bash
#SBATCH --job-name=novelty-formal-summary
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/dual_lora_novelty_20260902/logs/formal_summary_%j.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/dual_lora_novelty_20260902"
cd "${REPO_ROOT}"
exec "${REPO_ROOT}/.venv/bin/python" \
  "${EXPERIMENT_ROOT}/code/summarize_runs.py" \
  "${EXPERIMENT_ROOT}/formal" \
  --output-prefix "${EXPERIMENT_ROOT}/summary/formal"
