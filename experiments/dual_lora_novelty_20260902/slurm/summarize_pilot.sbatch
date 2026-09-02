#!/usr/bin/env bash
#SBATCH --job-name=novelty-summary
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/dual_lora_novelty_20260902/logs/pilot_summary_%j.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/dual_lora_novelty_20260902"
cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" \
  "${EXPERIMENT_ROOT}/code/summarize_runs.py" \
  "${EXPERIMENT_ROOT}/pilot" \
  --output-prefix "${EXPERIMENT_ROOT}/summary/hparam_pilot_seed42_200steps"
exec "${REPO_ROOT}/.venv/bin/python" \
  "${EXPERIMENT_ROOT}/code/select_pilot.py" \
  --summary "${EXPERIMENT_ROOT}/summary/hparam_pilot_seed42_200steps.json" \
  --output-json "${EXPERIMENT_ROOT}/summary/selected_novelty_hparams_from_200step_pilot.json" \
  --output-tsv "${EXPERIMENT_ROOT}/summary/selected_novelty_hparams_from_200step_pilot.tsv"
