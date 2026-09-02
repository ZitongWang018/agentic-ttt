#!/usr/bin/env bash
#SBATCH --job-name=novelty-formal
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --array=0-24%5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/dual_lora_novelty_20260902/logs/formal_%A_%a.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/dual_lora_novelty_20260902"
matrix="${EXPERIMENT_ROOT}/configs/formal_matrix.tsv"
line_number=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r name mode novelty_lr lambda0 seed < <(sed -n "${line_number}p" "${matrix}")
selected_hparams="${EXPERIMENT_ROOT}/summary/selected_novelty_hparams_from_200step_pilot.tsv"
if [[ ! -f "${selected_hparams}" ]]; then
  echo "Missing pilot selection: ${selected_hparams}" >&2
  exit 2
fi
if [[ "${mode}" != "off" ]]; then
  IFS=$'\t' read -r novelty_lr lambda0 < "${selected_hparams}"
fi

export RUN_DIR="${EXPERIMENT_ROOT}/formal/${name}"
export NOVELTY_MODE="${mode}"
export NOVELTY_LR="${novelty_lr}"
export NOVELTY_LAMBDA0="${lambda0}"
export NOVELTY_DECAY_END=350
export MAX_STEPS=500
export SEED="${seed}"
srun --export=ALL bash "${EXPERIMENT_ROOT}/slurm/common_run.sh"
