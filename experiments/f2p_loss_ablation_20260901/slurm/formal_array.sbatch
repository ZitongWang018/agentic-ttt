#!/usr/bin/env bash
#SBATCH --job-name=f2p-formal
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/f2p_loss_ablation_20260901/logs/formal_%A_%a.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/f2p_loss_ablation_20260901"
matrix="${EXPERIMENT_ROOT}/configs/formal_matrix.tsv"
if [[ ! -f "${matrix}" ]]; then
  echo "Missing ${matrix}; finish pilot selection first." >&2
  exit 2
fi
line_number=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r name mode alpha beta < <(sed -n "${line_number}p" "${matrix}")

export RUN_DIR="${EXPERIMENT_ROOT}/formal/${name}"
export LOSS_MODE="${mode}"
export LOSS_ALPHA="${alpha}"
export LOSS_BETA="${beta}"
export MAX_STEPS=500
srun --export=ALL bash "${EXPERIMENT_ROOT}/slurm/common_run.sh"
