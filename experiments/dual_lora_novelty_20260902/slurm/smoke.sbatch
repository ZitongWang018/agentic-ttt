#!/usr/bin/env bash
#SBATCH --job-name=novelty-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/dual_lora_novelty_20260902/logs/smoke_%j.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/dual_lora_novelty_20260902"

export RUN_DIR="${EXPERIMENT_ROOT}/smoke/main"
export NOVELTY_MODE="learned_fixed_cosine"
export NOVELTY_LR="0.000005"
export NOVELTY_LAMBDA0="0.5"
export NOVELTY_DECAY_END="35"
export MAX_STEPS=12
export SEED=42
srun --export=ALL bash "${EXPERIMENT_ROOT}/slurm/common_run.sh"
