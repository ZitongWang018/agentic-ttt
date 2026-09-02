#!/usr/bin/env bash
#SBATCH --job-name=f2p-checkpoints
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/dual_lora_novelty_20260902/logs/f2p_checkpoint_rerun_%j.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/dual_lora_novelty_20260902"

export RUN_DIR="${EXPERIMENT_ROOT}/f2p_checkpoint_rerun/f2p_only_seed42"
export NOVELTY_MODE=off
export NOVELTY_LR=0.000005
export NOVELTY_LAMBDA0=0.0
export NOVELTY_DECAY_END=350
export MAX_STEPS=500
export SEED=42
export AGENT_MEMORY_SAVE_FREQUENCY=5
export AGENT_LORA_SNAPSHOT_STEPS=100,200,300,400,500

srun --export=ALL bash "${EXPERIMENT_ROOT}/slurm/common_run.sh"
