#!/usr/bin/env bash
#SBATCH --job-name=novelty-relative
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/dual_lora_novelty_20260902/logs/relative_%A_%a.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/dual_lora_novelty_20260902"
matrix="${EXPERIMENT_ROOT}/configs/relative_hard_matrix.tsv"
line_number=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r name negative_selection < <(
  sed -n "${line_number}p" "${matrix}"
)

run_group="${RUN_GROUP:-relative_formal}"
export RUN_DIR="${EXPERIMENT_ROOT}/${run_group}/${name}"
export NOVELTY_MODE=learned_fixed_cosine
export NOVELTY_LR=0.000005
export NOVELTY_LAMBDA0=0.25
export NOVELTY_DECAY_END=350
export NOVELTY_NEGATIVE_SELECTION="${negative_selection}"
export NOVELTY_STRENGTH_MODE=relative_task_norm
export NOVELTY_SCORE_BATCH_SIZE=5
export NOVELTY_TRAIN_MICROBATCH_SIZE=2
export AGENTS_CONFIG="${EXPERIMENT_ROOT}/configs/f2p_novelty_qwen3_4b_fast.json"
export AGENT_MEMORY_SAVE_FREQUENCY=5
export MAX_STEPS="${MAX_STEPS_OVERRIDE:-500}"
export SEED=42
srun --export=ALL bash "${EXPERIMENT_ROOT}/slurm/common_run.sh"
