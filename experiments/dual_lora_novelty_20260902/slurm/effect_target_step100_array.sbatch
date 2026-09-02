#!/usr/bin/env bash
#SBATCH --job-name=effect-target-s100
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --open-mode=append
#SBATCH --output=experiments/dual_lora_novelty_20260902/logs/effect_target_step100_%A_%a.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/dual_lora_novelty_20260902"
MATRIX="${EXPERIMENT_ROOT}/configs/effect_target_step100_matrix.tsv"

ROW="$(awk -F '\t' -v line="$((SLURM_ARRAY_TASK_ID + 2))" 'NR == line {print $0}' "${MATRIX}")"
if [[ -z "${ROW}" ]]; then
  echo "No matrix row for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
IFS=$'\t' read -r SETTING TARGET_LOGPROB_DROP <<< "${ROW}"

RUN_DIR="${EXPERIMENT_ROOT}/step100_effect_target/${SETTING}_seed42"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/sunward/models/Qwen3-4B}"
AGENTS_CONFIG="${EXPERIMENT_ROOT}/configs/f2p_novelty_qwen3_4b.json"
PYTHON_HEADER_DIR="${REPO_ROOT}/experiments/f2p_loss_ablation_20260901/vendor/python3-devel-root/usr/include/python3.11"

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export C_INCLUDE_PATH="${PYTHON_HEADER_DIR}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
if [[ -f /home/software/nccl-env.sh ]]; then
  source /home/software/nccl-env.sh
fi

cd "${REPO_ROOT}"
exec srun --export=ALL "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/code/run_step100_action_memory.py" \
  --action_memory_mode exact_adapter \
  --action_memory_rank 4 \
  --action_memory_lr 0.001 \
  --action_memory_update_frequency 5 \
  --action_memory_window_size 25 \
  --action_memory_optimization_steps 10 \
  --action_memory_reference_beta 1.0 \
  --action_memory_apply_scale 1.0 \
  --action_memory_target_logprob_drop "${TARGET_LOGPROB_DROP}" \
  --action_memory_semantic_threshold 0.85 \
  --action_memory_microbatch_tokens 8 \
  --branch_step 100 \
  --game_name remnant \
  --agents_config "${AGENTS_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --max_steps 150 \
  --seed 42 \
  --run_dir "${RUN_DIR}" \
  --cumulative_config_save \
  --resume_from_step 100 \
  --agent_memory_save_frequency 5
