#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/sunward/models/Qwen3-4B}"
GAME_NAME="${GAME_NAME:-remnant}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:?MAX_STEPS is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
NOVELTY_MODE="${NOVELTY_MODE:?NOVELTY_MODE is required}"
NOVELTY_LR="${NOVELTY_LR:-0.000005}"
NOVELTY_LAMBDA0="${NOVELTY_LAMBDA0:-0.5}"
NOVELTY_DECAY_END="${NOVELTY_DECAY_END:-350}"
NOVELTY_NEGATIVE_SELECTION="${NOVELTY_NEGATIVE_SELECTION:-uniform}"
NOVELTY_STRENGTH_MODE="${NOVELTY_STRENGTH_MODE:-absolute}"
NOVELTY_SCORE_BATCH_SIZE="${NOVELTY_SCORE_BATCH_SIZE:-5}"
NOVELTY_TRAIN_MICROBATCH_SIZE="${NOVELTY_TRAIN_MICROBATCH_SIZE:-1}"
AGENTS_CONFIG="${AGENTS_CONFIG:-${EXPERIMENT_ROOT}/configs/f2p_novelty_qwen3_4b.json}"
AGENT_MEMORY_SAVE_FREQUENCY="${AGENT_MEMORY_SAVE_FREQUENCY:-1}"
AGENT_LORA_SNAPSHOT_STEPS="${AGENT_LORA_SNAPSHOT_STEPS:-}"
PYTHON_HEADER_DIR="${REPO_ROOT}/experiments/f2p_loss_ablation_20260901/vendor/python3-devel-root/usr/include/python3.11"

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${PYTHON_HEADER_DIR}/Python.h" ]]; then
  echo "Missing vendored Python headers: ${PYTHON_HEADER_DIR}/Python.h" >&2
  exit 1
fi
export C_INCLUDE_PATH="${PYTHON_HEADER_DIR}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"

if [[ -f /home/software/nccl-env.sh ]]; then
  source /home/software/nccl-env.sh
fi

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}"
exec "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/code/run_experiment.py" \
  --novelty_mode "${NOVELTY_MODE}" \
  --novelty_rank 4 \
  --novelty_alpha 8 \
  --novelty_lr "${NOVELTY_LR}" \
  --novelty_update_frequency 5 \
  --novelty_window_size 25 \
  --novelty_batch_size 5 \
  --novelty_lambda0 "${NOVELTY_LAMBDA0}" \
  --novelty_decay_end "${NOVELTY_DECAY_END}" \
  --novelty_negative_selection "${NOVELTY_NEGATIVE_SELECTION}" \
  --novelty_strength_mode "${NOVELTY_STRENGTH_MODE}" \
  --novelty_score_batch_size "${NOVELTY_SCORE_BATCH_SIZE}" \
  --novelty_train_microbatch_size "${NOVELTY_TRAIN_MICROBATCH_SIZE}" \
  --game_name "${GAME_NAME}" \
  --agents_config "${AGENTS_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --max_steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --run_dir "${RUN_DIR}" \
  --cumulative_config_save \
  --agent_memory_save_frequency "${AGENT_MEMORY_SAVE_FREQUENCY}" \
  --agent_lora_snapshot_steps "${AGENT_LORA_SNAPSHOT_STEPS}"
