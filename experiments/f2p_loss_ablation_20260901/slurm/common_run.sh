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
LOSS_MODE="${LOSS_MODE:?LOSS_MODE is required}"
LOSS_ALPHA="${LOSS_ALPHA:-1.0}"
LOSS_BETA="${LOSS_BETA:-0.0}"
PYTHON_HEADER_DIR="${EXPERIMENT_ROOT}/vendor/python3-devel-root/usr/include/python3.11"

# Disable Transformers generation graph compilation before Python starts. The
# isolated runner uses SDPA attention; Triton's small driver shim is supported
# by the private Python headers below.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Triton always builds a small CUDA driver shim, even when model generation
# uses eager attention.  Compute nodes have the Python runtime but omit
# Python.h, so expose the matching headers vendored inside this experiment.
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
  --loss_mode "${LOSS_MODE}" \
  --loss_alpha "${LOSS_ALPHA}" \
  --loss_beta "${LOSS_BETA}" \
  --game_name "${GAME_NAME}" \
  --agents_config "${EXPERIMENT_ROOT}/configs/f2p_local_qwen3_4b.json" \
  --model_path "${MODEL_PATH}" \
  --max_steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --run_dir "${RUN_DIR}" \
  --cumulative_config_save \
  --agent_memory_save_frequency 1
