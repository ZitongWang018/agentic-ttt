#!/usr/bin/env bash
set -euo pipefail

# Reproduction entry point for the paper-style Qwen3-4B
# LoRA SFT + Short-Term Memory agent.
#
# Usage:
#   bash scripts/run_sft_stm_repro.sh remnant
#   GAME_NAME=mark MAX_STEPS=500 bash scripts/run_sft_stm_repro.sh

GAME_NAME="${1:-${GAME_NAME:-remnant}}"
MAX_STEPS="${MAX_STEPS:-500}"
SEED="${SEED:-42}"
CONFIG="${AGENT_CONFIG:-configs/sft_stm_qwen3_4b.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output_repro_sft_stm}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
  echo "No usable Python interpreter found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

"${PYTHON_BIN}" eval.py \
  --game_name "${GAME_NAME}" \
  --agents_config "${CONFIG}" \
  --max_steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --cumulative_config_save \
  --debug
