#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-v1}"
ROOT="${AGENTODYSSEY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${HINDSIGHT_RUN_ROOT:-${ROOT}/runs/hindsight_long_horizon_ttt_${ROUND}}"
CONFIG="${AGENT_CONFIG:-${ROOT}/configs/hindsight_long_horizon_ttt_qwen3_4b.json}"
PY="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
MODEL_PATH="${AGENTODYSSEY_MODEL_PATH:-Qwen/Qwen3-4B}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}"

launch_one() {
  local game="$1"
  local gpu="$2"
  local run_dir="${RUN_ROOT}/${game}_qwen3-4b_hindsight-long-horizon-ttt_seed42_gpu${gpu}"
  local log_path="${RUN_ROOT}/${game}_gpu${gpu}_${ROUND}.log"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PY}" "${ROOT}/scripts/eval_feedback_to_policy_ttt.py" \
    --game_name "${game}" \
    --agents_config "${CONFIG}" \
    --model_path "${MODEL_PATH}" \
    --max_steps 500 \
    --seed 42 \
    --run_dir "${run_dir}" \
    --cumulative_config_save \
    --debug >"${log_path}" 2>&1 < /dev/null &
  echo $! >"${RUN_ROOT}/${game}.pid"
  echo "launched ${game} on GPU${gpu}, pid=$(cat "${RUN_ROOT}/${game}.pid"), log=${log_path}"
}

launch_one remnant 0
launch_one metropolis 1
echo "run_root=${RUN_ROOT}"
