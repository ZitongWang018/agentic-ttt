#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGENTODYSSEY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${LOOKAHEAD_RUN_ROOT:-${ROOT}/runs/lookahead_env_ttt}"
PY="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
MODEL_PATH="${AGENTODYSSEY_MODEL_PATH:-Qwen/Qwen3-4B}"
CONFIG="${AGENT_CONFIG:-${ROOT}/configs/lookahead_environment_ttt_qwen3_4b.json}"
mkdir -p "$RUN_ROOT"

start_one() {
  local game="$1"; local gpu="$2"
  local run_dir="$RUN_ROOT/${game}_qwen3-4b_lookahead-environment-ttt_seed42_gpu${gpu}"
  local log="$RUN_ROOT/${game}_gpu${gpu}.log"
  if pgrep -af "eval_feedback_to_policy_ttt.py.*--game_name ${game}.*${run_dir}" >/dev/null; then
    echo "already running ${game} gpu${gpu}"
    return
  fi
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" "$ROOT/scripts/eval_feedback_to_policy_ttt.py" \
    --game_name "$game" \
    --agents_config "$CONFIG" \
    --model_path "$MODEL_PATH" \
    --max_steps 500 \
    --seed 42 \
    --run_dir "$run_dir" \
    --cumulative_config_save \
    --debug >"$log" 2>&1 < /dev/null &
  echo $! > "$RUN_ROOT/${game}.pid"
  echo "started ${game} gpu${gpu} pid $(cat "$RUN_ROOT/${game}.pid")"
}

start_one remnant 0
start_one metropolis 1
