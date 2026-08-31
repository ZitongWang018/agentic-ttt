#!/usr/bin/env bash
set -u

ROOT="${AGENTODYSSEY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
MODEL_PATH="${AGENTODYSSEY_MODEL_PATH:-Qwen/Qwen3-4B}"
CFG="${AGENT_CONFIG:-$ROOT/configs/f2p_qwen3_4b.json}"
BASE="${F2P_RUN_ROOT:-$ROOT/runs/f2p}"
LOG=$BASE/queue_gpu0.log
mkdir -p "$BASE"
cd "$ROOT"
exec >> "$LOG" 2>&1
echo "[$(date)] queue started"

wait_for_absent() {
  local pattern="$1"
  while pgrep -af "$pattern" >/dev/null 2>&1; do
    echo "[$(date)] waiting for: $pattern"
    sleep 30
  done
}

wait_for_absent "eval.py --game_name mark .*mark_seed42_from438_gpu0"

run_one() {
  local game="$1"
  local gpu="0"
  if [ "$game" = "metropolis" ]; then gpu="1"; fi
  local run_dir="$BASE/${game}_qwen3-4b_feedback-to-policy-ttt_seed42_gpu${gpu}_retry4"
  local run_log="$BASE/${game}_gpu${gpu}_retry4.log"
  echo "[$(date)] starting $game"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" "$ROOT/scripts/eval_feedback_to_policy_ttt.py" \
      --game_name "$game" \
      --agents_config "$CFG" \
      --model_path "$MODEL_PATH" \
      --max_steps 500 \
      --seed 42 \
      --run_dir "$run_dir" \
      --cumulative_config_save \
      --debug > "$run_log" 2>&1 &
  local pid=$!
  echo "[$(date)] $game pid=$pid run_dir=$run_dir"
  wait "$pid"
  local rc=$?
  echo "[$(date)] $game finished rc=$rc"
  return $rc
}

run_one remnant &
pid_remnant=$!
run_one metropolis &
pid_metropolis=$!
wait "$pid_remnant"
rc_remnant=$?
wait "$pid_metropolis"
rc_metropolis=$?
echo "[$(date)] remnant rc=$rc_remnant metropolis rc=$rc_metropolis"
if [ "$rc_remnant" -eq 0 ] && [ "$rc_metropolis" -eq 0 ]; then
  echo "[$(date)] starting mark after remnant/metropolis"
  bash "$BASE/launch_mark_f2p_gpu0.sh"
fi
echo "[$(date)] queue finished"
