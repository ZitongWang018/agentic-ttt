#!/usr/bin/env bash
set -u

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${LOOKAHEAD_RUN_ROOT:-${SCRIPT_ROOT}/runs/lookahead_env_ttt}"
STAMP=$(date '+%F %T %Z')
alert=0
echo "[$STAMP] lookahead environment TTT monitor"

for game in remnant metropolis; do
  run_dir="$RUN_ROOT/${game}_qwen3-4b_lookahead-environment-ttt_seed42_gpu$([ "$game" = remnant ] && echo 0 || echo 1)"
  log="$RUN_ROOT/${game}_gpu$([ "$game" = remnant ] && echo 0 || echo 1).log"
  agent="$run_dir/qwen3_4b_lookahead_env_ttt/agent_log.jsonl"
  intermediate="$run_dir/qwen3_4b_lookahead_env_ttt/lookahead_intermediates.jsonl"
  pidfile="$RUN_ROOT/${game}.pid"
  pid="$(test -f "$pidfile" && cat "$pidfile" || true)"
  alive=0
  test -n "$pid" && kill -0 "$pid" 2>/dev/null && alive=1
  steps=0
  test -f "$agent" && steps=$(wc -l < "$agent")
  intermediate_steps=0
  test -f "$intermediate" && intermediate_steps=$(wc -l < "$intermediate")
  echo "$game alive=$alive agent_steps=$steps intermediate_steps=$intermediate_steps log=$log"
  if [ "$steps" -ne "$intermediate_steps" ] && [ "$steps" -gt 0 ]; then
    echo "ALERT $game agent/intermediate length mismatch"; alert=1
  fi
  if grep -Eqi 'Traceback|OutOfMemory|CUDA error|Killed|RuntimeError|Exception' "$log" 2>/dev/null; then
    echo "ALERT $game error signature in log"; alert=1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
fi
df -h "${DISK_PATH:-${SCRIPT_ROOT}}" | tail -1
exit "$alert"
