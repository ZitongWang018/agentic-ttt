#!/usr/bin/env bash
set -u

ROUND="${1:-v1}"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${HINDSIGHT_RUN_ROOT:-${SCRIPT_ROOT}/runs/hindsight_long_horizon_ttt_${ROUND}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
echo "time=$(date -Is)"
for game in remnant metropolis; do
  pid_file="${ROOT}/${game}.pid"
  echo "--- ${game} ---"
  if [[ -f "${pid_file}" ]]; then
    pid=$(cat "${pid_file}")
    ps -p "${pid}" -o pid=,etime=,stat=,cmd= || true
  else
    echo "pid file missing"
  fi
  agent_log=$(find "${ROOT}" -path "*${game}_qwen3-4b_hindsight-long-horizon-ttt*" -name agent_log.jsonl -print -quit 2>/dev/null || true)
  credit_log=$(find "${ROOT}" -path "*${game}_qwen3-4b_hindsight-long-horizon-ttt*" -name hindsight_intermediates.jsonl -print -quit 2>/dev/null || true)
  echo "agent_log=${agent_log:-missing} lines=$(if [[ -n "${agent_log}" ]]; then wc -l <"${agent_log}"; else echo 0; fi)"
  echo "credit_log=${credit_log:-missing} lines=$(if [[ -n "${credit_log}" ]]; then wc -l <"${credit_log}"; else echo 0; fi)"
  if [[ -n "${credit_log}" ]]; then
    "${PYTHON_BIN}" - "${credit_log}" <<'PY'
import json, pathlib, statistics, sys
p=pathlib.Path(sys.argv[1])
rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def vals(k): return [float(r[k]) for r in rows if isinstance(r.get(k),(int,float))]
def mean(k):
    v=vals(k); return round(statistics.mean(v),5) if v else None
print(json.dumps({
  "credits":len(rows), "last_source_step":rows[-1].get("source_step") if rows else None,
  "mean_e_t":mean("evidence_e_t"), "mean_R":mean("feedback_R"),
  "mean_A_t":mean("advantage_A_t"),
  "positive_A":sum(float(r.get("advantage_A_t",0))>0 for r in rows),
  "negative_A":sum(float(r.get("advantage_A_t",0))<0 for r in rows),
  "policy_updates":rows[-1].get("policy_updates_after") if rows else 0,
  "incomplete_windows":sum(bool(r.get("incomplete_window")) for r in rows),
}))
PY
  fi
  log=$(find "${ROOT}" -maxdepth 1 -name "${game}_gpu*.log" -print -quit 2>/dev/null || true)
  if [[ -n "${log}" ]]; then
    echo "fatal-log-matches:"
    grep -E "Traceback|CUDA error|out of memory|CUDACachingAllocator.*OOM|Episode finished" "${log}" | tail -n 8 || true
  fi
done
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader || true
