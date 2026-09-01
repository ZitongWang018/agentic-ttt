#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_one() {
  local name="$1"
  local mode="$2"
  local alpha="$3"
  local beta="$4"
  RUN_DIR="${EXPERIMENT_ROOT}/smoke/attempt5/${name}" \
  LOSS_MODE="${mode}" \
  LOSS_ALPHA="${alpha}" \
  LOSS_BETA="${beta}" \
  MAX_STEPS=6 \
    bash "${EXPERIMENT_ROOT}/slurm/common_run.sh"
}

run_one original original 1.0 0.0
run_one no_w no_w 1.0 0.0
run_one normalized normalized_logp_l2 1.0 0.5
