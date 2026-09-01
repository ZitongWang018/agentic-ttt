#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_ROOT}/../.." && pwd)"
cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" "${EXPERIMENT_ROOT}/code/test_loss_math.py"
bash "${EXPERIMENT_ROOT}/slurm/smoke.sh"
