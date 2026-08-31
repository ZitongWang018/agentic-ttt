# Agentic TTT experiment bundle

This directory contains the complete analysis bundle for the five methods implemented in this workspace:

- `SFT + Short-Term Memory`
- `Environment-prediction TTT`
- `Feedback-to-Policy TTT (F2P-TTT)`
- `Lookahead Environment-TTT`
- `Hindsight Long-Horizon TTT`

The detailed report is in [`report.md`](report.md). It keeps Metropolis, Remnant, and Mark in separate same-task tables and figures. The `raw/` directory contains the saved evaluation traces used by the analysis, while `outputs/` contains summary tables, failure cases, representative examples, and step-level diagnostics. The `figures/` directory contains the generated plots.

Hindsight logic and its completed Metropolis v1 main trajectory are included in the code and report. The large temporary Hindsight intermediate JSONL file is intentionally omitted; it is not required to run the agent and was excluded to keep the repository focused on reproducible code, main traces, and summarized analysis.

## Running the agents

The agent implementations are under `agents/parametric/`, and the corresponding Qwen3-4B configurations are under `configs/`. The shared evaluator now registers all five parametric agent types. For a local evaluation, use the repository's normal `eval.py` entry point with one of the JSON configurations and the matching generated game environment. The shell launchers under `scripts/` document the dual-GPU and monitoring commands used during the experiments; paths to local model checkpoints should be changed for the target machine.

The reported reward quantity called `net_reward_proxy` is the logged positive-event total minus `death`. It is a diagnostic aggregate, not a replacement for an official benchmark score if the benchmark defines different weights.
