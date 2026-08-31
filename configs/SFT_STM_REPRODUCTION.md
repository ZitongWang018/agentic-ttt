# Qwen3-4B LoRA SFT + Short-Term Memory reproduction

This configuration reproduces the paper-style SFT agent with a fixed-size short-term memory.
It is intended for the AgentOdyssey research experiments, not as a claim that every hidden
provider or hardware detail of the paper is known.

## Fixed configuration

| Item | Value | Source |
|---|---:|---|
| Backbone | `Qwen/Qwen3-4B` | Override with `--model_path /path/to/local/checkpoint` or `AGENTODYSSEY_MODEL_PATH` |
| Agent | `LoRASFTAgent` | Repository implementation |
| Provider | `huggingface` | Repository implementation |
| Short-term memory size | `5` tuples | Paper appendix / code default |
| Reflection | disabled | Vanilla SFT+STM comparison |
| Summarization | disabled | Vanilla SFT+STM comparison |
| LoRA target modules | `q_proj,k_proj,v_proj,o_proj` | Paper appendix / code |
| LoRA rank | `16` | Paper appendix / code |
| LoRA alpha | `32` | Paper appendix / code |
| LoRA dropout | `0.05` | Paper appendix / code |
| Maximum sequence length | `4096` | Current code default; paper does not clearly state this value |
| Learning rate | `5e-6` | Current code default; paper does not clearly state this value |
| Epochs per online update | `2` | Current code default; paper does not clearly state this value |
| Batch size | `2` | Current code default; paper does not clearly state this value |
| Gradient accumulation | `1` | Current code default; paper does not clearly state this value |
| Training cadence | Every 5 steps with STM size 5 | Paper appendix / code logic |
| Environment budget | `max_steps=500` | Challenge protocol; use for comparable runs |

## Run

On the Linux GPU host:

```bash
bash scripts/run_sft_stm_repro.sh remnant
GAME_NAME=mark bash scripts/run_sft_stm_repro.sh
GAME_NAME=metropolis bash scripts/run_sft_stm_repro.sh
```

The runner saves cumulative environment state, agent logs, memory, and the LoRA adapter under
`output_repro_sft_stm/`. Do not reuse a run directory for a different seed or game.

## Important unresolved details

The paper specifies the central method settings but does not fully lock down every engineering
choice needed to reproduce exact table values. Before reporting results, record the exact model
revision, dtype/quantization, Transformers/PEFT/PyTorch versions, GPU hardware, decoding settings,
optimizer settings, random seeds, invalid-action retry policy, and the exact generated game JSON.
The repository's current configuration makes the values above explicit so that later runs are
auditable rather than silently relying on defaults.
