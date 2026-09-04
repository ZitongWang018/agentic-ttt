# Rank-Matched Sparse Tri-LoRA v2

This experiment is a causal turning-point probe for online test-time adaptation.
It uses one always-on Task adapter and a Top-1-routed bank of two Free adapters:

\[
W_t = W_0 + \Delta W_T + \lambda \Delta W_{F_{z_b}},
\qquad z_b \in \{1,2\}.
\]

- Task LoRA: rank 12, normalized F2P action-token objective.
- Free LoRA-1/2: rank 4 each, block-return E2E objective.
- Active rank: `12 + 4 = 16`, matching the rank-16 baseline.
- Stored LoRA rank: `12 + 4 + 4 = 20`.
- Routing: deterministic balanced Top-1, one Free expert per block.
- Parameter writes: Task updates only Task; a block updates only its active Free.

## Free-expert records

Each expert gets a separate JSONL file. Every record stores `(x, y, A_b)`:

- `x.action_prompt`: environment state and causal history available at the step.
- `x.realized_reasoning_prefix`: the inference-time assistant prefix immediately
  before the action value. It is conditioning context, not a supervised target.
- `y`: the action tokens actually present in the causal source trajectory.
- `advantage`: normalized future block advantage `A_b`.
- `block_return`: discounted real benchmark return `G_b`.

In offline replay, `source` is explicitly marked
`historical_f2p_offline_replay`. Six turning-point diagnostic generations are
logged separately and never enter either Free expert's training data.

## Turning-point diagnostic

At each selected point the code records six paths:

1. Base
2. Base + Task
3. Base + Free-1
4. Base + Free-2
5. Base + Task + Free-1
6. Base + Task + Free-2

The probe restores RNG state afterward so evaluation does not alter subsequent
training. This six-forward diagnostic is not the deployed policy: normal online
execution routes first and generates once with `Base + Task + selected Free`.

## Smoke command

```bash
python experiments/turning_point_trilora_20260904/turning_point_trilora_v2.py \
  --source-log /path/to/f2p/agent_log.jsonl \
  --output-dir /path/to/smoke_output \
  --name sparse-trilora-rankmatched-v2-smoke \
  --swanlab-group smoke \
  --points 21 \
  --max-replay-step 21
```

Formal turning-point tests use SwanLab group `remnant-point-test`; feasibility
checks use `smoke`.
