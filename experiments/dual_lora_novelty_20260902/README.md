# Dual-LoRA novelty experiment

当前完成项、运行中任务、结果入口和清理记录统一列在
[`RESULTS_INDEX.md`](RESULTS_INDEX.md)。正式汇总文件均采用
`实验含义_seed_steps` 命名，不再使用含义模糊的 `formal.json` 或
`pilot.json`。

This experiment keeps the F2P task branch exactly at
`normalized_top1_a0.25_b0.25` and adds a separate rank-4 novelty LoRA.

The novelty branch keeps a FIFO multiset of the last 25 raw action strings.
Every five environment steps it samples five occurrences uniformly without
replacement and minimizes their length-normalized token log-probability under
the current decision context. No positive action, environment action list,
reward, or environment-specific state parser enters this loss.

Generation uses

```text
W = W_base + Delta_W_f2p + lambda(step) * Delta_W_novelty
```

The learned fixed-norm variants project the effective novelty update
`(alpha / rank) * B @ A` onto one global Frobenius sphere after every update.
The target radius is the F2P adapter's effective norm after its first update.
The main variant applies a cosine weight from `lambda0` to zero by step 350.

The experiment is isolated under this directory. Existing source agents and
the completed F2P ablation outputs are not modified.

## Step-100 action-memory branch

The follow-up experiment branches from the immutable F2P `step_0100` LoRA and
the exact environment state at step 100, then evaluates steps 100--149. It
compares a clean F2P continuation, exact action parameter memory, semantic
action parameter memory, and a prompt-history control.

The two parameter-memory variants use a rank-4 residual head only while the
model is generating the value of the real JSON `"action"` field:

```text
delta_z = W_up(SiLU(W_down(h_action_slot)))
```

Every five steps the head is reset to the same initialization and fitted on a
FIFO window of 25 real action-slot records. Its loss is action-token negative
unlikelihood plus `KL(base || adapted)` with coefficient 1.0. The semantic
variant greedily clusters frozen mean token embeddings at cosine threshold
0.85 and trains on a count-weighted medoid per cluster. The prompt control
places the same 25 raw actions in the user prompt and explicitly asks the
model to prefer materially different valid actions.

Entrypoints:

- `code/run_step100_action_memory.py`
- `code/action_memory_patch.py`
- `configs/action_memory_step100_matrix.tsv`
- `slurm/action_memory_step100_array.sbatch`

### Effect-target follow-up

The strength-controlled follow-up keeps the exact-action rank-4 adapter but
does not use a fixed inference scale. After each five-step refit, a bracketed
binary search selects the smallest scale whose weighted historical
action-token log-probability drops by the requested amount. The two initial
targets are 0.2 and 0.5 nats. This tests behavioral strength directly while
retaining the same KL reference loss and avoiding parameter-norm projection.

- `configs/effect_target_step100_matrix.tsv`
- `slurm/effect_target_step100_array.sbatch`
