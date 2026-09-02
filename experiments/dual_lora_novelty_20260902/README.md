# Dual-LoRA novelty experiment

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
