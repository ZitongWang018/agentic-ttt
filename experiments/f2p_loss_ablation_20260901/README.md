# F2P loss ablation

This directory is self-contained experiment code and output storage for five
formal F2P runs:

1. Original signed F2P loss.
2. Action NLL without `w_t`.
3. The best three normalized `(alpha, beta)` settings from a short pilot.

The normalized loss is

```text
L = alpha * (-mean(log p(a_t | H_t)))
    + beta * ||log p(a_t | H_t)||_2
```

Both terms are divided by robust scales measured on the first update batch;
the scales are then frozen for the rest of the run.  No source file outside
this experiment directory is modified by the experiment implementation.

All GPU entrypoints use Slurm.  Outputs are separated into `smoke/`, `pilot/`,
and `formal/`.

The cluster compute image omits `Python.h`, although Triton requires it to
build its small CUDA driver shim at runtime.  A matching TencentOS
`python3-devel` RPM is downloaded and extracted under `vendor/`; job scripts
add that private include directory through `C_INCLUDE_PATH` without changing
the system Python installation.

For F2P scoring, Qwen3's `logits_to_keep` interface restricts the LM head to
the exact action-token suffix.  This is mathematically equivalent to slicing
the same suffix from full-sequence logits, while avoiding a roughly 1 GB
temporary full-vocabulary tensor near the 4096-token context limit.

Pilot and formal arrays are requeueable and append Slurm logs.  Environment
state, F2P buffer/scales, and the LoRA adapter are checkpointed after every
completed transition, so a requeued or manually resubmitted task continues
from the last coherent step.

The formal array has an `afterok` summary job that writes `summary/formal.json`
and `summary/formal.csv` only after all five runs finish successfully.
