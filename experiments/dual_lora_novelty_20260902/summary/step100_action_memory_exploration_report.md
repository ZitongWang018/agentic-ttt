# Step100 action-memory exploration

Generated: 2026-09-03T02:48:10.110500+08:00

All runs branch from the same F2P environment/LoRA checkpoint at step 100.
Repeat rates include the 25 actions preceding the branch point.

## Completed 50-step comparison

| Setting | Unique | Exact repeat | Semantic repeat | Invalid | Kill | Death |
|---|---:|---:|---:|---:|---:|---:|
| exact_action_adapter | 23 | 60.0% | 60.0% | 1 | 1 | 2 |
| f2p_control | 29 | 50.0% | 52.0% | 2 | 2 | 1 |
| prompt_history25 | 30 | 42.0% | 48.0% | 2 | 2 | 3 |
| semantic_action_adapter | 26 | 56.0% | 58.0% | 1 | 1 | 3 |

Prompt-history25 reduced exact repetition most (42%) and produced the most unique actions (30),
but increased latency and deaths. Neither fixed-scale parameter adapter beat the F2P control.

## Stopped effect-target diagnostic

| Setting | Steps | Exact repeat | Invalid | Target/actual drop | Mean scale | Mean KL |
|---|---:|---:|---:|---:|---:|---:|
| exact_target_drop0.2 | 14 | 14.3% | 7 (50.0%) | 0.2/0.200 | 7.506 | 0.450 |
| exact_target_drop0.5 | 12 | 8.3% | 4 (33.3%) | 0.5/0.500 | 10.151 | 0.786 |

Both constraints were met on stored historical action-slot states, but the probability mass
moved toward spelling, punctuation, and language variants rather than new semantic actions.
Examples include `defender`, `pickup(torch)`, `pick uptorch`,
`attack goblins_warror_1`, and `攻击 goblinWarriot_1`.
The runs were therefore stopped early and are evidence against strengthening this token-level
action-slot objective without a semantic or planning-level mechanism.
