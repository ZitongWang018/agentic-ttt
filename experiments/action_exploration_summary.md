# F2P 与动作探索实验汇总

更新时间：2026-09-03（Asia/Beijing）

## 实验范围

本文档统计目前保留且可用于分析的 32 个结果行，包括：

- 6 组 30-step F2P loss 系数 pilot。
- 5 组 500-step F2P loss formal ablation。
- 7 组 200-step novelty 超参 pilot。
- 5 组 500-step absolute-strength novelty 正式对照。
- 2 组 500-step relative-strength/hard-negative 对照。
- 1 组 500-step F2P checkpoint 重跑。
- 4 组从 step100 分叉的 50-step action-memory 对照。
- 2 组提前停止的 effect-target 诊断。

已删除的 smoke、启动失败和被替代任务不计入结论。除早期 F2P loss
ablation 外，当前正式 novelty 实验均为 Remnant、Qwen3-4B、seed42，
F2P 使用 `normalized_logp_l2(alpha=0.25, beta=0.25)`。

## 指标定义

- `U`：不同原始动作字符串数，越高通常表示探索范围越广。
- `R`：全局重复率，定义为 `1 - U / N`。
- `R25`：当前动作是否出现在最近 25 个动作中，更直接衡量局部循环。
- `Inv`：无效动作率，越低越好。
- `P`：除 death 外各 score counter 之和；它不是严格的标量 reward，
  只建议用于同组实验比较。
- `K/UK/D`：`kill / unique_kill / death`。
- `Delta logp`：被惩罚历史动作的平均 log-prob 变化；负数表示动作确实受到压制。

## 1. F2P loss 选择

### 1.1 30-step 系数 pilot

这部分主要用于检查训练稳定性。各组前 30 步的行为指标相同，且没有
reward event，差异主要体现在 loss 和梯度规模。

| Setting | N | U | R | Inv | Mean loss | Mean grad norm | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| alpha=.25, beta=.25 | 30 | 28 | 6.7% | 0% | 0.441 | 1.386 | 最温和 |
| alpha=.5, beta=.25 | 30 | 28 | 6.7% | 0% | 0.608 | 2.054 | 进入 formal |
| alpha=1, beta=.25 | 30 | 28 | 6.7% | 0% | 0.959 | 3.292 | 进入 formal |
| alpha=1, beta=.5 | 30 | 28 | 6.7% | 0% | 1.216 | 4.077 | 未进入 formal |
| alpha=1, beta=1 | 30 | 28 | 6.7% | 0% | 1.702 | 5.343 | 梯度偏大 |
| alpha=2, beta=.5 | 30 | 28 | 6.7% | 0% | 2.021 | 7.045 | 梯度最大 |

### 1.2 500-step F2P formal ablation

| Setting | 含义 | U | R | Inv | P | K/UK/D | Mean loss / grad | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `original` | 原始带权 F2P loss | 134 | 73.2% | 5.2% | 13 | 6/2/10 | 2.681 / - | 整体较弱 |
| `no_w` | 去掉 `w_t` 的 action NLL | 140 | 72.0% | 4.0% | 21 | 13/4/10 | 6.365 / 46.144 | reward 提升但梯度很大 |
| `normalized alpha=.25 beta=.25` | 归一化 NLL + logp-L2 | **156** | **68.8%** | **3.6%** | **31** | **22/6/11** | 0.193 / 1.364 | **最终采用** |
| `normalized alpha=.5 beta=.25` | 更强 NLL | 139 | 72.2% | 4.8% | 29 | 18/6/13 | 0.297 / 2.078 | 不如 alpha=.25 |
| `normalized alpha=1 beta=.25` | 进一步增强 NLL | 120 | 76.0% | 6.2% | 15 | 9/2/8 | 0.553 / 3.916 | 明显退化 |

`alpha=.25, beta=.25` 同时具有最多 unique action、最低 invalid 和最高
productive score，因此后续实验都使用这一版 F2P loss。

## 2. 200-step novelty 超参 pilot

共同机制：rank-4 novelty LoRA，固定 effective norm，apply weight 从 `lambda0`
cosine 衰减到 0；FIFO window=25，每 5 步从历史中采 5 条训练。

表中括号分别对应前 140 步或后 60 步。

| LR / lambda0 | U 总/前140 | R 总/前140 | Inv | P 总/后60 | K/UK/D | Delta logp | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| F2P only | 73/65 | 63.5%/53.6% | 5.0% | 6/0 | 1/1/4 | - | pilot control |
| 2.5e-6 / .25 | 78/67 | 61.0%/52.1% | 17.0% | 5/0 | 1/1/5 | -0.0148 | invalid 太高 |
| 2.5e-6 / .5 | 64/55 | 68.0%/60.7% | 11.0% | 4/3 | 0/0/3 | -0.0066 | 探索退化 |
| 2.5e-6 / 1.0 | 77/61 | 61.5%/56.4% | **3.5%** | 9/6 | 2/2/3 | -0.0139 | reward 好，探索一般 |
| **5e-6 / .25** | **87/69** | **56.5%/50.7%** | 7.0% | **9/3** | 6/1/2 | -0.0151 | **pilot 选中** |
| 5e-6 / .5 | 69/53 | 65.5%/62.1% | 15.0% | 7/6 | 2/2/3 | -0.0205 | 压制强但行为差 |
| 5e-6 / 1.0 | 75/58 | 62.5%/58.6% | 4.0% | 3/2 | 1/1/5 | -0.0158 | reward 较差 |

Pilot 中 `LR=5e-6, lambda0=.25` 相对 control 将 U 从 73 提升到 87，
将重复率从 63.5% 降到 56.5%，death 从 4 降到 2。但这个改善没有在后续
500-step fixed-cosine 正式实验中复现。

## 3. 500-step novelty 正式实验

实际执行配置以各 run 的 `experiment_config.json` 为准：`lambda0=.25`。
`formal_matrix.tsv` 仍残留旧的 `.5`，但这不是实际跑出结果的配置。

`P 总/后150` 用于观察 novelty 衰减后是否恢复 exploitation。

| Setting | 机制 | U | R / R25 | Inv | P 总/后150 | K/UK/D | Delta logp | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| F2P only | matched control | 159 | 68.2% / 54.8% | 6.8% | 29/15 | 21/5/6 | - | 主对照 |
| Learned fixed constant | 学习；固定范数；lambda=.25 恒定 | 157 | 68.6% / 54.6% | 4.2% | **30/15** | 19/6/7 | -0.0300 | 与 control 基本持平 |
| Learned fixed cosine | 学习；固定范数；.25 衰减到0 | 146 | 70.8% / 55.4% | **2.4%** | 23/8 | 12/4/12 | -0.0303 | 重复和任务表现都变差 |
| Learned unconstrained cosine | 学习；不限范数；.25 衰减到0 | **168** | **66.4% / 48.6%** | 4.6% | 20/3 | 12/4/10 | **-0.1165** | 探索强，但 exploitation 受损 |
| Random fixed cosine | 随机 LoRA，不学习 | 149 | 70.2% / 54.0% | 4.4% | 16/2 | 8/3/13 | +0.0002 | 明显较差 |
| Relative uniform | novelty/F2P norm 比例 .25 衰减到0；随机历史负样本 | 126 | 74.8% / 58.0% | 4.0% | 13/6 | 6/3/12 | -0.0320 | 表现很差 |
| Relative hard | 同上；选择 task-logp 最高的历史动作 | 151 | 69.8% / **50.6%** | 4.6% | **31/14** | 19/6/11 | -0.0184 | **最好的参数去重版本** |
| F2P checkpoint rerun | 无 novelty；保存每100步权重 | **171** | **65.8% / 49.4%** | 5.0% | **31/15** | 21/6/11 | - | 强基线及 step100 来源 |

关键观察：

- Unconstrained LoRA 确实能强烈压制历史动作，提升 unique 并降低 R25，
  但后 150 步 productive score 只有 3，说明探索扰乱了任务能力。
- Hard-negative 明显优于 uniform-negative，但尚未证明超过纯 F2P。
- F2P checkpoint 重跑自身就达到 171 unique、49.4% R25 和 31 productive，
  优于大部分参数 novelty 设置。

## 4. Step100 action-memory 局部分支

四组从完全相同的 step100 环境状态和 F2P LoRA 分叉，各运行 50 步。
`R25` 包含分叉前的 25 个真实动作，因此比全局 R 更适合这个实验。

| Setting | 机制 | N | U | 全局R / R25 / 语义R25 | Inv | P | K/UK/D | Delta 历史logp | 秒/步 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F2P control | 不使用去重 | 50 | 29 | 42% / 50% / 52% | 4% | **6** | 2/2/1 | - | 40.9 | 局部对照 |
| Exact adapter | rank-4 action-slot residual head | 50 | 23 | 54% / 60% / 60% | 2% | 3 | 1/1/2 | -0.0092 | 40.5 | 比 control 更重复 |
| Semantic adapter | embedding 聚类后训练 medoid | 50 | 26 | 48% / 56% / 58% | 2% | 2 | 1/1/3 | -0.0275 | 42.8 | 有压制，但行为未改善 |
| Prompt-history25 | 每步加入最近25个动作并软性要求不重复 | 50 | **30** | **40% / 42% / 48%** | 4% | 4 | 2/2/3 | - | **55.1** | 重复最低，但时延和 death 增加 |

Prompt-history25 是目前唯一在相同 step100 分支上降低重复的方法，但 U
只从 29 增加到 30，平均每步多约 14 秒，death 从 1 增加到 3。

### 4.1 Effect-target 强压制诊断

| Setting | 状态 | N | U | R25 | Inv | 目标/实际下降 | Mean scale | KL | 结果 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| drop 0.2 nat | 提前停止 | 14 | 13 | 14.3% | **50.0%** | .2/.200 | 7.51 | .450 | 拼写/格式逃逸 |
| drop 0.5 nat | 提前停止 | 12 | 12 | 8.3% | **33.3%** | .5/.500 | 10.15 | .786 | 更强的分布扰动 |

这两组表面上的低重复不是有效探索，而是生成了 `pickup(torch)`、
`pick uptorch`、中文动作、错误实体拼写等无效变体，因此按失败路线处理。

## 5. 综合结论

1. **F2P 基线本身仍然最可靠**：
   `normalized alpha=.25 beta=.25` 是明确最优的 F2P loss。
2. **Hard-negative relative 是当前最好的参数空间候选**：
   明显优于 uniform-negative，但尚未稳健超过 F2P。
3. **Prompt-history25 有真实但很小的局部去重作用**：
   代价是约 35% 的额外推理时延和更多 death。
4. **Unconstrained novelty 能提高探索，但会损害后期利用**。
5. **Fixed-cosine 在 pilot 中的改善没有在 500 步中复现**。
6. **Exact/semantic action-slot adapter 和 effect-target 的当前形式不值得继续盲目加强**：
   前者行为影响太弱，后者会产生 token/surface escape。
7. **当前正式 novelty 结果只有 seed42**。F2P 两次 500-step 运行已经出现
   `159 -> 171 unique`、`death 6 -> 11` 的明显波动，因此小幅改善不能当作稳健结论。

## 6. 原始结果入口

- [F2P pilot](f2p_loss_ablation_20260901/summary/pilot_attempt2.csv)
- [F2P formal](f2p_loss_ablation_20260901/summary/formal.csv)
- [Novelty pilot](dual_lora_novelty_20260902/summary/hparam_pilot_seed42_200steps.csv)
- [Absolute-strength 500-step](dual_lora_novelty_20260902/summary/absolute_strength_ablation_seed42_500steps.csv)
- [Relative-strength/hard-negative](dual_lora_novelty_20260902/summary/relative_strength_hard_negative_seed42_500steps.csv)
- [Step100 action-memory](dual_lora_novelty_20260902/summary/step100_action_memory_exploration_seed42.csv)
- [Step100 action-memory 报告](dual_lora_novelty_20260902/summary/step100_action_memory_exploration_report.md)
- [Dual-LoRA 结果索引](dual_lora_novelty_20260902/RESULTS_INDEX.md)
