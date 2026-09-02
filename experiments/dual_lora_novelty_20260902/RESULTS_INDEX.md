# 实验结果索引

更新时间：2026-09-03（Asia/Beijing）

## 共同设置

- 游戏：Remnant
- 模型：Qwen3-4B
- seed：当前正式比较均为 42
- F2P 来源：`f2p_loss_ablation_20260901/formal/normalized_top1_a0.25_b0.25`
- F2P loss：`normalized_logp_l2`，`alpha=0.25`，`beta=0.25`
- F2P 更新：学习率 `5e-6`，每 5 个 environment steps 更新
- novelty LoRA：rank 4；动作 FIFO window 25；每 5 步更新；每次 5 条历史动作

## 已完成

### 0. 上游 F2P loss ablation

- 原始项目：`../f2p_loss_ablation_20260901/`
- 本项目采用：`formal/normalized_top1_a0.25_b0.25`
- 该设置定义：`normalized_logp_l2`，`alpha=0.25`，`beta=0.25`
- 本项目的 F2P-only 和所有 novelty setting 都使用同一 F2P loss 配置

### 1. Novelty 超参数 pilot

- 原始结果：`pilot/`
- 规模：7 个 setting，每组 200 steps，seed42
- 汇总：`summary/hparam_pilot_seed42_200steps.{json,csv}`
- 选择结果：`summary/selected_novelty_hparams_from_200step_pilot.{json,tsv}`
- 被选参数：novelty learning rate `5e-6`，初始强度 `lambda0=0.25`

### 2. Absolute-strength 五策略对照

- 原始结果：`formal/`
- 规模：5 个 setting，每组完整 500 steps，seed42
- 汇总：`summary/absolute_strength_ablation_seed42_500steps.{json,csv}`
- setting：
  - `f2p_only_seed42`：仅 F2P，novelty 关闭
  - `random_fixed_cosine_seed42`：随机 novelty LoRA，固定范数，cosine apply weight
  - `learned_fixed_constant_seed42`：学习 novelty，固定范数，constant apply weight
  - `learned_unconstrained_cosine_seed42`：学习 novelty，不限制范数，cosine apply weight
  - `learned_fixed_cosine_seed42`：学习 novelty，固定范数，cosine apply weight

### 3. Relative-strength 与 hard-negative 对照

- Slurm array：`1867`
- 原始结果：`relative_formal/`
- 规模：2 个 setting，各完整 500 steps，seed42
- setting：`uniform_relative_seed42`、`hard_relative_seed42`
- 说明：applied novelty/F2P effective-norm 比例按 cosine 从 0.25 衰减到 0
- 汇总：`summary/relative_strength_hard_negative_seed42_500steps.{json,csv}`

### 4. Step100 action-memory 四组短实验

- Slurm array：`1899`（tasks 0--3）
- 状态：已完成，每组 50 个新 environment steps
- 原始结果：`step100_action_memory/`
- 汇总：`summary/step100_action_memory_exploration_seed42.{json,csv}`
- 报告：`summary/step100_action_memory_exploration_report.md`
- 共同分叉点：F2P `lora_checkpoints/step_0100`、environment step100、
  STM5=step95--99、最近25个真实动作、空 F2P buffer
- 运行区间：environment steps 100--149，共新增 50 steps，seed42
- 四组：
  - `f2p_control_seed42`：从同一点继续 F2P，不使用动作去重
  - `exact_action_adapter_seed42`：真实 JSON action slot 上的 rank-4 参数记忆
  - `semantic_action_adapter_seed42`：冻结 action embedding 聚类后的参数记忆
  - `prompt_history25_seed42`：把最近25个动作放入 prompt 并显式要求减少重复
- adapter：每5步从固定初始化重拟合10步；lr `1e-3`；apply scale `1.0`
- loss：历史 action-token unlikelihood + `1.0 * KL(base || adapted)`
- semantic clustering：cosine threshold `0.85`，按 cluster count 加权 medoid
- 诊断：`action_memory_intermediates.jsonl` 记录历史目标压制、当步 action
  的 base/adapted log-prob 差、KL、logit delta、gating、聚类和耗时
- 结论：prompt-history25 的 exact repeat 最低（42%，control 为 50%），
  但平均决策时间更长且 death 更多；两种固定强度参数 adapter 均未超过 control

### 5. Effect-target 参数去重诊断

- Slurm array：`1905`（tasks 0--1）
- 状态：因 token/surface escape 提前停止，分别保留 14/12 steps
- 原始结果：`step100_effect_target/`
- setting：exact adapter 的历史 action-token 目标下降 `0.2`、`0.5` nats
- 强度：每次更新自动搜索满足目标的最小 apply scale，不限制参数范数
- 结论：历史动作的目标 log-prob 降幅可精确达到，但概率质量要
  转移到拼写、格式和语言变体，invalid 分别达 50%/33.3%

### 6. F2P-only 权重快照重跑

- Slurm job：`1887`
- 状态：已完成 500 steps
- 原始结果：`f2p_checkpoint_rerun/f2p_only_seed42/`
- 设置：500 steps，seed42，novelty 关闭
- 不可覆盖 LoRA：`lora_checkpoints/step_0100`、`step_0200`、
  `step_0300`、`step_0400`、`step_0500`
- `memory/lora` 是用于恢复的滚动 checkpoint，不计入上述五组

## 文件约定

- `experiment_config.json`：单次 run 的完整设置
- `config.jsonl`：逐步环境状态
- `agent_log.jsonl`：逐 environment-step 动作、reward、invalid 与训练 trace
- `f2p_intermediates.jsonl`：F2P 中间量
- `novelty_intermediates.jsonl`：novelty 选择、loss、范数和反事实影响日志
- `action_memory_intermediates.jsonl`：step100 分叉实验的 action-slot 参数记忆日志
- `memory/lora/`：最新滚动 LoRA
- `summary/*.json`：便于程序读取的汇总
- `summary/*.csv`：便于表格分析的汇总

Step100 action-memory 探索已做精简归档：每个 run 保留
`experiment_config.json`、`agent_log.jsonl` 和
`action_memory_intermediates.jsonl`；删除可从正式日志重建的
`config.jsonl`、`f2p_intermediates.jsonl` 和滚动 `memory/` checkpoint。

## 已清理的无效结果

以下内容已删除，不参与任何结论：初始 smoke、mode smoke、relative smoke、
hard-negative v2 smoke、顺序混杂的 job 1863 结果、失败的 relative 尝试日志、
未真正启动的旧 formal array 日志、被 job1899 取代的 job1895 临时副本，
以及 Python bytecode 缓存。pilot 与所有正式结果均保留。
