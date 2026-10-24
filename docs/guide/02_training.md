# 02 · 训练原理：模型怎样从打牌结果中学习

[架构与数据流](01_architecture.md) · [下一篇：操作手册](03_runbook.md) · [术语表](glossary.md)

> 适用范围：v1.4 `macro_decision_v1` + 宏归一化正则的 IPPO 主线。代码核对：2026-10-21。以下数字标为“示例”时不代表实测；阶段 4 配方是源码快照，不是阶段 5/6 后正式训练已冻结的声明。当前状态见 [CURRENT_EXECUTION.md](../marvel_v14_prompts/CURRENT_EXECUTION.md)。

快速定位：[先听懂](#intuition) · [模型](#model) · [PPO](#ppo) · [GAE](#gae) · [一次更新](#iteration) · [配置与配方](#config)

<a id="intuition"></a>

## 1. 不看公式也能理解的版本

模型打一批牌，训练器反复问三个问题：

1. **我当时觉得这局面有多好？** 由价值头估计。
2. **后续结果比预期好还是差？** 由奖励和 GAE 计算“优势”。
3. **哪些决定的概率应该提高或降低？** 由 PPO 在限制变化幅度的同时更新策略。

它不是“赢了就把所有动作一律加分、输了就一律减分”。优势会结合局面价值和后续回报；同一个结果在不同预期下可能产生不同学习信号。它也没有直接得到“唯一正确打法”的人工标签。

基础路径用一个共享策略控制四个座位；两队目标相反，但参数共享，每个座位只能使用自己的信息。可选对手池是另一种采样组织方式，不是基础自对弈默认已开启。

## 2. 训练数据从哪里来？

训练环境按训练数据域的 RNG 和上下文发牌，Sampler 用当前行为策略采样，收集**完整的单副牌局轨迹**。每轮用该批数据进行若干 epoch/minibatch 更新，再采集下一批；不能把任意很旧的 IPPO 轨迹当作当前 on-policy 数据无限重用。

记录内容见[架构篇字段表](01_architecture.md)：可见观测、合法候选、行动玩家、选择、旧策略/行为概率、终局、奖励、实际长度等。

### 为什么强调 on-policy？

PPO 需要比较“采样时的旧策略”与“更新中的新策略”。若采样偷偷用了另一种温度、强行动作或混入对手，却仍当成当前策略样本，概率比就没有所声称的含义。

`SamplerConfig` 因此要求 on-policy 训练温度 **恰好为 1.0**；评估的贪心/其他温度是不同模式。对手池步骤有 owner 与 train mask，冻结对手的动作不能当当前 Actor 的 on-policy 梯度样本。

源码： [sampler.py](../../guandan_marvel/runtime/sampler.py) 的 `SamplerConfig`、`rollout_step`，以及 [training/validation.py](../../guandan_marvel/training/validation.py)。

<a id="model"></a>

## 3. 模型内部做什么？

```text
每个观察者的事件历史 → Transformer / KV cache ─┐
对应观察者的结构化状态 channels ──────────────┤→ 状态特征 s
合法下一 token 的候选编码 ────────────────────→ 动作特征 a
                                                      │
                          状态—动作融合 → 策略 logits → 合法候选概率
                          状态特征 s   → Value Head  → 局面价值
```

- Transformer 帮助模型利用历史事件，而不是只看最后一手牌。
- 状态编码补充手牌、当前级牌、行动/名次等结构化信息。
- 动作编码针对当前合法候选，不直接预测所有可能的完整出牌组合。
- 非法/padding 候选被掩码排除；权重/先验也进入候选概率。
- 当前 IPPO Value Head 使用玩家信息集，不是全知裁判。

源码：[agent.py](../../guandan_marvel/model/agent.py)、[base.py](../../guandan_marvel/model/base.py)、[transformer.py](../../guandan_marvel/model/transformer.py)。

### v1.4 Actor Residual Fuse 为什么要加？

旧路径主要是逐元素乘法后接输出头：`z = s * a`。v1.4 保留它，再增加非线性残差：

```text
u = concat(s, a, s*a, |s-a|)
z_new = s*a + W_out · G(u)
logits = logit_head(z_new)
```

它让模型更灵活地表达“同一个动作在不同局面下是否合适”。独立输出投影 `W_out` 零初始化、无 bias、按规定不施加 weight decay，所以迁移之初残差为零，**首次优化前**可保留旧策略路径的输出。第一次反向主要更新输出门，后续梯度才进入残差主干；这不是训练失败。

这解释结构与迁移设计，不保证残差一定提升棋力。详见 [阶段 3 规范](../marvel_v14_prompts/03_actor_fuse_amp_checkpoint.md)。

## 4. 为什么把多个 token 合成“宏动作”？

假设一次完整出牌需要选择牌型、补充参数和具体牌，最后提交。环境经过多步，但牌手真正作出的是一个完整决定。

若按 token 独立加权，描述较长的决定会贡献更多样本，PPO 也可能只限制局部 token 变化，却不能准确限制整个决定的概率变化。宏动作把同一玩家从空前缀到 `COMMIT/PASS` 的连续策略步骤合在一起。

- `PASS` 是长度 1 的宏动作。
- 机会/环境步骤不是策略宏动作的一部分。
- 完整性被破坏或玩家在前缀中异常切换应失败，不能静默补齐。
- **宏动作长度**与**到同一玩家下次决定的距离**不是一个量。

示例：本次出牌用了 5 个 token，其他玩家合计用了 12 个 token，才再次轮到自己；两次决定起点的距离可能是 17，而不是 5。

源码：[macro_actions.py](../../guandan_marvel/training/macro_actions.py) 的 `segment_rollout_macros`。

<a id="ppo"></a>

## 5. Sequence PPO：对完整决定做一次裁剪

### 5.1 先用数字理解

**教学示例，不是当前 clip 配方**：某完整动作由三个被选 token 构成。

```text
旧策略的已选 token 概率：0.5 × 0.4 × 1.0 = 0.20
新策略的已选 token 概率：0.6 × 0.5 × 1.0 = 0.30
完整动作的概率比：0.30 / 0.20 = 1.50
```

若优势为正，希望提高它的概率；但假设本示例 `eps=0.2`，PPO 的裁剪分支到 `1.2`，不会继续按 `1.5` 放大奖励。这是稳定更新的约束，不是硬性保证所有参数变化或 KL 都小于某值。

### 5.2 公式与实现

对宏动作 `m` 的 token 集合：

```text
log π(m | S) = Σ_t log π(a_t | h_t)
r_m = exp(log π_new(m | S) - log π_old(m | S))
L_PG = mean_m max(-r_m A_m, -clip(r_m, 1-eps, 1+eps) A_m)
```

- `h_t` 包含该步玩家历史与私有动作前缀，不是假设所有 token 独立。
- 取负号是因为优化器最小化 loss。
- 每个有效宏动作作为一个策略样本，长动作不因 token 多而获得更多样本权重。
- 实际在 FP32 的对数域累加，避免直接连乘导致下溢。
- 不能把它改成“先逐 token clip 再求平均”。

宏优势在当前 minibatch 的有效可训练宏样本上做归一化；DDP 使用全局 count/sum/sumsq，不能各 rank 随意归一化后再简单平均。相关实现也为本 rank 无有效宏样本等边界保留正确的零梯度连接。

源码：[macro_ppo.py](../../guandan_marvel/training/macro_ppo.py) 的 `macro_sequence_logprob`、`compute_macro_pg_loss`。

<a id="gae"></a>

## 6. GAE：怎样把终局结果传回较早的决定？

### 6.1 直觉

Value Head 像一个对当前局势的预测。一次决定后，把收到的奖励和后继价值与原先预测比较，形成 TD 残差；GAE 再把后续残差按距离和 λ 组合，得到较平滑的优势估计。

λ 不是“胜率”，而是决定后续残差沿轨迹传播的强弱参数。当前宏决策算法的关键是：**λ 按同一玩家的一次次完整决定衰减，而不是按每个描述 token 衰减。**

### 6.2 先区分两个时钟

| 参数/量 | 当前宏决策含义 |
|---|---|
| `gamma` | 环境折扣，仍以基础 token 时间计；默认 1.0 |
| `Delta_m` | 本宏起点到同一玩家下个宏起点的真实 token 距离；包括其他人和机会步骤，排除 padding |
| `lambda_macro` | 同一玩家宏决策之间的 trace 衰减；未单设时取 `gae_lambda`，默认 0.95 |
| `c_m` | 是否存在同一玩家的可延续后继决定 |

到玩家最后一次决定时，奖励窗口覆盖其后直到真实终局，不能漏掉自己已出完牌后才产生的终局奖励。

### 6.3 公式

```text
R_m = Σ_(j=0..Delta_m-1) gamma^j · reward_p[start_m+j]
delta_m = R_m + gamma^Delta_m · c_m · V_p(next_m) - V_p(start_m)
A_m = delta_m + gamma^Delta_m · lambda_macro · c_m · A_next_m
value_target_m = V_p(start_m) + A_m
```

这里使用该行动玩家 `p` 的奖励通道；目标在 `no_grad` 下计算并 detach，不把目标计算路径当成应优化的预测分支。

旧 token trace 跨越 `Delta` 的衰减是 `(gamma * lambda_token)^Delta`，新宏 trace 是 `gamma^Delta * lambda_macro`。**这是有意更换 λ 的时钟，不是旧算法等价重排。** 即使某次宏动作只有一个 token，只要到自己下次决定还隔着别人行动，也不能假设 `Delta=1`。

源码：[macro_gae.py](../../guandan_marvel/training/macro_gae.py) 的 `compute_macro_decision_gae`；历史等价性对照见 [legacy_gae.py](../../guandan_marvel/training/legacy_gae.py)。

## 7. 奖励：不仅关心赢，也关心队伍名次

| 结果类型 | 含义 | 胜方每个座位的 raw reward | 训练尺度 |
|---|---|---:|---:|
| `DOUBLE_DOWN` | 胜队占前两名 | +3 | +1 |
| `HEAD_THIRD` | 头游与队友第三 | +2 | +2/3 |
| `HEAD_LAST` | 头游与队友末游 | +1 | +1/3 |

败方每个座位取相反数，`R_train = R_raw / 3`；不是把两个队友的同值奖励再加起来当单座位目标。非终局的普通协议步骤不因为出了一张好牌就自动获得人工 shaping 奖励。

这样赢法不同会产生不同收益，胜负符号不变。奖励尺度改变也会影响价值目标和损失尺度，旧 flat ±1 结果不能不加说明直接与新训练回报比较。

源码：[reward.py](../../guandan_marvel/reward.py)、[state.py](../../guandan/state.py)、Sampler 的奖励缩放和终局指标记录。

## 8. 总损失里还有什么？

宏决策 IPPO 的基本结构为：

```text
total = macro_PPO_loss
      + reg_coef × macro_normalized_KL
      + value_loss_coef × macro_value_loss
      + 只有明确启用且兼容的辅助项
```

- Value loss 在宏起点拟合目标，使用实现中的 Huber loss（`delta=10`），不是对宏内每个 token 都重复一份宏目标。
- 正则先对每个宏内部 token 的 KL 取平均，再对宏平均，避免长宏天然贡献更多正则。
- 实现中的参考分布由合法候选 `avail_weights` 归一化得到；等权合法候选时就是合法集合上的均匀分布。不要把字段名里的 uniform 理解成包括非法候选。
- `value_loss_coef` 默认 0.5。学习率、clip 和正则系数还有 schedule；初始值不是后期恒定值。
- NTP、对称性或 Belief 不因存在类名而自动进入损失；看实际配置和资格。对手池也有明确的组合限制。

源码：[local.py](../../guandan_marvel/training/local.py) 的 `_macro_train_iteration`，以及 `compute_macro_value_loss`、`compute_macro_normalized_kl_mean`、[schedules.py](../../guandan_marvel/training/schedules.py)。

<a id="iteration"></a>

## 9. 一次训练 iteration 的完整顺序

1. **进入更新边界**：验证 Trainer 健康，计算这一轮学习率、clip、正则系数。
2. **采样**：收集 `collect_batch_size` 对应的完整轨迹，验证真实 terminal 和 on-policy/owner 契约。
3. **分段**：识别每个完整宏动作的起止、玩家和同玩家后继。
4. **构造目标**：无梯度前向得到价值；计算宏 GAE、优势和价值目标。
5. **优化**：按牌局轨迹切 minibatch，在若干 epoch 中重新前向，计算损失、反向、梯度检查/裁剪并更新参数。
6. **完成边界**：更新 merged policy，写指标，推进 iteration、durable update 和完成局数。
7. **外层编排**：worker 根据实际预算检查保存、评估和是否开始下一轮。

一个 iteration 可以有多次 optimizer step；配置 epoch/minibatch 数也不能代替实际成功更新计数。中途发生异常，不能保存混有半轮更新的状态并伪称 durable checkpoint。

## 10. 三种训练语义不要混在一起

| `training_semantics_profile` | PPO 样本 | 优势/价值语义 | 用途 |
|---|---|---|---|
| `legacy_token_v1` | token | legacy token | 保留旧路径和兼容对照 |
| `macro_sequence_ablation_v1` | 完整宏动作 | 旧 token 优势取宏起点；价值仍 token 语义 | 隔离“只改 sequence ratio”的效果 |
| `macro_decision_v1` | 完整宏动作 | 同玩家宏 GAE、宏起点价值 | v1.4 阅读主线 |

`policy_regularizer_mode` 另行指定。代码会拒绝未定义的组合；VRPO 保留其 Q-critic/auxiliary 语义，不能请求宏 IPPO profile 后悄悄退化或套用。

配置校验见 [config.py](../../guandan_marvel/config.py) 的 `validate_training_semantics` 和 `validate_opponent_pool_combinations`。

<a id="config"></a>

## 11. 配置应该怎样读？

### 11.1 分清三层来源

1. `TrainConfig/ModelConfig/PPOConfig`：构造默认值与兼容约束；目前仍有 `algo_name="vrpo"`、`v1.3`、legacy token、AMP off 等默认值；device 默认还会根据 CUDA 可见性选择，正式操作必须显式指定。
2. 交付 builder / profile：选择某套实验配方，可能覆盖上面的默认值。
3. **某次运行的 resolved config、semantic contract 和 manifest**：确认真正运行了什么。

因此 v1.4 主线至少要明确 `algo_name="ippo"`、`model.architecture`、`ppo.training_semantics_profile`、`ppo.policy_regularizer_mode` 和 `device`。不能只写 `TrainConfig(game=...)` 就声称启动了 v1.4 主线，也不能只改 JSON 中一个字符串就宣称旧 checkpoint 可以 exact-resume。

### 11.2 阶段 4 交付 builder 的配方快照

以下直接对应 [phase4_delivery.py](../../guandan_marvel/kaggle/phase4_delivery.py) 的 `_training_block`，用于理解已有设计。**不是阶段 5/6 后最终生产配置已经冻结，不能直接覆盖旧交付执行。**

| 参数 | builder 中的值/选择 | 意义 |
|---|---|---|
| model | `v1.4-macro-fuse`，history 128 / feature 256，4 层 4 头历史编码 | 正式尺寸结构 |
| semantics / regularizer | `macro_decision_v1` / `macro_normalized_kl_uniform_v1` | 明确选择宏决策路径 |
| rollout / collect | global 4 / 20 | 并行活动环境与每轮完整采集批量不是同一概念 |
| minibatches / agent epochs | 8 / 4 | 优化组织参数；不保证恰好 8 个非空 minibatch |
| base lr / clip / reg | `4e-4` / `0.02` / `0.1` | 还会按 schedule 改变 |
| AMP / compile / NTP | on / off / weight 0 | 三个独立选择 |
| seed / rank stride | `570700` / `1000003` | 跨 rank 数据身份的一部分 |
| checkpoint / evaluation interval | 50 / 250 | 由外层编排按协议执行 |
| target iterations | 1500 | 总体目标，不是本次获准自动跑到的次数 |
| wall budget | 12600 秒 | 3.5 小时，为会话收尾留余量 |
| context | 13 等级 × 4 leader 的 7A schedule + curriculum | 默认该交付不采样贡还上下文 |

minibatch 实际大小由本 rank 的轨迹数与 `ceil(B/mini_batches)` 决定，不能把配置数字直接乘起来当完成更新数。训练分布由 [context_schedule.py](../../guandan_marvel/context_schedule.py) 版本化，并随实际发牌记录 provenance；贡还审计通过和实际分布覆盖仍需分别核对。

## 12. AMP、DDP、EMA 分别解决什么？

- **AMP**：让适合的模型算子使用较低精度，节省算力/显存；宏概率比、GAE、KL、目标与损失等关键数学保持 FP32。当前 CUDA 路径使用 FP16 与 scaler，不把 CPU BF16 测试当 CUDA 认证。
- **DDP**：多 GPU 各采一部分轨迹，并同步梯度。必须正确按全局有效宏数归一化；Windows 本地/Gloo 测试不等于 Kaggle 双 T4/NCCL。
- **merged/EMA policy**：区分可训练的 `agent_resume` 和合并输出的 `agent_merged`；权重、合并规则和恢复状态都要保留。不是因为有 merged 名字就一定做了非零强度的平滑；阶段 4 配方 `agent_merging_beta=0.0`。

## 13. 看哪些指标，不能怎么解读？

| 类别 | 例子 | 用来发现什么 |
|---|---|---|
| 计数 | `statistics/iterations`、`statistics/durable_updates`、`statistics/completed_games_total` | 是否真完成，而非只启动 |
| 健康 | 非法动作、截断、非有限 loss/梯度、宏分段违规 | 数据与数值是否有效 |
| 学习 | `macro/pg_loss`、`macro/value_ev`、KL、clip、梯度范数 | 更新是否退化或过猛；字段以实际日志为准 |
| 性能 | `perf/rollout_seconds`、`perf/learner_seconds` | 时间花在采样还是学习 |
| 棋力 | 独立评估胜率、分差、cluster CI | 是否优于指定对照 |

Value EV 的目标方差退化时可能未定义；不能把缺失强行记为一个“好分数”。自我对弈中队伍零和和策略共同变化，也使训练平均回报不能直接当固定对手棋力曲线。

## 14. 最常见的误解

- “把数据跑得更久就会更强”：需要评估；配方或学习信号不合适时可能停滞/退化。
- “有价值头就可以直接搜索”：先有资格与真实搜索实现，不能越过 Value Gate。
- “之前模型训练过，新代码也算训练过”：代码/语义改变要重新绑定证据。
- “加载了旧权重就是续训”：可能只是 warm start，见[操作手册](03_runbook.md)。
- “复制 Bootcamp 的全部建议就是当前方案”：它包含历史版本和已不允许的路线，见[文档地图](00_overview.md)。

更深的数学设计与反例标准见 [阶段 2 规范](../marvel_v14_prompts/02_macro_sequence_ppo_and_gae.md)、[独立数学验证记录](../marvel_v14_plan_verification.md)；实际算法判断仍以当前实现和测试为准。
