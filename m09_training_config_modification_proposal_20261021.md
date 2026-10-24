# M09 训练配置修改方案（提案）

- 日期：2026-10-21
- 状态：**提案，未实施**。按项目规矩（CLAUDE.md）需用户确认后才落地代码；启动训练需另行授权。
- 依据：《m09_vs_fabledan_assessment_20261021.md》的评估结论 + 本项目自身训练数据 + 已发表 RL 文献。
- 分级标记：**【必须】**= 本项目自身数据证明已造成损失；**【建议】**= 领域标准实践/低成本高期望；**【实验】**= 需 A/B 判定。

---

## 总表（一页速查）

| # | 配置项 | 现值（代码位置） | 建议值 | 级别 |
|---|---|---|---|---|
| 1 | `base_clip_coef` | 0.02，且衰减 → iter2000 实际 **0.005**；stage2 预备 0.01→0.0025（`config.py`、`schedules.py`、stage2 脚本） | **0.2，固定不衰减** | 【必须】 |
| 2 | `clip_coef_decay_power` | 1.0（`config.py::ScheduleConfig`） | **0（clip 不参与衰减）** | 【必须】 |
| 3 | 训练分布 | 固定 `Rank.TWO` + 无贡品 + leader=0（M09 profile / plan） | **208 上下文等权**（13 级 × 4 leader × 4 贡品结果；stage2 脚本 `context_schedule()` 已定义，激活即可） | 【必须】 |
| 4 | `rollout_batch_size` / `collect_batch_size` | 4 / 4 global（2/rank） | **16~32 / 16~32 global** | 【必须】 |
| 5 | `max_sampler_bytes_per_rank` | 4 GB | **6~8 GB**，或把缓冲 horizon 4096→1024 | 【必须】 |
| 6 | `base_reg_coef` | 0.1（衰减至 0.05） | **0.01**（KL-to-uniform）或等价熵加成 | 【必须】 |
| 7 | lr 衰减 schedule | `stable_iterations=500`、`agent_lr_decay_power=1.0`，从 iter 500 起线性衰减 | **段内不衰减**（`stable ≥ target`）；warm start lr=1e-4 | 【必须】 |
| 8 | 单段训练量 | 2000 迭代 × 4 局 = 8000 局 | **≥8 万局/段**（迭代数 × 单迭代局数） | 【必须】 |
| 9 | `ntp_weight` | None（M09 完全未用） | **0.02（A/B 开）** | 【建议】 |
| 10 | Belief | B0 REJECT / off | **暂不重开**；NTP 之后重跑 B0 探针再决定 | 【建议】 |
| 11 | `model.architecture` | v1.3 | pilot 用 v1.3 跑通 → 再 A/B `v1.4-macro-fuse` | 【实验】 |
| 12 | macro-sequence PPO | 代码就绪未用 | 等 #1 修完并验证后再 A/B | 【实验】 |
| 13 | pilot 准出门槛 | 无 | **rule WR > 85% 且 cluster CI low > 0.7**（≥256 配对局）才进长训 | 【必须】 |
| 14 | 测试/脚本同步 | `test_two_stage_training.py` 断言 0.01/0.05 | 随 #1/#6 同步更新 | 【必须】 |

**明确不建议改**（避免过度治疗）：GAE λ=0.95、γ=1.0、Huber value loss、`value_loss_coef=0.5`（标准且实现正确）；奖励 ±3/2/1 除 3（与 FableDan 一致）；评估框架（cluster CI / one-shot holdout / champion gate，这是项目优势）；warm start from iter2000 保留（该策略近均匀，warm start 与从头差异极小，但保留可复现链条）。

---

## 1.【必须】clip_coef：0.02（实际衰减到 0.005）→ 固定 0.2

**现状（代码事实）**
- `guandan_marvel/config.py`：`base_clip_coef: float = 0.02`。
- `guandan_marvel/training/schedules.py` + `ScheduleConfig.clip_coef_decay_power=1.0` + `lr_stable_iterations=500`：有效 clip = `0.02 × min(500/iter, 1)`，即 iter500=0.02、iter1000=0.01、iter2000=**0.005**（`docs/marvel_v14_plan_verification.md` 自己确认了该衰减）。
- stage2 预备配置更紧：`base_clip_coef=0.01`，同衰减 → iter2000 时 **0.0025**；且 `tests/unit/test_two_stage_training.py` 把 0.01/0.05 断言锁死了。

**为什么必须改（本项目证据）**
- 训练指标全程 `clip_frac ≈ 12~14%`：即使 clip=0.02，也有约 1/7 的样本比率被截断——说明优势信号推着策略动，而 clip 在强行刹车。
- `kl_ref`（新旧策略 KL）全程 ≈ 0.0005：每次迭代策略只挪动 0.05% 的量级，**PPO 退化成了"几乎不学习的迭代"**。
- 学习曲线平坦：iter500→2000（数据 ×4），rule 胜率 56.25% → 56.25% → 50.00% → 65.63%（holdout 无偏值 55.38%），无显著改善。
- 本报告 GPU 实测：策略在真正分歧点 top-choice 仅 ~71%、熵 0.79 nats（均匀上界 ~1.03）——策略"温平"。

**理论依据**
- Schulman et al. 2017《Proximal Policy Optimization Algorithms》（arXiv:1707.06347）：PPO 的 clip 范围 ε 在其全部实验中为 **0.1~0.2**（Atari 用 0.1，连续控制用 0.2）。clip 的理论角色是把单次更新限制在信任域内的"悲观界"估计器；ε 越小，每次更新的最大策略变化越小。ε=0.005 意味着对数比率 ±0.5% 即截断，等效于把学习率再压一个数量级——这与"用 4 局/迭代的小 batch 做 8 次 minibatch 复用"叠加，构成极慢的学习动态。
- token 级 PPO 的现代大规模实践（RLHF，InstructGPT，Ouyang et al. 2022，arXiv:2203.02155；以及后续所有 RLHF 工作）同样用 ε≈0.2 的序列级/ token 级 clip。**没有任何已发表工作使用 0.005~0.02 量级的 clip**——这是本管线从 doudizhu 血统继承的未标定参数。
- clip 随迭代衰减无发表先例；PPO 的"退火"对象是学习率，不是信任域半径。

**风险与缓解**：clip 放宽后策略更新变大，可能出现不稳。缓解：(a) 监控 `kl_ref`/`clip_frac`/normalized entropy，不稳时先降 lr 而非缩回 clip；(b) 健康检查（nonfinite/illegal/truncated）已存在会硬失败；(c) 因 iter2000 策略近均匀，放宽后的前几步更新正是需要的"解冻"，崩塌风险低于常规 sharp policy。

---

## 2.【必须】训练分布：激活 208 上下文全分布（脚本已写好，从未运行）

**现状（代码事实）**
- M09 主训练：`configs/marvel_m09_main_training_plan.json` 明确 `level_rank: Rank.TWO`、`previous_result: null`，并自认 `known_limit: "main training does not directly sample all 13 level ranks or tribute contexts … may prevent E2/E3 acceptance"`。leader 亦固定 seat 0（`GameConfig` 默认）。
- 修复代码**已经存在**：`scripts/build_m09_two_stage_stage2_delivery.py::context_schedule()` 定义了 `13 个级牌 × 4 个 leader × 4 个 previous_result`（None / DOUBLE_DOWN 进贡 / HEAD_THIRD 进贡 / HEAD_LAST 抗贡取消）= **208 个等权上下文**；`config.py` 的 `training_context_schedule` 验证管线也已就绪。但 stage1/stage2 都从未运行，当前任何训练用的还是旧分布。

**为什么必须改（本项目证据）**
- 本报告 GPU 实测：跨级牌 vs rule 胜率 lv15=56.2% / lv3=62.5% / **lv8=37.5%** / lv14=75.0%——同一个模型在不同级牌上从"输给规则机器人"到"稳定赢"剧烈摆动。单级牌训练没有教会模型"级牌变化 → wildcard 语义变化 → 出牌策略变化"。
- 无贡品训练的模型从未做过进贡/还贡/抗贡决策；Botzone 真实对局每轮都有这些决策，等于把其中一手牌交给随机。
- holdout 的 rule CI [25.12%, 85.65%] 与分布外泛化不稳一致。

**理论依据**
- 训练/评估分布匹配是统计学习的基本前提（i.i.d. 假设）；掼蛋是一个 **contextual MDP**：级牌决定哪些牌是 wildcard（逢人配）及比较序，贡品状态决定初始牌权与换牌策略。在单一 context 上训练的策略对其余 context 是分布外泛化——泛化"能工作"不等于"可靠工作"，实测不可靠。
- 同族工作的实践一致：FableDan 训练时 `level=rng.randrange(13)` + `random_tribute_mode`（1/3 无贡、1/3 单贡、1/3 双贡）；DanZero（arXiv:2210.17087）论文明确将"贡品阶段从启发式改为纯 RL"列为后续改进，说明贡品覆盖被视为强度的一部分。

---

## 3.【必须】训练量：4 局/迭代 → 16~32 局/迭代；单段 ≥8 万局

**现状（代码事实）**
- M09 profile：`rollout_batch_size=4`、`collect_batch_size=4`（global，2/rank）、`target_iterations=2000` → 全程 8000 局 / 2,348,788 env steps / 16,000 次优化器步。stage2 预备配置同样是 4/4。

**内存可行性（本方案实测计算，不是估计）**
- dense 轨迹缓冲 ≈ **91 MB/轨迹**（horizon 4096 预分配；其中 `avail_tokens` 4096×256×8×8B ≈ 67 MB 为主）。当前 2×rollout+collect = 6 轨迹/rank ≈ 548 MB（与训练指标 `memory/sampler_bytes_per_rank_estimate=548,290,608` 精确吻合）。
- 4 GB/rank 预算内：rollout 8/rank + collect 8/rank = 24 轨迹 ≈ 2.2 GB ✅。若 `max_sampler_bytes_per_rank` 提到 6~8 GB（Kaggle CPU RAM 充足，这是 CPU 内存不是显存）：16/rank ≈ 4.4 GB ✅。
- 另一条路：实际对局 token 步数 p99 ≈ 330，把缓冲 horizon 从 4096 收到 1024 可把每轨迹降到 ~23 MB，同样预算下容量 ×4。
- 附带收益：当前 0.7 局/s/rank 的收集速度受 Python 侧 per-step 开销主导；batch 从 2 提到 8~16/rank 后 GPU 前向 batch 从 8 提到 32~64，GPU 利用率与吞吐都会改善。

**为什么必须改（本项目证据）**
- 每迭代仅 ~1200 env step、8 次 minibatch 复用——等效"新鲜梯度比特"极少，这是曲线平坦的供给侧原因之一（与 #1 的需求侧叠加）。

**理论依据**
- 自我对弈 RL 的棋力近似随经验量对数增长；DanZero 用 160 CPU + 1 GPU × 30 天（约 600–800 GPU 小时）达到对人类 60%、对最强规则机器人 80%+；M09 全部训练史约 **10.6 GPU 小时**（4 个 Kaggle 段 × 80 分钟 × 双 T4），是 DanZero 的 1/60~1/75；局数只有 FableDan 的 ~1/145。**两个数量级的差距无法用调参弥补，只能靠规模+效率同时补。**
- PPO 的样本复杂度：batch 越小、复用越多，单样本信息利用率越低（on-policy 假设被复用破坏）；OpenAI baselines / Acme 的标准做法是大 batch + 少 epoch。

**时间成本诚实账**：16 局/迭代 × ~40 s/迭代 ≈ 1400 局/小时；8 万局 ≈ 57 机时（Kaggle 12 小时/段 ≈ 5 段）。这是场景 B→C 的必要投入，不是可选项。

---

## 4.【必须】reg_coef：0.1（衰减至 0.05）→ 0.01

**现状（代码事实）**：`base_reg_coef=0.1`，`reg_coef_decay_power=0.5` → iter2000 时 0.05。损失项 = `pg_loss + reg_coef × KL(π‖U_legal) + 0.5 × value_loss`（`local.py`）。

**为什么必须改（本项目证据）**
- 全程 `kl_unif ≈ 0.02~0.03 nats`：策略被钉在"合法动作近均匀"处。0.1 的系数意味着每个样本都有一个大小 0.1 的系统性"拉向均匀"梯度，而优势经 minibatch 归一化后 ~N(0,1)、其系统性分量（优势与特征的相关）在早期仅 ~0.1~0.2 量级——**熵拉力与优势信号同量级，平衡点自然被推到近均匀**。

**理论依据**
- 熵正则的标准权重比 1 小两个量级：A3C（Mnih et al. 2016，arXiv:1602.01783）用熵系数 0.01；Sutton & Barto《RL: An Introduction》(2018) §13.1 的熵加成实践同量级；Williams & Peng (1991) 的熵正则框架中系数需按奖励尺度标定。
- 本项目奖励归一化到 ±1/3、优势归一化到单位方差——按 A3C 的标定逻辑，熵/正则系数 0.01 才是同尺度；0.1 是奖励尺度未变化时从 doudizhu 继承的旧值。

**风险与缓解**：降正则后若策略过早尖锐（normalized entropy < 0.1）导致探索不足，回调到 0.03 或加熵下限；监控 `normalized_entropy_mean`（指标已存在）。

---

## 5.【必须】lr 衰减：从 iter 500 开始线性衰减 → 段内不衰减

**现状（代码事实）**：`lr_stable_iterations=500`、`agent_lr_decay_power=1.0`：lr 从 iter 500 起线性下降，iter2000 时有效 lr = 4e-4 × 0.25 = 1e-4；且 `clip_coef_decay_power=1.0` 让 clip 也参与衰减（见 #1）。

**理论依据**
- lr 衰减是标准做法，但标准形式是**末端退火**（cosine/linear-to-zero，Loshchilov & Hutter 2017 SGDR；AlphaZero 的 lr schedule 同理）——从训练中点开始线性衰减等于主动放弃后半程的学习能力。
- 对续训（warm start）：lr 应从 warm-start 值（1e-4）起步并在段内保持，段间按计划调整；把 warm start 与"段内衰减"叠加会使模型在新区间内迅速冻结。
- clip 不参与衰减（见 #1 的理论部分）。

---

## 6.【建议】NTP 辅助损失：ntp_weight = 0.02（A/B）

**现状（代码事实）**：M09 训练 `ntp_weight=None`——辅助表征学习完全未用；v1.4 的 `ntp_head` 与 `compute_next_observation_token_loss` 代码已就绪（训练期存在、推理期丢弃、`belief_weight=0` 时不构建的同款纪律已实现）。

**为什么建议（本项目证据）**
- Belief B0 REJECT 的直接含义：当前共享表征**不含**对手手牌的增量推断信息（improvement = −0.202，CI [−0.228, −0.182]）。不完全信息博弈的决策上限被此锁死。
- NTP（预测下一步哪些观测 token 变为可见）是同一能力的"免费"代理任务：要预测别人出什么，就必须记牌与推断——这正是 B0 缺的。

**理论依据**
- 多任务/辅助表征学习：Caruana 1997《Multitask Learning》——辅助任务迫使主干学到对主任务有用的表征。
- 同族实践：DanLM 路线（FableDan 的直接基线）默认 NTP 权重 0.02；FableDan README 明确"Everything that matters—card counting, who is dangerous—is learned from scratch through self-play and an auxiliary next-token-prediction objective"。
- 与 Suphx oracle guiding（arXiv:2003.13590）/ PerfectDou 蒸馏（arXiv:2203.16406）的关系：那些方法用 oracle 监督把信息注入表征；NTP 是不需要 oracle 标签的弱化版，风险更低、实现已就绪。

---

## 7.【建议】Belief：暂不重开，先修表征再探针

**理论依据**：B0 REJECT 说明问题在**表征**不在**头**。Suphx/PerfectDou 的前提是表征能承载推断信息；在坏表征上加强权 head 只会过拟合噪声。正确顺序：(a) 用 #6 的 NTP 训练后的模型重跑 B0 探针（项目已有完整探针框架，按 deal 切分、四类基线）；(b) 仍 REJECT 则做表征诊断（哪些通道被使用、信念信息的线性可探性）；(c) 才决定 belief 头或表征改造。**顺序颠倒会浪费一轮训练。**

---

## 8.【实验】架构：pilot 用 v1.3，再 A/B v1.4-macro-fuse

**为什么不是"必须"**：v1.3 的对角双线性打分器（`logit(s,a)=w·(s⊙a)`，无法表达 s_i·a_j 交叉项）确实是表达力限制，FableDan 的 `concat+MLP` 无此限制；v1.4 ActorFuse（`[s,a,s⊙a,|s−a|] → ResGatedMLP` 残差，零初始化输出投影保证迁移恒等）修补了它。但一次只变一个变量：先把 #1~#8 的配置修完并验证 rule WR，再 A/B 架构，否则无法归因。

**A/B 设计**（用项目已有框架）：同 warm start、同种子、同发牌、paired deals；判定用 promotion gate（cluster CI low）；监控 `grad_norm/logit` 组与 `grad_norm/decoder` 组（`parameter_group_gradient_norms` 已实现）。

---

## 9.【实验】macro-sequence PPO：排在 #1 之后

`macro_ppo.py`/`macro_gae.py` 的半 MDP 数学正确（Sutton, Precup & Singh 1999，arXiv:1903.03627 的 options 框架），但 token 级 clip 病（#1）会被 macro 级 clip 继承——宏比率 `exp(Σ log π)` 的方差比 token 级更大，clip 0.01 下会更紧地锁死。顺序：先修 token 级配置并验证 → 再 `macro_decision_v1` A/B。

---

## 10.【必须】测试与交付脚本同步（机械但必要）

- `tests/unit/test_two_stage_training.py` 断言 `base_clip_coef == 0.01` / `base_reg_coef == 0.05`：随 #1/#4 同步更新。这不是"放水改测试"——测试锁的是配置常量值，配置变更为既定决策时测试必须跟随（项目规矩禁止的是为掩盖失败而改测试）。
- `scripts/build_m09_two_stage_stage2_delivery.py` 的 `training` dict 更新为新值（含 `training_context_schedule` 激活、batch、预算）。
- `configs/marvel_m08_locked_config.json` 是冻结历史，**不动**；新配置走新 profile/新 plan 文件。
- `guandan_marvel/kaggle/profiles.py` 的 fallback 默认值同步（它 fallback 到 `PPOConfig` 默认）。

---

## 11.【必须】pilot 准出门槛（长训的前置条件）

- 新配置 pilot：约 1 万局（Kaggle 1~2 段），跑项目自己的 formal evaluation：**rule WR > 85% 且 cluster CI low > 0.7**（≥256 配对局）才允许进长训。
- 理论依据：这是项目已有 promotion policy（ci95_low ≥ 0.52）的前置严格化；M09 的教训是 8000 局之后才发现配置问题——把发现点从 8000 局提前到 1 万局内的 pilot，成本 ~1 GPU 天。
- pilot 同时是概率推演（报告附录 C）的预测器：rule WR > 90% → 支持场景 C（35~50%）；~60~70% → 问题比配置深，下调预期并先做表征诊断。

---

## 12. 落地顺序与授权边界

```text
第 0 步（本次授权范围）：按本提案修改配置常量 + 同步测试 + stage2 脚本     [本地文件改动]
第 1 步：本地 CPU smoke（规则/环境/统计测试全绿 + 小规模训练冒烟）          [本地]
第 2 步：Kaggle pilot（新配置，~1 万局，formal eval）→ 对照 #11 准出        [需授权]
第 3 步：达标 → 冻结配置 → 分段长训（≥8 万局/段）→ 每段 formal eval       [需授权]
第 4 步：NTP A/B → v1.4-macro-fuse A/B → macro PPO A/B（顺序执行）        [需授权]
第 5 步：belief 重探针 → value qualification → 部署搜索                   [需授权]
```

任何一步的 health 失败（illegal/truncated/nonfinite > 0）立即停止，按项目失败即停规矩等待用户确认。

---

## 附录：四项修改的精确落点（2026-10-21 核实，pilot 后更新）

**修改量结论：小。** 全部落点在 2 个文件的 1 个字典里，共 ~17 行，无架构改动、无新代码、无测试破坏（唯一断言 clip 值的 `test_two_stage_training.py` 走 stage2 脚本路径，不经此字典）。

| 修改 | 精确位置 | 改法 |
|---|---|---|
| clip 0.02→0.2 固定 | `guandan_marvel/kaggle/phase4_delivery.py::_training_block()` 第 182 行 | `"base_clip_coef": 0.2`；同时 `"lr_stable_iterations"` 500→≥target_iterations（`profiles.py:103` 已支持透传；stable≥target 即 clip/lr 全程不衰减） |
| reg 0.1→0.01 | 同字典第 183 行 | `"base_reg_coef": 0.01`；`"reg_stable_iterations"` 500→≥target |
| lr 4e-4→1e-4 | 同字典第 181 行 | `"base_lr": 1.0e-4` |
| NTP 0.02 | 同字典第 190 行 | `"ntp_weight": 0.02`。**零集成风险**：`local.py::load_warm_start_policy` 的 `allowed_missing_prefixes = ("ntp_head.", ...)` 明文允许源 checkpoint 缺 NTP 头（全新初始化），docstring 即以此为例 |
| 贡品 208 上下文 | 同文件第 203 行 | `build_rank_leader_schedule()` → `build_context_schedule(previous_results=[None, DOUBLE_DOWN, HEAD_THIRD, HEAD_LAST(抗贡)])`（~10 行 payload 定义）。**7B 审计已 PASS**（`context_schedule.py::TRIBUTE_OBSERVATION_AUDIT["passed"]=True`，原 pre-repair 反例已修复），schema v2 原生支持；`CURRICULUM`（warmup 20 + blend 100）会把贡品上下文在 100 迭代内从 0 平滑混到全量，old_context=None 合法 |
| （配套）评估牌副含贡品 | `guandan_marvel/kaggle/pilot_delivery.py:207` | `previous_results=(None,)` → 4 模式（否则训练 7B、评估 7A，测错对象） |

**效果预期（分三层，详见对话结论）：**
1. 指标层（几十迭代内立竿见影）：clip_frac 31-37%→5-10%；macro_kl 0.04→0.1-0.3；token KL-to-uniform 下降加速；NTP loss 出现；贡品率随 curriculum 爬升。**这些是“修改生效”的领先指标——若不变说明没生效。**
2. 自博弈层（几千局）：value head 在新分布下 value_loss 先 spike 再降、value_ev（现 0.2-0.4）回升；贡品引入期指标暂时恶化是 curriculum 预期行为。
3. 棋力层（不会立竿见影）：pilot 已证 1000 局+权重动 3.96% → 棋力零变化；self-play RL 棋力对数增长，修配置是把“被锁死的学习速率”释放（M09 8000 局≈有效 800 局），不是抬上限。现实预期：8 万局段结束 rule WR 70-85% 才算有效果，85%+ 需多段 program。

---

## 下一步决策论证（2026-10-21，combo 结果之后）

### 问题：直接训练 8 万局吗？

**结论：不是。推荐「全配方 + 按 Kaggle 段分批 + 每 5000 局评估判定门」——第一段（~12h）即风险控制段，过门才续。** 相对盲跑 80k，只多 ~15% 评估开销，但把“可能浪费 24 GPU·小时”变成“最多浪费 8 GPU·小时”，且拿到学习曲线形状这一最高价值信息。

### 论证 1：曲线形状是唯一未知量，而中途评估便宜（期望值计算）

- 已知：5000 局 → vs-parent h2h +5.9pp（p≈0.037，per-deal 配对 CI [2.014, 2.454] 排除平局）；macro_kl 0.109→0.226 仍在上升（250 迭代时未收敛）。
- 未知：80k 局后 rule WR 到哪？vs-parent 曲线是继续爬还是平台化？value_ev 卡在 ~0.3 会不会成为天花板？
- M09 的历史教训正是“跑完 8000 局才知道曲线是平的”——盲跑 80k 是在重复这个错误，且赌注大 10 倍。
- 评估成本：1024 局/次（vs-parent + vs-rule 各 512）≈ 40-60 分钟/次。每 5000 局评估一次，80k 全程约 16 次 × 1h ≈ 评估开销 ~17%。
- 判定门设计（Kaggle 12h 段边界 = 天然止损点）：
  - **门 1（第一段末，~2 万局）**：health 全零（硬性）；vs-parent h2h > 55%（外推预期 58-62%；若 <52% 曲线已平 → 停，诊断）；贡品 curriculum 混合完成后 value_loss 回落 < 0.12。
  - **门 2（~5 万局）**：vs-parent h2h 斜率仍显著为正；normalized_entropy > 0.05。
  - **门 3（8 万局末）**：rule WR 对照 85% 门槛（CI low > 0.7）判定“有效果”。
- 两个指标的分工（FableDan DESIGN.md 的同款智慧）：**vs-parent h2h = “还在变强”的过程指标**（rule 饱和后唯一可信的增长信号）；**vs-rule = 准出门槛指标**。combo 已展示两者会背离（vs-parent +5.9pp 但 vs-rule 平）——中途必须同时看两个，不能只看 rule。

### 论证 2：贡品 + NTP 是数据量买不来的结构性能力，现在不加成本一样

- 贡品：每局 1-2 次还贡选择是模型从未训练的决策（当前随机）；Botzone 实战每轮都有。80k 局 7A 数据对这个缺口零贡献——晚加不如早加（ curriculum 已在 100 迭代内平滑混合，第一段就会覆盖）。
- NTP：B0 REJECT 表明表征缺对手手牌推断能力；NTP 是零 oracle 成本的修补，且是 DanLM/FableDan 的默认配置（同一架构家族验证过）。长跑中表征缺陷会被中后期决策放大。
- 成本：~17 行代码（落点已核实）+ ~10-20% 单步计算 + 归因弱化（接受：首要目标是棋力不是归因，combo 本来就不是 matched A/B）。
- 修正上轮判断（数据）：entropy 并非线性下降撞线——decile 分析显示它在 **0.087 找到稳定地板**（0.187→0.087 后连续 100 迭代持平）。所以“熵坍缩迫使盲跑危险”这个理由不成立，仍需监控但非紧迫。真正支持中途评估的是论证 1（曲线未知 + macro_kl 高位大动）。

### 论证 3：什么情况下才应该“直接 80k”

仅当 GPU·小时是免费且充裕的（盲跑的机会成本≈0）。在 Kaggle 配额稀缺下，分批+判定门严格占优。若用户判定配额充裕，可直接 80k，但保留每 5000 局评估（成本不变，只是放弃“提前止损”期权）。

### 期望收益的定量框架（诚实外推）

- h2h-vs-parent（对数衰减假设）：5k→+5.9pp；10k→+8；20k→+10；40k→+12；80k→+14pp 上下。
- rule WR：63% → 75-85% 区间为成功；注意 vs-parent 涨 ≠ rule 涨（combo 已证），rule WR 可能在 65-70% 徘徊很久后才跳——不要因此中途误判。
- 对 FableDan：即便 80k 全成功（rule WR 85%），距 FableDan 的 97.66% 仍有量级差距，h2h 胜率预计仍 <20%。**80k 的目标是场景 B→C 边界的“站稳中上游 + 验证曲线仍在爬”，不是追上 FableDan**（那需要场景 C：10 万局+ 与 belief 补课）。
