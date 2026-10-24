# Bootcamp Week-1 训练流程说明书（供第三方评审）

- 文档版本：v3（2026-10-17，F-round 修复版：永久 record-only 闸门 / NTP 移除 / 数字修正 / 分片读数 / paired 主终点严格完整性）
- 对应交付：`guandan-marvel-m09-bootcamp-week1-v1`
- 交付 manifest hash：`e0106f861b6156ee6166f2e2e66e94d8c76be3622297e443e78deb98adbee6f0`
- source archive sha256：`691d2633d84f95800412a162a340c903539202c520b7eecb2c409f13f92f3f1f`
- 计划 hash：`3786ecf232662a399409aad936c6c3685665589b7db52b4343aa00ab5444f83c`

> 本文档的目的：完整描述一次"两周追赶计划"中第 1 周的训练流程（设计、理由、超参、编排、评估协议、已知限制），供外部评审者判断其合理性。文末（§14）列出了希望评审者逐条回答的问题。
>
> 术语约定：**M09** = 本项目上一代模型（IPPO，iter2000 候选）；**FD/FableDan** = 外部参照模型（Botzone 掼蛋天梯第 11 名，1190.75 分，公开权重可本地推理）；**anchor 评估** = 运行中周期性小样本对规则/随机基线的评估。

---

## 1. 背景与目标

### 1.1 上一代模型（M09）的实测结论（全部为同引擎、正式评估产出）

| 指标 | M09 iter2000 | FableDan |
|---|---|---|
| vs rule（规则启发式） | 55.4% | 97.7% |
| vs random | 85.8% | 99.8% |
| h2h 对 FableDan（1024 局） | **2.34%**（结果工件未归档，待 comparison 交付重跑后回填；同口径 `_wilson(24,1024)` 实算 CI95 = [1.58%, 3.46%]） | — |
| 参数量 | 2.06M | 4.35M |
| 训练总自对弈局数 | 8,000 局（iter0→2000，4 局/迭代） | 未知（规模远大于此） |

学习曲线证据（官方评估记录）：

| 对局锚点 | iter500（n=128） | iter2000（n=260） | 变化 |
|---|---|---|---|
| vs rule | 56.3% | 55.4% | **-0.9pp（平台化）** |
| vs random | 83.6% | 85.8% | +2.2pp |
| vs 旧 champion（iter100） | 57.8% | 66.9% | +9.1pp（自对弈漂移） |
| 晋级（promotion） | inconclusive | accepted=false | 两次都未晋级 |

### 1.2 诊断（被本周流程直接针对的三条）

1. **采样批量玩具化**：每迭代仅 4 局（每 rank 2 条轨迹），PPO 优势估计被单局结果噪声主导（PPO 梯度方差 ∝ 1/batch）。
2. **学习信号弱**：纯自对弈 + 从未成功晋级（对手永远是冻结的 iter100 旧自己），对固定锚点（rule/random）在 iter500 后完全停止增长。
3. **缺少样本效率机制**：无辅助目标；对比 FD 使用 DMC + NTP 辅助 + belief 头（oracle 监督）。本 round 用 19,039 步实测证明本协议的 NTP 是零信息量任务（§7.2），故不实施；该项留待 Week-2 用蒸馏/其他辅助目标处理。

### 1.3 本阶段（Week-1）目标与判定标准

- **目标**：修复训练配方（批量 + 新数据流），验证学习曲线是否恢复上升；不追求一步追平 FD。
- **Gate B（周终判定）**：
  - 主终点：paired **new vs M09 iter2000**（同 256 副牌、4 座位轮换、双方温度 0）的 cluster-bootstrap CI 下界 > 0；
  - 次要：h2h vs FD（1024 局）三个读数并列——全集 / level=15 子集 / OOD 分解（§11.1）；
  - 辅助：vs rule ≥ 75%、vs random ≥ 97%；
  - 若主终点 CI 下界 ≤ 0：不再继续同类训练，直接进入 Week-2 的 FD 蒸馏（KD）路线。
- **预期区间（设计时估计，非承诺）**：vs rule 65-80%，vs random 93-98%，h2h vs FD 5-15%。

---

## 2. 总体流程图

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 交付物（离线构建，已通过本地端到端验证）                                      │
│  session1_start-upload.zip   (full 模式 + warm start)                      │
│  session2_continue-upload.zip(resume 模式)                                 │
│  session3_continue-upload.zip(resume 模式)                                 │
│  guandan_marvel_bootcamp_week1.ipynb（通用 Notebook，自动识别模式）          │
└────────────────────────────────────────────────────────────────────────────┘
                │ 上传 Kaggle Dataset（T4×2）＋导入 Notebook
                ▼
┌─── Session 1（预计 8-10.5h，墙钟上限 11h）────────────────────────────────┐
│ 1. bootstrap：解压 source → 编译 C++ 扩展 → 环境校验                        │
│ 2. warm start：加载 M09 iter2000 权重（仅策略）                             │
│ 3. 训练 500 iter × 20 局 = 10,000 局                                       │
│    每 50 iter：durable checkpoint + recovery zip（原子）                    │
│    每 250 iter：anchor 评估（64 局 rule/random）+ plateau 告警（record-only）│
│ 4. 到点停机 → 输出 guandan-marvel-recovery.zip                             │
└────────────────────────────────────────────────────────────────────────────┘
                │ 下载 recovery.zip → 作为 session2 的第二个输入
                ▼
┌─── Session 2（resume，到 1000 iter / 累计 20,000 局）─────────────────────┐
│ exact-resume：哈希校验 env/sampler/RNG/计数/优化器 → 无缝续训               │
│ 同样每 50 iter 落 checkpoint；每 250 iter anchor 评估                       │
└────────────────────────────────────────────────────────────────────────────┘
                ▼
┌─── Session 3（resume，到 1500 iter / 累计 30,000 局）─────────────────────┐
└────────────────────────────────────────────────────────────────────────────┘
                ▼
┌─── 周终评估（本地/新 Kaggle 会话）─────────────────────────────────────────┐
│ 主终点：paired new vs M09 iter2000（256 副 × 4 轮换 ×2 臂，2048 局）        │
│ 次要：1024 局 h2h vs FD 三读数（全集/level15/OOD）+ rule/random 对照        │
│ 判定 Gate B                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                ▼
        Gate B 通过 → Week-2：FD 蒸馏（KD）
        Gate B 不通过 → 直接进 KD，不再堆同类训练
```

---

## 3. 游戏环境与数据

### 3.1 规则与协议

- 掼蛋四人两队（座位 0/2 一队，1/3 一队），标准 V2 规则实现；进贡/还贡存在于完整协议中。
- 协议版本：`GD-ACTION-0.2-MARVEL | GD-ENV-0.2-MARVEL | GD-ENCODING-0.2-MARVEL`。
- 环境：C++（`nanopokerlib._guandan_cpp`）为训练/评估的生产后端；另有 Python 参考实现用于交叉验证（差分测试）。
- 构建身份（C++ 扩展 sha256）写入 checkpoint 与 recovery，任何后端变化会导致评估/续训被拒绝（fail-closed）。

### 3.2 训练分布（本周刻意与 M09 保持一致）

| 字段 | 值 |
|---|---|
| level_rank | `Rank.TWO`（只练 2 级开局） |
| leader_seat | 0 |
| previous_result | `None`（**无进贡上下文**） |
| 说明 | M09 主训练即为此固定分布；本周不改分布以保持与 warm start 的可比性。已知代价：13 个等级与进贡上下文的泛化不覆盖（评估侧通过选择/终审数据集覆盖，但训练未见） |

### 3.3 发牌生成与数据隔离

- 发牌由 EnvBatch 的**连续 PRNG 流**生成：初始种子 = `cfg.seed + rank × 1,000,003`，每个 slot 之后按序列化流推进；`(seed, slot)` 只标识首副牌，后续牌以流内 reset 记录。
- 本周训练种子：**570700**（M09 用 470700，刻意换新，避免与旧训练数据重复）；`environment_rank_seed_stride = 1,000,003`。
- 数据域隔离（不重叠的种子区段）：
  - 训练：570700 系列（连续流）；
  - 运行中 anchor 评估：1,200,000,000（8 副牌）；
  - 选择评估：1,000,000,000 系列；终审：1,500,000,000 系列。
- 每副训练牌的牌面哈希全部记录进 checkpoint 的 `training_data_provenance`（含 deal_sha256 列表与记录哈希），评估数据集独立哈希，二者必须无交集（已有隔离校验机制）。

### 3.4 一局游戏的时间结构（用于理解张量形状与吞吐）

- 平均 commit 步（决策点）≈ 118/局；token 步 ≈ 310/局；训练张量时间轴按协议最大手数 4096 稠密分配（这是显存/磁盘开销的主要来源，见 §9.4）。

---

## 4. 观测与动作表示

| 项 | 值 |
|---|---|
| token 词表 | 320（含事件、牌、动作结构 token；pad=0） |
| 观测 | 每个观察者的"最近事件 token 袋"（≤48 tokens，求和嵌入）+ 512 维通道特征 |
| 全部候选动作 | ≤256 个完整合法动作；每个候选 = ≤8 tokens 的序列 |
| 决策方式 | 在候选集合上做 softmax 打分选 1 个；提交时逐 token 回放（stepwise commit 协议） |
| 时间步 | 每次决策提交为一个时间步 t；展开缓冲 T=4096 |

---

## 5. 模型架构（MARVEL v1.3，2,061,314 参数；本周 ntp_weight=0 不构建 NTP 头）

```
观测 token 袋(≤48) ──sum-embed(128)+座位嵌入──► 时序 Transformer 解码器
                                                 (4 层 × 4 头, d=128, KV cache)
每个玩家一条时间轴 ─────────────────────────────────────┐
                                                        ▼
行为方状态 = state MLP( decoded(128) ⊕ channels(512) ) → feature 256 (3 层 ResGatedMLP)
                                                        │
候选动作编码 = action MLP( sum-embed(token 序列, 256) ) → feature 256 (2 层)
                                                        │
       策略 logits = logit_head( player_state ⊙ action_feat ) → 每候选标量 → masked softmax
       value      = value_head(player_state) → 标量 V
```

- 架构字符串固定 `v1.3`；NTP 头由配置 `ntp_weight > 0` 决定是否构建。本周 `ntp_weight=0.0`，因此模型为 2,061,314 参数（`state_tokens_embed` 40,960 + `state_player_embed` 640 + `action_tokens_embed` 81,920 + `decoder` 722,560 + `state_mlp` 754,944 + `action_mlp` 459,776 + `logit_head` 257 + `value_head` 257；与 warm-start checkpoint 的 `agent_state_dict` 逐键求和一致）。
- 模型前向在 rollout 阶段使用 KV cache（每步单 token 时间步前向）；训练阶段无 cache 全序列前向。
- 无对手建模头、无 belief 头（对比 FD 有 belief 头；本项目推迟到 Week-2+ 评估是否引入）。

---

## 6. 训练算法：单策略 IPPO（纯自对弈）

### 6.1 一个策略占 4 座

- 采样器（`runtime/sampler.py` + C++ EnvBatch）只接受一个策略模块，**4 个座位全部由当前策略驱动**（纯自对弈）。
- 训练引擎**不支持**在 rollout 中混入快照/规则等外部对手（引擎层限制）；champion/对手池仅存在于评估侧。
- `agent_merged`（EMA 副本，β=0 → 每步等于当前策略副本）只用于评估冻结，不参与 rollout。

### 6.2 损失函数（每个 minibatch）

```
L_total = L_PPO + reg_coef · KL(π_new ‖ uniform_on_legal)
        + 0.5 · Huber(V − V_target, δ=10)
```

- **L_PPO**：标准 clipped surrogate，clip=0.02（随时间衰减）；优势在批次内做 z-score 标准化；行为概率来自 rollout 策略（严格 on-policy，温度=1.0，采样器契约强制）。
- **KL-to-uniform 正则**：对合法动作上的均匀分布做正向 KL，用于维持探索；系数 0.1，按 `(500/t)^0.5` 衰减。
- **value**：Huber 回归到 GAE 目标，系数 0.5。
- **GAE**：γ=1.0（分幕任务），λ=0.95；终局奖励为团队 ±1（赢队每名玩家 +1，负队 -1；无分差 multiplier）。
- **NTP 辅助损失**：本阶段已移除（`ntp_weight=0.0`，见 §7.2）；损失函数与单测代码保留，默认不参与训练。

### 6.3 优化器与调度

| 项 | 值 |
|---|---|
| 优化器 | AdamW（β=0.9/0.999，weight decay 1e-2；头/嵌入部分标记 no-weight-decay） |
| 学习率 | 4e-4；前 500 iter 恒定，之后 ×(500/t)（power=1） |
| clip 系数 | 0.02，同样 ×(500/t) |
| 正则系数 | 0.1 × (500/t)^0.5 |
| 梯度裁剪 | 全局 L2 范数 1.0 |
| 每次迭代 | 4 epochs × minibatch（每 rank 10 条轨迹按 mini_batches=8 → 5 箱×每箱 2 条 → 20 次梯度步/迭代） |
| 每局等价梯度更新 | 20 步 × 2 条 / 20 局 ≈ 2 次/局（与 M09 相当） |

### 6.4 采样器与"精确局数预算"契约

- `rollout_batch_size = 4`（全局；每 rank 2 条并行环境），`collect_batch_size = 20`（全局；每 rank 10 局/迭代）。`collect()` 会一直 rollout 直到攒满 20 局才返回本迭代批次。
- 每次迭代恰好完成 20 局（commit 缓冲吞掉余量），因此有**硬契约**：`target_completed_games = target_iterations × 20`，每 session 校验累计值；墙钟停机只能作为安全阀（允许 mid-budget 停止，但记录为 incomplete，续训后继续补齐同一绝对目标）。
- 每步张量、RNG、env 内部状态、deal 记录都可以序列化 → 支持**精确续训**（exact resume），不存在"从最近批量重来"的近似。

---

## 7. 本周相对 M09 的配方改动（两处生效 + 一处移除）

### 7.1 采样批量 4 → 20 局/迭代（+批次内 mini_batches 2→8）

- 依据：PPO 优势/梯度估计方差 ∝ 1/batch；4 局/迭代时每个梯度步只见过 ~1-2 局，信用分配基本被单局噪声主导（M09 指标 `adv_std≈0.2`、`adv_mean≈0`，且对 rule 曲线在 iter500 后完全平坦）。
- 约束推导（为什么不是 64+）：
  - checkpoint 必须保存稠密采样缓冲（`T=4096` 全量分配）：**87.1 MiB/轨迹**；
  - 每 rank 轨迹数 = `2×rollout/rank + collect/rank`，20 局配置下 = 14 条 → 采样缓冲 1.19 GiB/rank → checkpoint 约 2.55 GB（双 rank）；
  - Kaggle working 盘 ~19.5GB；keep_last=2 + 峰值（新 checkpoint 写入 + recovery zip 重建）≈ 9.5GB 峰值，余量 >2×。
  - 若用 collect=44：单 checkpoint ≈ 5.1GB，峰值会逼近磁盘上限 → 否决。

### 7.2 NTP 辅助损失：实测零信息量，本 round 移除（ntp_weight=0.0）

- 原设计：对每个"行为方决策步 t"，预测 t+1 时刻该玩家观测袋中"新增"的 token（delta 袋），BCE-with-logits，pos_weight=10。
- 实测反证（真实 C++ 引擎，60 局随机自对弈 19,039 个 NTP 步）：
  - 每步正例 2.107 个 / 被评 313.4 个词表项，pos_rate = **0.672%**；
  - 逐事件同一性检查：行为方 t+1 的袋 = 其 t 步动作事件本身，**19,039/19,039（100%）**；即 delta 全部是"预测自己刚做的动作"；
  - 正例归类：55.4% framing（BOS/EVENT_*/phase/seat/PASS/COMMIT）、26.0% 本人刚采样的动作 token、20.1% 已原样存于 `SELF_PREFIX_TOKENS` 通道、15.5% 本人手牌 card token（已在 hand 通道）、0.3% wild 赋值；扣除这些后剩余有效增量信息 = **0.00%**。
- 成本量化（warm start 权重 + 随机 ntp_head，真实 on-policy 数据实测）：初始 `ntp_loss`=0.735（ln2×[(1−ρ)+10ρ]，ρ=0.0067），加权项 0.03×0.735=**0.0221**；但对共享主干的隔离梯度仅 state_mlp≈1.0e-4、decoder≈7e-8，比 PPO/value 项小 1,000–10,000 倍（初始不污染主损失，也说明该头替换成本极低）。
- 结论：本周不训练 NTP 头（`ntp_weight=0.0`，模型回到 2,061,314 参数，warm start 零缺失键）；损失函数与单测代码保留，Week-2 若换靶（如"下一自身决策步"以上含他座事件）再评估启用。

### 7.3 新训练种子（570700）+ warm start（而非 exact resume）

- 换种子的原因：新的"训练域"数据流避免与 M09 已消费的 8,000 局重复（数据隔离原则）。
- 用 warm start 而不是 exact resume 的原因：本阶段修改了 `rollout_batch_size/collect_batch_size/mini_batches`（属于训练语义字段），exact-resume 契约**禁止**改动语义字段（否则 sampler/replay 状态不可比）；warm start 的语义是"只继承策略权重，重置一切训练状态"。

---

## 8. 初始化：从 M09 iter2000 warm start

- 来源文件：`warm_start_policy.pt`（M09 unified 终选候选 `ippo_iter0002000` 的评估 checkpoint，sha256 `af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8`）。
- 加载语义（代码级事实）：
  - 校验 schema/算法/架构/协议；
  - 权重加载采用"允许缺失白名单"：**只允许缺 `ntp_head.*`（新头，随机初始化）**；其余任何缺失/多余键都报错；
  - optimizer、sampler（环境流/缓冲/计数）、RNG、schedule、局数计数**全部重置**；
  - 新 run 的 iteration 从 0 起，`training_data_provenance` 从新种子重新记录。
- 已知取舍：优化器动量/二阶矩被丢弃（AdamW 热启动瞬态），换来与旧 buffer/RNG 的解耦；这是工程上常见的 warm-restart 折衷。

---

## 9. Kaggle 会话编排（T4×2）

### 9.1 会话与目标

| Session | 模式 | 目标 iteration（绝对） | 目标局数（累计） | 预估耗时 |
|---|---|---|---|---|
| session1_start | full + warm start | 500 | 10,000 | 8-10.5h |
| session2_continue | resume | 1,000 | 20,000 | 8-10.5h |
| session3_continue | resume | 1,500 | 30,000 | 8-10.5h |

- 吞吐依据（M09 r2 主训练归档 `runs/M09_full_training/remote/r2_main_segment0500_20261013T135603Z/extracted-output/metrics.jsonl`，iter101-500，n=400 行）：
  - 端到端：该段 1600 局 / 4795.4s = **0.334 局/秒**（`session_elapsed_seconds` 见 `docs/marvel_migration/evidence/M09_main_segment0500/segment_summary.json`）；
  - rollout 口径：`perf/completed_games_per_second` mean **0.842** / median **0.844**（该指标 = 完成局数 / rollout 秒，`training/metrics.py:77`）；
  - learner 占迭代时间 56%（rollout 均值 5.0s vs learner 均值 6.4s，400 行均值）。
- 规划区间：0.3-0.4 局/秒，每迭代（20 局）约 50-67 秒。依据：新旧配置每游戏 learner 成本同比例（均为 4 epoch；10 局/20 梯度步与 2 局/8 步的关系），批量增大后 rollout 时间线性上升、端到端吞吐略降。
- 每 session 10,000 局 = 0.334 局/秒下 8.3h 训练 + 约 1-2h 评估/打包 ≈ **8-10.5h**；每周配额 30h，三会话 + 启动/评估/打包 ≈ 27-30h。

### 9.2 时间上限的三层防护

1. Worker 层 `max_wall_seconds = 39,600s（11h）`：到点在下一次迭代后干净停机（保存 checkpoint + recovery，`progress.json` phase 记录 `budget_stopped`，退出码 2）；
2. Notebook 层 runner 超时 = 11h + 900s = 11.25h；
3. Kaggle 平台 12h 硬上限。
→ 正常路径永远由第 1 层触发，输出完整可下载。

### 9.3 checkpoint / recovery / 断点恢复

- 每 50 iter（约 30-50 分钟）：
  - 写 durable DDP checkpoint（每 rank 一个 shard + manifest，原子发布；rank 间哈希交叉校验）；
  - 原子重建 `guandan-marvel-recovery.zip`（含 checkpoint + 全部运行工件 + 原生扩展二进制），旧 zip 被覆盖，无累积；
  - 更新 `checkpoint_manifest.json` 与 retention（keep_last=2 + latest）。
- 恢复路径：
  - 新 session 的 resume profile 目标为绝对 iteration；恢复时校验 source/semantic/config/sampler/RNG/counters/optimizer 全部哈希一致，然后**精确续跑**；
  - 若 session 中途被杀：用最近一次 recovery（运行中可随时下载更新过的 zip）作为输入重跑该 session，实现"最多损失 50 iter"；
  - plateau 历史（`anchor_history.json`）随 recovery 跨 session 恢复。
- 失败哲学：所有关键写入原子化（临时文件 + fsync + replace）；评估失败/恢复异常 → fail-fast 停机，但**最后 durable checkpoint 与 recovery 已保存**，人工可续。

### 9.4 磁盘预算（审计结论）

| 项 | 大小 |
|---|---|
| 单个 checkpoint（双 rank） | ≈2.55 GB |
| 保留策略 | keep_last=2 + latest（稳定 ~5.1GB；写入瞬间 3 份 ≈7.7GB） |
| recovery zip | 1 份，每次覆盖（gzip ≈0.3-0.6GB） |
| output.zip | 明确排除 checkpoints/ 与 zip 自身（几十 MB） |
| 最坏瞬时峰值 | ≈9.5 GB / 19.5 GB working 盘 |

### 9.5 日志与可观测性

- Notebook 控制台：每 5 秒一条 `[M07 heartbeat]`（phase、iteration x/y、累计局数、env/s、ETA、最新 checkpoint、评估状态）；启动阶段每 5 秒 bootstrap 事件。
- 文件层：`metrics.jsonl`（每迭代：pg/value/kl/clip/adv/ntp/grad、健康计数、吞吐）、`telemetry.jsonl`（5s：GPU/CPU/内存）、`progress.json`（心跳快照）、`console.log`、`diagnostic.jsonl`（rank 粒度事件）、`evaluation_games.jsonl` + `evaluation_summary.json` + `anchor_history.json`。
- 无长静默窗口：最长"低信息段"为 64 局 anchor 评估（5-10 分钟，心跳继续）与 checkpoint 打包（1-3 分钟）。

---

## 10. 运行中评估与 plateau 闸门

### 10.1 anchor 评估（每 250 iter）

- 构成：8 副牌 × 4 座位轮换 × 2 对手（rule、random）= **64 局**，独立子进程、单卡、温度 0（argmax）。
- 数据集：种子 1.2e9，固定 8 副（跨评估复用同一批牌 → 配对比较，降低方差）；与训练/选择/终审数据域不重叠。
- 输出：逐局记录 + 汇总 win rate + Wilson 区间；写入 `anchor_history.json`（每档 iteration、rule/random 胜率、局数）。
- 统计局限（已知）：每个对手 32 局，有效独立样本 ≈ 8 副牌；按仓库 `clustered_mean_ci95`（`evaluation/stats.py:13`）在 8 副时 3·ln(60)/8 = 1.535 的有限样本修正项已超过 0.5，区间退化为 **[0, 1]**（实测：M09 的 8 副牌 anchor 无论数据如何都得到 [0,1]）。anchor 只作趋势/冒烟信号，**不作强度声明**；naive Wilson 口径也不构成强度依据（n=32 时半宽约 ±16pp）。

### 10.2 plateau 闸门（本 round 改为 record-only 告警，永不停机）

- 旧规则（已废弃）：metric = rule win rate；`min_evaluations=3`、`patience=2`、`min_delta=1pp`，连续 2 次不超历史最好值 +1pp 就在 checkpoint 边界停机。
- 废弃依据（实测量化）：anchor 评估是温度 0 + 固定 8 副牌，因此对固定 checkpoint 是确定性的；旧规则在真实恒胜率（55%/60%/65%）下 Monte Carlo（10,000 次）前 6 次评估内停机概率 **93.75%/93.91%/94.04%**（聚类口径 0.9400/0.9394/0.9421）；确定性口径下平坦策略或 +0.3pp/eval 的慢改进策略 **100% 在第 3 次评估（iter750）触发**；只有每 250 iter 稳定 ≥+1pp 才能存活。与"宁可多跑一轮也不错杀"相反。
- 新语义（永久 record-only）：闸门仍每次评估更新 `anchor_history.json`（`stale_count`、`plateau_alert`、entries），但 payload 与评估事件均无停机语义（无 `stop_authority`/`plateau_stop` 字段），worker 永不因它 break；`progress.json.phase` 只会是 `complete` 或 `budget_stopped`。告警只提示"8 副牌固定牌面的 rule 读数低于历史最好值 +1pp"，不构成统计结论（见 §10.1 的 CI 局限）。
- 设计含义：`min_delta=1pp` 在聚类口径下仅为每评估 SD（√0.1025 = 32pp）的 1/32；任何基于该锚点的绝对阈值都不可校准。人工决策请使用 §11.2 的 paired 主终点与 §11.1 的分片读数。

---

## 11. 周终评估协议（Gate B）

### 11.1 h2h vs FableDan（次要指标，三读数并列）

- 环境：同一个 C++ 引擎与协议；双方温度 0（argmax）。
- 适配约束：FD 每一步只能在其自身 `gen_moves` 与引擎候选动作的**交集**内选择（其自定义移动生成与 MARVEL 候选对齐；此前 3,584 局 0 次适配失败/0 次空交集）。
- 规模：256 副牌 × 4 座位轮换 = **1,024 局**。
- 统计口径（修正）：按牌副聚类的 95% 半宽为 **±9.3–10.5pp**（plug-in s²=0.1025 → ±10.5pp；二项聚类 s²=0.0625 → ±9.3pp），不是 naive Wilson 的 ±3.1pp。M09 基线 2.34% 与 5% 线相差 2.66pp，在该样本量下**不可分辨**；因此 h2h 不再作为主判定。
- 三读数并列（comparison 汇总的 `by_context`，`comparison/stats.py`）：① 全集 1024 局；② level=15 子集；③ OOD 分解（非 level15/leader0）。训练分布（level=15、leader=0）只覆盖 256 副中的 **5 副（1.95%）**，全集读数几乎完全测的是等级/leader 泛化，必须与 ②③ 一起解释。

### 11.2 paired 主终点：new vs M09 iter2000

- 臂：`bootcamp_final_vs_m09_iter2000`（candidate=本周 final，opponent=M09 iter2000）与镜像臂 `m09_iter2000_vs_bootcamp_final`，同 256 副牌、4 座位轮换、双方温度 0；共 2048 局。
- 统计：`paired_delta`（`comparison/stats.py:102`）按 deal 配对，输出 mean_delta 与 cluster-bootstrap 95% CI；CI 下界 > 0 即主终点通过。deal 效应与先后手在配对中被消掉，功效高于 h2h。

### 11.3 辅助对照

- 同场地 rule/random 各 128 副（与 M09 同口径，便于纵向对照 55.4% / 85.8%）。
- 合法性红线：0 非法局、0 截断局、四座位胜率偏差 <10pp（已内置统计）。

---

## 12. 与 Week-2 的衔接（简述，不属本流程验收范围）

- 若 Gate B 通过：用 FD 做教师做**候选动作级蒸馏**：
  - FD 权重（npz）已验证可无损载入其 PyTorch 网络并在 GPU 批量推理（与 numpy 实现数值偏差 ≤7e-7）；
  - 数据生成：FD 在 MARVEL 引擎中打牌，逐决策记录候选 Q 向量（与引擎候选通过 move_key 对齐）；
  - 损失：对候选分布做 `softmax(Q_FD/τ)`（τ≈0.5）的 CE/KL；权重 λ 从 1.0 随自对弈局数余弦退火到 0（约 20k 局内），RL 始终为主损失；
  - 依据：策略蒸馏（Rusu 等）+ DAgger 分布修正 + 防 teacher ceiling 的退火。
- 引擎层限制：训练 rollout 内仍无法混入外部对手；对手多样性只能靠蒸馏数据与后续引擎改造解决。

---

## 13. 全量超参数表（交付 profile 的事实来源）

```jsonc
{
  "algorithm": "ippo",
  "ddp": { "world_size": 2, "backend": "nccl" },
  "training": {
    "seed": 570700,
    "environment_rank_seed_stride": 1000003,
    "rollout_batch_size": 4,          // 全局；每 rank 2 并行环境
    "collect_batch_size": 20,         // 全局；每 rank 10 局/迭代 → 20 局/迭代
    "mini_batches": 8,
    "agent_epoch": 4,                 // PPO 轮数
    "replay_buffer_scale": 8,         // IPPO 不使用 replay，仅配置完整性
    "base_lr": 4e-4,
    "base_clip_coef": 0.02,
    "base_reg_coef": 0.1,             // KL-to-uniform
    "lr_stable_iterations": 500,
    "reg_stable_iterations": 500,
    "ntp_weight": 0.0,              // NTP 已移除（§7.2），头不构建
    "ntp_pos_weight": 10.0,
    "checkpoint_interval": 50,
    "evaluation_interval": 250,
    "checkpoint_keep_last": 2,
    "evaluation_keep_last": 4,
    "max_wall_seconds": 39600,
    "max_sampler_bytes_per_rank": 4294967296,
    "amp_enabled": false,
    "compile_enabled": false,
    "model": { "architecture": "v1.3", "history_dim": 128, "feature_dim": 256,
               "decoder_layers": 4, "decoder_heads": 4, "state_layers": 3,
               "action_layers": 2, "fuse_layers": 2 }
  },
  "evaluation": { "base_deals": 8, "seed_base": 1200000000,
                  "opponents": ["random", "rule"], "seat_rotations": [0,1,2,3],
                  "timeout_seconds": 1800 },
  "plateau_gate": { "metric": "rule_win_rate", "min_evaluations": 3,
                    "patience": 2, "min_delta": 0.01 },  // record-only：只写 plateau_alert
  "completed_game_budget": { /* games == collect_batch_size × target_iterations，精确校验 */ }
}
```

---

## 14. 已知限制与请评审者逐条回答的问题

### 14.1 已知限制（设计时就承认的）

1. **纯自对弈**：训练 rollout 无外部对手/快照池（引擎单策略限制）；对手多样性只能靠蒸馏数据与后续引擎改造。
2. **NTP 已移除**：实测仍是零信息量任务（60 局 19,039 步：pos_rate=0.672%；行为方 t+1 袋 100% 等于其自身动作事件；扣除 framing/自身动作/通道已剧透后增量信息 0.00%），本 round `ntp_weight=0.0`。损失函数与单测保留待 Week-2 换靶评估；当前无任何辅助目标。
3. **奖励**：团队 ±1 终局奖励，无分差/双下 shaping；γ=1.0。
4. **评估功效（修正后）**：运行中 anchor 为每对手 32 局（8 副固定牌、温度 0）；按仓库 `clustered_mean_ci95` 8 副时区间退化为 [0,1]，naive Wilson 半宽 ±16pp；终评 1024 局 h2h 的聚类半宽为 ±9.3–10.5pp。闸门已改为 record-only（§10.2），不再有统计误杀；所有强度判断依赖 §11.2 的 paired 主终点与 §11.1 的分片读数。
5. **warm start 代价**：优化器状态清零，Adam 矩估计存在瞬态；未做对照实验。
6. **配额耦合**：若吞吐 <0.25 局/秒，可能多花一个 session 才能达成 30k 局（规划依据见 §9.1：0.334 局/秒端到端）。
7. **分布覆盖（训练侧）**：训练只覆盖 level=Rank.TWO（=15）、leader=0、无进贡；更新引擎/协议支持多上下文（`training_context_schedule`），本周为保持与 M09 配方可比未启用。
8. **分布错配（评估侧）**：h2h 牌副 256 副覆盖 13 级 × 4 leader、无进贡；恰为 (level=15, leader=0) 的仅 **5/256 = 1.95%**，即 **98.05%** 的主指标牌副处于训练分布外。`configs/marvel_m08_locked_config.json` 的 `limitations` 自认 "training deal distribution still requires M09 review for level-rank/tribute coverage"。因此 §11.1 的 h2h 全集读数必须与 level=15 子集、OOD 分解并列解释。

### 14.2 希望评审回答的问题（按优先级）

1. 对"批量 4→20（5×）"这一改动幅度：结合 PPO 的方差-效率权衡，是否足够打破观测到的平台？是否应该直接上 64+（代价：checkpoint 磁盘与吞吐）？有没有更合理的中间量级或梯度累积方案？
2. NTP 已因零信息量移除（§7.2）；本协议下是否存在可用的替代辅助目标（例如下一自身决策步的他座事件预测、阴牌计数 belief 头、对手建模、对比学习）？请给出可量化的信息增益估计方法。
3. 在纯自对弈不可改的前提下，是否有比"warm start + 批量"更有效的补充手段（例如熵/探索调度、价值目标塑形、KL-to-old 正则、自我博弈课程）？
4. 评估协议：32 局/对手的确定性锚点 + record-only 告警 + 1024 局分片 h2h + paired new-vs-M09 主终点，统计上是否仍有缺陷（多重比较、固定牌复用、选择偏差）？若你们负责验收，会怎样改？
5. Gate B 主终点（paired new-vs-M09 CI 下界 > 0）与预期区间（h2h 5-15%）是否务实？如果你们判断这批训练注定失败（例如核心瓶颈是经验量 2-3 个数量级），请直接指出更根本的路线修正。

---

## 15. 附录：交付物与哈希

F-round 重建交付（`scripts/build_bootcamp_week1_delivery.py`，源为当前工作区快照，commit `fa4a7af3227dd8d76663bd9cf2b0e3a659d70373`）：

| 文件 | sha256 |
|---|---|
| delivery manifest（`runs/bootcamp_week1_delivery/delivery_manifest.json`） | `e0106f861b6156ee6166f2e2e66e94d8c76be3622297e443e78deb98adbee6f0` |
| 计划 hash（plan JSON 内 `plan_hash`） | `3786ecf232662a399409aad936c6c3685665589b7db52b4343aa00ab5444f83c` |
| source archive（打包自工作区快照，sha256） | `691d2633d84f95800412a162a340c903539202c520b7eecb2c409f13f92f3f1f` |
| session1_start-upload.zip | `51e4766e5467a7f0a47db760172b2494cce4370ea052b7554a9f83b3a999cdc9` |
| session2_continue-upload.zip | `82b2c91628b7dc590eac1c6c739b2c610c8a7d3b05486f9d3d3e95a76c53d488` |
| session3_continue-upload.zip | `ce069950d79b2a5f3b794a9aa386aeaba6795fea4a645a2c256780bf85e6bb3a` |
| warm start 权重（M09 iter2000，未改动） | `af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8` |

本地验证记录（F-round，Python 3.12 + CPU torch）：

- full（warm start）2 iter → recovery → resume 2 iter，全部退出码 0；预算精确 40→80 局（=20×iters）；
  `warm_start_loaded.source_iteration=2000` 且加载无缺失键；anchor 记录跨 session 恢复（1→2 entries），
  闸门永久 record-only（payload/事件均无 `plateau_stop`/`stop_authority` 字段）；
- 新交付为 FIXLOG 所记录的 F1–F7 修复版本；F8 仅留接口未启用（见交付目录 FIXLOG.md）；
- 最终工作区回归：全量 1567 passed / 2 skipped / 1 failed（失败为既有 `guance_phase2_declaration_handoff`
  native-build-identity 工件漂移，与 F-round 无关）；非 acceptance 完整回归 **1262 passed / 2 skipped**。
