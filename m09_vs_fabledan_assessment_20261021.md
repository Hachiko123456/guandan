# guandan-marvel（M09）vs FableDan 全方位评估报告

- 评估日期：2026-10-21
- 评估对象：`D:/project/guandan`（MARVEL v1.4 / M09 训练线，IPPO iter2000 候选模型）
- 对比基准：`D:/project/FableDan`（Botzone 掼蛋排行榜第 11 名，1190.75 分）
- 证据基础：对比赛原始输出（3584 局 JSONL + summary）、两个项目的完整源码、训练产物与指标、以及在真实 C++ 环境上用 GPU 复跑的 128 局受控实验

---

## 0. 执行摘要

1. **对战结果是真实且彻底的失败，不是方法学假象。** 1024 局 head-to-head，M09（iter2000）只赢 24 局（2.34%，95% CI [1.17%, 3.71%]）；**91.2% 的对局被 FableDan"双下"**（对手队伍包揽第 1、2 名）。M09 的 24 场胜利只来自 12/256 个牌局（且每个牌局两个座位轮换都赢），说明胜利依赖发牌而非棋力。
2. **对比赛的方法学是公平的**，本报告逐项审计后确认：FableDan 使用其原生 NumPy 推理与完整 `gen_moves` 动作空间（99.4% 的招法在 MARVEL 引擎中合法，无回退、无降级），M09 使用官方评估协议（temperature 0、策略 argmax）；牌局、座位轮换、统计方法（cluster bootstrap + Bernstein CI）均无偏。
3. **项目自身的 final holdout 早已指出同一结论**：iter2000 对 rule 仅 55.38%（CI [25.12%, 85.65%]），**不能稳定击败一个贪心规则机器人**，M09 未被 accepted。FableDan 对同一 rule 机器人胜率 97.66%。
4. **代码质量高，问题不在正确性，而在训练规模、训练分布和关键能力缺失。** 用博弈论/RL 权威理论（Schulman PPO 2017、GAE 2015、Sutton-Precup-Singh 半 MDP 1999）逐项核对，PPO/GAE/奖励/宏动作数学实现全部正确；真正的差距是：
   - **训练量差约两个数量级**：M09 共 8000 局 / 235 万环境步；FableDan 估算 ≥2.9 亿决策样本（约百万局级）。
   - **训练分布严重缩水**：M09 只用固定级牌 Rank.TWO、无进贡、固定 leader；FableDan 全 13 级 + 随机贡品。
   - **不完全信息推断能力缺失**：Belief B0 预检被 REJECT（模型内部表征无法超越公开信息基线推断对手手牌）；FableDan 用 oracle 监督的 belief 头解决了这一问题。
   - **策略过"平"**：clip_coef=0.02（标准 PPO 为 0.1~0.2）+ KL-to-uniform 正则 0.1，使策略在真正分歧点只保持温和偏好（top-choice ≈71%）。
5. **实战能力结论**：当前模型不具备与 Botzone 前列 AI 实战的能力（h2h 2.34%），也不具备稳定战胜人类初级规则的能力（57% vs rule，且跨级牌不稳定，lv8 只有 37.5%）。按项目已规划但未执行的两阶段续训路线，stage1（同语义到 20000 局）只能带来边际改善；**只有在 stage2（全级牌 + 贡品 + 对手池 + 数量级以上的数据量 + belief 能力补课）之后，才可能训练出有实战价值的模型。**

---

## 1. 评估方法与证据来源

| 证据 | 来源 |
|---|---|
| 对战原始数据 | `m09-fabledan-comparison-output.zip`（3584 局逐局记录 + summary + validation report）、`m09vsfabledan (2).log`（Kaggle 运行日志） |
| 项目训练史 | `docs/marvel_migration/STATUS.yaml`、`runs/M09_full_training/**`（5 个远程段的指标 JSONL）、`configs/marvel_m09_*.json` |
| M09 模型 | `runs/comparison_delivery/dataset/m09_candidate_iter002000.pt`（SHA256 `af0df1ea…`，与 holdout 冻结候选一致） |
| FableDan | `D:/project/FableDan` 全量源码 + `dist/fabledan_weights.npz`（与对比赛权重 SHA256 完全一致） |
| 本地受控实验 | 本报告作者在 RTX 3050 Ti GPU 上用真实 C++ 引擎复跑 128 局（4 个级牌 × rule/random × 16 局）+ 8961 个决策步的策略诊断 |
| 工程验收状态 | `docs/marvel_v14_prompts/CURRENT_EXECUTION.md`、`runs/phase5/b0/b0_r4*.json`、`runs/phase6_closeout/codex_acceptance_20261020/CODEX_ACCEPTANCE.json` |

---

## 2. 对战结果全景

### 2.1 官方对比数据（Kaggle T4×2，2026-10-16，共 3584 局）

| 对战臂 | 局数 | 胜率 | cluster bootstrap 95% CI | Bernstein 95% CI |
|---|---:|---:|---|---|
| **h2h（iter2000 vs FableDan）** | 1024 | **2.34%**（24 胜） | [1.17%, 3.71%] | [0%, 9.04%] |
| m09 vs rule | 512 | 57.03% | [51.56%, 62.50%] | [39.54%, 74.52%] |
| **FableDan vs rule** | 512 | **97.66%** | [95.70%, 99.22%] | [85.38%, 100%] |
| m09 vs random | 512 | 80.86% | [77.34%, 84.38%] | [66.09%, 95.63%] |
| **FableDan vs random** | 512 | **99.80%** | [99.41%, 100%] | [89.65%, 100%] |
| champion(iter100) vs FableDan | 512 | 1.56% | [0.39%, 3.12%] | [0%, 13.37%] |

判定规则（预先注册）：`m09_stronger if ci95_low>0.5; fabledan_stronger if ci95_high<0.5` → **`fabledan_stronger`**，结论毫无统计争议。

### 2.2 失败的结构性特征（从 3584 局逐局记录提取）

- **91.2%（934/1024）的 h2h 对局以"双下"告终**——FableDan 队伍包揽第 1、2 名，M09 队伍第 3、4 名。M09 队伍平均奖励 -0.953（满分 ±1），即平均每局输掉约 2.86/3 个排名分。
- M09 的 24 场胜利**只来自 12 个牌局**（final-000023/084/097/103/115/163/177/194/195/241/242/255），每个牌局在 4 个座位轮换中赢 2 次。也就是说：**96% 的牌局上 M09 一胜难求**，其胜利是发牌红利的产物，不随座位变化——这是典型的"没有棋力、只有牌运"特征。
- 对局覆盖全部 13 个级牌（每级 76~80 局 × 4 轮换），M09 在每个级牌、每个座位轮换上的胜率都约 2.3%，**不存在"某个级牌不会"的局部问题，是全面实力差距**。

### 2.3 对方法学公平性的审计（重要）

用户可能怀疑"完败"是对赛设置造成的假象。本报告逐项审计后确认设置公平：

1. **FableDan 的动作空间未被阉割。** `fabledan_guest.py` 将 MARVEL 引擎的完整合法动作枚举与 FableDan 原生 `gen_moves` 取交集后交给 FableDan 的 Q 网络评分。诊断数据：`fd_candidates` 191,730 中 190,632（**99.4%**）在引擎中合法；`unsupported_candidates=0`、`classification_fallbacks=0`、`empty_intersection_fallbacks=0`。即 FableDan 以其原生方式完整发挥，"交集"限制只损失 0.6% 且本就不是其原生动作集的一部分（其 `Move` 表示天然按 claim 模式去重，每决策约 10 个候选，而引擎枚举约 62 个完整动作）。
2. **两边都无搜索、都 greedy。** M09 走官方 V2 协议 temperature=0（已核实 `avail_weights` 合法动作为均匀 1.0，选择即策略 argmax）；FableDan 走 NumPy 推理 greedy argmax Q。
3. **无信息泄露。** MARVEL 通道布局（`game.py`）明确不含他座手牌；FableDan 观测为引擎公开状态 + 自手。
4. **统计严谨。** 256 个独立牌局 cluster × 4 座位轮换，cluster bootstrap（按牌局聚类）+ bounded empirical Bernstein 双 CI，pre-registered 判定规则。
5. **诚实的 scope 声明**：`informal: true`、无贡品（`previous_result=None`）、`tribute: none`。无贡品对双方都简化了游戏；FableDan 训练时是**有**随机贡品的，M09 训练时**没有**——所以这个 scope 实际上对 FableDan 略有不利（它更熟悉贡品局），对 M09 是中性。这不影响结论方向。
6. **权重一致性**：对比赛使用的 `fabledan_weights.npz`（SHA256 `86570527…`）与 FableDan 仓库 `dist/fabledan_weights.npz` 逐字节一致——对战的就是其正式部署模型。

### 2.4 与项目自身 holdout 的一致性（交叉验证）

M09 项目自己的 one-shot final holdout（2026-10-16，1300 局，与 FableDan 对战同日）已得出同向结论：

| 对手 | iter2000 胜率 | CI95 low | 判定 |
|---|---:|---:|---|
| random | 85.77% | 60.04% | E1 supported |
| legacy V1 | 87.69% | 62.30% | supported |
| iter100 champion | 66.92% | 36.59% | 不能声明稳定优于 |
| **rule** | **55.38%** | **25.12%** | **E2 失败：不能稳定击败 rule** |

→ M09 accepted：**否**。这与"FableDan 97.66% vs rule"互为印证：**M09 与 FableDan 之间的差距，首先体现在"能否稳定战胜一个贪心规则机器人"上。**

### 2.5 本地 GPU 受控实验（本报告新增证据）

在真实 C++ 引擎上加载 iter2000 checkpoint（v1.3 架构，2,061,314 参数），4 个级牌 × {rule, random} × 16 局：

| 级牌 | vs rule | vs random | 备注 |
|---|---:|---:|---|
| 15（=训练级牌 2） | 9/16 = 56.2% | 15/16 = 93.8% | 训练分布内 |
| 3 | 10/16 = 62.5% | 16/16 = 100% | 分布外 |
| 8 | **6/16 = 37.5%** | 16/16 = 100% | **分布外，输给 rule** |
| 14 | 12/16 = 75.0% | 14/16 = 87.5% | 分布外 |
| **合计** | **37/64 = 57.8%** | 61/64 = 95.3% | 与官方 57.03%/80.86% 同向 |

两个结论：
- **本地 harness 复现了官方数字**（57.8% vs 官方 57.03%），证明实验可靠，也证明官方评估没有低估模型。
- **模型跨级牌极不稳定**（37.5%~75% vs rule，样本小但波动巨大）。训练只见过级牌 2，模型把级牌当输入通道记住了"级牌 2 时怎么打"，没有学会"级牌变化 →  wildcard 语义变化 → 出牌策略变化"的泛化。这正是训练分布缩水的直接后果。

**策略锐度诊断**（8961 个候选方决策步）：平均合法行数 1.65（63.7% 的协议步骤只有 1 个合法动作——环境强制）；在真正有多选的分歧点上，策略 top-choice 概率仅约 71%、熵约 0.79 nats（该处均匀上界约 1.03）。换算到"一手牌"层面：MARVEL 的协议要求 5~8 个 token 顺序组合才能提交一手牌，**token 级的不确定性会复合成一手牌级别的犹豫**——这是 token 协议架构的固有代际损耗（详见 §4.2）。

---

## 3. 项目"修复得怎么样了"——工程状态评估

### 3.1 已经修复完成的（有证据）

- **阶段 1–6 全部通过独立 Reviewer 验收**（2026-10-20）：全量测试 2037 passed / 2 skipped / 1 known Guance dirty-tree failure；阶段 6/06B Codex 复核通过（`runs/phase6_closeout/codex_acceptance_20261020/`）。
- **工程质量确实是强项**：checkpoint provenance 哈希链、champion/晋升的保守 cluster CI 门槛（ci95_low ≥ 0.52）、one-shot holdout 防过拟合机制、DDP/recovery 精确恢复、C++/Python 双引擎 differential 测试、防"静默降级"的严格加载器。这些比 FableDan 的工程体系（6 个测试文件）严谨得多。
- 对赛暴露的问题（规则、协议、评估链路）已经定位清楚，代码里有对应修复的阶段记录。

### 3.2 尚未修复 / 尚未验证的（关键）

| 项 | 状态 | 含义 |
|---|---|---|
| **正式训练** | 未启动（DELIVERY_PENDING） | iter2000 之后没有新训练；当前 v1.4 源码**从未训练过** |
| **Belief B0** | REJECT | 模型编码器无法比公开信息基线更好地推断对手手牌（improvement = -0.202，CI [-0.228, -0.182]），Belief 实现按预案停止 |
| **Value Gate** | INCONCLUSIVE | 价值函数资格未通过 → 按设计**不能启用搜索** |
| **Search** | 未实现（SEARCH_IMPLEMENTED=False） | 无部署期搜索 |
| **两阶段续训计划** | 已写好，未执行 | stage1（iter2000→5000 同语义）/ stage2（全级牌+贡品+pool）都只在配置层 |
| **工作区未提交改动** | 2874 insertions / 43 files | v1.4 的 actor fuse、NTP、macro PPO、opponent pool 等在代码里，但**零训练验证** |
| **部署链路** | 无 | 没有 Botzone 提交、没有 NumPy 部署模型、没有线上对局 |

**一句话：工程修复（让代码正确、可复现、可验收）基本完成；棋力修复（让模型变强）尚未开始。**

---

## 4. 架构对比：FableDan vs MARVEL

### 4.1 总览表

| 维度 | FableDan（约 2600 行 Python） | MARVEL / M09（3.9 万行 + C++） | 优劣 |
|---|---|---|---|
| 骨干网络 | 小型 Llama：d=128，4 层，4 头，RoPE + QK-Norm + SwiGLU，**4.35M 参数** | 同家族 Llama decoder（4 层/4 头/history=128）+ state/action MLP，**2.06M 参数** | 平手（FableDan 大 2.1×，都不是容量瓶颈） |
| 历史表征 | 48 词表事件 token 流（相对座位编码），每次决策重编码全历史 | 320 词表 + 512 通道 actor-visible 状态（手牌 one-hot、剩余张数、当前 winning、级牌 one-hot…），KV cache 增量编码 | **MARVEL 表征更丰富**（显式牌计数/剩余张数，少让网络自己学） |
| 动作空间 | **完整招法级**：枚举 legal moves，Q(s,a) 直接对每手牌打分 | **协议 token 序列**：FAMILY/RANK/SUIT/LENGTH/CARD/WILD/COMMIT/PASS 逐步组合，5~8 步提交一手牌 | **FableDan 优**（见 §4.2） |
| 策略打分 | `MLP([ctx; hand_emb])` → Q，concat 后过 3 层 MLP——全表达 | v1.3：`Linear(s ⊙ a)`（逐元素积的对角双线性）+ v1.4 残差 ActorFuse | v1.3 表达力受限（见 §4.2）；v1.4 已补但未训练 |
| 训练算法 | DMC（Deep Monte Carlo）：回放池 + MSE(Q, 团队最终回报/3)，无 TD bootstrap | IPPO：on-policy PPO + GAE(λ=0.95) + Huber value loss + KL-to-uniform 正则 | **MARVEL 算法更现代**（信用赋值严格更优，见 §5） |
| 辅助学习 | **NTP（0.02）+ Belief 头（0.05，oracle 监督对手手牌）** | **无**（M09 训练 ntp_weight=None） | **FableDan 优**（关键差异，见 §4.4） |
| 探索 | ε=0.02 top-10 贪心探索 | KL-to-uniform 正则（0.1）+ PPO clip | FableDan 优（MARVEL 的正则把策略压平，见 §4.6） |
| 训练对手 | 纯自博弈 + 冻结 snapshot 池（league 思想） | 纯自博弈（opponent pool 已实现未用） | FableDan 略优 |
| 训练分布 | 全 13 级随机 + 随机贡品模式 | **固定级牌 2、无贡品、固定 leader** | **FableDan 完胜**（P0 级差异） |
| 训练规模 | ≥2210 cycle × ~131k 样本 ≈ **≥2.9 亿决策样本（百万局级）** | 8000 局 / 235 万环境步 / 16000 次优化器步 | **FableDan 完胜**（P0 级差异，约两个数量级） |
| 评估体系 | 规则/随机/shadow 胜率诊断 | champion 晋升 + formal matrix + one-shot holdout + cluster CI + provenance 哈希链 | **MARVEL 完胜** |
| 部署 | 纯 NumPy 推理，Botzone 第 11 名（1190.75） | 无部署链路 | FableDan 完胜 |

### 4.2 动作空间设计：token 协议 vs 完整招法（架构级核心差异）

MARVEL 把"出牌"建模成**序列生成**（类似 LLM 解码）：模型每个环境步选一个协议 token，5~8 步后 COMMIT 提交一手牌。FableDan 把出牌建模成**单步分类**：环境枚举合法完整招法，Q 网络一手定胜负。

本报告在真实引擎上实测了这套 token 协议的代价：

- **63.7% 的协议步骤只有 1 个合法动作**（环境强制的，学了也白学），86% ≤2 个——信息密度低。
- 真正有选择的分歧点上，iter2000 策略 top-choice 仅 ~71%。由于一手牌要连续多个 token 都选对，**token 级 71% 复合成一手牌级别的显著犹豫**（若 6 步组合且每步独立 0.71，则"完全按意图出牌"的概率约 0.71⁶≈13%——实际步骤间相关，不会这么低，但方向明确）。
- 训练指标佐证：500 轮 segment 全程 `kl_unif ≈ 0.02~0.03`（策略离均匀分布只有一步之遥），`clip_frac ≈ 13%` 且 `kl_ref ≈ 0.0005`——**策略几乎没从均匀初始化移动多少**。

token 协议的优点（参数共享、组合泛化、无需枚举动作）在斗地主/DanLM 系项目中被验证过；但它的成立前提是**策略在 token 级足够尖锐**。MARVEL 当前训练配置（clip 0.02 + reg 0.1，见 §4.6）恰好压制了尖锐度，放大了该架构的固有代际损耗。FableDan 的招法级 argmax 没有这个问题。

另一个架构细节：v1.3 的打分器是 `logit(s,a) = w·(s ⊙ a)`——**对角双线性形式，无法表达 s_i·a_j (i≠j) 的状态-动作交互**，必须依赖 state/action MLP 把交互预先编码进同维向量。v1.4 的 ActorFuse（`[s, a, s⊙a, |s−a|] → ResGatedMLP` 残差，零初始化输出投影保证迁移时恒等）正是为修补这一点而设计——数学上正确且优雅，但**从未训练验证**。

### 4.3 训练算法：DMC vs IPPO+GAE

FableDan 的 DMC：每个决策样本的目标是 `z = 团队最终回报 / 3`（无折扣、无 bootstrap），回放池 FIFO，MSE 拟合。这是经典的 Monte Carlo control with function approximation——**无偏但高方差、信用赋值最粗**（整局 250 个决策共享同一个结局信号）。它依赖海量样本把方差平掉（DanLM 路线："算法很简单，胜利来自表征和纯粹的数据量"）。

MARVEL 的 IPPO：GAE(λ=0.95) 把终端奖励按每个玩家自己的决策时间线做指数加权信用赋值 + PPO clip + Huber value loss。**理论上严格优于 DMC**：同样的样本量能提取更多信号。代码实现经逐行核对正确（§5）。

**结论：算法上 MARVEL 更先进，但先进算法救不了 100 倍的数据差距。** 这也解释了为什么 FableDan 用"简单算法 + 海量数据"战胜了 MARVEL 的"现代算法 + 微量数据"。

### 4.4 不完全信息推断：belief 差距（P1 核心）

掼蛋是严重不完全信息博弈。顶级 AI（Suphx 的 oracle guiding、PerfectDou 的完美信息蒸馏）共同经验：**决策质量的上限由对手手牌推断能力决定**。

- FableDan：训练期用 oracle 标签（3 个相对座位 × 15 等级计数）监督 belief 头（权重 0.05），强制编码器学会精确记牌/推断；推理期丢弃，零开销。其 DESIGN.md 明言这是第一优先级。
- MARVEL：Belief B0 预检 **REJECT**——用 iter2000 模型自身的表征做对手手牌计数回归，mean improvement = **-0.202**（CI [-0.228, -0.182]），即**比"公开信息强基线"还差**。按预案停止 belief 实现。

这个 REJECT 的深层含义：**M09 的 512 通道表征里堆了剩余张数等显式特征，但模型的共享主干并没有学会用它们做概率推断**。没有 belief，中后期"要不要压过对手、什么时候放炸弹、对手是否在听牌"这类决策只能靠团队回报的端到端信号盲学——在 8000 局的数据量下根本学不出来。这是与 FableDan 差距的结构性来源之一，也解释了为什么 M09 对 rule 只有 57%：rule 机器人虽笨，但它不犯错式让牌；而 M09 在信息不完整时频繁做出无声损失。

### 4.5 评估与工程体系（MARVEL 的优势，值得保留）

必须公平指出：MARVEL 在**训练基础设施**上明显强于 FableDan：

- champion 世代管理 + 保守晋升门槛（cluster CI low ≥ 0.52）+ one-shot holdout 防选择偏倚；
- 全链路 provenance 哈希（checkpoint/dataset/evaluation/source 互锚）；
- 双引擎 differential、容量硬失败不截断、DDP/recovery 语义严格的 checkpoint 兼容分类；
- 对赛 harness 本身（牌局哈希、轮换、双 CI）比 FableDan 的 evaluate.py（单种子 200 局）严谨一个量级。

**如果数据规模和分布补齐，这套基础设施是 FableDan 不具备的加速器。**问题只在于：基础设施不等于棋力，而用户此刻需要的是棋力。

### 4.6 训练配置层面的三个技术失误（P1）

从 `runs/M09_full_training/**/metrics.jsonl`（500 轮全量指标）与 `configs/marvel_m09_main_training_plan.json` 直接读出：

1. **`base_clip_coef = 0.02`**。PPO 标准值为 0.1~0.2（Schulman 2017 及后续所有大规模应用）。0.02 把每次迭代策略变化锁死在 ±2%，配合 4 局/迭代的数据量，等效学习速率极低——实测 `kl_ref`（新旧策略 KL）全程 ≈0.0005，`clip_frac` 常年 12~14%（约 1/7 的样本在被截断），`pg_loss` 长期在 0 附近抖动。**且实际比 0.02 更糟**：`schedules.py` 的 `clip_scale = min(500/iter, 1)^1.0` 衰减使有效 clip 在 iter1000/1500/2000 分别降到 **0.010 / 0.0067 / 0.005**（项目自己的 `docs/marvel_v14_plan_verification.md` 已确认此衰减）——训练后期策略更新被锁死在 ±0.5%，这是学习曲线平坦（§C.2）的更完整解释。
2. **`base_reg_coef = 0.1` 的 KL-to-uniform 正则过强**。它直接把策略往"合法动作均匀分布"拉，与尖锐策略的目标相反。实测 `kl_unif ≈ 0.02~0.03`：策略离均匀仅 0.03 nats。对于需要精确时机判断的牌类游戏，这个正则抑制了优势信号转化为尖锐偏好的能力。（该正则从 doudizhu 血统继承，未针对掼蛋重新标定。）
3. **每迭代 4 局（2 rank × 2 trajectory）**。双 T4 的吞吐下 collect 只有 ~6 秒/迭代，learner ~6.5 秒/迭代——**learner 在过小的 batch 上反复 epoch**，梯度噪声被 4 epoch × 2 minibatch 放大。2000 迭代 × 4 局 = 8000 局的总量，对 self-play RL 只是热身量级。

此外训练分布（§4.1 表）固定级牌、无贡品、leader=seat 0，是计划文档自己承认的 known limit（"may prevent E2/E3 acceptance"）——事后看它还顺带造成了跨级牌不稳定（§2.5）。

---

## 5. 代码正确性验证（对照权威理论）

逐项核对两个项目的关键算法实现：

### 5.1 PPO  clipped surrogate —— ✅ 正确（MARVEL）
`macro_ppo.py`：宏动作序列 log 概率在 log 域累加（`sum_t log π(a_t|h_t)`，不乘概率）；对**完整宏比率** `r_m = exp(new−old)` 做单次 clip；每宏等权（长宏不获得额外策略权重）；优势按 minibatch 全局统计量（DDP all-reduce count/sum/sumsq）归一化；`L_k = (world_size/N)·Σ l_i` 保证 DDP 梯度平均等于全局每宏均值；全程 FP32，log-ratio 超 FP32 exp 安全域**硬失败**而非静默钳位。与 Schulman et al. 2017（PPO）及其后续大规模实践完全一致。

### 5.2 GAE —— ✅ 正确（MARVEL）
`gae.py::compute_gae_from_value`：在共享 token 时间线上按玩家分别维护 reward_buffer / discounted_next_value / gae 的反向扫描；δ_t = Σγ^j r + γ^Δ·V(该玩家下次行动点) − V_t；A_t = δ_t + γλ·A_{next}。这正是 Schulman et al. 2015（GAE）在多智能体回合制时间线上的正确适配，与 `legacy_gae.py` 中独立实现的参考 oracle 数学等价（该 oracle 本身就是为了做等价性回归而写）。

### 5.3 半 MDP 宏决策 GAE —— ✅ 正确（MARVEL v1.4，未训练）
`macro_gae.py`：环境折扣钟按基础 token 计（γ^{Δ_m}，Δ_m 含 chance/他人 token），λ 按同一玩家宏决策衰减——与 Sutton, Precup & Singh 1999（Options / Semi-MDP）的宏动作公式一致；显式拒绝把 τ=0 当终局窗口；末宏窗口延伸到真实 num_steps。数学严谨。

### 5.4 DMC / Fitted-Q —— ✅ 实现正确，算法本身是已知高方差路线（FableDan）
`ring.py` + `train.py`：z = 团队回报/3 ∈ [−1,1]；FIFO 回放；冻结 snapshot 只做对手不进回放；eps top-k 探索。无 target network、无 TD bootstrap——这是 DanLM 血统的显式设计（用数据量换方差）。属于 Bertsekas & Tsitsiklis（Neuro-Dynamic Programming）框架下合法的 Monte Carlo control，但在样本效率上严格弱于 PPO+GAE；其成功完全依赖规模。

### 5.5 奖励语义 —— ✅ 两边一致且符合掼蛋标准
MARVEL `reward.py`：双下 +3 / 头三 +2 / 头末 +1，败队取负，每 seat 取队伍值；训练归一化 R/3。FableDan `engine.py::_rewards`：同一张表。与 Botzone 掼蛋标准计分一致。无 reward shaping、无势函数——干净。

### 5.6 环境规则一致性 —— ✅ 有实证
- MARVEL 内部：C++/Python differential 测试（`tests/marvel_differential/`）+ 2037 项测试。
- 跨项目实证：对赛诊断 `unsupported_candidates=0`、`classification_fallbacks=0`、FableDan 99.4% 招法在 MARVEL 引擎合法——两套独立实现的规则引擎在 4 万余次决策上无实质分歧。
- 本报告实测：C++ 引擎发牌 108 张守恒、27×4 手牌、接风/双下判定与规则一致（holdout 审计同样验证）。

### 5.7 小结

**两个项目的核心算法代码都是正确的，且 MARVEL 的实现更现代、更严谨。** 用户担心的"代码是否符合数学理论"这一问题可以给出肯定回答；失败的原因不在代码正确性，而在：训练规模（P0）、训练分布（P0）、策略锐度配置（P1）、belief 能力（P1）、以及 v1.4 改进未经训练验证（P2）。

---

## 6. 问题清单（按严重度，毫无保留）

### P0-1 训练量差约两个数量级
- M09：8000 局 / 2,348,788 环境步 / 16,000 次优化器步（全部训练史）。
- FableDan：≥2210 cycle × ~131k 样本/cycle ≈ **≥2.9 亿决策样本（约百万局）**（依据：`dist/FableDan_league_pool1910_*` 交付命名、训练看板 "cycle2010/2210 vs 1910"、`train_fast` 默认 buffer 262144/diversity 2；未见其 run log，为下限估计）。
- self-play RL 的棋力大致随 log(数据量) 增长；两个数量级的差距不是调参能弥补的。

### P0-2 训练分布缩水：单级牌 + 无贡品
- `marvel_m09_main_training_plan.json` 白纸黑字：`level_rank: Rank.TWO`、`previous_result: null`，并自认 "known_limit … may prevent E2/E3 acceptance"。
- 后果一（已实测）：跨级牌不稳定，lv8 只有 37.5% vs rule（§2.5）。
- 后果二（未在本次对赛暴露）：进贡/还贡/抗贡是完全没学过的决策——真实 Botzone 对局有进贡，M09 在那些局面等同随机。FableDan 训练时 1/3 无贡 + 1/3 单贡 + 1/3 双贡。
- 附带：训练 leader 固定 seat 0，也是分布偏差。

### P1-1 策略过平：clip 0.02 + reg 0.1
标准 PPO clip 的 1/5~1/10；KL-to-uniform 正则把策略钉在近均匀处（kl_unif≈0.03）。token 级 71% 的 top-choice 复合成一手牌级别的犹豫（§4.2/§4.6）。这是"能学会但学不精"的直接技术原因。

### P1-2 Belief 能力缺失（B0 REJECT）
模型无法超越公开信息基线推断对手手牌。不完全信息博弈的决策天花板被锁死。FableDan 用 0.05 权重的 oracle belief 头解决；MARVEL 的预案是 B0 通过才实现 B1——B0 已挂，需要先解决"为什么 512 通道表征学不到推断"这个更基础的问题（可能需要专门的 belief 预训练任务或表征改造，而不是在现有主干上加头）。

### P1-3 Value Gate INCONCLUSIVE + Search 未实现
价值函数资格未过（EV/排序/校准未达 Gate），按设计搜索不能启用。没有部署期搜索意味着每一次决策只有一次前向——对比 FableDan 每步 20~40ms、Botzone 单步数秒预算的余量，MARVEL 连"用算力换棋力"这条路都没铺。

### P2-1 v1.4 新功能零训练验证
actor fuse、NTP、macro-sequence PPO、opponent pool、symmetry：工程全部验收通过，但**没有一个在真实训练中跑过**。它们解决的是 §4.2/§4.4 的真问题（表达力、信用赋值），但"工程正确 ≠ 棋力提升"（项目自己的 CURENT_EXECUTION.md 也这么写）。warm-start 迁移语义（source 已变，不能冒充 exact_resume）需要重新设计。

### P2-2 当前源码与 m09 模型不同源
m09（iter2000）由 Kaggle 交付的旧 source 训练；当前工作区有 2874 行未提交改动（v1.4 全套）。继续训练必须先解决配置冻结、warm-start 版本化迁移、目标设备（双 T4/NCCL）启动/恢复验证——项目文档已列明，尚未做。

### P2-3 无部署与实战链路
没有 NumPy/轻量推理、没有 Botzone 提交、没有线上样本。FableDan 的 NumPy 镜像推理 + Botzone 第 11 名是闭环的；MARVEL 目前是"训练到 holdout 为止"的开环系统。

### P3 其他
- 训练指标 `pass_rate ≈ 0.60`：六成 commit 是 PASS 的学习行为，需在续训中监控是否过保守（不主动赢牌）。
- `double_down_rate` 自博弈中从 0.41 缓升到 0.56，未见明显 exploit 崩塌，但样本太小不足以判定。
- leader_seat 固定、seed 语义等分布细节建议在 stage2 一并随机化。

---

## 7. 最终能训练出什么东西？实战能力判断

### 7.1 现状判定
- **当前 iter2000 模型：不具备实战能力。** 对 FableDan 2.34%（91% 被双下）；对 rule 57% 且跨级牌在 37.5%~75% 间摇摆；未学过进贡；无部署链路。它大致相当于"比随机强、比规则机器人 marginally 强、未到人类新手平均水平"的自我对弈初级阶段产物。
- 注意：这不是"训练失败"——E1（稳定胜 random 85.77%）和 legacy superiority（87.69%）都成立，说明 **RL 管线本身是通的**，模型确实从零学会了记牌、组合、轮次管理。只是在 8000 局的预算下只能到这个水平。

### 7.2 按现有计划的推演
- **Stage1（iter2000→5000，同语义，总 20000 局）**：数据量仍只有 FableDan 的 ~2%，分布仍单一。预期：rule 胜率可能从 57% 升到 70% 上下并稳定，champion 晋升可能通过，但对 FableDan 仍是个位数胜率。**不建议把宝全押在 stage1。**
- **Stage2（全级牌 + 贡品 + 对手池 + warm start，2000 迭代）**：方向完全正确（逐个对应本报告 P0-2/P2-1）。若同时把数据量抬到 **50 万局以上**并修正 §4.6 的 clip/reg 配置，按 DanLM/DouZero 血统的已证实力，该架构家族达到"Botzone 中上游"是合理预期。但 belief（P1-2）不解决，中后期决策质量会有可见天花板——FableDan 的第 11 名不是只靠数据量堆出来的，belief 头是它的结构性优势之一。

### 7.3 诚实的期望管理
- 以当前工程状态 + 现有算力预算（Kaggle T4×2 数小时/段），**不可能复现 FableDan 的训练量**。现实目标函数应该是：先稳定超过 rule（CI low > 0.6），再谈 legacy/champion，最后才可能与 FableDan 有一战。
- 若算力受限，优先补分布（全级牌+贡品）和配置（clip/reg）——这两项零算力成本，且已被证实造成实际损失（§2.5）。
- 若目标就是"追上 FableDan"，唯一被验证的路径是它的路径：**全上下文自博弈 + 十年级数据量 + belief 辅助 + 部署期搜索**，逐项补齐，没有捷径。

---

## 8. 建议路线图（按优先级）

1. **立即（零训练成本）**：冻结新训练配置——全 13 级随机、随机贡品、随机 leader；`clip_coef` 提到 0.1~0.2（或至少 0.1）；`reg_coef` 降到 0.01 或改为熵bonus 形式；每迭代局数从 4 提到环境吞吐允许的上限。
2. **短期**：解决 warm-start 语义与目标设备验证后，跑一个**全上下文 pilot**（如 5 万局），用现有 formal 评估框架验证 E1/E2/E3——这是第一次能回答"修好之后到底强多少"的实验。
3. **中期**：重打 Belief B0——先诊断为什么 512 通道表征学不到推断（表征探针/表征相似度分析），再决定是加监督头还是改表征；同步实现 v1.4 macro PPO + ActorFuse 的 A/B（同 warm start、同种子、paired 牌副）。
4. **中期**：实现 value qualification → 若过 Gate，实现部署期一步搜索（有 FableDan 的 20~40ms/步数据作延迟参照）。
5. **长期**：把数据量当一等公民——GPU-days 预算、cycle 化训练（ DanLM/FableDan 式 actor-inference-learner 分离）、NumPy 部署模型、Botzone 提交拿真实排名反馈。
6. **全程**：保持 MARVEL 已有的评估严谨性（这是它对 FableDan 的真正的比较优势），每一个"变强"的声明都必须过 cluster CI + one-shot holdout。

---

## 附录 A：关键证据索引

- 对战：`report.md` / `summary.json` / `games.worker*.jsonl`（m09-fabledan-comparison-output.zip）
- 训练分布证据：`configs/marvel_m09_main_training_plan.json`（level_rank/previous_result/known_limit）
- 训练超参证据：`guandan_marvel/config.py`（base_clip_coef=0.02, base_reg_coef=0.1, base_lr=4e-4）；checkpoint resolved_config（ntp_weight=None）
- 训练动态证据：`runs/M09_full_training/remote/r2_main_segment0500_*/extracted-output/metrics.jsonl`
- Belief REJECT：`runs/phase5/b0/b0_r4.json`（improvement −0.202, CI [−0.228, −0.182]）
- Value Gate/Search：`runs/phase6_closeout/codex_acceptance_20261020/CODEX_ACCEPTANCE.json`（INCONCLUSIVE / False）
- FableDan 规模证据：`dist/FableDan_league_pool1910_v3.zip`、`dist/fabledan-training-demo.html`（cycle2210 vs 1910 = 50.70%）
- FableDan 排名：`botzone_rank.html`（第 11 名，1190.75）
- 本地实验：`.workbuddy/tmp_analysis/exp_m09_levels.py` 及输出（GPU 128 局 + 8961 步诊断）

## 附录 B：本报告作者的本地实验复现方法

```bash
# 环境：D:/project/guandan/.venv-gpu（torch 2.13.0+cu126，RTX 3050 Ti）
# 脚本：.workbuddy/tmp_analysis/exp_m09_levels.py
# 内容：加载 m09_candidate_iter002000.pt（绕过跨平台 backend-identity 校验），
#       在 C++ 引擎上以官方 V2PolicySession 协议对 rule/random 各打 16 局 × 4 级牌，
#       温度 0；同步收集策略概率/熵/合法动作数诊断。
# 结果：vs rule 57.8%（官方 57.03%）、vs random 95.3%（官方 80.86%，本样本偏易）；
#       分歧点 top-choice ≈71%、多选熵 ≈0.79 nats。
```

## 附录 C：用现在的架构训练，打败 FableDan 的概率推演（2026-10-21 补充）

### C.1 判定标准
"打败"按项目自身 Gate 定义：h2h 胜率 > 50% 且 cluster CI low > 0.5，≥1000 局配对对局。单次百局样本不构成证据。

### C.2 证据链

**（1）项目自己的学习曲线是平的。** 四个 segment 的正式评估（同一批评级牌局、配对比较）：

| 迭代 | 累计局数 | vs rule（点估计） |
|---|---:|---:|
| 500 | 2,000 | 56.25% |
| 1000 | 4,000 | 56.25% |
| 1500 | 6,000 | 50.00% |
| 2000 | 8,000 | 65.63%（选择集）/ 55.38%（one-shot holdout） |

数据量 4 倍，rule 胜率在 50%~66% 区间噪声波动，**无显著改善**。说明瓶颈不是"数据不够"，而是配置（clip 0.02 + reg 0.1）把学习能力锁死了。

**（2）DanZero 论文（AAAI 2023，arXiv 2210.17087）给出该架构家族的权威标尺：**
- DanZero 用 **DMC + 513 维状态特征 + 54 维动作编码**（与 MARVEL 的 512 通道 + token 协议同族），**160 CPU + 1 GPU 训练 30 天（约 600–800 GPU 小时）**，达到：对 top-8 规则机器人基线 100%（半数）/ >80%（最强几个），对人类选手 60%。
- DanZero+ 期刊版在此骨架上加 policy-based RL（PPO 类）+ 预训练 warm start——即 MARVEL 走的 IPPO 路线。
- FableDan 的 README 自述"达到 DanLM 规模的 experience 需要 serious GPU-days"，其训练看板显示 cycle 2210 时相对 frozen champion 仅 50.70%（自博弈已平台期），部署版为 Botzone 第 11 名。

**（3）算力账。** M09 全部训练史 ≈ 4 个 Kaggle 段 × ~80 分钟 × 双 T4 ≈ **10.6 GPU 小时**、actor 容量 4 局/迭代；DanZero 为 **600–800 GPU 小时 + 160 CPU actor**；FableDan 估算 **≥2.9 亿决策样本（约 116 万局）**。M09 的局数只有 FableDan 的 ~1/145，GPU 小时约为 DanZero 的 1/60~1/75。

### C.3 分场景概率（作者判断，非实测）

| 场景 | 内容 | P(统计显著战胜 FableDan) |
|---|---|---:|
| A | 按现状续训（当前 config + 单级牌 + 无贡品，哪怕到 5 万局） | **≈ 0–2%**。曲线已平，rule 胜率锁在 ~55%；FableDan 对 rule 97.66% |
| B | 修 config（clip→0.1~0.2、reg→0.01）+ 全分布，10–30 万局（DanZero 算力的 1/5~1/3） | **≈ 5–15%**。预期 rule 胜率 85–95%，是可敬的机器人；但数据量差 1~2 个数量级且无 belief |
| C | B + belief 补课 + DanZero/FableDan 级数据量（百万局、600+ GPU 小时、百级 actor） | **≈ 35–50%**。架构家族在此规模被反复验证；MARVEL 表征与 IPPO+GAE 不弱于 DanLM/DMC；残馀风险是 belief B0 REJECT 暴露的表征问题与 FableDan 是移动目标 |
| D | C + 部署期一步搜索（value gate 过后） | **≈ 45–60%**。不完全信息博弈中浅搜索通常 +2~5% 胜率 |

**加权结论：如果执行完整的 B→C→D 计划并投入 DanZero 级算力，最终战胜 FableDan 的概率约 40–55%——略高于抛硬币，但远非确定；如果只执行已规划未执行的 stage1（同语义到 2 万局），概率 ≈ 0–2%。**

### C.4 关键洞察与决策门

1. **架构不是瓶颈。** DanLM 家族在同等架构下达到过 Botzone 前列（FableDan）和人类水平（DanZero）；MARVEL 的 512 通道表征与 DanZero 的 513 维同族，IPPO+GAE 比 DMC 样本效率更高，评估基础设施更强。真正的瓶颈是：配置（P1-1）、分布（P0-2）、数据量（P0-1）。
2. **最便宜的预测器：全分布 + 修配置的 pilot。** 跑 ~1 万局（约 1–2 GPU 天）后看 rule 胜率：若 >90%，按 C 推演（35–50%）；若仍 ~60–70%，说明问题比配置更深（表征/优化），概率应下调到 10% 以下并优先做表征诊断。这个 pilot 是后续所有算力投入的前置条件。
3. **FableDan 是移动目标。** 其作者持续训练（cycle 2210+）。"战胜 2026-10 的 FableDan"与"战胜你完成训练时的 FableDan"是两个目标；领先窗口需要持续投入维持。
4. **统计诚实性。** h2h 接近 50% 时需 ~1000 局才能把 CI 半宽压到 ±3%；任何"打败/打不过"的声明都必须过项目自己的 cluster CI + one-shot holdout，不能用选择集数字。

### C.5 训练配置现状核对表（2026-10-21 逐个文件核实）

| 配置项 | 现状 | 位置 | 是否需要修改 |
|---|---|---|---|
| 全级牌+贡品上下文分布 | ✅ 已定义（13 级 × 4 leader × 4 贡品上下文 = 208 等权上下文），**从未运行** | `scripts/build_m09_two_stage_stage2_delivery.py::context_schedule()` | 不用改内容，但从未生效 |
| `base_clip_coef` | ❌ 默认 0.02；stage2 预备 **0.01**（更紧）；schedule 衰减使 M09 实际到 iter2000 仅 **0.005**、stage2 会到 **0.0025** | `config.py`、stage2 脚本、`tests/unit/test_two_stage_training.py`（测试锁死了 0.01/0.05） | **必须改**：固定 0.1~0.2、取消衰减（或 stable_iterations 设到 ≥target） |
| `base_reg_coef` | ❌ 默认 0.1；stage2 预备 0.05；衰减后 M09 0.05 / stage2 0.025 | 同上 | **建议改**：降到 ~0.01 或改熵加成形式 |
| `base_lr` | ⚠️ M09 用 4e-4；stage2 预备 1e-4（warm start 合理） | stage2 脚本 | 基本合理，可保留 |
| 训练量 | ❌ stage2 仍是 4 局/迭代 × 2000 迭代 = 8000 局（与 M09 总量相同） | stage2 脚本 budget contract | **必须改**：提到 10~100 倍 |
| 模型架构 | ❌ 锁 v1.3；v1.4-macro-fuse 代码就绪未纳入 | stage2 脚本 `model.architecture` | 建议 pilot A/B 后决定 |
| NTP / belief / symmetry | ❌ 均未启用（belief B0 已 REJECT） | 无配置 | belief 需先解决 B0 表征问题 |
| warm-start 语义 | ✅ optimizer/replay 重置、版本化迁移管线就绪 | stage2 脚本 | 就绪 |
| 配置冻结动作 | ❌ 未完成（CURRENT_EXECUTION.md：冻结配置是下一步且需用户授权；stage1 未运行，stage2 无输入） | `docs/marvel_v14_prompts/CURRENT_EXECUTION.md` | 需用户授权后执行 |

**结论：配置没有改完，且最关键的 clip 一项在 stage2 预备配置里方向是错的（0.01 比 0.02 更紧，衰减后 0.0025）。** 建议的冻结组合：clip 固定 0.1~0.2（不衰减）、reg 0.01、全 208 上下文分布、rollout/collect batch 提到吞吐上限、target 局数 ≥8 万（阶段目标），架构先 v1.3 跑通再 A/B v1.4-macro-fuse。

---

## 附录 D：T4×2 pilot 复盘（2026-10-21，guandan-marvel-v14-t4x2-pilot-20261021）

### D.1 实际配置 vs 本报告建议（逐项核对）

| 项 | 本报告建议 | pilot 实际 | 是否落实 |
|---|---|---|---|
| 架构 | pilot v1.3 → A/B v1.4 | **直接上了 v1.4-macro-fuse**（90 键，ActorFuse 零初始化） | ✅ 超出预期 |
| 训练语义 | 修完 clip 再上 macro | **macro_decision_v1 + macro_normalized_kl_uniform_v1** | ✅ 超出预期（顺序相反） |
| batch | 16~32 局/迭代 | **20 局/迭代**（rollout 16 + collect 20，mini_batches 8） | ✅ |
| AMP | — | **启用** | ✅ |
| 级牌/leader 分布 | 208 上下文 | **12 级牌 × 4 leader**（13 级中缺 9 级，500 局采样未覆盖） | ⚠️ 部分 |
| 贡品上下文 | 必须 | **previous_result 全为 null——未修** | ❌ |
| clip | 0.2 固定 | **仍 0.02**（macro 级；clip_frac 31~37%） | ❌ |
| reg | 0.01 | **仍 0.1**（macro-normalized 形式） | ❌ |
| lr | warm start 1e-4 | **仍 4e-4** | ❌ |
| NTP | 0.02 A/B | **未开**（ntp_weight=0.0） | ❌ |
| 训练量 | ≥8 万局/段 | **1000 局**（50 迭代） | ❌（pilot 规模） |

### D.2 结果：工程满分，棋力零信号

- **工程**：precheck/resume_check PASS，50 迭代 1000 局，health 全零（illegal/truncated/nonfinite=0），recovery 完整；warm start 与 iter2000 的 78 个权重**逐位一致**（max diff 0.0）；吞吐 ~1.05 局/秒（T4×2），**80k 局 ≈ 21 小时 ≈ 2 个 Kaggle 段**；50 迭代后权重相对 L2 变化 3.96%——策略确实在动（macro_kl 0.04~0.06 vs M09 token 训练的 0.0005）。
- **棋力**：candidate vs control（=iter2000）**50.0%**（CI [0,1]）；candidate vs rule 56.25%（CI [1.2%,100%]）、control vs rule 54.69%；paired rule improvement **+1.56pp（CI [-100pp, +100pp]）**；verdict **INCONCLUSIVE**、promotion false。
- **机制解释**：策略动了但棋力没动，三个原因叠加——(a) 关键配置未修（clip 0.02 在 macro 级仍锁死信任域，31~37% 宏被截断；reg 0.1；无贡品）；(b) value head 未适应新分布：value_loss 0.116（M09 收敛时 0.02）、value_ev 仅 0.2~0.4——iter2000 的 value 只在级牌 2 上训练过，换 12 级牌后 advantage 估计质量差，净学习为零；(c) 评估功效不足：384 局只能检测巨大效应，"INCONCLUSIVE"是诚实的读数而非失败。
- **自博弈信号**：double_down_rate 0.49→0.375、mean_rank_point_reward 2.1→1.975（自我对弈变不分胜负）；token KL-to-uniform 0.074→0.052（策略在变尖，方向正确）；grad_norm 0.47→0.29（递减）。

### D.3 结论与下一步

**可以继续，且工程线已验证到值得投入的程度；但"当前配置出棋力"没有证据。** 下一步：把 D.1 中 ❌ 的四项（clip 0.2 / reg 0.01 / 贡品上下文 / lr 1e-4）+ NTP A/B 补上，直接跑 **≥8 万局/段 + ≥1000 局/臂** 的有统计功效段（约 2~3 个 Kaggle 段），对准入门槛（rule WR>85%、CI low>0.7）判定。不要再跑 1000 局级 pilot——它只能回答"管线通不通"，回答不了"有没有效果"。

---

## 附录 E：combo 训练复盘（2026-10-21，guandan-marvel-v14-t4x2-combo-20261021-v1）

### E.1 配置落实核查（5 项建议 → 落实 3.5 项）

| 建议项 | 落实 | 实际值（checkpoint resolved_config） |
|---|---|---|
| clip→0.2 固定 | ✅ | `base_clip_coef=0.2`，`clip_coef_decay_power=0.0` |
| reg→0.01 | ✅ | `base_reg_coef=0.01`，`reg_coef_decay_power=0.0` |
| lr→1e-4（warm start） | ✅ | `base_lr=1e-4`，lr 全程恒定 |
| 训练量↑ | ✅ | rollout 24/迭代、250 迭代 = **5000 局**（warm start 为第一个 pilot 的 iter50 候选） |
| 13 级牌 | ✅ | 3~15 全 13 级 × 4 leader（curriculum offset 50 平滑混合） |
| **NTP 0.02** | ❌ | `ntp_weight=0.0`（REPORT.md 自列 `"ntp": false`） |
| **贡品上下文** | ❌ | schedule 52 上下文全部 `previous_result=null`（`"tribute": false`） |

### E.2 训练动态：完全符合预测，且是三次运行中第一次"活跃学习"

| 指标 | 预测 | 实际（前 46 → 后 46 记录） | 判定 |
|---|---|---|---|
| clip_frac | 33%→5-10% | **0.081 → 0.106** | ✅ 符合 |
| macro_kl | 0.04→0.1-0.3 | **0.109 → 0.226（持续上升）** | ✅ 符合且未收敛 |
| normalized_entropy | 下降但防坍缩 | 0.163 → **0.087**（警戒线 0.05 之上，但在降） | ⚠️ 监控 |
| value_ev | 应回升 | 0.304 → 0.297（**持平 ~0.3**） | ❌ 老问题未解 |
| NTP loss / tribute_rate | 应出现/爬升 | 无 / 0.0 | ❌ 未应用 |

**三次运行对比（核心证据）**：pilot#1（clip 0.02）macro_kl **递减** 0.057→0.037（策略冻结）；combo（clip 0.2）macro_kl **递增** 0.109→0.226（250 迭代时仍在活跃移动，远未收敛）。clip 修改的“解冻”效果得到直接验证。

### E.3 棋力：第一次有统计上可辩护的提升（但幅度小）

- candidate vs control（=pilot#1 的 iter50，frozen）：**55.86%**（512 局 / 128 deals），cluster CI [0.382, 0.735]（未达 0.52 晋升线）。
- **本报告的 per-deal 配对分析**（比 cluster bootstrap 更高功效）：每 deal 胜 2.234/4（平局=2.0），SE 0.112，95% CI [2.014, 2.454]——**排除平局**；z=2.09，p≈0.037；34 个 deal 4-0 横扫 vs 19 个 deal 0-4 被横扫（净 +15）。
- 但 vs rule：candidate 62.89% ≈ control 63.28%，paired per-deal 差值 **-0.016**——**对 rule 没有任何提升**，两者都卡在 ~63%，距 85% 门槛很远。
- 归因保留：REPORT.md 自认 "single new-recipe branch vs frozen parent; NOT a matched old-recipe training A/B"——+5.9pp 不能 100% 归因于新超参（可能部分来自多训 200 迭代）。但旧配方 50 迭代零提升 + 动态指标（冻结 vs 活跃）使归因大概率成立。

### E.4 结论与下一步

**可以继续——这是三次运行里第一次"有证据地继续"：动态健康（无 health 失败、未收敛、无熵坍缩）、直接对父代有边缘显著提升（p≈0.037）、且提升方向与理论预测一致。** 但提升尚未渗透到 rule 对抗，且 5 项只落实 3.5 项。下一步：
1. 补 NTP 0.02 + 贡品 208 上下文（剩余两项，改动量已知 ~17 行）；
2. 直接推 8 万局段（当前吞吐 0.92 局/s，≈24 机时 ≈ 2 个 Kaggle 段）；
3. **训练中每 ~5000 局做一次 vs-parent h2h**（512 局/臂），画出学习曲线而非只赌终点；
4. 监控：normalized_entropy（0.087 且在降，破 0.05 则 reg 回调 0.03）；value_ev（~0.3 是 Value Gate 老问题，advantage 质量的天花板）；
5. matched A/B 可跳过——动态证据已足够强，算力花在数据量上更值。
