# MARVEL 原生架构强化改造方案（v1.4 确认稿）独立验证报告

> 验证日期：2026-10-17（含第三方独立裁决复核，同日）
> 验证对象：《MARVEL 原生架构强化改造方案（确认稿）》
> 验证方式：① 逐条核对方案声称的"当前实现"与 `D:/project/guandan` 仓库真实代码；② 用数值实验验证全部数学公式（含对真实 `gae.py` / `losses.py` 的等价性测试）；③ 将关键争议（GAE 公式 vs 验收标准）交第三方模型独立裁决，与本报告结论互证。
> 复现脚本：`scripts/verify_marvel_v14_math.py`（系统 Python 3.12 + torch 2.13.0+cpu，退出码 0）

---

## 一、总结论

**方案整体成立，可以批准方向；Round 0–2 优先实施的判断正确。** 方案对现状的描述与代码高度吻合（20 项全部属实），数学主体正确，三项核心改动（完整动作序列级 PPO、commit-boundary GAE、actor 残差融合）方向正确且互不冲突。

存在 **2 个必须在编码前修正的 P0 级问题** 和 8 个需要写入设计的风险点。其中最关键的 GAE 争议已经第三方独立裁决确认，最终共识如下：

> **方案公式本身（作为新的宏决策 GAE）是正确的；错误的是它把这次改造描述成"旧 GAE 的等价重写"，并据此写下了一条不可能成立的验收标准。正确处置：保留公式、把改造目的改述为"有意切换 λ 的衰减时钟"、重写验收标准、并实现一个 legacy-equivalent oracle 供回归测试。**

第二个 P0（残差融合）同样已经第三方裁决确认：

> **"最后一层初始化为 0"不是可验收的设计描述（零化最后一个残差块无法得到零输出）。正确设计：`z'=z+W_out·G_φ(u)`，`W_out=0`、`bias=False`、no weight decay；第一次 backward 只有输出门有梯度（trunk 梯度为 0 是预期），第二次 backward 梯度才进入 trunk；一致性检查点在 optimizer step 之前。**

---

## 二、现状描述核对（方案声称的"当前是什么样" vs 代码）

| # | 方案声称 | 代码证据 | 结论 |
|---|---|---|---|
| 1 | 逐 token IPPO，每步一个协议 token | `runtime/sampler.py` 每步一个动作；`guandan/action_state.py:397-421` 合法动作集是 next-token 集合（PASS/COMMIT/FAMILY/RANK/SUIT/LENGTH/CARD/WILD） | ✅ 属实 |
| 2 | token 时间轴 GAE（γ=1, λ=0.95） | `training/gae.py:84-112`，逐 token 步递归，默认参数一致 | ✅ 属实 |
| 3 | actor 状态—动作逐元素乘法后直接线性头 | `model/agent.py:153-154`：`player_state * action_feats → logit_head`，无非线性 | ✅ 属实 |
| 4 | 257 参数线性 logit 头 | `logit_head = Linear(feature_dim=256, 1)` → 256+1=257 | ✅ 属实 |
| 5 | 固定等级 15、leader=0、无进贡 | M09 checkpoint 内 `level_rank:15, leader_seat:0, previous_result:null`；`configs/marvel_m09_main_training_plan.json` 明确"does not directly sample all 13 level ranks or tribute contexts" | ✅ 属实 |
| 6 | flat ±1 奖励 | `guandan/round.py:129` `_term`：`rewards=[±1 per seat]`，outcome_class 另行记录 | ✅ 属实 |
| 7 | clip=0.02 | `PPOConfig.base_clip_coef=0.02`；`schedules.py` 按 `min(500/iter,1)` 衰减，iter2000 时降到 0.005 | ✅ 属实（且比方案说的更弱：已衰减） |
| 8 | KL(π‖U) 系数 0.1 进入总损失 | `local.py:653`：`total = pg_loss + schedule.reg_coef * kl_unif + 0.5 * value_loss`；`base_reg_coef=0.1`，sqrt 衰减 iter2000→0.05 | ✅ 属实 |
| 9 | AMP 关闭 | `amp_enabled` 默认 False，且 `local.py:177-178` 对 `amp_enabled=True` **直接 raise**（代码级硬禁用，非配置开关） | ✅ 属实（但见 §6-P1-1） |
| 10 | 无 belief / 无 opponent pool / 无搜索 | 无 belief 头；训练为纯自博弈（pool 仅存在于评估矩阵）；全仓库无 MCTS/搜索 | ✅ 属实 |
| 11 | `fuse_layers=2` 只被 CentralCritic 使用 | `model/critic.py:33-38`；`agent.py` 完全不用 fuse_layers | ✅ 属实 |
| 12 | rollout 已存 `action_taken/behavior_action_prob/policy_probs/avail_is_commit`，不改 C++ 协议即可恢复完整动作段 | `buffer.py:23-36` field specs 五项俱全 | ✅ 属实 |
| 13 | 512 维状态 channels / 双 GPU DDP / 精确恢复 | `num_channels=512`；M09 world_size=2 nccl；`load_state_dict(strict=True)`+RNG+sampler+optimizer | ✅ 属实 |
| 14 | rollout 44% / learner 56% | 实测一次 smoke 记录为 51.7/48.3（`perf/rollout_seconds=6.50, learner=6.07`）。Amdahl 结论对两种拆分都稳健：S=1.32~1.39，S_max=1.93~2.27 | ✅ 方向属实（数值随运行波动） |
| 15 | NTP 关闭 | `ntp_weight` 默认 0.0，`Agent.ntp_enabled_for_config` 据此禁用 | ✅ 属实 |
| 16 | 当前只报告 value_loss（无 EV） | `metrics.py` 无 EV/梯度范数/宏动作指标（已有 commit_steps 计数） | ✅ 属实 |
| 17 | "不蒸馏 FableDan"与仓库架构一致 | FableDan 仅存在于 `comparison/fabledan_guest.py`（评估竞技场 guest，选完整动作），不在训练链路 | ✅ 一致 |
| 18 | collect_batch 4→20 为计划变更 | M09 计划 `collect_batch_size_global: 4` | ✅ 属实 |

**现状核对结论：20/20 属实，无一条描述失实。** 方案作者对代码库的掌握是准确的。

补充实测数据（来自 `runs/M09_full_training` 指标，对评估方案很重要）：
- 每局约 **332 个 token 步 / 144 个 commit 步** → 平均宏动作长度 ≈ **2.3** token；每玩家一局约 83 个自有 token 步、**约 36 个宏决策**；同一玩家相邻宏决策起点的平均 token 距离 τ̄ ≈ 332/36 ≈ **9.2**。
- `max_action_tokens=8` 是候选 action row 的固定编码宽度，**不是完整宏动作的 token 上限**；阶段 1 真实 C++ probe 实测宏长 max=9/10，合成测试覆盖到 24。env `max_token_steps=512`（实测 332–340，余量 ~35%，存在截断风险）。
- clip=0.02 时 smoke 记录 `clip_frac=0.309`（iter 2，4 epoch 后）——约三成 token 样本被裁剪，**信任域已饱和**，直接支持方案"clip 放宽到 0.05 + floor 0.02"的动议（注意：仅 2 局 smoke，属方向性证据）。
- 初始化期 `kl_unif ≈ 0.012–0.023`（策略接近均匀，符合预期）。

---

## 三、数学公式逐项验证

### 3.1 已验证正确的公式（数值实验通过）

| 公式 | 验证方式 | 结果 |
|---|---|---|
| 宏动作概率 = token 条件概率乘积（链式法则） | 数学恒等式 + B4 数值 | ✅ 正确；log 域求和可行 |
| 宏动作 ratio `r_m=exp(Σlogπ_new−Σlogπ_old)` | B2：length-1 环境下宏 PPO loss 与真实 token PPO loss 分别为 −1e-8 / −0.0 | ✅ 数值一致 |
| GAE 衰减表 0.95^20/50/100/200 | 数值计算 | ✅ 0.358 / 0.0769 / 0.00592 / 3.5e-5 全部吻合 |
| `D_KL(π‖U)=log n−H(π)` | 200 组随机分布，误差 <1e-10 | ✅ 正确（本仓库 avail_weights 为 0/1 掩码，U 即合法动作均匀分布，公式严格成立） |
| 熵归一化 `H(π_t)/log\|A_t\|∈[0,1]`，单合法动作为 0 | 边界值 | ✅ 正确 |
| Explained Variance `EV=1−Var(y−V)/Var(y)` | 定义检查 | ✅ 标准定义 |
| Amdahl：S_max=1/0.44≈2.27；S=1/(0.44+0.56/2)≈1.389 | 数值 | ✅ 正确；对实测 51.7/48.3 拆分同样稳健（1.93 / 1.32） |
| "目标放大 3 倍 → Huber/MSE 损失约放大 9 倍" | Huber(delta=10) 数值 | ✅ 精确 9.0（误差在二次段内） |
| `argmax E[R]=argmax E[cR], c>0` | 数学 | ✅ 正确；方案关于"flat ±1 与分差目标最优策略不同、只有整体缩放不改变新目标"的澄清也正确 |
| 座位旋转/副本置换的等变性与对称 KL | 数学 | ✅ 形式正确（注意事项见 §6-P2-3） |
| belief 分类分布 + 牌数/等级守恒损失 | 数学 | ✅ 形式正确 |
| 引用文献 | PPO=1707.06347、GAE=1506.02438、Option-Critic=1609.05140 | ✅ 均正确且贴切 |

### 3.2 GAE 公式争议：结论与第三方裁决互证（P0，本报告最重要的一节）

旧实现（`gae.py:84-112`）向后扫描，**每一个全局 token 步**都执行 `gae *= γλ`（只在行动玩家处更新）。压缩到"同一玩家的相邻决策点"后，等价递归为：

\[
A_m^{\rm old}=\delta_m+(\gamma\lambda)^{\tau_m}A_{m+1}^{\rm old},
\qquad
\delta_m=R_m+\gamma^{\tau_m}(1-d_m)V(S_{m+1})-V(S_m)
\]

其中 τ_m 是该玩家相邻两个宏动作起点之间的 **token 步数**（四人游戏恒 ≥1，实测均值 ≈9.2；只在所有宏动作长度 1 的合成环境中 ≈4）。

方案写出的公式是：

\[
A_m^{\rm plan}=\delta_m+\gamma^{\tau_m}\lambda A_{m+1}^{\rm plan}
\]

**两种写法的差别在 λ 的衰减时钟**，不在环境折扣：

- `γ^{τ_m}`：由 primitive-time 回报定义决定，**环境语义，必须保留**（γ=1 时数值上看不出，γ<1 时保留 γ^τ 才正确）；
- `λ` 的时钟：**估计器设计选择**。每 token 步衰减一次 = 旧实现 / token-time GAE；每宏决策衰减一次 = macro-decision SMDP GAE。

**裁决结论（本报告数值实验 + 第三方独立推导双向确认）：**

| 公式 | 与旧实现等价 | 修复 token 信用衰减 |
|---|---:|---:|
| `(γλ)^{τ_m}`（token 时钟） | ✅ 是（实测差 1.7e-7） | ❌ 否——总 token 距离未变，衰减原样保留 |
| `γ^{τ_m}λ`（宏决策时钟，方案原文） | ❌ 否（实测差 0.49） | ✅ 是——λ 有效尺度从 token 数变为玩家决策数 |

量化收益（每名玩家一局，γ=1）：旧时钟端到端信用权重 λ^{332}≈4e-8（早期动作的优势几乎被抹平）；宏时钟 λ^{36}≈0.158。合成环境上直接测得：同一玩家第一个动作 vs 最后一个动作的信用权重，token 轴 0.0566 vs 宏轴 0.4877（**约 9 倍**，脚本测试 F 精确复现 λ^56 与 λ^14）。

**因此最终处置（替换原验收标准 #3）：**

1. **保留方案公式 `γ^{τ_m}λ`**，但必须把改造目的从"等价重写"改述为：**"有意将 GAE 的 λ 时钟从基础协议 token 步切换为同一玩家的宏决策步，从而改变偏差—方差特性并修复协议 token 膨胀导致的过度 trace decay；环境折扣仍按基础时间保持 γ^{τ_m}。"**
2. **实现一个 legacy-equivalent oracle**：`A_m=δ_m+(γλ_{token})^{τ_m}A_{m+1}`，只用于验证宏分段、reward 聚合、value bootstrap 的正确性，不用于生产训练。
3. **重写验收标准**：原标准"长度 1 协议环境与旧 GAE 数值等价"**不成立**（四人交替下 τ_m>1 恒成立，长度 1 只消除动作内 token，不消除玩家间间隔）。正确条件见 §七-A/C/D/E。
4. **配置显式声明时钟**，不再用裸 `gae_lambda=0.95`：

```jsonc
{
  "advantage_timeline": "player_macro_decision",
  "gae_lambda_clock": "macro_decision",   // 或 "token"（legacy oracle）
  "gae_lambda": 0.95,                      // λ_macro 是超参数，候选 {0.90, 0.95, 0.97, 1.00}
  "environment_discount_clock": "token",
  "discount_gamma": 1.0
}
```

5. 附带澄清（对实现团队重要）：**mixed-length 环境下任何宏公式都不等于 token GAE**——token 时间轴在宏动作内部还有额外决策点（宏动作的每个 token 都是该玩家的行动步，都会重置 GAE buffer）。legacy oracle 的等价性测试因此只能在 all-commit（长度 1）环境上做（脚本 B1b 即此设计）。
6. 附带澄清（对效果预期重要）：若 value 已准确，局部 TD residual 本身就够用，长 trace 的收益会打折；宏时钟同时**提高方差**，是否净收益必须靠配对 A/B 判定，不能因"早期 advantage 变大"就认定有效。

### 3.3 两个精度/尺度细节（P2）

- **宏 logprob 公差 1e-6 偏紧**：L≤8 段在 float32 下累加与 float64 参考的最大差为 1.63e-6（实测）。建议公差写 `<1e-5`（或相对误差），或宏 logprob 内部用 float64 累加。对 PPO ratio 无实质影响，但验收线要可行。
- **乘积 ratio 的 clip 长度依赖**：`clip` 作用于 L 个 token 概率的乘积，L=8 时 1.1^8=2.14——即 clip=0.10 对长动作等效每 token 仅 ~1.3% 的信任域。建议 `macro/clip_frac` **按宏长度分桶**（L=1 vs L≥5）报告，否则 A/B 会被长度分布变化混淆。

---

## 四、对真实代码的数值实验（`scripts/verify_marvel_v14_math.py`）

| 编号 | 检验内容 | 结果 |
|---|---|---|
| A* | 纯数学：衰减表、Amdahl、KL 恒等式、Huber 9x、熵边界、乘积 clip | ✅ 全部通过（§3.1） |
| B1a | 方案原文宏 GAE（γ^τ·λ）≡ 现 token GAE（长度 1 环境） | ❌ 最大差 0.490（即 §3.2 的问题；已第三方裁决确认是预期行为而非 bug） |
| B1b | Token 忠实宏 GAE `(γλ)^τ` ≡ 现 token GAE（长度 1 环境） | ✅ 最大差 1.7e-7 |
| B2 | 长度 1 宏动作：sequence PPO loss ≡ 旧 token PPO loss | ✅ −1e-8 vs −0.0（给定相同 advantage 时严格一致） |
| B3 | 宏分段不跨玩家、每段以 COMMIT/PASS 结束 | ✅ |
| B4 | 宏 logprob = token logprob 之和（log 域） | ✅（float32 累加误差 1.63e-6，见 §3.3） |
| B5 | `chosen_is_commit = gather(avail_is_commit, action_taken)` 恰好标出 COMMIT/PASS 步 | ✅ |
| B6 | 终局奖励符号正确进入四名玩家的最后宏动作 | ✅ {0:+0.97, 1:−1.02, 2:+0.93, 3:−0.93} |
| **C** | **退化等价 τ_m≡1**（单玩家连续行动环境）：两种宏公式都 ≡ token GAE | ✅ 均 4.4e-8 |
| **E1** | **端点 λ=0/1**（all-commit 环境）：两种宏公式都 ≡ token GAE | ✅ 0.0 / 1.1e-7 |
| **E2** | **端点 λ=0/1**（mixed 环境）：两种时钟公式互相塌缩为同一形式 | ✅ 精确 0 |
| **F** | **信用传播**（仅终局 reward、V=0）：token 轴权重 = λ^56、宏轴 = λ^14 | ✅ 均精确复现；0.0566 vs 0.4877 |
| D1 | 混合长度环境下两种时间轴的 advantage 分歧 | 平均\|差\|=0.54、最大 0.95（advantage 为 O(1) 量级，分歧实质） |
| D2 | 现 token GAE 的动作内不一致性（方案核心论据） | 同动作内首 token 与 commit token 的 advantage 比：均值 0.54，范围 **−10.5 ~ +44.5（符号翻转实测严重）**——强力支持"一个动作内部 token 获得不一致战略评价"的批评 |
| **G1/G1b** | **残差融合零初始化**：`W_out=0` 时 fused 特征与 logits 对旧 `s*a` 路径逐位一致 | ✅ `torch.equal` 通过 |
| **G2/G2b** | **梯度门控时序**：第一次 backward 输出投影梯度 4.03（非零）、trunk/input_proj 梯度精确为 0 | ✅ 与裁决的预期门控行为一致 |
| **G3/G3b** | 第一次 optimizer step 后：第二次 backward 梯度进入 input_proj（1.21）与 trunk（0.108），且特征开始偏离旧路径 | ✅ 梯度门打开、学习激活 |
| **G4** | output_proj bias 经共享 logit 头后对所有候选是同一偏移、softmax 中抵消 | ✅ 数值证实（`bias=False` 有依据） |
| **G5** | 零化最终 RMSNorm scale 也可得精确零输出（裁决对我方绝对化表述的修正） | ✅ 数值证实；但 trunk 首步无梯度，`W_out` 方案仍优先 |

---

## 五、各项改动的理论合理性评估

| 改动 | 理论评价 | 关键保留 |
|---|---|---|
| 1. 完整动作序列级 PPO | **合理且必要**。PPO action 与战略动作对齐；链式法则保证与自回归架构严格兼容；变长动作保留。 | 乘积 ratio 方差随 L 增长；clip 有长度依赖（§3.3） |
| 2. commit-boundary GAE | **合理，是全方案最大收益点**（信用链衰减从 λ^332≈4e-8 修复到 λ^36≈0.158，约 7 个数量级）。公式正确、目的表述错误，按 §3.2 六点处置。 | λ_macro 需 A/B；宏时钟提高方差，净收益靠配对试验；mixed 环境不等价是预期（§3.2-5） |
| 3. actor 残差融合 | **合理**。保留旧乘法路径 + 零初始化残差，warm start 无损，可逐步学习跨维交互（s_i·a_j）。设计已经第三方裁决确认为当前仓库条件下最严密的方案：`z'=z+W_out·G_φ(u)`，`W_out=0`、`bias=False`、不参加 weight decay。 | **原方案"最后一层初始化为 0"不是可验收的设计描述**：零化最后一个残差块的 `down_proj` 不能使输出为 0（residual bypass + 末端 RMSNorm）；零化 `output_norm.weight`（RMSNorm scale）或输入投影虽可实现零输出，但 trunk 首步无梯度、且把门控与特征提取混在一层。必须改为显式零初始化输出投影（§6-P0-2、§七-B）。另有性能陷阱：fuse MLP 必须像 `_encode_sparse_actions_eager` 一样只对 legal action 稀疏计算（B×T×A×D 密集跑不可行）。 |
| 4. 分差奖励 + /3 归一化 | **合理**。保留 DOUBLE_DOWN>HEAD_THIRD>HEAD_LAST 的序结构；/3 控制 value 尺度；9x 损失论证正确。 | 可简化：outcome_class 已随 terminal info 流出（metrics.py 已在用），**C++ 改动（guandan.cpp）并非必需**，纯 Python reward adapter（按 trajectory 的 outcome_class 把 ±1 改写为 ±3/±2/±1 再 /3）即可，显著降低双引擎一致性风险 |
| 5. 全上下文训练（13 等级 × 4 leader × 进贡模式） | **理论论证正确**：p_train=δ(c0) 对实战分布无约束，这是目标函数层面的修改而非数据增强。warm-start 三阶段课程风险可控。 | schema 需扩 `previous_result`；provenance 已记录 hands/level/leader/previous_result，扩字段即可；验收强调"看实际完成牌局而非 manifest coverage"是对的 |
| 6. 宏归一化熵 | **数学正确**，消除动作长度对正则的偏置。 | **系数尺度不可直接平移**：旧形式为 0.1→0.05 · KL(π‖U)（nats，观测值 ~0.02 → 项 ~0.002），新形式为 0.01 · H̄∈[0,1]（近均匀策略时 ~0.01）——探索压力可能不降反升约 5 倍。过渡期必须**双轨监控** kl_unif 与 H̄，并重调 β；另建议 Round 1 增加"仅换熵形式"的消融臂，避免把两个变化混在一个 gate 里 |
| 7. value 可靠性指标体系（EV/相关/ranking/分桶校准） | **完全正确且必要**。constant-0 预测器的反例论证成立。 | 分桶（early/mid/late × outcome class）需注意小样本桶的 CI |
| 8. 座位/副本对称性 | 数学正确，减少无效自由度。 | **副本不变性与进贡规则冲突**：`tribute_card_for` 以 card_id 决胜（`round.py:118-119`），同牌两副本时强制进贡牌由 copy ID 决定——硬对称损失会与最优策略冲突。Round 4 的 gate 应在非进贡/还牌阶段评估，或对 symmetry loss 用软约束（小 λ_s） |
| 9. belief 辅助任务 | 方法学要求正确：deal 级切分（避免同牌相邻状态泄漏）、公开信息强基线、共享主干梯度比 ρ∈5–20% 作为工程目标。 | "完整 history probe 相对公开信息基线的增量"作为关键判定，方法论正确 |
| 10. NTP 替换 | **判断正确**：现 NTP 标签≈行为方自己刚提交的 token（非 commit 步 delta 为空、commit 步 delta=刚出的牌），确无信息量。 | 新候选（下一公共动作类型等）同样需要先过信息量预检，方案已写明 |
| 11. checkpoint opponent pool | 理论作用描述准确（不保证纳什，但抑制策略循环）；`policy_train_mask` 只让当前座位进梯度，与现有 `on_policy_valid` 校验兼容（对手动作仍进历史）。 | 建议初始就记录 pool 采样比例与对手版本号进 metrics，便于归因 |
| 12. AMP | Amdahl 论证正确且保守（rollout 前向若也 autocast，实际上界优于 1.389）。 | **当前是代码级 raise**（§6-P1-1）；GradScaler 存档/恢复方案正确 |
| 13. 实施顺序（Round 0–7） | Round 0 先建观测基线、Round 1/2 单变量验证、逐项 gate，符合受控实验原则；dev/selection/holdout 三集隔离 + cluster bootstrap 的最终评估体系正确。 | 微调建议见 §6-P1-3、P1-4 |

---

## 六、问题清单（按严重程度）

### P0 — 编码前必须解决

1. **GAE 公式的目的表述与验收标准 #3 错误**（§3.2，已经第三方裁决确认）：公式 `γ^{τ_m}λ` 本身作为宏决策 GAE 是正确的，但①不能声称与旧实现等价；②验收标准"长度 1 协议环境数值等价"不成立（正确条件是 τ_m≡1，且 λ∈{0,1} 时两时钟自然塌缩）；③必须同时交付 legacy-equivalent oracle（`(γλ)^{τ_m}`）供回归；④配置显式声明 `gae_lambda_clock`；⑤向团队明示"mixed 环境不等价是预期行为，0.49 量级的 advantage 差异不是 bug"。
2. **残差融合的零初始化设计不可验收**（§5-3，已经第三方裁决确认）：原方案"`F_φ` 使用 ResGatedMLP/SwiGLU；最后一层初始化为 0"不是可实现的设计描述——
   - 零化最后一个残差块的 `down_proj` **不能**使 F_φ(u)=0：残差恒通路会把非零输入传到末端，最终 RMSNorm 输出非零；
   - 零化 `output_norm.weight`（RMSNorm scale，等价于逐维 γ=0）或零化输入投影**可以**实现零输出，但 trunk 在第一次 backward 梯度为 0，且门控与特征提取混在同一层，不清晰；
   - **必须改为显式零初始化输出投影**：`z'=z+W_out·G_φ(u)`，`W_out` 权重与训练均零初始化、`bias=False`（bias 经共享 logit 头后对所有候选是同一偏移，softmax 中抵消，不可辨识）、`W_out` 不参加 weight decay（与仓库 `_mark_no_weight_decay` 惯例一致；且 AdamW 的衰减不会把零权重变成非零，但会持续把门压回零）；
   - 验收措辞必须修正：第一次 backward 时**只有** output_proj 获得非零梯度，trunk/input_proj 梯度为 0 是**预期行为**；第一次有效 optimizer step 后，第二次 backward 梯度才进入 input_proj 与 trunk；
   - 配套要求：fuse MLP 只对 legal action 稀疏计算（密集 B×T×A×D 不可行）；架构版本升为 v1.4；评估加载按架构 strict=True；warm-start 走白名单迁移（只允许缺 `actor_fuse.*`/`ntp_head.*`/`belief_head.*`，且加载后断言 W_out 全零 + 固定输入探针逐位一致）；v1.3→v1.4 的 exact resume 必须拒绝。

### P1 — 需要写入设计

3. **AMP 是代码级硬禁用**：`local.py:177-178` 对 `amp_enabled=True` 直接 raise。启用 AMP 必须改代码（删 raise + 实现 autocast/GradScaler/unscale_ 后裁剪/scaler 存档恢复），不是改配置。方案文件清单应补上 `training/local.py`（已列）并注明这一点。
4. **评估链路 strict 加载**：`evaluation/checkpoints.py:175` 用 `strict=True` 加载旧 checkpoint；新增 `actor_fuse` 模块后，旧 M09 iter2000 评估 checkpoint 会直接加载失败。warm-start 训练路径（`local.py:283-291`，只放行 ntp_head 缺失）和评估路径都要加兼容逻辑。方案已列出该文件，但需明确"两条路径"。
5. **熵系数尺度切换**（§5-6）：旧 `0.1·KL(nats)`（衰减到 0.05）与新 `0.01·H̄_normalized` 不是同一个尺度，观测数据表明探索压力可能不降反升。双轨监控 + 消融臂 + β 重调。
6. **乘积 ratio clip 的长度依赖**（§3.3）：clip_frac 按宏长度分桶；A/B 的 clip 候选（0.05/0.10）要分别看 L=1 与 L≥5 子集。
7. **分差奖励的 C++ 依赖可去除**（§5-4）：outcome_class 已在 terminal info 中，纯 Python adapter 即可，避免动 `nanopokerlib/cpp/guandan/guandan.cpp` 引入双引擎不一致风险。

### P2 — 防御性/细节

8. **截断尾段**：`max_token_steps=512`，实测每局 332–340 token（余量 ~35%），策略变化后可能触顶。截断可能发生在非 commit token 中途（`action_state.py:513-517`），此时最后一段不以 COMMIT 结束——分段代码必须显式处理（标记 incomplete，从宏 loss 排除或按 bootstrap 处理），不能假设"所有段都以 COMMIT/PASS 结束"。gate 里"0 truncated"要持续监控。
9. **宏 logprob 公差**：1e-6 在 float32/L=8 下偏紧（实测 1.63e-6），建议 1e-5 或 float64 累加（§3.3）。
10. **副本对称性 vs 进贡 tie-break**（§5-8）：`tribute_card_for` 按 card_id 决胜，Round 4 gate 应在非进贡阶段评估对称性误差。
11. **λ 语义变化**（§3.2）：从每 token 步变为每宏决策，λ=0.95 是合理起点但应纳入 A/B（候选 0.90/0.95/0.97/1.00）。

---

## 七、验收标准修订建议

### 7.1 GAE 部分（对应方案改动 2 的验收清单）

原清单第 1–6 条中，第 3 条作废，其余保留并补充。修订后的完整验收组：

- **A. Legacy oracle 等价性**（证明宏分段/reward 聚合/bootstrap 正确）：legacy-equivalent 实现 `(γλ_token)^{τ_m}` vs 当前逐 token 旧实现，在 **all-commit（长度 1）环境**上 `max_abs_diff < 1e-6`（float32）。测试矩阵须覆盖：四玩家交替、可变宏长 1–8、玩家提前出完、接风、终局、中间非零 reward、γ=1 生产参数与 γ<1 人工参数。
- **B. Macro GAE 数学正确性**：手工构造宏序列（τ、R、V、γ、λ_macro 显式给定），按公式手算 A_m 与实现比对 `< 1e-6`。
- **C. 退化等价**（替换原作废的第 3 条）：当同一玩家每个相邻决策满足 **τ_m=1**（单玩家环境或人工 TensorDict 使同玩家连续行动）时，新旧递归数值相等。**不能**用"四人环境 + 每宏动作长度 1"作为等价条件。
- **D. 有意不等价断言**：四人交替、每宏动作长度 1、τ=4、γ=1、λ=0.95 时，显式断言 legacy 系数 0.95^4 与 macro 系数 0.95 不相等——0.49 量级的 advantage 差异是预期算法变化，不得再被当作 bug。
- **E. 端点**：λ=0（两公式都退化为 A=δ）与 λ=1（都退化为 A=δ+γ^τA_next）时，两种时钟公式互相相等且与 token GAE 相等（脚本 E1/E2 已精确验证）。
- **F. 信用传播**：仅终局 reward、V=0 时，验证最早动作的终局 residual 权重：legacy = λ^{token 距离}、macro = λ^{宏决策数}（脚本 F 已精确复现）。
- **G. 实战 A/B**（数学正确 ≠ 棋力提升）：control=token 时钟 vs treatment=macro 时钟，其余全部固定（同 warm start、同发牌、同局数、同奖励、同 actor、同超参、同评测牌集），报告 advantage std、early/mid/late advantage 量级、value EV、clip fraction、grad norm、配对胜率（按 deal 聚类、cluster bootstrap），通过对规则/对 FableDan 不退化，DOUBLE_DOWN 率健康；通过线为 **paired CI95 lower > 0**。

其余验收项（commit 计数对齐、终局奖励进四名玩家最后宏动作、两步手搓游戏、EV 分桶）保留不变，本报告 §四 的 B3/B6/F 已为其提供了现成的数值基座。

### 7.2 残差融合部分（对应方案改动 3 的验收清单，已经第三方裁决确认）

原方案 6 条验收标准中，第 2、3 条的表述必须修正（"第一次反向传播：final fuse projection 有非零梯度"正确；但"fuse 前层获得非零梯度"必须注明发生在第二次 backward）。修订后的完整验收组（脚本测试组 G 已逐条数值验证）：

- **A. 初始化函数等价**：旧 v1.3 权重加载到 v1.4 后、**任何 optimizer step 之前**，FP32、`model.eval()`、autocast 关闭、dropout 关闭，相同输入下 `torch.equal(old_logits, new_logits)` 且 argmax 逐位相同。测试输入须覆盖：随机合成、真实 rollout、多合法动作、单合法动作、长 prefix、各牌型、配子、贡还牌。AMP 下不作为硬合同（`assert_close` rtol=1e-3/atol=1e-4 + argmax 一致 + 无 NaN/Inf）。
- **B. 初始化参数检查**：`output_proj.weight` 全零（`count_nonzero == 0`）；input_proj/trunk 权重非零；output_proj 已进 optimizer 参数组、**不在** weight decay 组。
- **C. 第一次 backward**：`output_proj.weight.grad norm > 0`；`trunk`/`input_proj` grad norm **== 0**（预期门控行为，不是失败——不得写成"第一次 backward 整个新支路所有参数都有梯度"）。
- **D. 第一次 optimizer step 后**：`output_proj.weight norm > 0`，new logits ≠ old logits——此时**不再要求一致**（否则等于要求新支路永远不学习）。
- **E. 第二次 backward**：`output_proj`/`input_proj`/`trunk` grad norm 均 > 0，证明梯度门已打开。
- **F. AMP**：无 NaN/Inf；GradScaler 正确（`scale→backward→unscale_→clip→step→update`）；AMP on/off 初始 argmax 一致；output_proj 梯度无 underflow。
- **G. Checkpoint 矩阵**：v1.3 评估 checkpoint 按 v1.3 构造加载 strict 成功；v1.4 按 v1.4 加载 strict 成功；缺 fuse key 的 v1.4 checkpoint strict **失败**（不允许静默降级）；v1.3→v1.4 policy-only warm start 走白名单成功 + 加载后断言 W_out 全零 + 探针逐位一致；v1.3→v1.4 exact resume **拒绝**；出现非白名单缺失或 unexpected key 拒绝。
- **H. DDP**：双 rank 初始 fuse 参数 hash 一致；第一次 step 后同步；recovery 后参数/optimizer state/输出一致；`find_unused_parameters=False` 不报错（trunk 的 grad 是零张量而非 None，正常不会触发；但双 rank 必须实测）。
- **I. 性能**：fuse 前向只对 legal `(row, action)` 稀疏计算（对齐 `_encode_sparse_actions_eager` 模式），显存/吞吐相对旧路径的增量在预算内。

---

## 八、最终结论

1. **方案可以批准**：20 项现状描述全部属实；数学主体正确；三项核心改动方向正确。Round 0–2 优先实施、逐项 gate 的判断成立。
2. **GAE 是全方案价值最大的单项，但表述必须修正**：把 λ 时钟从 token 步切换到宏决策步，可把每玩家一局的端到端信用衰减从 λ^332≈4e-8 修复到 λ^36≈0.158。公式保留；删掉"等价重写"的错误表述；按 §七-A~G 重写验收组；交付 legacy oracle；配置显式声明时钟。
3. **先修两个 P0 再动手**：GAE 表述/验收/oracle 三件套；残差融合改为显式零初始化输出投影设计（`W_out=0`、`bias=False`、no weight decay、稀疏计算、架构升 v1.4），并按 §七-2 重写验收措辞。
4. **建议的减风险简化**：分差奖励用纯 Python adapter（免改 C++）；Round 1 加熵形式消融臂；clip_frac 按宏长度分桶；截断尾段显式处理。
5. 若 Round 1–2 在同预算配对试验中仍不能显著超过 token IPPO，方案第十九节的退出准则（问题可能深入状态表示或自回归动作本身）是诚实且正确的。

---

## 九、独立裁决复核记录

2026-10-17，将本报告 §3.2 的核心争议（含旧实现递归推导、方案公式、合成环境实测数据）提交第三方模型独立裁决，结论与本报告完全互证：

- 旧实现压缩为 `(γλ)^τ`——**本报告正确**（裁决独立推导一致）；
- "长度 1 宏动作 ⇒ 新旧等价"——**方案错误**（τ_m 是玩家间距离，与宏长度无关；四人交替 τ≈4，真实协议均值 ≈9.2）；
- `(γλ)^τ` 可复现旧实现——**正确**；`(γλ)^τ` 可修复信用衰减——**错误，它保留旧衰减**；
- `γ^τ·λ` 作为宏决策 SMDP GAE——**正确**；作为旧实现等价式——**错误**；
- Semi-MDP 理论唯一强制的是 γ^τ（环境语义），λ 的时钟是估计器设计选择——**本报告 §3.2 的表述与此一致**；
- 最终判决：**反驳推翻了方案的"等价性论证和验收标准"，但没有推翻宏动作 GAE 本身**；方案必须把真实目的说清楚——这是一次有意改变 trace 时间尺度、牺牲旧数值等价性来修复长信用链的算法改造。

该裁决补充的两点已并入本报告：真实协议 τ̄≈9.2（原稿仅区分了合成环境的 4）；λ=0/1 是两个时钟自然塌缩的强实现检查点（已落入脚本 E1/E2 与 §七-1E）。

同日，第二份独立裁决针对残差融合（Prompt 2 争议）复核，结论：我方"零化最后一个 SwiGLU `down_proj` 不能使 ResGatedMLP 输出为 0"的推理**正确**（residual bypass + 末端 RMSNorm）；但我方"ResGatedMLP 无论如何都不可能零输出"的表述**过于绝对**——零化最终 `output_norm.weight`（RMSNorm scale，等价逐维 γ=0）或零化输入投影均可实现精确零输出（脚本 G5 已数值证实），代价是 trunk 首步无梯度、门控与特征提取混层。裁决终审推荐与我方一致：**显式零初始化输出投影 `z'=z+W_out·G_φ(u)`（W_out=0、bias=False、no weight decay）是当前仓库条件下最清晰、可验证、适合无损 warm start 的方案**；并修正了两条验收语言（第一次 backward 只有输出门有梯度、trunk 梯度为 0 是预期；一致性检查点在 optimizer step 之前而非之后）、补充了 bias 不可辨识性论证、架构升 v1.4 + 按架构 strict 加载 + warm-start 白名单 + exact-resume 拒绝、以及 fuse MLP 必须稀疏计算等工程约束。均已并入 §5-3、§6-P0-2 与 §七-2。

---

## 附：验证环境与复现
- 脚本：`scripts/verify_marvel_v14_math.py`（本次验证的全部断言；运行 `python scripts/verify_marvel_v14_math.py`，退出码 0。其中 B1a 的"失败"是 §3.2 记录的设计问题本身，已转为 `[FINDING]` 输出而非测试失败）。测试组：A 纯数学、B1–B6 对真实代码的等价性/分段/奖励归集、C τ=1 退化等价、E1/E2 λ 端点、F 信用传播、G1–G5 残差融合零初始化设计（bitwise 一致性、梯度门控时序、bias 不可辨识、RMSNorm scale 替代方案）。
- 代码核对范围：`guandan_marvel/{training,model,runtime,evaluation}/*.py`、`guandan_marvel/{buffer,config,env_batch}.py`、`guandan/{round,action_state,state,env_batch,evaluation_stats}.py`、`configs/marvel_m09_*.json`、`runs/M09_full_training/**` 指标与 checkpoint。
- 数值实验：真实 `compute_gae_from_value` / `compute_clipped_pg_loss` + 独立实现的宏 GAE/宏 PPO 参考，合成 rollout 严格遵循 MARVEL rollout 契约（字段、形状、掩码语义）。

---

# 十、分阶段实施总方案复核（2026-10-17）

对方案作者随后提交的《MARVEL 原生架构分阶段实施总方案》（阶段 0–12）的复核结论。

## 10.1 总评

**分期整体合理，可以按此推进，先启动阶段 0。** 该方案完整吸收了前两轮裁决结论（legacy oracle、λ 时钟显式声明、fuse 零初始化输出投影与两步梯度门控、v1.4 版本化、strict 评估/白名单 warm start/exact-resume 拒绝），并且在实验设计上比原方案更进一步：**阶段 2（只换 sequence ratio、保留旧 λ 时钟）与阶段 3（只切换 λ 时钟）被拆开**，这比本报告原 Round 1 的捆绑设计更干净——ratio 统计单位与信用时钟两个变化得以分别归因。

## 10.2 已代为核实的两个关键技术前提

1. **阶段 7B 的进贡观察担忧属实**。`guandan_marvel/game.py:388-411`：进贡/还牌阶段的 commit 事件对非行为方是私有的（`EVENT_PRIVATE_EXCHANGE_COMMIT`，仅行为方可见）；交换结算时所有玩家收到同一个 `EVENT_EXCHANGE_RESOLVED` 事件，全部 transfer 记录（TRIBUTE/RETURN × source/target/card × COMMIT）拼接进单一 token 序列，而 `model/base.py:454-457` 的 `_embed_tokens` 对 token 维度**求和**——多条转移的配对关系只隐含在 token 顺序中，无位置结构。channels 虽编码了 tribute_donors/order/selections/anti_tribute 等部分信息，但"能否从 policy observation 恢复公开贡还配对关系"确实是开放问题，7B 先探测后训练的 gate 设计正确。
2. **阶段 2 的 advantage 定义存在歧义，已量化**。方案写 `A_m = A_{t=s_m}^{legacy}`，可读作 (a) 宏级别 legacy oracle 输出的 A_m，或 (b) token GAE 在宏起点步的取值。二者在 mixed-length 环境不等：实测（合成 mixed 环境，λ=0.95）mean=0.022、max=0.267（advantage 量级 ~0.2）；差值解析形式为 γ(1−λ)×动作内 value 差异，在 λ=1 与长度 1 环境归零（与本报告 §3.2-5 的"动作内额外决策点"同源）。

## 10.3 必须明确/修正的点（按严重程度）

1. **（P1）阶段 2 的 advantage 定义二选一并写入验收**：建议取 (a) 宏级别 oracle 值（与"宏动作即决策单位"一致）；同时在阶段 1 验收中把 mixed 环境的差值（γ(1−λ)×动作内 value 差异，λ=1 或长度 1 时归零）列为**已知预期 delta** 的量化测试项，不要让实现者误以为 oracle 与 token GAE 应逐点相等。
2. **（P1）value loss 的归属要逐阶段写明**：阶段 2 必须保持现有 token 级 value 训练不变（每个行动 token 步都算 value loss），阶段 3 才改为只在宏起点训练。否则实现者可能"顺手"把 value 也改成宏级，阶段 2 的 A/B 就被污染。
3. **（P1）阶段 2 实际含两个变化**（PPO 统计单位 + 熵归一化）。建议增加第三臂（macro PPO + 旧式逐 token 平均熵）做消融，或至少在报告中双轨记录 kl_unif 与 H̄ 的实际量级——本报告 §5-6 已指出两者系数尺度不可直接平移（0.1·KL(nats) vs 0.01·H̄）。
4. **（P2）阶段 3 把 λ 固定为 0.95**：λ_macro 是超参不是理论值，建议 A/B 增加 λ∈{0.90, 0.95, 0.97, 1.00} 扫描臂（至少第二 seed）。
5. **（P2）阶段 3 同时改了 value 目标**（时钟切换 + value 只在宏起点训练）：二者捆绑有内在合理性（宏级 advantage 需要宏级 value 目标才自洽），但应在文档中明示，失败时知道回退哪一部分。
6. **（P2）阶段 5 必须删除 `training/local.py:177-178` 的 AMP 硬 raise**——当前 `amp_enabled=True` 会直接抛错，不是配置开关（本报告 §6-P1-1）。
7. **（P2）阶段 6 的 C++ 双引擎改造可免除**：outcome_class 已随 terminal info 流出（`training/metrics.py` 已在用），纯 Python reward adapter（按 trajectory 的 outcome_class 把 ±1 改写为 ±3/±2/±1 再 /3）即可，避免动 `nanopokerlib/cpp/guandan/guandan.cpp` 引入双引擎不一致风险（本报告 §5-4）。若坚持双引擎，需把序列化不变量测试列入 gate。
8. **（P2）阶段 8 的对称性 gate 必须排除进贡/还牌阶段**：`guandan/round.py:118-119` 的 `tribute_card_for` 以 card_id 决胜，同牌两副本时强制进贡牌由 copy ID 决定——副本不变性在此被规则本身破坏，硬对称损失会与最优策略冲突（本报告 §6-P2-3）。gate 应在 PLAY 阶段评估，或 symmetry loss 用小 λ_s 软约束。
9. **（P3）阶段 1 需定义 τ 对 chance 步的计量**：chance/environment 步（player_ids<0）不计入 policy macro，但 τ_m 的 token 距离应包含它们（与 GAE 的 token 时钟一致），建议在 `macro_actions.py` 文档中写死。

## 10.4 结论

分期顺序、依赖关系（9A 失败则取消 belief、11A 失败则取消搜索）、三态统计判定、逐阶段交付清单与独立验证流程均正确。按上述 9 点修正后即可启动；**阶段 0 的 probe 数据建议额外保存宏分段诊断基线值**（macro/count、length 分布、legacy oracle 与 token GAE 在固定 probe 上的差值），供阶段 1 验收直接比对。
