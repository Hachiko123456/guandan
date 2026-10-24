# 阶段 2 实施 Prompt（修订版）：完整宏动作 Sequence PPO 与玩家宏决策 GAE

> 将本文件完整交给 OpenCode 的 `orchestrator` 执行。阶段 1 的宏分段、Legacy GAE Oracle、padding/fuzz、真实 C++ 对账与 baseline manifest 已完成；实施者必须先复核前置门禁。实施结束后必须由全新的只读 `reviewer` 独立核验。Reviewer 明确 `VERDICT: PASS` 且 `NEXT_STAGE: ALLOW` 前，不得进入阶段 3。

## 任务

在工作区：

```text
D:/project/guandan
```

只实施 MARVEL v1.4 阶段 2：

1. 完整宏动作 Sequence PPO；
2. 保留逐位兼容的 legacy token PPO/GAE/value/regularizer 路径；
3. 提供只替换 policy ratio 的因果隔离模式；
4. 实现玩家宏决策时钟 GAE 与宏起点 value loss；
5. 实现独立、可关闭的宏归一化 uniform-KL/entropy 正则消融；
6. 完成单机、双 rank DDP、真实 C++ smoke 与 durable recovery/resume 验证。

本阶段不承诺棋力提升。没有正式、配对、独立评估数据时，只能声明实现正确或短程训练健康，不能声明已追平或超过 FableDan。

---

# 零、前置门禁与工作区安全

## 0.1 前置门禁

修改训练代码前必须先执行并保存结果：

```text
python -m pytest \
  tests/marvel_training/test_macro_actions.py \
  tests/marvel_training/test_macro_actions_cpp_env.py \
  tests/marvel_training/test_macro_actions_padding_fuzz.py \
  tests/marvel_training/test_marvel_v14_phase1_baseline.py \
  -q
```

并验证：

- `docs/marvel_v14/phase1_baseline_manifest.json` 的声明哈希全部可复算；
- `incomplete_count == 0`；
- `player_switch_violation_count == 0`；
- 真实 C++ probe 的 engine commit 数与宏数一致；
- Legacy GAE Oracle 在 `gamma=0/0.5/0.97/1`、`lambda=0/0.5/0.95/1` 下继续通过；
- baseline 中记录的阶段 2 Prompt 哈希与当前文件一致。

任一前置门禁失败时立即停止，不得在阶段 2 中顺手修复阶段 1；应输出 blocker 和单独修复 Prompt。

## 0.2 工作区安全

- 工作区存在未提交修改和未跟踪文件。
- 禁止 `git reset`、`git clean`、`git checkout`、`git restore`、`git commit`、`git push`。
- 不得覆盖、删除、回滚或批量格式化与本阶段无关的文件。
- 不得修改、删除或放宽阶段 1 的宏分段、Legacy Oracle、正数 padding player/action 反例、500 组 fuzz、manifest、自引用防护和真实 C++ 对账。
- 不得提前实施阶段 3 及以后内容：actor fuse、AMP/GradScaler、分差奖励、完整 context schedule、belief、训练期 opponent pool、部署搜索。
- 未经用户明确授权，不得启动长程 Kaggle 训练或高成本正式评估。
- 实施完成后停止，等待独立验收。

## 0.3 算法范围

本阶段的 macro PPO 与 macro GAE **只接入 IPPO**。

- `algo_name=ippo`：支持下文定义的阶段 2 profiles。
- `algo_name=vrpo`：保留现有 legacy 行为；若请求任何 macro profile，配置校验必须 fail-fast，不能静默退回 token 语义，也不能擅自把 IPPO 的宏 GAE 套到 VRPO Q-critic。

---

# 一、不可混淆的训练语义矩阵

不要用多个可任意组合的布尔值表达训练语义。推荐增加一个版本化枚举：

```jsonc
{
  "training_semantics_profile":
    "legacy_token_v1" |
    "macro_sequence_ablation_v1" |
    "macro_decision_v1",

  "policy_regularizer_mode":
    "legacy_token_kl_uniform_v1" |
    "macro_normalized_kl_uniform_v1"
}
```

字段名允许采用项目风格的等价命名，但必须是显式枚举、可序列化、进入 resolved config、semantic contract、checkpoint identity 和 recovery；不得依赖隐含布尔组合。

核心三种 profile 的语义必须固定为：

| Profile | policy objective | policy advantage | value objective | 默认 regularizer | 用途 |
|---|---|---|---|---|---|
| A `legacy_token_v1` | 旧 token PPO | 旧 token GAE，所有 active token | 旧 token value target，所有 active token | 旧 token KL-to-uniform | 逐位兼容基线 |
| B `macro_sequence_ablation_v1` | 完整宏动作 PPO | **生产旧 token GAE 在 macro start 的值** | **旧 token value loss，所有 active token，完全不变** | **旧 token KL-to-uniform，完全不变** | 只隔离 ratio 统计单位 |
| C `macro_decision_v1` | 完整宏动作 PPO | 玩家宏决策 GAE | 宏起点 value target，每宏一次 | 默认仍用旧 token KL-to-uniform | 最终宏语义 |

关键约束：

1. Profile B 只能改变 `token ratio -> macro sequence ratio`。
2. Profile B 不得把 value loss 改成只在宏起点计算；否则 A/B 同时改变 actor ratio 与 critic 采样单位，失去因果归因。
3. Profile B 不得默认切换 entropy/regularizer；否则 A/B 同时改变 ratio 与探索正则。
4. Profile C 才同时启用玩家宏决策 GAE 与宏起点 value target，因为两者在语义上必须配套。
5. `macro_normalized_kl_uniform_v1` 是独立消融开关，默认关闭；主要比较时 A/B/C 均先使用 `legacy_token_kl_uniform_v1`。
6. 如实现者不用单一 profile 枚举，而使用多个枚举字段，必须对支持矩阵做严格校验，并拒绝所有未定义组合。尤其必须拒绝 `token PPO + player macro GAE`。
7. `environment_discount_clock` 固定为基础 token 时间，不允许在本阶段配置为 macro 时间。

默认值必须是：

```text
training_semantics_profile = legacy_token_v1
policy_regularizer_mode = legacy_token_kl_uniform_v1
```

因此升级代码但不改配置时，训练行为保持 legacy。

---

# 二、完整宏动作 Sequence PPO

## 2.1 宏动作定义

一个 policy macro 是同一行动玩家从空 action prefix 开始，直到选中 `COMMIT` 或 `PASS` 的连续策略 token：

\[
m=(a_{s_m},a_{s_m+1},\ldots,a_{e_m}).
\]

- `s_m`：阶段 1 已验收的 `macro_start_index`；
- `e_m`：选中 `COMMIT` 或 `PASS` 的 token；
- `h_t`：token 决策时模型可见历史和 prefix；
- `chance`、`environment-controlled`、padding 不属于 policy macro；
- 它们不进入 macro policy probability，但位于两个同玩家宏起点之间时必须计入环境 token 距离。

必须复用阶段 1 已验收的宏分段语义；不得另写一套边界规则导致诊断和训练分叉。

## 2.2 宏动作概率与 ratio

根据自回归链式法则：

\[
\log \pi_\theta(m\mid S_m)
=
\sum_{t=s_m}^{e_m}
\log \pi_\theta(a_t\mid h_t).
\]

行为策略宏 log-probability：

\[
\log \pi_{\mathrm{old}}(m\mid S_m)
=
\sum_{t=s_m}^{e_m}
\log p_{\mathrm{behavior},t}.
\]

宏 log-ratio 与 ratio：

\[
\ell_m
=
\log \pi_\theta(m\mid S_m)
-
\log \pi_{\mathrm{old}}(m\mid S_m),
\qquad
r_m=\exp(\ell_m).
\]

宏 clipped PPO loss：

\[
L_{\mathrm{PG},m}
=
\max\left(
-r_m\hat A_m,
-\operatorname{clip}(r_m,1-\epsilon,1+\epsilon)\hat A_m
\right).
\]

每个有效可训练宏只产生一个 policy 样本：

\[
L_{\mathrm{PG}}^{\mathrm{macro}}
=
\frac{1}{N_{\mathrm{macro}}}
\sum_m L_{\mathrm{PG},m}.
\]

## 2.3 新策略 selected log-probability

优先从 FP32 masked logits 使用 `log_softmax` 计算 selected log-probability，避免先 softmax 再 `log(prob)` 的下溢。

若需要给 `AgentOutput` 增加非参数性的 logits/log-prob 输出字段：

- 不得改变网络结构或参数量；
- legacy forward 的现有字段、logits、probs、argmax 必须不变；
- checkpoint state_dict 必须不新增模型参数；
- 固定 probe 输出必须保持一致。

不得用 `log(clamp(prob))` 静默掩盖 active selected probability 为 0 的错误。

## 2.4 安全 gather 与有效性校验

所有 gather 必须遵循阶段 1 相同的防护原则：

1. 先由 `history_mask & (player_ids >= 0) & macro_token_mask` 得到 active macro token。
2. inactive/chance/padding 的 `action_taken` 可能是 `-1/99/999`，必须在 gather 前替换为安全索引。
3. active token 的 action index 必须在范围内且对应合法动作；非法 active index 必须 fail-fast，不能 clamp。
4. active token 的 `behavior_action_prob` 必须有限且严格大于 0。
5. active selected new log-probability 必须有限。
6. active token 必须全部 `on_policy_valid=True`。
7. active token 的 behavior selected probability 必须与 rollout reference policy 的 selected probability在既有容差内一致；不一致 fail-fast。
8. padding 中正数垃圾 `player_id` 或大 action index 不得影响任何真实宏结果。

## 2.5 实现硬要求

1. 必须在 log-domain 汇总，禁止直接累乘 token probability。
2. 必须先汇总完整宏 log-ratio，再执行**一次** PPO clip。
3. 禁止逐 token clip 后求积、求和或平均。
4. loss 按宏动作平均，宏长不得隐式增加 policy 样本权重。
5. 宏长为 1 时，sequence log-ratio、ratio、clip 和 policy loss 必须与 token PPO 数值一致。
6. 优势、value target 与宏分段在 policy epochs 开始前，用 rollout 初始模型在 `torch.no_grad()` 下计算一次并 detach；不得在每个 epoch 重新估计 target。
7. mini-batch 必须按完整 trajectory/batch 样本切分；不得按时间切片截断宏动作。
8. 每个 mini-batch 可以重新索引或重建 segmentation，但结果必须与全 batch 后按 batch index gather 完全一致。
9. 阶段 1 的 `macro_old_logprob` 可作诊断或交叉检查，但生产 loss 必须重新执行 on-policy/reference/finite 契约校验，不能无条件信任缓存字段。
10. 默认对超出 FP32 `exp` 安全范围的有限 log-ratio **fail-fast 并报告样本/宏长/数值**。不得默认静默 clamp。若实现可选 safety clamp，必须默认关闭、进入 semantic config、记录阈值与次数，并有证明正常范围样本不变的测试。

建议新增独立模块：

```text
guandan_marvel/training/macro_ppo.py
```

生产 loss 不得放入阶段 1 的诊断 Oracle 文件。

---

# 三、Profile B：只替换 Sequence Ratio 的对照模式

当 profile 为 `macro_sequence_ablation_v1`：

## 3.1 Policy advantage

使用生产 legacy token GAE 在宏起点的值：

\[
\hat A_m^{B}
=
\hat A_{s_m}^{\mathrm{legacy\ token}}.
\]

必须用 `macro_start_index` 在有效宏上安全 gather。阶段 1 Legacy Oracle只用于断言该 gather 与已验收诊断结果一致，不得让诊断 Oracle 成为生产 loss 依赖。

## 3.2 Value loss保持 legacy

Profile B 的 value prediction、target、mask、Huber delta、coefficient 和所有 active token 的采样权重必须与 Profile A 完全一致：

```text
Profile B value loss = existing token-level value loss
```

禁止只在 macro start 计算 value loss。

## 3.3 Regularizer保持 legacy

Profile B 默认继续使用现有：

\[
+\,c_{\mathrm{reg}}\;\operatorname{mean}_{\mathrm{active\ token}}
D_{KL}(\pi_{new}\|U_{legal}).
\]

现有 schedule、系数、mask 和 token 平均方式保持不变。

## 3.4 因果隔离声明

A 与 B 的差异只能是：

```text
token selected-action ratio/clip/sample mean
→
完整宏动作 ratio/clip/sample mean
```

若任何其他训练语义变化，必须视为 blocker，不能称为 Sequence-only 消融。

---

# 四、Profile C：玩家宏决策 GAE

## 4.1 同玩家 successor、区间终点与持续时间

对玩家 \(p\) 的宏动作 \(m\)，起点为 \(s_m^p\)。定义：

- 若存在同一玩家下一宏起点：`next_m >= 0`，令
  \[
  u_m=t_{m+1}^p,
  \qquad c_m=1.
  \]
- 若不存在同玩家下一宏：令
  \[
  u_m=\operatorname{num\_steps}[b],
  \qquad c_m=0.
  \]

定义本宏决策区间的基础 token 持续时间：

\[
\Delta_m=u_m-s_m^p.
\]

要求：

1. 有 successor 时，\(\Delta_m\) 必须等于阶段 1 `macro_tau`；
2. 无 successor 时，阶段 1 `macro_tau` 可以保持 0，但生产 macro GAE 必须显式计算 `num_steps - start`，不能把 0 当作终止区间时长；
3. \(\Delta_m\ge1\)；
4. `chance`、environment token、其他玩家 token、当前宏内部 token均计入 \(\Delta_m\)；
5. padding 不得计入；
6. 无 successor 可能表示该玩家已出完而全局游戏仍继续，仍必须把该玩家的奖励窗口累计到该轨迹真实 `num_steps`，但不得 bootstrap。

## 4.2 区间奖励

只取行动玩家 \(p\) 的奖励通道：

\[
R_m^p
=
\sum_{j=0}^{\Delta_m-1}
\gamma^j r_{s_m^p+j}^{p}.
\]

不得读取统一 batch horizon 的 padding reward。

## 4.3 Semi-MDP TD residual

\[
\delta_m^p
=
R_m^p
+
gamma^{\Delta_m}c_m V^p(S_{m+1})
-
V^p(S_m).
\]

- 当前 value 必须取 `state_values[b, s_m, p]`；
- bootstrap 只能取同一玩家下一宏起点 `state_values[b, u_m, p]`；
- 禁止使用全局下一宏或其他玩家 value；
- `c_m=0` 时 bootstrap 严格为 0；
- truncated/incomplete 轨迹不能被当作正常 terminal 零 bootstrap。若没有显式、经过测试的 horizon bootstrap 观测，macro profile 必须拒绝 truncated/incomplete rollout。

## 4.4 宏决策 GAE

\[
A_m^p
=
\delta_m^p
+
gamma^{\Delta_m}\lambda_{\mathrm{macro}}c_m A_{m+1}^p.
\]

展开为：

\[
A_m^p
=
\sum_{\ell\ge0}
\lambda_{\mathrm{macro}}^{\ell}
\gamma^{\sum_{q=0}^{\ell-1}\Delta_{m+q}}
\delta_{m+\ell}^p,
\]

直到该玩家最后一个宏决策。

必须在代码、docstring 和报告中明确：

1. 环境 discount 时钟仍是基础 token，因此是 \(\gamma^{\Delta_m}\)。
2. \(\lambda_{macro}\) 每个同玩家宏决策衰减一次。
3. 这是有意改变估计器语义，不与旧 token GAE一般等价。
4. 精确复现旧 token trace 的 successor 系数是
   \[
   (\gamma\lambda_{token})^{\Delta_m}
   =\gamma^{\Delta_m}\lambda_{token}^{\Delta_m}.
   \]
5. 只有 `Delta=1`、`lambda=0` 或 `lambda=1` 时，两种递归系数退化一致。
6. “宏长为 1”不能推出“同玩家 Delta 为 1”；四人轮流时通常大于 1。
7. 不得修改阶段 1 Legacy Oracle 后再声称新算法与旧算法等价。

## 4.5 生产接口

建议新增：

```text
guandan_marvel/training/macro_gae.py
```

输出至少包含：

- valid macro mask；
- macro advantage；
- detached macro value target；
- macro duration `Delta`；
- continuation mask；
- reward sum；
- same-player next index；
- terminal/last-for-player诊断。

所有 target 在 `torch.no_grad()` 中生成并 detach，不允许 value target 反向传播到用于构造 target 的 current/next value。

---

# 五、Profile C 的 Macro Value Loss

Profile C 使用：

\[
V_{target,m}^p
=
\operatorname{stopgrad}\left(V^p(S_m)+A_m^p\right).
\]

要求：

1. prediction 取宏起点、行动玩家的 scalar value；
2. 每个有效宏计算一次 value loss；
3. prefix 内其他 token 不参与 Profile C 的主 value loss；
4. 使用现有 Huber 形式、delta 与 `value_loss_coef`，除非已有配置明确控制；本阶段不得顺手改损失形式或系数；
5. padding/chance/non-policy 宏不得进入；
6. target 必须 finite、detached；
7. Profile A/B 仍使用现有 token value loss，绝不受本节影响；
8. NTP 等既有辅助 loss 按现有配置原样保留，默认关闭；不得借本阶段修改其定义、权重或 mask。

宏 value explained variance：

\[
EV=1-\frac{\operatorname{Var}(y-V)}{\operatorname{Var}(y)}.
\]

只在有效宏上统计。若 target variance 小于阈值，必须记录 `degenerate_target_variance_count` 并报告 `NaN/undefined`，不能伪造为 0 或 1。

---

# 六、Advantage Normalization 与空样本处理

## 6.1 Legacy

Profile A 保持现有 token normalization 逐位一致，包括其历史行为。不得以修复单宏方差问题为由修改 legacy 数值路径。

## 6.2 Macro profiles

Profile B/C 只能在当前 policy mini-batch 的有效可训练宏集合上归一化：

\[
\hat A_m^{norm}
=
\frac{\hat A_m-\mu_{macro}}
{\sqrt{\operatorname{Var}_{population}(\hat A_m)}+\varepsilon}.
\]

要求：

1. 使用 population variance（`unbiased=False` 或等价实现），避免单宏时 `std()` 产生 NaN；
2. padding、chance、非 on-policy、incomplete 或违规宏不得进入均值方差；
3. 不得把同一宏 advantage 复制给每个 token 后做 token normalization；
4. 单有效宏或零方差时，normalized advantage 应为有限值 0，并记录计数；
5. 无有效宏的训练 mini-batch 必须 fail-fast，不能对空 tensor 求 mean；
6. 在 DDP 下，macro advantage 的 count/sum/sumsq 必须 all-reduce，所有 rank 使用同一个按有效宏样本加权的全局 mean/std；日志也必须使用同一统计，禁止平均各 rank 的均值；
7. 记录归一化前和归一化后的 mean/std/min/max。

---

# 七、策略正则：不得污染 Sequence-only 因果对照

现有训练目标包含 token 级 `KL(policy || legal-uniform)`。本阶段不得静默删除、改符号或把系数直接套到不同尺度的 entropy 上。

## 7.1 Legacy token regularizer

`legacy_token_kl_uniform_v1` 必须逐位复用现有计算：

\[
\bar K_{token}
=
\operatorname{mean}_{t\in active\ token}
D_{KL}(\pi_t\|U_t).
\]

- Profile A 必须使用；
- Profile B 必须默认使用，以确保只改变 sequence ratio；
- Profile C 的首轮主验证也必须使用，以隔离 macro GAE/value；
- `base_reg_coef` 及其 schedule 不变。

## 7.2 宏归一化 KL 与 entropy（独立消融）

仅当显式设置：

```text
policy_regularizer_mode = macro_normalized_kl_uniform_v1
```

才使用每宏等权的正则：

\[
\bar K_m
=
\frac1{L_m}
\sum_{t=s_m}^{e_m}
D_{KL}(\pi_t\|U_t),
\qquad
\bar K
=
\frac1{N_{macro}}
\sum_m \bar K_m.
\]

同时计算纯诊断归一化 entropy：

\[
\bar H_m
=
\frac1{L_m}
\sum_{t=s_m}^{e_m}
\begin{cases}
\dfrac{H(\pi_t)}{\log |\mathcal A_t|}, & |\mathcal A_t|>1,\\
0, & |\mathcal A_t|=1,
\end{cases}
\]

\[
H(\pi_t)=-\sum_{a\in\mathcal A_t}\pi_t(a)\log\pi_t(a).
\]

要求：

1. `|A_t|` 是 `avail_weights > 0` 的合法动作数；
2. 0 概率项按极限 `0 log 0 = 0` 处理，不得产生 NaN；
3. 每宏只产生一个 KL/entropy 样本；长宏不因 token 多而获得更大总权重；
4. 训练目标仍为 `+ reg_coef * macro_mean_KL`；不要把 `-entropy` 与 KL 同时重复加入，除非另有显式、默认 0、进入 semantic contract 的实验系数；
5. 宏 KL 模式默认关闭；不得把旧 `reg_coef=0.1` 无论量纲直接解释为 normalized entropy coefficient；
6. 无论训练使用哪种 regularizer，均双轨记录 legacy token KL、macro mean KL、raw entropy 和 normalized macro entropy，便于量级对照；
7. regularizer 切换必须作为独立消融，不能把它的结果归因给 sequence PPO 或 macro GAE。

---

# 八、数值精度、finite 契约与梯度

以下计算强制 FP32，即使阶段 3 以后启用 AMP：

- masked logits 与 `log_softmax`；
- selected token log-probability；
- macro log-probability；
- macro log-ratio 与 ratio；
- PPO clip；
- KL/entropy；
- reward discount sum；
- macro GAE；
- value target；
- advantage statistics；
- loss accumulation；
- finite checks。

要求：

1. 每个关键中间量有 shape、dtype、device、finite 检查；
2. 禁止 `.item()` 或 CPU round-trip 参与生产 loss 数学；诊断日志除外；
3. 禁止就地写入破坏 autograd 所需张量；
4. macro old log-prob、advantage、value target均 detach；new log-prob、policy KL/entropy、value prediction保留梯度；
5. 分别记录 decoder/state/action/logit/value 参数组梯度范数；
6. Profile B/C 中 policy 与 value 梯度必须非零（允许具体 batch 某组为 0，但必须报告并解释）；
7. 不得实施 GradScaler 或 autocast；本阶段只保证未来 AMP 下这些区域可固定为 FP32。

---

# 九、训练、配置、checkpoint 与恢复集成

## 9.1 接线位置

新语义必须正确进入：

- `TrainConfig/PPOConfig` 校验；
- 序列化/反序列化与旧配置默认迁移；
- semantic configuration/contract；
- resolved config hash；
- source fingerprint组件清单；
- checkpoint schema/identity；
- exact/compatible resume分类；
- Kaggle profile新训解析；
- Kaggle resume不得覆盖checkpoint训练语义；
- local/IPPO trainer；
- DDP worker；
- metrics、progress、durable checkpoint；
- recovery/resume。

## 9.2 Legacy兼容

Profile A 必须保持：

- 模型参数量不变；
- state_dict key集合不变；
- fixed probe logits/probs/argmax/value不变；
- 相同 rollout 的 GAE、PG loss、value loss、KL 与 total loss逐位一致；
- 相同 seed 的短程训练参数更新在现有容差内一致；
- 旧评估 checkpoint 可继续 strict evaluation；
- M09 policy-only warm start继续可用；
- 新字段缺失的旧配置规范化为 legacy defaults，不能误判成 macro。

不得承诺修改源码后可对旧 checkpoint 做字节级 exact resume。必须明确区分：

- strict evaluation load；
- policy-only warm start；
- compatible resume；
- exact resume。

Macro checkpoint/recovery必须保存并验证 profile 与 regularizer mode；任何训练语义不匹配必须 fail-closed。

## 9.3 DDP

- DDP 每 rank 必须使用相同 semantic config；
- sampler/recovery保持现有 rank-specific RNG 与分片契约；
- macro advantage 的 count/sum/sumsq 跨 rank all-reduce，所有 rank使用相同全局 normalization统计；
- macro PG/value/regularizer必须实现“全局每样本等权”，不能让每个 rank 等权。

若 rank \(k\) 有 \(n_k\) 个有效宏、全局 \(N=\sum_k n_k\)，而 DDP 默认会再对梯度除以 `world_size`，则本 rank 的可微 loss必须等价于：

\[
L_k=\frac{\mathrm{world\_size}}{N}\sum_{i\in k}\ell_i.
\]

这样 DDP 梯度平均后才等于全局宏均值 \(N^{-1}\sum_i\ell_i\)。value和macro regularizer同理。count/all-reduce张量不参与梯度。

- legacy Profile A 保持现有 DDP 数值路径，不得为统一实现而改变旧语义；
- metrics同样按 token/macro样本数加权，不得简单平均 rank mean；
- 全局 `N=0` 必须 fail-fast；单个 rank `n_k=0` 但全局 `N>0` 时必须安全产生与模型相连的零 loss，避免 DDP collective/unused-parameter分叉；
- 至少运行一次 CPU `gloo` 双 rank smoke，并用两个 rank 宏数量不同的合成反例验证梯度/指标等于单进程拼接 batch reference；若当前测试基础设施只支持其他后端，必须给出等价可执行验证，不得只声称兼容。

## 9.4 阶段 1 模块边界

可以调用阶段 1 segmentation 生成训练输入，但不得：

- 改写或删除 Legacy Oracle；
- 让诊断 Oracle进入生产 loss；
- 绕过 `incomplete_count`、`player_switch_violation_count`；
- 在违规轨迹上继续训练；
- 复制第二套不一致的宏边界实现。

Macro训练前必须 fail-fast：

```text
incomplete_count == 0
player_switch_violation_count == 0
all valid macro tokens are on-policy
```

---

# 十、必须新增的测试

测试不能与生产实现共享同一个 helper 作为“独立期望值”。手算/reference 侧应使用清晰 Python 标量或独立张量表达式。

## 10.1 Sequence PPO

至少覆盖：

1. 单 token 宏：sequence ratio/loss 与 token PPO 一致；
2. 两 token 人工概率：`r_macro = r1 * r2`；
3. 三 token：macro selected log-prob 等于各 token selected log-prob之和；
4. 构造“逐 token clip”和“宏 ratio clip”结果不同的正、负 advantage 反例；
5. 宏长 `1/2/8/10/16/32`；
6. 长宏只计一个 policy 样本；
7. 两个相同 advantage、不同宏长的宏具有相同 policy 样本权重；
8. padding/chance/environment token完全排除；
9. inactive `action_taken=-1/99/999` 不触发 gather越界；
10. active action越界或选中非法动作 fail-fast；
11. 宏内任一 token `on_policy_valid=False` fail-fast；
12. behavior probability `0/负数/NaN/Inf` fail-fast；
13. behavior/reference selected probability不一致 fail-fast；
14. selected new log-prob非有限 fail-fast；
15. 极端正 log-ratio触发明确错误或显式启用的可审计 safety机制，禁止静默溢出；
16. mini-batch segmentation等于 full-batch 后索引结果；
17. Profile A 旧 loss逐位回归。

## 10.2 Profile B 因果隔离

使用完全相同 rollout/model/config，仅切换 A/B profile，断言：

- GAE tensor相同；
- token value target相同；
- token value loss相同；
- legacy token KL相同；
- NTP/其他辅助项相同；
- 唯一不同是 policy ratio/clip/sample aggregation；
- 生产 token GAE 在 macro start gather 与阶段 1 Oracle 在容差内一致；
- mixed macro length、非零 value、`gamma<1` 均覆盖。

## 10.3 Macro GAE 独立手算

至少覆盖：

1. `Delta=[4,7,3]`、`gamma=0.99`、`lambda=0.95`；
2. 四人严格轮流、宏长1但 `Delta=4`：macro与legacy明确不等价；
3. `Delta=1` 退化一致；
4. `lambda=0`；
5. `lambda=1`；
6. `gamma=0`；
7. `gamma=0.5/0.97/1`；
8. 玩家提前出完、无下一宏，但终局 reward 在更晚 token：奖励必须进入最后宏窗口，bootstrap为0；
9. 接风导致同玩家 successor距离变化；
10. 只有终局 reward且 value=0：legacy按 token 距离衰减，macro按玩家宏决策数衰减；
11. multi-batch不同 `num_steps`；
12. padding reward/value/player/action为垃圾值，不影响结果；
13. next value只能来自同一玩家下一宏起点；
14. 无 successor 时使用 `num_steps-start`，不得误用 `macro_tau=0`；
15. truncated/incomplete轨迹被拒绝；
16. float64与标量 reference高精度对账，float32在合理容差内。

## 10.4 Value、Normalization、Regularizer

- Profile C value loss只在 macro start计一次；
- Profile A/B token value loss逐位不变；
- target明确 detached；
- padding宏不进入 value/normalization；
- 单宏、零方差、空宏分别稳定处理；
- population std不产生 NaN；
- 单合法动作 normalized entropy=0；
- `0 log 0` 正确处理；
- macro KL中不同宏长等权；
- legacy token KL逐位不变；
- regularizer mode切换只改变正则聚合，不改变 PG/GAE/value；
- value EV零方差时报告 undefined与计数。

## 10.5 配置、checkpoint、DDP

- 新字段 round-trip；
- 缺字段旧配置规范化为 legacy；
- 未定义 profile/组合和 VRPO macro请求 fail-fast；
- semantic hash覆盖所有新字段；
- exact resume拒绝 profile/regularizer mismatch；
- Kaggle resume不能覆盖 checkpoint语义；
- full→durable checkpoint→resume保留 profile、optimizer、sampler、RNG、iteration；
- CPU gloo双 rank至少1次迭代；
- 全局统计按 macro/token count加权，而不是 rank mean平均。

---

# 十一、本地真实训练与恢复验证

使用真实 C++ 环境、小规模配置，分别运行：

### A. Legacy

```text
training_semantics_profile=legacy_token_v1
policy_regularizer_mode=legacy_token_kl_uniform_v1
```

### B. Sequence-only

```text
training_semantics_profile=macro_sequence_ablation_v1
policy_regularizer_mode=legacy_token_kl_uniform_v1
```

### C. Final macro semantics

```text
training_semantics_profile=macro_decision_v1
policy_regularizer_mode=legacy_token_kl_uniform_v1
```

### D. Regularizer aggregation ablation

```text
training_semantics_profile=macro_decision_v1
policy_regularizer_mode=macro_normalized_kl_uniform_v1
```

要求：

- 同一模型初始化或同一 M09 policy-only warm start；
- 相同 seed、发牌控制、完成局数、batch/epoch/学习率/clip/reg配置；
- A/B/C 各至少2 iter；
- D 至少1 iter，只验证独立开关和数值健康；
- 不得启动长训。

记录至少包括：

- token count、macro count、macro length、Delta/tau分布；
- macro old/new log-prob、log-ratio、ratio；
- macro clip fraction及按宏长分桶；
- legacy token KL、macro KL、raw/normalized entropy；
- advantage raw/normalized mean/std/min/max；
- value loss与匹配采样单位的 EV；
- decoder/state/action/logit/value梯度范数；
- finite/clamp/zero-variance/empty计数；
- illegal/incomplete/truncated；
- 每 iter rollout/learner/total耗时。

Profile C 必须完成：

```text
full 2 iter
→ durable checkpoint/recovery
→ 新进程 resume 2 iter
```

并验证模型、optimizer、sampler、RNG、iteration、durable counters和训练语义恢复正确。仅在同一进程调用 load 不算完整 recovery 验证。

另运行：

```text
2-rank CPU gloo, Profile B 或 C, 至少1 iter
```

---

# 十二、有限因果 A/B（资源允许时）

不得直接开始正式长训。若资源不足，明确写 `NOT RUN`，不能写成通过。

## 12.1 Ratio A/B

```text
Control: Profile A
Treatment: Profile B
```

保持 value、GAE、regularizer 和其他超参完全一致，只比较 token ratio 与 sequence ratio。

## 12.2 GAE/value A/B

```text
Control: Profile B
Treatment: Profile C
```

这项比较是“macro GAE + 配套 macro-start value”对 legacy token GAE/value 的整体比较，不得谎称只隔离了 lambda时钟。

如果希望单独隔离 GAE 时钟，必须增加一个显式实验 profile：

```text
macro sequence PPO
+ player macro-decision GAE
+ legacy token value loss
+ legacy token regularizer
```

该 profile 只能作为实验消融，必须进入 semantic contract；不允许用未记录的临时布尔值实现。

## 12.3 Regularizer A/B

```text
Control: Profile C + legacy token KL
Treatment: Profile C + macro-normalized KL
```

不能把本项结果归因给 macro GAE。

所有执行的 A/B 均要求：

- 从同一初始 checkpoint 分别开始；
- 相同 seed、初始牌副、完成局数和除目标变量外的配置；
- 不能从 Control 中途切换成 Treatment；
- 固定独立评测牌副；
- 四座位轮换；
- 按 deal 聚类；
- 输出只能为 `positive/negative/inconclusive`；
- 小样本不得宣称最终棋力提升或 FableDan parity。

---

# 十三、阶段专项验收

必须全部满足：

1. 三个核心 profile 语义清晰、可配置、可恢复。
2. Profile A fixed input/loss/短程更新与旧实现一致。
3. Profile B 只改变 sequence ratio聚合；value和regularizer保持 legacy。
4. Sequence ratio公式、log-domain实现与独立手算一致。
5. Profile C macro GAE使用明确的 `Delta/continuation`，终止宏不误用 `tau=0`。
6. Macro GAE/value target与独立reference一致。
7. 不再出现“宏长1即可与旧GAE等价”的错误声明。
8. 宏违规、off-policy、active非法action、truncated轨迹 fail-fast。
9. 宏长大于8、multi-batch、hostile padding、chance正确。
10. 单宏、零方差、单合法动作无NaN/Inf。
11. Regularizer消融独立，不污染Sequence-only。
12. full→durable recovery→新进程resume通过。
13. 双 rank DDP smoke通过，统计加权正确。
14. 阶段1所有测试继续通过，baseline manifest哈希可复算。
15. 全量回归无新增失败；既有失败必须与开工前对账。
16. 模型结构、参数量、state_dict key和fixed probe不变。
17. 未提前实施阶段3以后内容。

---

# 十四、禁止事项

- 不得实现 actor fuse。
- 不得实现 AMP/autocast/GradScaler。
- 不得修改 reward 或 outcome语义。
- 不得启用完整 context schedule。
- 不得添加 belief、训练期 opponent pool或部署搜索。
- 不得删除 legacy token模式。
- 不得删除、放宽或改写阶段1反例来使测试通过。
- 不得通过只支持 `gamma=1` 规避一般公式测试。
- 不得把 `macro_tau=0` 当作最后宏的真实持续时间。
- 不得在 Profile B 改 value采样单位或regularizer。
- 不得把 normalized entropy系数与现有 KL系数视为同量纲直接迁移。
- 不得修改 FableDan 相关代码或使用其输出作为训练标签/数据。
- 不得执行正式长程训练。

---

# 十五、Implementer 最终报告格式

必须包含：

1. `IMPLEMENTATION_STATUS: COMPLETE / PARTIAL / BLOCKED`；
2. 修改文件及用途；
3. 开工前 git状态与阶段1门禁；
4. profile/regularizer配置矩阵与非法组合；
5. Sequence PPO公式到代码映射；
6. Macro GAE `Delta/continuation/reward/bootstrap` 到代码映射；
7. A/B/C value与regularizer采样单位；
8. Legacy逐位兼容证明；
9. 新增测试及每个反例捕获内容；
10. 实际运行命令和完整结果；
11. A/B/C/D smoke指标；
12. full→durable→新进程resume结果；
13. 双 rank DDP结果；
14. 有限A/B结果或明确 `NOT RUN`；
15. 参数量、state_dict、probe与checkpoint兼容性；
16. baseline manifest验证；
17. `git status --short`、`git diff --stat`、tracked diff bytes/SHA256；
18. 未实施的后续阶段内容；
19. 遗留风险。

Implementer无权自行批准进入阶段3。

---

# 十六、Reviewer 独立核验要求

实现完成后必须启动一个全新的只读 Reviewer。Reviewer不得复用 Implementer 的推理上下文，不得编辑文件，不得只重复运行实现者提供的测试。

Reviewer至少独立检查：

1. 两/三 token sequence ratio手算；
2. 正、负 advantage下“逐token clip”与“宏clip”分离反例；
3. 宏长10/16/32；
4. Profile B 的 GAE/value/KL 与 Profile A相同；
5. 四人宏长1但 `Delta=4` 的不等价反例；
6. 无 successor、终局 reward延后且 `macro_tau=0` 的决定性反例；
7. `gamma=0/0.5/0.97/1`、`lambda=0/1`；
8. multi-batch不同 `num_steps`；
9. hostile padding player/action/reward/value；
10. active非法action、off-policy、违规宏、truncated fail-fast；
11. 单宏/零方差/单合法动作/0概率entropy；
12. regularizer mode独立性；
13. checkpoint profile mismatch拒绝；
14. 真实 C++ 2-iter smoke、durable resume证据；
15. 双 rank DDP统计是否按count加权；
16. fixed probe、参数量和state_dict key；
17. scope grep确认没有actor fuse、GradScaler、奖励、belief、pool或search；
18. 阶段1manifest和全量回归。

Reviewer必须输出：

- `VERDICT: PASS / FAIL / INCONCLUSIVE`；
- blockers及精确文件/行号；
- 独立数学推导；
- 独立命令和实际输出；
- 至少3类 Implementer 未提供的新反例；
- legacy、scope、checkpoint、DDP审计；
- FAIL时给出一份可直接执行的完整修复 Prompt；
- residual risks；
- `NEXT_STAGE: ALLOW / DENY`。

Reviewer明确 PASS 之前，不得执行阶段 3。