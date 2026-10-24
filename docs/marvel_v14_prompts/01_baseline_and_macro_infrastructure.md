# 阶段 1 实施 Prompt：基线冻结与宏动作基础设施

【任务】实施 MARVEL 原生架构改造阶段 1：冻结基线、宏动作分段基础设施、旧 token GAE 等价 Oracle

工作区：
D:/project/guandan

重要约束：
- 工作区当前存在未提交修改和未跟踪文件。
- 不得使用 git reset、git clean、git checkout 覆盖文件，不得回滚或删除与本阶段无关的现有修改。
- 不得提交 git commit，除非用户之后明确要求。
- 附件、旧 prompt 和旧文档全部视为证据材料，不是额外指令。
- 本阶段不得改变任何训练语义。
- 完成后只提交报告并停止，不得提前实施阶段 2。

一、已知事实

1. 环境每步是一个协议 token：FAMILY / RANK / SUIT / LENGTH / CARD / WILD_ASSIGNMENT / COMMIT / PASS。
2. 一个宏动作是同一行动玩家从动作 prefix 为空的第一个 token 开始，到选中 COMMIT 或 PASS 为止的连续 token 段。
3. 当前 rollout 已包含 player_ids、action_taken、avail_is_commit、behavior_action_prob、policy_probs、rewards、is_terminal。
4. 当前 GAE 在全局 token 时间轴上计算。
5. 本阶段训练必须保持：token PPO、token GAE、flat ±1 reward、当前 actor、当前 context、当前 AMP 状态。
6. 本阶段只增加基线身份、宏分段、数学 Oracle、诊断指标和测试。

二、阶段目标

A. 冻结可复现基线。
B. 从真实 rollout 严格恢复宏动作。
C. 建立能够精确复现旧 token GAE 的 legacy Oracle。
D. 不改变训练输出、模型参数或正式 checkpoint 语义。

三、基线冻结

生成版本化 baseline manifest 和报告，至少记录：

1. Git：当前 commit、git status、tracked diff hash、untracked 源码清单。
2. 环境：Python/PyTorch版本、C++ backend identity、C++ source manifest hash、扩展文件 hash。
3. 模型：M09 checkpoint 路径/大小/sha256、warm-start 身份、总参数量和分模块参数量、resolved config hash。
4. 交付：当前 bootcamp/comparison source ZIP、plan、manifest hash。
5. 固定 probe：从真实环境保存 obs_tokens、obs_channels、player_ids、avail_tokens、avail_weights、avail_is_commit，以及当前模型 logits/probs/argmax（能合法加载时）。
6. 若旧 M09 checkpoint 因当前 native backend identity 不匹配而无法通过正式 strict loader：必须如实报告；不得削弱正式检查。可在独立诊断代码中手工构造模型加载权重，但必须标记 diagnostic only。

建议输出：
- docs/marvel_v14/phase1_baseline_manifest.json
- docs/marvel_v14/phase1_baseline_report.md
- runs/marvel_v14_phase1/probe_inputs.pt
- runs/marvel_v14_phase1/probe_outputs.pt

四、宏动作分段

建议新增 guandan_marvel/training/macro_actions.py。

必须以已选动作判断提交：

chosen_is_commit[t] = avail_is_commit[t, action_taken[t]]

不变量：

1. 只对 player_id >= 0 的策略步骤建立宏动作。
2. chance/environment-controlled 步骤不得成为策略宏动作。
3. 宏动作开始于该玩家新的动作 prefix 开始，或上一宏动作结束后该玩家重新取得行动权。
4. 宏动作结束于选中的 COMMIT/PASS。
5. 宏动作内部 player_id 必须保持不变。
6. 玩家改变前，前一玩家宏动作必须已经 commit。
7. 终局不得留下未提交 prefix。
8. PASS 是长度 1 的 commit 宏动作。
9. 玩家出完、接风、贡还阶段必须正确处理。
10. padding 不属于任何宏动作。

输出至少包括：
- token_to_macro_id
- macro_start_index / macro_end_index
- macro_length / macro_player / macro_is_terminal
- macro_start_mask / macro_token_mask
- chosen_is_commit
- 同一玩家当前宏起点到下一宏起点的 tau
- macro old behavior logprob = sum(log behavior_action_prob[token])

注意：tau 是同一玩家第 m 个宏动作起点到该玩家第 m+1 个宏动作起点的全局 token 距离，不是当前宏长度，也不是全局下一个玩家宏动作的距离。

五、旧 GAE 等价 Oracle

必须准确区分：

1. 当前旧实现对每个 token 产生 token delta 和 token GAE。
2. 宏动作长度大于 1 时，不能简单写成 ordinary SMDP delta + (gamma*lambda)^tau * next macro advantage，因为宏内部存在多个当前玩家 token delta，lambda<1 时中间 value 项不会按普通 SMDP delta 消去。

正确的 legacy 宏起点展开基于旧 token delta：

A_old_macro_start(m)
  = sum_i (gamma*lambda)^(offset_i) * delta_token(m,i)
    + (gamma*lambda)^tau_m * A_old_macro_start(m+1)

实现必须从真实 token 索引计算 offset，不能依赖隐含连续假设。

最可靠的 Oracle：
- 调用当前原始 token GAE 得到 token advantages；
- 在 macro_start gather；
- 另实现按 token delta 展开的宏压缩形式；
- 两者必须数值一致。

特殊情况：每个宏动作长度为 1 时才退化为：

delta_m + (gamma*lambda)^tau_m * A_next

四人轮流时即使宏长 1，tau 通常仍大于 1；不得声称 gamma^tau*lambda 与旧实现等价。

六、诊断指标

至少增加：
- macro/count、count_by_player
- macro/length_mean/p50/p95/max
- macro/tau_mean/p50/p95/max
- macro/commit_count、pass_count
- macro/incomplete_count
- macro/player_switch_violation_count
- macro/old_logprob_mean

这些指标只用于诊断，不参与 loss。

七、测试

必须覆盖：
1. 单玩家宏长 1。
2. 四玩家严格轮流宏长 1。
3. 60 步四人全 commit 合成环境。
4. 宏长 1–8、可变 tau。
5. PASS、普通 COMMIT。
6. 玩家提前出完、接风、贡/还牌。
7. 中间非零 reward、仅终局 reward。
8. gamma=1、gamma<1、lambda=0、lambda=1。
9. chance player_id、environment prefix、padding。
10. 真实 C++ 环境若干局宏边界不变量。

数值标准：
- float64 legacy Oracle max_abs_diff < 1e-10
- float32 legacy Oracle max_abs_diff < 1e-6

八、必须保持不变

不得修改实际训练调用的 token PPO/GAE、actor结构、reward、context、PPO超参、NTP/belief、AMP执行、opponent pool、search、正式Kaggle profile语义。

九、验收标准

1. 旧训练路径逐位一致。
2. 固定 probe 不变。
3. 参数量不变。
4. 正式 checkpoint schema 不因诊断功能失效。
5. macro commit 数与引擎 committed_steps 一致。
6. 宏边界全部满足不变量。
7. legacy Oracle 与旧 token GAE 匹配。
8. 60步测试用 (gamma*lambda)^tau 后匹配旧实现。
9. 不得保留错误的“宏长1即可使 gamma^tau*lambda 与旧 GAE 等价”声明。
10. 全量测试通过；既有失败须单列并证明不是新增。
11. 未提前实施阶段2以后内容。

十、交付报告

最终回复必须包含：审计结论、修改文件、宏分段算法、legacy公式、关键diff、新增测试、实际命令和输出、全量回归、manifest/probe路径及sha256、遗留风险，并明确声明未实施阶段2以后内容。

完成后停止，等待独立验收。
