# A04 观测 API 说明 — GD-ENV-0.1 / GD-ENCODING-0.1

A04 在不改变已验收 A02/A03 规则和动作代码的前提下，对其进行封装。
`guandan.environment` 重新导出 `GameConfig`、`PreviousHandResult`、`ObservationSpec`、
`Observation`、`CommittedAction`、`StepResult`、`BatchObservation`、`BatchStepResult`、
`GuandanEnv` 和 `GuandanEnvBatch`。类型定义位于 `environment_types`，批量环境位于
`env_batch`。面向策略的提交回执不包含嵌入式原子动作。

- 默认维度保持为：4096 个观测 token、256 个 `float32` 通道、256 行填充后的合法 token、32 个动作 token，以及大小为 256 的词表/PAD 为 0。
- 构造函数不会自动发牌。显式 `reset()` 返回第一个策略观测。`reset()` 之前以及终局/截断之后，`observe()` 返回 `None`；reset 之前执行 step 会失败，终局/截断之后执行 step 会抛出 `EpisodeTerminatedError`。
- 显式 reset seed 会复现同一副牌。之后不带 seed 的 reset 会从该槽位自己的 NumPy PCG64 生成器取样。相同配置和 reset 历史会复现相同随机流；clone/serialize 会保留随机状态，但不共享可变状态。
- `reset(initial_state=...)` 接受新的 A03 快照或新的 A04 快照。计数器必须为零，前缀必须为空，牌局必须未终局且未截断。已有的进行中或已结束牌局必须使用 `deserialize()` 恢复，不能使用 `reset()`；这样可以区分“新建牌局”和“恢复牌局”，不会删除已出牌历史/牌张，也不会重置正在运行的牌局计数器。
- 每次 reset 和 step 都通过编码层执行事务操作。观测溢出时抛出 `ProtocolError`，既不发布游戏状态变更，也不推进随机数状态。
- 终局结果的观测为 `None`；`result.info` 包含 `ranking`、`winner_team`、`outcome_class`、`team_reward`、`finish_token_step` 和 `finish_committed_step`。前缀步骤和截断步骤的奖励为零，只有终局提交会支付 +/-1。
- 在贡还交换尚未完成时，动作回执会被脱敏。最后一次还贡完成后，`result.info.resolved_transfers` 才会发布所有交换的实体牌身份。`info` 中不得包含调试状态、隐藏手牌或候选动作列表。
- 完整信息的 `serialize()`/`full_state()` 是显式的检查点/调试 API，绝不是策略观测开关。返回的字典和数组必须是分离副本。
- 每个批槽位的 A03 路径记忆化使用完整牌局快照作为失效键。它不改变动作/前缀语义；不得引入全局缓存或共享可变游戏状态。缓存条目不会被序列化，也不会对策略可见。
- 批量行的首维为 B；active/self ID 为 `int32`，计数器为 `int64`，phase 为 Unicode，合法掩码为 `bool`，奖励为 `float32[B,4]`。V1 不存在 P 维观测者轴，也不存在扁平的完整动作张量。本说明替换 `observation_protocol.md` 中的临时适配器草图，但不改变已接受的标量 token 契约。
- 非活动槽位只接受 PAD=0，保持状态不变，并返回零奖励、`present=false` 以及全零策略数组。`reset_at` 只重置一个槽位。`auto_reset` 默认关闭；启用时，新结束牌局的 flags/reward/counters/info 会保留，而批量观测属于新牌局；`info.auto_reset` 和 `info.final_info` 明确标识这一边界。

token/通道在线布局见 `docs/encoding_layout.md`。
