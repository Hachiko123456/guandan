# M04 C++ Runtime 补修协议

- 日期：2026-10-09
- 原始实现：`75735031721a44304ffe4d7be7fc1532fadd0ba6`
- backend：`cpp`
- action/environment/encoding：保持 `GD-ACTION-0.2-MARVEL` / `GD-ENV-0.2-MARVEL` / `GD-ENCODING-0.2-MARVEL`
- serialization：保持 `guandan-marvel-cpp-runtime-v1`

## 1. 补修边界

本轮只修复两个 M04 接口/恢复缺陷：

1. C++ `EnvBatch.step()` 不能消费公共 `actions_from_np()` 返回的 `V2Action` 序列；
2. C++ `deserialize()` 接受损坏 RNG 文本和与 game state 不一致的 batch trajectory counter。

不改变 A01/A02 规则、IPPO/VRPO 数学目标、M02 sampler 失败禁用策略或 M03 checkpoint 契约。正式默认后端仍是 `python_reference`，M05 未开始。

## 2. 公共动作接口

C++ `EnvBatch.step(action_input)` 同时接受：

- NumPy integer rows，shape `[B,8]`；
- `actions_from_np(rows)` 返回的 `Sequence[V2Action]`。

两条路径统一生成完整 `ActionRow`，随后执行相同校验：

- row framing；
- phase；
- prefix stage；
- kind 与 V1 token payload；
- COMMIT flag；
- token vocabulary；
- decoded object 的 `token_id`、`phase_token`、`prefix_stage`、`kind`、`is_commit` 与 row 一致；
- action 数量等于 batch size；
- 完整 row 与当前 slot 的 live candidate 唯一匹配。

不能只根据 `token_id` 执行。raw rows 和 decoded actions 的 transition、event、channel、reward、terminal 和序列化结果必须相同。batch 继续使用 trial-copy；任一 slot 非法时所有 slot 均不提交。

## 3. 严格反序列化

RNG restore 使用 `std::mt19937_64` stream parser，并要求：

1. extraction 成功；
2. 完整状态存在，空文本/非法文本/截短文本失败；
3. extraction 后只允许空白；非空垃圾后缀失败。

Batch restore 还要求：

- `events`、`trajectory_steps`、`games` 长度一致；
- `0 <= trajectory_step <= max_horizon`；
- 每个 `trajectory_step == game.token_step`；
- 每个 game 的 `max_token_steps == batch.max_horizon`；
- `state.committed_step == history.size()`；
- `token_step == sum(canonical_tokens(history)) + live_prefix.size()`；
- level/previous-result 与 game config 一致；
- terminal finish counters 与 game counters 一致；
- `truncated` game 不可作为可用 `EnvBatch` snapshot；
- `auto_reset=true` 的内部 snapshot 不得保留 terminal game；
- live prefix 必须仍能扩展为合法动作。

所有检查完成且 `observe()` 成功后才返回实例。损坏输入抛 Python 可捕获异常，不返回部分恢复对象。

## 4. 兼容性

Wire 字段和正常 payload 内容没有变化，因此 serialization schema 不升级。有效旧 payload 继续可读并保持 byte-stable roundtrip；原实现曾接受但不符合协议的不合法 payload 现在明确拒绝。

`step(rows)` 保持兼容；新增 `step(actions_from_np(rows))` 是对既有公共 game interface 的修复，而不是新协议分支。

## 5. 验证要求

补修验收覆盖：

- single/multi-slot raw/decoded 等价；
- shape/dtype/count/stale/forged/multi-slot atomic rejection；
- 真实小型 Transformer Agent + 现有 MARVEL Sampler 的 on-policy first step；
- 两环境错峰终局、连续两次 collect 和 auto-reset；
- 空/非法/截短/尾随垃圾 RNG 拒绝；
- negative/over-horizon/mismatched counter 和 batch length 拒绝；
- 中途 prefix roundtrip 后继续真实终局；
- serialize/restore 后实际跨 auto-reset 的下一副发牌/RNG continuation；
- 强制重建、隔离 wheel、仓库外导入和缺扩展无 fallback。

这些仍是 M04 补修证据，不替代 M05 10k/100k 差分和 >=2x 固定工作负载门槛。
