# M02 Guandan MARVEL V2 Python Adapter Protocol

- 实现日期：2026-10-08
- backend：`python_reference`
- action：`GD-ACTION-0.2-MARVEL`
- environment：`GD-ENV-0.2-MARVEL`
- encoding：`GD-ENCODING-0.2-MARVEL`
- serialization：`guandan-marvel-python-adapter-v1`

## 1. 规则来源与边界

V2 adapter 包装已验收的 V1 `GuandanEnv` / `StepwiseActionState`。合法动作枚举、COMMIT、贡还牌、终局、奖励和状态转移仍由 A01-A04 Python 规则实现；V2 不复制或修改规则语义。默认 legacy trainer/evaluator 不切换到此接口。

`python_reference` 是显式 reference backend，不是高性能后端。传入其他 backend 名称会直接失败，不进行 fallback。

## 2. Game module

Canonical module：`guandan_marvel.game`。

| 常量 | 值 | 含义 |
|---|---:|---|
| `num_players` | 4 | 四个固定座位 |
| `num_tokens` | 320 | V1 0..255 加 V2 framing vocabulary |
| `num_channels` | 512 | 每座位 actor information channels |
| `pad_token` | 0 | padding |
| `max_game_horizon` | 4096 | 沿用 V1 token hard limit |
| `max_available_actions` | 256 | 每个不同 V1 legal-next token 至多一行 |
| `max_observation_tokens` | 48 | 单步增量事件上界 |
| `max_action_tokens` | 8 | 结构化候选 row 固定长度 |

模块暴露 `GAME_SPEC`、`EnvBatch`、`actions_from_np()`、`ACTION_SCHEMA`、`ENV_SCHEMA` 和 `ENCODING_SCHEMA`。

## 3. 结构化候选动作

每行严格为：

`[ROW_BEGIN, PHASE, PREFIX_STAGE, KIND, PAYLOAD_MARKER, V1_TOKEN_ID, IS_COMMIT, ROW_END]`

- `PHASE`：LEAD/FOLLOW/TRIBUTE/RETURN；
- `PREFIX_STAGE`：当前私有 canonical prefix 长度 0..32；
- `KIND`：PASS、COMMIT、FAMILY、RANK、SUIT、LENGTH、CARD 或 WILD_ASSIGNMENT；
- `V1_TOKEN_ID`：唯一执行载荷；
- `IS_COMMIT`：与 PASS/COMMIT payload 一致。

`actions_from_np()` 严格验证 row framing、kind、payload 和 commit flag，并返回携带完整 row 的 `V2Action`。`EnvBatch.step()` 还要求整行与当前状态的候选表恰好匹配一次，因此旧 row、伪造 metadata 或只碰巧合法的 payload 都不能执行。

容量不做 top-N。V1 legal-next 集合按 token ID 去重，词表共 256 个 ID，因此 256 是协议硬上界；实际 M02 扫描 15 局、4447 steps、23121 candidate rows，观察最大候选数 27。

## 4. 增量事件与 state channels

每步 observation 为 `tokens[B,4,48]` 和 `channels[B,4,512]`。

- reset 只发送 reset/phase/level/actor event；
- 私有 prefix token 只发送给当前 actor，其他座位该步 token row 全 PAD；
- play/PASS commit 发送一次公开 canonical action event；
- 未完成贡还牌 commit 只发送给选择者；
- 全部贡还牌完成后一次公开完整 resolved transfer records；
- 不重复历史 action list，也不重复 4096-token V1 observation。

48-token 上界来自：最长 V1 canonical action 32 加 framing 不超过 37；双贡双还最多四条、每条 5 token，加 framing 不超过 25。overflow 抛 `AdapterCapacityError`。

channels 仅含当前公开状态和 observer 自身私有状态：

- observer 自己的物理手牌；
- 当前 winning cards/declaration；
- 自己的 prefix；
- 自己已提交但未公开的贡还牌选择；
- 公开剩余张数、名次、阶段、级牌、leader/winner/pass 等；
- 已完成交换后的公开状态。

不含其他座位 ownership。RETURN 阶段已写入 A02 donor hand、但尚未公开的 incoming return card 会从 donor actor channels 中隐藏，直到交换完成。

## 5. Critic 隔离

`EnvStep` 不含 ownership/full state。训练期 privileged labels 只能通过独立的 `EnvBatch.critic_state()` 获取：

- `ownership[B,4,108]`；
- `escrow[B,108]`；
- `played[B,108]`。

该 API 与 actor reset/step/observe wire 分离，并执行 108 张物理牌守恒检查。

## 6. 轨迹与 sampler

`guandan_marvel.env_batch.EnvStepBatch` 把 NumPy wire 转为 TensorDict；`guandan_marvel.runtime.sampler.Sampler` 将每个 token step 写入 `TrajectoryBuffer`：

- `num_steps`、`player_ids`；
- `obs_tokens`、`obs_channels`；
- `avail_tokens`、`avail_weights`、`avail_is_commit`、`avail_probs`；
- `action_taken`、`rewards`、`is_terminal`。

只有真实 A02 terminal 才设置 `is_terminal` 和非零终局奖励。horizon overflow 抛 `TrajectoryHorizonError`，不伪造胜负。sampler 完整轨迹测试收集 296-step 真终局并存入/采样 MARVEL buffer。

## 7. 序列化

V2 batch 序列化保存：协议版本、backend、容量、每 slot V1 environment payload、增量 event 和 trajectory counter。反序列化会验证版本、shape、容量及当前 live frame；后续 row、state transition、event 和序列化 bytes 必须一致。

## 8. 验证证据

- `evidence/M02_guandan_adapter_v2/verification.json`
- `evidence/M02_guandan_adapter_v2/test_results.json`
- `tests/marvel_adapter/test_adapter.py`

这些是本地功能/正确性证据，不是模型强度、C++ 性能或 Kaggle 远端验收证据。
