# A04 编码布局 — GD-ENCODING-0.1

`encode_observation` 是当前 A03 `StepwiseActionState` 的确定性策略投影。它不会序列化 `RoundState`、对手手牌、交换托管区，或尚未完成的进贡/还贡选择。

## 观测 token 流

`observation_tokens` 是一个 `int32[4096]` 的零填充流。已使用长度保存在 `state_channels[24]` 中。头部为：

```text
BOS, PHASE, LEVEL_RANK, ACTIVE_SEAT, COMMIT
```

头部之后有五个带边界的区段。每个区段以 `BOS,LENGTH(section_id)` 开始，以 `COMMIT` 结束：

| 区段 | 内容 |
|---:|---|
| 0 | 当前行动玩家的实体牌 token，按 card ID 升序排列 |
| 1 | 当前行动玩家尚未完成的 A03 前缀 token |
| 2 | 当前获胜出牌声明的 payload；没有时为空 |
| 3 | 公开出牌/过牌历史；交换完成后还包含已公开的进贡/还贡历史 |
| 4 | 已确定的牌转移记录；没有时为空 |

历史和转移记录也各自使用 `BOS ... COMMIT` 的记录边界。在一条出牌记录中，逢人牌 token 紧邻对应实体牌 token；红桃级牌自然使用时还会额外携带 `LENGTH(1)`。所有输出 token 都是 V1 词表中的非 PAD token。容量溢出时抛出 `ProtocolError`，不得截断尾部。

## 状态通道

`state_channels` 是分离存储的 `float32[256]`：

- `0..7`：当前玩家、队伍、阶段 token、级别、token step、committed step、领出玩家、当前轮获胜玩家（`-1` 编码为 `-1.0`）；
- `8..11`：按座位排列的可见剩余牌数；
- `12..15`：按座位排列的完成名次（`0` 表示尚未排名）；
- `16..19`：实际已出完座位标记；
- `20..28`：过牌次数、前缀长度、是否可提交、历史长度、已用 token 长度、自身手牌数量、当前轮标记、交换是否公开、抗贡标记；
- `29..36`：上一副牌是否存在、获胜队伍、结果类别、名次、取消进贡标记；
- `37..47`：合法行数、最大 token steps、已排名座位数、已出牌数量、公开出牌/过牌数量、进贡数量、还贡数量、当前获胜牌张数、前缀牌张数、前缀牌型、前缀逢人牌数量；
- `48..51`：固定队伍映射；
- `52..55`、`56..59`、`60..63`：当前玩家/领出玩家/当前获胜玩家的 one-hot 座位；
- `64..67`：阶段 one-hot；
- `68..80`：印刷级别 one-hot；
- `81..96`：当前玩家手牌的点数计数，`96..100` 为花色计数；
- `101..208`：当前玩家实体牌 mask；
- `209..240`：未完成前缀副本；
- `241..250`：前缀牌型 one-hot/计数辅助通道；
- `251..255`：当前获胜牌型/点数/张数以及 commit/public 哨兵。

确切的实现下标通过 `REMAINING_COUNTS_SLICE`、`FINISHING_RANKS_SLICE`、`SELF_CARD_IDS_SLICE` 和 `PREFIX_TOKENS_SLICE` 导出。所有数组都拥有自己的存储；策略视图中的 `private_hand_card_ids` 始终为 `None`。特权 `full_state()`/checkpoint API 与策略视图分离。

## 合法 token 数组

`legal_next_tokens`、`legal_next_mask`、`legal_next_is_commit`、`legal_next_kinds` 和 `legal_next_values` 都是长度为 256 的分离数组。数组行按照确定性顺序从完整的 A03 合法 next-token 集合复制而来。无效行全部为零且 `mask=False`。编码器遇到容量溢出时抛出错误，不得通过只保留 top-N 行来规避。
