# A04 编码布局（GD-ENCODING-0.1）

本文档是 `guandan.encoding.encode_observation` 的规范布局。编码器是**纯投影**：不序列化或复制 `RoundState`，不读取对手/队友手牌，不把未完成交换的选择、escrow、转移配对或隐藏历史计数写入策略观测。`private_hand_card_ids` 在策略 `Observation` 中恒为 `None`。

## 1. 固定输出

| 字段 | 形状 | dtype | 说明 |
|---|---:|---|---|
| `observation_tokens` | `(4096,)` | `int32` | 非零部分是 token wire；尾部为 `PAD=0` |
| `state_channels` | `(256,)` | `float32` | 下面第 4 节的固定通道 |
| `legal_next_tokens` | `(256,)` | `int32` | A03 合法 next-token 行 |
| `legal_next_mask` | `(256,)` | `bool` | 有效行标记 |
| `legal_next_is_commit` | `(256,)` | `bool` | A03 行是否提交 |
| `legal_next_kinds` | `(256,)` | `int32` | A03 kind 编码 |
| `legal_next_values` | `(256,)` | `int32` | A03 value 编码 |

使用长度为 `state_channels[24]`。所有数组都是新分配的 detached 数组；任何容量溢出都抛出 `ProtocolError`，禁止尾部截断或 top-N 筛选。终止或 truncated protocol 由 `encode_observation` 抛出 `EpisodeTerminatedError`。

## 2. Token 词表与总规则

只使用 A03 已存在的 V1 词表，不使用保留区 `49..63`、`224..255`，也不把 `PAD=0` 放入已使用区：

- `BOS=1`；`LEAD_PLAY=2`、`FOLLOW_PLAY=3`、`TRIBUTE=4`、`RETURN=5`；`PASS=6`；`COMMIT=7`。
- `FAMILY=8..17`；`RANK=18..32`；`LENGTH=38..48`；`CARD=64..171`；`WILD_ASSIGNMENT=172..223`。
- `TokenCodec.length(seat)` 在 wire 中作为座位字段；`TokenCodec.length(section_id)` 作为区段字段；这两个位置由语法上下文区分。

`COMMIT` 只作为头部、区段或记录的边界。一个记录/区段边界只出现一个 `COMMIT`；显式牌型负载不会复制 A03 `canonical_tokens` 的 START、SUIT、LENGTH 或内部 COMMIT。

## 3. `observation_tokens` 完整 wire

### 3.1 头部

```text
BOS, PHASE, RANK(level_rank), LENGTH(active_seat), COMMIT
```

其中 `PHASE` 是当前观察阶段：

- `LEAD_PLAY`：`current_winning is None`；
- `FOLLOW_PLAY`：存在当前获胜牌；
- `TRIBUTE`、`RETURN`：交换尚未完成。

### 3.2 五个区段

每个区段严格为：

```text
BOS, LENGTH(section_id), PAYLOAD, COMMIT
```

`section_id=0..4`，对应 `LENGTH(0)..LENGTH(4)`。

| 区段 | PAYLOAD | 精确内容 |
|---:|---|---|
| 0 | 手牌 | 当前 active seat 的实体 `CARD(card_id)`，按 card ID 升序 |
| 1 | 前缀 | 当前 active seat 的 A03 未完成 prefix 原 token 序列 |
| 2 | 当前获胜牌 | 无当前获胜牌时为空；否则为 `LENGTH(winner_seat)` 加显式出牌负载 |
| 3 | 公开历史 | 每条记录见 3.3；未完成交换时过滤所有进贡/还贡记录 |
| 4 | 已解析转移 | 每条记录见 3.4；交换未完成时为空 |

### 3.3 历史记录与显式出牌负载

区段 3 的每条记录严格为：

```text
BOS, RECORD_PHASE, LENGTH(actor_seat), BODY, COMMIT
```

`RECORD_PHASE` 为：普通出牌使用 `LEAD_PLAY` 作为历史出牌记录标签，过牌使用 `FOLLOW_PLAY`，已解析进贡/还贡使用 `TRIBUTE`/`RETURN`。当前 observation 的真实阶段仍由头部 PHASE 和 `Observation.phase` 表示。

普通出牌 `BODY` 是完整且无歧义的：

```text
FAMILY, RANK(comparison_rank),
    CARD(card_id_0), [WILD(target_rank, target_suit)], [LENGTH(1)],
    CARD(card_id_1), [WILD(target_rank, target_suit)], [LENGTH(1)],
    ...
```

规则：

1. `FAMILY` 是十类之一；
2. `RANK` 始终是 declaration 的 `comparison_rank`，不是顺子窗口起点；
3. CARD 按实体 `card_id` 升序；
4. 逢人牌替代时，目标 `WILD_ASSIGNMENT` 紧跟其实体 CARD；
5. 逢人牌自然使用时，`LENGTH(1)` 紧跟其实体 CARD（自然使用不另发替代目标）；
6. 同一 CARD 不重复；一个 `WILD` 或自然 `LENGTH(1)` 只属于紧邻的前一个 CARD；
7. `FULL_HOUSE` 的另一组牌点数由实体牌与两个逐牌 wild 目标解析，不另设第二个 rank token；
8. `STRAIGHT`、`PAIR_SEQUENCE`、`TRIPLE_SEQUENCE` 的窗口由实体牌、wild 目标和 comparison rank 解析，不另设起点或长度 token；
9. `STRAIGHT_FLUSH` 的花色由实体自然牌和 wild 目标解析，不另设 suit token；
10. BODY 内没有 COMMIT，记录末尾的 COMMIT 是唯一记录终止符。

过牌 `BODY` 只有：

```text
PASS
```

### 3.4 已解析转移记录

区段 4 每条记录严格为：

```text
BOS, TRIBUTE|RETURN, LENGTH(source_seat), LENGTH(target_seat), CARD(card_id), COMMIT
```

只有整个交换结束后（`Phase.PLAY`；终局状态由环境另行处理）才允许输出这些身份。`TRIBUTE`/`RETURN` 阶段绝不输出未完成选择、escrow card、transfer pairing、transfer count 或隐藏 exchange history。

## 4. `state_channels[0:256]` 精确索引

除 one-hot/掩码外，通道存储的是可由 `float32` 精确表达的整数值。

| slice/index | 含义 |
|---|---|
| `0` | active player seat |
| `1` | active player team |
| `2` | PHASE token（2..5） |
| `3` | level rank 的整数值 |
| `4` | `token_step` |
| `5` | `committed_step` |
| `6` | leader seat |
| `7` | current winning seat；无则 `-1` |
| `8:12` | 按绝对座位 0..3 的可见剩余牌数（稳定上下文索引） |
| `12:16` | 按座位的 finishing rank；未完成为 0 |
| `16:20` | 按座位的 `emptied_seats` 二值标记 |
| `20` | consecutive passes |
| `21` | prefix 长度 |
| `22` | 当前 legal rows 中是否含 COMMIT |
| `23` | 可见历史记录数 |
| `24` | 已使用 observation token 数 |
| `25` | active seat 手牌数 |
| `26` | 是否存在 current winning |
| `27` | exchange identities 是否已公开 |
| `28` | anti-tribute 标记 |
| `29` | previous result 是否存在 |
| `30` | previous winner team；无 previous result 为 `-1` |
| `31` | previous outcome：0/DOUBLE_DOWN=1/HEAD_THIRD=2/HEAD_LAST=3 |
| `32:36` | previous ranking[4] |
| `36` | previous tribute_cancelled |
| `37` | 有效 legal rows 数 |
| `38` | `max_token_steps` |
| `39` | 已排名 seat 数 |
| `40` | 可见出牌 card 数 |
| `41` | 可见出牌/过牌记录数 |
| `42` | 可见进贡记录数 |
| `43` | 可见还贡记录数 |
| `44` | current winning card 数 |
| `45` | 当前 prefix 中 CARD token 数 |
| `46` | 当前 prefix 中 FAMILY 编号（1..10；无则 0） |
| `47` | 当前 prefix 中 WILD token 数 |
| `48:52` | 固定 `TEAM_OF=(0,1,0,1)` |
| `52:56` | active seat one-hot |
| `56:60` | leader seat one-hot |
| `60:64` | current winner one-hot；无则全 0 |
| `64:68` | PHASE one-hot，顺序 LEAD/FOLLOW/TRIBUTE/RETURN |
| `68:81` | level printed-rank one-hot，按 `PRINTED_RANKS` 顺序 |
| `81:96` | active hand 的 printed-rank counts；这是 15 个位置，右端不含 96 |
| `96:101` | active hand 的 suit counts，含 JOKER suit |
| `101:209` | active hand physical-card mask，card ID 0..107 |
| `209:241` | prefix token 副本，最多 32 个位置 |
| `241:251` | prefix family one-hot/辅助位置，family 1..10 |
| `251` | current winning family 编号；无则 0 |
| `252` | current winning comparison rank；无则 0 |
| `253` | current winning card 数；无则 0 |
| `254` | 当前 legal rows 是否含 COMMIT（与 22 相同的稳定哨兵） |
| `255` | policy encoding 版本存在哨兵，恒为 1 |

## 5. 合法数组与信息边界

五个 legal 数组直接使用 A03 `legal_token_arrays()` 的完整有效行，并复制到 `ObservationSpec.max_legal_next_tokens` 的 detached 数组中；无效行是 `mask=False` 且其余元数据为零。编码器不依赖 caller 提供的 metadata 来验证 action。

策略观测只包含：当前玩家手牌、当前玩家未完成 prefix、公开阶段/计数/轮面、公开出牌/过牌历史，以及完整交换结束后的身份。对手和队友手牌的任意重排（保持公开计数和当前玩家手牌不变）不得改变观测；未完成交换中其他 donor 的牌选择也不得改变观测。
