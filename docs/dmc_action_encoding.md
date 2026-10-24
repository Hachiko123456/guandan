# DMC 动作特征布局（GD-DMC-ACTION-1）

本文档是 `guandan_marvel/dmc/candidates.py:encode_action_features` 的规范布局。版本字符串为 `ACTION_FEATURE_VERSION = "GD-DMC-ACTION-1"`（检查点 schema `dmc-v1`）。编码器是**纯函数**：只读取 `CandidateAction` 与 `context`，不读取隐藏状态、不修改 `RoundState`、不消耗 RNG；同一输入两次调用必须逐位相等。

动作粒度是**完整候选动作**（一次出牌/过牌/进贡/还贡为一个动作），不是 token 级动作。`CandidateAction.tokens` 保存 A03 的唯一 canonical token 序列，提交时仍走 `StepwiseActionState.step`，本布局不重新定义规则。

## 1. 固定输出

| 字段 | 形状 | dtype | 说明 |
|---|---:|---|---|
| 返回值 | `(64,)` | `float32` | 每维定义见第 4 节；全部为有限值 |

## 2. 输入契约

`CandidateAction`（frozen，字段固定）：

| 字段 | 类型 | 语义 |
|---|---|---|
| `tokens` | `tuple[int, ...]` | A03 `canonical_tokens(action)`，非空 |
| `family` | `int` | `0` = 非牌型（PASS/进贡/还贡）；`1..10` = 第 3 节 family 顺序 |
| `comparison_rank` | `int` | declaration 的 `comparison_rank`（`Rank` 整数值）；非牌型为 `0` |
| `card_ids` | `tuple[int, ...]` | 实体牌 ID，升序；PASS 为空 |
| `wild_assignments` | `tuple[(card_id, represented_rank, represented_suit), ...]` | declaration 中每张百搭的表示目标，按 `card_id` 升序；包含自然使用的百搭 |
| `is_pass` | `bool` | 是否 PASS；进贡/还贡为 `False` 且 `family=0` |

`context` 必须提供：

- `level_rank`：本局级牌（非 Joker）；可为 `Rank` 或可转换的整数；
- `current_winning`：当前轮面的 `Combination`，无轮面为 `None`。

支持传入带同名属性的对象（如 `RoundState`）或带同名键的 `Mapping`。级别非法或缺少 `level_rank` 时抛出 `TypeError`/`ValueError`，不做静默默认。

## 3. 顺序表

**Family 顺序**（与 A03 token codec 的 `FAMILY_TOKEN_BASE` 顺序一致）：

| family | 含义 |
|---:|---|
| 0 | 非牌型（PASS / 进贡 / 还贡） |
| 1 | `single` |
| 2 | `pair` |
| 3 | `triple` |
| 4 | `full_house` |
| 5 | `straight` |
| 6 | `pair_sequence` |
| 7 | `triple_sequence` |
| 8 | `rank_bomb` |
| 9 | `straight_flush` |
| 10 | `four_kings` |

**15 格点数直方图顺序**（`RANK_HISTOGRAM_ORDER`，与 A03 `RANK_ORDER` 相同）：`3,4,5,6,7,8,9,10,J,Q,K,A,2,SMALL_JOKER,BIG_JOKER`。

**5 格花色顺序**：`CLUBS=0, DIAMONDS=1, HEARTS=2, SPADES=3, JOKER=4`。

**百搭定义**：`Card.is_level_wild(level_rank)`，即红桃且印刷点数等于 `level_rank` 的实体牌（两副牌最多 2 张）。自然使用与替代使用都计入百搭张数。

**claim rank**：若实体牌出现在 `wild_assignments` 中，取其 `represented_rank`；否则取实体牌印刷点数。该定义用于 12..27 的“声称点数”直方图，与 32..46 的“实体印刷点数”直方图刻意分开。

## 4. `features[0:64]` 精确索引

所有除法结果以 `float32` 计算并存储；计数先按整数累加再统一相除。

| 索引 | 含义 | 计算 |
|---:|---|---|
| `0` | is_pass | PASS 为 `1.0`，否则 `0.0` |
| `1..11` | family one-hot | `features[1 + family] = 1.0`，其余为 `0.0`；共 11 位（family 0..10）。进贡/还贡与 PASS 共用 slot 0，由索引 0 区分 |
| `12..26` | claim/comparison rank 计数直方图（15 格） | 每张实体牌的 claim rank 落桶计数 `÷ 4.0` |
| `27` | 直方图块内保留位 | 恒为 `0.0`（锁定表把 `12..27` 整块留给该直方图；为同时满足 15 格与 64 维总长，末位留空） |
| `28` | 出牌张数 | `len(card_ids) / 27.0` |
| `29` | 使用百搭张数 | 实体百搭张数 `÷ 2.0` |
| `30` | comparison_rank 标量 | `comparison_rank / 15.0`（非牌型为 0） |
| `31` | 是否压过当前轮面 | 存在 `current_winning` 且 `1 <= family <= 10` 时为 `1.0`，否则 `0.0`（PASS/进贡/还贡、无轮面均为 0） |
| `32..46` | 实体牌 printed-rank 直方图（15 格） | 每张实体牌的印刷点数落桶计数 `÷ 4.0`（百搭按实体点数计数） |
| `47..51` | 实体牌花色计数（5 格） | 每张实体牌的花色落桶计数 `÷ 27.0`（Joker 使用 `JOKER=4`） |
| `52` | 是否含红桃级牌（百搭） | 含至少一张百搭为 `0.5`，否则 `0.0`（按提示的 `÷2` 归一化） |
| `53..63` | 版本对齐保留位 | 恒为 `0.0`；后续版本若启用必须提升版本字符串 |

### 4.1 索引 31 的不变量

`enumerate_candidates` 只返回 `enumerate_legal_actions` 的合法动作：存在 `current_winning` 时，非过牌候选必然严格压过当前轮面（普通牌型同族同长度且更大，或炸弹按炸弹序更高）。因此对**枚举产生的候选**，索引 31 等价于“有轮面且更高”，编码器按 `current_winning is not None and family != 0` 置位，不重复调用规则比较。手工构造的非合法 `CandidateAction` 不在本契约内；如需校验，应在提交路径由 A03 拒绝。

### 4.2 直方图求和恒等式（可用于自检）

- `features[12:27].sum() == len(card_ids) / 4`，且 `features[27] == 0`
- `features[32:47].sum() == len(card_ids) / 4`
- `features[47:52].sum() == len(card_ids) / 27`
- `features[1:12].sum() == 1.0`

## 5. 确定性与可复现性

- 枚举：`enumerate_candidates` 按 `tokens` 升序稳定排序；A03 已对 token 序列去重并对歧义报错，因此同一局面两次枚举逐位相等。
- 编码：只依赖候选字段与 `level_rank`/`current_winning`；不读取 `active_seat`、手牌或隐藏信息。
- 不做任何随机采样；百搭表示取 declaration 的确定目标（含自然使用），与 canonical token 序列一一对应。
- 版本字符串进入 `DMCRuntimeConfig.action_feature_version`；加载不匹配版本的检查点必须拒绝，不得静默解释。
