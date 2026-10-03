# A03 分步动作协议 — 验收矩阵

**状态：** 仅为需求草案；尚未实现、测试或验收。本文件不改变 A02。

## 范围边界

A03 是构建在已验收 A02 原子规则引擎之上的 方案 B 分步层。其公开入口是 `guandan.action_state.StepwiseActionState`；它负责私有前缀构造、规范化 token 路径、试执行验证/回滚以及提交计数。它**不**引入或要求 `guandan.environment`、`GameConfig`、观测张量、批量执行或训练集成。这些属于 A04 或后续阶段的范围。

## A03 公开 API

来自 `guandan.action_state`：

- `StepwiseActionState(round_state, *, max_action_tokens=32, max_legal_next_tokens=256, max_token_steps=4096)`；`deal(...)`；`prefix`；`done`；`committed_step`；`token_step`；`truncated`。
- `step(token_id) -> ProtocolStepResult`；`clone()`；`serialize()`/`deserialize()`。
- `legal_next_tokens()`/`legal_tokens()`；`legal_token_rows()`；`legal_token_arrays()`。
- `ActionPrefix`，包含 `phase`、`family`、`declared_ranks`、`declared_length`、`declared_suit`、`selected_card_ids`、`wild_assignments`、`next_expected_kind`、`complete` 和 token 历史。
- `TokenCodec`，用于组合族（family）、牌点（rank）、花色（suit）、长度（length）、实体牌（physical-card）和逢人牌 token（wild-token）的编码/解码。
- `ProtocolStepResult`，包含 `observation: ActionPrefix`、`rewards: float32[4]`、`done`、`truncated`、`is_commit`、协议 `CommittedAction | None`、`token_step`、`committed_step` 和 `info`。
- 协议 `CommittedAction`，包含行动者/阶段/种类（actor/phase/kind）、组合族（family）、实体牌的 `card_ids`、声明的牌点/花色、逢人牌分配、公开索引，以及它所引用的 A02 原子动作。
- `LegalTokenRow(token_id, kind, value, is_commit)`，以及经过填充且带有掩码的合法 token 数组；无效行必须被掩码屏蔽、标记为非提交，并将值置零。
- `ProtocolError` 用于容量/配置错误；提交非法 token 时仍使用 A02 的 `IllegalActionError`；对终局后或截断后的步进调用，保留现有的终局错误契约。

## V1 token 契约

| token | 规范性含义 |
|---|---|
| `0`、`1` | `PAD`、`BOS`；仅作上下文使用；二者都不是动作决策 |
| `2..5` | 阶段上下文（`LEAD_PLAY`、`FOLLOW_PLAY`、`TRIBUTE`、`RETURN`）；仅作上下文使用，规范动作语法以 FAMILY 开头，不发出此类 token |
| `6`、`7` | `PASS`、`COMMIT` |
| `8..17` | 十类组合的序号：SINGLE、PAIR、TRIPLE、FULL_HOUSE、STRAIGHT、PAIR_SEQUENCE、TRIPLE_SEQUENCE、RANK_BOMB、STRAIGHT_FLUSH、FOUR_KINGS |
| `18..32` | 牌点序号映射 `3..A, 2, SMALL_JOKER, BIG_JOKER` |
| `33..37` | `Suit` 枚举值 |
| `38..48` | 长度值 `0..10` |
| `49..63`、`224..255` | 保留；永远不合法 |
| `64..171` | 实体牌 ID `0..107` |
| `172..223` | 逢人牌替代目标的牌点/花色 token，目标不能是王牌 |

`FAMILY` 是每个非过牌出牌的第一个动作 token。`PASS` 是单 token 提交，且仅允许用于存在当前赢家的 `FOLLOW_PLAY`。`TRIBUTE` 和 `RETURN` 的序列为 `CARD, COMMIT`；不允许逢人牌分配。v1 不对外提供 `CANCEL`。

## 对照 `docs/action_protocol.md` 的验收矩阵

| 条款 | A03 要求 | 证据/测试义务 |
|---|---|---|
| §§1–2 生命周期 | 每个对外提供的前缀都有合法的完成方式；前缀是私有的；只有 `COMMIT`/`PASS` 会改变 A02 状态。 | `test_no_dead_prefix_is_exposed`；构造前缀前后的公开状态快照。 |
| §§3–4 词汇/语法 | 按上述精确 v1 范围实现四个阶段和全部十类组合；仅当动作恰好合法且完整时，才允许 `COMMIT`。 | `test_public_api_and_v1_token_ranges`；覆盖各阶段/牌型类别的测试夹具。 |
| §4 规范化顺序 | 三带二先发出三同张组，再发出对子组；连牌中的各组按自然牌面点数 `2..A` 递增；每个实体牌组内的牌按 `card_id` 递增。 | 规范化路径/集合比较，以及排列测试夹具。 |
| §4 逢人牌编码 | 对每张选中的实体逢人牌，恰好发出一个 `WILD` token，即使其目标为该牌的自然牌点/花色也不例外；这些 token 按所选逢人牌的 `card_id` 升序发出，因此两张逢人牌的位置没有歧义。若自然使用与替代使用的语义声明具有相同的牌点/花色，则统一规范化为自然声明。 | 逢人牌/自然牌测试夹具；无重复的规范化路径；两张逢人牌用例。 |
| §5 前缀状态 | `ActionPrefix` 如实反映私有 token 的构造状态，并报告下一个预期的种类及当前是否完整。 | 前缀字段和序列化断言。 |
| §6 提交结果 | 每次成功提交都返回一个协议 `CommittedAction`，并且恰好关联一个 A02 原子动作。 | `test_each_exposed_commit_is_one_legal_canonical_action`。 |
| §7 计数/奖励 | 接受 token 时：`token_step += 1`；构造前缀时：`committed_step` 不变，奖励为零。提交/PASS 时：两个计数器各推进一次；只有终局提交可以携带 A02 奖励。 | 计数、PASS、前缀奖励、终局奖励测试。 |
| §8 合法行 | `legal_token_rows()` 必须完整列出合法 token；`legal_token_arrays()` 使用固定容量并进行填充；不允许只选择前 N 项（top-N）或静默截断。 | 行/数组掩码以及集合完整性测试。 |
| §9 相等性 | 可达的已提交动作键集合，必须与独立的完整动作枚举器产生的集合完全相等；不得因牌顺序或组顺序不同而产生重复项。 | 对缩减牌堆进行穷举，并对完整牌堆进行采样，验证集合相等性。 |
| §10 失败安全 | 阶段/token 错误、不完整的提交、重复牌、以王牌为替代目标、无法压过当前出牌的出牌，以及非法交换牌，都必须原子性地失败。 | 异常和逐字节状态回滚测试。 |
| §10 溢出 | 超出 `max_action_tokens=32` 或 `max_legal_next_tokens=256` 的容量时，必须抛出 `ProtocolError`，绝不能截断。 | 专门构造的边界/溢出测试夹具，以及诊断断言。 |
| §§7/10 截断 | 达到 `max_token_steps` 时设置 `truncated`，胜负奖励保持为零，记录 `info["termination_reason"]`，并拒绝后续步进调用。 | 截断测试。 |
| §§11–12 | 序列化/克隆保留前缀、计数器、合法行/数组以及下一次提交的结果；token 级 PPO/GAE 规则仍留待后续阶段处理。 | 序列化往返后的继续执行测试；A03 不包含训练测试。 |

## 试执行验证与回滚

`step(token_id)` 具有事务性。先为前缀、A02 公开状态、计数器、`done` 和 `truncated` 创建快照；再根据当前合法集合验证 token；在克隆副本上试着扩展前缀或应用提交。**在状态转移之后**，重新计算下一阶段/当前行动玩家及其合法行/数组。如果验证失败、候选动作不完整或存在歧义，或者任一容量在下一次状态转移之前或之后溢出，则抛出 `IllegalActionError`/`ProtocolError` 并恢复整个快照。计数器、前缀、公开历史、牌的归属、奖励和终局标志均不得残留任何变更。

成功的非提交步骤只发布新的私有前缀。成功的提交恰好发布一次 A02 状态转移，清空前缀，使 `committed_step` 恰好推进一次，并返回下一个协议状态。该试执行边界不得修改 A02 的实现或语义。

## 必需测试路径和主代理负责的复测

运行器使用的测试路径必须恰好是 `tests/acceptance/test_a03_actions.py`；`scripts/run_acceptance.py --stage A03` 必须收集该文件中的测试。该测试套件必须覆盖可达性/无死前缀（不存在无法合法完成的前缀）、与独立枚举结果的严格集合相等性、规范化顺序、逢人牌/自然牌等价性、所有阶段/牌型类别、原子回滚、计数器/奖励、序列化、信息边界（协议观测中不包含对手手牌），以及两个溢出点。

复测命令和证据审查由主代理负责。必需证据为 A03 专项测试结果，以及运行器生成的 JSON/报告；本文档不得声称测试已通过或阶段已验收。A04 单独负责 `guandan.environment`、观测张量、批量 API/等价性、填充/数据类型/通道，以及批量信息边界验收。
