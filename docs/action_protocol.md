# 掼蛋方案 B 动作协议

- **文档 ID：** GD-ACTION-0.1
- **兼容规则：** GD-RULES-0.1
- **兼容环境：** GD-ENV-0.1
- **方案：** 逐步 token 构造，并显式使用 COMMIT token。

## 1. 目标

策略不从扁平动作列表中选择完整牌型，而是一次选择一个合法 token。环境为当前行动玩家维护一个私有的未完成构造。只有 COMMIT 才会将该构造转换为一个原子的游戏动作。

该协议适用于四个决策阶段：

- LEAD_PLAY：选择任意合法的非过牌出牌；
- FOLLOW_PLAY：选择能够压过当前出牌的合法出牌，或选择 PASS；
- TRIBUTE：恰好选择一张符合条件的进贡牌；
- RETURN：恰好选择一张符合条件的还贡牌。

## 2. 动作生命周期

~~~text
phase starts
  -> PREFIX token 0
  -> PREFIX token 1 ...
  -> optional PREFIX tokens
  -> COMMIT
  -> atomic rule transition
  -> next phase/active player
~~~

只有当一个前缀至少能扩展成一个完整的合法已提交动作时，该前缀才有效。环境不得暴露无法完成合法动作的前缀。

未完成前缀属于当前行动玩家的私有信息，不会追加到公开历史中。被拒绝的 token 不改变状态。

## 3. Token 类型

首个词汇表必须支持以下语义类型。数值 ID 由实现定义，但必须进行版本化。

### 3.1 控制 token

- PASS：仅在 FOLLOW_PLAY 中有效；立即提交，且没有额外前缀。
- COMMIT：仅当当前前缀恰好描述一个完整的合法动作时有效。
- CANCEL：不是游戏动作，且在 v1 中不暴露。若要放弃构造，请恢复 clone 所得状态或构造前缀之前的状态。

### 3.2 出牌构造 token

语义类型包括：

- FAMILY: SINGLE, PAIR, TRIPLE, FULL_HOUSE, STRAIGHT, PAIR_SEQUENCE, TRIPLE_SEQUENCE, RANK_BOMB, STRAIGHT_FLUSH, FOUR_KINGS;
- RANK：牌面点数或声明点数；
- LENGTH：需要时使用的组/序列长度；
- SUIT：STRAIGHT_FLUSH 所需的花色；
- CARD：当前行动玩家手中的实体 card_id；
- WILD_ASSIGNMENT：为一张选定逢人牌声明的非王牌目标；
- GROUP_BOUNDARY：在规范语法要求时分隔三同张/对子或重复牌组。

### 3.3 进贡/还贡 token

- CARD 选择一张实体牌。
- COMMIT 将其最终提交。
- TRIBUTE 或 RETURN 中不允许逢人牌指定。

## 4. 规范语法

实现必须选择一种规范语法，并且只暴露该语法的前缀。以下语法是 v1 的规范；只有在产生相同已提交动作集合且将该选择记录为协议修订版时，才允许使用等价的确定性编码。

### 4.1 LEAD_PLAY 与 FOLLOW_PLAY

FOLLOW_PLAY 还会额外暴露 PASS。

对于非过牌出牌：

~~~text
FAMILY
  -> family-specific declaration tokens
  -> CARD selections and WILD_ASSIGNMENT tokens as required
  -> COMMIT
~~~

环境不得接受会产生重复规范动作的出牌顺序。规范化规则如下：

- 每个语义组内的实体牌按 card_id 排序；
- 各组按声明点数递增排序，但 FULL_HOUSE 先输出其主要三同张组；
- 逢人牌指定按实体 card_id 排序；
- COMMIT 前不得出现未使用的 token。

允许的各牌型专用结构：

- SINGLE: FAMILY, CARD, 可选 WILD_ASSIGNMENT, COMMIT;
- PAIR/TRIPLE: FAMILY, RANK, CARD..., 可选 WILD_ASSIGNMENT..., COMMIT;
- FULL_HOUSE: FAMILY, TRIPLE_RANK, PAIR_RANK, CARD..., 可选 WILD_ASSIGNMENT..., COMMIT;
- STRAIGHT: FAMILY, START_RANK, CARD..., 可选 WILD_ASSIGNMENT..., COMMIT;
- PAIR_SEQUENCE: FAMILY, START_RANK, LENGTH, CARD..., 可选 WILD_ASSIGNMENT..., COMMIT;
- TRIPLE_SEQUENCE: FAMILY, START_RANK, LENGTH, CARD..., 可选 WILD_ASSIGNMENT..., COMMIT;
- RANK_BOMB: FAMILY, RANK, LENGTH, CARD..., 可选 WILD_ASSIGNMENT..., COMMIT;
- STRAIGHT_FLUSH: FAMILY, SUIT, START_RANK, CARD..., 可选 WILD_ASSIGNMENT..., COMMIT;
- FOUR_KINGS: FAMILY, CARD, CARD, CARD, CARD, COMMIT.

CARD 和 WILD_ASSIGNMENT token 的数量由牌型、声明和长度共同确定。只有在精确的必需结构完整后，COMMIT 才可以合法。

### 4.2 TRIBUTE

~~~text
CARD
COMMIT
~~~

只允许暴露 GD-RULES-0.1 中规定的符合条件的实体牌集合。

### 4.3 RETURN

~~~text
CARD
COMMIT
~~~

只允许暴露牌面点数为 2..10 的自然非王牌，并排除红桃级牌实体逢人牌。

## 5. 前缀状态

私有前缀记录至少包含：

~~~python
@dataclass
class ActionPrefix:
    phase: str
    family: str | None
    declared_ranks: tuple[int, ...]
    declared_length: int | None
    declared_suit: int | None
    selected_card_ids: tuple[int, ...]
    wild_assignments: tuple[tuple[int, int, int | None], ...]
    next_expected_kind: str
    complete: bool
~~~

当 complete 为 false 时，不修改实体手牌。成功 COMMIT、reset 或恢复 clone 状态后，丢弃该前缀。

## 6. COMMIT 语义

COMMIT 是唯一授权规则转移的 token。

成功执行 COMMIT 时：

1. 根据规则引擎验证完整前缀；
2. 构造一个规范的 CommittedAction；
3. 为出牌、进贡或还贡移除/转移选定的实体牌；
4. 更新公开历史和当前阶段；
5. 在本局结束时计算终局结果/奖励；
6. 清除前缀；
7. 暴露下一位行动玩家的观测。

如果验证失败，则抛出 IllegalActionError，并保持所有字段（包括计数器）不变。

CommittedAction 必须包含：

~~~python
@dataclass(frozen=True)
class CommittedAction:
    actor: int
    phase: str
    kind: str                 # PLAY, PASS, TRIBUTE, RETURN
    family: str | None
    card_ids: tuple[int, ...]
    declared_ranks: tuple[int, ...]
    declared_suit: int | None
    wild_assignments: tuple[tuple[int, int, int | None], ...]
    public_index: int
~~~

## 7. token 步与已提交游戏步记账

观测、步骤结果、轨迹采样记录和诊断中都必须包含这两个计数器。

### 7.1 token 步

- 重置时 token_step 从零开始。
- 每个被接受的 token（包括前缀 token 和 COMMIT）都会使其加一。
- 被拒绝的 token 不会使其增加。
- 它衡量神经决策粒度，并作为首个 MARVEL 适配器的时间轴。

### 7.2 已提交游戏步

- 重置时 committed_step 从零开始。
- 每个成功产生一个原子规则动作的 COMMIT 都使其加一。
- 非提交前缀 token 不会使其增加。
- PASS、PLAY、TRIBUTE 和 RETURN 各计为一个已提交游戏步。
- 单 token 的 PASS 仍计为一个已提交游戏步。

### 7.3 必需标志与奖励

每个步骤结果都包含：

~~~text
is_commit: bool
committed_action: CommittedAction | None
reward: float[4]
done: bool
truncated: bool
token_step: int
committed_step: int
~~~

非提交步骤的 is_commit 为 false、没有已提交动作且奖励为零。成功提交时 is_commit 为 true；只有终局提交可以有非零奖励。

## 8. 合法下一 token 集合

每个非终局观测都会返回一个填充后的合法下一 token 行列表。每行包含 token_id、语义类型、语义值、is_commit 和有效掩码。

该集合必须完整覆盖当前前缀和阶段下的所有合法下一 token。不得只选择 top-N 个完整动作来生成它。容量溢出属于环境配置错误，必须抛出诊断，而不是静默截断。

## 9. 动作相等性与去重

如果两个 token 序列选择了相同的实体牌、牌型、声明和逢人牌指定，则它们是同一个已提交动作。环境只暴露一条规范序列。

- 重排对子中的牌不构成第二个动作。
- 交换三带二中的牌组不构成第二个动作。
- 当最终声明不同时，将逢人牌指定为不同目标属于不同动作。
- 只有在已提交动作不同时，按本牌使用的红桃级牌与其逢人牌指定才属于不同声明；相同的规范动作只暴露规范的本牌形式。

## 10. 失败与安全行为

- 来自其他阶段的 token 不合法。
- 不在 legal-next-token 集合中的 token 不合法。
- 未完成的 COMMIT 不合法。
- 重复选择实体牌不合法。
- 将逢人牌指定为王牌不合法。
- 不能压过当前轮次领先出牌的出牌不合法，PASS 除外。
- 领出时或不存在当前领先者时使用 PASS 不合法。
- 超过 max_action_tokens 属于协议错误，而不是截断。
- 达到 max_token_steps 时，回合将因诊断原因被截断，且不产生胜负奖励。

## 11. 验收前必需的协议测试

实现必须证明：

1. 每个合法完整动作都有一条暴露的前缀路径；
2. 每个暴露的 COMMIT 都恰好产生一个合法的已提交动作；
3. 不暴露无法完成合法动作的前缀；
4. 不暴露非法牌或逢人牌指定；
5. 规范化会移除排列重复；
6. 独立的完整动作参考枚举器与逐步协议在缩减牌堆及采样的完整牌堆状态上，产生完全相同的已提交动作集合；
7. 非提交步骤不修改公开游戏状态或已提交游戏步计数；
8. 提交步骤只修改一次状态，并且只增加一次已提交游戏步计数；
9. 序列化会保留前缀和下一合法 token 集合；
10. token 级奖励和终局奖励与 GD-ENV-0.1 一致。

## 12. V1 固定决策与延后范围

1. V1 使用 `GD-ENV-0.1` 中的全局 token 词汇表和数值范围。
2. `max_action_tokens=32` 和 `max_legal_next_tokens=256`；溢出属于协议错误，绝不静默截断。
3. 一次策略前向传递选择一个标量下一 token。
4. 首个训练适配器使用 token 级 PPO/GAE。前缀扩展行的奖励为零；只有终局提交可以有非零的单副牌局奖励。已提交动作的对数概率聚合被延后处理，不得混入 v1 检查点。
5. 后续协议版本可以增加不同的目标或词汇表，但必须拒绝不兼容的 v1 检查点。
