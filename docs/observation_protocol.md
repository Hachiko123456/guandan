# 观测协议

本文档描述预期的适配器形态。只有在 A00 规则决策通过后，具体维度才会固定。

一次批处理步骤会暴露兼容 NumPy 的数组：

- `observations.tokens`：`[B, P, L_obs]`，每个观测者对应的整数 token ID。
- `observations.channels`：`[B, P, C]`，公开/自身状态通道。
- `player_ids`：`[B]`，当前行动玩家，或表示机会事件/管理操作的负数 ID。
- `available_actions.tokens`：`[B, A, L_act]`。
- `available_actions.weights`：`[B, A]`，合法动作先验/权重。
- `available_actions.is_commit`：`[B, A]`，选择该动作是否会提交一个完整动作。
- `rewards`：`[B, P]`，按队伍对齐的奖励向量。
- `is_terminal`：`[B]`。

策略视图可以包含当前行动玩家的手牌和全部公开信息，但绝不能包含对手或队友的私有牌。集中式价值评估器和仅供训练使用的信念标签与策略观测分离。
