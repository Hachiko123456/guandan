# 评估、Champion 与 Opponent Pool 规范

## 1. 数据隔离

### Training deals

用于采样训练。不得用于正式模型选择统计。

### Selection deals

固定、版本化，用于周期性 candidate/champion 比较。可多次使用，因此它只负责选择，不负责最终无偏报告。

### Final holdout deals

最终候选锁定后只运行一次。任何根据 final holdout 调参、选 checkpoint 或重训的行为都会使该 holdout 失效，必须生成新版本。

三者 seed/domain 不得重叠，manifest 必须记录生成器版本、hash 和级牌/贡还范围。

## 2. 评估矩阵

每个 candidate 至少评估：

- vs current champion；
- vs rule；
- vs random；
- vs selected legacy V1 checkpoint；
- vs opponent-pool 抽样历史 champion。

首阶段不比较 FableDan。FableDan 需要单独的跨引擎公平性方案。

## 3. 公平性

- 固定基础发牌；
- 四个座位角色轮换；
- 相同 candidate/opponent policy mode；
- 相同动作温度；
- 相同规则、级牌和贡还条件；
- 每个基础发牌的四轮结果作为一个 cluster；
- 不把同一 deal 的四个旋转当四个独立样本计算置信区间。

## 4. 指标

至少报告：

```text
eval/win_rate
eval/clustered_ci95_low
eval/clustered_ci95_high
eval/team02_win_rate
eval/team13_win_rate
eval/seat_rotation_gap
eval/mean_team_reward
eval/double_down_rate
eval/head_third_rate
eval/head_last_rate
eval/illegal_count
eval/truncated_count
eval/games_per_second
```

斗地主的 landlord/peasants 指标不直接复制。参考仓库中名为 `ci` 的统计必须重新审查其数学含义；V2 正式输出明确使用 clustered 95% CI。

## 5. Champion 晋升

初始建议门槛，M06 可根据统计审查形成 ADR，但不得降低为“点估计略高就晋升”：

1. candidate vs champion 完整四座位矩阵；
2. win-rate 点估计 > 50%；
3. clustered 95% CI 下界达到晋升阈值或通过预先定义的 sequential gate；
4. 对 rule 不出现超过容忍度的回退；
5. 对 random 不出现异常回退；
6. illegal/truncated 都为 0；
7. checkpoint、协议和源码身份完整；
8. selection 数据没有被训练采样。

若样本不足，状态应为 `inconclusive`，不得自动晋升。

## 6. Opponent pool

保留：

- 当前 champion；
- 最近若干 champion；
- 代表不同阶段/策略风格的 checkpoint；
- rule/legacy anchor。

训练抽样概率必须配置化和记录；不能只和最新自己训练导致策略遗忘。pool 变更写入 manifest，恢复训练时可重建。

## 7. 日志和产物

每次评估保存：

- evaluation plan；
- candidate/opponent checkpoint hash；
- deals manifest；
- 每局记录；
- cluster 统计；
- seat rotation coverage；
- summary；
- promotion decision；
- champion manifest 更新或保持不变的理由。

## 8. 发布强度层级

建议使用分级，而不是一个模糊的“可用”：

- E0：管线可运行；
- E1：稳定击败 random；
- E2：对 rule 达到可竞争水平；
- E3：稳定击败 rule 且优于 legacy champion；
- E4：通过独立强基线/人类测试；
- E5：完整多副实战发布。

M09 至少目标为 E3。E4/E5 需要额外基线和完整比赛管理器。

## 9. M06 已实现协议

M06 canonical schemas、四座位 role mapping、deal-cluster t interval、promotion 状态机、manifest 原子性、legacy opponent-only adapter、pool recovery、training hook 和 final one-shot audit 见 `M06_EVALUATION_PROTOCOL.md`。本地小规模运行只证明 E0 管线，不构成模型强度或 Kaggle 性能结论。
