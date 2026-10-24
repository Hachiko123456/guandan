# 04 · 评估说明：怎样判断模型真的变强

[操作手册](03_runbook.md) · [下一篇：开发指南](05_development.md) · [术语表](glossary.md)

> 适用范围：当前 V2 评估机制、v1.4 训练与条件资格的结论边界。代码核对：2026-10-21。以下数值例子不是项目实测。动态资格、残余问题与授权范围统一见 [CURRENT_EXECUTION.md](../marvel_v14_prompts/CURRENT_EXECUTION.md)。

## 1. 能更新参数，不代表更会打牌

| 看到的现象 | 最多说明 | 不能直接说明 |
|---|---|---|
| 程序能启动 | 入口和部分依赖可用 | 完整牌局、训练或恢复正确 |
| loss 有限、参数变化 | 有数值上有效的更新 | 策略变强 |
| 自我对弈平均回报变化 | 训练分布内读数变化 | 对固定对手进步 |
| 小样本赢了多数局 | 这次样本中赢得更多 | 稳定优于对手 |
| 严格隔离的足量公平评估 | 对指定范围有统计支持 | 所有规则/设备/人类实战都更强 |

棋力评估需要固定比较对象、数据和协议；工程正确性测试不能取代它。

## 2. 三类数据不能混用

| 数据域 | 可以做什么 | 不能做什么 |
|---|---|---|
| training | 采集轨迹并更新模型 | 当最终无偏棋力证据 |
| selection/development | 比较 checkpoint、调配方和选候选 | 冒充从未被选模接触的最终考试 |
| final holdout | 候选冻结后进行一次独立判断 | 看结果后继续选模，却沿用原 holdout 结论 |

配置应记录 domain、生成器版本、seed/区间、规则上下文和 manifest hash，并核对实际训练 provenance。仅声明“使用不同 seed”不能覆盖分布/身份检查；qualification 等额外域也需显式隔离。

最终数据是会被使用行为消耗的资源：候选、协议和数据集应预先冻结，按 reserve-before-touch 等机制记录。用最终结果反复调参会破坏其独立性，应重新建立隔离的最终协议，而不是删记录重跑。

源码：[manifests.py](../../guandan_marvel/evaluation/manifests.py)、[provenance.py](../../guandan_marvel/evaluation/provenance.py)、[host.py](../../guandan_marvel/evaluation/host.py)。细则见 [M06 修复协议](../marvel_migration/M06_EVALUATION_REPAIR_PROTOCOL.md)、[最终数据集绑定协议](../marvel_migration/M06_FINAL_DATASET_BINDING_REPAIR_PROTOCOL.md)。

## 3. 与谁对战？

| 对手/对照 | 作用 | 局限 |
|---|---|---|
| random | 检查明显异常和最低基线 | 胜过随机不代表会合作或具备竞争力 |
| rule | 检查能否应对固定启发式 | 可能针对规则弱点过拟合 |
| 历史 checkpoint / champion | 判断是否优于已有候选、有无遗忘 | 相对提升依赖对手身份和分布 |
| evaluation opponent pool | 扩大评估对手覆盖 | 不等于训练时已启用 checkpoint pool |
| FableDan 等外部基线 | 单独协议下检验外部竞争力 | 需要动作/规则适配与公平性检查 |

必须记录具体 checkpoint hash 和 policy mode，不能只写“打了旧模型”。M06 的默认晋升矩阵与某个后续专项比较协议也不是同一份计划；按本次锁定 evaluation plan 检查覆盖，不能自己删掉不方便的对手。

FableDan 只能作为单独授权的外部评估对手，不提供训练数据、监督标签或蒸馏目标。适配失败不能静默退到 random/rule 后继续算外部胜率。相关专项代码位于 [comparison](../../guandan_marvel/comparison)。

## 4. 为什么要相同发牌与座位轮换？

不同模型拿到的手牌强弱可能不同。公平比较需要同一基础发牌，按协议轮换四个座位/队伍角色，并使用一致的级牌、贡还、规则和决策模式。

**教学示例**：同一发牌四次轮换，候选的胜负为 `[1,0,1,0]`，该 deal cluster 的均值为 0.5。不能把它们当四份独立发牌，假装样本量扩大四倍。

```text
独立发牌 A → 4 个座位轮换 → cluster A 的结果
独立发牌 B → 4 个座位轮换 → cluster B 的结果
独立发牌 C → 4 个座位轮换 → cluster C 的结果
                        ↓
             按独立 cluster 估计不确定性
```

32 个独立发牌、每个 4 次轮换，是 128 局对战、32 个独立 cluster；不是 128 份独立证据。多对手场景还要区分每个对手的覆盖和统计，不能把混合对手总局数直接当某个单对手样本量。

## 5. 统计读数怎样解释？

| 指标 | 主要问题 |
|---|---|
| win rate | 在这套协议中赢了多少 |
| clustered CI | 有限独立发牌下，不确定性有多大 |
| team02/team13、seat rotation gap | 是否对某组座位/角色偏斜 |
| mean team reward / rank points | 赢法与队伍名次收益怎样 |
| double-down/head-third/head-last rate | 胜负结构是否变化 |
| illegal/truncated/incomplete | 结果是否有效、计划是否完成 |
| throughput | 评估速度，不是棋力 |

### 置信区间不是装饰

**仅为读数示例**：若某协议得到点估计 55%、区间跨过 50%，不能只凭 55% 就宣布稳定优于对照。若样本不足，应报告 `INCONCLUSIVE`。正式晋升还要满足预先冻结阈值、健康、覆盖和身份要求。

当前 [stats.py](../../guandan_marvel/evaluation/stats.py) 的 `clustered_mean_ci95` 使用 **bounded empirical-Bernstein 95% interval**，输入为独立 deal 均值，0/1 个 cluster 时不提供有效结论。不要拿旧迁移文档的 t-interval 描述替代当前实现，也不要与 Value qualification 的 cluster bootstrap 混成一种算法。实际报告应读取 metric/schema/method。

大量重复、全胜或全败的少量数据，不能自动获得零宽度区间。原始样本、分组方法和统计版本都需要保留，才可复算。

## 6. 怎样选 checkpoint 和晋升 champion？

典型流程是：锁定 selection plan → 比较候选与对照 → 验证矩阵、身份和独立 cluster → 计算统计 → 按预先约定政策决定晋升或保持。

[PromotionPolicy](../../guandan_marvel/evaluation/promotion.py) 有 formal/smoke 区分、最少 cluster 与胜率/CI 门槛；这里不把默认阈值复制成所有未来任务都适用的标准。**一次运行要保存它实际冻结的 policy。**

smoke 只检查管线，不能因其门槛低就把结果作为 formal 晋升证据。champion manifest 应原子更新并绑定实际 checkpoint/协议/源码/数据身份；失败时保留原因，不用新文件名代替合格晋升。

## 7. 三类 Gate 必须区分

| 名称 | 判断什么 | 不会自动证明什么 |
|---|---|---|
| 训练阶段 Gate | 到达实际计数与预算、健康、保存恢复及所需证据 | 棋力已提升、搜索可启用 |
| 棋力/晋升 gate | 在锁定评估协议下是否满足对照和统计条件 | 所有功能都具备资格 |
| Value/Search qualification gate | Value 是否足以支持被允许的搜索用途 | 已有搜索实现、最终对战更强 |

当前训练 Gate 工具定义绝对迭代目标 50/250/500/1000/1500。`dry_run` 只输出计划；`complete` 或计数达到目标也不等于所有资格/健康/恢复和棋力检查通过。实现中的 `formal_gate_reached` 还有其他约束，调用者必须提供真实检查和训练 factory。

工具来源：[training_gates.py](../../guandan_marvel/phase6/training_gates.py)、[design.py](../../guandan_marvel/phase6/design.py)。当前实际推进和每次授权仍看当前入口。

## 8. 为什么价值拟合好，不一定适合搜索？

训练 Value 的目标是预测某策略与分布下的后续收益。搜索还需要在相邻候选中可靠地比较“选哪个更好”，两者并不等同。

资格可能考察：独立 deal 数、EV/相关性/校准、不同牌局阶段与上下文切片、候选排序、身份/数据隔离和原始样本可复算性。训练日志中的 Value EV 不能直接代替正式 qualification。

### 并列候选的陷阱

若真实最佳候选集合为 `T`、候选总数为 `k`，随机选择命中最佳的基准是 `|T|/k`，不总是 `1/k`。所有候选真实收益都相同时，命中率 100% 与随机基准也都是 100%，没有排序信息，不能算有效提升。

当前 tie-aware 实现、历史 design 文本、artifact schema/provenance 与原始样本持久化还需要按当前入口列出的要求核对；不得把 metadata-only 的执行报告当可复算资格。详见 [value_gate.py](../../guandan_marvel/phase6/value_gate.py)、[搜索条件入口](../../guandan_marvel/phase6/search.py) 和 [06B 收尾要求](../marvel_v14_prompts/06B_value_gate_ranking_closeout.md)。

**即使未来资格 PASS，也必须有真实搜索实现和部署验证。** 反过来，基础策略训练不必为了开始训练强迫 Value Gate PASS；不具资格时保持搜索关闭。

## 9. 一份可信评估报告至少有什么？

- [ ] 明确回答比较对象、规则/上下文范围和 policy mode。
- [ ] 锁定 plan、对手矩阵、seed/domain 与数据 manifest。
- [ ] candidate/opponent checkpoint、source/native/config/metric 身份可追溯。
- [ ] 记录每局结果、deal cluster、座位轮换与实际完成数量。
- [ ] 缺失、失败、非法、截断如实报告；不以计划值补实际值。
- [ ] 提供点估计、不确定性、样本不足判定和原始可复算数据。
- [ ] selection 与 final 用途区分；final 的一次性使用有记录。
- [ ] 结论不超过实际样本、对手、上下文和版本。

本篇只解释阅读与设计方法，没有运行任何正式对战、晋升或最终 holdout。更多协议见 [评估与 champion 规范](../marvel_migration/06_EVALUATION_CHAMPION_SPEC.md)，它的历史描述需结合当前代码及后续修订。
