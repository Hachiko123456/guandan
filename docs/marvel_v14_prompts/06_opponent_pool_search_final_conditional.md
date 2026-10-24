# 条件阶段 6 实施 Prompt：Opponent Pool＋搜索＋最终训练工具

> 只有核心阶段 1–4 已通过且条件阶段 5 已审查后才执行。不得未经用户授权启动正式长训。

## 任务

在 `D:/project/guandan` 实施 checkpoint opponent pool、value资格验证、条件部署搜索和最终分级训练/评估工具。不得使用 FableDan 蒸馏；FableDan只能作为最终外部评估对手。

## 一、Opponent Pool

支持 current policy 与冻结 historical checkpoint 按队伍/座位对战。新增明确字段：

- policy_owner
- policy_train_mask
- opponent_checkpoint_id
- opponent_policy_version

规则：

1. current-controlled token：`policy_train_mask=True`，必须严格on-policy。
2. frozen opponent token：mask=False，动作进入历史但不进入PPO，不要求其behavior probability等于current policy。
3. 对手eval模式、requires_grad=False、不进入optimizer。
4. current/opponent队伍和座位轮换平衡。
5. chance/environment步骤保持原规则。
6. macro不得跨policy owner。
7. policy loss只选current宏动作。

支持 current-vs-current 和 current-vs-historical 的可配置混合；具体权重通过A/B选择，不写死为理论常数。Pool条目包含checkpoint sha256、architecture identity、权重、active状态，可覆盖recent/spaced/champion历史版本。

Recovery保存pool manifest、采样RNG、当前opponent identity和assignment。验证 frozen参数hash不变、不在optimizer，opponent步骤不进PPO，current/current模式与旧单策略逐位一致，DDP/recovery精确恢复，不兼容checkpoint拒绝。

## 二、Value资格验证

搜索前生成 value qualification report：

1. explained variance：(1-Var(target-pred)/Var(target))；
2. Pearson/Spearman；
3. early/mid/late；
4. DOUBLE_DOWN/HEAD_THIRD/HEAD_LAST；
5. seat/team/context；
6. calibration；
7. 固定状态反事实候选排序准确率。

Value Gate：EV长期为正且稳定，中后局排序显著优于随机，无严重seat/context偏差。失败时不得实施搜索，只输出否决报告。

## 三、部署搜索（仅Gate通过）

只在evaluation/deployment路径启用，训练零影响、默认关闭。

确定化必须只用行动玩家公开信息，满足108张守恒、自手牌/已出牌/公开贡还/剩余张数约束。若belief被否决，使用公开信息一致先验；若belief通过，可作proposal但仍须满足硬约束。

候选必须是完整宏动作或可正确映射的top-k自回归sequence；不得把下一个token top-k当完整动作top-k。

前瞻必须沿真实引擎推进，正确处理PASS、牌权、出完、接风、贡还；team perspective符号正确；分支不得修改主环境。评分明确使用value/Q/先验混合，参数可配。

## 四、搜索A/B

固定独立牌副、四座位轮换：纯argmax vs argmax+search。按deal配对和cluster bootstrap，报告win-rate与rank-point delta、outcome、延迟p50/p90/p95/p99、timeout/fallback。只有CI下界>0且延迟满足预算才建议默认开启。

## 五、最终训练Gate工具

只实现工具，不自动长训：

- Gate1：50 iter健康检查；
- Gate2：250 iter（约5000局）；
- Gate3：500 iter（约10000局）；
- Gate4：1000 iter；
- Gate5：1500 iter（约30000局）。

每个Gate冻结evaluation checkpoint，运行new-vs-M09 paired、rule/random、context分片、value、DOUBLE_DOWN、health和recovery。CI lower>0可继续；upper<0停止；跨0增加seed/牌副，不能自动烧完预算。

## 六、数据隔离与最终比较

严格区分 training/development anchor/selection/final holdout。Final在候选冻结后只运行一次，不用于调参，不得运行后改选checkpoint。

最终FableDan比较只作外部评估，报告全集、level、leader、tribute、candidate team、rotation、outcome、win rate、rank-point、cluster CI及adapter诊断。“超过FableDan”使用预锁统计标准，例如CI95下界>0.5，不得只看点估计。

## 七、禁止事项

不得蒸馏FableDan，不得自动长训，Value Gate失败不得搜索，opponent步骤不得进current PPO，不得破坏current/current legacy，不得用selection/final训练，不得选择性报告。

## 八、交付报告

包含修改文件、pool结构、on-policy mask、frozen验证、DDP/recovery、value完整数字和Gate结论；若通过再报告搜索合法性/A-B/延迟；报告训练Gate工具、数据隔离、全量回归和风险，并明确是否实际启动长训（未经授权必须为否）。完成后停止。
