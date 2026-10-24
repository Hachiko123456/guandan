# 条件阶段 5 实施 Prompt：对称性＋Belief

> 只有核心阶段 1–4 全部通过后才执行。本阶段不得实施 opponent pool 或 search。

## 任务

在 `D:/project/guandan` 实施座位/重复牌对称性诊断与条件训练约束，并先完成 belief 信息量预检；只有预检通过才实现 belief head。不得使用 FableDan 权重、策略、标签或蒸馏。

## 一、对称性诊断

构建合法变换：

1. 座位整体旋转：observer/player/leader/team/partner、状态和候选动作同步映射。
2. 双副等价物理牌 copy ID 交换：手牌、历史、current winning、候选动作、wild declaration同步映射。

比较：

[
pi(Pamid Ps)quad	ext{vs}quad Ppi(amid s)
]

报告 max/mean probability difference、KL、argmax一致率，并按phase/action kind/context分桶。先只诊断，不立即改loss。

## 二、对称性训练

若诊断显示显著不一致，再增加可配置 seat rotation augmentation、duplicate-card permutation augmentation 和：

[
L_{sym}=D_{KL}(pi(cdotmid s)Vert P^{-1}pi(cdotmid Ps))
]

要求变换可逆、只比较映射后相同合法候选、不泄漏隐藏信息、`symmetry_weight=0` 时不增加forward/推理开销。暂不直接重写全部card embedding；若一致性仍不足，只提出下一阶段建议。

## 三、Belief B0 信息量预检

- 真实C++引擎，至少数百局，多等级/leader/贡还。
- 样本为行为方宏动作起点。
- 输入只能是策略实际可见观测。
- 标签为相对座位 next/partner/previous ×15等级剩余计数。
- train/validation/test必须按deal切分；同一副牌的任何决策不得跨集合。

基线：

A. 无条件均匀先验。
B. 公开信息强基线：自己手牌、已出牌、各家剩余张数、公开贡还。
C. 当前channels线性/浅层probe。
D. 完整history+channels probe。

报告每座位count L1、rank MAE、exact-count accuracy、手牌守恒误差、校准、按deal聚类bootstrap CI，以及D相对B的增量。

Gate不能只要求击败均匀先验。必须证明完整history相对强公开基线有稳定增量。若失败：停止belief实现，保留数据/代码/指标和否决报告；不得筛样本或按step随机切分。

## 四、Belief B1（仅B0通过）

建议输出 `3 relative seats × 15 ranks × counts 0..8` categorical logits。

主损失为count cross entropy；辅助守恒约束：每家预测总牌数匹配公开剩余张数，各等级三家预测总量匹配可推导剩余等级数。

belief head从player_state构建，训练期存在，推理不构建。要求：

- `belief_weight=0` 时不构建；
- policy-only warm start白名单明确；
- strict评估checkpoint按配置构建；
- 推理参数量和延迟不增加。

测共享主干梯度比例：

[
ho=rac{|lambda_b
abla L_{belief}|}{|
abla L_{RL}|}
]

分别报告decoder/state_mlp/actor_fuse。不能因为外部项目使用0.05就直接固定0.05，应按实测选择候选。

## 五、A/B与验收

分别做 symmetry off/on、belief off/on；同warm start、seed、发牌和局数，固定牌副配对评估。辅助loss下降不等于棋力提升。

验收：变换规则合法、动作映射完整、无隐藏泄漏、symmetry KL下降且实战不退化；belief标签与oracle一致、B0按deal切分、history显著优于强基线、梯度进入共享主干、推理零开销。只有paired棋力CI下界>0才默认开启。

## 六、交付报告

包含修改文件、变换定义、旧模型诊断、symmetry A/B、B0数据和切分、四组基线、bootstrap CI、B0通过/否决；若通过再报告belief结构/梯度/A/B、全量回归和风险。明确未实施条件阶段6。完成后停止。
