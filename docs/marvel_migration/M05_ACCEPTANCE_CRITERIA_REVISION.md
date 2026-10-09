# M05 验收标准修订记录

- 修订 schema：`guandan-marvel-m05-acceptance-revision-v1`
- 修订日期：2026-10-10
- 适用阶段：`M05_differential_performance`
- 原实现 commit：`9b4cb016c690650a18d3bc39137f703491c207d4`
- 原阻塞登记 commit：`ad964b4d146553780a19099ffc37d7fe72709230`
- 状态：approved

## 1. 修订原因和边界

经用户明确批准，M05 的本机 CPU 性能验收口径调整如下。本修订只改变性能门槛分类，不改变任何规则、动作、观测、隐私、恢复或 checkpoint 语义，也不授权继续进行 M05 性能调优。

旧报告、旧协议和旧机器证据中“真实冻结 Transformer rollout 未达到 2×”的结论保持原样，仍是执行当时真实且可审计的历史结果。新验收结论通过本修订记录和独立的修订后验证文件叠加表达，不回写或改造旧结果。

## 2. 修订后硬门槛

以下门槛继续阻塞 M05，任一失败均不得接受：

1. Python/C++ 规则、状态、合法动作、reward、terminal 和完整 trajectory 语义一致；
2. 四座位信息隔离、privileged private state 边界和隐私验证通过；
3. 牌守恒、非法动作拒绝、整数边界、serialization/resume 和 checkpoint/backend identity 安全通过；
4. 10k 完整终局、100k 中间状态、13 level 与 directed complex-action/corpus replay 证据有效；
5. fixed-action single-env 同机固定工作负载 Python→C++ 加速 `>=2.0×`；
6. fixed-action batch 同机固定工作负载 Python→C++ 加速 `>=2.0×`；
7. scoped tests 和目标依赖回归没有新的 blocking failure；
8. C++ 仍无 silent fallback，Python reference 仍可作为显式 oracle。

## 3. 非阻塞优化目标

以下项目必须公开实测值，但不再单独阻塞 M05：

- 本机 CPU 真实冻结 Transformer rollout；
- 预先固定的额外 Transformer rollout 集合；
- learner 不变的完整 IPPO iteration 总吞吐。

它们仍用于识别端到端固定开销和后续优化方向，不得被描述为达到 2×。

## 4. 保留的历史实测结果

- 主 Transformer rollout：`1.6359769458438593×`，报告时四舍五入为 `1.636×`；
- 额外集合：`1.5852854548521904×`，报告时四舍五入为 `1.585×`；
- 完整 IPPO iteration：`1.0840783354992432×`，报告时四舍五入为 `1.084×`；
- fixed-action single-env：`3.202622103753068×`；
- fixed-action batch：`2.050500469537274×`。

## 5. 远端和强度声明边界

- Kaggle T4×2 的实际吞吐、显存、稳定性和双卡效率仍必须在 M08 以真实远端证据验证；
- 本修订不代表 Kaggle 性能、双卡效率、训练收敛、模型强度或可实战性已经通过；
- M06 仅可开展本地评估/champion/opponent-pool 管线工作；不得开始 Kaggle 正式训练。

## 6. 修订后验收结论

对当前仓库和既有机器证据复核后：

- 正确性与安全证据有效；
- 自 M05 实现 commit 起没有训练/规则/C++/测试代码变化；
- fixed-action single-env 与 batch 均达到修订后的 `>=2.0×` 硬门槛；
- 唯一旧 blocker 是现已改为非阻塞目标的真实 Transformer rollout 2× 条款；
- 未发现新的正确性、安全或恢复阻塞。

因此 M05 按 `guandan-marvel-m05-acceptance-revision-v1` 接受，并允许启动 M06。M05 的旧 blocked 报告和旧 `accepted: false` 验证文件继续保留为历史记录。
