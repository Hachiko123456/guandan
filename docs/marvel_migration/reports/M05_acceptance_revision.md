# M05 验收标准修订登记报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 核验基线：`ad964b4d146553780a19099ffc37d7fe72709230`
- M05 实现 commit：`9b4cb016c690650a18d3bc39137f703491c207d4`
- 修订 commit：`c26ed0caafe28c493d5f506a1fe7bc19a345ecf5`
- 状态：`accepted_under_revised_criteria`

## 范围

本次只修订 M05 性能验收分类并登记验收，不修改规则、训练、模型、C++ runtime、checkpoint 或测试代码，不继续 M05 性能调优。

## 依据

- 新规范：`docs/marvel_migration/M05_ACCEPTANCE_CRITERIA_REVISION.md`
- 历史性能报告：`docs/marvel_migration/reports/M05_performance_finish.{md,json}`
- 历史验证：`docs/marvel_migration/evidence/M05_performance_finish/verification.json`
- 修订后验证：`docs/marvel_migration/evidence/M05_acceptance_revision/verification.json`

## 复核结论

- 既有 correctness 和 tests 均有效；
- M05 实现后没有代码变化；
- fixed single `3.203×`、fixed batch `2.051×`，均达到修订后的 `2×` 环境硬门槛；
- Transformer 主集合 `1.636×`、额外集合 `1.585×`、完整 IPPO iteration `1.084×` 均公开保留，属于非阻塞优化目标；
- 未发现新的正确性、安全、隐私、合法动作、牌守恒或恢复 blocker。

## 结论和边界

M05 按修订后的标准接受，解除 M06 blocker。该结论不构成 Kaggle T4×2 远端性能、双卡效率、模型强度或实战能力证明。
