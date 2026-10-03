# STATUS.yaml 模式定义

每个阶段都记录 `status`、`implementation`、`tests`、`accepted`、`commit`、`evidence` 和 `blockers`。

规则：

- `accepted: true` 要求同时具备 `status: accepted`、完整的 Git SHA 和证据路径。
- 自动生成的报告始终保持 `accepted: false`。
- 即使针对性测试通过，未完成的代码仍须保持 `accepted: false`。
- 阻塞问题必须具体且可复现。
