# STATUS.yaml 模式定义

每个阶段都记录 `status`、`implementation`、`tests`、`accepted`、`commit`、`evidence` 和 `blockers`。

规则：

- `accepted: true` 要求同时具备 `status: accepted`、完整的 Git SHA 和证据路径。
- 自动生成的报告始终保持 `accepted: false`。
- 即使针对性测试通过，未完成的代码仍须保持 `accepted: false`。
- 阻塞问题必须具体且可复现。


## Profile 验证

A05/A06/A07 等阶段可在同一个阶段下分别记录：

```yaml
verification:
  local_fast: not_run|passed|failed
  remote_full: not_run|passed|failed
  stress: not_run|passed|failed
```

`local_fast` 通过只能说明本机接线和最小正确性通过，不能自动设置阶段 `accepted: true` 或替代 `remote_full`。
