# 续接提示词

```text
继续执行 D:\project\guandan 的总体执行计划。

阅读 project_status/STATUS.yaml、docs/MASTER_EXECUTION_PLAN.md、docs/AGENT_EXECUTION_PROTOCOL.md、docs/completion_criteria.md 和当前阶段的验收文档，查看 Git status/diff 及最近的提交。

使用 C:\Users\yhx\.conda\envs\guandan_train\python.exe。不得修改 D:\project\doudizhu 或 D:\project\FableDan。不得使用或复制 FableDan。保持项目为纯 Python。

找到第一个 accepted=false 的阶段，并且只继续推进该阶段。为子代理分配互不重叠的可写文件集合。检查每个子代理的差异，运行针对性测试，执行阶段验收命令，并运行完整测试套件。不得通过自动化将阶段标记为 accepted；只有在完成主代理审查和一次 Git 提交之后，才能更新 STATUS.yaml。不得重新开始已验收通过的阶段，也不得跳过未通过的验收门槛。仅在确实存在尚未解决的产品/规则决策或需要外部权限时提问。

报告阶段、文件、测试、证据、提交、阻塞问题，以及是否可以开始下一阶段。
```
