# guandan · 掼蛋训练项目

独立实现的纯 Python 掼蛋训练项目。

- 规则和游戏引擎在本仓库中独立实现。
- 动作协议逐步构造 token，并显式提交。
- 训练目标：使用 PyTorch 进行 MARVEL 风格的策略/价值训练，然后在 Kaggle 上运行。
- FableDan 既不是运行时依赖，也不是源码依赖。

项目状态记录在 `project_status/STATUS.yaml` 中，验收证据保存在 `project_status/history/` 中。

## 仓库说明

本仓库包含独立实现的源码、测试、配置和阶段验收文档。
本地训练检查点、运行日志和 `project_status/history/` 下的验收产物不纳入 Git；
克隆仓库后，需要在自己的环境中重新生成这些产物。

A05/A06 的本机就绪状态不等于远端完整验收，具体进度与待验证事项以
`project_status/STATUS.yaml` 为准。
