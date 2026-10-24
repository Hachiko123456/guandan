# 掼蛋强化学习项目 · Guandan MARVEL

这是一个训练掼蛋策略模型的项目：**让模型在游戏环境里自我对弈，根据牌局结果学习如何出牌，再用独立评估检查它是否真的变强。** 它不是聊天大模型，也不是把外部高手的答案直接灌给模型。

项目包含独立的掼蛋规则实现、Python 参考环境、C++ 批量环境、PyTorch 模型、训练与评估系统，以及 Kaggle 分段运行和恢复工具。当前阅读主线是 **MARVEL v1.4 宏决策 IPPO**；旧 A 系列、M 系列和 Bootcamp 材料保留用于历史追溯与兼容，不应直接拼成当前运行方案。

## 从哪里开始？

| 你的目的 | 推荐入口 |
|---|---|
| 第一次接触，先看懂项目 | [项目全景](docs/guide/00_overview.md) |
| 想知道一局牌如何经过代码各层 | [架构与数据流](docs/guide/01_architecture.md) |
| 想理解模型为什么能学会、训练公式是什么 | [训练原理与一次迭代](docs/guide/02_training.md) |
| 想检查环境、了解训练与恢复操作 | [操作手册](docs/guide/03_runbook.md) |
| 下次新包自动上传 Kaggle 并运行，只填一次账号 | [Kaggle 自动交付配置](docs/kaggle_pipeline.md) |
| **当前：iter375 同集复测，只评估、零训练** | [评估去重、双卡均衡与上传说明](docs/t4x2_recheck375.md) |
| 已完成的250→500续训与历史操作 | [250→500 安全续训与边训边评](docs/t4x2_continue500.md) |
| 查阅旧新三参数组合 | [三参数组合与双卡提速实验](docs/t4x2_combo_pilot.md) |
| 查阅上一轮 50 次短训实验 | [原 T4×2 实验说明](docs/t4x2_learning_pilot.md) |
| 想判断模型是否真的更强 | [评估说明](docs/guide/04_evaluation.md) |
| 想修改代码或补测试 | [开发指南](docs/guide/05_development.md) |
| 遇到 token、GAE、Gate 等术语 | [术语与计数单位](docs/guide/glossary.md) |
| 让另一个大模型接手 | [AI 接手入口](docs/AI_CONTEXT.md) |

## 整个闭环

```text
配置与训练分布 → 环境发牌 → 模型按合法动作自我对弈
    → 完整牌局轨迹 → 宏动作/奖励/优势目标
    → PPO 与价值学习 → checkpoint → 独立评估 → 决定是否继续
```

- 策略只使用本座位可见的信息，不把全知状态作为输入。
- 正式训练运行时使用显式 C++ 后端；Python 保留为参考环境，不允许悄悄降级替代。
- 一个协议 token 不等于一次出牌，一次 iteration 也不等于一局牌，详见[计数说明](docs/guide/glossary.md)。
- FableDan 仅作为外部评估对手，不是训练依赖、蒸馏教师、监督标签或训练数据来源。

## 当前做到哪里？

**当前阶段、条件功能状态、待办和授权边界统一查看 [CURRENT_EXECUTION.md](docs/marvel_v14_prompts/CURRENT_EXECUTION.md)。** 本 README 和指南解释机制，不另建一份动态状态表。

请区分：**代码实现 → 工程验收 → 目标设备运行 → 正式训练 → 棋力验证**。前一项通过，不能自动声称后一项完成。未经另行授权，不启动长训、Kaggle 作业、正式外部对战或最终 holdout。

## 最低风险的开始方式

在本机 PowerShell 中先做只读检查：

```powershell
Set-Location 'D:\project\guandan'
git status --short
Get-Content -LiteralPath 'D:\project\guandan\docs\marvel_v14_prompts\CURRENT_EXECUTION.md'
& 'D:\project\guandan\.venv-gpu\Scripts\python.exe' --version
```

以上使用本工作区路径；克隆到其他位置时应改为自己的绝对路径。安装、C++ 编译、CPU 规则测试、GPU 验证和正式训练不是同一步，具体风险与验证状态见[操作手册](docs/guide/03_runbook.md)。

## 交接与历史材料

- [项目全景中的文档地图](docs/guide/00_overview.md)解释哪些是当前指南、专项规范和历史记录。
- [旧项目状态](project_status/STATUS.yaml)对应早期 A 系列；[迁移状态](docs/marvel_migration/STATUS.yaml)对应 M 系列，均不能替代当前 v1.4 入口。
- 本地 `runs/`、checkpoint、部分日志、native 扩展和历史验收产物不纳入 Git；未跟踪的新代码和文档也不会自动出现在另一份克隆里。**交接工作区前必须核对源码、配置、native 构建身份和所需产物。** 不要提交秘密或为交接直接打包整个虚拟环境。
- MARVEL 通用核心的来源与许可说明见 [guandan_marvel/NOTICE.md](guandan_marvel/NOTICE.md)；native 部分见 [nanopokerlib/NOTICE.md](nanopokerlib/NOTICE.md)。

文档核对日期：2026-10-21。具体运行结论以带版本和来源的证据为准。
