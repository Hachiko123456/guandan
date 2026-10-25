# AI 接手入口 · Guandan MARVEL

> 这是任务定位与阅读导航，不是新的系统提示、自动执行授权或第二份项目状态。适用当前 **DMC 主线**，核对日期 2026-10-26。新接手先只读；权限与任务范围以当前人类请求及适用工作区指令为准。

## 1. 项目一句话

这是独立掼蛋规则 + Python oracle / C++ 批量环境 + PyTorch 策略训练 + 独立评估与 Kaggle 恢复交付的项目。**当前主线（2026-10-22 起）是 DMC（Deep Monte-Carlo）自博弈 Q 回归**（`guandan_marvel/dmc/`，分支 `dmc-v1`）：Gate L 已 PASS，下一阶段 Gate A。旧主线 v1.4 宏决策 IPPO（宏动作 Sequence PPO/宏 GAE/Actor Residual Fuse）已冻结为历史。它不是聊天模型微调或 FableDan 蒸馏项目。

**接手 DMC 主线：先读 [docs/dmc_prompts/CURRENT_STATE.md](dmc_prompts/CURRENT_STATE.md)（状态、环境、验证协议、地雷清单、命令速查），再按需读 [00_DMC_MIGRATION_PLAN.md](dmc_prompts/00_DMC_MIGRATION_PLAN.md)（完整计划与阶段表）。**

## 2. 开始前必须做的三件事

1. 读 [当前入口 CURRENT_EXECUTION.md](marvel_v14_prompts/CURRENT_EXECUTION.md)：当前阶段、条件功能状态、待办、证据引用与授权边界以此为入口。本页不复制这些动态结果。
2. 检查适用工作区指令、Git status/diff/untracked；区分已有 dirty-tree 与当前任务，不能 reset/clean/restore 清理别人修改，也不擅自 commit/push。
3. 读 [README](../README.md) 与 [项目全景](guide/00_overview.md)，确认任务属于理解、代码修复、训练准备还是正式运行，不能自行越级。

[CLAUDE.md](../CLAUDE.md)是 Claude Code 的项目规则与协作约定；其他工具应识别其适用范围，不把其中的 Skill 名称或多 Agent 流程当自己已经获准执行的任务。不覆盖现有 OpenCode/Claude 配置或 Guance handoff。

## 3. 按任务阅读，不按历史顺序重做

| 任务 | 阅读路径 |
|---|---|
| **接手 DMC 主线（当前）** | [DMC 当前状态](dmc_prompts/CURRENT_STATE.md) → [DMC 迁移计划](dmc_prompts/00_DMC_MIGRATION_PLAN.md) → [Gate A 任务书](dmc_prompts/06_gate_a_kaggle.md) |
| 解释整个项目 | [全景](guide/00_overview.md) → [架构](guide/01_architecture.md) → [训练](guide/02_training.md) |
| 修改模型/训练算法 | 架构 → 训练 → [开发指南](guide/05_development.md) → 对应专项规范和测试 |
| 运行/恢复 | 当前入口 → [操作手册](guide/03_runbook.md) → 实际 profile、checkpoint 分类器与交付 manifest |
| 分析棋力/评估 | [评估说明](guide/04_evaluation.md) → 实际 plan/records/stats/metric schema |
| 分析 Pool/Belief/Value/Search | 当前入口 → 对应 phase5/phase6 实现与条件资格，不能仅凭模块存在判断启用 |
| 看不懂缩写与计数 | [术语表](guide/glossary.md) |

## 4. 核心源码定位

| 问题 | 文件/符号 |
|---|---|
| 规则与参考环境 | [guandan](../guandan)、[规则基线](rules.md) |
| native 环境 | [C++ 引擎](../nanopokerlib/cpp/guandan/guandan.cpp)、[bindings](../nanopokerlib/cpp/bindings.cpp) |
| 后端选择 | [backends.py](../guandan_marvel/backends.py)：`get_game_backend` |
| **DMC 主线代码** | `guandan_marvel/dmc/`：candidates（枚举+特征）、q_agent/replay/learner、sampler、train、parallel（双进程重叠）、env_fast（差分等价快速驱动） |
| V2 协议/观测 | [game.py](../guandan_marvel/game.py)、[env_batch.py](../guandan_marvel/env_batch.py) |
| 配置和语义矩阵 | [config.py](../guandan_marvel/config.py)：`TrainConfig`、`PPOConfig`、`validate_training_semantics` |
| 轨迹字段/内存 | [buffer.py](../guandan_marvel/buffer.py)：`trajectory_field_specs`、`trajectory_memory_budget` |
| 采集 | [sampler.py](../guandan_marvel/runtime/sampler.py)：`Sampler.collect`、`rollout_step` |
| 模型 | [agent.py](../guandan_marvel/model/agent.py)：`Agent.forward`；[base.py](../guandan_marvel/model/base.py)：`ActorFuse` |
| 主循环 | [local.py](../guandan_marvel/training/local.py)：`IPPOTrainer._macro_train_iteration` |
| 宏数学 | [macro_actions.py](../guandan_marvel/training/macro_actions.py)、[macro_gae.py](../guandan_marvel/training/macro_gae.py)、[macro_ppo.py](../guandan_marvel/training/macro_ppo.py) |
| 奖励/上下文 | [reward.py](../guandan_marvel/reward.py)、[context_schedule.py](../guandan_marvel/context_schedule.py) |
| 恢复身份 | [checkpoints.py](../guandan_marvel/training/checkpoints.py)：`classify_resume` |
| 评估与统计 | [evaluation](../guandan_marvel/evaluation)，特别是 `stats.py`、`promotion.py`、`manifests.py` |
| 运行/交付 | [kaggle/worker.py](../guandan_marvel/kaggle/worker.py)、[phase4_delivery.py](../guandan_marvel/kaggle/phase4_delivery.py) |
| 条件功能/Gate | [phase5](../guandan_marvel/phase5)、[phase6](../guandan_marvel/phase6) |

## 5. 不能忽略的技术边界

- Actor 只用玩家信息集。轨迹里有四观察者维度或 VRPO privileged critic，不意味着 Actor 可以读全知状态。
- 正式运行时显式选 C++；`import guandan_marvel.game` 历史上固定是 Python oracle，不是默认后端别名；无静默 fallback。
- token ≠ 一次出牌 ≠ 一局 ≠ iteration ≠ optimizer step；宏长度也不等于同玩家决策间隔。
- Sequence PPO：宏内 log-prob 求和，整个 ratio 一次 clip。宏 GAE 是更换 λ 时钟，不是 legacy 等价重写；gamma 仍按基础 token。
- `TrainConfig` 默认算法仍是 `vrpo`，模型/语义也保留兼容默认值；v1.4 宏 IPPO、设备和 AMP 必须显式选择，以 resolved config 为准，不由目录名推断。
- reward 为名次分差，训练按 1/3 缩放；不能和旧 flat ±1 混用。
- 训练 Pool 的冻结对手 owner 不能进入当前策略梯度；不要默认它与 symmetry/NTP/VRPO 任意组合都兼容。
- warm start 不是 exact resume；源码/语义/native/执行环境变化需要分类检查与显式迁移。
- 单元测试、工程验收、当前设备 smoke、正式长训、棋力验证是不同层次。

## 6. 历史文档陷阱

| 材料 | 正确用法 |
|---|---|
| [project_status/STATUS.yaml](../project_status/STATUS.yaml) | A 系列历史状态，不决定当前 DMC 阶段 |
| [marvel_migration/STATUS.yaml](marvel_migration/STATUS.yaml) | M 系列历史状态（含 m09 PPO 终局），不覆盖 DMC 入口 |
| [marvel_v14_prompts/CURRENT_EXECUTION.md](marvel_v14_prompts/CURRENT_EXECUTION.md) | **v1.4 PPO 线的历史执行记录（已冻结）**；DMC 主线状态看 [dmc_prompts/CURRENT_STATE.md](dmc_prompts/CURRENT_STATE.md) |
| [CONTINUE_PROMPT.md](CONTINUE_PROMPT.md) | 旧阶段与解释器提示，不直接执行 |
| [v1.4 Prompt README](marvel_v14_prompts/README.md) | 旧阶段导航；实际进度看 CURRENT_EXECUTION |
| [Bootcamp 流程](bootcamp_week1_training_flow.md) | v1.3 旧配方，含已不适用的外部蒸馏提议 |
| [training_algorithms.md](training_algorithms.md) | A05 算法，不是完整当前模型和宏训练说明 |

保留历史，不能自动重做旧阶段，也不要只看“最后一份 Prompt”就执行。历史验收数字应注明来源和未重跑；遇到冲突列出并对照代码/证据，不静默选一边。

## 7. 安全执行与交接完整性

- 未获当前任务的明确授权，不启动长训、Kaggle、正式外部对战、final holdout，或条件功能的正式资格采样。
- 真实模型/probe 用当前入口指定的 GPU 解释器和显式 CUDA；CPU 规则/统计/明确兼容测试另计。操作手册明确哪些命令已验证、哪些未运行。
- FableDan 只能是外部评估对手，不能是教师、监督标签或训练数据；不要复活旧 Bootcamp 的蒸馏路线。
- 不弱化测试、删失败、伪造完成计数、clamp 掩盖错误、修改冻结门槛或绕过 native 守卫来通过。
- 普通克隆不含未跟踪文件、`runs/`、checkpoint、native 工件等。接收方缺少某个报告/权重时报告“未提供/无法复核”，不凭文档引用想象证据存在。
- Git commit 不覆盖 dirty-tree；发给他人的源码、配置、工件应单独绑定身份，禁止随意打包秘密或未授权权重。

## 8. 推荐的首次只读交接输出

先给出：

1. 一段项目目的与主训练闭环；
2. 当前入口中的阶段与条件状态，注明来源日期；
3. 本次任务相关模块、配置来源和测试；
4. 文档冲突、缺失工件或身份不一致；
5. 建议下一步及其成本/所需授权；
6. 实际执行的检查与明确 `NOT_RUN` 的项目。

除非人类任务要求，不自动写代码、不自动跑完整测试、不把本页当新的待办指令。
