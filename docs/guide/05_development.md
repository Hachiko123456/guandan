# 05 · 开发指南：改哪里、测什么、怎样交接

[评估说明](04_evaluation.md) · [术语表](glossary.md) · [AI 接手入口](../AI_CONTEXT.md)

> 适用范围：当前源码维护与文档维护。代码核对：2026-10-21。这里是变更地图，不授予历史阶段重跑、规则重定义、正式训练或最终评估的权限。实际工作先看 [CURRENT_EXECUTION.md](../marvel_v14_prompts/CURRENT_EXECUTION.md)。

## 1. 开始改动之前

1. 确认任务是解释、修复、实验、正式训练还是验收，范围不要互相替代。
2. 读当前入口、工作区指令、相关专项规范；检查 Git status/diff 与未跟踪文件。
3. 记录改动前的代码/配置身份，分清原有 dirty-tree 与本任务新增变化。
4. 找到真实数据流和现有测试，不只读某个设计 Prompt 的“预期实现”。
5. 决定哪些兼容性必须保持，以及本次哪些验证不能运行。

不得为了获得干净工作区而 reset/clean/restore 他人文件；不擅自 commit/push，不清理与任务无关的输出、OpenCode/Claude 配置或 Guance handoff。

## 2. 按“想改什么”定位

| 任务 | 主要源码 | 应核对的契约/测试 | 典型风险 |
|---|---|---|---|
| 改牌与组合 | [cards.py](../../guandan/cards.py)、[combos.py](../../guandan/combos.py)、[C++ 规则](../../nanopokerlib/cpp/guandan/guandan.cpp) | [规则](../rules.md)、`tests/unit/test_cards.py` / `test_combos.py`、`tests/marvel_cpp/` | 双引擎规则漂移、地方规则混入 |
| 改轮次/贡还/终局奖励 | [round.py](../../guandan/round.py)、[state.py](../../guandan/state.py)、C++、[reward.py](../../guandan_marvel/reward.py) | `tests/unit/test_round.py`、`tests/marvel_training/test_phase4_reward_and_context.py` | 奖励坐标、公开信息、恢复身份失配 |
| 改协议/观测 | [action_state.py](../../guandan/action_state.py)、[game.py](../../guandan_marvel/game.py)、native bindings | [V2 接口](../marvel_migration/04_GAME_INTERFACE_V2.md)、`tests/marvel_adapter/` | 隐藏信息泄露、容量/版本不兼容 |
| 改模型/Fuse/AMP | [agent.py](../../guandan_marvel/model/agent.py)、[base.py](../../guandan_marvel/model/base.py)、[config.py](../../guandan_marvel/config.py) | `test_macro_fuse*.py`、`test_agent_active_head_optimization.py` | 旧权重迁移、漏进 optimizer、精度退化 |
| 改 PPO/GAE | [macro_ppo.py](../../guandan_marvel/training/macro_ppo.py)、[macro_gae.py](../../guandan_marvel/training/macro_gae.py)、[local.py](../../guandan_marvel/training/local.py) | `test_macro_actions*.py`、`test_macro_ppo.py`、`test_macro_gae.py`、legacy/recovery/DDP 测试 | 时钟、均值单位、padding、空 rank 错误 |
| 改采样/缓存 | [sampler.py](../../guandan_marvel/runtime/sampler.py)、[buffer.py](../../guandan_marvel/buffer.py) | sampler 契约、`test_training_repair.py`、`tests/marvel_cpp/test_repair_sampler.py` | 概率与行为不符、截断、缓存恢复不一致 |
| 改训练分布 | [context_schedule.py](../../guandan_marvel/context_schedule.py)、profile、Sampler | reward/context、tribute audit、manifest/provenance | 声明覆盖但未实际采样 |
| 改对手池 | [phase6](../../guandan_marvel/phase6)、Trainer、Sampler | `test_phase6_policy_owner.py`、`test_phase6_pool_masks.py`、pool identity/recovery/DDP | 对手梯度混入、冻结策略变化 |
| 改评估/统计/晋升 | [evaluation](../../guandan_marvel/evaluation) | `tests/marvel_evaluation/`、统计 schema 与隔离协议 | 把相关轮换当独立、误晋升、数据泄漏 |
| 改 Value qualification | [value_gate.py](../../guandan_marvel/phase6/value_gate.py)、qualification CLI | `test_phase6_value_gate*.py`、`test_phase6_qualification_planner.py` | all-tie 假信息、metadata 冒充原始证据 |
| 改交付/保存恢复 | [training/checkpoints.py](../../guandan_marvel/training/checkpoints.py)、[kaggle](../../guandan_marvel/kaggle) | backend identity、macro recovery、`tests/marvel_kaggle/`、phase4 delivery | 新源码接旧包、半轮状态、native 身份错配 |

表内 `test_*.py` 简称属于 [tests/marvel_training](../../tests/marvel_training)，除非显式写了其他目录。文件名中的 `*.py` 是检索模式，不是一份实际文件。

规则修复不等于“顺手改训练器”的授权。若专项规范冻结了规则语义，先提交最小反例、影响与独立修复范围，再决定是否扩大改动。

## 3. 测试怎样分层？

| 层次 | 要回答的问题 | 不能替代 |
|---|---|---|
| 纯规则/统计/张量单位测试 | 局部计算是否正确、边界是否覆盖 | 真实模型与完整环境集成 |
| 协议与双引擎差分 | Python/C++ 相同输入是否语义一致 | 真实 Transformer 训练吞吐 |
| 真实模型小 smoke | 能否采样、有效更新、保存恢复 | 正式模型尺寸和长时间稳定性 |
| DDP/AMP/故障恢复 | 全局归一化、精度、事务边界是否正确 | 其他设备/通信后端认证 |
| 完整回归/阶段验收 | 本次范围是否满足冻结标准 | 棋力提升 |
| 公平棋力评估 | 指定对照下是否更强 | 未覆盖场景的普遍能力 |

运行前阅读测试夹具：文件名叫 unit/smoke 不保证无模型或无 GPU。当前入口要求真实模型/probe 使用显式 CUDA，CPU 只用于允许的规则/统计/明确兼容测试；不要照搬历史 CPU 模型脚本作为当前认证。

本次文档任务只执行了 [操作手册](03_runbook.md) 列出的低风险检查，不运行表中整套测试。

## 4. 数学与边界变更检查单

改宏训练时至少考虑：

- [ ] token、宏动作、同玩家决策间隔、真实牌局是否区分。
- [ ] 终局奖励是否传到玩家最后一次决定；是否误读 padding/chance。
- [ ] 已选动作合法且概率匹配实际行为；新 log-prob 使用正确掩码。
- [ ] PPO 裁剪对整个序列一次执行，而不是逐 token 裁剪。
- [ ] 宏优势、价值与正则的样本单位一致；无样本 rank/全局归一化正确。
- [ ] 新分支参数进入 optimizer，冻结对手排除，必要的零梯度连接保留。
- [ ] AMP 的关键数学仍 FP32，不静默 clamp、吞掉 overflow 或 skip。
- [ ] 对更改设计构造旧实现会失败的反例，而不只是正常输入测试。

改变任何规则/观测/模型/奖励/采样/owner 语义时，都要评估协议和 checkpoint 兼容性；不能只改代码、不改身份或迁移规则。

## 5. 配置变更的检查单

- 配置类默认值、builder/profile、resolved config 三层分别核对。
- 新 feature 必须显式选择、校验不兼容组合；关闭时不能留下隐形行为变化。
- 训练分布、对手身份、采样权重和 seed 必须进入应有的 manifest/provenance。
- `total_iterations`、Gate 绝对目标、本次预算和实际完成计数分别记录。
- checkpoint 的源码/执行/native/semantic 身份不匹配要失败并解释，不偷删字段。
- 正式新交付应使用新路径与身份，不覆盖已冻结历史包。

## 6. 一次变更如何形成可信交付？

```text
范围与基线 → 最小反例 → 代码/文档修改 → 定向验证
    → 集成/回归（按获准成本） → 差异和身份核对 → 如实报告
```

报告至少包含：精确文件、行为变化、实际命令/退出码、未运行项目、遗留失败、产物路径与身份、能支持和不能支持的结论。未重跑的历史结果注明“读取既有报告”，不写成本次测试。

如果执行失败或证据不完整，用 `FAIL/INCONCLUSIVE/NOT_RUN` 等准确状态；不改容差、删测试、伪造局数或凭执行者总结跳过验收。阶段性协作方式还要遵循当前工作区适用指令，不能因文档中出现 Agent 字样就自行启动多代理。

## 7. 文档以后怎样维护？

### 内容放在哪里

| 内容 | 维护位置 |
|---|---|
| 当前阶段、下一步、条件状态 | 现有 `CURRENT_EXECUTION.md`；其他地方链接，不复制动态清单 |
| 架构与稳定机制 | 本套 guide，对应源码变更时同步核对 |
| 某次运行的参数与结论 | 该 run 的 manifest、resolved config、日志和报告 |
| 专项接口/数学契约 | 对应协议与阶段规范，按其冻结要求版本化修订 |
| 历史阶段/旧实验 | 保留原记录，新入口说明适用范围 |

不把旧报告悄悄改成当前结果。根 README 是更新后的导航入口；旧阶段目录索引和冻结记录仍保留，不用移动目录来“整理”。

### 每次指南变更怎么自查

- [ ] 核对日期/适用版本清楚，示例与实测分开。
- [ ] 新概念先给直觉，再给流程/例子，最后给公式与源码。
- [ ] 相对链接从本文位置能解析；函数名真实存在。
- [ ] 命令参数确实有入口，未验证命令显式标注。
- [ ] 高成本命令不放在无警告的一键脚本里。
- [ ] 公式、张量和样本单位与当前实现一致。
- [ ] 动态状态没有维护出多份互相漂移的版本。
- [ ] Mermaid 有文字版说明，不依赖插件才能理解。

## 8. 交给别人或另一大模型之前

1. **源码完整**：检查 tracked diff 与 untracked；`git diff` 不展示未跟踪文件的内容。
2. **只交必要文件**：不要擅自 add/commit/push 或把秘密、缓存、虚拟环境和无权分发的权重全部打包。
3. **文档完整**：README、guide、AI_CONTEXT、当前入口、相关专项规范一起提供。
4. **声明缺失证据**：`runs/`、checkpoint、日志、native 和部分验收产物被忽略，不随克隆出现。
5. **若需复现**：另给获准产物、依赖/平台说明、source/native/config/data/metric hash 清单。
6. **接收方先只读**：按 [AI_CONTEXT.md](../AI_CONTEXT.md) 验证现状，不自动重做历史阶段或启动训练。

本指南不建立第二份项目进度，也不承诺没有拿到原始工件的接收方能复核本地历史结论。
