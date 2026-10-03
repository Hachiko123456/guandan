# A05 训练集成

## 范围

把 A04 环境接入 IPPO/VRPO 训练流水线。本阶段同时支持本机快速 profile 和 Kaggle/远端完整 profile，但本机快速通过只证明接线和最小正确性，不证明模型能力，也不等价于远端完整验收。

## Profile 契约

配置文件的实现目标为 `configs/acceptance_profiles.json` 的 `profiles-0.2`。本轮文档更新不修改配置文件本身；若当前文件仍是 `profiles-0.1`，主代理必须先实现/审查迁移，不能把旧字段或默认值视为满足本验收契约。

### `local_fast`

- `ippo`、`vrpo` 各完成 5 次更新。
- 使用 4 个环境。
- `token_steps_per_update=256` 是跨环境聚合总量，即每环境每次更新 64 ticks，而不是每个环境 256 steps。
- 每个算法 5 次更新合计 1280 token steps。
- 第 5 次更新保存 checkpoint；实际加载后再完成 1 次恢复更新。
- 时间预算为 0.5 小时；超时且未达到目标为 `incomplete`，不得假通过或自动缩放目标。

### `remote_full`

- `ippo`、`vrpo` 各完成 100 次更新。
- 使用 8 个环境。
- `token_steps_per_update=1024` 是跨环境聚合总量，即每环境每次更新 128 ticks。
- 每 10 次更新保存 checkpoint；实际恢复后再完成 2 次更新。
- 时间预算为 12 小时；只有显式 `--profile remote_full` 才能执行此规模。
- 报告必须记录实际 GPU/设备、Python、PyTorch、profile、Git commit、实际更新数、环境数、聚合 steps、checkpoint 和恢复证据。

这些数量是执行和证据门槛，不是胜率、强度、泛化或排名保证。

## Runner 合约

`run_acceptance.py` 只负责配置解析、profile 选择、环境变量传播、测试编排和 write-once 证据；它不实现 trainer。`--profile <name>` 必须：

1. 加载并校验 `profiles-0.2`；
2. 将规范 profile 名称写入 `GUANDAN_PROFILE`；
3. 将解析后的完整 canonical JSON 写入 `GUANDAN_RESOLVED_PROFILE_JSON`；
4. 让 A05 测试实际读取这两个 canonical 输入，并按其中计数执行/记录。

`--show-profile --profile <name>` 只打印配置并退出，不运行 tests，不创建验收通过结论，不修改 `STATUS.yaml`。未知 profile、配置缺失、类型/范围错误、版本不匹配或无效 JSON 都必须失败退出；不得回退到默认 profile。硬件检测只能报告不兼容或能力，不能因硬件自动改变 env 数、更新数、steps、checkpoint 或时间预算。

## 必须实际验证的正确性

每个 profile 都必须提供实际执行证据，而不是只验证对象能构造：

- loss、gradient 和必要的数值统计为有限值；
- 合法动作概率归一化，非法动作概率为零；
- actor 不获得对手私有牌；critic 的训练专用信息若存在，必须与 actor 输入隔离；
- 玩家视角与队伍视角保持正确，不能不小心使用全知或错误玩家视角；
- token step、committed game step、`done`、`truncated`、reset 与 GAE/mask 的边界分别测试；截断不得伪造真实终局奖励，真实终局奖励必须来自实际 terminal；
- checkpoint 保存、加载、兼容性校验和恢复后实际继续更新均通过；规则、动作、环境、编码或模型版本不兼容时拒绝加载；
- 如果实现是 mean-pooled MLP，必须如实记录为 mean-pooled MLP，不能宣称为原始 MARVEL Transformer；
- VRPO 必须具有可审查的独立目标/更新语义，不能只是将 PPO 重新命名或 relabel。

## 证据与不完整处理

报告至少记录 profile、算法、目标/实际更新数、env 数、每次聚合 token steps、实际总 steps、checkpoint/resume 事件、运行时长、解释器/库/设备、seed、Git commit 和失败/截断原因。任何低于 profile 目标的更新数、steps、恢复更新或超时都只能是 `incomplete`/`failed`；不得用计划值、启动次数、部分输出或硬件降级伪造 passed。

## 命令

```powershell
python scripts/run_acceptance.py --stage A05 --profile local_fast
python scripts/run_acceptance.py --stage A05 --profile remote_full
python scripts/run_acceptance.py --stage A05 --profile local_fast --show-profile
```

第一条命令用于 `local_ready` 前的本机完整 scoped 验证，第二条才是完整稳定性门槛；第三条只检查配置打印，不是验收运行。

## 审查与状态

自动报告不得设置监督层的 `accepted: true`。主代理必须检查差异、检查实际证据、运行 profile 对应测试和目标依赖测试，并按项目规则安排完整回归。A05 通过 local_fast 后最多进入 `local_ready`，仍保持 `accepted: false`；只有 remote_full 实际证据、主代理需求/差异审查、完整回归和提交都具备时，才可考虑接受。A05 的这些文档要求不改变 A00-A04 已接受状态。


## 主代理 local_fast 证据

本机 `local_fast` 已按 `profiles-0.2` 实际运行 IPPO/VRPO：每算法 5 次更新、4
个真实 108-card 环境、每次 256 aggregate token steps、合计 1280 token steps；
update 5 checkpoint 后各恢复并追加 1 次更新。报告记录真实终局快照、GAE/Q-boost
目标类型、动作掩码、玩家/队伍视角、terminal/truncated/reset 边界、有限梯度以及
resume digest。

证据路径：`project_status/history/20261003T232220411107Z_9c0ae0690d5d491088a43fbf02dc9039/A05/report.json`。

本机 local_ready 不等于远端完整验收；remote_full 未运行，`accepted` 必须保持 false。
本轮另有前置规则缺陷审计：`docs/prerequisite_defects_A05_A06.md`。
