# 主代理/子代理执行协议

## 主代理

主代理负责需求、集成、Git 和验收。对于每一份子代理结果，主代理必须检查具体差异，拒绝超出范围的文件改动，运行针对性测试，执行阶段验收命令，检查实际执行证据，运行适用的完整回归，并对照需求进行审查；完成这些步骤后才能更新 `STATUS.yaml`。

A05/A06 采用 review loop：每轮都按“差异审查 → scoped 测试 → profile 验收/实际计数审查 → 需求与信息边界审查 → 记录缺口”执行。失败时，主代理必须把失败测试、违反的需求、预期行为、允许修改的文件和准确复测命令退回；修复后重新走同一轮。主代理必须及时向上层报告精确修改、已知问题、证据路径和是否达到 gate，不得用笼统的“已完成”替代。

主代理还必须维护两道门槛的区别：

- `local_ready` 只表示本机 scoped `local_fast` A05/A06 测试、目标依赖、实际证据和人工审查完成；它可以开启后续 A05 → A06 → A07 打包准备，但不能设置 `accepted: true`。
- `remote_full` 需要实际达到 profile 的训练/评估数量并通过主代理审查；只有它以及完整回归、提交等条件满足后，才可考虑阶段接受。A07 仍需真实全新 Kaggle 会话；没有访问权限时停止在 ready package。

本机每次 A05/A06 变更运行 scoped 的完整 `local_fast` 与目标依赖即可，不要求每次都重跑 A02 的 10,000 seeds；远端/release 再运行完整回归。不得为提速删除或弱化安全、信息隔离、终局/截断、reset、GAE 边界、牌守恒和错误处理测试。

## 子代理

- 使用互不重叠的可写文件集合。
- 不得编辑 `project_status/STATUS.yaml`，也不得将阶段标记为已验收通过。
- 不得扩大工作范围；不得修改用户未授权的仓库或文件。
- 报告具体的文件、命令、测试、实际计数/证据、假设及尚存缺口。
- 完成后立即向主代理报告 exact changes、known issues、未覆盖项和下一步建议；不等待主代理猜测差异。
- 不得使用或复制 FableDan。
- 不得把 mean-pooled MLP 说成原始 MARVEL Transformer；不得把 VRPO 实现成仅改名的 PPO relabel。
- 必须保留 actor/critic 的私有信息边界、玩家/队伍视角、token/committed step、truncation/reset/GAE 边界和真实 terminal reward 语义。

## 当前 A05/A06 执行契约

目标配置为 `profiles-0.2`：

- `local_fast`：IPPO/VRPO 各 5 updates，4 envs，`token_steps_per_update=256` 为跨环境聚合总量（64 ticks/环境，1280 steps/算法），checkpoint 每 5 次，恢复追加 1 次，0.5 小时；
- `remote_full`：各 100 updates，8 envs，跨环境聚合 1024 steps/update，checkpoint 每 10 次，恢复追加 2 次，12 小时；
- A06 `local_fast`：4 deal groups × `[0,1,2,3]` = 每个 `random/rule/snapshot` 对手 16 局；`remote_full`：250 × 4 = 每个对手 1000 局；时间不足即 incomplete，无强度保证。

`--profile` 必须把 canonical `GUANDAN_PROFILE` 与 `GUANDAN_RESOLVED_PROFILE_JSON` 传给测试。`--show-profile` 只能打印配置，不跑测试、不授予验收。非法 profile/config 必须失败；硬件不得触发自动缩放，只有显式 `remote_full` 才能选择远端规模。验收测试必须实际加载配置、使用计数并提供实际执行证据。

## 状态含义

- `not_started`：尚无已验收通过的实现；
- `in_progress`：正在实现/审查；
- `verified`：实现/测试已通过，等待主代理审查；
- `accepted`：所有验收门槛及主代理审查均已通过；
- `blocked`：存在具体且反复出现的阻塞问题；
- `rejected`：经审查，实现未达到验收门槛。

pytest 通过不等于监督验收通过。 主代理必须在验收前运行完整测试套件。`local_fast` 通过不等于 `local_ready`，`local_ready` 也不等于 `accepted`；`accepted` 还不能由自动生成报告自行设置。

## 失败处理协议

退回工作时，主代理应提供失败的测试、违反的需求、预期行为、允许修改的文件，以及准确的复测命令。在可行的情况下，继续由同一子代理处理。任何超时、目标计数不足、只有计划没有实际证据或实际行为被硬件降级的运行，都必须按 incomplete/failed 处理，不能补写为通过。
