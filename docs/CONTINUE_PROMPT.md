# 续接提示词

```text
继续执行 D:\project\guandan 的总体执行计划。

先读取 project_status/STATUS.yaml、docs/MASTER_EXECUTION_PLAN.md、docs/AGENT_EXECUTION_PROTOCOL.md、docs/completion_criteria.md、docs/acceptance/A05_training_integration.md、docs/acceptance/A06_evaluation.md，并查看 Git status/diff 及最近提交。A00-A04 已有要求和 accepted 状态必须保持不变；先找到第一个 accepted=false 的阶段，当前应从 A05 开始，不得跳阶段。

使用 C:\Users\yhx\.conda\envs\guandan_train\python.exe 做本机验证。不得修改 D:\project\doudizhu 或 D:\project\FableDan，不得使用或复制 FableDan，保持项目为纯 Python。未经主代理审查不得采纳子代理结论；子代理不得编辑 project_status/STATUS.yaml、不得设置 accepted=true、不得扩大文件范围。

先实现并审查 profiles-0.2 的 acceptance profile 契约：local_fast 为 IPPO/VRPO 各 5 次更新、4 个环境、每次 256 个跨环境聚合 token steps（每环境 64 ticks，合计 1280 steps/算法）、每 5 次 checkpoint、恢复后追加 1 次更新、0.5 小时；remote_full 为各 100 次更新、8 个环境、每次 1024 个跨环境聚合 token steps、每 10 次 checkpoint、恢复后追加 2 次更新、12 小时。A06 为 local_fast 4 个 deal groups × 座位轮换 [0,1,2,3] = 每个 random/rule/snapshot 对手 16 局；remote_full 为 250 × 4 = 每个对手 1000 局。局数必须是实际完成局数，超时不足为 incomplete；不作任何强度保证。

Runner 的 --profile 只负责校验/传播/证据编排，必须向测试转发 canonical GUANDAN_PROFILE 和 GUANDAN_RESOLVED_PROFILE_JSON；--show-profile 只打印配置、不跑测试、不授予验收。非法 profile/config 必须失败；硬件不能自动缩放，只有显式 remote_full 才能运行远端规模。A05/A06 验收测试必须实际加载配置、使用配置计数并提供实际执行证据，不能用默认值、计划值或伪造报告。

A05 必须审查 actor/critic 私有信息边界、玩家/队伍视角、token/committed step、truncation/reset/GAE 边界、真实 terminal reward、动作掩码、有限 loss/gradient 和 checkpoint 恢复。mean-pooled MLP 必须如实命名，不能声称原始 MARVEL Transformer；VRPO 不能只是改名后的 PPO relabel。A06 必须实际完成四个座位轮换和三类对手，错误不能静默回退。local_fast 通过只可能形成 local_ready，不等于 accepted 或 remote_full 通过。

执行顺序为：scoped 完整 local_fast A05 + 目标依赖测试 → 主代理逐项审查并记录 local_ready → local_fast A06 + 目标依赖测试 → 主代理逐项审查 → 继续 A07 打包准备。每次小改动不必重复 A02 的 10,000 seeds；远端/release 再做完整回归，但不得删除安全测试。A07 的 accepted 必须依赖真实全新 Kaggle 会话；若无 Kaggle 权限，只交付 ready package 并明确阻塞，不用本机替代。

每个子代理完成后立即向主代理报告：精确修改文件和行为、运行命令与实际结果、证据路径、假设、已知问题和未覆盖项。主代理对每轮执行差异审查 → 针对性测试 → 验收测试/证据审查 → 需求审查；失败则给出精确失败用例、违反需求、允许文件和复测命令，退回同一子代理修复。最后再决定是否更新 STATUS.yaml；只有 remote_full 实际证据、主代理审查、完整回归和提交都具备时才可考虑 accepted=true。
```
