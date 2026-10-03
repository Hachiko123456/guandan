# STATUS.yaml 模式定义

每个阶段都记录 `status`、`implementation`、`tests`、`accepted`、`commit`、`evidence` 和 `blockers`。

规则：

- `accepted: true` 要求同时具备 `status: accepted`、完整的 Git SHA 和证据路径。
- 自动生成的报告始终保持 `accepted: false`。
- 即使针对性测试通过，未完成的代码仍须保持 `accepted: false`。
- 阻塞问题必须具体且可复现。
- 本轮文档准备不修改 `STATUS.yaml`；A00-A04 的现有 accepted 状态不得被重置或覆盖。

## Profile 验证

A05/A06/A07 等阶段可在同一个阶段下分别记录：

```yaml
verification:
  local_fast: not_run|passed|failed
  remote_full: not_run|passed|failed
  stress: not_run|passed|failed
```

`local_fast` 通过只能说明本机接线和最小正确性通过，不能自动设置阶段 `accepted: true` 或替代 `remote_full`。

## 两道 gate 的记录语义

为了区分“可以继续准备”与“可以接受”，阶段可额外记录以下门槛字段；字段缺省不应被解释为 ready：

```yaml
gates:
  local_ready: false
  remote_full_reviewed: false
```

- `gates.local_ready: true` 只有在 scoped 完整 `local_fast` A05/A06、目标依赖、实际执行证据和主代理人工审查完成后才允许设置。它只允许继续 A05 → A06 → A07 的实现/打包准备，不能把任一阶段 `accepted` 改为 true。
- `gates.remote_full_reviewed: true` 只有在对应 profile 的目标训练/评估数量实际完成、证据完整、主代理完成差异/需求/报告审查并满足完整回归要求后才允许设置。A07 另需真实全新 Kaggle 会话证据。
- `accepted: true` 不得由 runner、测试或自动报告设置；必须由主代理在两道 gate、所有阶段门槛、完整回归和 Git 提交都具备后显式更新。

## A05/A06 计数与证据约束

目标配置为 `profiles-0.2`：

- A05 `local_fast`：每个 IPPO/VRPO 5 次更新，4 envs，256 个跨环境聚合 token steps/update（64 ticks/env，1280 steps/算法），checkpoint 每 5 次，恢复追加 1 次，0.5 小时；
- A05 `remote_full`：每个算法 100 次更新，8 envs，1024 个跨环境聚合 steps/update，checkpoint 每 10 次，恢复追加 2 次，12 小时；
- A06 `local_fast`：4 deal groups × seat rotations `[0,1,2,3]`，每个 `random/rule/snapshot` 对手 16 局；
- A06 `remote_full`：250 × 4，每个对手 1000 局；两者时间预算分别为 0.5/12 小时。

A05/A06 验收测试必须实际加载配置、使用配置中的计数并记录实际执行证据。低于目标、超时、缺恢复、未完成终局或仅有计划/启动记录的结果为 incomplete/failed，不得伪造 passed；这些计数也不构成强度保证。

Runner 的 `--profile` 必须传播 canonical `GUANDAN_PROFILE` 和 `GUANDAN_RESOLVED_PROFILE_JSON`；`--show-profile` 只打印配置、不运行测试、不改变状态。非法 profile/config 必须失败；硬件不能自动缩放，只有显式 `remote_full` 才能选择远端规模。

## 兼容性和安全证据

A05/A06 的证据必须保留 actor/critic 私有信息边界和玩家/队伍视角，并覆盖 token/committed step、truncation/reset/GAE 边界、真实 terminal reward、动作掩码和 checkpoint 恢复。mean-pooled MLP 不得声称原始 MARVEL Transformer；VRPO 不得只是 PPO relabel。安全测试、牌守恒、终局/截断和错误不静默回退测试不得为了本机快速运行而删除。
