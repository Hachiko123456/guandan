# M06 集中补修协议

- Stage：`M06_evaluation_champion_repair`
- 日期：2026-10-10
- 原实现：`10cb3452f69cd23a3bed848498e4353dcfbe3b9c`
- 原登记：`a46204c756802541c36d49f90ab922cc1f4b6fb8`
- 修复目标：评估安全、身份绑定、真实数据隔离、final 状态机和保守统计；不以模型强度验收

## 1. 异步评估隔离

`TrainingEvaluatorHook` 升级为 `guandan-marvel-training-evaluator-hook-v2`：

- 只使用 `spawn` 进程执行评估，禁止线程内构造模型；
- worker 只接收 immutable snapshot path/SHA、iteration、versioned entrypoint、JSON config 和产物路径；
- 不接收 trainer、optimizer、sampler、replay 或训练 RNG；
- 每个任务保存 `work.json`、`worker_status.json`、`result.json`；父进程追加 fsync journal；
- successful/failed/interrupted/cancelled 先进入 durable result，再允许 delivery；
- 下一次调度会 harvest 已完成 Future，不删除未交付结果；
- duplicate iteration 明确 `duplicate_rejected`；pending 超限明确 `skipped_pending`；
- close(wait=false) 只记录真实 cancelled/shutdown_pending，不伪造 completed。

Trainer 每个 durable iteration 自动调用 `collect_ready()` 并写入 metrics；训练结束通过 `close_evaluation_hook()` 收集尾部结果。Worker 失败不 poison 已完成训练更新。

## 2. Final holdout 原子状态机

Schema：`guandan-marvel-final-holdout-audit-v2`。

状态：

```text
unused -> reserved -> running -> consumed
                         |-> failed
                         |-> interrupted
```

- holdout 身份使用 final dataset hash，不依赖 `created_at` 或 run ID；
- reservation 原子绑定 dataset、candidate SHA、plan identity、config hash、evaluation_id；
- 使用 O_EXCL lock file 串行化并发请求；
- reserve 在 deal materialization、模型打分和输出覆盖检查之前执行；
- consumed/failed/interrupted 都不自动恢复 unused；当前实现不支持续跑；
- candidate mismatch、重复/并发请求在 0 次 game callback 时拒绝；
- final 输出已存在时拒绝覆盖，audit 记录 failed。

## 3. Promotion 和 pool 事务边界

Schemas：

- promotion policy：`guandan-marvel-promotion-policy-v2`
- promotion decision：`guandan-marvel-promotion-decision-v2`
- champion manifest：`guandan-marvel-champion-manifest-v2`
- opponent pool：`guandan-marvel-opponent-pool-v2`
- transaction：`guandan-marvel-promotion-transaction-v1`

Champion 更新前验证：decision schema/hash、status/promote 一致、candidate identity 精确相等、checkpoint 文件内容、algorithm/architecture/protocol/backend/build identity、当前 champion manifest hash/generation、selection mode、plan/summary/evaluation/deals identity。

晋升事务：prepared -> champion published/pool pending -> completed。Pool 失败时 champion manifest 保留 `pool_sync=pending`，transaction 标记 `partial_failure`，run 不得报告 complete；后续可依据 transaction ID 恢复。

Checkpoint cache 使用 path + 当前文件 SHA；同路径内容替换会重新加载或拒绝身份。

## 4. 实际训练来源

Training checkpoint 升级为 `guandan-marvel-training-checkpoint-v5`；sampler state 升级为 `guandan-marvel-sampler-state-v2`；semantic contract 升级为 v4。

Checkpoint 保存 `guandan-marvel-training-data-provenance-v1`：

- domain=`training`；
- source manifest hash；
- generator version；
- initial EnvBatch seed；
- 明确连续 PRNG stream 语义；
- actual deal SHA-256 集合和记录 hash；
- deal count 和完整性状态。

Training seed 不是“每局 seed”。只有初始 reset 使用 seed+slot；后续 auto-reset 来自每个 slot 的连续 PRNG 状态。Selection/final 在任何 policy forward 前 materialize 固定 deal identity，并同时检查已知 initial seed 和 actual deal hashes。

旧 v4 checkpoint 不伪造 provenance：

- training resume：migration；
- selection：仅 `smoke_only`，禁止正式晋升；
- final：拒绝；
- 即使 smoke-only，只要已知 training seed 与 selection 重叠仍立即拒绝。

## 5. Committed-step 统计

- 使用 `available_actions.is_commit`；
- row[6] 必须严格等于 `ACTION_COMMIT`/`ACTION_NON_COMMIT`，并与 bool flag 交叉验证；
- PASS、普通 COMMIT、贡还提交和前缀 token 使用统一检查；
- 终局累计 token/commit 数分别与 `finish_token_step`、`finish_committed_step` 完全一致，否则评估失败；
- 旧 M06 日志保留原错误统计，不倒写。

## 6. 保守 clustered CI 与计划完整性

CI schema：`guandan-marvel-bounded-cluster-ci-v1`，方法为 bounded empirical-Bernstein 95% interval。四个 rotations 先形成 deal mean；有限样本 range correction 保证全胜/全败不会产生退化 `[1,1]`/`[0,0]`。

正式 promotion policy：

- mode=`formal`；
- 至少 32 个独立 deal clusters；
- CI lower threshold 默认 0.52；
- smoke mode 永不自动晋升。

32 的依据：对零方差全胜 clusters，empirical-Bernstein finite-range correction 仍给出非退化下界，同时避免 1–2 个 smoke deals 自动晋升。该门槛是 M06 的初始正式安全门槛，后续只能通过版本化决策修改。

Summary 必须对照 plan 的完整 `opponent × deal × rotation` 键集合，拒绝 missing whole deal、missing rotation、duplicate、extra、candidate/opponent checkpoint mismatch、NaN/Inf 和非法范围。

Overall 跨 opponent 共用 base deal，故只报告描述性 point metrics，不报告误导性 clustered CI；promotion 只使用 per-opponent CI。

## 7. 验收边界

修复验收要求真实 v5 C++ checkpoint、真实 spawn worker、完整五 lane/四换座、小规模 selection、真实 commit counters、pool recovery 和未使用 final audit。小样本必须得到 no-promotion/inconclusive，不构成模型强度或 Kaggle 性能证据。
