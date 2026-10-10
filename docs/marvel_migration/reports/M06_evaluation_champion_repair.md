# M06 集中补修与重新验收报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 开始 HEAD：`a46204c756802541c36d49f90ab922cc1f4b6fb8`
- 原实现：`10cb3452f69cd23a3bed848498e4353dcfbe3b9c`
- 原登记：`a46204c756802541c36d49f90ab922cc1f4b6fb8`
- 补修实现 commit：由本报告所在的第一笔提交确定，第二笔登记准确 SHA
- 状态：`accepted`（repair verification）

## 范围

本轮只修复 M06 的评估安全、数据来源、final 状态机、promotion 身份绑定、committed 统计和小样本政策。没有修改 A01/A02 规则、IPPO/VRPO 数学、M02/M03 on-policy/中断保护；没有运行 Kaggle/DDP/长训练；没有使用正式 final holdout；没有比较 FableDan。

原 M06 报告、日志、checkpoint、champion/pool/final audit 和机器证据保持历史只读。

## 六类问题复现与修复

### A. 异步 RNG 污染和结果丢失

**已复现。** 原 ThreadPool 真实 loader 改变共享 Torch RNG；iteration 1 failure 在 iteration 2 调度时被删除。

修复：spawn ProcessPool；worker 只收 immutable snapshot/config/path；work/worker status/result/journal 全部持久化；success/failure/interrupted/cancelled 先 harvest 再 delivery；duplicate/pending/shutdown 状态显式记录。Trainer 每个 durable iteration 自动 collect，训练结束由 trainer close 收尾。

真实 loader、真实 C++ 两次训练更新测试证明启用/禁用评估后的模型、trajectory counters、后续 RNG 完全一致。Snapshot 后修改 live model，worker 仍读取原冻结权重。

### B. Final 一次性检查过晚

**已复现。** consumed 后重复请求仍执行 20 次 callback 并写 summary 才失败。

修复：final audit v2 状态机 `unused -> reserved -> running -> consumed|failed|interrupted`；用 stable dataset hash、candidate、plan/config、evaluation_id 原子预约。Reservation 在 final deal materialization、score 和 output mutation 前完成；O_EXCL lock 防并发；当前不支持失败重跑。

测试证明 consumed 和 candidate mismatch 均 0 次 callback；并发最多一方成功；首局异常、KeyboardInterrupt、plan write failure 分别进入 failed/interrupted；旧 summary 不覆盖。

### C. Promotion 未绑定批准 candidate

**已复现。** Decision 批准 A，但原 store 接受并写入 B。

修复：decision v2 验证 schema/hash/status/mode、candidate exact identity、真实文件 SHA、算法/架构/协议/backend/build、champion manifest hash/generation、plan/summary/evaluation/deals。Path cache 加入实时 SHA。

Champion/pool 使用事务状态：pool 失败时 champion 显式 `pool_sync=pending`，transaction=`partial_failure`，run 状态不是 complete。

### D. 数据隔离未绑定实际训练来源

**已复现。** 旧 checkpoint training seed 93001 被放入 selection，原流程仍 complete。

修复：training checkpoint v5、sampler state v2、training provenance v1；记录 domain/source manifest/generator/真实 seed 语义/actual deal hashes/count。Selection/final 在任何 policy forward 前比较已知 initial seed 和真实 deal hash。旧 v4 不补造 provenance，只允许 smoke-only selection；final 拒绝；已知 seed overlap 即使 smoke 也拒绝。

### E. Committed-step 统计错误

**已复现。** `bool(row[6])` 将 ACTION_NON_COMMIT=305 也当真；旧 records 全部 committed==token。

修复：使用 `available_actions.is_commit`，并与 ACTION_COMMIT/ACTION_NON_COMMIT row token 交叉核验。终局累计数与 engine `finish_token_step`/`finish_committed_step` 强一致，否则失败。

新真实演练：3708 token steps，1722 committed steps；20 局全部与 engine counters 一致，且 committed < token。旧错误日志未改写。

### F. 小样本和完整性错误放行

**已复现。** 两个全胜 clusters 得到 `[1,1]` 并 promote；缺整个 deal 仍 complete。

修复：bounded empirical-Bernstein 95% deal-cluster interval；finite sample range correction；formal auto-promotion 最低 32 deals；smoke 永不自动晋升。Summary 对照 plan 的 exact opponent×deal×rotation keys，拒绝 missing/duplicate/extra/checkpoint mismatch/nonfinite/malformed。Overall 跨 opponent 不报告误导性 clustered CI。

## 真实异步评估演练

运行目录：`runs/M06_evaluation_champion_repair/20261010_repair_acceptance_final2`

- 真实 C++ IPPO 小模型训练两次 iteration；
- 生成 provenance-complete v5 champion 和 evaluation snapshot；
- spawn worker 执行 candidate/champion/rule/random/legacy/pool 固定矩阵；
- 1 deal × 4 rotations × 5 lanes = 20/20 完整终局；
- illegal=0，truncated=0；
- formal data isolation eligible=true，无 seed/deal hash overlap；
- promotion=`inconclusive`，原因是 1 cluster < formal 32；
- champion 不变，pool manifest 可恢复；
- final audit 保持 `unused`；
- 等待真实后台评估期间 model/merged/optimizer/RNG/sampler/provenance hash 全部不变。

该演练证明功能闭环，不是模型强度或远端性能证据。

## 测试

1. M06 repair + MARVEL training：70 passed，0 failed，0 skipped，exit 0。
2. M01–M05 backend regression：180 passed，0 failed，0 skipped，exit 0。
3. A00–A04 protected acceptance：157 passed，0 failed，0 skipped，exit 0。
4. Legacy evaluation regression：155 passed，0 failed，0 skipped，exit 0。

正式非重叠合计：562 passed，0 failed，0 errors，0 skipped。

额外 backend import/identity diagnostic：10 passed，0 failed，0 skipped，exit 0。

报告与 STATUS 完成后的最终 A00 recheck：5 passed，exit 0。

Repair artifact verifier：exit 0。

## 修改文件

核心：

- `guandan_marvel/config.py`
- `guandan_marvel/runtime/sampler.py`
- `guandan_marvel/training/checkpoints.py`
- `guandan_marvel/training/local.py`
- `guandan_marvel/evaluation/{hooks,workers,manifests,provenance,checkpoints,host,stats,promotion,pool,evaluator}.py`
- `scripts/run_marvel_m06_repair.py`
- `scripts/verify_marvel_m06_repair.py`

测试：

- `tests/marvel_evaluation/test_hook_process.py`
- `tests/marvel_evaluation/test_final_audit_state.py`
- `tests/marvel_evaluation/test_evaluator_integrity.py`
- `tests/marvel_evaluation/test_committed_steps.py`
- 现有 M06/training tests 的 v2/v5 compatibility 更新。

规范：

- `docs/marvel_migration/M06_EVALUATION_REPAIR_PROTOCOL.md`
- `00_README.md`、`05_TRAINING_SPEC.md`、`06_EVALUATION_CHAMPION_SPEC.md`
- `08_ACCEPTANCE_MATRIX.md`、`09_CHECKPOINT_COMPATIBILITY.md`
- `10_RISK_AND_DECISIONS.md`、`M06_EVALUATION_PROTOCOL.md`、`M06_SOURCE_AUDIT.md`

## 机器证据

目录：`docs/marvel_migration/evidence/M06_evaluation_champion_repair/`

- `reproduction.json`
- `issue_resolution.json`
- `async_hook_verification.json`
- `final_state_machine.json`
- `promotion_binding.json`
- `data_provenance.json`
- `committed_steps.json`
- `statistics_policy.json`
- `real_exercise.json`
- `test_results.json`
- `verification.json`
- `artifact_manifest.json`

## 已知限制

1. 真实演练只有一个 selection deal，按正式政策必须 inconclusive。
2. Repair checkpoint 是本地小规模功能训练，不代表强度。
3. Final 状态机只用临时测试 holdout；正式 final 仍未使用。
4. 未做 Kaggle、DDP、长训练或 FableDan 对比。
5. Promotion 成功路径使用真实 checkpoint identity + 合成完整矩阵测试，不是强度证据。

## Blockers 与下一阶段

六类安全/身份/隔离/统计 blocker 均已修复并通过失败路径测试。M06 可重新接受；M07 可恢复为待启动状态，但本轮不执行 M07。
