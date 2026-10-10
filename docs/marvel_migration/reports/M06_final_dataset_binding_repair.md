# M06 Final Dataset Binding 小范围补修报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 开始 HEAD：`597392519c776d65874cc646bc1787261f66a837`
- 前次补修实现：`afcd8d9817a2e993b1fae5652bb339fce3294809`
- 本轮实现 commit：`60dbaa5bc57acd3067822ffaab898d8c62653675`
- 状态：`accepted`

## 范围

本轮只修复 FinalHoldoutAudit 与 runner 实际 final manifest 之间的 dataset identity 绑定，并勘误 `test_hook_process.py` 的历史 source hash。没有修改异步 worker、training provenance、champion/pool、统计方法、规则、训练数学或 C++ 引擎；没有运行 M07/Kaggle/DDP/正式训练；没有消费真实 final holdout。

## 问题复现

已在基线实现上复现：audit A 绑定 final seed 760030，runner 使用 manifest B（final seed 760040）时没有拒绝，进入 B callback，随后 A audit 被写为 failed 并产生 evaluation output。

复现测试在修复前 exit 1；日志保存在 `runs/M06_final_dataset_binding_repair/20261010_reproduction/reproduction.log`。

## 修复

1. Runner 在入口深拷贝并验证调用方 manifest，后续始终使用该冻结副本。
2. Final 分支计算实际 `dataset_identity(manifest, "final")`、generator version 和 deal count。
3. 在加载 policy、materialize deal、game callback、output mutation 和 reservation 前调用 audit read-only lock validation。
4. `reserve()` 新增三个必填参数，并在独占锁内再次执行三方比较：调用值、audit object 预期值、persisted holdout identity。
5. Schema v2 和持久化状态机不变；旧 reserved/running/terminal 文件不迁移、不重置。

## 关键验证

### A audit + B manifest

- A identity：`470c1d6fc7a7024e2f3741123e2fe124051450789128fa428c1923ccd6437a67`
- B identity：`3d31537428fd54f7ba2f8019d14594f59da34833faece5650888686ca72d3cfa`
- 错误：`ValueError: final dataset identity mismatch between evaluation manifest and audit object`
- materialize：0
- policy forward：0
- game callback：0
- A audit before/after SHA：相同
- output 只保留预置 `existing.json`

### 正确请求

- 20 个受控 callback 完成；
- audit 正常进入 consumed；
- active reservation 保存实际 dataset identity。

### 一次性和并发

- 仅 created_at 改变时 dataset identity 不变；consumed holdout 仍在 0 callback 拒绝；
- consumed B 不能借未使用 A audit 绕过；A/B bytes 均不变；
- audit object/persisted identity 冲突在锁内拒绝；
- 并发 reservation：1 成功、1 拒绝；
- 原 candidate mismatch、异常、中断、输出不覆盖测试继续通过。

## Source hash 勘误

历史 manifest 的错误记录：8915 bytes / `698968a0e357b02e550b263440b19e864b3ee9a417f6483a5485cc0e0657358e`。

基线提交和当前文件实际为：8914 bytes / `d5a00eca8bf415f3c7ba0380be3cd3be778a8186157582437209ab136885d2d6`。

错误 hash 精确对应真实文件末尾额外增加一个换行的版本。只修正历史 source manifest 该条目及上层 artifact 引用；没有倒写其他历史源码 hash。本轮另有独立 source/artifact manifest。

## 测试

- `python -m pytest tests/marvel_evaluation -q`：49 passed，exit 0；
- `python -m pytest tests/marvel_evaluation/test_final_audit_state.py -q`：10 passed，exit 0；
- evaluator/checkpoint/hook integration subset：21 passed，exit 0；
- A00：5 passed，exit 0；
- machine binding verifier：exit 0。

正式非重叠集合：54 passed，0 failed，0 errors，0 skipped。

没有重跑 M05 10k/100k、C++ build、训练长跑或前次 20 局演练。前次 20 局只作为继承证据引用，不冒充本轮重跑。

## 修改文件

- `guandan_marvel/evaluation/evaluator.py`
- `guandan_marvel/evaluation/manifests.py`
- `tests/marvel_evaluation/test_final_audit_state.py`
- `scripts/verify_m06_final_dataset_binding.py`
- `docs/marvel_migration/M06_FINAL_DATASET_BINDING_REPAIR_PROTOCOL.md`
- M06 规范、决策、STATUS、报告和本轮 evidence
- 历史 `M06_evaluation_champion_repair/source_manifest.json` 单项勘误及 artifact 引用同步

## 证据

`docs/marvel_migration/evidence/M06_final_dataset_binding_repair/`

主要文件：

- `binding_verification.json`
- `reproduction.json`
- `hash_erratum.json`
- `test_results.json`
- `source_manifest.json`
- `artifact_manifest.json`
- `verification.json`

## 结论

Final actual-dataset identity 绑定 blocker 已解决；历史 hash 勘误和当前证据链一致。M06 可重新接受，M07 可恢复待启动，但本轮不执行 M07。
