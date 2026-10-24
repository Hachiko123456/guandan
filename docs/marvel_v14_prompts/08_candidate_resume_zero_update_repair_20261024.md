# 08：candidate 恢复预检零更新修复

日期：2026-10-24。来源日志：`C:\Users\yhx\Downloads\guandan-t4x2-pilot (10).log`。

## 失败证据与根因

v10 在 Kaggle T4×2 已通过输入检查、control 的训练/恢复/四场评估，以及 candidate 首次两迭代训练。candidate 从 `iter000002` 精确恢复后，唯一预检迭代最终没有应用任何 optimizer step；`run_training_arm` 随后以 `applied_updates=0` 失败。candidate 启用了 `target_kl=0.015`，该结果与合法的更新前KL早停路径一致；外层日志没有保留该失败迭代的minibatch计数，因此不能排除AMP跳步等另一种零更新来源。正式两臂训练尚未开始。

这不是 CUDA、NCCL、checkpoint 缺失、v10 候选池 batch 隔离修复或上传混包错误。可以由日志直接证明的触发条件是 `ProductionActions.preflight()` 将恢复验证限定为 `iteration_limit=1`，同时要求该唯一迭代必须产生更新。候选配方存在合法零更新路径；用一个随机迭代作为恢复正确性的唯一判据会产生假失败。

## Spec

- 不更改候选/控制配方、target_kl、PPO、模型、奖励、数据域、种子、正式训练预算或评估协议。
- 初次预检仍运行两个迭代并产生恢复点。
- 恢复预检改为有界的三个迭代窗口。`iteration_limit` 在训练入口中表示“从恢复点新增 N 次迭代”；窗口内只要产生至少一次真实 optimizer update 才能通过。
- 保留 `run_training_arm` 的零更新拒绝。因此三个恢复迭代全部无更新时仍失败，不能把 target-KL 跳过冒充有效训练。
- 保留 start/final iteration、精确恢复、模型导出、双rank、四场真实预检评估和两个预检都通过前禁止正式训练的门槛。
- 在 preflight 证据中记录恢复窗口、实际新迭代数、更新数及 KL stop 数，便于再次诊断。
- 新增测试必须能复现：一迭代窗口零更新失败；三迭代窗口中首个被KL跳过、后续至少一个更新则通过；三次均零更新仍失败。
- 生成新交付版本，不覆盖 v10。最终包必须运行 actual payload QA，并证明相对v10只有本修复所需 driver/文档证据变化（生产 overlay、模型、native、plan与发牌不变）。

## Acceptance Criteria

1. 日志中的 candidate 恢复命令不再是 `--iteration-limit 1`，而是明确的有限窗口。
2. 单次 target-KL 零更新不会导致假失败；整个窗口零更新仍 fail-closed。
3. control 与 candidate 均继续验证“首次训练有更新、恢复窗口有更新、恢复迭代前进、模型可评估”。
4. 相关 serial compare、target-KL、恢复事务、启动和交付 QA 测试通过。
5. 主代理审查实际 diff、测试断言、构建日志、最终 payload 内容与哈希后才可交付。
6. 不上传、不启动 Kaggle。用户使用已配置 token 自行执行一键上传脚本。

## 修改范围与计划

- 允许修改：`tools/serial_compare/pipeline.py`、相关 `tests/marvel_kaggle/` 测试、交付构建生成物、当前状态和本修复报告。
- 禁止修改：训练器核心、target-KL算法、plan配方、模型/native、v10已有目录、正式预算和评估数据。
- deepseek-coder 子代理实现最小代码和专项测试；主代理独立复核与运行测试。
- 主代理构建新 delivery，运行 `scripts/verify_t4x2_serial_compare_delivery.py` 并检查新旧包差异。
- 一键上传入口沿用 `scripts/kaggle.ps1 run -Delivery <新交付目录>`；本轮只验证本地准备和 dry-run，不调用真实上传。
