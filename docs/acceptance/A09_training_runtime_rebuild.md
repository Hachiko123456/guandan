# A09 训练运行时重构验收

本阶段只验收训练运行时、checkpoint/resume、日志、性能、双 GPU 调度和 Kaggle 交付。A00-A04 规则与环境验收结论不因本阶段改变。

## 1. Resume compatibility

Checkpoint 必须同时记录：

- `checkpoint_schema_version`
- `semantic_contract_hash`
- `implementation_fingerprint`
- `model_schema_hash`
- `algorithm`
- `optimizer_schema`
- `rollout_schema`
- `source_commit`
- `resume_mode`

允许三种明确模式：

- `exact_resume`: semantic contract、实现指纹、RNG/rollout 顺序全部一致；要求 continuation replay 可逐项一致。
- `compatible_resume`: semantic contract 一致，但允许日志、性能、设备、调度和实现变化；必须通过短 continuation verifier；不宣称 bitwise reproducibility。
- `migration`: 独立离线工具；不在训练主路径隐式迁移旧状态，只能显式输出新的初始化 checkpoint。

以下变化必须拒绝默认 resume：模型结构/参数含义、观测/动作/规则协议、reward/termination、RNG/reset/order、optimizer 或 objective 语义、rollout/minibatch 语义。以下变化不得阻塞 `compatible_resume`：日志、telemetry、progress、checkpoint 间隔、session 时间、输出目录、设备绑定和不改变语义的 IO 优化。

必须有测试覆盖：

- exact resume continuation 与 uninterrupted baseline 一致；
- compatible resume 能加载并完成 continuation，且明确标注非 bitwise；
- semantic mismatch、model mismatch、optimizer mismatch、损坏/伪造 digest 被拒绝；
- migration 只能显式执行，不被普通 resume 自动触发；
- 部分 rollout/optimizer update 不得被当成 durable update。

## 2. Structured logging

每个 run 必须生成：

- `run.json`
- `events.jsonl`
- `metrics.jsonl`
- `system_metrics.jsonl`
- `checkpoints/index.jsonl`
- `live/progress.json`
- `summary.json`
- `errors/failures.jsonl`（发生错误时）

每条事件至少包含：

- `run_id`, `session_id`, `phase`, `algorithm`
- `timestamp_utc`, `elapsed_seconds`
- `source_commit`, `semantic_contract_hash`
- `update`, `checkpoint_id`（适用时）
- `event`, `status`

训练指标必须覆盖 loss、policy/value/Q、entropy、KL、clip fraction、gradient norm、reward、terminal/truncated、committed steps、token steps 和各阶段耗时。

性能指标必须拆分 rollout、encoding、H2D、policy forward、sampling、environment step、next-value/critic、backward、optimizer、checkpoint、logging。

系统指标必须覆盖 CPU、RAM、磁盘、每张 GPU 的 utilization、allocated/reserved/peak memory；NVML 不可用时必须记录 `unavailable`，不能静默缺失。

热路径不得同步写逐 transition 日志。正常模式按 update/时间批量 flush；checkpoint 和 phase end 必须 atomic + fsync；异常时保留最近窗口和完整错误上下文。

## 3. Runtime performance

在相同 `remote_full` 配置、相同 T4 x2 环境上，与旧基线比较：

- IPPO 和 VRPO 的端到端 token throughput 至少达到旧基线的 3 倍；
- 评估 games/s 至少达到旧基线的 3 倍；
- 训练与评估都必须报告 CPU、GPU、IO 分项时间；
- 两张 GPU 都必须有真实 worker、显存分配和 telemetry 样本；不能只记录 `gpu_count=2`；
- 如果小模型导致 GPU 利用率不高，必须有 CPU/GPU benchmark 证明设备选择是有依据的。

旧基线参考：IPPO 约 28.7 token steps/s，VRPO 约 27.0 token steps/s；旧运行中 optimizer update 只占总墙钟时间约 7%–9%。

## 4. Training/evaluation correctness

- token_step、committed_step、reward、terminal/truncated 语义保持不变；
- IPPO/VRPO 的目标、critic 隔离和 observation privacy 测试继续通过；
- evaluation snapshot 与 training checkpoint 分离；
- 一个评估 worker 对同一 checkpoint 只加载一次；
- 不完整评估不能被计入完成统计；
- 评估重启时从该算法的第 0 局重新计数，不拼接缺失局数；
- 完整评估必须覆盖所有 opponent、deal group 和 seat rotation。

## 5. Kaggle package

完成修改后必须从当前 clean source 重新生成：

- 一份带 manifest、source commit、semantic contract hash 和 file digest 的源码 ZIP；
- 一份唯一 Run-All notebook；
- notebook `plan` 模式必须能够运行；
- notebook 不得自动猜旧 progress bundle；
- 旧源码包和旧 notebook 不作为恢复输入；
- package 测试必须验证路径安全、manifest 完整性、source/profile/contract 一致性。

## 6. Final gate

只有以下全部满足，才允许用户上传 Kaggle 进行长训练：

1. checkpoint/resume 测试通过；
2. structured logging 测试通过；
3. targeted training/evaluation tests 通过；
4. 端到端 benchmark 达到性能门槛；
5. 双 GPU telemetry 证明两张卡真实工作；
6. clean source package 和 Run-All notebook 生成成功并通过本地 plan/package gate；
7. 当前 run 从旧 checkpoint 完全脱钩，正式训练从新 run、新 checkpoint 开始。
