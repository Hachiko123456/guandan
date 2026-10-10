# M07 Kaggle T4×2 集中补修协议

- 日期：2026-10-10
- 基线：`4dbba7ae502ff21b3ca4e4f4da0dc3ad6879f65b`
- 范围：只修复 M07 交付链路；不执行 M08、不启动 Kaggle、不运行正式长训练
- 状态语义：补修完成后仍仅为 `remote_ready`，不是 `remote_verified`

## 1. 正式 bounded evaluation

M07 bounded evaluation 使用规则真值 `Rank.TWO`，其协议整数为 15。不得用裸整数 2，也不得放松 deals manifest 校验。

训练内评估必须由 `spawn` 子进程执行。子进程只接收冻结 checkpoint 路径、输出路径、设备和 JSON 配置；不接收 trainer、optimizer、sampler、replay 或训练 RNG。父训练进程在成功和异常路径都不加载评估模型，因此 Python、NumPy、Torch CPU/CUDA RNG 和训练状态不被评估改变。

`eval_only` 没有 live trainer，可直接调用同一实际 C++ evaluation core。CPU 本地验证使用实际 C++ backend、小模型、1 个 base deal、1 个 rotation 和 random opponent；这只证明关键路径，不是 formal selection/final holdout。

## 2. Durable checkpoint 先于评估

每个 checkpoint boundary 的固定顺序：

1. 两 rank 完成 learner update；
2. 保存并原子发布 DDP checkpoint envelope；
3. rank 0 立即写入 `checkpoint_manifest.json`；
4. rank 0 立即生成可直接下载和恢复的 recovery ZIP；
5. 之后才冻结 evaluation checkpoint 并启动隔离评估；
6. 评估完成或失败后，单独原子更新 evaluation 状态和 recovery bundle。

因此首次或后续评估失败都不能使已完成 update 丢失。Evaluation failure 必须写 `evaluation_summary.status=failed`、checkpoint index 的 `evaluation_status=failed`，并使所有 rank 共同失败退出；不得写 complete。

Recovery ZIP 使用临时文件加 `os.replace`。重打 bundle 失败时，旧 bundle 字节保持不变。Runner finally 可从已发布 checkpoint index 重建 bundle，但不得要求用户从 output ZIP 手工拼接 rank shard。

## 3. Resume 继承语义

Resume profile 不再携带 algorithm/model/batch/optimizer/sampling/replay 等训练语义，只允许：

- `additional_iterations`；
- `checkpoint_interval`；
- `evaluation_interval`；
- `max_wall_seconds`。

构建 trainer 前必须验证：

- recovery manifest schema/hash 和逐文件 hash；
- source commit/source archive hash；
- DDP envelope schema/hash/world size；
- rank coverage 和 shard SHA-256；
- rank training checkpoint schema；
- source/backend/semantic/resolved config/provenance declared hash；
- rank summary 与实际 shard 的模型、配置和 continuation state identity。

随后从 rank shard `resolved_config` 重建 TrainConfig，只覆盖允许的运行控制字段。Exact 或仅含既定 NON_SEMANTIC_CONFIG_PATHS 的 compatible resume 可继续；其他差异必须拒绝并提示使用创建 recovery 的精确 Dataset/source ZIP。

## 4. Continuation identity

Rank summary 新增 `guandan-marvel-ddp-checkpoint-v2` rank summary 的 `continuation_state_hash`，覆盖：

- resume/merged actor、critic；
- actor/critic optimizer；
- replay；
- sampler/environment state；
- Python/NumPy/Torch CPU/CUDA RNG；
- schedule、merge state、iteration/durable/learner counters。

本地验收必须比较“连续两次 update”和“一次 update -> recovery -> resume -> 下一次 update”的每 rank continuation hash，而不只比较 iteration。

## 5. 统一预算停止

每次完整 learner update 和 metrics 汇总后，所有 rank 在同一 collective 中提交本地 elapsed/budget vote。共享决策采用任一 rank 达到预算即停止。`checkpoint_due`、durable checkpoint、break 和最终 phase 对所有 rank 使用同一结果。

预算判断不在未完成 update 中生效；未完成 update 不计 durable。测试通过 rank 延迟和 elapsed offset 制造边界偏差，必须证明两 rank 在 timeout 内共同退出、保存同一 iteration、模型 hash 一致且 recovery 有效。

## 6. Notebook 实时进度

每 rank 写版本化 heartbeat；rank 0 每 5–10 秒更新 `progress.json` 并向 stdout flush 一行摘要，Notebook 和 `console.log` 均可见。至少包含：

- mode/phase；
- iteration/target；
- 已完成 games/env steps/game steps；
- rollout/learner/metrics/checkpoint/evaluation phase；
- 已测吞吐、session elapsed、session ETA；
- 每张 GPU utilization/显存或 unavailable；
- 最近 durable checkpoint、evaluation status；
- rank/pid/phase/heartbeat age。

第一个 iteration 完成前 counters 和 ETA 使用 unknown，不写伪 0。Resume ETA 使用 `iteration - session_start_iteration`，不能用历史 iteration 除本次 elapsed。Heartbeat 只调用 psutil/nvidia-smi 和文件 I/O，不执行模型 forward、不改变训练 RNG。

## 7. 故障和证据分类

证据必须区分：

- 实际 C++ bounded evaluation；
- 实际双进程 CPU/gloo DDP；
- 故障注入；
- 受控 unit callback；
- 静态 package/Notebook/source 检查。

CPU/gloo、缩小模型或 smoke horizon 不得写成 T4/NCCL/Kaggle verified。CUDA RNG 和双 T4 性能仍由 M08 验证。

## 8. 交付身份

旧 M07 包与补修包不可混用。正式 source ZIP 必须从 clean implementation commit 生成，`workspace_snapshot=false`；验收登记可使用后续独立 commit，报告必须同时记录 implementation SHA、registration SHA、source archive SHA 和 delivery manifest hash。
