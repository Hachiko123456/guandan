# M09 Unified Train → Commit → Formal Eval Pipeline 本地实施报告

- 日期：2026-10-14
- 阶段：`M09_full_training`
- 状态：`unified_pipeline_local_implementation_complete_remote_abc_pending`
- M09 accepted：**否**
- 真实 Segment1000：**未运行**
- final holdout：**未生成、未运行**
- 本地 smoke：**通过，但不构成 Kaggle 验收**

## 1. 本轮完成范围

本轮按三个可独立回退的阶段完成了本地实现：

1. `d4c5406a4338a5b9fc7eeb082a08f8033539b108` — `feat(migration): add unified segment train-eval pipeline`
   - 固定绝对 schedule：500/2000、1000/4000、1500/6000、2000/8000；
   - 自动从 recovery iteration/completed-games 选择下一段；
   - durable、hash-protected pipeline state；
   - 训练 checkpoint/recovery commit 严格先于 evaluation；
   - evaluation 失败后保留 immutable training recovery；
   - 相同段重跑跳过训练；
   - 版本化 continuation bundle。
2. `e5fdc70c184ee854c9c20c720a12d9e2f3fb8bd2` — `feat(evaluation): add resumable plan-bound formal evaluation`
   - 每个 task/record 绑定 plan、candidate、opponent、deal、rotation、backend/protocol、deterministic policy 和 action trace；
   - JSONL/shard 恢复只运行缺失任务；
   - 一致重复安全去重；冲突重复、未知任务、旧 candidate、foreign plan 和损坏 JSON fail closed；
   - exact matrix 完整前不写 summary/result，也不允许 promotion；
   - promotion transaction 支持 `prepared`、`champion_published_pool_pending`、`completed`、`partial_failure` 幂等恢复。
3. `07fc7e94fba756b512d3ab28bf977d7e64b88cef` — `feat(evaluation): add deterministic dual-gpu evaluation`
   - `task_index % world_size` 固定分片；
   - 独立 worker/GPU、独立 shard、父进程 plan-order merge；
   - worker 非零退出、缺失 task 或冲突记录阻止 summary/promotion；
   - 单/双 worker deterministic records、ordering、summary hash 和 promotion input 对比；
   - 双 GPU formal 默认仍关闭，正式默认 worker 数为 1；
   - 版本化自动连续 Segment 配置已实现但默认 `enabled=false`，且永远不能触发 final holdout。

随后增加统一 Kaggle 入口、控制 overlay、远端 worker/runner、本地 smoke、交付验证和本报告。训练 source 继续使用已验证的 r2 精确身份，不改变模型架构或训练语义；新流水线控制代码以独立 hash-bound overlay 随 Unified Dataset 发布。

## 2. 最终用户入口

最终交付根目录：

`D:\project\kaggle_train\guandan-marvel-m09-unified-pipeline-v1`

唯一长期 Dataset 目录：

`D:\project\kaggle_train\guandan-marvel-m09-unified-pipeline-v1\dataset`

唯一长期 Notebook：

`D:\project\kaggle_train\guandan-marvel-m09-unified-pipeline-v1\guandan_marvel_m09_unified_pipeline.ipynb`

长期入口只有：

1. Unified Dataset；
2. Unified Notebook；
3. 上一次 Unified Notebook Output。

当前 v1 control 明确设置 `production_enabled=false`；无 Remote validation request 时只会默认运行小预算 Remote A，不会直接启动 Segment1000。Remote A/B/C 通过后，在**同一个 Kaggle Dataset** 上发布 reviewed control 新版本；之后日常 Run All 才会自动选择 1000/1500/2000。无需按 Segment 新建 Dataset 或 Notebook。

## 3. Source / Control / Artifact 身份

训练 source 保持 M09 r2 exact-resume 身份：

- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`；
- source archive SHA-256：`d08b8743970b8cce38a5ffe7768a3169b117a76ba2c03c596acafadd7e7cd465`；
- source content manifest hash：`fd8853e379b3bb2a6eefc00474bda9c200aa49139bcbea841a4ffa7c105e6fca`；
- main plan hash：`96f324f43ab391c3912a9284fd1dc862424b4c19d63ae9b375a0c15627095f88`；
- initial iter500 recovery SHA-256：`2d395ae4293e42c0b07f28e9b6d8508aa3c07aa1419d11ce65cb53598a43cb44`；
- iter100 champion seed SHA-256：`b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c`；
- legacy V1 anchor SHA-256：`880e74c1c41a5be252c1319f9be3ccb1f19553c4e63fe4658198b57c850b1f1d`；
- Unified control manifest hash：`dd229f94f70e47e45d18cdd2f9043014b61305752f39f15451f0f69f45c3b658`；
- delivery manifest hash：`463bf9b2e7c827d9489945a0556ba07fcac6e9f8bc9af8425ec6a37ab6cb1df4`。

完整 hash 清单：

`docs/marvel_migration/evidence/M09_unified_pipeline_local_implementation/source_control_artifact_hashes.json`

## 4. Pipeline 边界

Notebook orchestration 顺序固定为：

```text
Discover Unified Dataset + latest compatible recovery + previous continuation
  -> materialize exact r2 training source and verified runtime binary
  -> select next absolute Segment or pending evaluation repair
  -> subprocess: existing torchrun/DDP training runner
  -> DDP process exits
  -> verify exact iteration/completed-game budget, two rank shards, recovery manifest, CRC and hashes
  -> write immutable committed recovery + training_commit.json
  -> subprocess(es): standalone formal evaluation worker(s)
  -> plan-order merge and deterministic summary
  -> promotion/champion/pool transaction
  -> continuation packaging
```

formal evaluation 从不在存活的 DDP collective 中运行。训练 commit 完成前不会启动 evaluation。

## 5. Pipeline 状态和恢复

实现的正常状态：

`DISCOVER_INPUTS` → `PREFLIGHT` → `TRAINING` → `TRAINING_CHECKPOINT_COMMITTED` → `TRAINING_RECOVERY_COMMITTED` → `EVALUATION_PREPARING` → `EVALUATING` → `EVALUATION_COMPLETE` → `PROMOTION_COMMITTED` → `PACKAGING` → `COMPLETE`。

实现的失败/修复状态：

- `TRAINING_FAILED_RECOVERABLE`；
- `TRAINING_COMPLETE_EVALUATION_FAILED`；
- `EVALUATION_INCOMPLETE_RESUMABLE`；
- `PROMOTION_REPAIR_REQUIRED`；
- `PACKAGING_FAILED_RECOVERABLE`。

状态 JSON 使用临时文件、flush/fsync 和原子 replace；console 文本不是恢复真值。

当 previous Output 表示训练已 commit 但评估失败时，Notebook 从 continuation 恢复 pending Segment。即使输入 recovery 已达到该段 target，也不会错误选择下一段训练；它验证原 `training_commit.json` 和 immutable recovery 后直接补跑 evaluation。

## 6. Continuation Bundle

每段输出：

`/kaggle/working/guandan-marvel-unified-output/guandan-marvel-continuation.zip`

其内容包括：

- committed training recovery；
- `training_commit.json`；
- pipeline state；
- current champion manifest 和 checkpoint history；
- opponent pool manifest；
- promotion transaction（若存在）；
- 当前及历史 evaluation plans/shards/summaries/decisions；
- candidate checkpoints；
- next target identity；
- source/control/plan hashes；
- `final_holdout_generated=false`。

下一次运行只需 Unified Dataset + 上一次 Notebook Output。

## 7. Formal Evaluation 恢复语义

formal plan 将 5 个 opponent lanes × 32 independent selection clusters × 4 rotations 枚举为 640 个稳定 task。唯一键为：

`opponent_id + deal_id + seat_rotation`

每条 durable record 绑定：

- resumable plan hash 和原 M06 base plan hash；
- task index/hash；
- candidate checkpoint SHA-256；
- opponent checkpoint SHA-256（rule/random 为 `null`）；
- deal id/materialized deal SHA-256；
- seat rotation；
- backend identity hash、protocol version；
- deterministic policy hash；
- outcome/health/token/committed steps/action trace hash；
- formal record content hash。

正式 record 排除 wall-clock scheduling noise；真实吞吐和 GPU 信息保留在 heartbeat/worker log 中。因此单/双 worker 的 formal records 和 summary identity 可确定性比较。

## 8. Dual-GPU 状态

本地实现已具备：

- GPU0：偶数 task index；
- GPU1：奇数 task index；
- 独立 `games.gpu0.jsonl` / `games.gpu1.jsonl`；
- 父进程验证后按 plan index 合并；
- 任一 worker 失败时保留合法 shard，阻止 summary/promotion。

但 Unified control 冻结：

- `formal_workers_default=1`；
- `dual_gpu_formal_remote_validated=false`。

只有 Remote C 通过后，才允许在新版本化 control 中考虑把双 GPU 设为 formal 默认。

## 9. Local Smoke

机器可读结果：

`docs/marvel_migration/evidence/M09_unified_pipeline_local_implementation/local_smoke.json`

本地 deterministic fixture smoke 证明：

- 训练 hook 只调用 1 次；
- training recovery commit 先完成；
- evaluation 在 3 条记录后受控失败；
- failure 前后 committed recovery SHA-256 完全相同；
- 第二次运行使用 target recovery + pending target，跳过训练；
- 只补缺失 evaluation tasks；
- summary 完成；
- promotion 为 `inconclusive`/不晋升语义；
- 单/双 worker 20-task corpus 的 records、ordering、records hash、summary hash、promotion input 完全一致；
- continuation 指向 segment1500；
- 未生成 final holdout。

这只是本地 fault-injection，不是 Kaggle T4×2 验收。

## 10. 测试

已执行：

```text
python -m pytest --import-mode=importlib tests/marvel_kaggle -q
# 70 passed

python -m pytest --import-mode=importlib tests/marvel_evaluation -q
# 62 passed

python -m pytest --import-mode=importlib tests/marvel_training tests/unit/test_checkpoint_compatibility.py tests/unit/test_kaggle_recovery.py -q
# 114 passed

python -m pytest --import-mode=importlib tests/marvel_kaggle/test_m09_unified_pipeline.py tests/marvel_evaluation/test_resumable_formal.py tests/marvel_evaluation/test_dual_gpu_formal.py -q
# 24 passed

python scripts/run_m09_unified_local_smoke.py --output runs/m09_unified_local_smoke.json
# passed
```

专项覆盖：auto segment selection、previous continuation discovery、training commit before evaluation、evaluation failure preserves recovery、target-recovery pending retry skips training、partial evaluation resume、corrupted/foreign rejection、duplicate conflict rejection、promotion idempotency、deterministic sharding/merge、worker failure gate、single Dataset/Notebook validation、no final holdout artifact。

## 11. Kaggle Remote A/B/C

### Remote A — Unified Pipeline Smoke

1. 用 `dataset/` 创建一个私有 Unified Dataset并导入唯一 Notebook；
2. Accelerator 选择 T4×2；
3. 只添加 Unified Dataset，Run All；v1 control 在 production gate 未打开时自动选择 Remote A（显式设置 `GUANDAN_M09_REMOTE_VALIDATION=A` 也等价）；
4. 不修改 Notebook/target/mode cell；
5. 验证只从 iter500/2000 games 运行到 iter502/2008 games，而不是 Segment1000；
6. 验证 DDP 完全退出后才出现独立 evaluation worker；
7. 固定 smoke selection 为 2 clusters × 4 rotations × 5 lanes = 40 games，policy 为 smoke、不晋升；
8. 下载 continuation、committed recovery、pipeline state、evaluation summary 和日志，并审查 heartbeat、两 rank、CRC/hash、training commit identity。

v1 production gate 会阻止无 validation request 的 Segment1000。

### Remote B — 故障与恢复

首次从 Unified Dataset 单独运行，设置：

```text
GUANDAN_M09_REMOTE_VALIDATION=B
GUANDAN_M09_EVAL_FAIL_AFTER=8
GUANDAN_M09_EVAL_FAIL_WORKER=0
```

它只训练到 iter502/2008 games。期望 recovery 先 commit；evaluation worker 非零退出；partial shard 和 continuation 可下载；training recovery 未变化。

第二次添加 Unified Dataset + 第一次 B Output，保留 `GUANDAN_M09_REMOTE_VALIDATION=B`，移除 failure injection。期望自动进入 `remote_b_smoke0502` evaluation-only repair、training subprocess 不启动、只补缺失 tasks、完成 smoke summary/不晋升 decision/continuation。

### Remote C — 单/双 GPU 一致性

添加 Unified Dataset + 已完成的 Remote A Output，设置：

```text
GUANDAN_M09_REMOTE_VALIDATION=C
```

该路径不训练，复用 A 的 candidate/plan/单 worker summary，自动启动两个独立 GPU worker并生成一致性报告。逐项比较 game key、candidate win、ranking/outcome、illegal/truncated、token/committed steps、action trace、merge ordering、records hash、summary hash 和 promotion input。通过前 production formal 继续默认单 worker。

精确机器计划：

`docs/marvel_migration/evidence/M09_unified_pipeline_local_implementation/remote_validation_plan.json`

## 12. Remote A bootstrap repair（2026-10-14）

第二次 Kaggle 尝试已通过 Unified control 校验、recovery materialization 和 native binary import，但在训练启动前加载 legacy V1 anchor 时失败：exact-resume source 内没有 `configs/acceptance_profiles.json`。根因是 Unified control Dataset 已携带并校验 acceptance profile catalog，但 runner 在 champion/pool 初始化前没有将它 materialize 到 source runtime 路径。

修复后，`scripts/m09_unified_pipeline.py` 在加载 champion/legacy/pool 前：

1. 验证 Dataset 中 `acceptance_profiles.json` 的 SHA-256；
2. 复制到 `<source_root>/configs/acceptance_profiles.json`；
3. 再次验证目标 hash；
4. 输出 `acceptance_profiles_materialized` 事件；
5. 才继续加载 legacy anchor。

该失败发生在 training/DDP 启动前，没有修改 checkpoint、recovery、champion，也没有运行 Segment1000。修复上传包：

`D:\project\kaggle_train\guandan-marvel-m09-unified-pipeline-v1\guandan-marvel-m09-unified-dataset-upload-v3.zip`

- bytes：69,291,142；
- SHA-256：`9acf9f3aa93c15525bd878b8dca3b0c7574ce71761407294cf6cc665d4738aed`；
- 19 entries；
- 不含会与 raw `control_overlay/` 冲突的内层 `control_overlay.zip`。

机器证据：`docs/marvel_migration/evidence/M09_unified_pipeline_local_implementation/remote_a_bootstrap_repair.json`。

## 13. 已知限制与下一步

- Remote A/B/C 尚未执行；当前只能称 `local implementation complete / remote validation pending`。
- 不得直接用未经 Remote A/B/C 验证的 Unified 高级路径运行正式 Segment1000。
- 当前训练分布仍是锁定的 Rank.TWO / no-previous-result；广上下文强度门槛仍可能阻止 E2/E3。
- 当前 champion 仍是 iter100；iter500 candidate 未晋升。
- 当前模型仍约 0.002061B 参数，架构未改变。
- 自动连续 Segment 默认关闭；每轮最多一个 Segment；final holdout 始终禁止。
- M09 仍 incomplete，不能声明 E1/E2/E3，也不能声明 M09 accepted。

下一步由用户依次执行 Remote A、Remote B、Remote C，返回全部 Notebook Outputs 和 console logs。本地审查三项都通过后，在同一个长期 Kaggle Dataset 上发布 `production_enabled=true` 的 reviewed control 新版本；只有该版本才授权自动选择并运行真实 Segment1000。
