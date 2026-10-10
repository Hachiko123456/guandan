# M07 Kaggle T4×2 集中补修与重新验收报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 补修基线：`4dbba7ae502ff21b3ca4e4f4da0dc3ad6879f65b`
- 实现 commit：`77a3674993d14d8c87fcab3a5084b1cf0c399b67`
- 验收登记 commit：本报告所在提交；精确 SHA 在最终答复报告
- 阶段状态：`remote_ready` / `accepted`
- 远端状态：`remote_verified=false`

## 范围

本轮只修复 M07 Kaggle 交付链路的 A–F 问题，重新生成并验证正式交付。没有实施 M08，没有上传或启动 Kaggle，没有正式长训练，没有消费 final holdout，没有修改真实 champion/pool、M05 性能门槛或 IPPO/VRPO 数学。

## A：正式评估级牌与真实入口

- `run_smoke_evaluation` 改为使用规则枚举 `Rank.TWO`，manifest 中实际编码为 15；
- pilot 周期评估使用 spawn 子进程真实加载 C++ checkpoint；
- eval_only 使用同一真实 C++ evaluation core；
- CPU/gloo 验收不再强制关闭 evaluation；
- 本地实际运行采用小模型、1 deal、1 rotation、random opponent，不涉及 final holdout。

结果：正式 pilot 首次周期评估 `complete`，1/1 游戏完成、0 illegal、0 truncated；eval_only 同样 `complete`，两者 `level_ranks=[15]`。

## B：评估失败与 durable recovery

Checkpoint boundary 已改为：

`save DDP checkpoint -> publish checkpoint index -> build recovery -> isolated evaluation -> atomic evaluation update/rebundle`。

首次评估失败和 iteration 2 后续评估失败均真实注入：

- 两 rank 都收到相同错误并退出，无 barrier 无限等待；
- progress 为 `phase=failed`、`complete=false`；
- 首次失败保留 iteration 1 recovery，并直接用于一次 resume update；
- 后续失败保留最新 iteration 2 recovery；
- bundle failure 注入后仍存在有效 recovery；
- 独立 unit test 验证 `os.replace` 失败时旧 recovery ZIP 字节不变。

下载的 recovery ZIP 直接用于 resume，用户不需要从 output ZIP 手工拼 shard。

## C：Resume 继承原训练语义

Resume manifest 已删除 algorithm/model/batch/epoch/replay 等重复配置，只保留追加迭代、checkpoint/evaluation 周期和会话预算。

Worker 在 trainer 构建前验证 recovery、DDP envelope 和当前 rank shard，再从 shard `resolved_config` 重建算法、模型、batch、optimizer、sampling/replay、训练来源和 DDP 配置。非 run-control 差异 fail closed。

验证：

- pilot：连续 2 updates 与 `1 update -> recovery -> resume -> next update` 的每 rank `continuation_state_hash` 完全相同；
- full：使用原配置 `history_dim=128`、`feature_dim=256`、`decoder_layers=4`、global batch 4、`agent_epoch=4` 完成 smoke-horizon update 和 recovery resume；恢复后配置逐项相同；
- 旧包/新包 source commit、archive 或 schema 不一致时给出可操作拒绝信息。

## D：预算停止共同决策

所有 rank 在完整 learner update 后通过 collective 汇总本地 elapsed/budget vote，任一 rank 达到预算即形成统一决定。Checkpoint due、break 和最终 phase 使用同一共享结果。

实际双进程测试给 rank 1 增加 1000 秒 elapsed offset 和边界延迟：

- rank 0 vote=false，rank 1 vote=true；
- 两 rank 看到相同 votes 和 decision=true；
- 两 rank 都在 iteration 1 写 `budget_stopped` 并退出；
- checkpoint iteration 相同，跨 rank 模型 hash 一致，recovery 有效；
- runner 和外层测试均有明确 timeout。

## E：评估 RNG/训练状态隔离

训练内评估改为 spawn 子进程。子进程只接收 immutable checkpoint 路径和 JSON 配置，不接收 live trainer、optimizer、sampler、replay 或 RNG。

真实模型加载测试覆盖成功和异常路径，父进程 Python/NumPy/Torch CPU RNG 均保持不变。Evaluation-enabled 连续/恢复训练的完整 continuation hash 相同，覆盖模型、optimizer、sampler、RNG、schedule、replay 和 counters。

本机 PyTorch 为 CPU build；CUDA RNG 的真实硬件验证留给 M08，没有伪造通过。

## F：Notebook 实时进度与故障诊断

新增 `guandan-marvel-progress-v2` 和每 rank heartbeat。rank 0 每 5–10 秒向 stdout flush 摘要并更新 `progress.json`，runner 将 stdout 实时写入 Notebook 和 `console.log`。

实际短程日志包含：

- mode/phase；
- rollout、learner、metrics、checkpoint、evaluation；
- iteration/target；
- completed games、env/game steps 和吞吐；
- session elapsed/ETA；
- GPU utilization/显存或 unavailable；
- 最近 checkpoint/evaluation；
- 两 rank phase、pid 和 heartbeat age。

首个 iteration 完成前计数与 ETA 显示 `unknown`；resume ETA 使用本次会话完成 iteration 数。

## 测试

阻塞、非重叠 pytest：

1. `tests/marvel_kaggle`：20 passed；
2. `tests/marvel_training` + `tests/marvel_core/test_core.py` + `tests/marvel_evaluation`：87 passed；
3. `tests/acceptance/test_a00_specification.py`：5 passed。

总计：112 passed，0 failed，0 errors，0 skipped。

额外真实/故障/静态验证：

- 精确 implementation commit source ZIP 构建与 package validation；
- Notebook JSON/AST/占位符检查；
- ZIP 解压后离线 pybind build/import；
- 解压源码 CLI preflight；
- 两次连续 update、split resume、full-config resume；
- pilot 首次真实 C++ evaluation、eval_only；
- 首次/后续 evaluation failure、bundle failure；
- rank budget skew；
- Notebook-visible heartbeat；
- M03 IPPO/VRPO smoke 回归。

## 正式交付

- 路径：`D:\project\kaggle_train\guandan-marvel-t4x2-20261010`
- Source commit：`77a3674993d14d8c87fcab3a5084b1cf0c399b67`
- Source archive SHA-256：`92b6a40bf3f23d7b0bff9c895072f1a15070e3300ec066d4ddf482c0f1ba784b`
- Delivery manifest hash：`15ab0e4899dc1f00e61895eedeb73008b27b13b53ad4ba7152d2f051fd954287`
- `workspace_snapshot=false`
- `remote_ready=true`
- `remote_verified=false`

旧 M07 包不能和本补修包的 recovery 混用。

## 证据

- `docs/marvel_migration/evidence/M07_kaggle_t4x2_repair/`
- 大型日志与 checkpoint：`runs/M07_kaggle_t4x2_repair/20261010_acceptance/`
- 原始问题复现：`runs/M07_kaggle_t4x2_repair/20261010_reproduction/`

## 受保护范围

未修改 `guandan/cards.py`、`guandan/combos.py`、`guandan/round.py`、`guandan/state.py`、`docs/rules.md`。未写入 `D:\project\doudizhu`。

## 已知限制

1. 没有真实 Kaggle/T4/NCCL/CUDA 训练；
2. 本地实际 DDP 为 CPU/gloo，物理 GPU 仅由 nvidia-smi 遥测可见；
3. 本地 bounded evaluation 每次仅 1 局，是管线证据，不是强度证据；
4. full 验证使用原模型/batch/epoch 配置和 smoke horizon，仅 2 updates，不是正式长训练；
5. CUDA RNG 隔离、T4 显存、双卡 speedup 和 ETA 仍需 M08。

## 结论

A–F 均修复并有回归、实际双进程、故障注入和最终包证据。M07 可恢复为 `remote_ready / accepted:true`，M08 可恢复为 `ready`，但本轮不执行 M08。
