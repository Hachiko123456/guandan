# M07 Kaggle T4×2 DDP 阶段报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 开始 commit：`d44c92bc7bb30efa89ad75ec67272be43e5e4649`
- 结束 commit：本报告与实现位于同一个原子 commit；精确 SHA 由最终答复和 clean-commit source ZIP manifest 报告，避免在 commit 内制造自引用哈希
- 执行环境：Windows 11；Python 3.12.0；PyTorch 2.13.0+cpu；本进程 CUDA unavailable；`nvidia-smi` 可见 RTX 3050 Ti，但未作为训练设备
- 阶段状态：`remote_ready` / `accepted`

## 范围

实现 M07 Kaggle T4×2 交付生成、输入发现、Run-All Notebook、单模型双 rank DDP、版本化 durable checkpoint/recovery、主日志与每 rank 诊断、有限评估、离线 source build，以及本地 package/CLI/CPU 双进程恢复验证。没有运行或伪造 Kaggle `/kaggle` 成功证据，没有上传 Kaggle，没有执行正式 full run。

## 实际修改文件

核心实现：

- `guandan_marvel/runtime/distributed.py`
- `guandan_marvel/training/local.py`
- `guandan_marvel/evaluation/provenance.py`
- `guandan_marvel/kaggle/` 全部版本化 delivery/runtime 模块
- `configs/marvel_kaggle_profiles.json`
- `scripts/build_marvel_kaggle_delivery.py`
- `scripts/run_m07_local_ddp_recovery.py`
- `tests/marvel_kaggle/`

规范和控制面：

- `docs/marvel_migration/M07_KAGGLE_DDP_PROTOCOL.md`
- `docs/marvel_migration/M07_SOURCE_AUDIT.md`
- `00_README.md`、`05_TRAINING_SPEC.md`、`07_KAGGLE_T4X2_RUNBOOK.md`
- `09_CHECKPOINT_COMPATIBILITY.md`、`10_RISK_AND_DECISIONS.md`
- 本报告、M07 evidence、`STATUS.yaml`

逐文件 SHA-256 见 `docs/marvel_migration/evidence/M07_kaggle_t4x2/source_manifest.json`。

## 未修改的受保护范围

未修改 `guandan/cards.py`、`guandan/combos.py`、`guandan/round.py`、`guandan/state.py` 或 `docs/rules.md` 的规则语义；未删除 legacy V1 trainer/model/checkpoint 只读评估能力；未写入只读参考仓库 `D:\project\doudizhu`。

## 测试命令与原始结果

阻塞、非重叠 pytest 集合：

1. `python -m pytest tests/marvel_kaggle -q --junitxml=...`：8 passed；
2. `python -m pytest tests/marvel_training tests/marvel_core/test_core.py tests/marvel_evaluation -q --junitxml=...`：87 passed；
3. `python -m pytest tests/acceptance/test_a00_specification.py -q --junitxml=...`：5 passed。

总计：100 passed，0 failed，0 errors，0 skipped。

动态/包验证：

- dirty staging delivery build + full manifest/hash/Notebook validation：exit 0；
- source ZIP 解压后离线 `python setup.py build_ext --inplace` 和 C++ import/build identity：exit 0；
- 从解压 source 启动本地 CLI preflight：exit 0，明确 `local_only`、`remote_preflight_passed=false`；
- 双进程 CPU/gloo pilot：iteration 1，world size 2，跨 rank 模型 hash 相同；
- recovery ZIP 自动发现、校验和 resume：iteration 1 -> 2，exit 0；
- M03 IPPO/VRPO 单 rank 恢复 smoke 回归：两算法各 2 durable updates/2 games，exit 0。

完整命令、JUnit、退出码和日志 hash 见 `test_results.json`。本机 Windows PyTorch elastic store 缺 libuv，因此本地双进程测试以显式 RANK/WORLD_SIZE 环境启动两个 gloo worker；正式 Kaggle 路径仍固定使用 `torchrun --standalone --nproc_per_node=2`，并由 source ZIP validator 检查。

## 机器可读证据

目录：`docs/marvel_migration/evidence/M07_kaggle_t4x2/`

- `protocol_schema.json`
- `package_validation.json`
- `local_ddp_recovery.json`
- `source_build_validation.json`
- `cli_integration.json`
- `environment.json`
- `test_results.json`
- `source_manifest.json`
- `verification.json`
- `artifact_manifest.json`

大型编译、checkpoint、日志和本地恢复输出保存在 `runs/M07_kaggle_t4x2/20261010_acceptance/`，不提交 Git。

## 性能结果

M07 没有真实 T4×2 性能结论。本地 CPU 双进程只验证同步和恢复语义；不报告双卡 speedup、GPU utilization、T4 显存或 full ETA。M08 必须在真实 Kaggle T4×2 下载日志后验证至少 1.5× 目标和参数可行性。

## 正确性与隐私审查

- DDP 是两个 rank 对同一 actor/critic 做同步梯度，不是两卡各跑独立算法；
- global batch 被 2 整除并按 rank 分配，模型初始化相同、环境 seed 不同；
- durable manifest 比较跨 rank resume/merged/critic state hash；
- 每 rank 保存 sampler/RNG/replay shard，rank 0 原子发布 envelope；
- recovery 保存实际 native binary，避免跨会话重编译后静默冒充 exact resume；
- 聚合 provenance 包含所有 rank seed/deal hash，数据隔离检查支持多 seed；
- source/recovery 路径、schema、manifest 和逐文件 hash fail closed；
- 有限 random/rule evaluation 明确不是 formal selection、champion promotion 或 final holdout。

## 与参考仓库的来源映射

参考 `D:\project\doudizhu` commit `acdcf199cf422e2b683cb22142b8e7cb33b14460` 的 distributed/orchestra/workflow 组织方式；掼蛋 DDP envelope、Kaggle discovery、Notebook、recovery binary 和 telemetry 为目标项目重写。完整映射与许可证见 `M07_SOURCE_AUDIT.md`。

## 已知限制

1. 没有真实 Kaggle kernel、T4、NCCL 或 CUDA 训练证据；
2. 本地 torch 是 CPU build，可见的物理 GPU 未被 PyTorch 使用；
3. pilot/full profile 是 M07 可运行默认，吞吐、显存和 ETA 仍需 M08 定型；
4. 当前 exact resume 只支持相同 world size=2；
5. recovery 保存 native binary，因此 bundle 体积高于只保存 rank 0 模型；
6. bounded evaluation 只证明 E0 管线，不证明模型强度；
7. staging package 的 `workspace_snapshot=true` 仅用于提交前验证；原子 commit 后生成的正式目录必须为 `workspace_snapshot=false` 并再次验证。

## Blockers

无 M07 本地交付 blocker。M08 仍需用户在 Kaggle T4×2 真实运行并下载证据。

## 是否允许启动下一阶段

是。M07 接受语义仅为 `remote_ready`；允许启动 M08 真实 remote pilot，不允许把本报告称为 T4×2 verified、正式强度或可实战结论。
