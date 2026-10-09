# M05 差分与性能补修报告

- 日期：2026-10-09
- 分支：`codex/marvel-guandan-migration`
- 原始实现 commit：`4ba65bfa20df2843fbc3cefb5299e885c14e2479`
- 补修实现 commit：`758b2e97fd3bf0103fa1046a72bf6ed4a8cf3600`
- 阶段状态：`blocked`（正确性与实现补修通过，真实性能门槛未通过）
- 下一阶段：M06 保持 blocked

## 范围

本轮复现并修复默认 C++ backend 的循环导入、导入顺序和公共模块身份问题；把实际 backend、backend-specific source 和加载的 C++ binary/build identity 纳入 checkpoint 恢复判定；保留原 10k/100k corpus 并执行 hash/计数/13-level replay；修正原 2.976× benchmark 的描述并新增真实 frozen Transformer Agent rollout；修复 A00 中文主标题失败。

未修改 A01/A02 规则真值、IPPO/VRPO 数学目标、原始 corpus、原 benchmark JSON 或原阶段报告；未实施 M06、Kaggle、DDP 或长训练。

## 1. 导入顺序故障复现与修复

在原始 commit 的隔离源码副本中真实复现：

- `import guandan_marvel`：默认 C++ 和 selector 均有 `EnvBatch`；
- `from nanopokerlib import guandan; import guandan_marvel`：native module 完整，但 package-level C++ 和 selector 的 `EnvBatch` 均缺失，输出为 `True False False`；
- package `game` 为 `guandan_marvel.cpp_game`，直接导入 `guandan_marvel.game` 为 Python oracle，二者身份不一致；
- `guandan_marvel.python_game` 不是可导入模块；
- 无扩展时，包初始化在显式选择 Python oracle 前即失败。

修复采用 extension-free lazy package API，不重载 `sys.modules`、不 reload、不吞掉异常：

- `default_game` / `get_game_backend()`：默认 C++；
- `cpp_game` / `get_game_backend("cpp")`：显式 C++；
- `python_game` / `get_game_backend("python_reference")`：显式 Python oracle；
- 历史 `game` 与 `import guandan_marvel.game`：始终为 Python oracle。

先导入 native 或先导入 package 均得到完整 C++。Config-only import 不加载扩展。无扩展隔离安装中，显式 Python oracle 正常；C++ 请求抛 `BackendUnavailableError`，无 fallback。

## 2. Checkpoint backend 与构建身份

Checkpoint schema 升级为 `guandan-marvel-training-checkpoint-v4`，semantic contract 升级为 `guandan-marvel-training-contract-v3`。

新 checkpoint 保存并校验：

- 实际 `backend`；
- hash-protected `backend_identity`；
- 实际 backend 对应的 source fingerprint；
- C++ extension path、binary SHA-256、embedded native-source digest；
- compiler、C++ standard、Python implementation/version/cache tag；
- checkout 有源码时，当前 native source manifest 与 embedded digest 是否一致。

最终本地扩展：

- binary SHA-256：`ce6bb30501ca8967776ba7bbbc11bdd7e7f0255bf496d240f62d23551c103d00`；
- embedded source SHA-256：`40058234af8227e9f0a0aa8c555959717e277f4ac16ed5094a918a32d46f6898`；
- source/build match：true；
- compiler：GCC 11.2.0，C++20。

实际 C++ checkpoint bounded 验证：

- same backend/source/build：`exact_resume`，并继续到 iteration 2；
- 仅 logging/run name 变化：`compatible_resume`；
- C++ checkpoint -> Python trainer：拒绝并 poison；
- extension binary identity 变化：拒绝并 poison；
- 失败实例不能继续训练或保存。

允许变化仍限于既有非语义白名单；backend、binary/build、wrapper/header/native source、semantic contract、采样/优化语义变化均不得 exact resume。旧 v3 缺 backend identity：训练为 migration，显式评估为 evaluation-only，不自动补全。

## 3. 强制重建与隔离 wheel

执行：

- `python setup.py build_ext --inplace --force`：exit 0；
- `python -m pip wheel . --no-deps --no-build-isolation --no-cache-dir`：exit 0；
- 最终 wheel：`guandan_train-0.1.0-cp312-cp312-win_amd64.whl`；
- wheel SHA-256：`74e9c38093dd892ad1461e4cd878dc2a65619323fda4a3a367c31ff18f69fec4`。

从仓库外 isolated target 验证以下场景全部 exit 0：package-first、native-first、config-only 不加载扩展、删除 wheel 内 native artifact 后显式 Python 可用且 C++ 明确失败。

## 4. 原 10k/100k 证据保留与本轮重放

原证据未修改：

- 10,000 完整局；
- 3,243,977 actions；
- 3,253,977 compared states；
- 100,000 个已比较轨迹中的随机非终局状态子集；
- 10,000 serialization roundtrips；
- 0 mismatch。

本轮验证：

- 全部 6 个 corpus shard 的 SHA-256 与大小匹配；
- 解压计数仍为 10,000 records / 3,243,977 actions；
- 从原 corpus 完整回放全 13 level，trajectory digest 全部匹配；
- directed 全牌型、wild、rank bomb、straight flush、four kings 重跑通过；
- 四观察者隐私重排重跑通过；
- 规则文件、Python adapter、C++ `guandan.cpp/.hpp` 相对原实现不变。

因此不机械重跑完整 10k；本轮证据明确分为继承的大规模结果和新执行的抽样/专项回放。

## 5. 性能口径修正

原 2.976× lane 使用：

`position = (case.seed + 17 * step_index) % legal.numel()`

其准确含义是 deterministic selector / fixed-action-rule + observation conversion + environment rollout，**没有执行 Transformer Agent**。原 JSON 和数值保留，但不再称为 frozen Transformer。

### 新真实 frozen Transformer Agent rollout

配置：同一组冻结 Transformer 权重、`eval()` + `torch.inference_mode()`、greedy temperature=0、同一物化 deal、CPU 单线程、4 完整局、3 次测量、交替 backend 顺序。每次均执行 1,027 个完全相同动作，动作/奖励 digest：

`76f3c5a4ff937e1afa035a2b0962644a945fe03f11a17fb9dae1f25d2b75453c`

| backend | wall samples (s) | median (s) | actions/s |
|---|---|---:|---:|
| Python | 6.311765 / 6.309451 / 6.348970 | 6.311765 | 162.71 |
| C++ | 4.015675 / 3.963421 / 3.993576 | 3.993576 | 257.16 |

真实 Agent rollout speedup：**1.580×**。

代表性分段：

- Python：environment 3.325 s；model forward+KV 2.433 s；conversion 0.060 s；sampling/marshalling 0.177 s；buffer/commit 0.219 s。
- C++：environment 1.110 s；model forward+KV 2.370 s；conversion 0.066 s；sampling/marshalling 0.165 s；buffer/commit 0.204 s。

环境段显著加速，但相同 Transformer forward 成为主要瓶颈，完整真实 rollout 未达到 2×。

### 完整 IPPO iteration

相同 deal/model/seed、338 env steps、151 committed steps：

- total speedup：1.060×；
- rollout speedup：1.092×；
- learner ratio：1.037×。

原 1.075× 结果未隐藏；本轮重新测量仍表明完整训练没有 2×。

## 6. 性能验收决策

总体计划把 end-to-end rollout 放入 >=2× 比较，而原 lane 不含真实模型。补修协议在运行前采用保守口径：真实 frozen Transformer rollout 必须 >=2× 才能重新 accepted，learner 本身不要求 2×。

实测 1.580× < 2×，因此本轮**不能重新接受 M05**。这不是正确性失败，而是未满足保守的真实模型端到端性能门槛。M06 必须保持 blocked。

## 7. A00 标题修复

仅修改以下四个文档首标题，正文逐行不变：

- `docs/marvel_migration/adr/ADR-005-006-marvel-core-namespace.md`
- `docs/marvel_migration/M02_ADAPTER_PROTOCOL.md`
- `docs/marvel_migration/M03_TRAINING_PROTOCOL.md`
- `docs/marvel_migration/M04_CPP_RUNTIME_PROTOCOL.md`

A00–A04 protected acceptance：157 passed。完整 `--import-mode=importlib`：1394 passed，10 skipped，A00 标题失败已消除。原证据中的旧 hash 继续表示历史版本；本轮 `document_revisions.json` 保存 old/new hash 和 title-only 证明。

## 8. 测试结果

Blocking commands：

1. Backend import/checkpoint identity：10 passed，exit 0。
2. M05 + C++ scoped：118 passed，exit 0。
3. M01–M03 affected regression：86 passed，exit 0。
4. A00–A04 protected acceptance：157 passed，exit 0。

主要不重叠集合：361 passed，0 failed，0 skipped。

完整回归：

- `python -m pytest --import-mode=importlib -q`：1394 passed，10 skipped，exit 0。
- 默认 `python -m pytest -q`：exit 2，原因是既有两个 `test_training_smoke.py` basename 在默认 import mode 下发生 collection identity collision；不是测试失败。两种行为和日志分别保留，未混算。

正式运行前曾有一次 Python wrapper 未镜像所有公共常量导致的 19-test 失败，已保留日志并在最终 86-test rerun 中全部通过。

## 9. 机器可读证据

目录：`docs/marvel_migration/evidence/M05_differential_performance_repair/`

大型日志、wheel、checkpoint 和完整输出：`runs/M05_differential_performance_repair/20261009_repair_review/`

## 10. 修改范围

主要新增/修改：

- lazy backend API：`guandan_marvel/__init__.py`、`backends.py`、`python_game.py`、`cpp_game.py`；
- native identity：`nanopokerlib/build_identity.py`、`guandan.py`、`cpp/bindings.cpp`、build header、`setup.py`；
- checkpoint v4：`training/checkpoints.py`、`training/local.py`、artifact verifier 和专项测试；
- sampler optional timing instrumentation；
- repair benchmark/import/corpus/checkpoint scripts；
- M05 repair protocol、checkpoint compatibility、架构/接口/风险文档；
- 四个 A00 标题；
- STATUS 和本报告/证据。

未修改受保护规则语义文件或 C++ 规则引擎 `guandan.cpp/.hpp`。

## 11. Blocker 与下一阶段

Blocker：真实 frozen Transformer end-to-end rollout 仅 1.580×，低于保守 2× 门槛。

结论：

- M05：`accepted=false`；
- M06：blocked；
- 不允许启动 M06、Kaggle、DDP 或正式训练；
- 后续需明确批准调整门槛口径，或继续优化真实 Agent rollout 至 >=2× 后再验收。
