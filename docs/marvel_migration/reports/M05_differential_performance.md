# M05 Python/C++ 差分与性能门槛阶段报告

- 日期：2026-10-09
- 分支：`codex/marvel-guandan-migration`
- 开始 commit：`c1fecb05cf37f57ad8462d8c43d1a5b7bb0868c0`
- 结束 commit：由本报告所在的单一原子提交确定；SHA 在最终回复和 Git 历史中报告（提交不能包含自身 SHA）
- 执行环境：Windows 11；Python 3.12.0；NumPy 2.3.5；PyTorch 2.13.0+cpu；AMD 6C/12T；14.89 GB RAM；RTX 3050 Ti 4 GB 已枚举但当前 PyTorch 无 CUDA
- 阶段状态：`accepted`

## 范围

本阶段建立可重放的匹配 deal/state/action corpus，对 Python reference 与 C++ backend 执行逐状态规则、协议、四座位观测、privileged private state、reward、terminal、serialization/resume 和完整 trajectory digest 差分；执行同机同工作量 single-env、batch、冻结策略端到端 rollout 和 learner 不变完整 IPPO iteration benchmark；通过后把 C++ 切换为正式训练默认，同时保留显式 Python oracle 和 no-silent-fallback 后端选择。

没有修改 A01/A02 规则语义，没有删除 legacy V1 训练器、模型或 checkpoint 只读评估能力，没有开始 Kaggle 正式训练。

## 实际修改文件

### 运行时与选择

- `guandan_marvel/__init__.py`
- `guandan_marvel/backends.py`
- `guandan_marvel/cpp_game.py`
- `guandan_marvel/training/local.py`
- `guandan_marvel/training/smoke.py`
- `guandan_marvel/testing/complex_actions.py`
- `guandan_marvel/testing/differential.py`

### 验证、benchmark 与测试

- `scripts/verify_marvel_m05_differential.py`
- `scripts/benchmark_marvel_m05.py`
- `tests/marvel_differential/test_m05_differential.py`
- 现有 M02/M04 显式 Python oracle import 更新：
  - `scripts/report_marvel_m03_memory_budget.py`
  - `scripts/verify_marvel_adapter_v2.py`
  - `scripts/verify_marvel_cpp_runtime.py`
  - `scripts/verify_marvel_sampler_contract.py`
  - `tests/marvel_adapter/test_adapter.py`
  - `tests/marvel_adapter/test_complex_actions.py`
  - `tests/marvel_adapter/test_memory_budget.py`
  - `tests/marvel_adapter/test_sampler_contract.py`
  - `tests/marvel_cpp/test_batch.py`
  - `tests/marvel_cpp/test_round_protocol.py`

### 文档、状态和证据

- `docs/marvel_migration/M05_DIFFERENTIAL_PERFORMANCE_PROTOCOL.md`
- `docs/marvel_migration/00_README.md`
- `docs/marvel_migration/03_TARGET_ARCHITECTURE.md`
- `docs/marvel_migration/04_GAME_INTERFACE_V2.md`
- `docs/marvel_migration/10_RISK_AND_DECISIONS.md`
- `docs/marvel_migration/STATUS.yaml`
- `docs/marvel_migration/reports/M05_differential_performance.md`
- `docs/marvel_migration/reports/M05_differential_performance.json`
- `docs/marvel_migration/evidence/M05_differential_performance/*`

## 未修改的受保护范围

相对开始 commit，以下文件 SHA-256 全部不变：

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `guandan/state.py`
- `docs/rules.md`

只读参考仓库 `D:\project\doudizhu` 保持 `acdcf199cf422e2b683cb22142b8e7cb33b14460` 且 clean。

## 测试命令与原始结果

### 正式 blocking tests

1. `python -m pytest tests/marvel_differential tests/marvel_cpp -q --junitxml=.../m05_scoped_rerun.junit.xml`
   - exit 0；113 passed；0 failed；0 skipped；34.92 s。
2. `python -m pytest tests/marvel_core tests/marvel_adapter tests/marvel_training -q --junitxml=.../m01_m04_regression.junit.xml`
   - exit 0；81 passed；0 failed；0 skipped；99.74 s。
3. `python -m pytest tests/acceptance/test_a01_cards.py tests/acceptance/test_a02_rules.py tests/acceptance/test_a02_a03_defect_fix.py tests/acceptance/test_a03_actions.py tests/acceptance/test_a04_environment.py -q --junitxml=.../a01_a04_regression.junit.xml`
   - exit 0；152 passed；0 failed；0 skipped；10.36 s。

Blocking 集合合计 346 passed，0 failed，0 skipped。

### 透明保留的非 blocking 诊断

- 首次 PowerShell wrapper 因参数名 `$args` 被 shell 自动变量遮蔽，实际误执行 `python -m pytest`，exit 2；日志保留，随后用正确参数重跑全部正式命令并通过。
- 全套诊断：`python -m pytest --import-mode=importlib -q ...`
  - exit 1；1383 passed，1 failed，10 skipped；413.77 s。
  - 唯一失败为未修改且开始 commit 已存在的 `docs/marvel_migration/adr/ADR-005-006-marvel-core-namespace.md` 英文首标题触发 A00 中文标题检查；本阶段未修改该文件。该诊断不替代且不削弱正式 A01-A04 与 M01-M05 blocking 回归结果。

完整日志、JUnit、exit code 和 SHA-256 在 `runs/M05_differential_performance/20261009_acceptance/tests`；索引见 `test_results.json`。

## 机器可读证据

- `docs/marvel_migration/evidence/M05_differential_performance/differential_summary.json`
- `docs/marvel_migration/evidence/M05_differential_performance/benchmark_report.json`
- `docs/marvel_migration/evidence/M05_differential_performance/corpus_manifest.json`
- `docs/marvel_migration/evidence/M05_differential_performance/mismatch_manifest.json`
- `docs/marvel_migration/evidence/M05_differential_performance/protocol_schema.json`
- `docs/marvel_migration/evidence/M05_differential_performance/verification.json`
- `docs/marvel_migration/evidence/M05_differential_performance/test_results.json`
- `docs/marvel_migration/evidence/M05_differential_performance/source_manifest.json`
- `docs/marvel_migration/evidence/M05_differential_performance/environment.json`
- `docs/marvel_migration/evidence/M05_differential_performance/artifact_manifest.json`

大型完整 corpus 与测试日志位于 `runs/M05_differential_performance/20261009_acceptance`，不提交 Git；6 个 gzip shard 总计约 8.4 MB，每个均有 committed SHA-256 manifest。

## 性能结果

同一机器、匹配 deal、相同 action row 或相同冻结策略、相同全局 transition count：

| workload | Python | C++ | speedup | gate |
|---|---:|---:|---:|---:|
| single-env 完整局，5,150 transitions | 414.98 step/s | 1,334.17 step/s | **3.215×** | pass |
| batch=16，4,096 transitions | 144.31 step/s | 313.58 step/s | **2.173×** | pass |
| 冻结策略端到端 rollout，4,400 transitions | 350.55 step/s | 1,043.08 step/s | **2.976×** | pass |

硬门槛取三条 backend-sensitive lane 的最小值：**2.173× >= 2.0×**。5× 目标未达到，因此不宣称 5×。

完整 learner 不变 IPPO iteration 使用相同 deal、模型 state、seed、338 token steps、151 committed steps：

- Python total 4.172 s；rollout 2.445 s；learner 1.701 s；
- C++ total 3.882 s；rollout 2.119 s；learner 1.742 s；
- total 1.075×；rollout 1.154×；learner ratio 0.977×。

该完整 iteration 真实公开模型/learner bottleneck，不被写成 >=2×；硬 gate 按阶段协议只使用已隔离且工作量完全匹配的 backend-sensitive lanes。CPU time、peak RSS、分段时间和 GPU inventory 均在 benchmark JSON 中。没有 GPU 空转工作负载。

## 正确性与隐私审查

正式差分命令：

`python scripts/verify_marvel_m05_differential.py --games 10000 --intermediate-states 100000 --workers 6 --output-dir runs/M05_differential_performance/20261009_acceptance`

结果：

- 10,000 / 10,000 完整终局；
- 3,253,977 states compared；
- 3,243,977 actions executed；
- 100,000 随机非终局中间状态；
- 10,000 serialize/deserialize 后续轨迹；
- 0 mismatch；0 mismatch fixture；
- 全 13 level；
- 十类组合全部覆盖；
- four kings 314 次、rank bomb 37,297 次、straight flush 12,960 次；
- natural wild 817 次、substituted wild 19,183 次；
- tribute/return 各 10,321 次；anti-tribute 1,249 局；contact 7,166 次；
- DOUBLE_DOWN 4,443、HEAD_THIRD 1,855、HEAD_LAST 3,702；
- 四观察者 hidden-hand reorder 全部 Python 内、C++ 内和跨 backend 一致；
- 完整 trajectory digest 全部一致。

Corpus 首条已独立 replay：game 0，317 actions，digest `ce70566d9bb77024304ece70e8703125ca2268f572a46840e8ed86925dcda18a`。

## 与参考仓库的来源映射

M05 没有从 `D:\project\doudizhu` 复制新的规则或游戏代码。继续使用 M01/M03 已登记的 MARVEL 通用 sampler/model/learner 来源；本阶段新增的是掼蛋本地 differential、replay corpus、benchmark 和 backend selector。许可证和既有 NOTICE 未改变。

## 已知限制

1. Python 与 C++ 内部发牌 PRNG 算法不同；相同整数直接调用两边 `reset(seed)` 不产生相同物理 deal。差分和 benchmark 使用 corpus seed 生成并物化同一 deal，再通过 Python initial state 与 C++ `reset_from_deals` 注入。规则/协议等价已证明，但不宣称跨实现 PRNG bitstream 等价。
2. 5× 目标未达到；正式最低实测为 2.173×。
3. learner 不变完整 iteration 只有 1.075×，因为相同模型推理和 learner 占主导；已分段公开，未混入环境加速结论。
4. 本机 PyTorch 为 CPU build，RTX 3050 Ti 未用于 benchmark；无 Kaggle/T4×2 结论。
5. 全套诊断仍有一个与本阶段无关、开始 commit 已存在的 A00 文档标题失败；正式 blocking scopes 全通过。
6. 这些结果只证明规则/协议正确性和本机吞吐，不证明模型强度或可实战。

## Blockers

无 M05 blocker。

## 是否允许启动下一阶段

是。M05 的 10k/100k 正确性、隐私、trajectory、serialization 和 >=2× backend-sensitive 性能硬门槛全部满足；M06 可以启动。不得跳到 Kaggle 正式训练。
