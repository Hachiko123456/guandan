# M04 C++ Runtime 补修与重新验收报告

- 日期：2026-10-09
- 分支：`codex/marvel-guandan-migration`
- 原始实现 commit：`75735031721a44304ffe4d7be7fc1532fadd0ba6`
- 补修实现 commit：待第一笔实现提交后登记
- 验收登记 commit：待第二笔登记提交
- 状态：`locally_verified_pending_identity_registration`
- 环境：Windows 11；Python 3.12.0；PyTorch 2.13.0+cpu；NumPy 2.3.5；pybind11 2.13.6；MinGW g++ 11.2；CUDA unavailable

## 范围

本轮只补修 M04 的公共动作接口和 C++ batch 反序列化验证，并重新执行 M01-M04 受影响回归。没有开始 M05，没有执行 10k/100k 差分、>=2x 性能门槛、正式训练、Kaggle 或 DDP。

原始 M04 报告、证据和日志保持不变；补修结果写入独立的 `_repair` 报告和 evidence 目录。

## 原始问题复现

### 1. `actions_from_np()` 与 C++ `step()` 不兼容：已复现

Python reference：

```text
actions = game.actions_from_np(rows)
env.step(actions) -> success
```

原始 C++：

```text
actions = cpp.actions_from_np(rows)
env.step(actions) -> TypeError: step() accepts numpy.ndarray only
```

相同调用进入现有 MARVEL Sampler 后，真实小型 Transformer Agent 在第一步 `environment_step` 失败，sampler 按既有 `poison_on_failure` 契约标记 `failed=true`、`exact_resume_allowed=false`。

### 2. 损坏 RNG 和异常 trajectory counter 被接受：已复现

原始 C++ `deserialize()` 接受：

- `rng_state="NOT_A_VALID_RNG_STATE"`；
- `trajectory_steps=[-1]`；
- game `token_step=0` 但 `trajectory_steps=[123]`。

非法 RNG 再次 serialize 时变成另一份默认/部分解析状态，没有加载错误。

原始复现见 `review_findings.json` 和 `runs/M04_cpp_runtime_repair/20261009_pre_repair/`。

## 动作接口修复

C++ `EnvBatch.step(action_input)` 现在同时接受：

1. NumPy integer structured rows `[B,8]`；
2. `actions_from_np(rows)` 返回的 `Sequence[V2Action]`。

两条路径统一验证完整 `ActionRow`：framing、phase、prefix stage、kind、payload、COMMIT flag、vocabulary，以及 decoded object metadata 与 row 的一致性。随后仍要求完整 row 与当前 slot 的 live candidate 唯一匹配，不能只取 `token_id`。

raw/decoded single-slot 和 multi-slot transition 完全相同。shape/dtype/count/stale/forged 输入显式失败；多 slot 任一非法时 trial batch 不发布，序列化内容保持不变。

没有修改通用 Sampler，也没有增加 backend 特判；修复发生在 C++ backend 的公共 game interface。

## 反序列化修复

RNG restore 现在要求：

- `std::mt19937_64` extraction 成功；
- 空、非法、截短文本拒绝；
- extraction 后只允许空白；非空垃圾后缀拒绝。

Batch/game restore 新增：

- batch games/events/trajectory 长度一致；
- trajectory counter 非负、不超过 horizon；
- `trajectory_step == game.token_step`；
- game `max_token_steps == batch.max_horizon`；
- level/previous-result 与 config 一致；
- committed step/history、token step/canonical history+prefix 一致；
- terminal finish counters 一致；
- 不接受 truncated EnvBatch snapshot；
- `auto_reset=true` 不接受内部残留 terminal game。

Wire 未变化，serialization schema 保持 `guandan-marvel-cpp-runtime-v1`。有效 payload byte-stable；原实现错误接受的非法 payload 现在明确拒绝。

## 真实 Agent + C++ Sampler

使用现有 `guandan_marvel.runtime.sampler.Sampler`、真实 `Agent` v1.3 小型 Transformer（8/8 hidden，1 layer/head）、`purpose=on_policy` 和显式 C++ game module：

- on-policy 首步成功，`on_policy_valid=true`，sampler 未 failed；
- memory preflight：2 rollout + 2 collect 小样本低于 128 MiB；
- 第一次 collect 两条真实完整轨迹：289、320 token steps；
- 第二次连续 collect：351、339 token steps；
- 每条 trajectory 恰有一个 terminal，位于最后 step；
- terminal reward 为固定团队 `{-1,+1}`；
- policy/behavior probability finite；
- 两 slot 错峰终局；
- auto-reset 后 player IDs 仍为 live seats；
- trajectory 与下一局 rollout 未混合；
- sampler `failed=false`，计数持续增长。

未执行 optimizer update 或正式训练。

## 序列化与 RNG continuation

- 中途 prefix serialize/deserialize byte-stable，恢复后与原实例逐 step 相等并到达相同真实终局。
- RNG continuation 测试在 serialize 后继续到 terminal，并实际触发 `reset_auto()`；新一副四手牌完全相同，post-reset `token_step=0`，序列化 bytes 相同。
- 正常 RNG 尾随空白可接受并 canonicalize；非空尾随内容在加载时拒绝。

## 强制构建和 wheel

强制重建：

```text
python setup.py build_ext --inplace --force
```

工具：

- Python：`C:\Users\yhx\AppData\Local\Programs\Python\Python312\python.exe`
- compiler：`C:\mingw11.2.0\bin\g++.exe`，11.2.0
- gendef：`C:\mingw11.2.0\bin\gendef.exe`
- dlltool：`C:\mingw11.2.0\bin\dlltool.exe`

完整 gendef/dlltool/g++ 命令在 `runs/M04_cpp_runtime_repair/20261009_repair_acceptance/build.log`。扩展、源码和 wheel hash 见 `build_artifacts.json`。

新 wheel 安装到全新隔离目录，从仓库外 cwd 导入，确认加载隔离 `.pyd`；`step(actions_from_np(rows))`、reset/step/serialize/deserialize 均通过。去除扩展后导入抛 `ModuleNotFoundError`，没有 Python fallback。

## 测试结果

- `tests/marvel_cpp`：64 passed，0 skipped，exit 0，25.151s；
- M01/M02/M03 (`tests/marvel_core tests/marvel_adapter tests/marvel_training`)：81 passed，0 skipped，exit 0，100.630s；
- A01-A04 protected acceptance：152 passed，0 skipped，exit 0，10.303s；
- legacy unit/integration/property：897 passed，10 skipped，exit 0，225.002s JUnit / 228.11s wall。

合计：1204 tests；1194 passed；0 failed；0 errors；10 skipped。10 个 skip 均为既有 CUDA/symlink 主机能力限制；补修测试无新增 skip。

原 13 个复杂动作、3 个 Python/C++ 完整小样本对局、batch-size 8 真终局/auto-reset 已重新运行通过。本地吞吐观测不是 M05 性能倍率结论。

## 修改文件

代码/构建：

- `nanopokerlib/cpp/bindings.cpp`
- `nanopokerlib/cpp/guandan/guandan.cpp`
- `setup.py`

新增测试/验证：

- `tests/marvel_cpp/test_repair_action_interface.py`
- `tests/marvel_cpp/test_repair_deserialize.py`
- `tests/marvel_cpp/test_repair_sampler.py`
- `scripts/verify_marvel_cpp_runtime_repair.py`

规范/状态/报告：

- `docs/marvel_migration/M04_CPP_RUNTIME_REPAIR_PROTOCOL.md`
- `docs/marvel_migration/00_README.md`
- `docs/marvel_migration/04_GAME_INTERFACE_V2.md`
- `docs/marvel_migration/STATUS.yaml`
- `docs/marvel_migration/reports/M04_cpp_runtime_repair.*`
- `docs/marvel_migration/evidence/M04_cpp_runtime_repair/*`

## 受保护范围和来源

`guandan/cards.py`、`combos.py`、`round.py`、`state.py`、`docs/rules.md` 相对原始 M04 commit 的 blob 未改变。`guandan_marvel/runtime/sampler.py` 和 IPPO/VRPO 数学代码未修改。参考仓库仍为 `acdcf199cf422e2b683cb22142b8e7cb33b14460` 且 clean。

## 机器证据

- `review_findings.json`
- `protocol_schema.json`
- `repair_verification.json`
- `original_scope_rerun.json`
- `repair_summary.json`
- `test_results.json`
- `source_manifest.json`
- `build_artifacts.json`
- `artifact_manifest.json`

## 已知限制

- M05 10,000 完整局、100,000 中间状态和 >=2x 固定工作负载门槛尚未执行。
- 默认 backend 仍是 `python_reference`。
- 本地 Windows CPU build 不构成 Linux/Kaggle/T4×2 证据。
- 本轮不产生模型强度、正式训练或实战结论。

## 当前结论

代码和验证硬门槛已本地通过，但本报告尚未登记补修实现的稳定 commit SHA。第一笔实现提交完成后，由第二笔验收登记提交写入准确 SHA，并决定 M04 是否重新 `accepted`。当前不得启动 M05。
