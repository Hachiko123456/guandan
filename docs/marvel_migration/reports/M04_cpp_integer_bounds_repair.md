# M04 C++ Action Row 整数边界补修报告

- 日期：2026-10-09
- 分支：`codex/marvel-guandan-migration`
- 原始 M04 实现：`75735031721a44304ffe4d7be7fc1532fadd0ba6`
- 前次补修实现：`ff0d1d097d8989ff9037c303493002a2a4c1388a`
- 前次验收登记：`1703456d0de1b0643513a37fcc7fd8f45482a7ef`
- 本次补修实现：待第一笔提交后登记
- 本次验收登记：待第二笔提交
- 状态：`locally_verified_pending_identity_registration`

## 范围

本轮仅修改 C++ action row 原始整数范围校验，并新增针对性测试、验证、构建和报告。没有修改 Guandan Python 规则、通用 Sampler、IPPO/VRPO、RNG 算法或恢复策略，没有切换默认 backend，也没有开始 M05。

## 原始问题复现

在前次已验收实现上，取合法 payload token `8` 并替换为：

```text
4294967304  = 8 + 2**32
-4294967288 = 8 - 2**32
```

Python reference 的 raw/decoded 两条入口均抛 `ProtocolError`，环境序列化保持不变。

原 C++ 的两条入口均错误接受：

```text
actions = cpp.actions_from_np(bad_rows)
env.step(actions)

env.step(bad_rows)
```

两个非法值均在 `static_cast<int>` 后变回合法 token `8`，动作被实际执行，环境状态改变。复现见 `review_findings.json`。

## 修复

`nanopokerlib/cpp/bindings.cpp::parse_action_row()` 现在在任何 C++ `int` 窄化前：

- signed NumPy integer 无损读取为 `std::int64_t[8]`；
- unsigned NumPy integer 无损读取为 `std::uint64_t[8]`；
- 对八个原始值全部检查 `0 <= value < NUM_TOKENS`；
- 全部通过后才转换为 C++ `int`。

unsigned 路径不经过 `int64_t`，所以 `INT64_MAX+1` 和 `UINT64_MAX` 不会在检查前溢出。范围检查后仍执行原有 framing、phase、stage、kind、payload、COMMIT、decoded metadata 和 live-candidate 唯一匹配校验。

没有裁剪、取模、饱和转换或 silent fallback。

## 八字段和边界覆盖

八个 row 位置均覆盖：

- signed `原值 + 2**32`；
- signed `原值 - 2**32`；
- unsigned `原值 + 2**32`。

共 24 个字段/偏移组合；raw rows 和 decoded actions 两条入口全部在解码阶段拒绝，环境 bytes 不变。

另覆盖 13 个整数边界：

- `-1`；
- `NUM_TOKENS`；
- int32 min/max；
- int64 min/max；
- `INT64_MAX+1`；
- `UINT64_MAX`；
- signed/unsigned `2**32 + token`；
- negative `2**32` offset。

测试数组均先断言 Python/NumPy 中保存的原始值与目标值相同，未在测试构造阶段预先溢出。

## Batch 原子性和合法兼容

两 slot 测试中 slot 0 合法、slot 1 使用越界 `uint64`。整个 batch 失败，所有 slot 的 state、prefix、RNG 和 counters 的 serialize bytes 完全不变。

合法 `int32`、`int64`、`uint32`、`uint64` 的 raw/decoded transition 相同；非连续和非本机字节序合法 arrays 也按数值正确解析。

Python/C++ 对同一正负 `2**32` payload 均拒绝且环境不变。

## 构建和 wheel

强制重建：

```text
python setup.py build_ext --inplace --force
```

工具：

- Python：`C:\Users\yhx\AppData\Local\Programs\Python\Python312\python.exe`
- pybind11：2.13.6
- compiler：`C:\mingw11.2.0\bin\g++.exe`，11.2.0
- gendef：`C:\mingw11.2.0\bin\gendef.exe`
- dlltool：`C:\mingw11.2.0\bin\dlltool.exe`

工作区扩展：

```text
D:\project\guandan\nanopokerlib\_guandan_cpp.cp312-win_amd64.pyd
SHA-256 54d54e1afb372059c7a796f61fedfa5484dffd42f595368646d53055b7ebc34b
```

对应 `bindings.cpp`：

```text
SHA-256 50cac479a57c4a2409c0824510bf1986f58ab382cc23428a4e956090be3b7bc3
```

Wheel：

```text
guandan_train-0.1.0-cp312-cp312-win_amd64.whl
SHA-256 56914e40f56267344569fdb94ab546f61048658cfd3562d6793ccb3744c4ca12
```

隔离 wheel 从仓库外加载新扩展，合法 decoded step 和 serialize roundtrip 通过；正负 `2**32` 及 `UINT64_MAX` 在 raw/decoded 两条入口全部拒绝。

## 本轮测试

- 新整数边界测试：45 passed，0 skipped，exit 0；
- `tests/marvel_cpp`：109 passed，0 skipped，exit 0；
- 真实 Agent + C++ sampler 专项：2 passed，0 skipped，exit 0；
- RNG/counter/serialization 专项：13 passed，0 skipped，exit 0；
- M01/M02/M03 回归：81 passed，0 skipped，exit 0；
- A01-A04 protected acceptance：152 passed，0 skipped，exit 0。

非重复主验收集合为 `109 + 81 + 152 = 342 passed`，0 failed、0 skipped。专项 45/2/13 已包含在 109 个 C++ tests 中，单独重跑用于保存聚焦证据。

前次补修的 legacy unit/integration/property 897 passed、10 environment skips 本轮仅核对历史证据，未描述为本轮重新运行。

## 修改文件

- `nanopokerlib/cpp/bindings.cpp`
- `tests/marvel_cpp/test_action_integer_bounds.py`
- `scripts/verify_marvel_cpp_integer_bounds_repair.py`
- `docs/marvel_migration/M04_CPP_INTEGER_BOUNDS_REPAIR_PROTOCOL.md`
- `docs/marvel_migration/00_README.md`
- `docs/marvel_migration/04_GAME_INTERFACE_V2.md`
- `docs/marvel_migration/STATUS.yaml`
- `docs/marvel_migration/reports/M04_cpp_integer_bounds_repair.*`
- `docs/marvel_migration/evidence/M04_cpp_integer_bounds_repair/*`

## 受保护范围

相对前次验收登记 commit，以下文件 Git blob 未改变：

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `guandan/state.py`
- `docs/rules.md`
- `guandan_marvel/runtime/sampler.py`

参考仓库仍为 `acdcf199cf422e2b683cb22142b8e7cb33b14460` 且 clean。

## 机器证据

- `review_findings.json`
- `protocol_schema.json`
- `verification.json`
- `repair_summary.json`
- `test_results.json`
- `source_manifest.json`
- `build_artifacts.json`
- `artifact_manifest.json`

## 当前结论

最小代码修复和全部要求的本地验证已经通过。当前等待第一笔实现提交生成稳定 SHA，再由第二笔登记重新验收。在身份登记完成前 M04 保持 `accepted=false`，M05 保持 blocked。
