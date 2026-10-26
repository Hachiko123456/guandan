# DMC 阶段 06A：Gate A 交付修复——m09 bundle 布局不一致（第 1 轮修复）

日期：2026-10-26 · 状态：待执行 · 失败证据：Kaggle run `dmc-gate-a-4163325ccf704c77ccc5947b` 第 27 秒 `M09LaneUnavailable: frozen m09 candidate is missing from the payload: m09/ippo_iter002000_final_candidate.pt`（快速失败按设计工作，仅损失 ~30s 配额）

## 根因（已诊断，不要重新调查）

- **driver 契约**（`tools/dmc_gate_a/m09.py:90-100` 的 `verify_bundle_files`）：全部 m09 文件按 `dataset`（notebook 传入的 `input_dir`）相对寻址。
- **notebook 实际布局**（交付内 `guandan_serial_compare_v1.ipynb` cell 1）：三个归档解包映射为 `('driver.zip', RUN_ROOT/driver), ('overlay_source.zip', SOURCE_ROOT), ('m09_bundle.zip', SOURCE_ROOT)` —— m09 文件落在 `SOURCE_ROOT/m09/…`、`SOURCE_ROOT/nanopokerlib/…`，`input_dir/m09/…` 不存在。
- **为何测试没抓到**：本地预演在"解释器门"（本机非 CPython 3.13.15）被按设计拒绝，**从未执行到 m09 预检**；`tests/dmc/test_gate_a_driver.py` 的 fixture 目录恰好把 m09 文件直接放在 dataset 下，符合 driver 契约但与 notebook 布局矛盾。集成层（notebook 布局 ↔ driver 契约）零测试覆盖。

## 修复范围

1. **notebook（`scripts/build_dmc_gate_a_delivery.py` 内嵌模板）**：`m09_bundle.zip` 解包到**两个目标**——`input_dir`（满足 driver 的 dataset 相对寻址与哈希校验）与 `SOURCE_ROOT`（满足 `import nanopokerlib._guandan_cpp` 与 `check_runtime_identity` 的纯净绝对路径）。两次解包都走既有 `unpack()` 的成员哈希校验。driver 侧契约**不许改**。
2. **新增回归测试（本修复的核心验收）**：`tests/dmc/test_gate_a_driver.py` 增加"Kaggle 布局仿真"用例——用真实交付构建产物（或最小同构 payload），按 notebook cell 1 的**原样解包映射**（driver→driver_root、overlay→source_root、m09→双目标）落盘后，以 `dataset=input_dir` 跑 `m09_precheck` 的文件校验部分（`verify_bundle_files`），断言通过；再断言 `SOURCE_ROOT/nanopokerlib/_guandan_cpp*.so` 存在且哈希一致。布局映射应提取为可测试的共享常量/函数（notebook 与测试同源），禁止两边各写一份。
3. **本地预演口径补齐**：预演必须覆盖"解释器门之后的路径"——用 stub/monkeypatch 跳过解释器比较，继续执行到 m09 预检的文件校验（不含真实原生导入），证明集成路径在本地可达。
4. 重建交付（`delivery_v2`，不覆盖 v1），`verify_dmc_gate_a_delivery.py` 校验 PASS，`kaggle_pipeline.py prepare --delivery … --dry-run` 通过（.venv-kaggle）。

## 明确禁止

- 不改 `m09.py` 的 dataset 相对寻址契约、不放松任何哈希/身份校验；
- 不动 4 项门槛、预算、并行拓扑（那些都没问题）；
- 不重新上传（由复核者/用户执行）。

## 验收

- 新增布局仿真测试 + 既有 47 项（driver/builder/verifier/parallel）全绿；
- 重建交付三哈希（payload/notebook/plan）更新进 REPORT，plan_hash 变化属预期（新交付身份）；
- `runs/dmc_gate_a_repair1_<date>/REPORT.md` 含根因、修复、测试证据、NOT_RUN 清单。
