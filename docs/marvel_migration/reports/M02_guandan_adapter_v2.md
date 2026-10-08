# M02 Guandan MARVEL V2 Python Adapter 阶段报告

- 日期：2026-10-08
- 分支：`codex/marvel-guandan-migration`
- 开始 commit：`d4638ab4eb7886335c8f438deadcaff12066f1bf`
- 结束 commit：`71500c5ac580cfda935d7bbd52de6a3793110c60`
- 执行环境：Windows 11；Python 3.12.0；PyTorch 2.13.0+cpu；NumPy 2.3.5；CUDA unavailable
- 阶段状态：`accepted`

## 范围

实现 `GD-ACTION-0.2-MARVEL`、`GD-ENV-0.2-MARVEL`、`GD-ENCODING-0.2-MARVEL` Python reference adapter：四座位 actor information sets、8-token 结构化候选 row、增量 event、512 current-state channels、complete-trajectory sampler/buffer wire、版本化序列化和独立 privileged critic labels。

规则、合法动作、COMMIT、贡还牌、终局和奖励继续由已验收 V1 A01-A04 Python oracle 执行。未写 C++，未替换 legacy/default 正式训练后端，未进行训练或强度比较。

## 实际修改文件

- 新增 `guandan_marvel/game.py`：V2 game module、schema/constants、`EnvBatch`、`actions_from_np()`、actor/critic privacy boundary、hard-fail capacities、serialization。
- 新增 `guandan_marvel/env_batch.py`：NumPy V2 wire 到 MARVEL TensorDict 的转换。
- 新增 `guandan_marvel/runtime/sampler.py`：complete-trajectory sampler。
- 更新 `guandan_marvel/buffer.py`：增加经 TensorDict 验证的 trajectory slice assignment。
- 更新 `guandan_marvel/__init__.py` 和 `runtime/__init__.py`：导出 game/sampler。
- 新增 `tests/marvel_adapter/test_adapter.py`：13 个 M02 tests。
- 新增 `scripts/verify_marvel_adapter_v2.py`：完整局/capacity/等价扫描。
- 新增 `M02_ADAPTER_PROTOCOL.md`、`M02_SOURCE_AUDIT.md`、协议/测试/扫描证据和本阶段报告。
- 更新 `02_SOURCE_MAPPING.md`、`04_GAME_INTERFACE_V2.md`、`10_RISK_AND_DECISIONS.md`、`STATUS.yaml`。

## 未修改的受保护范围

未修改：

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `guandan/state.py`
- `docs/rules.md`

未删除或替换 legacy V1 trainer/model/checkpoint evaluator。`D:\project\doudizhu` 审计前后均 clean，参考 HEAD 保持 `acdcf199cf422e2b683cb22142b8e7cb33b14460`。

## 测试命令与原始结果

1. M02 adapter + M01 core：

`python -m pytest -q tests/marvel_adapter tests/marvel_core --disable-warnings --junitxml=runs/M02_guandan_adapter_v2/test_logs/m02_scoped.junit.xml`

结果：exit 0；25 passed，0 failed，0 errors，0 skipped；44.271s。

2. A01-A04 protected regression：

`python -m pytest -q tests/acceptance/test_a01_cards.py tests/acceptance/test_a02_rules.py tests/acceptance/test_a02_a03_defect_fix.py tests/acceptance/test_a03_actions.py tests/acceptance/test_a04_environment.py --disable-warnings --junitxml=runs/M02_guandan_adapter_v2/test_logs/legacy_a01_a04.junit.xml`

结果：exit 0；152 passed，0 failed，0 errors，0 skipped；10.906s。

3. Legacy unit + integration：

`python -m pytest -q tests/unit tests/integration --disable-warnings --junitxml=runs/M02_guandan_adapter_v2/test_logs/legacy_unit_integration.junit.xml`

结果：exit 0；889 passed，0 failed，0 errors，10 skipped；223.294s。skip 为既有可选依赖/环境 skip，非 M02 新增跳过。

4. M02 verification scan：

`python scripts/verify_marvel_adapter_v2.py --output docs/marvel_migration/evidence/M02_guandan_adapter_v2/verification.json`

结果：exit 0；15 个完整真终局、13 个 level ranks、2 个贡还牌局；4447 token steps、2809 commits、23121 candidate rows；0 diff/illegal/conservation failure；序列化 continuation 已检查。

5. Hygiene：

`git diff --check && git diff --cached --check`：exit 0。

完整 stdout/JUnit/exit code 位于 `runs/M02_guandan_adapter_v2/test_logs/`（被 `.gitignore` 排除）；hash 和计数见提交的 `test_results.json`。

## 机器可读证据

- `docs/marvel_migration/evidence/M02_guandan_adapter_v2/protocol_schema.json`
- `docs/marvel_migration/evidence/M02_guandan_adapter_v2/verification.json`
- `docs/marvel_migration/evidence/M02_guandan_adapter_v2/test_results.json`
- `docs/marvel_migration/evidence/M02_guandan_adapter_v2/source_manifest.json`

## 性能结果

仅做本地 CPU 功能扫描：15 局共 4447 token steps，扫描脚本约 30.37s。该数字包括 Python rule enumeration、序列化 continuation 和证据检查，不能描述为正式吞吐 benchmark、高性能后端、模型强度或 Kaggle 远端性能。

## 正确性与隐私审查

- 每个 V1 legal-next token 在 V2 恰好对应一条 row；整行 stale/forged 校验后才执行。
- V2 选择后底层 V1 serialized state、commit flag、reward 和 terminal 一致。
- 每步输出四座位 actor information sets；四玩家 opponent/team hand swap 测试通过。
- 未完成 tribute/return 只向选择者发送 private event；其他座位 event 全 PAD，RETURN target 尚未公开的 incoming card 从 actor hand channels 中隐藏。
- full ownership 只从独立 `critic_state()` 返回，未附加到 actor `EnvStep`。
- observation 为增量 event；扫描最大 25 token，未重复 V1 4096-token history。
- action/observation/horizon/buffer overflow 均显式失败，不截断合法候选、不伪造终局。
- 4 个专项完整局（含贡还牌）和 15 局扫描均验证真实 terminal、奖励、名次和 108 张牌守恒。
- MARVEL sampler 完成 296-step 真终局，TrajectoryBuffer 可写入和采样。

## 与参考仓库的来源映射

- `guandan_marvel/env_batch.py` 派生自 `marvel/env_batch.py`。
- `guandan_marvel/runtime/sampler.py` 派生自 `marvel/runtime/sampler.py`。
- 两者保留 SPDX、参考 commit 和来源路径；MIT LICENSE/NOTICE 已随 M01 分发。
- `guandan_marvel/game.py` 是 Guandan 专用新实现，不含斗地主规则。
- 详细映射见 `M02_SOURCE_AUDIT.md`。

## 已知限制

- backend 仅为 `python_reference`；不是高性能后端。C++/pybind 属于 M04。
- M02 sampler 只验证完整 trajectory wire；IPPO/VRPO 更新、checkpoint resume 和训练 smoke 属于 M03。
- 扫描策略是确定性 commit-first 功能策略，不是模型、强度或实战证据。
- `max_available_actions=256` 是 V1 legal-token 词表硬上界；本阶段实际扫描最大 27，不是 M05 大规模差分/性能证据。
- 默认正式训练 backend 未切换；legacy V1 wire/checkpoint evaluator 保留。

## Blockers

M02 无 blocker。M03 必须在本接口上实现本地 IPPO/VRPO update/resume/nonfinite/replay smoke；不得把本阶段功能轨迹称为训练完成。

## 是否允许启动下一阶段

是。M02 硬门槛已满足，可启动 M03_marvel_training_smoke。此结论只代表本地 adapter/trajectory 正确性验收，不代表 C++、Kaggle 或模型强度验收。
