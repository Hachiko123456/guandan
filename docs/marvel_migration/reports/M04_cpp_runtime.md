# M04 C++/pybind Guandan Runtime 阶段报告

- 日期：2026-10-09
- 分支：`codex/marvel-guandan-migration`
- 开始 commit：`a3177967ba2095d2201a6f97d54895de579905cb`
- 结束 commit：本原子提交自身（自引用 SHA 不嵌入提交内容；以 Git 和最终答复为准）
- 执行环境：Windows 11；Python 3.12.0；NumPy 2.3.5；PyTorch 2.13.0+cpu；MinGW g++ 11.2；CUDA unavailable
- 阶段状态：`accepted`（本地功能/专项验收；M05 未开始）

## 范围

本阶段新增显式 `nanopokerlib.guandan` C++20/pybind backend，重新实现 Guandan 牌、组合、轮次、贡还牌、A03 prefix/COMMIT、V2 observation 和批量环境。实现保留 M02 的 `GD-ACTION-0.2-MARVEL`、`GD-ENV-0.2-MARVEL`、`GD-ENCODING-0.2-MARVEL`，新增 `guandan-marvel-cpp-runtime-v1` 序列化。

`guandan_marvel.game` 仍是默认 `python_reference`。`guandan_marvel.cpp_game` 只是显式 opt-in；本阶段没有切换正式训练默认后端，没有修改 IPPO/VRPO 数学目标，也没有删除 Python oracle 或 legacy checkpoint 评估路径。

## 实际修改文件

### C++ runtime / build

- `nanopokerlib/cpp/core/types.hpp`
- `nanopokerlib/cpp/guandan/guandan.hpp`
- `nanopokerlib/cpp/guandan/guandan.cpp`
- `nanopokerlib/cpp/bindings.cpp`
- `nanopokerlib/__init__.py`
- `nanopokerlib/guandan.py`
- `nanopokerlib/LICENSE`
- `nanopokerlib/NOTICE.md`
- `nanopokerlib/README.md`
- `guandan_marvel/cpp_game.py`
- `setup.py`
- `pyproject.toml`
- `MANIFEST.in`
- `.gitignore`

### 测试/验证

- `tests/marvel_cpp/test_cards_combos.py`
- `tests/marvel_cpp/test_round_protocol.py`
- `tests/marvel_cpp/test_batch.py`
- `scripts/verify_marvel_cpp_runtime.py`

### 规范/报告/证据

- `docs/marvel_migration/M04_CPP_RUNTIME_PROTOCOL.md`
- `docs/marvel_migration/00_README.md`
- `docs/marvel_migration/02_SOURCE_MAPPING.md`
- `docs/marvel_migration/03_TARGET_ARCHITECTURE.md`
- `docs/marvel_migration/04_GAME_INTERFACE_V2.md`
- `docs/marvel_migration/evidence/M04_cpp_runtime/*`
- `docs/marvel_migration/reports/M04_cpp_runtime.md`
- `docs/marvel_migration/reports/M04_cpp_runtime.json`
- `docs/marvel_migration/STATUS.yaml`

## 未修改的受保护范围

以下 A01/A02 规则真值文件相对开始 commit 的 Git blob 未改变：

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `guandan/state.py`
- `docs/rules.md`

`D:\project\doudizhu` 保持只读、clean，HEAD 为 `acdcf199cf422e2b683cb22142b8e7cb33b14460`。目标 C++ Guandan 目录没有复制 `cpp/games/doudizhu/*`；唯一 `Doudizhu` 字样是说明“未复制斗地主游戏代码”的来源注释。

## 实现结果

- 108 张物理牌、13 级牌上下文、有效牌力与两张红桃级牌 wild 已实现。
- 十类牌型、自然/替代 wild、普通牌型可比边界、炸弹顺序和四王已实现。
- 发牌、轮转、PASS、接风、名次补全、三类终局和固定团队奖励已实现。
- 单贡/双贡、最高合法贡牌、还牌限制、抗贡、交换隐私和完成后公开已实现。
- A03 canonical token、唯一 prefix、COMMIT、8-token structured row 和 stale/forged row 拒绝已实现。
- 4-seat `[B,4,48]` event tokens 与 `[B,4,512]` actor channels 已实现；privileged ownership 只通过 `critic_state()` 暴露。
- `EnvBatch` 支持 reset、fixed-deal reset、observe、step、auto reset、debug、critic state 和 byte-stable serialization。
- batch/step 使用 trial copy；非法动作、容量、horizon 和牌守恒错误显式失败，不提交部分状态。

## 测试命令与原始结果

### Build / package

- `python setup.py build_ext --inplace --force`：exit 0；生成并成功导入 `.pyd`。
- `python -m pip wheel . --no-deps --wheel-dir runs/M04_cpp_runtime/20261009_acceptance/wheel`：exit 0；生成 `guandan_train-0.1.0-cp312-cp312-win_amd64.whl`。
- wheel 安装到隔离 target、切换到仓库外目录后导入 `nanopokerlib.guandan` 并 `EnvBatch(2).reset(7)`：exit 0。

### Scoped / regression

- `python -m pytest tests/marvel_cpp -q ...`：41 passed，0 failed，0 skipped，exit 0，9.727s。
- `python -m pytest tests/marvel_adapter tests/marvel_training -q ...`：69 passed，0 failed，0 skipped，exit 0，94.946s。
- A01-A04 protected acceptance：152 passed，0 failed，0 skipped，exit 0，10.439s。
- legacy unit + integration + property：897 passed，0 failed，10 skipped，exit 0，228.502s JUnit / 231.57s wall。

合计：1169 tests；1159 passed；0 failed；0 errors；10 skipped。10 个 skip 全部是既有主机能力限制：2 estimator CUDA、4 rollout CUDA equivalence、1 training objective CUDA、1 training support CUDA、1 Windows file symlink、1 Windows directory symlink。M04 C++ scoped tests无 skip。

完整命令、退出码、JUnit、stdout hash 和 skip 明细见 `test_results.json`；大型日志位于 `runs/M04_cpp_runtime/20261009_acceptance/`，未提交 Git。

## 机器可读证据

- `docs/marvel_migration/evidence/M04_cpp_runtime/protocol_schema.json`
- `docs/marvel_migration/evidence/M04_cpp_runtime/verification.json`
- `docs/marvel_migration/evidence/M04_cpp_runtime/test_results.json`
- `docs/marvel_migration/evidence/M04_cpp_runtime/source_manifest.json`
- `docs/marvel_migration/evidence/M04_cpp_runtime/build_artifacts.json`
- `docs/marvel_migration/evidence/M04_cpp_runtime/artifact_manifest.json`

验证器实际完成：

- 13 个 directed cases，覆盖当前十类牌型、natural/substitution/multiple/mixed wild；
- 3 个完整 Python/C++ 同路径小样本终局，逐 token 比较 action row、channels、events、player、reward 和 terminal；
- batch size 8 的真实独立终局，8/8 slot 至少完成一次；
- byte-stable serialize/deserialize 和 108-card critic conservation。

这些是 M04 专项/smoke 证据，不是 M05 大规模差分。

## 性能结果

本机验证器的 batch-size 8 smoke：2808 token transitions，2.0419s，约 1375.2 token transitions/s。该数字受本机 CPU、随机策略和工作负载影响，只是初步观测；没有与固定 Python workload 做门槛化比值，不能声称 M05 的 >=2x/目标 5x 已通过。

## 正确性与隐私审查

- C++ 与 Python oracle 的所有 directed action declaration/canonical row/event/channel 相等。
- 双贡双还阶段逐步与 Python oracle 相等；未完成 exchange event 只有 actor 可见，完成后四座位同时收到完整 transfer records。
- actor channel self-hand slice 只包含 observer 自己物理手牌；未附加 ownership、escrow、played。
- RETURN 中尚未公开的 incoming return card 对 donor actor channel 隐藏。
- stale/forged metadata、非法 payload、容量溢出和牌守恒错误均为显式异常。

## 与参考仓库的来源映射

通用 `core/types.hpp` 的类型和 batch shape 概念来自 MIT `nanopokerlib` core；C++ Guandan 规则、protocol 和 encoding 是依据本仓库 Python oracle 的新实现。逐文件分类和 hash 见 `source_manifest.json`。

## 已知限制

- M05 的 10,000 完整对局、100,000 中间状态、全 13 级统计差分和序列化后长链差分尚未执行。
- 固定工作负载 Python/C++ >=2x 性能门槛尚未执行；当前 1375.2 transitions/s 不是倍率结论。
- 正式 MARVEL sampler/trainer 默认仍使用 Python reference；C++ 只可显式导入。
- 本地验证是 Windows CPU；没有 Kaggle、T4×2、DDP 或模型强度结论。
- 当前阶段实现单手 Guandan runtime，不扩展为完整多副升级比赛管理器。

## Blockers

无 M04 blocker。M05 仍为 `not_started`，必须执行完整差分和固定工作负载性能门槛后才可接受并切换默认后端。

## 是否允许启动下一阶段

是。M04 硬门槛已满足，可启动 M05；M05 本身尚未通过。
