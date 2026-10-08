# M01 MARVEL 通用核心阶段报告

- 日期：2026-10-08
- 分支：`codex/marvel-guandan-migration`
- 开始 commit：`44efa2fd28536cb9863c6173ea3fbc92626df506`
- 结束 commit：本报告所属的单一原子提交（自引用 SHA 不能写入其自身内容；实际 SHA 见 Git HEAD 与最终答复）
- 执行环境：Windows 11；Python 3.12.0；PyTorch 2.13.0+cpu；NumPy 2.3.5；CUDA unavailable
- 阶段状态：`accepted`

## 范围

完成与具体斗地主规则无关的 MARVEL 通用核心迁移；建立独立命名空间、版本化配置/缓存/checkpoint/日志协议、fake game 形状测试、来源/许可证记录和 CPU 本地回归。未接入掼蛋 V2 环境，未运行正式训练。

参考仓库在实施前后均为 clean，只读审计 commit 为 `acdcf199cf422e2b683cb22142b8e7cb33b14460`（`Isolate Dou Dizhu routines`）。读取的 `LICENSE` 为 MIT，Copyright (c) 2026 AtaraxosAI；原始和随包副本 SHA-256 均为 `c40411b9f09e2927c13b18331fee1d3f2caad5dcac4200c3280c198b8c05c95b`。

## 实际修改文件

- 新增 32 个 `guandan_marvel/**` 文件：配置/GameSpec、Transformer/KV cache、Agent/Critic、TensorDict、optimizer/Muon、trajectory buffer、versioned checkpoint、compile/DDP gate、logging、GAE/loss/schedule、test-only fake game、MIT LICENSE/NOTICE。
- 新增 `tests/marvel_core/test_core.py`，覆盖 12 个 M01 核心测试。
- 新增 `docs/marvel_migration/M01_SOURCE_AUDIT.md`：逐文件来源、改写程度、残留假设、M02 接口和延后模块。
- 新增 `docs/marvel_migration/adr/ADR-005-006-marvel-core-namespace.md`。
- 新增机器证据：`docs/marvel_migration/evidence/M01_marvel_core/verification.json`、`source_manifest.json`。
- 更新 `02_SOURCE_MAPPING.md`、`10_RISK_AND_DECISIONS.md`、`STATUS.yaml` 和 `pyproject.toml`。

完整逐文件映射见 `M01_SOURCE_AUDIT.md` 和 `evidence/M01_marvel_core/source_manifest.json`。

## 未修改的受保护范围

未修改以下规则真值或其语义：

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `guandan/state.py`
- `docs/rules.md`

未删除或重命名 legacy V1 trainer/model/checkpoint evaluator。未复制 `nanopokerlib/cpp/games/doudizhu`、斗地主 evaluator agent 或角色/叫地主规则。

## 测试命令与原始结果

1. M01 core：

```text
python -m pytest -q tests/marvel_core --disable-warnings --junitxml=runs/M01_marvel_core/test_logs/marvel_core.junit.xml
exit 0; 12 passed, 0 failed, 0 errors, 0 skipped; 14.42s
```

2. legacy unit + integration：

```text
python -m pytest -q tests/unit tests/integration --disable-warnings --junitxml=runs/M01_marvel_core/test_logs/legacy_unit_integration.junit.xml
exit 0; 889 passed, 0 failed, 0 errors, 10 skipped; 222.71s
```

10 个 skip 均为既有环境能力限制：CPU host 无 CUDA（8 个）或 Windows host 不支持测试所需 symlink（2 个）；不是 M01 新增 skip。

3. A01-A04 受保护规则/协议 acceptance：

```text
python -m pytest -q tests/acceptance/test_a01_cards.py tests/acceptance/test_a02_rules.py tests/acceptance/test_a02_a03_defect_fix.py tests/acceptance/test_a03_actions.py tests/acceptance/test_a04_environment.py --disable-warnings --junitxml=runs/M01_marvel_core/test_logs/legacy_a01_a04.junit.xml
exit 0; 152 passed, 0 failed, 0 errors, 0 skipped; 10.84s
```

4. wheel/clean interpreter path：

```text
python -m pip wheel . --no-deps --wheel-dir runs/M01_marvel_core/wheel
python -m pip install --no-deps --target runs/M01_marvel_core/install <built-wheel>
python -I -c <prepend isolated install path; import guandan and guandan_marvel; fake Agent/Critic forward>
exit 0; Agent [2,3,4]; Critic [2,3,4,4]
```

最终 wheel：`runs/M01_marvel_core/wheel/guandan_train-0.1.0-py3-none-any.whl`，226,507 bytes，SHA-256 `b9b165335ef351a2cc9c5d6ec7f69c5d288b13fdff168c9a7f267018797a595b`。wheel 包含 `guandan_marvel/LICENSE` 和 `NOTICE.md`，不提交 Git。

5. Git/静态门禁：

```text
git diff --check
git diff --cached --check
```

最终均须 exit 0；结果记录在 `verification.json` 和 `runs/M01_marvel_core/test_logs/`。

## 机器可读证据

- `docs/marvel_migration/evidence/M01_marvel_core/verification.json`
- `docs/marvel_migration/evidence/M01_marvel_core/source_manifest.json`
- 大日志/JUnit：`runs/M01_marvel_core/test_logs/`（ignored，不提交）
- wheel/隔离安装：`runs/M01_marvel_core/{wheel,install}/`（ignored，不提交）

## 性能结果

无性能或模型强度结论。M01 只执行 CPU 正确性、打包和导入验证；没有 GPU、DDP throughput、Kaggle 或正式训练证据。

## 正确性与隐私审查

- `Agent` 只读取每座位 observation tensor；`CentralCritic` 保留 centralized `[P]` 输入能力，但 M02 尚未定义真实 actor/critic 信息边界。
- `GameSpec` 只描述容量/协议身份，不包含任何斗地主或掼蛋规则语义。
- buffer horizon overflow hard fail，不截断。
- checkpoint 强制 `guandan-marvel-checkpoint-v1`；未版本化 payload 拒绝加载。
- compile 默认关闭；显式启用时不可用会报错；`eager` backend 已实际执行测试。
- DDP 默认关闭；`WORLD_SIZE>1` 与关闭配置冲突、不完整 torchrun 环境、无 CUDA 请求均明确失败，无 silent fallback。
- fake game 标记为 `fake-core-test-only`，仅位于 `guandan_marvel.testing`，不作为掼蛋 adapter。

## 与参考仓库的来源映射

完整映射见 `docs/marvel_migration/M01_SOURCE_AUDIT.md`。核心结论：

- 通用模型、KV cache、loss/GAE、optimizer 以 MIT 来源迁移并改 namespace；
- 配置、buffer、checkpoint、distributed、logger 做了较大泛化/重写；
- 删除 `nanopokerlib`/斗地主默认值和游戏 module 自动解析；
- env batch、sampler、workflow、IPPO/VRPO trainer、evaluation hook 分别延后到 M02/M03/M06，避免机械复制游戏专用假设。

## 已知限制

- 尚无 Guandan V2 adapter、真实 V2 token/channel/action semantics 或完整终局 trajectory。
- 尚无 sampler、IPPO/VRPO trainer/workflow 或正式训练 CLI；这不是训练完成证据。
- DDP 仅验证配置 fail-closed；本 CPU host 未初始化真实多 GPU process group。
- compile 只验证 `eager` backend 和开关契约；未验证 inductor 动态 shape 性能。
- centralized critic 的真实训练期 full-state privacy contract 必须在 M02 定义并测试。
- legacy 回归的 10 个环境 skip 未被 M01 解除；其原因已在 JUnit 记录。

## Blockers

M01 无 blocker。M02 需要实现并验证版本化 Guandan `GameSpec`、actor information set、critic full state、逐 token/COMMIT 与容量上界；不得复用 fake game 作为实现。

## 是否允许启动下一阶段

是。M01 硬门槛已满足；`STATUS.yaml` 将 M02 解锁为 `not_started`。这仅允许启动 M02，不表示掼蛋已经接入 MARVEL、模型可实战或远端验收完成。
