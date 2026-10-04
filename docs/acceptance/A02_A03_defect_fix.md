# A02/A03 缺陷修复验收记录

- **记录日期：** 2026-10-05
- **规则基线：** `docs/rules.md`（GD-RULES-0.1）
- **目的：** 对 `docs/prerequisite_defects_A05_A06.md` 中的 A02/A03 已知问题进行可审计修复或显式隔离；本记录不改写历史验收证据，也不把局部冒烟测试当作完整规则验证。
- **本记录状态：** `passed`（仅表示下列缺陷修复证据通过，不等于重写 A00-A04 历史 `accepted` 状态）
- **A05/A06 阻塞政策：** A05/A06 的完整规则前置仍需按 `STATUS.yaml` 和本记录边界审查；本轮没有修改训练算法，也没有把规则测试通过表述成模型变强。

## 验收矩阵

| ID | 原始缺陷 | 正式规则来源 | 实际修改文件 | 可观察通过条件 | 测试命令 | 证据文件 | 状态 | 是否阻塞 A05/A06 |
|---|---|---|---|---|---|---|---|---|
| A02-RANK | `effective_rank_value(Rank.TWO, Rank.TWO) == 0`，且有效牌力顺序在级牌、比较、炸弹/连牌边界可能不一致。 | `docs/rules.md` §3.2、§4.1、§5.1–§5.2 | `guandan/cards.py`, `guandan/combos.py`, `guandan/round.py`, `tests/unit/test_cards.py`, `tests/unit/test_combos.py`, `tests/unit/test_defect_fixes.py` | `level_rank=TWO` 时 `3..A < TWO < 小王 < 大王`；非 2 级牌仍只有一个级牌位置；对子/炸弹比较与贡牌排序共享有效牌力；连牌仍使用自然窗口。 | `python -m pytest -q tests/unit/test_cards.py tests/unit/test_combos.py tests/unit/test_round.py tests/unit/test_defect_fixes.py`；退出码 0，35 passed。 | `project_status/history/20261005T000000000000Z_A02_A03_defect_fix/acceptance_evidence.json` | `passed` | 否（该缺陷已修复；A05/A06 仍受其他发布门槛约束） |
| A02-TRIBUTE | 贡牌候选排除了王，且逢人配/级牌/王优先级未由统一规则定义。 | `docs/rules.md` §7.2，尤其第 134–136 行 | `guandan/round.py`, `tests/unit/test_defect_fixes.py`, `tests/acceptance/test_a04_environment.py` | 最高非逢人配实体牌候选包含王；逢人配不作为贡牌；级牌强度、抗贡、双副牌实体 card_id 和转移状态可观察。 | 同上；A04 贡牌专项也通过；退出码 0。 | 同上；另见 `project_status/history/20261005T000000000000Z_A02_A03_defect_fix/report.json` | `passed` | 否（该缺陷已修复；不得据此宣称 A05/A06 完整规则覆盖） |
| A02-RETURN | 还牌只排除红桃级牌，未排除所有当前级牌。 | `docs/rules.md` §7.3、第 142 行及 §12 第 212 行 | `guandan/round.py`, `tests/unit/test_defect_fixes.py`, `tests/acceptance/test_a04_environment.py`, `docs/acceptance/A04_observation_and_batch_environment.md` | 普通当前级牌、红桃当前级逢人牌、王和普通 2..10 按正式规则处理；非法还牌原子失败；贡还往返后牌数守恒。 | `python -m pytest -q tests/acceptance/test_a04_environment.py -k 'A04_016 or A04_017 or A04_024'`；退出码 0，3 passed，78 deselected；A01–A04 集中命令退出码 0，152 passed。 | 同上 | `passed` | 否（该缺陷已修复；A05/A06 完整规则门槛仍需单独审查） |
| A02-FULL-GAMES | 旧的“10,000 seed”测试只提交单牌和少量 PASS，不是 10,000 局真实终局对局。 | `docs/completion_criteria.md` A02；`docs/acceptance/A02_round_rules_engine.md` §16–§20 | `scripts/run_full_game_regression.py`, `tests/acceptance/test_a02_a03_defect_fix.py`, 本记录 | 每个 seed 从发牌开始，经合法贡/还/出牌动作到终局；记录终局、受控 truncation、非法动作、牌数守恒、最终手牌数、奖励/排名及贡还执行。 | `python scripts/run_full_game_regression.py --count 32 --report project_status/history/20261005T000000000000Z_A02_A03_defect_fix_scoped.json`；退出码 0，32/32 终局；`python scripts/run_full_game_regression.py --count 10000 --report project_status/history/20261005T000000000000Z_A02_A03_defect_fix/report.json`；退出码 0，10000/10000 终局、0 truncation、0 非法动作、0 failed checks。 | `project_status/history/20261005T000000000000Z_A02_A03_defect_fix/report.json`；摘要见 `acceptance_evidence.json` | `passed` | 否（旧证据未改写；A05/A06 仍不能据此自动 accepted） |
| A03-CANONICAL | seed=2、`level_rank=SIX` 等局面出现自然/逢人配声明差异却 canonical token 冲突。 | `docs/acceptance/A03_stepwise_action_protocol.md` §§4、6、8–10；`docs/rules.md` §3.3 | `guandan/combos.py`, `tests/unit/test_combos.py`, `tests/acceptance/test_a02_a03_defect_fix.py`, `tests/property/test_action_protocol_defect_regression.py` | 同 rank/suit 的自然/替代声明按正式等价规则稳定 canonicalize；语义不同动作不合并；seed=2/SIX 的 token 序列唯一；每个 COMMIT 唯一；非 COMMIT 不改公开状态；短动作 encode/commit/re-encode round-trip 稳定。 | `python -m pytest -q tests/acceptance/test_a02_a03_defect_fix.py`；退出码 0，3 passed；`python -m pytest -q tests/property/test_action_protocol_defect_regression.py`；退出码 0，1 passed；A01–A04 集中命令退出码 0，152 passed。 | 同上 | `passed` | 否（缺陷已修复；不等于 A05/A06 模型/训练质量证明） |

## 实际修改文件

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `docs/acceptance/A02_A03_defect_fix.md`
- `docs/acceptance/A04_observation_and_batch_environment.md`
- `scripts/run_full_game_regression.py`
- `tests/unit/test_cards.py`
- `tests/unit/test_combos.py`
- `tests/unit/test_defect_fixes.py`
- `tests/acceptance/test_a02_a03_defect_fix.py`
- `tests/acceptance/test_a04_environment.py`
- `tests/property/test_action_protocol_defect_regression.py`

## 运行与范围边界

- `python -m pytest -q tests/acceptance/test_a01_cards.py tests/acceptance/test_a02_rules.py tests/acceptance/test_a03_actions.py tests/acceptance/test_a04_environment.py tests/acceptance/test_a02_a03_defect_fix.py`：退出码 0，152 passed。
- `python -m pytest -q tests/unit/test_cards.py tests/unit/test_combos.py tests/unit/test_round.py tests/unit/test_action_protocol.py tests/unit/test_defect_fixes.py tests/property/test_card_conservation.py tests/property/test_round_invariants.py tests/property/test_action_protocol_property.py tests/property/test_action_protocol_defect_regression.py`：已作为 A01–A04 相关集中运行的一部分执行；集中命令退出码 0。
- `python -m pytest --collect-only -q`：退出码 0，1019 tests collected。
- `python -m pytest -q`：未完成。命令在仓库全量测试约 28% 处由本代理主动中断，退出码 1；这不是通过或失败的完整全量结论，剩余约 72% 未验证。原因是全量套件包含长时间训练/评估路径，不能把部分进度写成全量通过。
- `git diff --check`：退出码 0，无 whitespace error。

## A00–A04 与 A05/A06 状态说明

- 没有修改 `project_status/STATUS.yaml`，没有删除或改写 `project_status/history/` 下既有历史证据；A00–A04 的历史 `accepted` 字段未被本轮重新声称或伪造。
- 本记录的 `passed` 只表示已列出的 A02/A03 缺陷修复证据通过，不改变阶段验收门槛。
- A05/A06 仍不能被表述为具备完整规则覆盖：`STATUS.yaml` 中的 `remote_full` 门槛仍未完成，且本轮全量 pytest 未跑完。不得据本轮测试声称模型变强。
- 不存在需要“隔离而不是修复”的 A02/A03 规则缺陷；未完成的仓库全量 pytest 是 `unverified` 范围，不是静默跳过。

## 建议

在保留本记录和新 JSON 证据的前提下，建议在独立资源窗口重新运行完整 `python -m pytest -q`，再由主代理复核差异、测试和 A05/A06 的 `local_ready`/`remote_full` 状态。A05/A06 训练/评估可继续准备，但在规则覆盖门槛解除前不应宣称完整规则验证或模型质量提升。

## 证据索引

- `project_status/history/20261005T000000000000Z_A02_A03_defect_fix/acceptance_evidence.json`
- `project_status/history/20261005T000000000000Z_A02_A03_defect_fix/report.json`
- `project_status/history/20261005T000000000000Z_A02_A03_defect_fix_scoped.json`
