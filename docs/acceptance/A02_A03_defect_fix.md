# A02/A03 缺陷修复验收记录

- **记录日期：** 2026-10-05
- **规则基线：** `docs/rules.md`（GD-RULES-0.1）
- **目的：** 对 `docs/prerequisite_defects_A05_A06.md` 中的 A02/A03 已知问题进行可审计修复或显式隔离；本记录不改写历史验收证据，也不把局部冒烟测试当作完整规则验证。
- **初始状态：** `pending`
- **A05/A06 阻塞政策：** 在 A02/A03 本记录中所有适用项目取得真实证据前，A05/A06 仍被规则缺陷阻塞；不得仅凭训练/评估冒烟结果解除阻塞。

## 验收矩阵

| ID | 原始缺陷 | 正式规则来源 | 修改涉及的文件（计划/实际） | 可观察通过条件 | 测试命令 | 证据文件 | 状态 | 是否阻塞 A05/A06 |
|---|---|---|---|---|---|---|---|---|
| A02-RANK | `effective_rank_value(Rank.TWO, Rank.TWO) == 0`，且有效牌力顺序在级牌、比较、炸弹/连牌边界可能不一致。 | `docs/rules.md` §3.2、§4.1、§5.1–§5.2 | `guandan/cards.py`, `guandan/combos.py`, `guandan/round.py`, 相关测试 | 对 `level_rank=2` 及至少一个非 2 级牌，普通牌、级牌、小王、大王顺序一致；顺子/同花顺按自然窗口；同点数组/炸弹比较与进贡排序共用明确语义。 | 待实现后填写 | 待生成的 pytest/JSON 报告 | `pending` | 是 |
| A02-TRIBUTE | 贡牌候选排除了王，且逢人配/级牌/王优先级未由统一规则定义。 | `docs/rules.md` §7.2，尤其第 134–136 行 | `guandan/round.py`, 相关测试 | 候选集合包含最高非逢人实体牌；王可按规则成为贡牌；逢人配不作为贡牌；抗贡、双下配对、实体 card_id 和转移状态可审计。 | 待实现后填写 | 待生成的贡还回归报告 | `pending` | 是 |
| A02-RETURN | 还牌只排除红桃级牌，未排除所有当前级牌。 | `docs/rules.md` §7.3、第 142 行及 §12 第 212 行 | `guandan/round.py`, 相关测试 | 对普通当前级牌、红桃当前级逢人牌、普通 2..10、王分别得到规则一致的合法性；非法还牌被拒绝，往返后牌数守恒。 | 待实现后填写 | 待生成的贡还回归报告 | `pending` | 是 |
| A02-FULL-GAMES | 旧的“10,000 seed”测试只提交单牌和少量 PASS，不是 10,000 局真实终局对局。 | `docs/completion_criteria.md` A02；`docs/acceptance/A02_round_rules_engine.md` §16–§20 | `tests/acceptance/...`, `scripts/run_full_game_regression.py` | 每个 seed 从发牌开始；记录终局/受控 truncation、非法动作、牌数守恒、最终手牌数、奖励/排名、贡还执行；scoped 与 release/full 范围明确。 | 待实现后填写 | `project_status/history/<run-id>/A02_A03_defect_fix/` | `pending` | 是 |
| A03-CANONICAL | seed=2、`level_rank=SIX` 等局面出现自然/逢人配声明差异却 canonical token 冲突。 | `docs/acceptance/A03_stepwise_action_protocol.md` §§4、6、8–10；`docs/rules.md` §3.3 | `guandan/action_state.py`, `guandan/combos.py`, 相关测试 | 找到最小复现；语义不同动作保留，公开结果完全等价动作按稳定规则去重；每个合法前缀唯一确定下一步；COMMIT 一对一；非 COMMIT 不改公开状态；encode/decode 往返稳定。 | 待实现后填写 | 待生成的 A03 回归/property 报告 | `pending` | 是 |

## 证据记录

实现和测试完成后，补全实际修改文件、完整命令、退出码、测试数量、证据路径、覆盖范围和未验证内容。`accepted`、A05/A06 状态和历史报告不得在没有真实证据时修改。

## 当前边界

- 不修改 A05/A06 训练算法，不访问 Kaggle，不接入 Guance。
- 不删除、改名、缩小或静默跳过既有测试。
- 如果某条规则无法安全确定，必须保留 `unsupported`/`blocker` 隔离和回归测试，而不是把状态写成 `passed`。
