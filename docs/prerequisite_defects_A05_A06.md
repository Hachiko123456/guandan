# A05/A06 前置阶段缺陷审计

2026-10-04，主代理在 `deb351a` 基线上复现。没有因为旧 accepted 标记忽略这些问题；本轮不重启 A00–A04，也不修改其牌局实现。

| 来源 | 具体复现 | 违反/风险 | 本轮影响与限制 |
|---|---|---|---|
| A01 级牌强度 | `effective_rank_value(Rank.TWO, Rank.TWO)==0`，A 为 12 | rules.md 明确级别 2 应高于 A | A05/A06 固定级别 FIVE；不据本机报告宣称级别 2 已验证 |
| A02 贡牌 | seed=0, SIX, PreviousHandResult((1,2,3,4),0,HEAD_THIRD)，座位3有王106/107，却选69 | 贡牌最高非逢人规则包含王，现有选择排除所有王 | 本机 counted runs 使用 standalone 第一副，无上局贡还；该问题待专门规则修复 |
| A02 还牌 | 上例贡牌后 return_cards_for 中有级牌13/67/69 | rules.md 末尾v1政策排除所有当前级牌，原实现只排除红桃级牌 | 同上；不要以 A05/A06 证据证明贡还符合性 |
| A02 验收证据 | test_ten_thousand_seeded_bounded_legal_rounds 只提交单牌和三次PASS | 不是 10,000 次完整终局对局；CASE_IDS 名称也不等于真实fixture覆盖 | 保留 accepted 历史但不把它当作新的完整规则证明；本轮实际终局独立记录 |
| A03 声明碰撞 | StepwiseActionState.deal(seed=2, level_rank=SIX).legal_next_tokens 抛相同 canonical token 冲突 | 相同自然红桃同花顺有两种自然flag声明 | A04 _CachedProtocol 有显式等价去重；A05/A06只经A04，不能声称原始A03已修复 |

本审计不将缺陷静默消除、降级为通过或篡改既有证据。local_ready 只代表本轮明确范围的训练/评估管线计数、GAE、奖励视角、checkpoint与错误处理。remote_full 及规则问题审查仍是发布前待办，accepted 保持 false。
