# A04 观测与批量环境验收记录

## 范围与边界

A04 验收 `guandan.environment` 的策略安全观测、单环境、批环境和检查点语义。它不重定义 A02 规则、A03 Scheme-B 语法或训练损失，也不授权策略侧读取 `full_state()`。

实现边界：

- `guandan.environment` 必须导出冻结的 `GameConfig`、`ObservationSpec`、`Observation`、`StepResult`、`GuandanEnv` 和 `GuandanEnvBatch`；传输 dataclass 位于 `guandan.environment_types`。
- 单环境构造后必须显式 `reset()`；`reset(initial_state=...)` 只接收计数器为零、前缀为空、未终局且未截断的新 A03/A04 快照；进行中或已结束牌局必须通过 `GuandanEnv.deserialize()` 继续。
- 策略观测只包含当前行动座位的私有手牌投影、公共状态、当前前缀和合法 next-token 行。对手/队友手牌、未完成贡还选择、托管区和 A02/A03 对象必须留在特权 debug/checkpoint 通道。
- 单环境和批环境必须按相同 seed、初始快照和 token 序列给出相同结果；批槽位独立，失败写入全批原子回滚，`auto_reset` 只改变结束槽位的后续观测。

## 必须证明的验收条款

### A04-T01：公共 API、固定传输形状和类型

证据必须覆盖：导出及类型重导出、默认 v1 数值、`Observation` 全字段、`StepResult` 全字段、数组形状/数据类型、批首维 B、奖励 `(B,4)`、计数器整数类型、`private_hand_card_ids is None`。对应测试：A04_001–A04_008、A04_028、A04_042。

### A04-T02：reset、fresh fixture 和 continuation 分界

证据必须覆盖：未 reset 不可 step/serialize、seed 和 leader 可复现、已知 seed 0 的 DOUBLE_DOWN 从第一贡牌者开始、A03 fresh bytes 与 A04 fresh bytes 可 reset、带 prefix/非零 counters/已完成 commit 的 payload 不得被 reset 当作新牌局、`deserialize` 保留 continuation。对应测试：A04_004、A04_007–A04_011、A04_037–A04_040、A04_055。

### A04-T03：合法 next-token 集合完整且不可静默截断

证据必须覆盖：lead/follow/tribute/return 各阶段、PASS/COMMIT 标记、PAD/保留区/非法阶段 token 不得出现、无效填充行全零且 mask 为 false、合法 token 无重复、返回牌集合与 A02 `return_cards_for` 精确相等（包括 `Rank.TWO`，并排除所有当前级牌而不只是红桃级牌）、容量不足必须报 `ProtocolError` 而非 top-N 过滤。对应测试：A04_012–A04_017、A04_057–A04_060。

### A04-T04：隐藏信息与贡还回执隐私

证据必须覆盖：交换当前行动座位以外的对手/队友手牌互换不改变完整策略观测；当前座位手牌变化可见；policy dataclass 和 `StepResult` 不含 engine/debug 对象；第一笔贡牌、第一笔还牌在交换未完成时 `card_ids` 为空；最后一笔还牌之后才公开完整的贡牌/还牌实体牌身份。对应测试：A04_018–A04_024、A04_053–A04_054。

### A04-T05：token-step、commit-step、奖励和终止

证据必须覆盖：前缀 token 只推进 `token_step`、不推进 `committed_step`、不改变公共 round state 且奖励为零；COMMIT/PASS/贡/还各只计一次；从 `_checkpoint()` 经过真实合法路径“座位 0 单牌、座位 1 PASS、座位 2 单牌”到 DOUBLE_DOWN 终局；终局 reward 为 `[+1,-1,+1,-1]`、零和、ranking/finish counters 正确；base env 不自动 reset；max-token 截断无终局奖励且终局/截断后继续 step 被拒绝。对应测试：A04_025–A04_032。

### A04-T06：非法输入和完整事务性

证据必须覆盖：非法 token、错误类型、重复物理牌选择、编码/观测失败均不改变任何序列化字节、counter、prefix 或 observation；clone 具有独立状态；返回数组和 debug 字典不与 engine alias。对应测试：A04_033–A04_036、A04_041、A04_057。

### A04-T07：序列化、版本和已知自然红桃级牌回归

证据必须覆盖：中途 prefix 续跑、终局/截断 guard、RNG 后续 reset 流、规则/动作/环境/编码/序列化版本拒绝；seed=2、level=Rank.SIX 的自然红桃 5–9 straight-flush collision 必须 reset 成功，并由自然 canonical token path 只提交一次，不得把等价声明当成不同动作。对应测试：A04_037–A04_040、A04_056。

### A04-T08：批环境等价、独立槽位和 inactive/auto-reset

证据必须覆盖：批/单同轨迹等价、leading dimensions、不同阶段/不同合法行数、槽位无共享 mutable state、`reset_at` 只改目标槽位、inactive slot 只接受 PAD=0、后槽位失败时前槽位不发布、serialized load 原子、`auto_reset` 保留结束 transition 的 flags/reward/counters/info 并返回新牌局 observation，其他 live slot 继续不变。对应测试：A04_042–A04_052、A04_060。

## 证据要求

- 运行时必须使用项目指定解释器，不使用 doudizhu/FableDan：

  `C:\Users\yhx\.conda\envs\guandan_train\python.exe`

- 集中验收命令：

  `C:\Users\yhx\.conda\envs\guandan_train\python.exe scripts/run_acceptance.py --stage A04`

- 本地快速复核命令：

  `C:\Users\yhx\.conda\envs\guandan_train\python.exe -B -m pytest -p no:cacheprovider tests/acceptance/test_a04_environment.py -q`

- 证据必须记录 pytest 的 collected/passed/failed/skipped/error 状态、解释器路径、工作区和完整失败 traceback；不能以 case-ID 数量代替行为证据。
- A04 测试不得用 `pytest.skip` 掩盖已知 seed/phase；已知 seed 0 的贡还路径和 seed 2 的自然红桃级回归必须是硬断言。失败、缺失 API、异常溢出或 privacy 泄漏均不接受。
- 接受前由 parent 复核本文件与测试 diff，运行 A04 集中命令和 full suite；本 worker 不写 `STATUS`、不提交 commit。

## 当前执行记录

本轮 worker 已对 `tests/acceptance/test_a04_environment.py` 完成无 skip 的聚焦复核；正式 acceptance runner 仍应由 parent 在最终工作区运行并保存其报告。实现或测试变化后必须重新生成证据，不得沿用旧结果。
