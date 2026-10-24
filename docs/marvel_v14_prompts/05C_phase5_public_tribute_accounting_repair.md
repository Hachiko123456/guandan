# 阶段5定向修复 Prompt：公开贡还牌当前归属与 Baseline B 记账

更新日期：2026-10-19。本文件为05C定向修复入口，不是新增阶段；完成本轮不代表阶段5已通过，也不进入阶段6。

## 0. 本次明确授权与禁止项

用户执行本Prompt，即授权在当前代码上核验并修复“公开已知但仍在手中的牌”在B0公开信息特征/Baseline B中的记账问题，并补反例、工程回归和一次有界16局CUDA pilot。

**明确允许最小修改生产Baseline B、公开信息特征、公开事件tracker以及对应schema/design定义。** 这不是仅修fixture的任务，不必因“生产公式需要改变”再次弹窗让用户在物理不一致选项中选择。修改必须有独立规则依据，保留旧设计与数据身份，不能悄悄改完冒用旧Gate。

本轮不授权512局完整B0、A/B/C/D真实probe训练、B1、对称性训练、完整全量suite、正式长训、Kaggle、双T4/NCCL、final holdout、FableDan正式评估、阶段6或重建正式delivery。阶段5原目标仍以05A为准，本轮只解决下面的公开信息记账阻塞。

一次非预期失败立即停止并等待用户，不自动修复重跑。本Prompt不是对未来新failure的预先解锁授权。

## 1. 必须完整读取的输入

项目根目录：`D:/project/guandan`。

1. `CLAUDE.md`、`.claude/agents/phase-implementer.md`、`.claude/skills/phase/SKILL.md`。
2. `docs/marvel_v14_prompts/00_local_gpu_execution_contract.md`。
3. `docs/marvel_v14_prompts/05A_symmetry_belief_gpu_execution.md`、`05B_phase5_resume_after_collection_repair.md`：理解原阶段5目标和失败停止规则，不从头实施。
4. `runs/phase5/review_checks/repair3_audit/audit.json`：Codex对真实pilot的只读统计与反例。
5. `runs/phase5/pilot/resume_pilot16_after_repair3.json`及其`.samples.jsonl`。
6. `runs/phase5/execution_guard/b9623635147e4e9e82cc510f121955c3.json`和`.log`、`5708a3fc1a5e421b8ebe6a6f10702c5f.json`和`.log`：上轮99项工程测试与pilot只作为历史证据，不是新修复的成绩。
7. 真实实现：`guandan_marvel/phase5/{belief_data,belief_collect,belief_probe,b0_runner,design}.py`及相关测试；Python/C++规则和编码文件用于只读核验事件语义。

旧pilot样本文件的SHA256：

```text
5ca3ba9512e791d2b8beb79404af38523d88c7da84483785ab78d31d87cf17b0
```

先核对文件身份、guard状态和当前工作区，记录tracked diff及未跟踪源码hash；不靠mtime或tracked diff单独证明一致。若有并发写入或非预期failure锁，停止报告，不清锁、不换Agent绕过。

## 2. 已证实的反例；不要把推测当根因

上轮原7项失败的最小/工程测试已通过，当前无须重做那些修复。但报告“真实采集必然U≥R”的保证已被其自己的数据推翻。

按旧18维public特征定义：

```text
R = sum(public[0:3])       # 三个相对座位的当前手牌张数
U_old = sum(public[3:18])  # 按旧规则扣掉所有已观察实体牌后的计数
```

对固定128个pilot样本重新统计，应得到：

```text
R = U_old : 93
R < U_old : 7
R > U_old : 28
```

这些不是可随便排除的“非法数据”，其实体守恒审计通过，R>U_old出现在DOUBLE_DOWN贡还context。

具体反例（旧samples.jsonl第62行，1-based）：

```text
deal_index=52, token_step=18, actor=2
relative_remaining=[27,27,21], R=75, U_old=72
printed-rank index=14（按项目编码核对王的名称）
old unseen count=0
oracle relative-seat counts=[0,1,0]
old Baseline B prediction=[0,0,0]
audit.conserved=true
```

“实际持有”和“公开已知持有”不能混为一谈：用因果事件轨迹独立核实其中哪些牌/归属在该时刻已经公开。**不能仅凭oracle非零，就宣称所有这类牌必须由B确定预测。** 需要证明的是当前public tracker和B是否正确使用了本应可见的贡还归属。

初步代码证据：`_public_cards_from_event_row`把PUBLIC_COMMIT与EXCHANGE_RESOLVED中的CARD统一加入一个集合；`public_unseen_by_rank`扣掉该集合；B仅按此unseen分配，没有显式区分仍在手中的公开已知牌与已出牌。此链路是待确认根因，不是可以跳过规则审计的结论。

## 3. 先建立可见信息账本，不能靠隐藏牌填输入

用真实协议区分以下实体状态及其时间顺序：

1. 行动方当前自手牌。
2. 已提交出牌、永久离开各家手牌的实体牌，PASS不增加此集合。
3. 公开贡还后，已知由某座位持有、但尚未公开打出的实体牌。
4. 托管/待分配牌；区分牌面/归属已经公开、仅张数公开，以及行动方自身可见的私有选择。
5. 确实没有公开的对手手牌。

对公开交换，解析完整合法event grammar、giver/receiver、encoded seat，不使用“出现过CARD就永久排除”规则。牌的当前已知归属必须随公开转移更新，公开打出后移入played；同一实体不能在多个bucket重复计数。

tracker必须因果更新，只使用行动方实际可见观测、公开事件和自身合法可见信息。不得读取`debug_state()['state']['hands']`、未公开贡还选择、其它observer的private前缀、未来事件或oracle label来构建输入。真实引擎隐藏状态只允许独立label/audit，验证账本声明不能反过来影响feature。

必须处理：贡还未完成/托管、公开resolve、抗贡、单/双贡、还回同一实体、公开已知牌后续打出、finished player、auto-reset清空新旧deal状态。不得把上局贡还归属带入下局。

若旧样本只保存108-bit“出现过”集合，无法恢复giver/receiver/当前归属，不要猜或用oracle补齐。保留旧样本作反例证据；在本轮新pilot保存足够的**可见因果事件/可重放公开轨迹**和账本快照，以便独立审计。

15分钟内先交出反例统计、最小账本/测试设计和具体阻塞；不要再做一小时泛化源码探索或重复GPU吞吐测速。

## 4. Baseline B 的修复语义与守恒

首先用独立手算明确每个量的含义，再实现。以下为推荐的约束表达；若真实协议需要不同实现，先给出等价推导与公开信息边界，不能牺牲已知归属或守恒。

设：

```text
C_r        : 全部实体牌在printed rank r上的张数（普通8，大小王各2）
O_r        : 行动方当前自手牌rank计数
P_r        : 已公开打出的rank计数
K_{s,r}    : 当前可确定仍在相对座位s手中的公开已知牌计数
R_s        : 相对座位s的公开当前手牌张数
E_known_r  : 确定在托管/其它合法非手牌bucket中的已知牌计数
h_s        = R_s - sum_r K_{s,r}
u_r        = C_r - O_r - P_r - sum_s K_{s,r} - E_known_r
```

各bucket不相交；若包含只知张数、不知牌面的托管牌，必须为其保留未知容量，不把它们全分配给对手。若还有其它公开可推导bucket，显式说明，不隐式丢弃。

无未知托管时，合法完整账本应满足：

```text
h_s >= 0, u_r >= 0
sum_r u_r = sum_s h_s
pred_{s,r} = K_{s,r} + h_s * u_r / sum_r u_r
```

存在未知托管/未归属容量时，可以把它作为额外分配bucket；在分母中保留该容量，保证每个对手分配量仍为h_s。若协议无法从可见信息唯一确定所需数量，使用明确的约束/概率处理或报告该路径INCONCLUSIVE，不偷读隐藏状态、不删样本。

未知池为空时直接返回合法K；不能除0、制造NaN或给已完成玩家分牌。对已知仍由s持有的牌，预测不低于K_{s,r}；公开已知牌已打出后不能保留该手牌下界。

**不再断言旧U_old≥R。** 公开转移牌仍在对手手中时，U_old<R可能完全合法。新版应检查重新定义的未知容量守恒和已知归属，而不是修改断言或丢数据。

保留形状API和物理上限：输入`[...,45]`或`[...,3,15]`的clamp输出形状不变；普通rank≤8、joker≤2。但不能依赖clamp掩盖负库存、双计、错归属或破坏守恒。针对旧/畸形public输入，显式版本化或fail-closed，不能静默把旧18维数据当新完整账本。

B必须是规则/公开信息基线，不得读取隐藏label学习修正，不得变成oracle、故意削弱B或改变评估目标帮D通过。

## 5. 显式允许范围与版本化

允许最小修改：

```text
guandan_marvel/phase5/belief_data.py
guandan_marvel/phase5/belief_collect.py
guandan_marvel/phase5/belief_probe.py
guandan_marvel/phase5/b0_runner.py
guandan_marvel/phase5/design.py
scripts/run_phase5_collection_pilot.py
scripts/run_phase5_b0.py（如果当前实际存在且需接线）
tests/marvel_training/test_phase5_belief_data.py
tests/marvel_training/test_phase5_belief_probe.py
tests/marvel_training/test_phase5_collection_boundaries.py
相关test_phase5_*新增定向反例与新公开账本helper
```

需要更改public特征/样本schema、Baseline B协议、输入维度或design hash时，本次已经明确授权这类**必要生产变更**。保留原设计快照和旧pilot，使用新schema/version/hash；旧Gate不得用于新设计。若C/D吃public特征，最小同步维度/可见性校验与单元测试，但不运行真实probe训练。

不得修改PPO/GAE/reward/context/Fuse核心数学、native引擎、clamp物理上限、旧冻结Prompt/阶段1manifest/report/probe、OpenCode配置或全局Python/驱动。不得删断言、降低容差、加skip/xfail、筛掉贡还样本或重写历史数据来获得通过。确需越界时先停止说明。

## 6. 必须新增的决定性反例

至少覆盖：

1. 公开贡牌/还牌在接收方手中：即使未知池中该rank为0，K对应手牌的B预测仍保留正确确定下界；测试独立于生产tracker/helper。
2. 同一实体公开转移→换owner→打出：逐步核对K/P/托管，防止双计；包括自己收到、自己送出和两名对手之间的转移。
3. 两个可见历史完全相同但隐藏分配不同的合法world：B输入/输出逐位相同，label可不同；拒绝未来信息和其它observer private泄漏。
4. 合法R=U_old、R<U_old、R>U_old三个用例，分别手算，说明其中差异来自已知手牌/托管而非统一当非法。
5. 旧真实pilot反例的可审计重放/新收集同类case；没有完整旧轨迹时明确不能恢复，并用新引擎case证明，不伪造旧case owner。
6. 空未知池、零剩余/finished player、非法负库存/重复owner/多bucket双计、joker上限、相对座位旋转、copy-ID重命名。
7. auto-reset、非零start_index、快槽位padding、timeout partial样本沿用已修边界；公开历史与账本不能串deal。
8. 单/双贡、还贡、抗贡、resolved/unresolved/private escrow；未支持路径单列INCONCLUSIVE，不假装支持。
9. shape/physical clamp、MAX_COPIES tile、新进程CUDA初始化回归继续通过；本轮不撤销已正确修复的旧7项。
10. 新旧schema/design/hash不兼容必须明确拒绝，旧数据不被静默升级；baseline指标比较用同一group split和目标定义。

负例必须用pytest.raises或显式反例断言在测试内部捕获，整个测试exit=0。不要故意启动一个必定exit!=0的旧测试集触发熔断，再自行解锁。

## 7. 执行顺序、预算与失败即停

先检查guard状态，当前历史观察为无failure锁，但实际执行必须重读。任何现存非预期failure均不在本次自动acknowledge权限内。

所有Implementer/协调器测试和采集经：

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase5_guarded.py run --label <unique-label> --timeout-seconds <bounded-seconds> -- <script-or-module-arguments>
```

共用规则继续生效：非预期失败一次锁存并停止，返回failure_id、命令、日志、已改文件，等待用户。不换seed/batch/命令、加timeout、清锁、改guard或派新Agent绕过；本轮没有自动修复重试额度。初始只读审计15分钟无具体结论也停止报告。

允许顺序：

1. 对现有128样本做只读统计与最小因果反例，无训练。
2. 修改公开记账与B，并完成最小CPU规则/数学测试；生产神经模型不在CPU跑。
3. 最小测试通过后重跑原9文件工程门禁（timeout≤180秒）并加本轮新测试。参考：

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase5_guarded.py run --label phase5-public-accounting-engineering --timeout-seconds 180 -- -m pytest tests/marvel_training/test_phase5_collection_import.py tests/marvel_training/test_phase5_collection_boundaries.py tests/marvel_training/test_phase5_execution_guard.py tests/marvel_training/test_phase5_symmetry.py tests/marvel_training/test_phase5_belief_data.py tests/marvel_training/test_phase5_diagnostics.py tests/marvel_training/test_phase5_belief_probe.py tests/marvel_training/test_phase5_feature_off.py tests/marvel_training/test_phase5_cli_contract.py -q -p no:cacheprovider --import-mode=importlib
```

新增测试文件必须显式加入，不能只报告旧99项通过。

4. 工程全通过后只跑一次16局CUDA pilot：deals=16，batch_slots=4，start_index=48，base_seed=90000，max_wall_seconds=90，guard timeout=120。输出到新路径，例如`runs/phase5/pilot/public_accounting_v2/pilot16.json`，保留旧pilot；若当前CLI输出目录约束不接受该目录，使用同一允许pilot根目录内新文件名，不削弱约束。
5. 保存新样本、足够公开事件轨迹、schema/design/source/checkpoint身份。核验16个complete deals、indices48..63、实际13等级/贡还覆盖、实体守恒、可见账本守恒/已知owner下界、CUDA FP16 forward与冻结模型权重不变。
6. 若16局未覆盖某个必需边界，使用独立小型引擎/规则单元测试补证，不能自动加大采集、换seed直到出现想要的统计。

这里的GPU模型执行仅采集推理；规则/账本/统计/单元fixture可CPU。不要把“CPU规则测试”误认成CPU生产训练。没有GPU就停止，不能fallback。

不运行完整B0或全量tests，不重新打开test数据挑指标，不启动训练/backward或优化器更新。不要执行原05B中“继续原05A完整B0”的后续部分，本轮边界以本Prompt为准。

## 8. 验收与最终返回

工程门禁和pilot全部通过后返回完整报告并停止；本轮不自动启动Reviewer或继续B0。可列出待用户授权的全新只读Reviewer范围，但不能把Implementer自测写成独立验收。

报告至少包含：

```text
TARGETED_REPAIR_STATUS: COMPLETE | PARTIAL | BLOCKED
OLD_REPORT_CLAIM_RECHECK: 旧R/U统计与28个真实反例
ROOT_CAUSE: 已证明事实与仍待验证假设分开
PUBLIC_INFORMATION_CONTRACT: 时序/实体状态/当前owner/可见边界
BASELINE_B_DERIVATION: K/未知池/托管/守恒/零分母
SCHEMA_AND_DESIGN_VERSION: 旧新身份和不兼容拒绝
INDEPENDENT_COUNTEREXAMPLES: 手算/真实引擎/泄漏/贡还
TEST_RESULTS: 每条命令、真实exit、完整日志、时间
GPU_PILOT: actual device/dtype、样本/牌组、账本与实体审计
BEFORE_AFTER_SOURCE_HASHES / CHANGED_FILES
NOT_RUN: 完整B0、probe训练、B1、全量、Reviewer、阶段6、长训、final
PHASE5_STATUS: PARTIAL_NOT_ACCEPTED
NEXT_STAGE: DENY
```

只要公开牌归属或未决交换信息边界未解决，即使工程测试都绿也不得放行B0；返回明确阻塞，不猜规则。不能靠换fixture、丢真实样本或静默改指标结束任务。
