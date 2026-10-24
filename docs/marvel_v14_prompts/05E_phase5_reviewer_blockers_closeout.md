# 阶段5收尾授权：修复Reviewer阻塞，复用现有B0，不重采不重训

日期：2026-10-20（Asia/Shanghai）。05E是阶段5的收尾入口，不是新增阶段。只有用户明确授权执行后才开始；旧05D预算已结束，不能继续沿用其授权。

## 0. 当前状态和本轮目标

阶段5仍为FAIL/PARTIAL，NEXT_STAGE DENY。不要进入阶段6、正式长训或最终评估。

已经取得的成果必须保留、核对后复用：

- 512 complete deals / 4096 samples / 512 canonical groups，B0预注册Gate REJECT。
- gate: `runs/phase5/b0/b0_r4.gate.json`，gate_hash `9f9310cd1d1bb935d0879e0e5cc93151fb2f0b8faef6086e9640317381d8f9fb`。
- count MAE改善B-D：mean -0.2023317048960521，95% CI [-0.227987811666637,-0.182306075375724]，epsilon=0.05；三seed改善均为负。
- 因此当前B1必须SKIPPED_B0_REJECTED；不重新训练probe、挑seed/指标或重开确认集来追求PASS。这只是当前D方案未达到门槛，不是“证明所有历史信息都无用”。
- 对称性诊断 `runs/phase5/symmetry/symmetry_diagnostic_r3.json` 已SIGNIFICANT，mean restricted KL约0.02857、argmax disagreement约0.11263，augmentation_recommended=true。
- 原全量日志：`runs/phase5/execution_guard/a2b6b5fab9384b9896c4d071c73efbe6.log`，1917 passed / 2 skipped / 1 failed，唯一Guance authority dirty-tree。它是旧源码证据，不覆盖本轮新改动。
- 原05D授权及已补全台账：`runs/phase5/bounded_autonomous/authorization.md`，保留不覆盖，另建本轮授权/计数。

本轮一次性处理以下3项Reviewer阻塞并完成最终验证、全新只读Reviewer复验：

1. 对称性非零配置被接受却无训练消费者；
2. Baseline B未拒绝跨座位重复K、K+u总量超全局实体上限；
3. records_hash因int/str字典键排序造成交付JSONL不可复算。

同时补齐最终源码pilot身份、账本/日志/轮次完整性。不是再改一个fixture就回报完成。

## 1. 必须先读与授权边界

完整读取05A原目标、共用GPU契约、05B/05C、05D历史范围、当前CLAUDE.md与角色规则、最新Reviewer全文/工具证据和真实代码。

先冻结当前tracked/untracked源码、配置、数据与产物hash，检查无并发源码写入与活跃任务。当前历史观察为无failure锁，但必须读取真实状态。

若有陈旧active.lock，只能在确认没有持有者/子进程、没有有效任务后按runner锁语义处理，记录旧内容/时间/依据；不能把仍活跃的任务当陈旧锁。不得删除failure.json或改写失败历史。

本次允许修改阶段5相关config/training接线、symmetry/diagnostics/公共账本与hash、checkpoint语义接线和必要tests/CLI；不得改原PPO/GAE/Fuse/reward/context数学、native引擎、全局Python/驱动/OpenCode、阶段1冻结manifest/历史Prompt或Guance handoff。

不得通过把触发条件改高、把SIGNIFICANT改为不显著、删除对称性需求或让非零开关静默无效来绕过P1。若确需缩减阶段5范围，必须返回用户决策，不能自改标准后宣称原阶段完成。

## 2. 一次性有界自主修复

用户执行本Prompt时，明确授权本轮普通工程错误自行定位/修复，无须逐条断言确认：

- 本轮最多3轮“根因 → 修改 → 最小/工程测试”；同根因最多2轮。
- Reviewer退回最多2次定向修复，计入3轮总数；每次复验用全新只读Reviewer。
- 总墙钟预算90分钟，包括代码、测试、pilot、全量和Reviewer；启动时记录绝对deadline，阶段/长命令前检查剩余时间，不能中断后把deadline重置。
- 10–15分钟一次紧凑进度；15分钟没有具体可测进展则报告阻塞，不做无边界源码探索。
- 超预算/同根因2轮未解决/规则数学无法裁定/需要越界时停止，一次性交付明确PARTIAL，不无限运行。

保留guard锁存、编译/import/export预检、真实exit和全部日志。Implementer不得清锁或修改安全机制；主协调器在本轮剩余预算和授权范围内可核对failure_id、记录根因后acknowledge再安排有理由的修复，不再逐条请求用户。非修复地原样重复失败命令仍禁止。

先登记新授权与计数，最小同步本轮策略例外，结束后恢复strict。不要重做guard系统。旧05D的“每次失败/预算”不作为本轮无限续期许可。

## 3. P1：对称性必须真正进入训练，且默认零开销

### 3.1 artifact门禁

审计`SymmetryConfig`、诊断artifact与训练入口的真实消费路径。对正weight/enabled augmentation要求：

- artifact真实存在、schema/hash完整、统计有限、达到预注册样本/阈值、判定SIGNIFICANT；
- 与实际模型/观测/变换定义/初始化checkpoint的兼容身份可核验；错误artifact、旧schema、tamper、错误模型/变换必须fail-closed；
- 验证绑定在启用时的配置/初始化与checkpoint语义上；不要要求训练后每次权重更新仍等于初始checkpoint SHA，从而无法继续训练；resume保留已验证启用contract。
- 若旧诊断与修改后的source身份不兼容，可使用原16局级预算进行一次必要的对称性重新诊断或从可核验冻结输入离线重放；不重采512局B0。

### 3.2 真实增强与aligned KL

按既有合法SeatRotation/CopySwap及可逆动作映射实施默认关闭的训练一致性分支。当前PPO轨迹、old behavior prob、宏动作归属与原GAE时钟不得因本轮被偷换。

正确对齐为`pi(P_T(a)|T(s))`与`pi(a|s)`，完整macro/完整序列或明确的restricted候选单位；不能把next-token KL冒充完整macro KL。说明mask、合法候选一一映射、padding、finished player、PASS/wild/贡还、两侧梯度/detach、macro归一化及DDP全局计数。

优先将变换用于额外的一致性loss分支，不把变换后的行为概率直接当作原on-policy概率。若要让增强轨迹本身进入PPO，先独立证明行为分布和ratio，不能复用错误old log-prob。

所有被config接受的enabled组合都必须实际生效；无法支持的组合在初始化前拒绝，不能只写semantic hash而不计算loss。也不能用对所有非零配置抛NotImplementedError来冒充已实现。

### 3.3 硬验证

- weight=0、augmentation=false时：参数/输出/轨迹/RNG/forward计数与旧路径逐位相同，无额外推理或训练forward。
- enabled时：真实CUDA训练中symmetry项被消费、合法映射后的KL有限且可反传，实际影响目标或梯度，参数产生有限有效更新；1–2个PPO iteration足够做工程smoke，不扩长训。
- 无标签泄漏；原PPO/GAE/reward参考结果不变；checkpoint保存恢复新配置/状态，跨feature不兼容不冒充exact_resume。
- 若触及DDP梯度路径，补零本地macro/rank不均衡的CPU双rankgloo测试；不声称双T4/NCCL通过。
- 默认仍关闭。没有独立paired棋力CI收益时，不宣传增强有效或默认开启；本轮不跑正式棋力评估。

## 4. P2：Baseline B全局实体上限与恶意输入拒绝

目前只查每个seat的K<=8/2，不足以防止跨seat重复。必须独立验证：

```text
for each printed rank r:
    sum_s K[s,r] <= C_r
    sum_s K[s,r] + u[r] <= C_r
C_r = 8 for normal ranks, 2 for each joker
```

如果还有可见确定托管/owner bucket，按schema定义同步验证全局不重叠与实体守恒；不能只看期望值shape或事后clamp。

同时检验finite、非负、整数计数语义、published hand lower bounds、sum(u)=H+X、空未知池和finished player；不把合法贡还的旧R>U_old当非法。

必须拒绝以下反例，且在模型/参数/正式状态被修改前失败：

```text
R=[8,8,0]
K[0,0]=8; K[1,0]=8
u=0; X=0
```

另测joker跨座位超2、K+u联合超限、NaN/Inf、非法fractional count和负库存。验证已有4096个真实样本在新约束下是否全部合法，不筛样本、不改旧数据。

若新约束只拒绝畸形输入而所有真实样本的B预测逐位不变，可保留其原B0统计作为待修复provenance的科学结果。若真实样本受影响或预测发生变化，不能继续声称旧Gate有效；停止并报告影响，未经新授权不重采/重训。

## 5. P2：records_hash canonicalization与非破坏式证据修复

实际问题：known_owner等嵌套mapping使用int键，JSON保存/重载后键变str，sort_keys次序改变，导致内存hash与交付JSONL hash不一致。

设计唯一可复算的canonicalization：对嵌套mapping键类型、排序、数值/字符串、list顺序和JSON编码作显式规范。int键转str时必须检测例如`1`和`"1"`这类碰撞，不能静默覆盖。拒绝非有限JSON数值，不改变样本顺序、特征、label或group split。

至少证明：

```text
canonical_hash(in_memory_records)
 == canonical_hash(json_roundtrip_records)
 == canonical_hash(records_loaded_from_delivered_jsonl)
```

原样保留`b0_r4.samples.jsonl`、manifest、gate与报告，不能悄悄重写原hash。若算法/manifest schema变更，使用新版本与新路径生成**派生provenance修复产物**，明确记录：原样本文件SHA、旧记录hash/旧gate hash、新canonical hash、规范版本、转换源码身份和source差异。

只允许对既有4096样本做离线重算与审计；不重新采集、不重新fit probe、不重新打开确认集调参、不修改epsilon/seed/统计标准。可以从已冻结预测/聚类误差记录核对原mean/CI/REJECT，必须标为“证据规范化修复”，不是“新实验”。

派生gate的科学数值不变时，给出字段级diff并标注origin/source lineage；新全局source hash因训练增强变动不等于B0当时source，不伪造新source曾跑过512局。需要特征/标签/预测重新计算才能保持结果时，先报告失效，保持Belief禁用而不擅自重训。

## 6. 数据与统计的诚实边界

- 保留B0 REJECT、B1关闭，不因为工程修复需要“阶段PASS”就把科学结果改成通过。
- 报告primary mean/CI与3 seeds指标各自的聚合口径；若集成预测与单seed指标不同，明确说明，不能混用。
- action_kind缺桶、tribute_mode与实际anti_tribute差异：能从已保存可见事件/元数据推导的离线补报；不能推导就明确未覆盖，不通过新增数据或改标签掩盖。
- 任何原报告过度声明都要纠正，但不把全部已完成实验推倒重来。

## 7. 验证顺序与预算

每轮先最小新反例，再阶段5工程suite；失败仍保留原exit/guard记录，由协调器按本轮预算处理，不扩大重跑范围。

至少包括新symmetry训练/artifact/disabled-parity tests、Baseline全局上限、hash roundtrip/collision和旧阶段5全部相关测试。counterexample放pytest.raises/独立oracle中，测试进程正常exit=0。

最终源码冻结后：

1. 一次最终16局CUDA pilot：start_index=48、base_seed=90000、batch_slots=4、采集max90秒、guard120秒，输出新目录，不覆盖旧pilot。证明当前schema/source身份、账本、实体与GPU精度；不使用早于sample_matrix修复的旧pilot冒充。
2. 必要的1–2 iteration symmetry CUDA训练smoke、checkpoint/resume及相关gloo，原feature-off路径继续通过。
3. 全量tests一次，GPU Python、完整stdout/stderr、真实exit及前后源码hash；不tail。唯一既有Guance failure仍单列为exit1，不能写全绿，也不为了它重跑；协调器可在本轮授权下核对完整失败列表后acknowledge这一次已知副产物锁存，不删守卫/测试/日志。
4. 全新只读Reviewer独立构造至少三类新反例，检查三个原blocker均闭环，核对新派生证据的原始文件和数值不变，复核完整全量日志；不重采B0、不训练probe、不重复全量。

若全量后因新缺陷改源码，定向重验并明确旧全量的source差异与证据适用边界，不声称全量绑定仍相同；是否补跑第二次全量须说明必要性并计入90分钟。

## 8. 最终交付与停止

完整返回：修复前后源码/Prompt/hash、三项blocker逐项证据、新增反例、所有命令/真实exit/完整日志/耗时、轮次和绝对deadline、当前源码pilot、训练增强CUDA更新/恢复证据、旧B0与派生manifest/gate的provenance对照、科学REJECT与默认关闭项、最终全量、完整Reviewer报告。

Reviewer必须独立输出PASS/FAIL/INCONCLUSIVE及NEXT_STAGE ALLOW/DENY。未解决P1/P2就不能进入阶段6。通过时只表示阶段5工程验收闭环，仍不授权阶段6、正式50+iter训练、Kaggle、双T4/NCCL或final holdout。

只在预算/安全/数学不可裁定边界停止请求用户，不再因普通fixture或可定位bug逐条确认。预算耗尽时如实交付PARTIAL，不再跨夜无期限继续。结束恢复默认strict策略，保留全部台账和证据。
