# 阶段6收尾修复：tie-aware Value Gate、qualification隔离与身份绑定

日期：2026-10-20（Asia/Shanghai）。06B是06A的定向修复入口，不是新增阶段。旧两轮自动修复已耗尽；只有用户明确按本文授权新窗口后执行，不能沿用旧额度自动继续。

## 0. 当前状态与本次目标

当前阶段6为FAIL / NEXT_STAGE DENY。核心阻塞不是“当前value弱、Gate INCONCLUSIVE”，而是Gate工具存在可复现的false-PASS路径。

保留已完成的Pool/owner/macro GAE、冻结历史推进/恢复、resident模型上限、training Gate预算工具及历史GPU证据，不从头实施，不重复长审计。当前科学状态保持：

```text
VALUE_GATE = INCONCLUSIVE
SEARCH_IMPLEMENTED = False
SEARCH_STATUS = NOT_RUN_VALUE_INCONCLUSIVE
DELIVERY_PENDING = true
```

本次授权目标是让资格工具正确、失败关闭、具备可信证据，而不是强迫value通过或实现搜索。

## 1. 先读真实输入与基线

完整读取06A、共用GPU契约、CLAUDE.md/phase角色规则，及：

```text
runs/phase6_final/PHASE6_FINAL_REPORT.md
runs/phase6_repair2/summary.json
runs/phase6_repair2/full_suite.log
runs/phase6_repair1/value_gate/samples.json
guandan_marvel/phase6/value_gate.py
guandan_marvel/phase6/design.py
guandan_marvel/phase6/search.py
scripts/run_phase6_value_qualification.py
scripts/run_phase6_reevaluate_gate.py
相关test_phase6_value_gate*及CLI/planner测试
```

当前value_gate.py SHA256：

```text
b5d6a420ab5cd8f337d7f82d7bfa51c0275c818c1cc0a217f0830ed2ba523486
```

冻结当前tracked diff及tracked/untracked源码逐文件hash，核对无并发写入。旧证据/失败/Review保留不覆盖，不能以mtime或tracked diff单独证明新文件身份。

历史全量2015 passed / 2 skipped / 1 known Guance failure，仅绑定旧源码，不能冒充本轮验证。

## 2. 授权范围、预算与停止

只允许最小修改Value Gate/ranking指标与评估入口、qualification planner/CLI、registration/metadata验证、search状态声明和相应测试/证据，不重做Pool、PPO/GAE、模型结构、Belief、native引擎或Guance。

- 新窗口总墙钟90分钟，启动即记录绝对deadline，断线/重连不重置。
- 最多3轮明确根因→修改→最小测试，同根因最多2轮。
- 最多2轮Reviewer修复并计入3轮总数；每次独立复验用全新只读Reviewer。
- 15分钟无最小可测进展/明确阻塞就报告，禁止再次200-turn只读审计后才写代码。
- 普通scope内工程错误由协调器核验后自主修复，无须每条断言询问用户；保持日志、真实exit、失败历史和轮次台账，不盲目原样重跑。
- 超预算、同根因超限、数学/数据隔离无法裁定、需放宽预注册标准/扩资源时停止交付PARTIAL，等用户决定。

不授权正式64/128-deal qualification采集、搜索实现、正式Gate1–5、长训、Kaggle、双T4/NCCL、FableDan正式评估、final holdout或交付上传。

## 3. P1：统一的tie-aware ranking语义，修复所有false-PASS路径

### 已有真实反例

`runs/phase6_repair1/value_gate/samples.json`：

```text
deal_id = qualification-000000
candidate_count = 2
model_scores = [-0.0020379414781928062, -0.0024632406421005726]
realized_returns = [-1.0, -1.0]
old accuracy = 1.0
old random_baseline = 0.5
```

旧代码用`max((value,-index))`选唯一realized_top，再与同样有索引偏好的model_top比较；全tie可凭索引获得正delta。这不是有效value排序证据。

### 本次明确采用的指标契约

将如下规则版本化并在任何新评估前锁定，保留原Gate EV/CI/epsilon/最小样本等阈值，不为过Gate调参：

```text
k = 合法完整候选数
T = realized return达到最佳值的候选集合
M = model score达到最佳值的候选集合
accuracy = |M ∩ T| / |M|
random_baseline = |T| / k
delta = accuracy - random_baseline
```

这是对模型并列选择均匀随机打破时的期望success，与随机候选策略一致比较；不必实际抽随机数，指标应可复算、对候选顺序置换不变。

- realized全部tie：accuracy=1、baseline=1、delta=0；**不提供排序信息，不能计入最小informative ranking样本门槛或mid/late充分性**。
- model全部tie但target非全tie：accuracy=baseline、delta=0，不能凭索引获得正优势。
- 部分tie：按上式手算，不能继续用1/k基线。
- 只有1个候选、不足可比较候选、非有限值、长度不匹配等显式处理，不抛弃不利case凑充分性。
- tie tolerance/数值类型必须写入metric schema；在观察新结果前固定，不能为了得出更好结论临时改epsilon。当前离散realized return的exact tie与近似model score tie分别说明。
- 保留total/informative/all-target-tie/model-tie/bucket计数，all-target-tie样本可以报告零优势，但不填充informative门槛。全部tie或有效ranking数不足时Gate为INCONCLUSIVE，不能PASS，也不能把缺证据当科学REJECT。

采集、ranking point序列化、离线重评和`evaluate_value_gate`必须使用同一版本契约。Gate不得信任调用者传来的旧`accuracy/random_baseline/top_index`，要从原始有限model_scores/realized_returns/候选定义独立重算并验证；未知旧schema不能静默采信错误分数。

必须封堵所有入口：全tie64个mid/late point不能PASS；伪造serialized accuracy=1也不能PASS；改变候选顺序不能改变delta或verdict。

同步纠正旧测试中全tie仍期望accuracy=1、baseline=1/k的fixture，但不删断言/宽容差/skip/xfail。预期负例用pytest.raises或明确assert，使测试进程正常exit=0。

## 4. P2/P3：qualification planner必须真的隔离、采rank证据且不混淆context

`run_phase6_value_qualification.py`当前execute没有构造/校验protected manifest，且固定`max_ranking_points=0`，不能满足资格证据。

### 隔离与入口

- execute必须在模型/引擎/采集前加载并校验training/development/selection/final等受保护集合的真实身份及seed/canonical deal保护约束，调用实际isolation gate。
- final holdout只允许使用已冻结的seed-domain/hash/访问策略作保护，不读取其结果或生成/消费final样本。
- 缺manifest、重叠seed、canonical deal重复/旋转/copy-ID别名、错误source/config identity都fail-closed；绝不把不相交seed单独当成实体牌副必不重叠的证明。
- 回归中用fake collector证明无效manifest时没有启动任何模型/engine。不能只给plan写一个isolation=true字段。

### Context计划

现有4level×4leader×4tribute若按同一个index同时取mod4，会只出现4个对角组合。修复为真正Cartesian分配或等价独立分层方案，输出joint cell与各marginal count，不把4个组合说成64个。

保持当前注册4个level的限定范围与原min_slice_deals不变；64/128只是计划，不声称覆盖完整13等级。不能通过减少context、改min_slice_deals或将不足slice过滤掉来放行。

### Ranking预算

execute路径必须有非零、受总wall/deal/候选/分支预算约束的ranking计划，并覆盖mid/late；不能固定0后写“资格计划可PASS”。显式统计预期及实际informative点，数量不足保持INCONCLUSIVE。

本轮仅测试planner/dry-run与mock执行接线。可以从已有真实样本离线校验schema和tie-aware结果，不能自动执行64/128-deal大采集来补足资格。

## 5. P3：registration/checkpoint/source/config绑定与search声明

- 明确每个hash的语义：qualification producer源码、checkpoint原训练源码、resolved config、registered design、sample target contract不一定是同一个hash。应分别绑定，不靠字段同名强行相等，也不能只把调用者同一字典复制两份就说交叉验证。
- 从实际严格加载的evaluation-only cpp/ippo checkpoint/冻结registration验证checkpoint SHA、state_dict内容指纹、architecture/feature/backend、checkpoint origin source与config及资格设计身份；metadata必须可追溯到真实验证结果。
- 错checkpoint、同权值换文件名、错registration source/config、result污染、旧registration混用均fail-closed，不能caller伪造metadata后PASS。
- Value资格PASS只表示VALUE_QUALIFIED，不等于搜索已实现。`SEARCH_IMPLEMENTED=False`时，任何artifact都不得写`search_status=IMPLEMENTED`，执行搜索入口仍拒绝。
- 搜索未实现的状态与资格结果分别记录，不用把SEARCH_IMPLEMENTED改True来迁就错误artifact。

## 6. 验证与证据复用

按顺序：

1. 新tie/partial tie/model tie/排序置换/serialized造假/全tie充分性独立手算反例。
2. Planner Cartesian覆盖、缺保护manifest不执行、真实CLI参数传递、非零ranking预算、错source/config/checkpoint拒绝测试。
3. 阶段6相关工程suite；不收集0 tests当PASS，不把old 67 passed当本轮结果。
4. 已有真实value样本离线重评到新目录，明确新metric/schema与旧统计差异；不覆盖旧样本/registration/gate，不重新fit模型或采集。
5. 如果只改Gate/planner，Pool相关生产路径/hash没变则复用既有CUDA pool A/B/C证据，精确说明适用边界；确有改变才补最小GPU smoke，不能无意义重复全部训练/恢复。
6. 最终源码冻结后全量一次（GPU Python、完整stdout/stderr、真实exit与前后tracked/untracked hash），唯一已知Guance failure如实单列，不绕过/删测试/清历史。不要每轮修复都先跑20分钟全量。
7. 全新只读Reviewer独立构造至少三类新反例，核对tie-aware Gate、planner/isolation/identity和默认关闭路径，复核精确source身份的完整全量日志；禁止重采qualification或正式训练。

本轮不需要Value Gate变成PASS。**Gate工具在证据不足时可靠给INCONCLUSIVE且保持搜索关闭，就可以作为正确条件分支接受工程验收。** 相反，即使67/100项测试绿，只要无信息ranking可false-PASS仍必须FAIL。

模型/真实神经smoke坚持独立GPU Python与显式CUDA，无CPU fallback；纯指标/规则/planner单元测试可CPU。所有profile都保留阶段5已验收source边界，不顺手修改其对称性实现。

## 7. 最终交付与训练边界

最终一次性返回：

```text
IMPLEMENTATION_STATUS
SOURCE_AND_CONFIG_HASHES_BEFORE_AFTER
TIE_METRIC_VERSION_AND_DERIVATION
FALSE_PASS_COUNTEREXAMPLES
QUALIFICATION_PLAN_AND_ISOLATION / INFORMATIVE_RANKING_BUDGET
CHECKPOINT_REGISTRATION_PRODUCER_IDENTITY
SEARCH_IMPLEMENTED=False / honest SEARCH_STATUS
TEST_COMMANDS_REAL_EXITS_COMPLETE_LOGS
REUSED_GPU_EVIDENCE_SCOPE / NOT_RUN
FINAL_FULL_SUITE / KNOWN_GUANCE_FAILURE
FRESH_REVIEWER_REPORT
VERDICT / NEXT_STAGE
DELIVERY_PENDING
```

全部工程门禁与Reviewer通过后，可说明阶段6开发验收通过、Value仍INCONCLUSIVE/搜索关闭，并停止等用户授权正式训练准备；不能直接启动长训。

阶段6通过后仍要冻结启用功能、重建包含阶段5/6的新正式交付、校验warm-start/迁移与目标硬件AMP/显存/双卡恢复，然后另行授权Gate1。旧04A ZIP、本地smoke和source已变的旧训练checkpoint都不能冒充当前可直接exact resume的生产交付。
