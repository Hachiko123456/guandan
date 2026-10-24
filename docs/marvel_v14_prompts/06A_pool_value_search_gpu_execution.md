# 阶段 6 修订执行 Prompt：Opponent Pool、Value Gate、条件搜索与训练 Gate 工具

更新日期：2026-10-19。此文件替代旧 06 的执行规范，不改其冻结历史内容。06A 仍然是“阶段 6”。当前只准备本 Prompt，不表示获准启动阶段6。

## 0. 前置门禁与任务边界

完整读取本文、`00_local_gpu_execution_contract.md`、`CLAUDE.md`、阶段5实施/Reviewer报告及真实代码。

仅当核心阶段1–4通过、阶段5已完成独立审查且用户明确执行本文件时实施。阶段5允许 B0 被有证据否决、belief保持关闭，不以“必须有belief”作为阶段6前提；B0 INCONCLUSIVE 的未决项必须由用户明确决定，不自行当作通过。

任务：实现可选 checkpoint opponent pool、value 资格检验工具、资格通过才实现的 evaluation/deployment 搜索，以及正式训练的分级 Gate 工具。只跑本文的本地有界 smoke；不自动启动正式训练Gate、Kaggle、FableDan正式评估或final holdout。

始终显式使用独立 GPU Python 和 `--device cuda`。环境、规则、非神经统计在CPU；current/frozen policy/value/神经proposal的模型推理及训练在GPU。4GB显存不足时显式改变本地容量并留证，不能回落CPU后声称GPU通过。

## 1. 先审计、版本化与默认关闭

先审计 sampler/trajectory buffer、on_policy_valid、macro segmentation/GAE/PPO、old behavior prob、value loss/regularizer、checkpoint/源指纹、sampler RNG、evaluation adapter。

冻结新字段和 contract/schema、预算、seed、Gate阈值、loss sample unit。所有新增 feature 默认关闭；current-vs-current且pool/search关闭时保持旧采样轨迹、参数、loss、RNG、推理开销等价。不要顺便改阶段2/3/4公式或修Guance handoff。

新增职责明确的 pool / qualification / search / training-gates 模块与test_phase6_*测试；按需最小修改现有连接处。不得扩大legacy容差、放松strict loader、重写历史manifest或修改旧冻结05/06/README。

## 2. Opponent Pool：行为归属与可训练样本严格分离

### 2.1 数据契约

每个参与决策的token/macro记录清晰字段：

```text
policy_owner
policy_train_mask
opponent_checkpoint_id
opponent_policy_version
assignment_id / assigned_team_seats
actual_behavior_probability_or_logprob
```

明确区分active player token、chance/environment步骤、padding和terminal。macro不得跨policy_owner；同一macro的owner/train mask必须一致，出现混杂直接fail-closed。

### 2.2 Loss/trajectory不变量

1. current-controlled active token的policy_train_mask=true，behavior prob必须来自实际采样的current policy snapshot，严格on-policy，不能拿更新后的policy冒充old行为分布。
2. frozen-opponent token mask=false；动作依然推进真实引擎、进入可见历史，不能从时间轴、reward、discount跨度或public state里删除。
3. frozen对手的behavior概率不要求等于current policy；不能将其写成current行为概率，不允许用伪造on_policy_valid规避验证。
4. current的policy/regularizer/value辅助训练样本选择单位明确。默认只对current-owned玩家的合格宏决策学习；若value训练范围不同，必须在设计中先说明目标和mask并独立证明，不可把对手步骤当on-policy current样本。
5. GAE按原玩家宏决策/原base-token discount时钟计算，opponent行动作为环境间隔保留；不能通过过滤opponent token缩短discount时钟、丢terminal reward或改变partner视角。
6. frozen模型eval、requires_grad=false，不进optimizer、不accumulate梯度；训练前后state/hash不变。记录model并发数量和显存预算，不能把整个历史pool常驻4GB。
7. 当前/对手按队伍与座位平衡轮换。current-current模式、single legal action、全对手/零current样本rank等边界都要定义。
8. 如训练rank可能没有本地current宏样本，保持各rank collective次数与模型连图的零anchor；按全局current样本数加权，不以rank均值代替样本均值。

### 2.3 Pool身份与恢复

条目至少包含immutable checkpoint SHA256、architecture/feature/config/来源身份、权重、active状态、recent/spaced/champion标签。采样权重必须配置化且预注册A/B，不写死为理论常数。

不兼容architecture/编码/后端checkpoint在修改trainer状态前拒绝；不因“只是对手”而略过缺key/shape/hash。没有合适多个生产snapshot时，仅用已验收项目checkpoint和明示synthetic测试夹具做工程验收，不伪造champion或多样性。

保存pool manifest/hash、sampling RNG、episode/seat assignment、当前opponent身份和训练owner状态。episode中不能无说明热切对手；resume不能忽略pool变动声称exact。显式policy-only warm start与exact resume分别验收。

## 3. Value资格：只有足够证据才能搜索

### 3.1 数据及指标

qualification状态来自真实C++引擎、玩家实际可见输入，独立于训练牌副/selection/final holdout；固定evaluation-only checkpoint与源身份，按canonical deal聚类切分，状态之间的多步相关性不得伪装独立样本。

至少报告：

```text
EV = 1 - Var(target - prediction) / Var(target)
Pearson / Spearman / calibration
sample_count / independent_deal_count / zero_target_variance_count
by early-mid-late / DOUBLE_DOWN-HEAD_THIRD-HEAD_LAST
by seat-team / level-leader-tribute
counterfactual full-macro candidate ranking accuracy
```

target定义为同一玩家/队伍视角、同一rank-point/3尺度、同一discount约定的后续真实回报；不能混入raw reward后将尺度不一致解释成value弱。Var(target)=0时EV为undefined，明确报告，不用epsilon伪造正EV或排除不利分桶。

counterfactual只能通过可复核引擎分支/共同随机数评估完整候选，控制状态重复与配牌相关性，不用模型自身value排序当ground truth。

### 3.2 Gate必须预注册

在查看独立qualification结果前锁定最小独立牌副数量、样本分桶、EV稳定窗口/跨checkpoint要求、主要ranking比较、随机/公开基线、最小效应和CI统计。

Gate至少要求预定独立切片上EV稳定为正、中后局排序优于基线且CI支持、无严重seat/context偏差。单个小smoke checkpoint无足够长期/跨样本证据时只能INCONCLUSIVE；不能让实现者为解锁搜索临时改阈值。

```text
VALUE_GATE PASS -> 才可实现/验收搜索
VALUE_GATE REJECT -> 搜索禁止构建，保留资格工具与否决报告
VALUE_GATE INCONCLUSIVE -> 停止搜索实现，保留工具与缺证据说明，不自动长训补样本
```

REJECT/INCONCLUSIVE不阻止可独立验收的pool和Gate工具按关闭搜索方式交付，但未运行的搜索不得声称实现通过。

## 4. 部署搜索（仅Value Gate PASS）

仅evaluation/deployment路径，训练默认零影响，`search_enabled=false`时不额外forward、不改变动作/采样RNG/延迟。

### 4.1 确定化与信息边界

只读行动玩家的可见观测、已公开历史/贡还、自手牌和公开剩余张数，满足108张实体守恒、copy-ID、printed rank/suit和既有约束。没有合法determination时明确fallback原policy，不读真实隐藏手牌兜底。

belief否决/关闭时使用满足公开约束的一致先验；belief通过可作proposal但必须投影/拒绝采样满足硬约束，不把B0标签oracle带入deployment。

给出observation-equivalent hidden-world反例，保持可见信息相同而隐藏手牌不同不能令确定化输入偷变；random proposal要有可复现seed/状态。

### 4.2 完整动作与真实分支

候选为完整合法宏动作或能正确还原的top-k完整自回归sequence。next-token top-k不是完整宏动作top-k。报告候选生成的截断/搜索近似、log-prob与合法映射，不能声称未做的完整枚举。

前瞻沿真实引擎处理PASS、牌权、wild声明、出完、接风、贡还、terminal reward；正确team perspective符号。分支环境、cache和RNG必须隔离，运行前后主环境序列化身份不变。

score的value/Q/prior混合公式、视角、reward尺度、折扣与timeout都显式配置。只能在dev集合选参数，不能用qualification确认集或final holdout反复调参。

### 4.3 A/B与成本

固定独立development deals、四座位轮换，paired比较argmax vs argmax+search。按deal聚类bootstrap报告win-rate/rank-point delta及95% CI、outcome/context切片、latency p50/p90/p95/p99、timeout/fallback/非法动作率、GPU显存。

实际未证明CI下界>0或延迟未满足预算时，报告STRENGTH_NOT_ESTABLISHED，默认关闭，不能把某个成功例子解释成可部署提升。

## 5. 正式训练Gate工具：只实现，不自动执行

以下是绝对iteration目标，不是5个自动连续训练任务：

| Gate | 绝对iteration | 若每iter实际完成20局时的累计局数 |
|---|---:|---:|
| 1 | 50 | 1000 |
| 2 | 250 | 5000 |
| 3 | 500 | 10000 |
| 4 | 1000 | 20000 |
| 5 | 1500 | 30000 |

实际budget以completed_games计数为准；改变collect_batch_size/DDP时不可沿用上表估算冒充精确局数。resume按绝对目标和实际counter计算remaining，不能每次续跑额外增加250/1500。

每Gate生成健康/恢复/强度证据：loss/finite/梯度/scaler skips、value、DOUBLE_DOWN/outcome/context、固定development paired new-vs-M09、rule/random与完整checkpoint恢复。记录CI、独立牌副数、evaluation checkpoint/hash及测试耗时。

健康门禁和强度门禁分别定义。固定默认决策可为：CI lower>0且健康/恢复通过才建议继续；CI upper<0建议停止；跨0为INCONCLUSIVE，输出建议而非自动加seed/牌副或烧完预算。跨Gate重用牌副/顺序检验的多重比较/重复窥视规则要预先声明，避免每个checkpoint挑最优。

所有runner提供dry-run、max_iterations/max_games/max_wall_seconds与显式正式训练确认参数；默认dry-run或拒绝正式目标，不能仅凭生成计划就启动50 iter。提供达到预算后的iteration-boundary checkpoint/recovery/output机制，不依赖平台强杀保存。

允许用tiny/smoke目标1–2 iter测试Gate工具边界，但必须标smoke，不写成Gate1已达标。

## 6. 数据隔离与外部比较工具

training、development/fast anchor、selection、final holdout的seed/hash/用途严格区分。final候选/阈值/CI/对手adapter先冻结，final只跑一次，不能运行后换checkpoint。

FableDan只能作为将来外部评估对手，禁止蒸馏、监督标签、策略/权重训练来源。本文只实现可审计比较工具，不执行FableDan正式比较或final holdout。

“超过FableDan”的未来声明必须预锁统计标准，例如pair rotation/按deal聚类后胜率CI95下界>0.5，并报告rank-point与adapter失败率；此标准只是未来协议，不是当前已达成结论。

## 7. 本地执行预算与必须测试

正式Gate绝不运行。默认本地pool smoke为batch=1、每配置1–2 iter、最多3个预注册配置；资格pilot最多32个独立deals，最多128个独立deals的有界qualification尝试，总采集预算1200秒。资格证据仍不足时INCONCLUSIVE/搜索不实施；不能为了完成搜索悄悄增加到正式长训。

若Gate确有既存合格证据，搜索smoke最多16个独立deals、每决策配置化节点数和时间上限。它只证明合法性/隔离，未获大样本收益CI则保持search关闭。

必须新增并亲跑反例：

1. mixed owner同macro拒绝、opponent token不能进current PPO、chance/padding正确、对手行为概率不被假定等于current。
2. current-current全关闭与旧模型的轨迹/loss/RNG等价；冻结对手参数hash和optimizer membership。
3. GAE跨opponent环境间隔、终局/玩家出完/接风、team perspective、single legal action。
4. 空current样本rank与非空rank的DDP全局加权/零anchor/collective一致性；CPU双rank不冒充NCCL。
5. pool SHA/architecture/assignment/RNG恢复、episode中断、manifest缺失/多key/伪造exact resume拒绝。
6. EV零target variance、错尺度/错视角、重复deal聚类、排序随机基线、资格不足不得启用搜索。
7. 若搜索实施：108实体守恒、隐藏信息不泄漏、完整macro映射、分支主环境/RNG不变、timeout可复现fallback、合法动作率。
8. Gate dry-run零训练、绝对iteration/remaining/预算停止、已超目标、恢复输出、inconclusive不自动扩预算。
9. 显式CUDA无fallback、真实FP16/scaler有效参数更新、同设备保存恢复、关闭路径推理零开销。
10. phase2/3/4与已验收phase5兼容回归；旧全量Guance失败不扩大，新增失败不得伪装历史项。

测试文件使用 `tests/marvel_training/test_phase6_*.py` 等真实实现路径；先反例，再单GPU小smoke，最后冻结源码后全量一次并保存真实exit code与完整日志。缺GPU不能宣称GPU门禁已过。日志/Reviewer规则遵循共用契约。

## 8. 报告、条件通过与停止

报告必须包含：

```text
IMPLEMENTATION_STATUS / SOURCE_AND_PROMPT_HASHES
POOL_SCHEMA / POLICY_OWNER_MASK / ON_POLICY_DERIVATION
FROZEN_OPPONENT_PROOFS / LOSS_MASK / GLOBAL_WEIGHTING
GPU_AMP_UPDATES_MEMORY_TIMING
CHECKPOINT_DDP_RECOVERY / MIGRATION_VS_EXACT_RESUME
VALUE_DATASET_AND_TARGET / METRICS_BY_SLICE
VALUE_GATE: PASS | REJECT | INCONCLUSIVE
SEARCH_STATUS: IMPLEMENTED | SKIPPED_VALUE_REJECTED | NOT_RUN_VALUE_INCONCLUSIVE
SEARCH_AB_OR_NOT_RUN / DEFAULTS_OFF
TRAINING_GATE_TOOLS / DRY_RUN_PROOF / FORMAL_GATES_NOT_RUN
DATA_ISOLATION / NO_FABLEDAN_TRAINING / FINAL_NOT_RUN
TESTS_AND_COMPLETE_LOGS / DELIVERY_PENDING / LEGACY_FAILURES
```

全新Reviewer独立检查数学、运行至少三类新反例，不能复用Implementer上下文。value gate未通过且搜索关闭，可以对pool/Gate工具给工程PASS，但必须单独标明条件搜索未实施，不能声称“所有可选功能完成”。无必要证据则INCONCLUSIVE，不开长训来强行补齐。

最终写 `VERDICT: PASS|FAIL|INCONCLUSIVE`。通过也仅表示开发阶段验收，不自动授权正式Gate1–5、Kaggle、final holdout或FableDan正式评估。完成后停止等用户。
