# Claude Code 实施 Prompt：一次上传、双配方串行训练与统一评估

日期：2026-10-23（Asia/Shanghai）
项目：D:/project/guandan
阶段标识：single_upload_serial_recipe_comparison_v1

> 这是交给 Claude Code 的完整实施任务。请先完整阅读，不依赖此前聊天的隐藏上下文。任务是修改代码、补测试并准备新 Kaggle 交付包；不是现在启动正式训练。实施后停止，交由用户与 Codex 独立检查。本文作为一个实施阶段，内部步骤是开发依赖，不是要求用户反复上传或逐项扫参。

## 0. 用户目标、授权和边界

用户希望：只上传一个 Dataset payload、导入一个 Notebook、执行一次 Run All；同一个 Kaggle 作业自动串行完成：两臂预检 → 现有配方训练 → 候选配方训练 → 统一评估 → 中文报告/模型/完整恢复/结果ZIP。

关键区别：一个作业内的两条隔离训练分支，不是在同一个模型上先跑旧配方、接着改参数跑新配方。两臂从同一375策略起点独立初始化，不互相继承训练后的权重、optimizer、采样器、cache或随机状态。

本任务授权：实现新入口和必要的核心适配、单元/回归测试、已有项目规则允许的有界本地验证、构建新交付包。未授权：代理启动Kaggle、提交远端作业、本地正式长训、正式50+iter训练、高成本对战、FableDan运行、final holdout。最终由用户在Codex复核通过后上传运行。

真实模型前向/训练/探针按CLAUDE.md使用显式独立GPU Python和CUDA；只读CPU权重/元数据检查、规则/统计/小型明确单测可以用CPU。不能静默CPU fallback，不能改变全局Python或CUDA环境。没有可用设备则如实NOT_RUN，不伪造目标设备认证。

遵守 D:/project/guandan/CLAUDE.md 与相关AGENTS.md。工作树已存在大量dirty/untracked，禁止reset/clean/checkout/restore/commit/push，禁止覆盖或格式化无关文件。保留OpenCode配置和既有Claude规则，不删除失败锁或绕过hook。如现有guard阻断，应报告而非换路径规避。历史阶段5规则不表示重开阶段5，也不能借本任务清除其锁。

按项目现有Claude流程：全新phase-implementer实施，全新只读phase-reviewer独立核验；最多按现有规则两轮修复，失败不降标准。Reviewer需独立构造至少三类实施者未提供的反例。不要求重新执行旧05/06阶段。

## 1. 必须先读的材料

1. D:/project/guandan/CLAUDE.md
2. D:/project/guandan/docs/marvel_v14_prompts/CURRENT_EXECUTION.md（历史入口，不能把旧“只评估”当成本次任务；也不得覆盖其历史记录）。
3. D:/project/guandan/runs/diagnosis_and_plan_review_20261022/REVIEW.md
4. D:/project/guandan/runs/diagnosis_and_plan_review_20261022/AUDIT.json
5. D:/project/guandan/runs/diagnosis_and_plan_review_20261022/ARCHITECTURE_CHECK.json
6. D:/project/guandan/runs/t4x2_continue500_analysis_20261022/REPORT.md
7. D:/project/guandan/runs/t4x2_recheck375_analysis_20261022/REPORT.md
8. D:/project/guandan/docs/guide/01_architecture.md、02_training.md、04_evaluation.md
9. D:/project/guandan/docs/t4x2_continue500.md、docs/t4x2_recheck375.md
10. 现有tools/continuation500、tools/recheck375、phase6池组件及相应测试。

事实、机理解释与待验证假设严格分开。不要照搬已被审查纠正的diagnosis报告结论。尤其：没有确诊非传递性；EV不是梯度噪声比例；熵不是单调坍塌；ratio_max不是KL；权重变化不证明棋力收益；target-KL不是硬信任域；本任务不做所谓“改advantage即复现DouZero”。

开工前落盘：git状态、tracked diff bytes/SHA256、未跟踪文件清单、相关输入与源码SHA256、本文SHA256、实际Python/设备/后端版本。只在新证据目录记录，不动冻结历史。

## 2. 默认实验方案：比较完整配方，不声称独立因果归因

### 2.1 初始输入

同一原始375评估策略作为两臂策略起点，250保留作池项与开发参考，500保留作未入池历史对手。375只是候选起点，不宣称已显著最强。

优先核验这些本地文件；不得只相信本文hash，必须读取实际文件：

- D:/project/guandan/runs/t4x2_continue500_analysis_20261022/model_iter000375.pt
  原始SHA256预期：d523e557975ec10b4e6e8347d49a0a9226c188a30bb22cf9088bb78ac16374ee
- 同目录anchor_iter250.pt
  原始SHA256预期：965c60603b6aa8b5fe107d00c1a998e1389b64d104c4610386786b9487849f5f
- 同目录model_iter000500.pt
  原始SHA256预期：2139e12728e64c6dba63de34e9a2673b9fc15b8f4251c3e019edb8c2cb80ba42

如路径或身份不符，先说明实际来源及差异，不允许修改预期hash掩盖错误。不得覆盖这三个原始模型、原结果ZIP或旧交付。

### 2.2 两臂明确配置

| 项目 | control | candidate |
|---|---|---|
| 起点 | 同一375策略 | 同一375策略 |
| 网络主结构 | v1.4-macro-fuse | 相同 |
| PPO/宏动作/奖励 | 真实v4基线 | 相同 |
| lr / clip / reg | 读取真实配置，预期1e-4 / .2 / .01 | 相同，不做额外衰减或扫参 |
| NTP | 保留真实基线，预期.02 | 关闭，显式受控头迁移 |
| 训练对手 | 保留现有self-play行为 | 冻结250、375静态池 |
| 池权重 | 不适用 | 默认1:1，工程默认，不宣称理论最优 |
| target-KL | None，严格保留旧路径 | .015，按下文明确停止口径 |
| optimizer/sampler/RNG | 新运行、明示初始化 | 一致规则独立初始化 |
| 上下文/贡还课程 | 冻结真实有效基线语义 | 完全相同 |
| 预算 | 相同训练墙钟配额 | 相同训练墙钟配额 |

注意：实际v4/continue500已含NTP与贡还课程。不能按早期“暂不新增贡还/NTP”描述把现有贡还关闭，也不能先改成理想化208上下文分布。不新增或修改贡还覆盖、模型层数、奖励、GAE、搜索、Belief、Symmetry、DMC、学习率调度等旁支。

必须读取resolved_config及真实源码，而非仅凭旧文档猜默认。若预期lr/clip/reg等与真实有效配置不同，先明确差异再冻结来源，不偷偷多改参数。

两臂课程应相同且不因fresh计数归零意外重放早期课程。必须显式定义课程的逻辑起点与新运行计数的关系，生成可核验上下文日程/版本；必要时使用独立curriculum offset，而非伪造完整resume或把新训练iteration写成旧375。历史origin_iteration=375与本臂new_iterations分开记录。

固定使用真实v4的macro_decision_v1与macro_normalized_kl_uniform_v1语义（核实其在resolved_config中的实际嵌套位置），维持gamma/GAE/value loss/AMP等其余设置；若来源并非该语义，先报告身份冲突，不把默认配置悄悄代入。Symmetry、Belief、search保持关闭；不以Value qualification通过作为基础策略训练的新增前置条件。

本实验比较配方整体收益和效率。NTP头不同、池可训练座位数不同，因此不能宣称严格单变量实验或各项独立归因。日志同时报告墙钟、局数、token、有效current宏、attempted/applied optimizer step。不要拿相同轮数/局数直接当作相同有效训练样本量。

## 3. 已知阻断必须显式修复

### 3.1 池模式fresh初始化

D:/project/guandan/guandan_marvel/training/local.py 的load_warm_start_policy目前在policy_router非空时拒绝加载：

    policy-only warm start cannot reconstruct opponent-pool assignments; use an exact checkpoint resume with the same pool manifest

375是evaluation-only，training_resume_allowed=False；现有完整latest恢复是500。不得把375变成假完整恢复，不得删guard或直接改metadata声称exact_resume。

请实现明确版本化的新运行策略导入/迁移入口：验证原始checkpoint身份、架构、张量形状、协议、语义、允许迁移差异；导入保留权重；初始化新的optimizer/scaler/merge状态、sampler、池assignment、RNG、计数器、课程状态。方案按真实实现选最小安全修改，不要求使用某个函数名。旧不合法路径仍应拒绝；新入口不能宽泛接受任意不匹配checkpoint。

control也使用同类fresh初始化规则，不能control完整resume而candidate重置。共享backbone、policy/fuse/value的保留权重在初始化后逐位一致；所有可训练参数被optimizer准确覆盖一次，冻结池参数不进入optimizer。补零更新推理等价检查：在同一可见输入、合法候选和明确数值口径下，去除训练用NTP头不得改变策略/价值输出；核对缓存增量与完整历史重算的一致性，容差须有dtype依据，不能为通过临时放宽。

### 3.2 NTP和冻结池模型兼容

权重0会不创建NTP头，但375包含ntp_head.weight、ntp_head.bias。现有loader允许部分missing却拒绝unexpected。

只允许本文定义的NTP头去除，输出迁移清单、保留键hash/形状、丢弃键及原因；不准strict=False后忽略全部missing/unexpected。control保留NTP权重；candidate移除头/训练分支按实际实现冻结契约。

冻结250/375的加载器也要真实验证：它们原配置可能仍含NTP头。允许冻结模型保留不用的训练头或受控只推理迁移，但需独立身份、来源与完整兼容验证，不能假定candidate关NTP后两个旧对手自然能严格加载。

### 3.3 后端/source身份与新源码

旧模型绑定Linux native、Python/Torch、原source/backend身份；tools/recheck375等存在严格旧路径。新源码包不能冒充旧source继续exact_resume，也不能未经证明复用不匹配native。

先审计当前源码、目标环境、旧checkpoint的真实兼容边界，设计受控“旧策略导入新运行”契约。保留原始身份，另记target身份、迁移版本、白名单与验证证据；不得把旧checkpoint/backend/source hash直接改成当前值以骗过检查。若必须重新构建native，按当前项目隔离构建流程，明确目标ABI，并验证规则/协议及来源；不要偷偷弱化全局模型加载校验。

这是高风险项：必须在本地和目标启动预检验证，包括冻结池与评估模型；解决不了就BLOCKED，不能交一个明知Kaggle会卡死的包。

### 3.4 resident与owner-mask

OpponentPoolConfig.max_resident_policies默认1，两个历史条目至少需2；现实现没有惰性逐出。必须同时审计PoolPolicyRouter的session/cache容量，不能只改一处上限。

冻结对手eval且requires_grad=False，hash不变。仅current owner的token/宏参与policy、value和相关训练统计；宏不能跨owner。对手动作仍正常进入可见历史。chance/环境步骤不改变。队伍/座位轮换平衡。pool manifest、assignment、采样RNG和cache语义都必须支持完整恢复。

## 4. 可选target-KL：明确数学和DDP行为

新增选项默认None，关闭时旧训练行为应保持不变；不能影响所有旧用户/旧交付。

candidate初始值.015，不做自动调参；定义停止阈值直接为target_kl（不是暗中乘1.5），日志明示公式。若真实实现需其他停止因子，必须在签名计划中显式暴露并解释，不得与本文默认混用。

宏级log_ratio为current macro logprob减behavior macro logprob；采样近似KL采用exp(log_ratio)-1-log_ratio（可用expm1）。仅有效current宏参与，按跨rank的KL总和/有效宏数加权。

示例需独立reference测试：

- rank0有1个宏、局部KL=.04，rank1有9个宏、KL=0：全局KL=.004，不能因直接平均=.02错误停。
- rank0有9个宏、KL=.02，rank1有1个宏、KL=0：全局KL=.018，不能因直接平均=.01错误不停。

明确监测时点。推荐最小实现使用当前minibatch更新前的宏KL，超过阈值则所有rank一致跳过该待执行更新，并退出本轮后续minibatch/epoch。它不是已完成更新的回滚或整轮最终KL硬保证。全局零有效宏、空rank、极端log_ratio、非有限值与AMP跳步必须有独立行为和证据；不准clamp/置零掩盖异常。

collective调用和退出路径必须两rank一致，不能单rank先raise/先break导致其他rank卡死。保留正确的梯度权重，不因加日志或停止判定重复缩放loss。scheduler、merge updater、计数和resume按实际applied更新语义处理，并测试停后保存/恢复。

累计记录每轮/epoch的sum/count、均值、最大值、可解释分位数、长度分桶、停止位置、attempted/applied/skipped step；最后minibatch指标保留时明确命名，不把它冒充整批最终KL。全局分位数不能通过平均rank分位数伪造；可用有界样本并标记近似口径。

## 5. 一个Notebook的串行状态机

建议新模块目录D:/project/guandan/tools/serial_compare/，复用已审计的通用能力但不把旧特定驱动改成万能入口。状态和路径必须显式分离：

    bootstrap/input_check
      -> preflight/control
      -> preflight/candidate
      -> freeze_execution_budget
      -> train/control
      -> train/candidate
      -> freeze_final_checkpoints
      -> evaluate/confirmation_primary
      -> evaluate/development_diagnostics
      -> summarize_and_package

两个preflight全部通过后，才允许任一正式训练臂开始。不是先跑control长段再测试candidate。

每个preflight在隔离目录完成真实小量双rank启动/采样/有效更新、保存、恢复后再更新、导出、小量完整评估与打包；覆盖该臂真实结构和池/早停路径。预检更新不进入正式臂，不继承其权重/RNG/数据/计数。正式两臂从不可变初始输入重新开始，训练发牌命名空间与preflight分离。

每个正式臂顺序占用两卡DDP，不能一张卡一个臂。阶段切换关闭本阶段子进程/进程组、释放模型与cache，下一阶段使用独立目录和进程；只管理本任务创建的进程，不kill全部Python或无关任务。

每个阶段具有明确入口前置、输出完整性、原子完成标记、错误报告、超时和恢复位置。完成标记须绑定plan/source/model/output hash，不能只看“目录存在”。控制臂模型绝不成为候选初始化来源。

### 5.1 恢复与幂等

- formal训练定期在安全边界保存完整恢复，频率由冻结计划给出（按时间/迭代均可，需测保存成本）；保留至少一个已验证旧完整版本，再原子切换latest指针。
- 包含每rank optimizer/scaler/RNG/sampler、pool manifest/assignment/RNG、课程状态、预算与已用训练时间、KL/更新计数、source/plan身份。
- 单独区分evaluation-only模型与exact新运行恢复。旧500恢复包不是本实验恢复包，不得混用。
- 重启/续跑读取相同计划及已有阶段产物，完成的control不重训；candidate从自己的完整恢复继续；完成训练后的恢复仅补评估/打包，零额外训练。
- 评估逐局保存进度、唯一任务键、checkpoint/协议/发牌hash；重复结果冲突须拒绝，不能重复累计胜场。
- 预算恢复不能重置为全新4小时或重新发放每臂40分钟。分别记录累计工作预算与当前会话deadline，剩余额度取安全较小值；不依赖跨进程/跨机器不可比较的monotonic绝对值。
- 检查缺文件、半写文件、篡改manifest、错误arm恢复、错误source/模型、损坏zip、阶段标记与实际文件不一致等。
- 正常预算停止安全保存；技术失败也尽最大努力产出已有完整状态与错误，但不得将失败重写成“complete”。SIGKILL/平台强制销毁可能无法执行finally，必须靠周期已落盘恢复降低损失，不承诺必能抢救。

### 5.2 默认总预算（应用上限，不代表平台额度承诺）

Notebook一键默认总预算14400秒，明确从bootstrap开始计时：

| 阶段 | 默认上限/预留 |
|---|---:|
| 所有依赖、编译、校验和两臂预检 | 1200秒 |
| control训练 | 2400秒 |
| candidate训练 | 2400秒 |
| 评估硬预留 | 6000秒 |
| 最终保存/汇总/打包预留 | 600秒 |
| 尚余共享机动空间 | 1800秒，不自动追加给某一训练臂 |

单臂新迭代数安全上限256，首要比较预算是相同训练墙钟配额；达到上限/安全停止时记录实际差异，不声称精确相同计算量。actual applied updates必须为正，不得“运行成功但零训练”。这只是短段路线筛选，不承诺足够检测小幅棋力差异。

全部启动预检后，若需缩减预算，只可在两臂正式训练前按公开确定公式对称分配，并记录不可变resolved预算/subplan hash。不得control已跑40分钟后再给candidate任意少一点却宣称等预算；无法为两臂与完整主评估保留最低计划资源则不启动正式训练。

每次新迭代前考虑最近实测p95耗时、保存时间和后续硬预留，不能把安全检查放在两小时以后。评估不得被训练挤掉。不得自动扩大horizon、batch、训练时长、评估范围或重跑失败阶段以耗尽配额。

OOM/身份/非有限/illegal/truncated等技术失败不自动调小batch/horizon、切CPU、改规则、换source或跳过错误继续。失败与统计INCONCLUSIVE分开，输出明确failed_phase和可恢复位置。

## 6. 性能要求：尽量提高有效吞吐，不追求虚假100%利用率

复用当前明确的NCCL两rank、AMP、有效前缀布局、去padding与必要cache优化。不要同时重写Transformer、采样算法或动作协议。基线batch/collect/horizon应从真实有效配置读出（已有v4文档记载rollout48/collect40/horizon1024，须核实是global还是per-rank）。

新池含两个冻结模型及路由cache，必须做实际内存预算，区分buffer、activations、模型、optimizer、cache、allocated/reserved和每rank峰值。既有dense每轨迹估算不能直接移植到新池。

实现阶段可基于可复现预算选择两臂共同安全的profile，选择理由与差异写入计划；但不开展长程HPO，不静默改变数据语义。目标启动需用生产模型尺寸、真实batch/合法horizon与有代表性的完整轨迹验证，不得用tiny/horizon缩小smoke冒充双T4全尺寸认证。若无法覆盖最坏峰值，明确估算与未覆盖边界。

每rank记录GPU设备/后端、有效更新、采样和learner耗时、GPU遥测、显存峰值、同步等待/可测瓶颈；记录games/sec、current macros/sec、learner steps/sec、评估games/sec与阶段端到端耗时。

评估两GPU按唯一任务分片并恢复时重均衡，不使用DDP同步推理造成无谓等待。候选对control对战可在每张卡各载双方，内存验证后并发任务。禁用不必要的重编译、每轮模型重载和重复对局。

不保证“打满GPU”或固定加速倍数；优化是否有效用测量回答，不能只报reserved显存很高。遥测/诊断本身耗时需统计，避免为日志显著拖慢训练。

## 7. 架构与训练诊断同次采集

不改变网络结构，仅低频记录/采样：

- decoder、state/action、actor_fuse、policy/value/NTP的梯度与更新幅度，标明grad unscale、裁剪前后及AMP跳步口径。
- Fuse残差对原乘积路径的输出范数/比例，使用有界抽样、no_grad/不改变训练RNG或模块状态；不能引入额外梯度或把冻结对手输出当current输出。
- value EV和loss的batch/样本口径，宏target与target variance，不能称EV为梯度信噪比。
- 强制动作数量/比例、非强制决策上的归一化熵、宏长度分桶；单合法动作计0的总体熵与非强制熵分开。
- NTP loss只在control记录，candidate清楚标记DISABLED而不是用0伪造更好loss。
- ratio、KL到旧策略、KL到均匀分布明确区分；每rank与全局样本数清楚。
- 上下文、级牌、leader、贡还结果和队伍座位实际覆盖，以及训练数据provenance。

可增加已有模型的轻量诊断，但不要求为本阶段另训v1.3、DMC或历史消融臂。不要将权重非零、梯度非零等价为棋力优势，不承诺一次运行解决全部架构问题。

## 8. 评估与统计：主问题优先，独立新牌局，不碰final holdout

训练开始前冻结数据manifest、评估策略（建议沿用确定性策略协议）、主指标、置信方法、任务数量和输出模型选择规则。

### 8.1 输入隔离与固定终点

- training、preflight、development、confirmation使用不同命名空间和可核验不重叠牌局；以实际发牌身份/哈希检查，不只比较seed名字。新数据不得复用用于训练/挑选的旧开发集伪装确认集。旧模型更早训练的隔离按可获得的真实provenance核对；若历史记录不完整，明确无法证明的边界，不伪造完整历史隔离证明。本实验新阶段的数据隔离必须严格成立。
- 新建本实验confirmation集，不使用项目保留的final holdout，也不运行FableDan。确认集不反过来影响训练、池权重、参数、提前停止或模型选择。
- 每臂选预声明的训练终点最新完整有效checkpoint，评估前冻结双方hash；不按开发胜率临时挑“最好”后拿同一数据宣称独立确认。
- 为避免诊断耗尽预算，先跑主确认，再跑开发辅助；两套数据不互用。

### 8.2 默认对局量

主确认：candidate_final vs control_final，128副独立新发牌，每副两个唯一整队分配，共256场。

开发辅助：另一组32副独立新发牌，每副两个唯一整队分配，以下每臂64场，共448场：

1. candidate vs control（诊断集，不合并到主确认的正式统计）；
2. control vs375；3. candidate vs375；
4. control vsrule；5. candidate vsrule；
6. control vs500；7. candidate vs500。

500不入训练池，作为未入池旧策略探针；375在池内，只能评估针对性收益，不能当唯一外部泛化指标。250已经入池，无需再为完整矩阵多烧预算。rule不是FableDan替身。

共704场正式评估，不含另行计数的少量工程preflight。若当前引擎协议必须采用不同分配计数，先证明唯一性与对称性、固定后公开调整，不得把重复座位轮换当独立样本。

如果实测成本不支持预算，实施时说明成本并合理冻结更小的辅助集/更长评估预留；不得在结果出来后缩减主确认、改CI、改成功阈值。任何主确认不完整标记EVALUATION_INCOMPLETE，不对已有方便子集出正式PASS。资源不足可以正确产出INCONCLUSIVE，不保证4小时证明5pp提升。

### 8.3 统计与解释

正式主指标是candidate对control的团队胜率，以独立整副牌为cluster处理两分配相关性。默认沿用项目已有按deal聚类bounded empirical-Bernstein 95%区间；记录其保守性和当前样本数精度。可另报预声明的cluster bootstrap/近似区间作为探索性敏感性分析，但不得替换正式口径制造显著。

候选配方有正式胜率收益证据：主区间下界>.5；上界<.5支持较差；跨.5为INCONCLUSIVE。配合健康完整性单独报告，不能凭点估计自动晋级。

这不是生产全分布promotion/final benchmark，本次formal_production_promotion=false始终成立；“推荐继续投入该配方”与“正式生产晋级”分开。开发辅助按pair/cluster报告胜率、团队原始得分、结局、context/seat分片和必要区间；小分片不足不得强行排名。

从375出发对375的无变化基线是50%，不得搬用375对250的52.7%。不以rule胜率85%作为本实验必须通过的通用门槛，不宣称结果已说明优于FableDan。

所有记录含发牌身份、arm、双方模型hash、队伍/座位分配、动作轨迹hash、终局与原始奖励、耗时/进程/设备。明确技术失败、数据缺失、预算停止与统计不显著的区别。

## 9. 交付产物与文件范围

建议目录与命名（不存在时新建；存在则创建新版本，不覆盖）：

- 新实现：D:/project/guandan/tools/serial_compare/
- 构建入口：D:/project/guandan/scripts/build_t4x2_serial_compare.py
- 离线验证入口：D:/project/guandan/scripts/verify_t4x2_serial_compare_delivery.py
- 新测试：tests/marvel_kaggle/test_t4x2_serial_compare*.py及必要training/runtime反例。
- 用户说明：D:/project/guandan/docs/t4x2_serial_compare.md
- 交付：D:/project/guandan/runs/t4x2_serial_compare_delivery_20261023/delivery_v1/
- 实施证据：D:/project/guandan/runs/t4x2_serial_compare_implementation_20261023/

交付目录至少含：

1. guandan_serial_compare_payload_v1.bin —— 唯一Dataset上传文件，内部含全部输入模型/新源码/driver/依赖/数据或确定性数据构建协议/manifest。延续已验证的opaque传输与严格哈希，不再要求用户手工上传多个ZIP、恢复包或挑模型。
2. guandan_serial_compare_v1.ipynb —— 单独导入Notebook，选双T4后Run All；默认不改变量。仅恢复或挂载歧义时才有明确高级选项。
3. README_UPLOAD.md —— 中文最短步骤、预期阶段、时间/资源上限、下载入口、失败与恢复办法。
4. plan.json、DELIVERY_MANIFEST.json（或等价清晰命名）、LOCAL_VALIDATION.json、ARTIFACT_QA.json、SHA256SUMS。

一次上传指一个Dataset payload加一个Notebook导入，不是让用户为两臂分别上传和点击。不要自行创建/更新用户的Kaggle Dataset或启动作业。

核心允许改动：guandan_marvel/config.py、training/local.py、相关checkpoint/init迁移、phase6池/路由兼容、必要runtime/DDP统计、新编排/评估/打包与测试。修改前读真实代码，保持最小作用域；不要为了迎合清单制造无意义文件。若必须触及其他文件，解释依赖并记录，不扩展为规则/架构重构。

可更新CURRENT_EXECUTION.md顶部加入本任务“实施/待Codex复核、未远端运行”的指向，保留历史。不得把本文或CLAUDE.md改成降低守卫的依据。不得覆盖旧delivery、模型、分析报告、原始ZIP、全局依赖和无关dirty改动。

### 9.1 运行结果包

默认一份guandan_serial_compare_results.zip，包含：

- REPORT.md、机器可读RESULT.json、阶段耗时/比较表、可选HTML报告；
- 两臂final模型、初始化迁移证明、各自最新完整恢复与全局阶段恢复信息；
- 逐局记录、聚类统计、开发与确认发牌manifest、provenance；
- 两rank训练日志、KL停止日志、架构诊断、GPU遥测、性能汇总；
- 原始/解析配置、pool manifest、运行计划/预算、依赖/source/native/model/hash身份；
- 错误阶段与traceback、NOT_RUN/未完成项和安全恢复步骤。

结果ZIP在训练前先创建最小有效报告，阶段结束原子更新；定期保存完整恢复。最终终止路径统一调用汇总/打包，正常异常也保留已有证据。遇空间不足优先保留可恢复状态与原始关键证据，不删除唯一恢复或悄悄丢失原始评估记录。

### 9.2 输入与归档安全

Notebook/payload/内层manifest/driver/source/模型hash互相绑定。拒绝路径穿越、绝对路径、反斜杠/NUL/符号链接、重复或大小写冲突条目、超限归档、未声明文件、多个冲突payload、错误恢复版本。不取消身份校验解决输入错误。

新源码必须包含本任务新增/未跟踪模块，不能只git archive漏掉它们。核验真实payload解码后源码逐字节等价，不能“本地测的是新工作区，包里还是旧source”。

依赖与native使用隔离运行时；尽量复用已验证离线交付能力，准确绑定ABI。若必须构建，应可复现且预算有界，不能未经说明要求联网安装浮动最新版。不要把绝对旧路径身份兼容问题留给用户在Kaggle手修。

## 10. 验证矩阵：不能只写报告，必须实际运行

下面是最低验证类别，测试名可自行设计；关键reference独立手算，不调用同一个生产helper自证。既有相关测试也要回归。

### A. 初始化与兼容

1. 两臂保留的decoder/state/action/policy/fuse/value参数逐位相同；control保留NTP、candidate只去白名单NTP键。
2. 非白名单missing/unexpected、错误架构/维度/协议/模型hash、非法迁移必须失败。
3. 池模式旧warm-start仍拒绝，新显式fresh入口成功；不能把evaluation-only变成假exact resume。
4. 候选optimizer无冻结对手，所有current可训练参数准确覆盖一次；池加载不污染learner初始化RNG。
5. source/backend身份不兼容反例拒绝；获准迁移保留来源与新身份，实际模型前向验证而非仅metadata。

### B. 池与KL数学

6. 两池项驻留上限/加载、队伍座位轮换、冻结hash恒定、macro owner边界。
7. opponent token/宏不进入policy/value loss和current指标；对手历史正常可见，pool off兼容旧行为。
8. 上文KL不等宏数的两组手算反例，另含一个rank空/两个rank空、恰好阈值、非有限/极端值。
9. 两rank同停/不停、正确跳出双层循环、无死锁；实际applied/attempted计数与日志一致。
10. target_kl=None不改变旧更新；启用后正常路径梯度/权重无额外错误缩放。
11. KL触发、未触发、AMP跳步之后的保存/恢复一致；模型/optimizer/scaler/sampler/pool/RNG/课程/计数完整。

### C. 编排、预算、恢复、评估

12. 两臂preflight都先执行；candidate预检故障时control长段调用次数必须0。
13. preflight不污染正式初始权重/随机数/数据；candidate从共享375起点而非control最终权重开始。
14. 配置/课程一致且不意外倒退，起点origin与new_iterations分开。
15. 使用可控fake clock/故障注入验证所有预算边界、eval/finalize硬预留、对称分配、p95安全边界、恢复不重置预算。
16. control完成/candidate中断后恢复：control零新增训练；两臂完成评估中断后恢复：两臂训练调用均为0。
17. 已有完成标记但文件缺失/坏hash必须拒绝；半写checkpoint/损坏ZIP不当有效恢复。
18. 评估唯一两分配、重复任务幂等、冲突记录拒绝、恢复重新均衡、按deal聚类而非按场假独立。
19. train/dev/confirmation身份隔离；final holdout/FableDan未读未跑；确认结果不反馈选模。
20. 用独立小样本手算胜率、原始得分、聚类统计；CI跨.5输出INCONCLUSIVE而不是FAIL或promotion。
21. 主评估缺失输出EVALUATION_INCOMPLETE；不要合并开发集偷增确认样本。
22. 技术错误、超时、预算停止、打包异常都保留已验证进度和明确失败；不能异常被捕获后return假complete。

### D. 真实接线与实际交付

23. 显式本机CUDA上的生产模型维度有界接线：两套配方各真实有效更新、保存、恢复再更新、导出和少量对战。更新迭代限制在smoke级，不跑125/256轮或正式评估；若本地批准范围不足，报告缺项与待目标预检，不擅自扩大授权。
24. 实际双进程CPU/Gloo小型数学/恢复测试可补充DDP语义；有条件才做已允许范围的真实CUDA-DDP。单GPU/Gloo不是双T4/NCCL认证，结果清楚分层。
25. 完整构建后，从真正交付.bin解码到隔离目录，验证Notebook绑定、全部文件SHA256、源码与工作区对应、依赖/native身份、导入路径；避免PYTHONPATH或cwd使测试误用工作区模块。实际离线bootstrap/dry-run/小型编排测试运行于包内driver，不只是本地源码mock。
26. 新source包必须包含完整phase6/new modules；Notebook代码语法/入口/下载路径、恢复选项、真实结果包结构可用。
27. 回归macro/fuse/phase6 pool/旧continue500与recheck375/评估统计/源身份/打包安全，确保旧默认关闭路径无回归。

全量测试能跑则跑，实际记录命令、退出码、通过/失败/跳过/选择排除数和原因；资源不支持或已知无关dirty-tree失败明确说明。不能只把deselected当passed，不能删测试/放宽容差/改正式门槛。现有未通过项逐一归因，不照抄旧“2037 passed”作为本次证据。

## 11. 实施和独立审查产物

证据目录至少：

- PRE_EDIT_AUDIT.json及diff/hash清单；
- DESIGN_AND_CONTRACT.md：独立设计、两臂精确配置、初始化/source迁移、课程、KL口径、预算、统计与失败路径；
- IMPLEMENTATION_REPORT.md：修改文件、变更理由、实际测试与风险；
- 原始命令/stdout/stderr/exit code/耗时日志；
- LOCAL_VALIDATION.json：按CPU unit / Gloo / 单卡CUDA / 包内验证 / 双T4目标分别标记；
- ARTIFACT_QA.json：实际最终payload，不是中间包；
- CLAUDE_REVIEWER_REPORT.md：完整只读独立审查输出；
- HANDOFF_TO_CODEX.md：本文SHA、开工前后diff hash、文件列表、测试索引、交付路径及SHA、未运行项、复验命令与已知风险。

Reviewer先独立看本文、真实源码/diff和数学，再参考Implementer报告，保持只读，不自行修源码。必须增加至少三类新的反例，优先检查不等宏数DDP停机、错误阶段恢复/预算重置、最终实际包仍含旧source/错误native等，而非只重跑实施者happy path。所有未跑项标NOT_RUN。

新包只能在最终源码测试完成后构建；若Reviewer修复任何影响交付的源码，重新构建新版本并对最终包重新QA。不能修完工作区后仍交前一个hash的旧包。保存中间版本便于追溯但README只指向一个最新待复核版本。

## 12. 完成定义与最终回复格式

完成需要：功能和安全门实现；关键反例实际通过；真实有界本地验证或清楚列出硬件未运行边界；最终新payload/Notebook/文档与证据相互绑定；独立Claude Reviewer结论；Codex可直接开始复核。

不要因为用户想操作简单，就省掉本地测试；也不要为了论文级归因继续扩成十几个训练臂。核心是两套固定配方、一次提交、自动串行、预算可控、失败可恢复、证据可查。

最终用中文输出以下字段与路径，所有路径用绝对路径：

    IMPLEMENTATION_STATUS: COMPLETE / BLOCKED / FAILED
    CLAUDE_REVIEW_VERDICT: PASS / FAIL / INCONCLUSIVE
    CODEX_REVIEW: PENDING
    KAGGLE_EXECUTION: NOT_RUN
    TARGET_DUAL_T4_VALIDATION: NOT_RUN（未实际运行时必须如此）
    USER_RUN_APPROVAL: PENDING_CODEX_REVIEW
    FORMAL_PRODUCTION_PROMOTION: false

并给出：

1. 实施改动摘要和文件清单，哪些结构/规则没有改；
2. 实际测试命令、结果及原始日志；
3. 两臂配置、预算、评估数量、数据隔离与统计口径；
4. 最终交付目录、唯一Dataset文件和Notebook路径、各SHA256；
5. HANDOFF_TO_CODEX.md路径；
6. 未运行项、已知风险和硬件边界。

本次任务完成后停止，不启动Kaggle或正式训练、不自动进入下一阶段。Claude自审PASS不等于Codex已批准。用户会让Codex另做真实代码、反例与最终包复核，再决定上传。
