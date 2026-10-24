# MARVEL v1.4 当前执行入口（2026-10-22）

本文件可更新；旧README、阶段1manifest/report/probe与原05/06属于冻结历史，不覆盖。

## 最新：v12 candidate恢复预检窗口修复（2026-10-24）

- 用户v10真实Kaggle日志通过输入检查、control训练/恢复/四场评估、candidate首次两迭代；candidate从iter2精确恢复后的唯一迭代产生0 optimizer update（与target-KL早停路径一致，外层日志不足以排除其他零更新来源），旧单迭代门槛报错。正式两臂训练未开始。
- 根因是预检只抽一个恢复迭代却强制该随机样本必须更新；不是GPU/NCCL、checkpoint、上传混包或v10候选池batch修复回归。
- v12将恢复预检改为完整3迭代窗口，窗口内仍必须至少一次真实更新；三次全0、恢复不完整或预算截断继续fail-closed。配方、target-KL、plan、模型/native、正式预算和评估均不变。
- 当前唯一交付为`runs/t4x2_serial_compare_delivery_20261023/delivery_v12/`；actual payload QA PASS，payload SHA256 `d05a0e1663724b8448f2ebbd0e1c2eebd93c683768227f8cf7cdd86eeecdbb56`。修复规范`docs/marvel_v14_prompts/08_candidate_resume_zero_update_repair_20261024.md`，证据`runs/t4x2_serial_compare_resume_window_fix_20261024/`。v11是本地中间签包，v10及以下只作历史，均不再上传。
- 一键入口为`scripts/upload_kaggle_v12.ps1`；本轮只运行`-DryRun`，未上传、未启动Kaggle。

## 最新：v10候选池batch隔离校验修复（2026-10-24）

- v9真实目标日志：control更新、保存、恢复、评估全部通过；candidate collect因assigned_seats跨batch误比第一局而失败，正式训练未开始。
- policy_owner.validate_pooled_trajectory现按每B行的有效acting前缀验证座位恒定，独立牌局允许不同分配；局内assignment/owner/mask等保护保留。
- 新增23项批量边界测试+15项相关owner/GAE/mask/DDP测试通过；本机CUDA生产网络batch4连续2轮+恢复后第3轮，共12条多分配轨迹、11次有效更新及导出2局评估通过。
- 当前唯一交付`runs/t4x2_serial_compare_delivery_20261023/delivery_v10/`。相对v9配方/数据/计划不变，旧包不覆盖；证据`runs/t4x2_serial_compare_pool_batch_fix_20261024/`。双T4/NCCL与完整正式训练仍需目标预检，不由单卡证据代替。
- 以下v9及以前仅作历史。

## 最新：v9评估数据语义与真实GPU评估修复（2026-10-23）

- v8已通过control更新/保存/恢复后，评估因level_rank=2不是合法内部Rank而失败；同生成器的DOUBLE_DOWN/HEAD_THIRD排名也已纠正。
- 当前唯一新入口`runs/t4x2_serial_compare_delivery_20261023/delivery_v9/`，旧v8及此前不再上传；模型/训练配方/预算/种子不变，评估manifest升级v2并重新签hash。
- 新增开训前168条真实配置/native reset；本机CUDA实际17场评估（13局覆盖场景+2局真实CLI+2局候选更新/恢复后导出评估）及记录/汇总/结果ZIP验证。候选单次有效更新和at-target恢复亦通过，使用生产结构、小batch与明确本地评估迁移，无CPU模型训练、无正式长训/远端提交；双T4/NCCL仍由目标preflight验证。
- 证据`runs/t4x2_serial_compare_evaluation_fix_20261023/`；下方v8及更早记录仅历史。

## 最新：v8 torchrun启动入口修复（2026-10-23）

- 用户回传v7约59秒在control预检失败，尚未进入正式两臂训练。v7暂停使用。
- Codex直接修复worker从torchrun环境读取world/rank/local_rank并按local_rank选CUDA，由训练器初始化DDP；补逐rank错误文件和父进程原始异常摘要。
- 新交付唯一入口：`runs/t4x2_serial_compare_delivery_20261023/delivery_v8/`，使用同目录的`.bin`与Notebook；来源/配方/计划不变。
- 本轮证据：`runs/t4x2_serial_compare_launch_fix_20261023/`。本机启动/通信与错误传播测试不等同双T4/NCCL认证；首次Run All必须通过两臂目标preflight。未启动远端任务。
- 下方v7等记录仅作历史，不作为本轮上传入口。

## 当前任务：一次上传、双配方串行训练与统一评估（2026-10-23，实施/待Codex复核、未远端运行）

新任务 `docs/marvel_v14_prompts/07_single_upload_serial_recipe_comparison_20261023.md`（SHA256 `6fd303ee0ab80b0160aea8e10f0d1a1c67e276fb4e914833b4cbe8f76573ab2b`）已按本会话完成**实现与本地验证**：

- 新编排模块 `tools/serial_compare/`；新构建/校验脚本 `scripts/build_t4x2_serial_compare.py`、`scripts/verify_t4x2_serial_compare_delivery.py`；用户说明 `docs/t4x2_serial_compare.md`。
- 交付目录 `runs/t4x2_serial_compare_delivery_20261023/delivery_v7/`（continuation identity v2 最小修复后的唯一候选版本，待 Codex 复核；v1–v6 仅追溯且不可上传）；实施证据 `runs/t4x2_serial_compare_implementation_20261023/`（含 `repair_round3/`、`repair_round4_state_identity/`），修复说明见 `REPAIR_ROUND1_REPORT.md`、`REPAIR_ROUND2_REPORT.md`、`repair_round3/REPAIR_REPORT.md` 与 `repair_round4_state_identity/FIX_REPORT.md`。
- 07A 第三方修复 Prompt `docs/marvel_v14_prompts/07A_checkpoint_save_transaction_repair_20261023.md`（SHA256 `95e31ff0…db99`）；后续最小修复单 `runs/t4x2_serial_compare_codex_acceptance_v6_20261023/MINIMAL_FIX_HANDOFF.md`（SHA256 `8d11c21b…8168`）。独立 Reviewer PASS / ALLOW（限构建一次 v7）；双 T4/NCCL、Kaggle、704 场、FableDan、holdout 均 NOT_RUN，用户上传仍需 Codex 复核与批准。
- 核心允许改动：`guandan_marvel/config.py`（可选 `target_kl`，默认 None 惰性）、`guandan_marvel/training/local.py`（可选宏样本 KL 早停、课程逻辑起点、CHECKPOINT 记录）、`guandan_marvel/training/macro_ppo.py`（全局 sum/count KL 与分位数/分桶）、新增 `guandan_marvel/training/policy_import.py`（版本化 fresh 初始化迁移）、`guandan_marvel/phase6/opponent_pool.py`（路由 session/cache 容量审计）。
- **未运行 Kaggle、未提交远端作业、未做正式训练/50+iter、未运行双 T4/NCCL、未运行 FableDan、未触碰 final holdout**；`formal_production_promotion=false`。
- 本地边界：单卡 RTX 3050 Ti 4GiB + Gloo；双 T4/NCCL、生产 batch 全尺寸 CUDA、真实 704 场确认评估均标 `NOT_RUN`。详见交付包 `LOCAL_VALIDATION.json` / `ARTIFACT_QA.json`。
- 本条目只记录“实施完成、待 Codex 复核”，不是批准，也不改变下文历史记录。

## 当前任务：iter375同集复测，只评估不训练（2026-10-22）

用户授权准备可上传Kaggle的下一步文件；本轮落实“评估优化＋375同集复测”，不启动历史对手池A/B或继续训练。

- 用户回传continue500已完成250→500、新增10000局，终点500对250为44.53%、对rule62.89%（250为66.02%），正式INCONCLUSIVE；详见 `runs/t4x2_continue500_analysis_20261022/REPORT.md`。
- 新操作入口：`docs/t4x2_recheck375.md`；独立交付目录 `runs/t4x2_recheck375_delivery_20261022/delivery_v1/`，旧包不覆盖。
- 两个唯一整队分配；旧1536终点记录→768唯一缓存，启动前验证所有重复动作轨迹、胜率、得分与按牌局CI等价。
- 复用250/500，不重跑整套旧基线；375新增对rule、对250、对500直接对战，共128副×2分配×3臂=768场。另24场缓存环境重放；375冒烟24场已计入768、不重复执行。
- 每个arm都分到两张卡，缺失任务恢复时重新均衡。先真实缓存重放，再375小冒烟，再余下744场；任一门槛失败立即停止，正常错误保存已验证逐局进度。
- 这是旧开发集上的事后诊断；formal promotion=false，无FableDan，无final holdout，无训练/optimizer/sampler恢复数据。
- 原CPython3.13.15 / Torch2.11.0+cu128 / Linux native原绝对路径身份保留；不修改生产模型头，不重新编译或静默回退。
- 全流程90分钟安全预算；本机单GPU接线测试不冒充双T4认证，目标机实际缓存重放门槛仍必须通过。
- Dataset只上传 `guandan_recheck375_payload_v1.bin`；Notebook单独导入 `guandan_recheck375_v1.ipynb`。恢复使用本包的整个 `guandan_recheck375_results.zip` 或其被Kaggle展开的目录，不是旧训练恢复包。

下文500续训包已由用户完成，保留历史操作与来源追溯，不作为当前新增训练授权。

## 当前任务：v4 完整状态续训 iter250 → iter500（2026-10-22）

用户明确授权：直接增加到500、修复流程错误、整理新Kaggle包，由用户上传运行；不代表代理可以自行启动远端训练。

- 当前操作说明：`docs/t4x2_continue500.md`；新交付目录：`runs/t4x2_continue500_delivery_20261022/delivery_v1/`，不覆盖旧v4。
- 起点是用户回传 **iter250 / 10000局** 的完整双rank恢复包，而不是iter50 policy-only warm start。绝对终点500 / 累计20000局，新增最多10000局。
- 原v4 source ZIP、Linux native、optimizer/sampler/RNG/scaler不变。独立 `tools/continuation500/` 驱动只负责新调度、观察和评估；恢复记为兼容目标/日志变更，不伪造旧source hash。
- 保留真实v4配方：clip=.2、reg=.01、lr=1e-4，NTP=.02、原贡还课程已启用，rollout48/collect40，horizon1024。下文旧v1的“暂不启用NTP/贡还、5000局、4096horizon”等不描述v4或本次续训。
- 旧native绝对源码路径和Python3.13.15属于严格后端身份：新驱动把原源码/二进制绑定到原路径，先验证backend hash，不修改checkpoint字段。
- 修复旧评估写死4096、摘要/恢复导出写死每轮20局的问题。开训前真实48局评估；2轮更新/保存+1轮恢复/保存后，再48局新导出评估；门槛失败立刻停，不进入长段。
- 固定iter250基线，375监控，500预声明开发终点评估；分座位、逐局hash、聚类CI、原始得分/胜率/配对rule增量。无FableDan、无最终holdout；统计INCONCLUSIVE与技术失败分开。
- 4小时全流程预算、90分钟终点评估预留，不足安全保存实际进度；已500恢复只补评估，不新增训练。双T4/NCCL训练与双GPU并发评估均在目标机预检，不承诺100%占用率或绝对不会发生平台故障。
- 本地验证范围以交付 `LOCAL_VALIDATION.json` 和 `ARTIFACT_QA.json` 为准；本机单卡CUDA与CPU/Gloo兼容性测试不冒充真实双T4验收。
- Dataset仅上传 `guandan_continue500_payload_v1.bin`，Notebook单独导入 `guandan_continue500_v1.ipynb`，两者不可与旧版混用。

以下是历史交付记录，不是当前上传指令。

## 最新交付：三参数组合与双 T4 提速（2026-10-21）

用户选择新组合 clip=0.2 / reg=0.01 / lr=1e-4，暂不加入贡还和 NTP；授权本地实现、验证与交付准备，不代表代理可自行启动 Kaggle。

- 新入口：`docs/t4x2_combo_pilot.md`；交付目标 `runs/t4x2_combo_delivery_20261021/delivery_v1/`，不覆盖旧 v2/v3。
- 从用户回传的 candidate_iter000050.pt（SHA256 `1f4de9ad180398c3041e6737588a4f92f59e2e97d07d5868e9066efbee725eda`）policy-only warm start，保留已训练 ActorFuse；重置 optimizer/sampler/counters，curriculum 显式 offset=50。
- 本轮最多新增250迭代/5000局，2+1+247=250；三参数段内恒定。不是旧50轮的exact resume，也不是8万局长训授权。
- 并发采样12环境/卡，学习collect仍10局/卡；4GiB sampler预算与4096 horizon不放宽。目标设备全尺寸峰值仍须Kaggle预检。
- 每50轮可靠保存，ZIP快速压缩，验证恢复副本后提前退役上一阶段冗余raw分片；NCCL梯度 + Gloo元数据侧组，新增存档分段计时、采样host分段和新旧策略approx-KL日志。
- 新开发集128独立发牌×4轮换×3臂=1536局；3.5小时总预算、90分钟评估预留。无FableDan、无final holdout，不把rule或单分支改参结果当严格超参数A/B。
- **本地交付就绪，目标设备待运行**。相关回归275 passed / 3 deselected；另1项真实双进程CPU/Gloo新配置更新/保存/恢复通过；本机CUDA生产模型继承90键、两次有效更新及exact resume通过（小batch/horizon512，不是双T4认证）。完整边界见新包 LOCAL_VALIDATION.json。
- Dataset只上传 `guandan_t4x2_combo_v1.bin`；Notebook单独导入 `guandan_t4x2_combo_v1.ipynb`。Payload SHA256=`4874c281908230e33516e7c36492330838958b3140197c9ddcc3870de8867f81`；内层manifest=`15eb63eed37cc11c7818495306b463d1e0194691210eaff90e8fe7c4adc366c7`。
- 实际产物解码/哈希/186源码文件逐字节比对、52个离线依赖文件完整性、6项输入反例拒绝均通过，见新包 ARTIFACT_QA.json。预检通过前不声称全尺寸显存、双T4/NCCL或端到端提速已经认证。
- 上一轮v3已由用户在Kaggle完成50迭代/1000局和384局开发评估，棋力主判定INCONCLUSIVE；新包是其后的新配方实验，不覆盖/混用旧v3，不运行新远端训练，也不承诺100%利用或加速倍数。

下文为旧实验与历史约束，不再把旧50轮交付当成新三参数包。

## 最新：Kaggle 输入传输热修复 v3（2026-10-21）

用户上传 v2 后在约29秒的输入校验阶段失败，指向 `guandan-marvel-source.zip`，尚未编译/训练。日志预期哈希与本地v2一致；因未包含实际文件状态，归档被展开、挂载布局变化或内容变化尚不能仅凭该日志定论。

- 当前上传入口改为 `runs/t4x2_pilot_delivery_20261021/delivery_v3/`。
- Dataset只上传 `pilot_payload_v3.bin`；Notebook导入 `guandan_t4x2_pilot_v3.ipynb`，选择T4×2。
- 传输SHA256：`c7fa72b2e65632aa6fff862a9234739e43f1f0f281d70f46a8e8a6bdea4db3b3`；内层原始manifest保持 `7735964d48096476ba0fc8a256ce8e78c81afd1cb0bd56d002894c514ea66d4b`，源码/配置/权重/发牌逐字节未变。
- 新增固定哈希外层、Base64还原、文件清单/实际哈希诊断，不取消任何身份守卫。操作见该目录 `README_KAGGLE_V3.md`。
- 仅修复交付传输，不重训、不改变50iter/1000局上限，不把本地传输验证称为真实Kaggle认证。

下文v2上传步骤是历史交付记录，当前不要据此继续上传ZIP。

## 本轮：T4×2 50-iteration 学习效果实验交付（2026-10-21）

用户已授权准备短训实验交付，并由用户上传 Kaggle 手动运行；本会话未启动远端作业或 50 iter 训练。

- 新入口：[T4×2 上传与恢复说明](../t4x2_learning_pilot.md)。交付目录：`runs/t4x2_pilot_delivery_20261021/delivery_v2/`。
- 上传 `pilot-upload.zip` 作为 Dataset，导入 `guandan_t4x2_pilot.ipynb`，明确选择 T4×2，Run All。
- manifest hash：`7735964d48096476ba0fc8a256ce8e78c81afd1cb0bd56d002894c514ea66d4b`；Notebook 绑定这个精确版本，不可混用旧 Dataset/source/recovery。
- 2 iter 预检 → 恢复到 3 → 继续到绝对50，总计1000局；总预算12600秒，预留评估与保存时间。不会自动继续250/500/1500。
- 维持宏决策IPPO/fuse/既有超参，新增显式性能profile：global rollout=16、collect=20、FP16 learner AMP、NCCL两rank、有效前缀布局去尾部padding与cache prefill无效计算（合法horizon仍4096）。
- 训练后用32独立开发发牌×4轮换，对照训练前control、rule；两GPU分片共384局。不运行FableDan、不消耗final holdout，不宣称棋力提升。
- 本地最终相关回归152 passed；pilot契约单测35 passed（与前者有重叠，不相加）；生产模型尺寸、小批量/horizon512的真实M09 warm-start、两次CUDA更新与exact resume通过；两rank CPU/Gloo兼容运行与恢复通过。**不是双T4/NCCL认证**。
- 一次中间工作区全量回归为2069 passed / 2 failed / 2 skipped：其一为原有Guance dirty-tree守卫，其二为新增测试读取migration字段层级错误，已修复并在最终相关回归中通过；未把这次全量记录称为全绿或最终源码全量认证。完整本地边界见交付 `LOCAL_VALIDATION.json`。
- 若已训练到50但评估中断，同源recovery会只补跑开发评估，零新增训练更新。
- 旧 `delivery/` 是本地中间产物，不再作为上传入口；仅使用 `delivery_v2/` 中相互绑定的Notebook和ZIP。
- 若远端预检、显存、有效更新、保存/恢复或评估完整性失败，停止并下载日志/`latest_recovery.zip`，不自动换CPU、缩horizon、扩大显存守卫或跳过失败。

本次pilot交付就绪不等于既有生产交付、条件功能或模型强度全部认证。下文保留上一轮工程验收与准备约束，发生入口冲突时本节仅覆盖本次明确的50iter实验交付范围。

## 开发验收状态（2026-10-20 阶段证据）

- 阶段1–4含2A/04A已有历史独立验收；阶段5含actor/prefix修复已有Reviewer PASS/ALLOW。
- 阶段6/06B工程收尾现已Reviewer PASS / NEXT_STAGE ALLOW，无P1/P2；Codex核对实际Reviewer日志、所有报告after文件hash、qualification producer与design身份、完整全量日志hash，并重跑22项tie-aware/planner测试，全部通过。
- 完整复查：`runs/phase6_closeout/codex_acceptance_20261020/CODEX_ACCEPTANCE.json`及`CLAUDE_REVIEWER_PASS.md`。
- 全量：2037 passed / 2 skipped / 1 known Guance dirty-tree failure，exit=1，不能说全绿；Guance handoff/守卫未改。
- 当前qualification producer hash=9545496e411712718aba7eb26c55a71a5552257c559f10c490749d680e217344，design hash=90f11870dc817e0149cca662ca4845c91244ea16fe286e9c93e7175e4fbcd1e1。

这是阶段6条件分支工程验收，不是所有可选功能或正式生产运行都验证完毕：

```text
BELIEF_B0: REJECT
BELIEF_B1: SKIPPED / OFF
VALUE_GATE: INCONCLUSIVE
SEARCH_IMPLEMENTED: False
SEARCH_STATUS: NOT_RUN_VALUE_INCONCLUSIVE
DELIVERY_PENDING: true
```

Symmetry/Pool默认开启与否必须在正式训练配置中明确冻结，工程正确不代表棋力提升。对称性默认关闭、Belief关闭、search关闭，不因验收通过自行启用。

## 既有生产训练准备约束（2026-10-20 背景）

不要重做旧05B/05C/05D/05E或从头重跑06A/06B，不再新增功能开发阶段。若用户授权准备，按以下顺序：

1. 冻结最终代码/启用功能/模型架构/AMP/seed/局数与wall预算，区分纯self-play与Pool可选训练。
2. 根据当前阶段5/6源码重建新版本正式交付，不覆盖旧04A包；所有archive/profile/warm-start/source/semantic/native/hash核对一致。
3. 检查旧checkpoint加载语义：source已变不能冒充exact_resume；policy-only warm start或明确版本化迁移后，optimizer/sampler/counters按协议重置。
4. 当前源码、目标设备、真实配置做最小GPU启动/有效更新/保存/恢复验证；本机小smoke不能证明Kaggle双T4/NCCL或正式显存预算通过。
5. 只有准备检查通过且用户明确授权，才启动50 iter Gate1，之后每个绝对目标Gate单独检查；不得自动一路跑满1500。

既有CUDA pool A/B/C证据只覆盖未变路径的历史实现，不等于当前source重新GPU认证。若要正式value qualification，先解决Reviewer P3里的样本/ranking持久化、metric schema/provenance、mixed-all-tie偏严及最小当前源码GPU接线验证；不能把只写metadata的execute产物当可复算资格证据。

不必强迫Value Gate PASS才能训练基础策略；资格不足就保持search关闭。未来再做64/128正式qualification或搜索，须单独授权、冻结协议与原始证据。

## 上次索引更新的执行边界（2026-10-20 历史记录）

本索引更新没有启动训练准备、GPU测试、长训、Kaggle、双T4/NCCL、FableDan正式评估或final holdout。

所有真实模型/probe执行用 D:/project/guandan/.venv-gpu/Scripts/python.exe 与显式CUDA；CPU规则/环境/统计和明确单位/兼容测试可CPU，无静默fallback。

保留OpenCode和全局环境，不改Guance handoff、不改历史冻结标准。阶段5/6已结束的自主预算不沿用到训练准备或长训，新的授权单独登记。
