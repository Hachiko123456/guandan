# 风险登记与架构决策

## 1. 已确认原则

### D001 保留 Python 规则 oracle

决定：A01/A02 Python 实现不因 C++ 迁移而删除。

理由：需要独立差分参考、回放和调试。

### D002 V2 增量协议

决定：新增 MARVEL V2 action/env wire，不原地改变 V1。

理由：保护 legacy checkpoint 和已有评估证据。

### D003 双 GPU 训练同一模型

决定：正式 T4×2 使用 DDP。

理由：目标是提高单模型训练量，而不是并行占卡。

### D004 先直接策略，后 belief/search

决定：M10 在 M09 后。

理由：避免搜索掩盖基础模型和训练问题。

## 2. M01 决策（已完成）

### D005 Python 包命名

日期：2026-10-08
状态：accepted
候选方案：保留顶级 `marvel` 并强制隔离；使用独立命名空间。
决定：使用 `guandan_marvel`。
证据：`adr/ADR-005-006-marvel-core-namespace.md`；同进程冲突测试可同时导入外部 `marvel` sentinel、`guandan` 和 `guandan_marvel`。
影响：参考 import 需要显式改名，但同一环境不会因 `sys.path` 顺序加载错误包。
回滚方式：仅可在隔离发布物增加 `marvel` compatibility facade；仓库内 canonical namespace 不回滚。

### D006 通用代码复制方式

日期：2026-10-08
状态：accepted
候选方案：vendor 派生子包；共享本地依赖；运行时引用参考 checkout。
决定：在当前仓库 vendor 许可证保留的派生子包，记录逐文件 provenance，不依赖参考绝对路径。
证据：`M01_SOURCE_AUDIT.md`、`guandan_marvel/NOTICE.md`、`guandan_marvel/LICENSE` 和 wheel 安装验证。
影响：上游同步需要人工审查；Kaggle 打包和复现无需额外 checkout。
回滚方式：未来可拆成独立版本化 wheel，但不得恢复绝对路径依赖。

## 3. M02/M04 必须决策

### D007 结构化动作 row 的最大长度和词表

日期：2026-10-08
状态：accepted
候选方案：完整 canonical action 放入单 row；只放 scalar token；带 current-state metadata 的固定 row。
决定：采用 8-token row `[ROW_BEGIN, PHASE, PREFIX_STAGE, KIND, PAYLOAD_MARKER, V1_TOKEN_ID, IS_COMMIT, ROW_END]`，词表 320。V1 legal-next token 是唯一执行载荷，但必须整行匹配当前候选，避免 stale/forged row。
证据：`M02_ADAPTER_PROTOCOL.md`、`tests/marvel_adapter/test_adapter.py`、`evidence/M02_guandan_adapter_v2/verification.json`。V1 token 词表仅 256 个 ID，因此候选容量 256 是硬上界；15 局扫描实际最大 27。
影响：保持逐 token/COMMIT 时间轴，candidate row 可被 MARVEL action encoder 消费；不会枚举后静默 top-N。
回滚方式：新协议版本可扩展 row，但不得原地改变 `GD-ACTION-0.2-MARVEL`。

### D008 C++ 物理牌表示

需兼容双副 108 张物理牌、相同牌面不同 card_id、逢人配声明和确定性 canonicalization。

### D009 发牌和 chance step

日期：2026-10-08
状态：accepted（Python reference backend）
候选方案：reset 内确定性发牌；显式 chance node。
决定：M02 Python reference adapter 在 `reset(seed)` 内完成发牌，首个输出直接给实际 active player；普通 PLAY/TRIBUTE/RETURN 决策的 `player_ids` 均为真实座位。
证据：`M02_ADAPTER_PROTOCOL.md`、M02 reset/step/sampler tests。
影响：M02 sampler 无额外 chance transition；M04 C++ 若改变该决策必须升级环境协议并做差分。
回滚方式：只能在新 `GD-ENV` 版本引入显式 chance node，不得静默改变 V2 wire。

## 4. M03/M08 必须决策

### D010 Muon

默认只作为可选实验；先以 AdamW 建立可复现基线，再决定是否启用。

### D011 torch.compile

必须测首次编译时间、图重编译和总吞吐；若动态 shape 导致退化可关闭。

### D012 AMP/TF32

T4 支持 fp16/TF32 相关路径，但正确性、稳定性和速度必须实测。不得牺牲非有限值检查。

### D013 Replay 设备

VRPO replay 放 GPU、CPU pinned memory 或分级缓存，由显存和吞吐决定。

### D014 模型尺寸

128/256、4-layer 是基线，不是已证明最优。pilot 后才能调整。

## 5. 主要风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| C++ 与 Python 规则漂移 | 训练错误且难发现 | M05 大规模差分、Python oracle 不删除 |
| 动作容量低估 | 合法动作被截断 | 扫描上界、overflow hard fail |
| 四玩家观测泄露 | 评估虚高 | 重排隐私测试、actor/critic tensor 分离 |
| 完整轨迹显存过高 | OOM | compact channels、分批 learner、pilot 定型 |
| CPU 仍供不上 GPU | 利用率低 | C++ batch、profile rollout/learner 分段 |
| DDP 通信收益不足 | 双卡变慢 | 相同全局工作量短测、>=1.5× gate |
| selection 过拟合 | champion 虚高 | final holdout 隔离、版本化 deals |
| 旧 checkpoint 混用 | 错误 resume | schema/contract hash、evaluation_only 分类 |
| 文档状态再次过期 | 错误推进 | 单一 STATUS、每阶段提交时更新 |
| 盲目照抄斗地主默认值 | 资源/算法不适配 | M08 实测、ADR 记录 |

## 6. ADR 格式

后续决策追加：

```text
### Dxxx 标题
日期：YYYY-MM-DD
状态：proposed/accepted/superseded
候选方案：...
决定：...
证据：...
影响：...
回滚方式：...
```
### D015 On-policy sampler 概率契约
日期：2026-10-08
状态：accepted
候选方案：记录温度后完整分布；仅记录模型分布；限制正式采样温度。
决定：正式 on-policy 只允许 temperature=1，保存 `policy_probs` 和已选动作的 `behavior_action_prob`，evaluation/test 标为 `on_policy_valid=false`。
证据：M02 repair sampler tests 和 `sampler_contract.json`。
影响：非 1 温度与 forced action 不可静默进入 PPO。
回滚方式：若未来支持 off-policy/temperature correction，必须升级 sampler/buffer semantic contract。

### D016 Sampler 失败状态
日期：2026-10-08
状态：accepted
候选方案：完整回滚 cache/env/buffer；失败后 poison。
决定：采用 poison-on-failure。agent/cache 执行后的异常不增加 completed counters，sampler 禁止继续或 exact resume。
证据：horizon、multi-slot forged action、buffer write injection 和真实小型 Agent cache tests。
影响：实现简单且 fail closed；调用方必须从已知一致状态重建。
回滚方式：只有提供 env/buffer/cache/RNG 原子 rollback 证据后才能改为可继续策略。

### D017 M03 dense trajectory 内存门槛
日期：2026-10-08
状态：accepted
候选方案：直接沿用 512/2048；降低合法动作；分配前预算。
决定：M03 必须在分配前进行字段级预算并设置 per-rank hard limit。禁止 top-N。首个本地 smoke 使用显式 test-only T=512、rollout=1、collect=1。
证据：`memory_budget.json` 和 `test_memory_budget.py`。
影响：默认 512/2048/T4096 不可直接本地或双 T4 分配。
回滚方式：紧凑 dtype、裁剪、分桶/chunk 实现并验证后可重新定型。
### D018 M03 首个优化器/编译/精度 profile
日期：2026-10-08
状态：accepted（local smoke）
候选方案：Muon/compile/AMP 默认开启；全部关闭；逐项试验。
决定：首个语义 smoke 使用 AdamW，Muon=false、compile=false、AMP=false。Muon 与 compile 保持显式配置；AMP=true 在 M03 local runtime fail closed。
证据：M03 training tests、JSONL metrics 和 `training_smoke.json`。
影响：最小化语义验证变量，不代表正式配置最优。
回滚方式：M08 pilot 提供稳定性和吞吐证据后重新定型。

### D019 M03 checkpoint/replay persistence
日期：2026-10-08
状态：accepted
候选方案：只存模型 warm restart；保存 sampler/RNG；保存 sampler/RNG/replay exact resume。
决定：IPPO 保存 sampler/RNG/optimizer，VRPO 额外持久化 replay；schema 为 `guandan-marvel-training-checkpoint-v2`。
证据：exact continuation tests 和 `checkpoint_verification.json`。
影响：checkpoint 较大，但 local smoke 可证明 exact resume；缺 replay 自动分类 warm restart。
回滚方式：未来若改为外部 replay artifact，必须升级 schema 并明确恢复分类。
### D020 M03 catchable interruption strategy
日期：2026-10-10
状态：accepted
候选方案：每 optimizer step 深拷贝回滚；异常后 poison。
决定：采用 trainer+sampler poison。捕获 `BaseException` 后记录 partial work、拒绝 payload/save/retry，并重新抛出。恢复只能新建实例并加载最后完整 checkpoint。
证据：M03 repair interruption tests 和 `repair_verification.json`。
影响：正常 step 无完整深拷贝开销；不承诺平台硬杀恢复。
回滚方式：只有实现并验证模型/optimizer/env/buffer/RNG/cache 原子回滚后才可允许原实例继续。

### D021 M03 semantic/source compatibility policy
日期：2026-10-10
状态：accepted
候选方案：resolved config hash 不同即 compatible；完整语义/源码逐项比较。
决定：checkpoint v3 嵌入 semantic config、组件级 source fingerprint 和 execution fingerprint。仅批准的日志/显示/运行控制字段可 compatible；模型、gamma、optimizer、schedule、sampling、replay、源码、设备和依赖变化保守 migration/reject。
证据：真实 gamma/decoder-head checkpoint tests、source fixture mutation 和 artifact verifier。
影响：pre-repair v2 不再作为 exact/compatible training resume。
回滚方式：未来若增加安全迁移器，必须使用新 schema 和逐字段转换证据。

### D012 M05 性能门槛分段与默认后端切换

决定：M05 的硬性能门槛使用三个同机同工作量、backend-sensitive workload：single-env、batch 和冻结策略端到端 rollout，三者均须 >=2×。完整 learner iteration 仍必须执行并分段报告，但不把 backend-independent 的相同 learner/model 时间混算成环境加速或用来否定已隔离的环境门槛。

决定：正确性 10k/100k 为 0 mismatch 且性能门槛通过后，正式训练默认切换为 C++；Python reference 继续保留为显式 oracle，不允许 silent fallback。

影响：报告必须同时公开完整 iteration 总比值、rollout 比值和 learner 比值。当前本机完整 IPPO iteration 总比值仅约 1.075×，不得写成 >=2×；C++ 默认资格来自已定义且全部通过的 backend-sensitive 门槛。

### D013 默认 backend 导入、恢复身份与真实性能补修

决定：包根使用 extension-free lazy API；`default_game` 表示默认 C++，历史 `game` 固定为 Python oracle。禁止通过 `sys.modules` 重载、吞掉 ImportError 或 fallback 掩盖循环导入。

决定：checkpoint v4 的 exact/compatible resume 必须匹配实际 backend、backend-specific source fingerprint 和原生 binary/build identity。v3 缺失这些字段时只允许 migration/evaluation-only，不自动补全。

决定：原 2.976× lane 重新命名为 deterministic selector rollout。总体计划对真实模型 end-to-end 门槛存在歧义时采用保守口径：真实 frozen Transformer rollout 也须 >=2× 才重新 accepted。当前补测未达到，因此 M05 保持未接受，M06 继续 blocked；learner 本身仍不要求 2×。

### D014 真实 Agent 性能三轮停止决策

决定：真实性能收尾固定 4 局/1,027 actions 主集合、8 局泛化集合、同一冻结权重和至少 5 次交替配对样本。最多实施三轮 profiler 驱动候选，不无限试错。

Round 1 保留 active-only post-decoder heads；四座位 history/KV 仍完整更新，仅消除最终被 mask 的三份 state/action/policy head 重复计算。Round 2 event snapshot 和 Round 3 outer-transaction mutation 因收益不足且扩大 C++ 差分风险而撤回。

最终真实 Agent speedup 1.636×，未达到 2×。尽管 fixed-action single/batch 继续超过 2×，不得据此替代真实模型门槛。M05 继续 blocked；继续深度优化或调整门槛必须由用户明确决定。


### D015 M05 性能验收分类修订

日期：2026-10-10
状态：accepted
批准来源：用户明确批准调整 M05 验收标准。

决定：正确性、隐私、合法动作、牌守恒、serialization/resume、checkpoint/backend identity 和 no-silent-fallback 继续为硬门槛。fixed-action single-env 与 batch 的同机 Python→C++ 加速继续要求分别 `>=2×`。本机 CPU 真实冻结 Transformer rollout、额外 Transformer 集合和完整 IPPO iteration 改为必须公开的非阻塞优化目标。

证据：fixed single `3.203×`、fixed batch `2.051×`；真实 Transformer 主集合 `1.636×`、额外集合 `1.585×`、完整 iteration `1.084×`。旧 blocked 报告和旧门槛失败证据保持不变，由 `M05_ACCEPTANCE_CRITERIA_REVISION.md` 和新的修订验证文件叠加登记修订后接受结论。

影响：M05 在没有新硬性 blocker 时可接受并解除 M06 blocker。该决定不证明 Kaggle T4×2 性能、双卡效率、模型强度或可实战性；这些仍分别由 M08/M09 验证。

回滚方式：如后续发现任何正确性、安全、隐私、恢复或身份缺陷，立即重新阻塞 M05/M06；不得以本修订豁免。


### D016 M06 selection、final 和 champion 边界

日期：2026-10-10
状态：accepted

决定：training/selection/final seed domain 在一个 versioned deals manifest 中全局去重。Selection 使用固定五 lane 矩阵和四座位轮换；同一 base deal 的四局只作为一个统计 cluster。Final 使用 candidate-locked one-shot audit，永远不产生 promotion。

决定：promotion 采用 `promote/reject/inconclusive` 三态。少于两个完整 clusters 或 CI 不足为 inconclusive；点估计 50.1% 不晋升。Champion 和 pool 都使用 hash-protected atomic manifest。

决定：legacy V1 只经 Python mirror adapter 作为 opponent；V2 candidate/champion 必须匹配 C++ checkpoint/build/protocol identity。

影响：M06 本地演练可以验证管线和恢复，但不构成强度、final holdout 或远端吞吐结论。


### D017 M06 集中补修安全边界

日期：2026-10-10
状态：accepted

决定：异步评估从线程改为 spawn process，所有 terminal results durable 后再 delivery；trainer 自动收集，close 收集尾部结果。

决定：final 使用 stable dataset identity 和 unused/reserved/running/consumed|failed|interrupted 状态机，reserve-before-touch，不支持失败后重新开新 evaluation。

决定：promotion decision v2 精确绑定 candidate/checkpoint content、champion manifest generation、selection plan/summary/evaluation/deals；champion/pool 采用显式 pending transaction recovery boundary。

决定：training checkpoint v5 保存 actual deal provenance；v4 只允许 smoke-only selection。

决定：formal promotion 至少 32 independent deal clusters，使用 bounded empirical-Bernstein 95% CI；overall 跨 opponent 不给 clustered CI。

影响：原 M06 accepted 结论在 repair 完成前撤回；旧报告和产物保持历史只读。


### D018 Final audit actual-dataset binding

日期：2026-10-10
状态：accepted

决定：final audit reservation 不再隐式使用 `self.dataset_hash` 代替外部核验。Runner 计算并显式传入实际 dataset identity、generator version 和 deal count；audit 在独占锁内对调用值、对象预期值和持久化值做三方比较。

兼容性：schema v2 和持久化状态不变；reserve API 新增必填 identity 参数。旧终态记录不重置。
## M07 决策：DDP rank shard 与 native binary 随 recovery 保存

单一 rank 0 checkpoint 无法精确恢复另一个 rank 的 sampler/RNG/replay；重新编译 native extension 又可能改变 M05 要求的 binary identity。因此 M07 采用 rank 0 原子发布的两 shard envelope，并在 recovery ZIP 中携带实际 extension binary。代价是 recovery 体积增加，但避免伪 exact resume、rank stream 丢失和跨会话 binary hash 漂移。
