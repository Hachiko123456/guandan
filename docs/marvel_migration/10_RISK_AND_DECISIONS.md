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
