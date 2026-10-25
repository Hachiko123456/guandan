# DMC 迁移总计划（2026-10-24）

> 本文件是 DMC（Deep Monte-Carlo）训练路线的总纲与阶段索引。实现通过 `docs/dmc_prompts/01..06` 分阶段执行，每阶段一个独立 Claude Code 会话。执行前先读本页"操作规程"。PPO/IPPO/VRPO 代码全部保留不动，DMC 以 `algo_name="dmc"` 并行注册。

## 1. 方向依据（论文 + 本地实证）

| 证据 | 结论 |
|---|---|
| DouZero（You et al., 2021, arXiv:2106.06135） | 斗地主 DMC 自博弈达到人类顶尖水平；消融显示 DMC 优于策略梯度基线 |
| **DanZero（Liu et al., AAAI 2022, arXiv:2210.17087）** | **掼蛋专用 DMC：第一个掼蛋 AI，160 CPU + 1 GPU 训练 30 天达到与人类玩家相当水平** |
| DanZero+（2023, arXiv:2312.02561） | DMC 路线在掼蛋上的直接改进与增强 |
| GuanZero（2024, arXiv:2402.13582）、Luo et al. 2024 | DMC 路线的后续变体，进一步验证可行性 |
| FableDan（本地实证） | 同为 DMC：cycle1910 ≈ 2.5 亿决策样本 ≈ 400 万局，Botzone #11；对 rule 97.7% |
| 本项目 m09（PPO 线） | 8000 局训练对 rule 仅 55~66%；iter500 打不过 iter250；熵塌缩 + critic 不稳（见 `runs/` 与 `docs/marvel_migration/reports/`） |

结论：DMC 是掼蛋上被 DanZero/FableDan 直接验证的路线；本项目 PPO 线在同等甚至更大局数下已确认平台化。

## 2. 设计决策锁定表（所有阶段不得偏离）

| 决策项 | 锁定值 | 理由 |
|---|---|---|
| 决策粒度 | **完整候选动作**（一次出牌/过牌为一个动作） | DanZero/DouZero/FableDan 全部如此；token 级 Q 语义不明 |
| 候选枚举 | Python 层复用 `guandan/action_state.py:383 _candidate_sequences()`（底层 `enumerate_legal_actions`，`guandan/environment.py:25` 有缓存）；C++ 导出仅作为后期吞吐优化 | 枚举器已存在且有缓存，避免先做 C++ |
| Q(s,a) 结构 | 复用 `model/agent.py` 主干 + 候选打分路径，输出每候选标量 Q；**不加载 ippo checkpoint**，fresh init | `agent.py:170` 已是 per-candidate 打分 |
| 动作特征 | 新版本化布局 `GD-DMC-ACTION-1`（64 维，见阶段 2 规格） | 对齐 FableDan 80 维动作块 |
| 回归目标 z | 该座位队伍终局名次分（现有 `guandan_rank_points`，±3/2/1 **÷3**），gamma=1 无折扣 | 与现有 reward.py 及 FableDan 完全一致 |
| 训练样本 | 仅"≥2 个合法候选"的决策点产生样本；单候选自动执行不进 buffer | FableDan 同规则；省数据 |
| 座位处理 | 4 座共享一个网络；每个决策样本带行动座位自己的 z | FableDan/DanZero 同 |
| 回放缓冲 | 扁平 FIFO，容量默认 500,000，均匀采样，无 target network，每 cycle 后用最新权重 | FableDan 262k/半换；DouZero 1M |
| 优化器 | AdamW lr=1e-4（恒定），weight_decay=0，grad clip 1.0，fp16 autocast + GradScaler | FableDan 同；T4 无 bf16 |
| 梯度/数据比 | 默认 1.0（每 1 个新样本 ≈ 1 次梯度样本通过），cycle=256 局 | FableDan 0.5；从保守值起步 |
| 探索 | ε-greedy：合法候选内均匀随机；ε 从 0.10 线性退火到 0.02（前 500 万样本） | 简单可复现；top-k 变体留作配置项默认关 |
| 评估 | ε=0 贪心 argmax；复用现有评估 lanes（rule/random/legacy/m09 h2h） | 现有 harness 直接用 |
| 语义版本 | `algo_name="dmc"` + `training_semantics_profile="dmc_v1"` + checkpoint schema `dmc-v1` | 仓库版本化纪律，禁止与 ippo 混淆 |
| 禁止事项 | 不蒸馏 FableDan；不删/不改 ippo·vrpo 路径；不弱化任何现有测试；不静默回退后端 | 项目红线 |

## 3. 阶段总览与门槛

| 阶段 | 内容 | 交付 | 门槛（不过则停） | 预计会话 |
|---|---|---|---|---|
| 1 | 配置/语义骨架 | `dmc` 注册 + 校验矩阵 + 测试 | pytest 全绿 | 1 |
| 2 | 完整候选枚举 + 动作特征 | `guandan_marvel/dmc/candidates.py` + 布局文档 + 模糊测试 | 枚举与 token 提交路径完全一致 | 1~2 |
| 3 | Q 头 + 回放缓冲 + 学习器 | `dmc/q_agent.py / replay.py / learner.py` + 单测 | 合成数据收敛测试通过 | 1~2 |
| 4 | ε-greedy 采样器 + 训练循环 | `dmc/sampler.py / train.py` + 200 局冒烟 | illegal=0、确定性、吞吐实测 | 1~2 |
| 5 | **Gate L：Kaggle 小规模信号验证**（~4h T4×2，含上传管线 DMC 适配） | 首跑结果已归档：**FAIL**（vs rule 59.8%<60%线、吞吐 0.711<0.85 局/s；health/Q校准/vs random 96.1% 三项 PASS） | 见 05 与 `runs/kaggle_pipeline/dmc-gate-l-*/gate_l_results_manual.zip` | 已完成（2026-10-25） |
| **07A** | **吞吐修复**：双卡采集/训练重叠、envs↑、cycle↑，目标 ≥1.5 局/s | `docs/dmc_prompts/07A_gate_l_repair_throughput.md` | 本机 ≥2x 基线；学习语义零改动 | 1~2 会话 |
| **07B** | **效率修复**：观测窗口 head+tail(128+896)、梯度累积 4×256=1024 | `docs/dmc_prompts/07B_gate_l_repair_window_batch.md` | 等价性/组成/确定性测试全绿 | 1 会话 |
| 5' | Gate L 重跑（07A+07B+07C 合并） | **PASS（2026-10-26）**：vs rule **65.2%≥65%** 且 56.3→65.2 上升；random 89.1%；qz_r2 0.193 上升；12,289 局 @1.024 局/s（早停外推 12,316>12,000 正确未触发） | 已完成 |
| 6 | **Gate A：Kaggle 方向验证**（9h session，≥10 万局 ≈ 800 万样本） | 结果包 + 与 m09-iter2000 h2h | 对 rule ≥ 75% 且 h2h m09 ≥ 60% | 1~2（操作型） |
| 7 | 规模化 + 增强（**仅大纲，Gate A 通过后再细化**） | 吞吐工程、belief、冠军池、多局实战 | 见 07 文件 | 事后 |

**放弃点**：Gate L 不过 → 修配方再上（不再加 session）；Gate A 时 ≥500 万样本仍 <65% vs rule → 停止扩量，回来查实现（大概率是特征或枚举 bug，不是 DMC 不行）。

## 4.1 本机与 Kaggle 的分工（2026-10-24 修订）

本机（RTX 3050 Ti 4GB）只承担：单元测试、阶段 4 的 200 局冒烟与确定性验证（约半小时级）。**一切规模数据（含 Gate L 的 2 万局）都在 Kaggle T4×2 上跑**，经 `scripts/kaggle.ps1 prepare/run -Delivery <DMC交付目录>` 提交（管线适配见 05 文件前置任务 A）。注意：现有管线的 `verify_delivery` 只认 serial_compare 交付格式（`tools/kaggle_pipeline/core.py:240`），DMC 必须先配自己的校验器；且每次不同内容的上传都会在 Kaggle 新建一个 dataset+notebook（`core.py:344`），VERIFIED 并归档结果后应清理旧的远端资源（Kaggle 网页手动删，或后续补 `cleanup_remote` 命令）。

## 4. Claude Code 操作规程（每阶段）

1. **一个阶段 = 一个全新会话**（`/clear` 或新开），避免上下文污染。输入：
   `读 docs/dmc_prompts/00_DMC_MIGRATION_PLAN.md 的"设计决策锁定表"，然后严格执行 docs/dmc_prompts/0X_*.md。`
2. 阶段 1~4 是代码阶段：优先用推理能力强的模型；阶段 5~6 是操作/分析阶段，轻量模型可用。
3. 每阶段结束后**你本人**执行该 prompt 里的"验收命令"，不采信模型自述。
4. 验收通过才 commit（一个阶段一个 commit），然后才开下一阶段会话。
5. 每阶段要求模型产出报告 `runs/dmc_stageN_YYYYMMDD/REPORT.md`（含 NOT_RUN 清单），沿用仓库现有证据纪律。

## 5. 预算速查

- FableDan 目标线：≈2.5 亿决策样本 ≈ 400 万局（3.5 个月 T4×2）。
- Gate L 门槛线：~4h T4×2 session，≥2 万局 ≈ 150 万+ 样本。
- Gate A 门槛线：≈800 万~1000 万样本 ≈ 10~15 万局（9h session ×1~2）。
- 本机：仅单测 + 阶段 4 冒烟（半小时级），不跑规模数据。

## 6. 主要风险与对策

| 风险 | 对策 |
|---|---|
| Python 枚举吞吐不足 | 阶段 4 实测 games/sec；DMC 每局只需 ~40 次前向（PPO 是 ~290 次 token 前向），天然省 7 倍；仍不足才做 C++ 导出（07 阶段） |
| 弱模型实现出隐性 bug | 每阶段验收命令可独立复核；阶段 2 的"枚举==可提交"一致性模糊测试是防线上界 |
| Q 值发散/不校准 | 学习器监控 corr(Q,z)、R²，Gate L 硬门槛 |
| 误改 PPO 路径 | 所有新代码进 `guandan_marvel/dmc/`，config 仅加法注册；git diff 审查清单见各阶段 |
