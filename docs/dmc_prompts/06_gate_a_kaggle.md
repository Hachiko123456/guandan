# DMC 阶段 6：Gate A —— Kaggle 方向验证（第一个远端 DMC 训练）

日期：2026-10-24 · 状态：待执行（依赖阶段 5 PASS）

## 目的

用 1~2 个 T4×2 session（9h 预算）回答一个问题：**DMC 在 10 万局量级是否显著越过 PPO 平台期**。这是是否投入规模化（阶段 7）的唯一依据。

## 交付包

- 复用现有 Kaggle 交付/恢复/评估骨架（`guandan_marvel/kaggle/package.py`、`worker.py`、pilot 模式），新 profile 名 `m10-dmc-gate-a-v1`；
- 上传统一走 `scripts/kaggle.ps1 prepare/run -Delivery <目录>`（阶段 5 已完成 DMC 交付校验器适配，见 05 前置任务 A）；VERIFIED 后归档结果并清理旧的远端 dataset/notebook；
- 数据/身份纪律照旧：source manifest、native 构建校验、recovery zip、心跳；DMC 侧新增 replay 统计与 ε 进度进入 checkpoint 身份；
- 训练目标：≥100,000 局（≈800 万+ 决策样本）；每 5,000 局 checkpoint；wall 9h 不足则按预算安全停并保留恢复。

## 评估矩阵（训练后同一 session 内完成，复用现有 lanes）

对终点与中点（≈50k 局）两个 checkpoint：

| 对手 | 局数 | 说明 |
|---|---:|---|
| rule | 512 | 与 m09 历史数字直接可比（m09: 55~66%） |
| random | 512 | 健康下限 |
| legacy | 512 | 与历史可比 |
| **m09-iter2000 h2h** | 512 | 冻结对手，复用 recheck375 式配对发牌（128 副 × 4 轮换） |

## 预声明门槛（进 REPORT，不许事后改）

| 指标 | PASS（进阶段 7） | ABORT |
|---|---|---|
| vs rule | ≥75% 且中点→终点上升 | <65%，或 ≥500 万样本仍平台 |
| h2h m09 | ≥60% | <55% |
| vs random | ≥90% | <85% |
| 健康 | illegal/truncated/nonfinite=0 | 任何违例 |
| 吞吐 | 记录 games/sec（外推阶段 7 日历） | — |

ABORT 处置：不换算法、不加算力；回本地按"特征/枚举/z 标签/回放比例"四个怀疑点逐项差分测试，修完重跑 Gate A。

## 明确禁止

- 不跑 FableDan（省配额；h2h FableDan 是阶段 7 的周期性监控项）；
- 不碰 final holdout；不启用 belief/对手池；不改模型结构。

## 交付物

Kaggle 结果 zip（指标、checkpoint、评估、recovery）+ `runs/dmc_gate_a_<date>/REPORT.md`（门槛判定 + 吞吐外推 + 阶段 7 规模预算建议）。
