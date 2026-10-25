# DMC 阶段 6：Gate A —— Kaggle 方向验证（第一个正式规模训练）

日期：2026-10-24 · 修订：2026-10-26（Gate L PASS 后按实测校准执行参数）· 状态：待执行

## 目的

回答一个问题：**DMC 在 5~10 万局量级是否显著越过 PPO 平台期并击败 m09**。这是是否投入规模化（阶段 7）的唯一依据。

## 执行参数（按 Gate L 实测校准）

- **启用 07A 并行拓扑**（collector `cuda:1` / learner `cuda:0`）——Gate L 重跑实为串行（1.024 局/s），训练块仍未隐藏，并行预计 ~1.4 局/s；
- 单 session 9h：train_budget ≈ 30,000s，平台 timeout 32,400s；单 session 目标 ≥40,000 局；**100,000 局通过恢复链跨 2~3 个 session 达成**（runner 的 resume 契约沿用）；
- 每 2,048 局 checkpoint；envs=64、cycle=512、batch=1024(4×256)、head+tail 窗口、fast 驱动——全部沿用 Gate L v2 已验证配置。

## 评估矩阵（终点 checkpoint，训练后同一 session 内完成）

| 对手 | 局数 | 说明 |
|---|---:|---|
| rule | 512 | Gate L 65.2% 的延续测量 |
| random | 512 | 健康下限 |
| legacy | 512 | 与 m09 历史可比 |
| **m09-iter2000 h2h** | 512 | **需将 m09 冻结候选（SHA256 `af0df1ea…4ca9c8`）嵌入 payload + 官方加载器 lane**，配对发牌 128 副 × 4 轮换 |

## 预声明门槛（不许事后改）

| 指标 | PASS（进阶段 7） | ABORT |
|---|---|---|
| vs rule | ≥75% 且相对 Gate L(65.2%) 上升 | <68% 或平台 |
| h2h m09 | ≥60% | <55% |
| vs random | ≥90% | <85% |
| 健康 | illegal/truncated/nonfinite=0 | 任何违例 |
| 吞吐 | 记录 games/s（外推阶段 7 日历） | — |

ABORT 处置：不换算法不加算力，按"特征/枚举/z 标签/回放比例"四个怀疑点逐项差分测试。

## 明确禁止

- 不跑 FableDan（留到 ~2500 万样本后的监控节点）；不碰 final holdout；不启用 belief/对手池；不改模型结构。

## 交付物

新 builder/driver（Gate A 配置 + m09 lane + 并行启用）+ 结果包 + `runs/dmc_gate_a_<date>/REPORT.md`（门槛判定 + 吞吐外推 + 阶段 7 预算建议）。上传统一走 `scripts/kaggle.ps1`；VERIFIED 后归档并清理旧远端资源。
