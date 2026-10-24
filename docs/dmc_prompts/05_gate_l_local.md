# DMC 阶段 5：Gate L —— Kaggle 小规模学习信号验证（本机不跑重负载）

日期：2026-10-24 · 状态：待执行（依赖阶段 4）
修订：2026-10-24 按用户约束改为 Kaggle 执行。本机只承担单测与阶段 4 的 200 局冒烟，一切规模数据都在 Kaggle 上跑。

## 目的

在投入 9h 大 session 之前，用一个 **~4h T4×2 session** 验证两件事：DMC 学习信号成立；实测吞吐供阶段 6/7 预算外推。PPO 对照数字：8000~20000 局时对 rule 55~66%（`docs/marvel_migration/reports/M09_final_holdout.md`、`docs/marvel_v14_prompts/CURRENT_EXECUTION.md`）。

## 前置任务 A：上传管线适配（一次性，完成前不得上 Kaggle）

现有 `tools/kaggle_pipeline` 的 `verify_delivery`（`tools/kaggle_pipeline/core.py:240`）硬绑定 serial_compare 校验器，只认那种交付格式。DMC 交付需要：

1. 新校验器 `scripts/verify_dmc_delivery.py`：校验 payload/notebook/plan.json 结构、预算字段（plan 内 `budgets.total_seconds` 必须 ≤ 平台 timeout − 300 秒，对齐 core.py:263 的既有约定）；
2. `configs/kaggle_pipeline.json` 的 `delivery_parent`/`experiment` 指向 DMC 交付目录与新实验名（如 `dmc-gate-l`）；
3. `prepare/run/watch/fetch` 全流程复用，不改管线代码本身；
4. 用假后端补 `tests/unit/test_dmc_delivery_verify.py`。

## 运行配置（configs/dmc_gate_l.json，打进交付包）

- envs=32、wall 4h、目标 ≥20,000 局（≈150 万+ 决策样本）；buffer=500k、batch=1024、梯度比 1.0、ε 退火按锁定表；
- 每 2,500 局 checkpoint；训练与评估同一 session 完成（沿用现有 pilot 内嵌评估模式）；
- 评估：~50% 时间点 mini 评估 vs rule 96 局；终点 vs rule 256 局 + vs random 256 局（ε=0 贪心）。

## 预声明门槛（写进 REPORT，不许事后改）

| 指标 | PASS | FAIL 处置 |
|---|---|---|
| 健康 | illegal=0，nonfinite=0，truncated=0 | 停，回阶段 2/4 查 bug |
| Q 校准 | 终点 qz_r2 ≥ 0.10 且总体上升 | 不升 → 查 z 通路/座位标签/特征 |
| vs rule | 终点 ≥ 65% 且 mini→终点上升 | <60% → ABORT 回查实现；60~65% → 允许一次续跑（同恢复链） |
| vs random | 终点 ≥ 85% | 否则同上 |
| 吞吐 | 4h 内 ≥20,000 局；记录 games/sec | 不足 → 先修吞吐（热点在枚举或前向批），不进阶段 6 |

## 明确禁止

- 不跑 FableDan、不碰 final holdout、不启用 belief/对手池；
- 不改算法代码；调参仅限 envs/batch（写进 REPORT 说明）。

## 交付物

Kaggle 结果 zip（checkpoint、指标 jsonl、评估原始数据、recovery）经 `kaggle.ps1 fetch` 归档到 `runs/dmc_gate_l_<date>/`，加 `REPORT.md`（门槛判定表、吞吐、NOT_RUN 清单、run_id→Kaggle 资源映射）。
