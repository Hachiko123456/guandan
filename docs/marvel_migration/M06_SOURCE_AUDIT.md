# M06 评估来源审计

- 目标阶段：`M06_evaluation_champion`
- 参考仓库：`D:\project\doudizhu`
- 只读参考 commit：`acdcf199cf422e2b683cb22142b8e7cb33b14460`
- 参考许可证：MIT
- 日期：2026-10-10

## 1. 参考的通用组织概念

| 参考路径 | 使用方式 | Guandan 实现 |
|---|---|---|
| `evaluations/core/types.py` | 参考 versioned job/result 数据边界 | M06 使用独立 versioned plan/record/result dict；未复制 class |
| `evaluations/core/plan.py` | 参考不可变 evaluation plan 概念 | `evaluation/evaluator.py` 重写固定矩阵 plan |
| `evaluations/core/recording.py` | 参考逐局落盘和可恢复记录概念 | JSONL + fsync，Guandan schema 全新实现 |
| `evaluations/core/stats.py` | 参考 summary 分层组织 | `evaluation/stats.py` 重写 deal-cluster t CI 和四队座指标 |
| `marvel/eval/hooks.py` | 参考训练中周期评估 hook 边界 | `evaluation/hooks.py` 重写 merged-policy freeze 和后台 executor |

## 2. 未迁移的游戏专用代码

以下代码没有复制、改名或作为 Guandan host 使用：

- `evaluations/doudizhu/host.py`
- `evaluations/doudizhu/controller.py`
- `evaluations/doudizhu/scenario.py`
- `evaluations/doudizhu/summary.py`
- `evaluations/doudizhu/agents/*`

斗地主 landlord/peasants、叫地主、底牌、炸弹倍数和外部 PerfectDou/DouZero/FPDou 连接器均未进入 M06。

## 3. Guandan 专用重写

- `evaluation/host.py`：C++ V2 EnvBatch、四座位固定队伍、贡还阶段、V2 canonical row；
- `evaluation/agents.py`：V2 KV session、rule/random row agent、legacy V1 mirror adapter；
- `evaluation/stats.py`：按 base deal 聚类，不采用斗地主角色 CI；
- `evaluation/promotion.py`：Guandan champion gate 和原子 manifest；
- `evaluation/pool.py`：历史 champion/anchor 权重和恢复；
- `evaluation/manifests.py`：training/selection/final 隔离和 final one-shot audit；
- `evaluation/checkpoints.py`：V2 C++ identity fail-closed 与 legacy opponent-only；
- `evaluation/evaluator.py`：candidate/champion/rule/random/legacy/pool 固定矩阵。

## 4. 许可证和只读保证

实现沿用仓库已有 `guandan_marvel/LICENSE` 和 `NOTICE.md`。参考仓库未写入缓存、格式化结果、构建产物或测试文件；阶段结束再次检查其 Git 状态。

## 5. Repair additions（2026-10-10）

`evaluation/workers.py`、`evaluation/provenance.py`、final state machine、promotion transaction 和 bounded statistics 为 Guandan 项目新写实现。ProcessPool 仅采用 Python 标准库执行隔离；没有复制斗地主 host/controller/agent。
