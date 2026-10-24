# DMC 阶段 4：ε-greedy 采样器与 DMC 训练循环

日期：2026-10-24 · 状态：待执行（依赖阶段 1~3）

## 背景事实（已核实）

- 现有采样器的环境驱动模式见 `guandan_marvel/runtime/sampler.py`（并行环境、批量前向、终局奖励收集）；后端选择在 `guandan_marvel/backends.py`（Python oracle 与 C++ 显式选择，无静默回退）。
- 完整动作的 token 序列可以被逐 token 提交回环境（`guandan_marvel/comparison/fabledan_guest.py:234` 起的用法：决策返回 `tokens` 元组后逐 token 走提交路径）。
- 奖励：`guandan_marvel/reward.py`，训练尺度 ÷3（`reward/training_scale=0.33333`）。

## 必读文件

1. `guandan_marvel/runtime/sampler.py`（环境批驱动、健康计数、终局收集的结构）
2. `guandan_marvel/env_batch.py`、`guandan_marvel/backends.py`
3. `guandan_marvel/comparison/fabledan_guest.py:230-300`（枚举→选择→token 提交的完整闭环）
4. 阶段 2/3 交付：`guandan_marvel/dmc/candidates.py`、`q_agent.py`、`replay.py`、`learner.py`

## 任务范围（新文件 `guandan_marvel/dmc/sampler.py`、`guandan_marvel/dmc/train.py`）

1. `DMCSampler`：
   - N 个并行环境（先接 Python oracle 后端，显式参数，不静默换）；发牌/上下文沿用现有 context schedule 接口；
   - 决策点检测 → `enumerate_candidates`（阶段 2）→ 全候选 `encode_action_features` → 跨环境拍平为 `[ΣB·K]` 一次 GPU 前向 → 还原每环境 Q 列表；
   - **单候选决策**：直接提交，不产生训练样本、不进 ε；
   - **多候选决策**：ε-greedy（均匀于合法候选；ε 按已见样本数线性退火 0.10→0.02，锁定表）；
   - 把所选候选的 canonical tokens 逐 token 提交回该环境；
   - 终局：每座位 z = 本队 `guandan_rank_points ÷ 3`；把该座位全部多候选决策样本（打包 obs_tokens/channels/动作特征/z/seat/game_id）推入 replay；
   - 健康计数对齐现有风格：illegal、truncated、nonfinite、每局决策数/样本数、ε 实际值、Q-argmax 命中率。
2. `train.py` 主循环（CLI：`python -m guandan_marvel.dmc.train --config <json>`）：
   - cycle = `games_per_cycle`(256) 局采集 → push replay → 梯度步数 = `round(新样本数 × gradient_replay_ratio / batch_size)`，默认比 1.0；
   - checkpoint（schema `dmc-v1`：模型/优化器/replay 统计/RNG/ε 进度/counter）与恢复：恢复后 replay 允许为空冷启（记录 `replay_reset=true`），其余必须精确；
   - 指标 jsonl（新 schema `guandan-marvel-dmc-metrics-v1`），心跳日志沿用仓库风格；
   - wall budget 与安全停止沿用现有 worker 约定。
3. 冒烟测试 `tests/dmc/test_train_smoke.py` + 本地脚本跑通：
   - 小配置（envs=8、games=200、buffer=50k、batch=256）端到端：illegal=0、truncated=0、每局样本数在 [15, 120] 区间、loss 有限；
   - **确定性**：同 seed 两次 50 局，产生的 (game_id, 决策数, 所选动作哈希) 序列完全一致；不同 seed 不同；
   - **吞吐**：REPORT 里给出 games/sec 与 samples/sec（GPU 与 CPU 各一次）。

## 明确禁止

- 不改 `runtime/sampler.py`、`training/local.py` 及一切 ippo 路径；不实现对手池/冻结对手（阶段 7 再说）。
- 不把温度采样、on-policy 契约混进来：DMC 的数据生成策略就是 ε-greedy，写进新契约文档字符串。

## 验收标准

```
python -m pytest tests/dmc -q
python -m guandan_marvel.dmc.train --config configs/dmc_smoke.json   # 200 局冒烟通过
python -m pytest tests/unit tests/marvel_adapter -q                  # 无回归
```

## 交付物

两模块 + 测试 + `configs/dmc_smoke.json` + `runs/dmc_stage4_<date>/REPORT.md`（吞吐表、确定性证据、样本量分布）。

## 本机边界（2026-10-24 修订）

本机只跑本文件的小冒烟与确定性测试（约半小时级）；**不跑任何规模数据，吞吐以阶段 5 Kaggle Gate L 实测为准**。本地粗测 games/sec 仅用于发现数量级异常（如 <0.1 局/秒说明有实现级性能 bug）。
