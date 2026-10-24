# DMC 阶段 1：语义与配置骨架（dmc_v1）

日期：2026-10-24 · 状态：待执行 · 执行方式：全新 Claude Code 会话

## 背景事实（已核实）

- `guandan_marvel/config.py:422` `algo_name: str = "vrpo"`；`:199-203` 校验宏训练 profile 仅允许 `algo_name=ippo`；`:492-498` `validate_training_semantics` 是算法/语义组合的准入点，且 `opponent_pool requires algo_name=ippo`。
- 本仓库纪律：新算法必须显式注册 + 语义矩阵校验 + 版本化，禁止静默降级。

## 必读文件

1. `docs/dmc_prompts/00_DMC_MIGRATION_PLAN.md` 第 2 节（设计决策锁定表）
2. `guandan_marvel/config.py`（全文，重点是 `validate_training_semantics`、`TrainConfig`、`PPOConfig`）
3. `tests/unit/test_two_stage_training.py` 中配置校验相关测试（了解测试风格）

## 任务范围

1. `TrainConfig.algo_name` 接受 `"dmc"`（默认仍为 `"vrpo"` 不变）。
2. 新增 `DMCRuntimeConfig` dataclass（放 `config.py` 或新 `guandan_marvel/dmc/__init__.py` 后从 config 引用，选耦合最小的方式）：
   - `buffer_capacity: int = 500_000`
   - `batch_size: int = 1024`
   - `games_per_cycle: int = 256`
   - `gradient_replay_ratio: float = 1.0`
   - `epsilon_start: float = 0.10`、`epsilon_final: float = 0.02`、`epsilon_decay_samples: int = 5_000_000`
   - `learning_rate: float = 1e-4`、`weight_decay: float = 0.0`、`grad_clip_norm: float = 1.0`
   - `action_feature_version: str = "GD-DMC-ACTION-1"`
   - `checkpoint_schema: str = "dmc-v1"`
3. 扩展 `validate_training_semantics`：`algo_name="dmc"` 时
   - 必须且只能配 `training_semantics_profile="dmc_v1"`（新增该枚举值）；
   - 拒绝（fail fast，复用现有报错风格）：`ppo.target_kl`、`opponent_pool`、一切 macro_* profile、`policy_regularizer_mode`（DMC 无此概念）；
   - `dmc` 不读取 `PPOConfig` 任何字段，文档字符串里写明。
4. 新测试 `tests/unit/test_dmc_config.py`：接受矩阵（dmc+dmc_v1 通过）、拒绝矩阵（dmc+macro_decision_v1、dmc+opponent_pool、dmc+target_kl、ippo+dmc_v1 均报错）、默认值快照。

## 明确禁止

- 修改 `vrpo`/`ippo` 的任何现有行为或默认值；改动 `config.py` 只允许"加法"。
- 引入任何训练/采样逻辑（本阶段只有配置与校验）。
- 弱化或跳过任何现有测试。

## 验收标准（执行人本人运行）

```
python -m pytest tests/unit/test_dmc_config.py -q        # 全绿
python -m pytest tests/unit -q -k "config or semantics"  # 现有配置测试不回归
git diff --stat   # 只涉及 config.py、guandan_marvel/dmc/__init__.py（如新建）、新测试文件
```

## 交付物

代码 + 测试 + `runs/dmc_stage1_<date>/REPORT.md`（列出改动文件、测试输出原文、NOT_RUN 清单）。
