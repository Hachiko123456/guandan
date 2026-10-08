# MARVEL V2 训练规范

## 1. 基线模型

首个可比较版本使用 MARVEL v1.3 风格：

```text
architecture=v1.3
history_dim=128
feature_dim=256
decoder_layers=4
decoder_heads=4
state_layers=3
action_layers=2
fuse_layers=2
```

M03 允许更小尺寸做 smoke，但报告必须明确写出实际配置，不能把小模型称为正式基线。

## 2. 采样单位

正式 iteration 以 `collect_batch_size` 个完整终局 trajectory 为学习批次，而不是固定 token-transition 数。运行时同时记录：

- env/token steps；
- committed game steps；
- completed games；
- trajectories；
- learner updates；
- iterations。

## 3. IPPO

初始算法基线参考：

```text
base_lr=4e-4
agent_epoch=4
gae_lambda=0.95
base_clip_coef=0.02
base_reg_coef=0.1
discount_gamma=1.0
value_loss_coef=0.5
grad_max_norm=1.0
```

流程：完整轨迹 -> multi-player GAE -> trajectory shuffle -> mini-batch -> clipped PG + uniform/reference KL regularization + value loss -> gradient clip -> optimizer step -> merged/EMA policy。

## 4. VRPO

VRPO 必须是独立实现：

- actor 不含 value head；
- centralized action-value critic；
- fresh rollout 上计算 Q-based GAE；
- policy PPO 更新；
- trajectory replay buffer；
- first critic epoch 使用 fresh rollout；
- 后续 critic auxiliary epochs 从 replay 采样；
- 记录 lambda-return 和 TD(1) Q loss；
- actor/critic 独立 optimizer 和 schedule。

参考基线：

```text
vrpo_base_lr=4e-4
gae_lambda=0.95
aux_epoch=4
replay_buffer_scale=64（只作上限参考，M08 实测决定）
```

## 5. 优化器和调度

迁移框架应支持：

- AdamW；
- 可选 Muon + AdamW 参数分组；
- weight decay 排除 norm/embed/bias；
- agent LR decay；
- critic LR decay；
- clip coefficient decay；
- regularization decay；
- gradient norm clipping。

是否启用 Muon、compile、TF32 和 AMP 必须由配置和 pilot 证据决定，不能成为不可关闭的隐式默认。

## 6. merged/EMA policy

训练权重和评估权重必须区分：

- resume actor：继续优化；
- merged/EMA actor：评估和发布候选。

`agent_merging_beta=0.0` 等价于跟随当前模型，不能描述成已启用平滑。M08 应测试合适 beta 或保持 0 并记录原因。

## 7. DDP

T4×2 正式训练：

- 每 rank 一张 GPU；
- 每 rank 独立环境 seed；
- 全局 rollout/collect batch 能被 world size 整除；
- 梯度通过 DDP 同步；
- rank 0 写日志和 checkpoint；
- 全局 env/game step 通过 reduce 汇总；
- checkpoint 恢复后 world size 变化必须有明确兼容策略。

## 8. 必需训练指标

通用：

```text
statistics/env_steps_total
statistics/game_steps_total
statistics/completed_games_total
statistics/env_steps_mean
statistics/game_steps_mean
statistics/trajectory_length_mean
statistics/return_team0_mean
statistics/return_team1_mean
```

IPPO：

```text
agent_initial/pg_loss
agent_initial/value_loss
agent_initial/kl_unif
agent_initial/kl_ref
agent_initial/clip_frac
agent_initial/adv_mean
agent_initial/adv_std
agent/pg_loss
agent/value_loss
agent/kl_unif
agent/kl_ref
agent/clip_frac
agent/adv_mean
agent/adv_std
agent/learning_rate
agent/clip_coef
agent/reg_coef
```

VRPO 追加：

```text
critic_initial/qv_lbd_loss
critic_initial/qv_td1_loss
critic/qv_lbd_loss
critic/qv_td1_loss
critic/learning_rate
buffer/replay_buffer_size
```

性能和健康：

```text
perf/env_steps_per_second
perf/game_steps_per_second
perf/completed_games_per_second
perf/rollout_seconds
perf/learner_seconds
perf/evaluation_seconds
perf/gpu_utilization/<rank>
perf/gpu_memory_bytes/<rank>
health/illegal_actions
health/truncated_games
health/nonfinite_losses
health/nonfinite_gradients
health/action_capacity_overflows
health/horizon_overflows
```

掼蛋行为：

```text
guandan/token_steps_per_game_{mean,p50,p95,p99}
guandan/commit_steps_per_game_{mean,p95}
guandan/pass_rate
guandan/bomb_rate
guandan/straight_flush_rate
guandan/four_joker_rate
guandan/wild_use_rate
guandan/tribute_rate
guandan/anti_tribute_rate
guandan/double_down_rate
guandan/head_third_rate
guandan/head_last_rate
```

## 9. checkpoint

checkpoint 至少保存：

- 模型架构和 state dict；
- resume/merged actor；
- critic；
- optimizer；
- schedules/iteration；
- env/game/completed-game counters；
- RNG；
- replay buffer 恢复策略；
- protocol versions；
- source commit；
- resolved config；
- semantic contract hash。

## 10. 强度声明

训练 loss 下降、GPU 利用率、完成 iteration 或超过 random 都不能单独证明可实战。强度只由 `06_EVALUATION_CHAMPION_SPEC.md` 的独立评估决定。
