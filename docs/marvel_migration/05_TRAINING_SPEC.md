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

## 2.1 On-policy 采样概率契约（M02 补修后）

正式 IPPO/VRPO fresh rollout 必须使用：

```text
purpose=on_policy
sampling_temperature=1.0
forced_action_indices=None
```

trajectory 字段语义：

- `policy_probs[B,T,A]`：rollout actor 原始策略分布，用于 reference KL 和 Q-value policy expectation；
- `behavior_action_prob[B,T]`：实际行为分布对已选动作的概率，用于 PPO 旧概率；
- `on_policy_valid[B,T]`：正式 learner eligibility。

`purpose=evaluation|test` 可使用 greedy、非 1 温度和 forced legal actions，但必须写 `on_policy_valid=false`。PPO 入口必须拒绝这些样本，不能把评估/测试 trajectory 静默混入 fresh on-policy batch。

sampler 在 agent/cache 执行后失败时采用 fail-closed poisoned state：completed counters 不增加，后续 rollout/collect 拒绝，不能称 exact resume。

## 2.2 Dense trajectory 分配前内存门槛

M03 在构造 `TrajectoryBuffer`/`Sampler` 前必须调用 `trajectory_memory_budget()` 或 `require_trajectory_memory_budget()`：

1. 打印字段级 shape、dtype、单 trajectory、rollout buffer、commit buffer 和每 rank 估计；
2. 配置显式 `max_sampler_bytes_per_rank`；
3. 超预算时在任何 dense buffer 分配前失败；
4. 预算还必须另列模型、optimizer、KV cache、activations、replay 和临时复制，不能把 sampler estimate 当总内存。

当前默认协议 `T=4096,A=256,L_action=8,P=4` 下，修复后 dense layout 单 trajectory 约 87.148 MiB。若全局 `rollout=512, collect=2048`，仅 rollout+commit buffers 约 261.445 GiB；两 rank 均分仍约 130.723 GiB/rank。

M03 推荐本地 smoke：

```text
backend=python_reference
purpose=on_policy
rollout_batch_size=1
collect_batch_size=1
replay_buffer_scale=1
mini_batches=1
history_dim=16
feature_dim=24
decoder_layers=1
decoder_heads=4
state_layers=1
action_layers=1
fuse_layers=1
smoke-only max_game_horizon=512
```

`T=512` 必须使用带独立 identifier 的测试专用 `GameSpec`，所有接收的 trajectory 都要在 512 前自然终局；不得把该结果描述为默认 `T=4096` 协议验证。该 sampler 形状约 32.681 MiB（1 rollout + 2-slot commit capacity）。

紧凑 dtype、按实际长度裁剪、分桶和 chunk/spill 是后续优化方案，本轮不通过 top-N 删除合法动作，也不假设 C++ 自动解决 tensor buffer 内存。

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
## 11. M03 本地实现状态（2026-10-08）

M03 本地 runtime 已实现：

- IPPO value head、多玩家 GAE、2+ policy epochs、clip/KL/value loss、gradient finite/clip；
- VRPO centralized Q critic、fresh actor GAE、complete replay、fresh-first/replay-next critic aux epochs、lambda/TD1 metrics；
- AdamW 默认，Muon/compile 可配置，首个 smoke 显式关闭 Muon/compile/AMP；
- resume actor 与 merged evaluation copy；beta=0 标记为 `current_policy_copy`；
- JSONL core/health/Guandan metrics；
- V2 checkpoint、sampler/RNG/replay exact continuation；
- incomplete/nonterminal batch 不增加 durable update。

实际验收 profile 使用 CPU、16/24 小模型、`T=512`、rollout/collect/replay=1；它只证明训练语义。
## 12. M03 interruption/durable boundary repair（2026-10-09）

`iteration` and `durable_updates` advance only after all actor/critic work and metric persistence complete. `learner_updates` may record partial work for diagnostics, while `durable_learner_updates` records the last checkpointable boundary. Catchable interruption/exception poisons trainer and sampler, rejects payload/save/retry, and requires reconstruction from the last complete checkpoint.

Resume compatibility now compares the actual semantic configuration and component-level implementation fingerprint; resolved-config hash differences are not automatically compatible.

## 12. M06 training evaluator hook

训练器可在 durable iteration 边界附加 `TrainingEvaluatorHook`。每 N iteration 冻结 merged policy 为 `guandan-marvel-evaluation-checkpoint-v1`，后台执行 selection evaluation。Snapshot 不含 optimizer、sampler、replay 或 RNG resume state，不能训练恢复。已有评估 pending 时允许记录并跳过，不阻塞下一训练 iteration；调度错误不 poison trainer。Final holdout 不得由 training hook 调用。

## 13. M06 repair training provenance

Canonical training checkpoint 为 v5，sampler state 为 v2。On-policy sampler 记录初始 stream seed、连续 PRNG 语义和每个实际训练 deal hash；checkpoint 保存 hash-protected `training_data_provenance`。v4 缺 provenance，训练 resume 分类为 migration，仅允许 smoke-only evaluation。Training evaluator 使用 spawn process；trainer 在 durable iteration 自动收集结果，worker 永远不接收 live training state。
## 14. M07 单模型双 rank DDP

M07 将既有 trainer 扩展为显式 DDP runtime：global rollout/collect batch 按 world size 等分，每 rank 独立环境 seed，模型初始化一致，trainable actor/critic 由 `DistributedDataParallel` 包装并同步梯度。远端只允许 `WORLD_SIZE=2`、NCCL、CUDA；CPU/gloo 仅用于明确标记的本地双进程恢复测试。

Durable checkpoint 由 rank 0 原子发布 `guandan-marvel-ddp-checkpoint-v1` envelope；每 rank shard 保存 sampler/RNG/replay，manifest 验证跨 rank resume/merged/critic state hash 一致并汇总全局 counters。当前恢复策略只接受相同 world size。完整协议见 `M07_KAGGLE_DDP_PROTOCOL.md`。
## 15. M07 补修 DDP resume 与预算边界

DDP resume 必须继承 checkpoint 的模型、batch、optimizer、sampling/replay 和训练数据来源，只允许日志/输出/追加迭代/会话预算等明确 run-control 差异。连续执行和 evaluation-enabled checkpoint/resume 的下一状态以完整 continuation hash 验证。预算停止只在完整 update 后由所有 rank collective 决定，并在共同 checkpoint boundary 停止。
