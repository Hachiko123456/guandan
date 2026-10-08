# M03 MARVEL IPPO/VRPO 本地训练阶段报告

- 日期：2026-10-08
- 分支：`codex/marvel-guandan-migration`
- 开始 commit：`6997cf8526f2852115f2df06ed279f1846682138`
- 实现 commit ref：`refs/tags/m03-marvel-training-smoke-accepted`（提交后由 tag 固定；准确 SHA 见 Git/final）
- 环境：Windows 11；Python 3.12.0；PyTorch 2.13.0+cpu；NumPy 2.3.5；CUDA unavailable
- 阶段状态：`accepted`

## 范围

在 `python_reference` backend 上接通小型 MARVEL Transformer Agent、IPPO value head、VRPO centralized Q critic、complete-trajectory sampler、trajectory replay、版本化 checkpoint/resume、JSONL metrics 和内存预分配门槛。

本阶段只验证本地训练语义。没有 C++、DDP、Kaggle、selection/champion 或强度评估。

## 实际修改文件

核心实现：

- `guandan_marvel/training/local.py`：本地 IPPO/VRPO trainer；
- `training/ippo.py`、`training/vrpo.py`：canonical entry points；
- `training/checkpoints.py`：V2 checkpoint 和 resume 分类；
- `training/validation.py`：完整轨迹、on-policy、finite、privacy、replay gate；
- `training/metrics.py`：05 spec 的本地 core/health/Guandan metrics；
- `training/smoke.py`：显式 test-only `T=512` game identity；
- `runtime/sampler.py`：checkpointable sampler state 和 terminal info queue；
- `config.py`：AMP/memory/replay persistence 显式配置。

工具和测试：

- `scripts/run_marvel_training_smoke.py`；
- `scripts/verify_marvel_training_artifacts.py`；
- `tests/marvel_training/test_training_smoke.py`。

文档：

- `M03_TRAINING_PROTOCOL.md`；
- `M03_SOURCE_AUDIT.md`；
- 训练规范、checkpoint 兼容性、source mapping、STATUS 和阶段证据。

## 内存预检

所有 sampler 分配前要求显式 `max_sampler_bytes_per_rank` 并执行 `require_trajectory_memory_budget()`。

本地验收 profile：

```text
rollout=1
collect=1
replay_buffer_scale=1
mini_batches=1
T=512 (test-only identifier)
history_dim=16
feature_dim=24
1 decoder layer / 4 heads
1 state/action/fuse layer
agent_epoch=2
critic_aux_epoch=2
AdamW
Muon=false
compile=false
AMP=false
```

估计：

- sampler dense buffers：34,268,184 bytes，约 32.681 MiB；
- VRPO replay capacity 1：11,422,728 bytes，约 10.894 MiB。

默认 `T=4096, rollout=512, collect=2048` 仍禁止直接分配。所有 smoke trajectory 在 512 前自然终局；不把 reduced horizon 描述为默认协议验证。

## IPPO 实际 smoke

执行方式：真实未训练小型 Transformer policy，自主 `purpose=on_policy`、temperature=1 采样，无 forced actions。

结果：

- 2 iterations；
- 2 durable updates；
- 4 optimizer learner updates（每 iteration 2 policy epochs）；
- 2 个真实完整终局；
- trajectory lengths：334、311；
- total env steps：645；
- total committed game steps：293；
- checkpoint iteration 1 保存后 exact resume，再完成 iteration 2；
- merged identity：`current_policy_copy`，beta=0 未称 EMA。

最终本地 loss/health：

```text
agent/pg_loss        -0.0004685098
agent/value_loss      0.0263503380
agent/kl_ref          0.0000833612
agent/grad_norm       0.15398598
nonfinite_losses      0
nonfinite_gradients   0
illegal/truncated     0
```

## VRPO 实际 smoke

结果：

- 2 iterations；
- 2 durable updates；
- 8 learner updates（每 iteration 2 actor + 2 critic）；
- 2 个真实完整终局；
- trajectory lengths：304、330；
- total env steps：634；
- total committed game steps：266；
- replay capacity/size：1；
- first critic aux epoch 使用 fresh rollout，second 使用 replay；
- checkpoint 保存 replay，exact resume 后继续 iteration 2。

最终本地 loss/health：

```text
agent/pg_loss              -0.0009128628
critic/qv_lbd_loss          0.0155547289
critic/qv_td1_loss          0.0015842289
agent/grad_norm             0.08449832
critic/grad_norm            0.05026963
nonfinite_losses            0
nonfinite_gradients         0
illegal/truncated           0
```

这些 loss 仅证明有限且更新路径可运行，不代表策略变强。

## Checkpoint 和 resume

Schema：`guandan-marvel-training-checkpoint-v2`。

保存内容包括：

- resume actor、merged/current-copy actor；
- IPPO value head；
- VRPO critic；
- actor/critic optimizers；
- schedule/iteration/learner/durable counters；
- config/source/semantic hashes；
- Python/NumPy/Torch RNG；
- sampler/environment/partial trajectory state；
- VRPO replay；
- selection/champion identity 为 null。

验证：

- IPPO exact resume 后第二次更新与不中断 control 模型权重、计数和 metric 一致；
- VRPO replay 恢复后继续更新；
- 损坏/nonterminal replay 被拒绝；
- exact、compatible、warm restart、migration、evaluation-only 全部分类通过；
- legacy V1 checkpoint 被拒绝作为 V2 training resume。

Checkpoint 位于 `runs/M03_marvel_training_smoke/20261008_cpu_smoke_acceptance/`，不提交 Git；路径、大小和 SHA-256 记录在提交的小型证据中。

## Metrics

每 iteration 写版本化 JSONL，包含：

- env/game/completed/learner/durable counters；
- returns、trajectory/commit lengths；
- PPO/critic loss、KL、clip、advantages、learning rates、grad norms；
- replay size；
- rollout/learner timings；
- illegal/truncation/nonfinite/overflow health；
- pass/bomb/straight-flush/four-kings/wild/tribute/outcome metrics。

CPU smoke 的 GPU utilization 为 null，未伪造 GPU 证据。

## 测试命令与结果

1. M03 + M01/M02 scoped：69 passed，0 skipped，exit 0，88.461s。
2. M03 training smoke tests：7 passed，0 skipped，exit 0，29.833s。
3. A01-A04 protected acceptance：152 passed，0 skipped，exit 0，15.704s。
4. Legacy unit + integration：889 passed，10 skipped，exit 0，226.309s。
5. Real IPPO/VRPO smoke runner：exit 0。
6. Checkpoint artifact verifier：exit 0。
7. Clean wheel import：exit 0。
8. `git diff --check`：exit 0。

10 个 skip 是既有 CUDA/symlink 环境限制；M03 没有新增 skip。完整命令、JUnit、日志 hash 和 skip 原因见 `evidence/M03_marvel_training_smoke/test_results.json`。

## 机器证据

- `protocol_schema.json`
- `memory_preflight.json`
- `training_smoke.json`
- `checkpoint_verification.json`
- `metrics_manifest.json`
- `test_results.json`
- `source_manifest.json`
- `artifact_manifest.json`

## 未修改的受保护范围

未修改 A01/A02 规则真值：

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `guandan/state.py`
- `docs/rules.md`

未删除 legacy V1 trainer/model/checkpoint evaluator。参考仓库保持只读和 clean。

## 已知限制

- 仅 CPU/local smoke；无 local GPU，因为当前环境 CUDA unavailable。
- `T=512` 是显式 reduced-horizon smoke identity，不是默认 `T=4096` 验证。
- Python reference 吞吐不是正式训练性能。
- checkpoint 较大，因为 exact resume 保存 dense sampler/replay state；紧凑/裁剪属于后续工作。
- AMP 首个 smoke 显式关闭；M03 local runtime 对启用 AMP fail closed。
- 未实现 selection/champion、C++、DDP 或 Kaggle。

## 是否允许启动下一阶段

是。M03 本地训练语义硬门槛已满足，可以启动 M04_cpp_runtime。此结论不包含模型强度、性能或远端声明。
