# M03 训练中断与恢复契约补修报告

- 日期：2026-10-09
- 分支：`codex/marvel-guandan-migration`
- 原始实现：`4968840679ea031ec7b69132cebd862efa302c17`
- 原始验收标签：`refs/tags/m03-marvel-training-smoke-accepted`（保持不动）
- 补修实现 commit：待第一笔补修提交，由第二笔验收登记固定
- 状态：`locally_verified_pending_registration`

## 范围

本轮只修复：

1. catchable interruption 可绕过 trainer 失败保护；
2. checkpoint semantic contract 未覆盖真实模型/学习参数及完整源码实现。

保留原 IPPO/VRPO 数学目标、完整轨迹、replay、metrics 和正常恢复结构。未开始 M04，未执行 C++、DDP、Kaggle、champion 或长训练。

## 原始问题复现

### KeyboardInterrupt 绕过保护：已真实复现

使用原始 iteration 1 IPPO checkpoint，在 iteration 2 第一笔 optimizer step 后注入 `KeyboardInterrupt`：

```text
中断前: iteration=1 durable=1 learner=2
中断后: iteration=1 durable=1 learner=3
changed weight tensors=33
trainer.failed=false
sampler.failed=false
checkpoint_payload succeeded=true
payload resume_classification=exact_resume
```

环境、模型、optimizer、sampler 已不再属于同一完整 iteration 边界，但原实现仍允许保存。

### Semantic contract 漏项：已真实复现

同一原始 checkpoint：

- `discount_gamma: 1.0 -> 0.5`：成功加载并返回 `compatible_resume`；
- `decoder_heads: 4 -> 2`：权重 shape 仍兼容，成功加载并返回 `compatible_resume`。

这证明只比较 architecture 标签和 resolved-config hash 差异不足以保护训练语义。

复现证据：`review_findings.json`，原始运行日志保存在 `runs/M03_marvel_training_smoke_repair/20261009_pre_repair/`。

## 中断修复策略

采用禁用策略 B，不为每个 optimizer step 做昂贵深拷贝。

Trainer 生命周期：

```text
idle -> running -> completed
idle/completed -> loading -> completed
running/loading -> failed
```

修复后 `RuntimeError`、`KeyboardInterrupt` 及其他可捕获 `BaseException`：

1. 标记 trainer `failed`、`iteration_status=failed`；
2. 同时 invalidate sampler；
3. 记录 active iteration、partial learner updates、durable boundary；
4. 重新抛出原异常，不吞掉停止请求；
5. 拒绝后续 `train_iteration()`、sampler collect、`checkpoint_payload()` 和 `save_checkpoint()`；
6. 只能新建实例并加载最后一个完整 checkpoint。

Checkpointable boundary 同时要求：

- trainer/sampler healthy；
- 没有 running/loading operation；
- `iteration == durable_updates`；
- `learner_updates == durable_learner_updates`；
- 至少一个 durable iteration。

本轮不承诺无法被 Python 捕获的平台硬杀或断电恢复。

## 中断修复实际证据

新 v3 checkpoint iteration 1 后，在 iteration 2 第一笔 optimizer step 后中断：

```text
iteration=1
durable_updates=1
learner_updates_observed=3
durable_learner_updates=2
partial_learner_updates=1
changed weight tensors=33
iteration_status=failed
trainer.failed=true
sampler.failed=true
previous checkpoint unchanged=true
```

中断后：

- `checkpoint_payload()`：拒绝；
- `save_checkpoint()`：拒绝；
- `train_iteration()` retry：拒绝；
- sampler collect retry：拒绝；
- 原异常 `KeyboardInterrupt`：重新抛出。

新建实例加载最后完整 checkpoint 后返回 `exact_resume`，并正常完成 iteration 2。

专项测试还覆盖：

- IPPO 多 epoch 中途 `RuntimeError`；
- VRPO actor 已更新、critic 尚未执行时 `KeyboardInterrupt`；
- sampler cached forward 已推进后 `KeyboardInterrupt`；
- checkpoint load 失败后的 trainer/sampler invalidation；
- 运行中直接请求 payload 的拒绝。

## Checkpoint/semantic contract 升级

版本升级：

```text
runtime:  guandan-marvel-m03-local-runtime-v2
checkpoint: guandan-marvel-training-checkpoint-v3
semantic contract: guandan-marvel-training-contract-v2
semantic config: guandan-marvel-semantic-config-v1
source fingerprint: guandan-marvel-source-fingerprint-v1
execution fingerprint: guandan-marvel-execution-fingerprint-v1
```

Semantic configuration 覆盖当前实现实际消费的：

- model architecture、history/feature dimensions；
- decoder layers/heads、state/action/fuse layers、FFN ratio；
- 完整 game spec 和协议容量；
- gamma、IPPO/VRPO GAE、epoch、clip/reg/value/Q 规则和 learning rates；
- optimizer 类型、betas、weight decay、gradient clipping；
- schedule powers 和 stable iterations；
- merged beta/identity；
- seed、rollout/collect/replay/minibatch；
- replay persistence/device；
- deterministic/TF32、AMP、compile、DDP；
- sampler/buffer/Q-target contract。

所有声明 hash 在加载时重新计算，不只信任 payload 字符串。

## Resume 分类边界

### exact_resume

要求：

- semantic contract/config 完全一致；
- component source fingerprint 完全一致；
- Python/PyTorch/NumPy/device/CUDA execution fingerprint 一致；
- resolved config 无差异；
- optimizer、sampler、RNG、replay 等必要状态完整。

不对跨设备或跨依赖版本做未经验证的逐位 exact 承诺。

### compatible_resume

仅允许明确批准的非语义变化：

- project/run name；
- logging directory；
- debug/log/checkpoint/evaluation display intervals；
- total requested iterations；
- resume path。

实际 logging_dir/run_name 变化已加载为 `compatible_resume` 并继续训练。

### migration/reject

以下变化不允许普通 compatible：

- gamma；
- decoder heads 或其他模型结构；
- optimizer 参数；
- schedule 参数；
- sampling/replay/merged 定义；
- source fingerprint；
- device/dependency execution fingerprint；
- contract/config/source 声明 hash 不一致。

错误信息包含具体字段路径，例如 `semantic_config.discount_gamma`、`model.decoder_heads`，而不只显示 hash mismatch。

### warm_restart

缺 replay、optimizer、sampler、RNG 等 continuation state 时分类为 warm restart，不称 exact。

## Source fingerprint 覆盖

按组件保存文件级 path/hash/bytes：

- `guandan_rules`：cards、combos、round、state、action_state、encoding、environment、env_batch；
- `game_adapter`：V2 game、env adapter、buffer；
- `model`：Agent、CentralCritic、Transformer、cache、tensordict、checkpointable/trainable/types；
- `optimization`：AdamW grouping、EMA、Muon；
- `runtime`：config、sampler、compile、logger、utils；
- `training`：checkpoint、GAE、loss、schedule、IPPO/VRPO/local runtime、validation、metrics、smoke profile。

不包含 runs、日志、checkpoint、模型输出、临时目录或只读参考 checkout。临时 source fixture 修改测试证明只改变对应 component 和 aggregate fingerprint。

## 旧 checkpoint 边界

- 原始 M03 `guandan-marvel-training-checkpoint-v2` 和报告完整保留；
- 原始 tag 仍指向 `4968840679ea031ec7b69132cebd862efa302c17`；
- v2 缺少 semantic-config/source/execution fingerprint，训练恢复分类为 `migration`；
- v2 可用于显式 evaluation/artifact inspection，但不能补填当前配置后称 exact；
- 新 exact/compatible 验收只使用 v3 checkpoint。

## 新 IPPO/VRPO smoke

继续使用预算内 CPU profile：rollout/collect/replay=1、16/24 小模型、test-only `T=512`、AdamW、Muon/compile/AMP=false。

### IPPO

- 2 real terminal games；
- 2 durable iterations；
- 4 learner updates；
- trajectory lengths 334、311；
- exact v3 resume；
- env/game steps 645/293；
- nonfinite/illegal/truncated 皆为 0。

### VRPO

- 2 real terminal games；
- 2 durable iterations；
- 8 learner updates；
- trajectory lengths 304、330；
- actor/critic/optimizer/RNG/sampler/replay exact v3 resume；
- env/game steps 634/266；
- nonfinite/illegal/truncated 皆为 0。

这些仅是本地训练语义证据，不是模型强度或性能结论。

## 测试结果

- M01/M02/M03 scoped：81 passed，0 skipped，exit 0，114.137s；
- interruption/semantic repair regressions：12 passed，0 skipped，exit 0，33.561s；
- A01-A04 protected acceptance：152 passed，0 skipped，exit 0，16.507s；
- legacy unit/integration：889 passed，10 skipped，exit 0，229.625s；
- new v3 IPPO/VRPO smoke：exit 0；
- v3 checkpoint artifact verifier：exit 0；
- clean wheel import：exit 0；
- `git diff --check`：exit 0。

10 个 skip 仍为既有 CUDA/symlink 环境限制，本轮新增 skip 为 0。

## 机器证据

- `review_findings.json`
- `protocol_schema.json`
- `repair_verification.json`
- `training_smoke.json`
- `checkpoint_verification.json`
- `source_fingerprint.json`
- `test_results.json`
- `source_manifest.json`

## 受保护范围

未修改 `guandan/cards.py`、`combos.py`、`round.py`、`state.py`、`docs/rules.md`。`D:\project\doudizhu` 保持原 HEAD 和 clean。

## 当前结论

补修代码和本地证据已通过，但第一笔补修 commit SHA 尚未登记。因此 `STATUS.yaml` 保持 M03 `accepted=false`、M04 blocked。第二笔提交只固定补修实现 SHA、证据 hash 和重新验收状态，不执行 M04。
