# Checkpoint 兼容性和迁移规则

## 1. Legacy V1

特征：

```text
GD-ACTION-0.1
GD-ENV-0.1
mean-pooled-token-MLP
hidden_dim 通常为 32
固定 token-transition collector
```

允许：

- 原训练器 resume；
- V1 evaluator 评估；
- 作为 legacy baseline/opponent；
- 生成历史行为数据。

禁止：

- 加载为 V2 Transformer resume；
- 把部分匹配权重称为继续训练；
- 用同名文件覆盖后声称协议兼容。

## 2. MARVEL V2 IPPO

至少保存：

- Agent resume state；
- merged/EMA Agent state；
- value head；
- optimizer；
- schedules；
- iteration/counters；
- config/protocol/source hashes；
- RNG；
- selection/champion 身份。

## 3. MARVEL V2 VRPO

在 IPPO 字段之外保存：

- CentralCritic；
- critic optimizer；
- replay buffer policy 和必要状态；
- actor/critic 独立 schedule；
- Q target/critic contract 版本。

若 replay buffer 不持久化，resume 报告必须明确 warm-restart 语义，不能称 exact resume。

## 4. Resume 分类

- `exact_resume`：语义、实现、RNG、采样顺序和必要 buffer 全部一致；
- `compatible_resume`：语义一致，允许设备/日志/非语义实现变化，需 continuation verifier；
- `warm_restart`：只恢复模型/optimizer 或缺 replay；
- `migration`：架构/协议变化，需要新训练，不称 resume；
- `evaluation_only`：仅加载推理权重。

## 5. 文件命名

文件名必须带架构和算法信息，例如：

```text
guandan-marvel-v2-ippo-iter000500.pt
guandan-marvel-v2-vrpo-iter000500.pt
guandan-legacy-v1-ippo-update1702.pt
```

`best.pt` 只能是 manifest 指向的副本/链接概念，不得作为唯一身份。

## 6. Champion manifest

至少包括：

- checkpoint filename/hash；
- algorithm；
- architecture；
- protocol versions；
- source commit；
- resolved config hash；
- training counters；
- selection deals hash；
- promotion evaluation hash；
- promotion timestamp；
- previous champion。
## 7. M02 sampler/buffer schema 修复

M02 补修将 trajectory probability 字段从含义模糊的 `avail_probs` 拆为：

- `policy_probs`；
- `behavior_action_prob`；
- `on_policy_valid`。

buffer state schema 从 `guandan-marvel-buffer-v1` 升级为 `guandan-marvel-buffer-v2`。v1 buffer snapshot 必须显式拒绝，不能作为 v2 exact/compatible resume。M02 未产生正式训练 checkpoint，因此该变更分类为 `migration`，不是丢失正式 resume 能力。M03 新 checkpoint 的 semantic contract hash 必须包含 sampler contract、buffer schema 和 on-policy purpose/temperature。
## 8. M03 training checkpoint v2

M03 canonical schema：`guandan-marvel-training-checkpoint-v2`。至少保存：

- algorithm、architecture、完整 game/sampler/buffer/Q-target semantic contract hash；
- resolved config/hash、source hash 和 source parent commit；
- resume actor、merged/current-copy actor、IPPO value head；
- VRPO critic；
- actor/critic optimizer；
- schedule、iteration、learner/durable counters；
- Python/NumPy/Torch/CUDA RNG；
- serialized sampler/environment/partial rollout state；
- VRPO replay state（exact resume）；
- selection/champion identity（M03 为 null，不伪造）；
- merged policy identity，beta=0 明确为 `current_policy_copy`。

文件名带算法和 iteration：`guandan-marvel-v2-{ippo|vrpo}-iterNNNNNN.pt`。M03 continuation verifier 已验证 exact resume 后模型权重、采样计数和下一次更新一致。
## 9. M03 repair checkpoint v3（2026-10-09）

Canonical training schema upgrades to `guandan-marvel-training-checkpoint-v3`; semantic contract upgrades to `guandan-marvel-training-contract-v2`. New fields include hashed semantic configuration, component-level source fingerprint, execution/dependency fingerprint, durable learner boundary and explicit iteration lifecycle.

- pre-repair v2: training resume classification is `migration`; explicit evaluation/artifact inspection remains allowed;
- exact: semantic/source/execution fingerprints and all required runtime state match;
- compatible: only approved logging/display/run-control fields differ;
- warm restart: continuation state such as replay/optimizer/sampler is missing;
- migration/reject: gamma/model heads/optimizer/schedule/sampling/replay/source/device/dependency semantics differ or declared hashes are inconsistent.

A trainer whose load fails is poisoned and cannot train or save. A catchable interruption during an iteration poisons trainer+sampler; only the last complete checkpoint may be loaded into a new instance.

## 10. M05 backend/build identity repair（2026-10-09）

Canonical training schema upgrades to `guandan-marvel-training-checkpoint-v4`; semantic contract upgrades to `guandan-marvel-training-contract-v3`. Every training checkpoint records the actual runtime `backend` and a hash-protected `guandan-marvel-backend-identity-v1` object.

For `python_reference`, identity and source fingerprint are generated without importing the native extension. For `cpp`, identity includes the loaded extension path, binary SHA-256, compiler/C++ standard/Python ABI, compile-time embedded native-source digest and, when source files are available, a comparison against the current native-source manifest. Wrapper, native source/header, build identity or binary changes are not exact-compatible.

`python_reference` and `cpp` checkpoints are never automatically cross-resumed. A v3 checkpoint that lacks backend/build identity is `migration` for training or `evaluation_only` under an explicit evaluation load; current backend data must not be injected to relabel it exact. Logging/run-directory-only changes remain compatible only when backend, source, binary, semantic and execution identities otherwise match. Any failed load continues to poison trainer and sampler.
