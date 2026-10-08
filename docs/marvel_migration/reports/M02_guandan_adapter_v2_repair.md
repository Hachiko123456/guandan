# M02 Guandan MARVEL V2 Adapter 补修与重新验收报告

- 日期：2026-10-08
- 分支：`codex/marvel-guandan-migration`
- M01 原始实现：`d4638ab4eb7886335c8f438deadcaff12066f1bf`
- M02 原始实现：`71500c5ac580cfda935d7bbd52de6a3793110c60`
- 补修实现 commit：`9f481f3c68b1e9dcc6bc66cdcfd6775811379600`
- 状态：`accepted`
- 环境：Windows 11；Python 3.12.0；PyTorch 2.13.0+cpu；NumPy 2.3.5；CUDA unavailable

## 范围

本轮只补修 M02 sampler/trajectory 正确性、复杂动作实际执行覆盖、证据身份和 M03 内存预算。未开始 M03 learner，未修改 A01/A02 规则文件，未写 C++，未运行 Kaggle。

## 审查问题复现

### 1. 行为概率与训练参考概率不一致：已复现

在原提交 `71500c5` 上使用非 one-hot 分布和 `sampling_temperature=0`：greedy 实际行为概率为 1.0，trajectory `avail_probs` 对已选动作记录约 0.2。根因是 sampler 用温度/forced 逻辑选动作，却把原始模型概率作为 PPO old probability。

修复：

- sampler contract 升级为 `guandan-marvel-sampler-contract-v2`；
- trajectory 改为 `policy_probs`、`behavior_action_prob`、`on_policy_valid`；
- `purpose=on_policy` 只允许 temperature=1，禁止 forced action；
- evaluation/test 可 greedy、0.5/2.0 温度或 forced legal action，但 `on_policy_valid=false`；
- PPO 用 `behavior_action_prob` 算 ratio，用 `policy_probs` 算 reference KL，并拒绝非 on-policy 数据；
- buffer schema 升级为 `guandan-marvel-buffer-v2`，v1 snapshot 显式不兼容。

验证中 on-policy 已选动作 policy/behavior 均为 0.20000000298；evaluation 的行为概率分别为 1.0、0.2842105329、0.1553921103；forced test trajectory 被 PPO 入口拒绝。

### 2. 失败 step 计数/cache 不一致：已复现

原实现的三 token horizon 复现：失败前 `(env=2, game=0, buffer=2)`，失败后 `(env=3, game=1, buffer=2)`，环境序列化不变，异常为 `TrajectoryHorizonError`。

修复采用方案 B `poison_on_failure`：

- completed counters 仅在 environment step、buffer write、completed commit 全部成功后增加；
- attempted/failed/rejected 使用独立诊断计数；
- agent/cache 执行后的任何异常将 sampler 标为不可继续；
- 后续 rollout/collect/cache prepare 抛 `SamplerFailedStateError`；
- failure info 明确 `exact_resume_allowed=false`；
- forced on-policy/forced shape/legal preflight 在 cache 前拒绝，不 poison sampler。

真实小型 Transformer cache 测试显示失败前 cache length 2、失败后 3，证明 cache 已推进；completed counters/buffer 保持 2/0/2，sampler 随即 poison 并拒绝重试。另覆盖多 slot forged action 原子失败和环境成功后的 buffer write injection。

同时修复了复查测试暴露的真实 Agent 直接 `rollout_step()` inference-tensor 更新问题：该入口现在显式运行在 `torch.inference_mode()`。

### 3. 复杂牌型实际执行缺口：已复现并补齐

原 15 局扫描复测保持：15 真终局、4447 steps、2809 commits、23121 candidate rows；candidate 枚举看见 9 类 family，但实际完整出牌仅 `SINGLE=810`，因此原扫描只作为候选/终局/守恒证据。

新增 13 个定向执行案例，从当前 `CombinationKind` enum 覆盖全部十类：single、pair、triple、full_house、straight、pair_sequence、triple_sequence、rank_bomb、straight_flush、four_kings。每类均从空 prefix 执行到 COMMIT；额外覆盖：

- natural wild；
- substituted wild；
- two wildcard assignments；
- natural+substituted mixed；
- 最长 15-token prefix；
- 复杂公开 commit event；
- prefix 中途序列化恢复；
- 独立 V1 oracle transition/reward/commit 比较；
- 108 张牌守恒。

定向案例保留一张额外 actor card，故用于复杂动作 commit 而非终局声明；真实 terminal 仍由独立 15 局扫描证明。

新增两环境不同时间终局、auto reset、连续两次 collect 测试，以及真实小型 Transformer Agent + test-only forced legal action 的完整真终局接口测试。后者 `on_policy_valid=false`，不作为自主策略或训练数据。

### 4. M03 dense buffer 内存风险：已复核

按修复后实际字段：

- 单 trajectory：91,381,768 bytes = 87.148445 MiB；
- global rollout=512、collect=2048：仅 rollout+commit = 261.445335 GiB；
- 两 rank 均分：130.722668 GiB/rank；
- 不含 model、optimizer、KV cache、activations、replay 和临时复制。

新增纯 shape/dtype `trajectory_memory_budget()` 和 fail-before-allocation `require_trajectory_memory_budget()`。M03 Prompt/训练规范要求启动前打印预算并按 per-rank limit 拒绝。

推荐本地 smoke：rollout=1、collect=1、replay scale=1、mini-batches=1、16/24 小模型、显式 test-only T=512；sampler 约 32.681 MiB。T=512 结果不得冒充默认 T=4096 验证。

## 修改文件

主要代码：

- `guandan_marvel/runtime/sampler.py`
- `guandan_marvel/buffer.py`
- `guandan_marvel/env_batch.py`
- `guandan_marvel/training/losses.py`
- `guandan_marvel/training/gae.py`
- `guandan_marvel/testing/complex_actions.py`

测试/证据工具：

- `tests/marvel_adapter/test_sampler_contract.py`
- `tests/marvel_adapter/test_complex_actions.py`
- `tests/marvel_adapter/test_memory_budget.py`
- `scripts/verify_marvel_sampler_contract.py`
- `scripts/verify_marvel_adapter_v2_complex_actions.py`
- `scripts/report_marvel_m03_memory_budget.py`
- `scripts/verify_marvel_adapter_v2.py`

规范/身份：

- `M02_ADAPTER_PROTOCOL.md`
- `05_TRAINING_SPEC.md`
- `09_CHECKPOINT_COMPATIBILITY.md`
- `prompts/M03_marvel_training_smoke.md`
- M01/M02 原报告中的 `self` 身份固定为准确 SHA。

## 实际测试结果

- Scoped core+adapter：62 passed，0 skipped，exit 0，65.280s。
- Sampler probability/failure：17 passed，0 skipped，exit 0，7.909s。
- Complex actions/lifecycle：16 passed，0 skipped，exit 0，22.527s。
- A01-A04 protected acceptance：152 passed，0 skipped，exit 0，12.969s。
- Legacy unit+integration：889 passed，10 skipped，exit 0，223.160s。

10 个 skip 均为原有环境能力：2 estimator CUDA、4 rollout GPU equivalence、1 training objectives CUDA、1 training support CUDA、1 Windows file symlink、1 Windows directory symlink。补修没有新增 skip。

完整命令、JUnit、stdout hash 和 skip 来源见 `evidence/M02_guandan_adapter_v2_repair/test_results.json`。大型日志位于 `runs/M02_guandan_adapter_v2_repair/20261008_repair_validation/`，不提交 Git。

## 机器证据

- `review_findings.json`
- `protocol_schema.json`
- `sampler_contract.json`
- `original_scan_rerun.json`
- `complex_action_coverage.json`
- `memory_budget.json`
- `test_results.json`
- `source_manifest.json`
- `artifact_manifest.json`

## 受保护范围

`guandan/cards.py`、`combos.py`、`round.py`、`state.py`、`docs/rules.md` 与 M02 原始提交对象 hash 相同。参考仓库 HEAD 仍为 `acdcf199cf422e2b683cb22142b8e7cb33b14460` 且 clean。

## 已知限制

- 仍是 `python_reference`，不是性能证据。
- M03 learner/checkpoint smoke 尚未开始。
- directed complex cases 不是完整局；与 15 局真终局证据分开。
- 默认 dense buffer 仍很大；本轮只增加预算/preflight，没有实施紧凑存储、裁剪、分桶或 spill。
- poisoned sampler 必须重建，当前不提供昂贵的全状态 rollback。

## 当前结论

补修实现提交已固定为 `9f481f3c68b1e9dcc6bc66cdcfd6775811379600`，所有硬门槛和证据 hash 已重新核验。`STATUS.yaml` 重新接受 M02 并解除 M03 blocker；本轮没有执行 M03。验收登记提交自身 SHA 不写入自身内容，准确值由 Git 和最终答复给出。
