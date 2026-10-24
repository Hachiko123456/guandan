# 阶段 4A 定向修复 Prompt：交付包接入 v1.4 Actor Fuse、AMP 与真实 M09 Warm Start

> 本 Prompt 不重新实施阶段 4 的奖励、52-context、curriculum 或评估语义。下文 blocker 是修复前的历史证据，不代表当前实现仍未修复。2026-10-19 的设备接线及交付集成已由 Codex 接续完成，下一步是使用 `/review-phase` 对现有实现进行全新只读复核，不能原样重跑 `/phase` 或恢复旧暂停任务。全新 Reviewer PASS 前不得进入阶段 5/6或正式训练。

## 0. 当前入口与约束（优先于历史描述）

1. 先完整读取本文件和 `docs/marvel_v14_prompts/04A_GPU_runtime_addendum.md`，两者共同构成验收规范。
2. GPU 环境固定为 `D:/project/guandan/.venv-gpu/Scripts/python.exe`，子进程使用 `sys.executable`。不得改全局 Python、Guance、驱动、OpenCode 配置或重建 native 扩展。
3. 完整证据索引：`runs/gpu_runtime_completion/IMPLEMENTATION_REPORT.md`；运行记录和最终源码冻结证据同目录保存。报告是待核实声明，不代替 Reviewer 自行核验。
4. 当前任务只核验 04A，不重新实施 04、不进入 05/06、不启动长训/Kaggle/final holdout。先确认没有并发源码写入。
5. 本地 probe 优先 `--device cuda`、单卡 batch=1、1 iteration；不可用或 OOM 必须明确失败，不能静默回落 CPU。有意运行 CPU 路径时明确 `--device cpu`，保留原 BF16 精度门禁。
6. Reviewer 必须在临时目录运行有写入的 probe，禁止改正式交付、manifest、checkpoint 或已有证据；可通过 pytest 的 tmp_path/独立临时目录产生测试产物。不得重建交付。
7. 同一最终源码上的全量只运行一次。Reviewer 核对冻结前后源码 hash 和完整全量日志，亲跑设备/迁移/AMP/篡改反例，不重复整套长耗时 smoke/recovery/全量。

## 一、修复前已确认 blocker（历史复现依据）

工作区：

```text
D:/project/guandan
```

阶段 4 主 Prompt：

```text
docs/marvel_v14_prompts/04_reward_context_delivery.md
SHA256=7640840ee34337aa0e7c529d5fd661e248a56bcfa9a44058a34444d444075050
```

当前阶段 4 candidate tracked diff：

```text
bytes=210323
sha256=57dcf1750c0d3ee2a084891debcade0d63b8def7da20455dff42c68dbd3bf275
```

以上是修复前基线，不要求当前文件回到旧 SHA256。当前源码以本次冻结证据为准；tracked diff 不包含新增未跟踪文件，必须同时核对源码逐文件 hash。执行前重新核对当前状态，仅在发现并发写入时停止，禁止为匹配旧 hash 回滚。

### Blocker A：交付配置绕过阶段 3

当前实际 start/resume plan与dataset profile包含：

```jsonc
{
  "amp_enabled": false,
  "model": {
    "architecture": "v1.3",
    "history_dim": 128,
    "feature_dim": 256,
    "decoder_layers": 4,
    "state_layers": 3,
    "action_layers": 2,
    "fuse_layers": 2
  }
}
```

这意味着正式 Kaggle训练不会使用：

- `v1.4-macro-fuse` Actor Residual Fuse；
- CUDA fp16 autocast；
- GradScaler。

阶段 3 已验收通过，阶段 4 的正式交付不得无说明回退到 v1.3/AMP off。

### Blocker B：随包 warm-start 与目标模型形状不兼容

当前：

```text
runs/marvel_v14_phase4/delivery/datasets/start/warm_start_policy.pt
bytes=158028
source architecture=v1.3
source feature_dim=24
source history_dim=16
source state_dict keys=33
actor_fuse keys=0
```

而当前交付目标profile是生产尺寸：

```text
feature_dim=256
history_dim=128
state_dict keys=78（v1.3）
```

只读实测：

```text
shape_mismatch_count=31
model.load_state_dict(..., strict=False) -> RuntimeError
```

首批不匹配包括：

```text
state_tokens_embed.weight: [320,16] -> [320,128]
action_tokens_embed.weight: [320,24] -> [320,256]
state_mlp.0.weight: [24,528] -> [256,640]
```

因此当前start Dataset虽然hash自洽，但在worker调用 `load_warm_start_policy` 时会失败，不能开始训练。

### Blocker C：验证器只检查文件/hash，没有验证真实启动链

当前 `verify_marvel_v14_phase4_delivery.py` 能返回 `valid=true`，但没有：

1. 从交付dataset profile真实构建 `TrainConfig`；
2. 构造目标 `IPPOTrainer`；
3. 加载随包warm-start；
4. 验证v1.3→v1.4 migration；
5. 运行至少1 iteration。

必须补上该门禁，确保“交付包自洽”同时意味着“交付包可启动”。

---

# 二、必须达到的最终交付语义

## 2.1 目标模型

start profile必须声明：

```jsonc
{
  "amp_enabled": true,
  "model": {
    "architecture": "v1.4-macro-fuse",
    "history_dim": 128,
    "feature_dim": 256,
    "decoder_layers": 4,
    "decoder_heads": 4,
    "state_layers": 3,
    "action_layers": 2,
    "fuse_layers": 2,
    "ffn_ratio": 2
  }
}
```

并保持：

```text
training_semantics_profile=macro_decision_v1
policy_regularizer_mode=macro_normalized_kl_uniform_v1
reward_semantics=guandan_rank_points
reward_scale=1/3
52-context 7A schedule
warmup=20
blend=80
target_iterations=1500
max_wall_seconds=12600
world_size=2
backend=nccl
required_gpu=T4 x2
```

不得改变已验收的 macro PPO/GAE、Fuse、AMP、reward或context数学。

## 2.2 真实 warm-start

交付builder默认必须使用与生产尺寸匹配的正式M09 v1.3 policy-only checkpoint，例如：

```text
runs/comparison_delivery/dataset/m09_candidate_iter002000.pt
```

或仓库中经阶段1冻结、与其SHA256相同的正式policy checkpoint。

必须验证源checkpoint：

```text
architecture=v1.3
feature_dim=256
history_dim=128
parameter_count=2,061,314
state_dict keys=78
iteration=2000（或冻结manifest声明值）
```

目标：

```text
architecture=v1.4-macro-fuse
parameter_count=2,783,234
state_dict keys=90
```

调用真实：

```python
trainer.load_warm_start_policy(payload)
```

必须返回：

```text
source_architecture=v1.3
target_architecture=v1.4-macro-fuse
fuse_migrated=true
output_proj_zero_verified=true
optimizer_reset=true
sampler_reset=true
counters_reset=true
```

并验证fixed probe在optimizer step前：

```text
old/new logits torch.equal
old/new probs torch.equal
old/new argmax torch.equal
actor_fuse.output_proj bitwise zero
```

禁止继续使用：

```text
runs/marvel_v14_phase3/ab-smoke/synthetic-v13-warm-start.pt
```

作为正式delivery默认输入。该文件只允许用于单元测试和小模型smoke。

---

# 三、允许修改范围

按需最小修改：

```text
guandan_marvel/kaggle/phase4_delivery.py
scripts/build_marvel_v14_phase4_delivery.py
scripts/verify_marvel_v14_phase4_delivery.py
tests/marvel_training/test_phase4_delivery.py
tests/marvel_training/test_phase4_probe_devices.py
tests/marvel_training/test_macro_fuse_amp.py
```

GPU 接线不得改全局 `build_train_config` 的生产行为；仅在本地 probe 复制配置、声明单 rank 并显式覆盖 device。Prompt、索引和 Claude 使用说明可同步更新，OpenCode 配置不得改动。

如果真实启动门禁需要专用helper，可新增：

```text
tests/marvel_training/_phase4_delivery_runtime_probe.py
```

或 `guandan_marvel/kaggle/phase4_delivery.py` 内职责明确的验证函数。

如需让标准配置文件与实际交付保持一致，可最小修改：

```text
configs/marvel_kaggle_profiles.json
```

但不得破坏既有M09历史profile；优先新增/更新明确的v14 Phase4 profile，而不是覆盖历史M09语义。

禁止修改：

- reward公式和scale；
- context schedule/curriculum数学；
- 7A/7B判定；
- macro PPO/GAE；
- Actor Fuse数学；
- AMP核心实现；
- checkpoint fail-closed规则；
-阶段1 manifest/report/probe和Prompt；
- belief、opponent pool、search；
- FableDan代码或数据。

禁止 `git reset/clean/checkout/restore/commit/push`。

---

# 四、Builder与Verifier精确要求

## 4.1 Builder

`build_marvel_v14_phase4_delivery.py`：

1. 默认warm-start改为正式M09生产checkpoint；
2. 不允许默认回落到synthetic checkpoint；
3. 在复制/打包前验证checkpoint尺寸、architecture、state_dict和SHA256；
4. 构造v1.4+AMP start semantic contract；
5. 构建start/resume使用同一source package identity；
6. 重建plan、dataset manifests、notebook、README、delivery manifest和upload ZIP。

## 4.2 Verifier

`verify_marvel_v14_phase4_delivery.py` 和生产验证函数必须新增fail-closed检查：

- plan/profile architecture必须为 `v1.4-macro-fuse`；
- `amp_enabled`必须为true；
- start warm-start必须存在且SHA匹配；
- source checkpoint生产尺寸正确；
- 从profile真实调用 `build_train_config`；
- target模型参数量和key集合正确；
- 真实调用 `load_warm_start_policy`成功；
- migration报告符合预期；
- output_proj严格全零；
- 对一个固定probe确认迁移前后policy逐位一致；
- 至少完成1个本地单设备 iteration 且 loss finite；本机优先 CUDA FP16+GradScaler，显式 CPU 时使用 BF16；
- 不仅记录 learner_updates，还必须验证参数实际变化，全部 AMP skip 不能冒充训练通过；
- 同设备重载必须逐位恢复模型、optimizer、counters 和 GradScaler；仅允许原有 strict resume 规则接受的 run-control 差异；
- CPU/GPU migration probe 都在同设备 FP32 下逐位比较，且显式关闭外层 autocast；
- checkpoint/recovery语义记录目标architecture和amp=true；
- resume profile不得覆盖checkpoint中的训练语义。

验证必须能拒绝：

1. synthetic小模型warm-start；
2. feature/history dim不匹配；
3. architecture仍v1.3；
4. amp=false；
5. v1.4 warm-start缺actor_fuse key却声称same-architecture；
6. 篡改warm-start hash；
7. start/resume semantic contract不一致。

---

# 五、必须新增测试

至少新增：

1. `test_phase4_delivery_targets_v14_fuse_with_amp`；
2. `test_phase4_delivery_default_warm_start_is_production_m09`；
3. `test_phase4_packaged_warm_start_loads_into_real_profile`；
4. `test_phase4_packaged_warm_start_rejects_synthetic_small_checkpoint`；
5. `test_phase4_runtime_probe_is_bitwise_equal_before_first_step`；
6. `test_phase4_resume_inherits_v14_amp_semantics`；
7. `test_phase4_verifier_rejects_v13_or_amp_off_plan`；
8. `test_phase4_verifier_rejects_shape_mismatch_even_when_hashes_match`。

旧实现必须至少失败于第1、3、4、7、8项。

---

# 六、本地验证与证据复用

所有命令在项目根目录执行。不要把裸 `python`/`pip` 当成 GPU 环境；以下是 PowerShell 语法。

## Tier 1：设备接线与真实交付

```powershell
& D:/project/guandan/.venv-gpu/Scripts/python.exe -m pytest tests/marvel_training/test_phase4_probe_devices.py tests/marvel_training/test_phase4_delivery.py -q -p no:cacheprovider
```

## Tier 2：最相关的专项

```powershell
& D:/project/guandan/.venv-gpu/Scripts/python.exe -m pytest tests/marvel_training/test_phase4_reward_and_context.py tests/marvel_training/test_macro_fuse.py tests/marvel_training/test_macro_fuse_amp.py tests/marvel_training/test_macro_fuse_checkpoints.py tests/marvel_training/test_macro_fuse_norm_precision.py -q -p no:cacheprovider
```

CPU BF16 测试显式选 CPU；CUDA 专项有 GPU 时真实运行、无 GPU 时明确 skip。不删除原测试，不降低容差。

## Runtime start probe

使用交付目录中真实 profile 和 M09 warm-start，核对 source package identity，完成：

```text
build TrainConfig（production v1.4 model，local single rank / batch=1 / horizon=512）
→ construct IPPOTrainer on selected device
→ load_warm_start_policy
→ same-device FP32 bitwise equality and zero fuse gate
→ 1 CUDA FP16 iteration with effective optimizer update
→ durable checkpoint
→ reload model/optimizer/counters/scaler on the same device
→ validate resume profile inheritance
```

本地 CUDA 初始 scale=1024 只用于短 probe 并必须记录；正式 trainer/交付保留生产默认 scale=65536。不能把本机 512-step 单卡 checkpoint 当作 4096-step 双 T4 的正式 recovery，也不能声称 NCCL 已验收。

## 重建交付（仅 Implementer；Reviewer 禁止）

```powershell
& D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/build_marvel_v14_phase4_delivery.py --repository D:/project/guandan --output D:/project/guandan/runs/marvel_v14_phase4/delivery
& D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/verify_marvel_v14_phase4_delivery.py --delivery D:/project/guandan/runs/marvel_v14_phase4/delivery --device cuda --runtime-iterations 1 --runtime-max-game-horizon 512 --runtime-init-scale 1024 --runtime-work-dir D:/project/guandan/runs/gpu_runtime_completion/packaged_gpu_probe --output D:/project/guandan/runs/marvel_v14_phase4/delivery_verification.json
```

`--no-runtime` 只说明结构校验，不能替代 runtime PASS。源码冻结后重建交付、核对源文件与 ZIP 内容一致，再运行最终 GPU probe，报告所有新 SHA256。

## 全量回归

仅最终源码和交付物冻结后执行一次：

```powershell
& D:/project/guandan/.venv-gpu/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --import-mode=importlib
```

保存完整日志、exit code、耗时、前后 tracked diff bytes/hash 和所有 tracked/untracked 源码逐文件 hash。允许保留已知 Guance dirty-tree failure，但不得新增失败；有遗留失败时不得声称“全量全绿”。Reviewer 在 hash 未变时复核这份证据，不再重复整套全量；源码变化时旧证据失效。

---

# 七、Guance已知问题

当前 `test_guance_declaration_handoff.py` 在dirty tree下会被 `_assert_authority_clean`拒绝；绕过该守卫后，旧handoff authority blob也不匹配已修改的 `guandan/round.py`。这是阶段4前已存在的跨工作流债务，但当前reward语义修改使其继续过期。

本修复不得通过削弱测试或绕过fail-closed来制造PASS。报告中继续明确列为独立待处理项；如用户未授权重建guance handoff，不把它混入本次delivery修复。

---

# 八、Reviewer要求

使用全新的只读Reviewer。Reviewer必须亲自：

1. 解包start dataset并读取真实profile；
2. 检查architecture=v1.4、amp=true；
3. 检查warm-start为生产M09尺寸而非synthetic；
4. 调用真实`load_warm_start_policy`；
5. 验证actor_fuse零门与bitwise policy等价；
6. 使用 GPU 环境在 pytest 临时目录运行至少 1 iter runtime probe，验证 CUDA/FP16/scaler/有效参数更新；不得写正式交付或复用旧 CPU checkpoint 冒充 GPU exact resume；
7. 篡改architecture、amp、shape和hash验证fail-closed；
8. 验证resume继承v1.4/AMP语义；
9. 验证delivery ZIP/manifest/hash；
10. 检查 reward/context/fuse/AMP 核心数学未被本次接线改动；核对 OpenCode 配置身份未变。
11. 至少三类新的独立反例，并标记 `REVIEWER_RUN / EXISTING_REPORT / NOT_RUN`。
12. Windows 本机单卡不能证明 Linux 双 T4/NCCL 通过；该项必须保留 NOT RUN。

最终必须输出：

```text
VERDICT: PASS / FAIL / INCONCLUSIVE
NEXT_STAGE: ALLOW / DENY
```

---

# 九、硬验收标准

1. 正式start profile使用 `v1.4-macro-fuse`。
2. 正式start profile `amp_enabled=true`。
3. 随包warm-start为生产M09尺寸并可真实加载。
4. v1.3→v1.4 migration报告和零门正确。
5. 固定probe迁移前后policy逐位一致。
6. 交付 runtime probe 至少 1 iter，通过明确 CUDA 设备、FP16/scaler、有效参数更新与同设备恢复门禁。
7. durable checkpoint记录v1.4/AMP语义。
8. resume严格继承且不能覆盖语义。
9. verifier拒绝旧v1.3/AMP-off/小模型/shape mismatch/tamper包。
10. start/resume source package identity一致。
11. delivery verifier valid=true且新增runtime gate通过。
12. Phase4 reward/context专项继续通过。
13. Phase3 fuse/AMP/checkpoint专项继续通过。
14. 无新增全量失败。
15. 未启动长训、Kaggle或final holdout。
16. 未实施阶段5/6。

---

# 十、报告格式

必须包含：

1. `IMPLEMENTATION_STATUS`；
2. 开工diff与并发检查；
3. 原delivery不可启动的独立复现；
4. 修改文件；
5. 最终profile完整模型/AMP语义；
6. warm-start source身份、尺寸、参数量、keys和SHA256；
7. migration结果与bitwise probe；
8. runtime 1-iter和checkpoint结果；
9. start/resume semantic contract；
10. tamper/fail-closed测试；
11. Tier1/Tier2/全量结果；
12. delivery所有hash；
13. 最终diff bytes/SHA256；
14. guance独立遗留项；
15. residual risks；
16. Python/torch/CUDA/GPU、实际 device/dtype/scaler、skips、参数更新、显存与耗时；
17. 源码逐文件冻结证据及其与 ZIP 的一致性；
18. 明确声明未长训、未实施阶段 5/6、双 T4/NCCL NOT RUN。

若验收仅受已知 Guance 遗留失败影响，必须单独说明：04A 集成门禁与全量遗留项分别判定，禁止通过绕过 Guance 守卫伪造全绿。通过独立 Reviewer 只表示 04A 通过，不表示用户已授权下一阶段或正式长训。

完成后停止，等待全新Reviewer。