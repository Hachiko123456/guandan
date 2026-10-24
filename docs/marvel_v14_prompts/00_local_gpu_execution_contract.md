# 后续阶段本地 GPU 执行与证据契约（05A / 06A 共用）

更新日期：2026-10-19。本文是 05A、06A 的必读组成部分。它规定执行约束，不启动任何任务。

## 1. 已完成范围与历史文件

核心阶段 1、2（含 2A）、3、4（含 04A）已有验收。04A 的独立 Reviewer 为 PASS / ALLOW；当前最终源码上的全量结果为 1796 passed / 1 failed / 2 skipped，唯一 failure 是已知 Guance authority dirty-tree/handoff。原始 Reviewer 输出与 Codex 复查证据在：

```text
runs/gpu_runtime_completion/review_audit_20261019/
```

这是历史验收基线，不是给未来变动沿用的测试成绩。实际开工时重新读取文件、计算 hash、检查并发，不靠 mtime 或 tracked diff 单独证明工作区未变。

原 `05_symmetry_and_belief_conditional.md`、`06_opponent_pool_search_final_conditional.md` 和原 `README.md` 属于阶段 1 冻结输入，不得覆盖。新阶段规范以 05A / 06A 为准；这些名称是修订入口，不代表多出两个开发阶段。

## 2. GPU 环境与强制设备门禁

只使用此独立 Python：

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe
```

GPU 训练/模型验收的命令必须显式指定 `--device cuda`。新入口的模型执行设备默认 cuda，不把 auto 作为训练验收设备，也不以裸 `python` 或全局 CPU torch 运行模型训练。子进程必须使用 `sys.executable`，并将设备和预算参数显式传下去。

开工先运行：

```powershell
& D:/project/guandan/.venv-gpu/Scripts/python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is required; no CPU fallback'; print(torch.cuda.get_device_name(0))"
```

本机验收环境已实测 RTX 3050 Ti Laptop，4GB；每次执行仍报告真实环境。严禁修改全局 Python、Guance、NVIDIA 驱动、OpenCode 配置，或无约束升级依赖。

### 设备分工

| 工作 | 设备要求 |
|---|---|
| 策略/value/可训练 probe/Belief 的真实 forward、backward、optimizer step | CUDA；有效 AMP 路径为 FP16 autocast + GradScaler |
| 冻结神经策略/对手的真实模型推理 | CUDA；与 CPU 小型参考测试明确区分 |
| CPU C++ 规则引擎、合法动作、环境推进、样本索引、文件/JSON/hash | CPU 正常运行；不声称整个程序都在 GPU |
| 公开信息解析基线、bootstrap/聚类统计、无神经运算的计数 oracle | CPU 可用，不属于把训练退回 CPU |
| 纯数学单元测试、CPU BF16 精度兼容、CPU 双 rank gloo | 保留显式 CPU；不得整套 pytest 强行切 CUDA |
| GPU 缺失时的单元测试 | GPU 专项可 skip，但本阶段 GPU 验收 NOT RUN，不能整体 PASS |

GPU 不可用、显存不足、非有限值等必须明确失败并留证，不准自动重试到 CPU。可以显式减小 microbatch、梯度累积或本地 horizon，但先记录适配，不能改变采样数据、数学、精度容差或正式配置以制造通过。

## 3. 小显存与运行预算

- 先 batch=1 的真实 GPU smoke，至少一个有效 optimizer update；确认参数实际变化，不能仅观察 learner_updates 字段。
- 不把 04A 的 horizon=512、single-rank checkpoint 当作正式 4096-step/T4x2 recovery。
- 本地 CUDA smoke 可显式使用 init_scale=1024 并写报告；生产 trainer 默认初始 scale=65536 保持原样。不得把修改全局 scaler 当作修复。
- 神经 probe 的训练也在 GPU。数据应分块送入 GPU，禁止一次缓存全部长轨迹或把多个生产模型与全部 buffer 常驻 4GB 显存。
- 05A / 06A 各自的 pilot、采集和小规模实验必须有 max_games、max_updates、max_wall_seconds；超过预算则在一致性边界保存可复核进度，报告 INCONCLUSIVE 或相应条件分支 NOT RUN，不自动加预算或改用 CPU。
- 本协议只允许阶段 Prompt 指定的本地有界测试。50/250/500/1000/1500 iter 正式训练 Gate、Kaggle、双 T4 作业、selection/final holdout 和 FableDan 正式评估均不得自动启动。

## 4. 记录真实设备、精度和恢复语义

每个模型训练/验收结果至少包含：

```text
python_executable / torch_version / cuda_build / gpu_name
requested_device / selected_device / parameter_device
autocast_dtype / scaler_enabled / initial_scale / final_scale / skipped_steps
optimizer_updates / parameter_changed / finite_loss / finite_grad
peak_allocated_bytes / peak_reserved_bytes / rollout_seconds / learner_seconds
seed / checkpoint_sha256 / config_hash / source_hash / dataset_split_hash
all_local_adaptations / budget / completed_work / unexecuted_items
```

迁移等价性在同一设备、同一 FP32 下比较，显式关闭 ambient autocast；不要求 CPU 与 CUDA 输出逐位一致。同配置、同设备 resume 必须恢复模型、optimizer、scaler、RNG、sampler 和本阶段新增状态；新的 feature/schema/source 变化必须使用显式版本化迁移或 policy-only warm start，不能伪造 exact_resume。保留 strict/fail-closed 规则。

## 5. 测试证据与全量预算

1. 先针对性反例与最小 GPU smoke，修复后再扩大测试；不得已知 blocker 存在时反复跑长测试。
2. 每个最终源码版本最多完整跑一轮全量，不跳过 GPU 专项，不删测试、不加 skip/xfail/宽容差制造通过。
3. 全量命令使用 GPU Python，具体测试内部仍按设备分工：

```powershell
& D:/project/guandan/.venv-gpu/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --import-mode=importlib
```

4. Implementer 保留完整 stdout/stderr、真实 pytest 退出码、耗时、测试前后 tracked diff 与所有 tracked/untracked 生产/测试源码 hash；不要用 `| tail` 丢弃全量日志，不把 pipeline 最后一个命令的 exit code 当 pytest 的 exit code。
5. Reviewer 必须独立运行新的决定性反例和 GPU 路径；可在精确源码 hash 一致时复核最终全量日志，不再为了“更放心”重跑整套；若变动仅为说明文档，逐项解释差异和受影响定向复测，不冒称旧清单 hash 等于新清单。
6. Reviewer 遵守只读 hook；允许 pytest 的临时产物，禁止重建正式交付、改源码或覆盖已有证据。协调器可将 Reviewer 的完整工具输出存入忽略的 `runs/phase5/` 或 `runs/phase6/`，不得要求 Reviewer 绕过 hook。
7. `runs/` 是忽略证据目录，不属于当前 git-ls-files 源码指纹；保存证据后仍要复核指纹，不能声称“任意新增报告必然改变源码”。
8. 已知 Guance failure 单列，不绕过守卫、不动其 handoff；任何新增 failure 必须查明，不默认为遗留。

## 6. 阶段 1 冻结证据与源码变更

不得修改阶段 1 manifest/report/probe、历史冻结 Prompt 或削弱 frozen-input 检查。新阶段需要新文件时使用新路径；若确实必须修改被冻结的生产输入，先列出路径与原因，设计单独的后续阶段 provenance/验收证据，不靠扩大旧 whitelist 或重写历史 manifest 通过。不能处理时先停止该变更并报告，不扩大修复范围。

05A/06A 改到训练语义、模型、checkpoint、sampler 或评估后，04A ZIP 就是历史交付。保留旧交付，后续如需新交付，在新目录重建、版本化、核对 archive 与源码，不能沿用旧 source hash 声称新功能已交付。未实际重建则明确 DELIVERY_PENDING，不上传或启动远程训练。

## 7. 条件分支与独立验收

条件特性被科学否决不等于工程失败。允许：工程 PASS + Belief/Search DISABLED，只要否决证据、默认关闭、legacy/恢复/设备/回归验证完整。

数据不足、预算耗尽、统计定义不充分则必须 INCONCLUSIVE，不能偷换为 gate 通过或科学否决。只有全新 Reviewer `VERDICT: PASS` 且 `NEXT_STAGE: ALLOW` 才允许用户选择下一阶段；不自动继续。
