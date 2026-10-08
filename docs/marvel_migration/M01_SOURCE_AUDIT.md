# M01 MARVEL 通用核心来源审计

- 审计日期：2026-10-08
- 只读参考仓库：`D:\project\doudizhu`
- 参考 commit：`acdcf199cf422e2b683cb22142b8e7cb33b14460`
- 参考 commit subject：`Isolate Dou Dizhu routines`
- 参考许可证：MIT，Copyright (c) 2026 AtaraxosAI
- 目标包：`guandan_marvel`
- 目标版本：`0.1.0`

## 分类结论

| 领域 | 参考模块 | M01 分类 | 处理 |
|---|---|---|---|
| 配置 | `marvel/config.py`, `game_identity.py` | 半通用/游戏耦合 | 重写为版本化 `GameSpec` 注入；删除 `nanopokerlib` 解析和 `doudizhu` 默认值 |
| Transformer/KV cache | `model/transformer.py`, `model/cache.py` | 通用 | 迁移，只有命名空间/来源头调整 |
| TensorDict | `model/tensordict.py` | 通用 | 迁移并加强 batch shape、device、赋值校验 |
| 模型基类/Agent/Critic | `model/base.py`, `agent.py`, `critic.py`, `types.py`, `legacy.py` | 半通用 | 迁移；只依赖 `GameSpec` 容量，不包含规则实现 |
| 优化器 | `optim/common.py`, `optim/muon.py` | 通用 | 迁移；Muon 保留为显式 opt-in，默认 AdamW |
| Buffer | `buffer.py` | 半通用 | 重写为 `GameSpec` 驱动，增加 overflow hard fail 和 versioned state round trip |
| Checkpoint | `runtime/checkpoints.py`, `model/checkpointable.py` | 通用 | 重写本地原子 I/O；强制 `guandan-marvel-checkpoint-v1`；不静默加载无版本 payload |
| Compile | `runtime/compilation.py` | 通用 | 迁移；默认关闭，请求不可用 backend 时失败而非 fallback |
| Distributed | `runtime/distributed.py` | 通用 | 重写显式 DDP gate；不完整 torchrun/CUDA 环境失败关闭 |
| Logging | `runtime/logger.py`, `intervals.py` | 通用 | 迁移 JSONL/CSV serializer/sink 和 interval；W&B 生命周期延后 |
| Training utilities | `training/losses.py`, `gae.py`, `trainer.py` | 通用/半通用 | 迁移 loss、GAE；从 trainer 提取纯 schedule 工具，不迁移训练 workflow |
| Env/sampler | `env_batch.py`, `runtime/sampler.py`, `rollout_conversion.py` | 游戏接口相关 | M01 不迁移；等待 M02 V2 adapter |
| IPPO/VRPO/workflow | `training/ippo.py`, `vrpo.py`, `workflow.py`, `runtime/orchestra.py` | 依赖 env/sampler | 延后 M03；避免未接环境的伪训练路径 |
| Evaluation | `eval/hooks.py` | 评估专用 | 延后 M06 |
| CLI | `train.py` | 完整训练入口 | 延后 M03 |
| 斗地主专用实现 | `nanopokerlib/cpp/games/doudizhu/*`, `evaluations/doudizhu/*` | 游戏专用 | 禁止迁移 |

## 文件级来源映射

“轻改”表示许可证头、命名空间和小型兼容调整；“中改”表示增加校验或替换局部接口；“重写”表示保留职责/契约思想但重新实现。

| 目标文件 | 参考来源 | 改写程度 | 保留的游戏假设 | M02 待适配接口 |
|---|---|---:|---|---|
| `guandan_marvel/config.py` | `marvel/config.py`, `game_identity.py` | 重写 | 固定玩家数、token/channel/capacity 由数据描述 | 产出真实 `GameSpec` 和协议版本 |
| `guandan_marvel/utils.py` | `marvel/utils.py` | 中改 | snake/UPPER attribute 兼容 | 无规则语义；adapter 可复用 seed/commit count |
| `guandan_marvel/model/transformer.py` | `marvel/model/transformer.py` | 轻改 | causal history sequence | 确认 V2 horizon/position 上界 |
| `guandan_marvel/model/cache.py` | `marvel/model/cache.py` | 轻改 | 固定最大 horizon 和 flattened player batch | M02 rollout shape 必须满足 cache contract |
| `guandan_marvel/model/tensordict.py` | `marvel/model/tensordict.py` | 中改 | 无 | adapter 生成字段需满足 batch/device 校验 |
| `guandan_marvel/model/checkpointable.py` | 同名参考文件 | 轻改 | 无 | M02/M03 添加协议/semantic hash 到 manager config |
| `guandan_marvel/model/trainable.py` | 同名参考文件 | 轻改 | 无 | 无 |
| `guandan_marvel/model/legacy.py` | 同名参考文件 | 轻改 | 只保留 MARVEL v1 模型加载结构，不含 legacy Guandan MLP | 无；legacy Guandan 仍由原包评估 |
| `guandan_marvel/model/base.py` | 同名参考文件 | 中改 | observation 为 `[B,T,P,L]`；candidate 为 `[B,T,A,L]` | 定义真实 obs/action token/channel 含义 |
| `guandan_marvel/model/agent.py` | 同名参考文件 | 中改 | 每步一个 acting player；环境步可用 `player_id=-1` | M02 映射逐 token/COMMIT 与环境控制步骤 |
| `guandan_marvel/model/critic.py` | 同名参考文件 | 中改 | centralized critic 可见 `[P]` 训练态 | M02 分离 actor information set 与 critic full state |
| `guandan_marvel/model/types.py` | 同名参考文件 | 轻改 | 输出 shape contract | M02/M03 对齐 sampler 字段 |
| `guandan_marvel/optim/common.py` | 同名参考文件 | 轻改 | 无 | 无 |
| `guandan_marvel/optim/muon.py` | 同名参考文件 | 轻改 | 无 | M08 才决定是否启用 |
| `guandan_marvel/buffer.py` | `marvel/buffer.py` | 重写 | complete trajectory tensor layout | M02 供应 V2 capacities；M03 决定 replay persistence policy |
| `guandan_marvel/runtime/checkpoints.py` | 同名参考文件 | 重写 | flat state binding | M02/M03 添加 action/env/encoding/critic contract metadata |
| `guandan_marvel/runtime/compilation.py` | 同名参考文件 | 中改 | 无 | M03 用真实动态 shape 测试决定开关 |
| `guandan_marvel/runtime/distributed.py` | 同名参考文件 | 重写 | 正式多 rank 目标为 CUDA DDP | M07/M08 验证 T4×2，不在 M01 声明性能 |
| `guandan_marvel/runtime/intervals.py` | 同名参考文件 | 轻改 | 无 | 无 |
| `guandan_marvel/runtime/logger.py` | 同名参考文件 | 重写/缩减 | 无 | M03 扩展必需训练指标；M06 接评估指标 |
| `guandan_marvel/training/losses.py` | 同名参考文件 | 轻改 | rollout field names/masks | M03 连接真实 trajectory |
| `guandan_marvel/training/gae.py` | 同名参考文件 | 轻改 | multi-player rewards/player ids | M02/M03 验证掼蛋终局奖励/环境步语义 |
| `guandan_marvel/training/schedules.py` | `training/trainer.py` | 重写提取 | iteration-based decay | M03 连接 trainer |
| `guandan_marvel/testing/fake_game.py` | 新写 | 新写 | 仅测试 shape 的四玩家假数据 | **不得**作为掼蛋 adapter 或训练 backend |
| `guandan_marvel/*/__init__.py` | 参考 export surface + 新命名空间 | 重写 | 无 | 后续只追加版本化 API |
| `guandan_marvel/LICENSE`, `NOTICE.md` | 参考 `LICENSE` + 新说明 | 复制/新写 | 无 | 保持随包分发 |

## 清理的硬编码依赖

- 运行时代码无 `nanopokerlib` import。
- 运行时代码无 `doudizhu` 游戏默认值、地主/农民角色、叫地主、底牌或斗地主牌型。
- 运行时代码无 `D:\project\doudizhu` 绝对路径依赖；该路径只出现在 provenance 文档。
- 顶级包不使用 `marvel` 名称，不会覆盖同环境中的参考安装。
- fake game 只位于 `guandan_marvel.testing`，identifier 明确为 `fake-core-test-only`。

## 未迁移原因摘要

M01 不迁移环境 batch、sampler、完整 workflow、IPPO/VRPO trainer、evaluation hook、train CLI，原因是这些模块仍依赖具体 environment wire、动作解码、终局/奖励、评估 routine 或正式训练生命周期。机械复制会形成斗地主假设或无环境可执行的伪训练路径，分别交由 M02、M03、M06 处理。
