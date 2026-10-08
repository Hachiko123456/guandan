# 目标架构

## 1. 分层图

```text
现有 Python A01/A02/A03/A04
  ├─ 规则规范与 reference oracle
  ├─ legacy checkpoint 只读评估
  └─ V2 Python adapter（M02/M03）
                │
                ├──────── 差分验证 ────────┐
                ▼                           │
     nanopokerlib.guandan C++ EnvBatch       │
                │                           │
                └───────────────────────────┘
                │
                ▼
        MARVEL Complete-Trajectory Sampler
                │
        ┌───────┴────────┐
        ▼                ▼
   MARVEL IPPO       MARVEL VRPO
 Agent+Value Head   Agent+Central Q Critic
                         + Replay/Aux Epochs
        └───────┬────────┘
                ▼
      Selection / Champion / Opponent Pool
                ▼
          Kaggle T4×2 DDP
                ▼
       Final Holdout / Release Artifact
```

## 2. 运行时模块

目标运行时应具有：

- versioned game module；
- C++ `EnvBatch`；
- `EnvStepBatch` TensorDict 转换；
- complete trajectory buffer；
- commit buffer；
- Transformer KV cache；
- DDP；
- `torch.compile` 可开关；
- checkpoint/resume；
- JSONL 指标；
- 控制台进度；
- 系统遥测；
- evaluation hook。

## 3. 模型基线

第一版忠实实现 MARVEL v1.3 风格：

```text
history_dim=128
feature_dim=256
decoder_layers=4
decoder_heads=4
state_layers=3
action_layers=2
fuse_layers=2
```

Actor 只能使用各座位自己的信息集。IPPO value head 使用相同 actor 信息。VRPO centralized critic 可以使用训练期完整状态，但其输出不能回流为 actor 输入。

## 4. 游戏后端切换

M02/M03 使用 Python adapter，目的仅为验证接口和训练语义。M04/M05 完成后，正式训练默认切换至 C++。保留显式 backend 选项：

```text
backend=python_reference
backend=cpp
```

不允许 silent fallback。请求 `cpp` 而扩展不可用时必须失败。

## 5. 兼容性

- V1 环境、模型、checkpoint 保持只读兼容；
- V2 使用新协议和新 checkpoint schema；
- V1/V2 评估报告必须明确标注版本；
- 不允许用同一个 `best.pt` 文件名覆盖不同架构；
- champion manifest 必须包含模型、协议、源码、配置和评估数据身份。
