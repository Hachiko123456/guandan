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
## 6. M04 实现状态（2026-10-09）

`nanopokerlib.guandan` 提供显式 `backend=cpp` 的 pybind `EnvBatch`，并保持与 M02 相同的 tensor shape/protocol version。原 M05 已完成 10,000 完整局和 100,000 中间状态且 0 mismatch；补修保留该证据。

公共导入契约使用延迟加载：`guandan_marvel.default_game` / `get_game_backend()` 取得默认 C++，`guandan_marvel.cpp_game` 显式取得 C++，`guandan_marvel.python_game` 显式取得 Python oracle；历史 `guandan_marvel.game` 与直接导入同名模块均固定为 Python oracle。配置和模型基础模块不强制加载扩展，显式 C++ 缺失时 fail closed。

性能补测发现真实 frozen Transformer rollout 未达到保守的 2× end-to-end 门槛，因此当前 M05 补修未重新 accepted，M06 保持 blocked。
