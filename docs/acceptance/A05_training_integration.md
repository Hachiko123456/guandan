# A05 训练集成

## 范围

把 A04 环境接入 IPPO/VRPO 训练流水线。本阶段同时支持本机快速 profile 和 Kaggle 完整 profile。

## Profile

配置文件：`configs/acceptance_profiles.json`。

### `local_fast`

用于本机接线和正确性验证：

- IPPO、VRPO 各 5 次更新；
- `rollout_envs=4`；
- 每次采集 256 个 rollout steps；
- 每个算法至少保存一次检查点，并恢复后继续更新；
- CPU 必须通过；本机存在 CUDA 时额外运行一次 CUDA smoke；
- 不用于判断模型能力。

### `remote_full`

用于 Kaggle/远程 GPU：

- IPPO、VRPO 各至少 100 次更新；
- `rollout_steps=1024`，并行环境数量由远端资源决定；
- 定期保存检查点；
- 运行更长的稳定性测试和恢复训练；
- 记录实际 GPU、Python、PyTorch、profile 和 Git commit。

## 最小正确性门槛

无论使用哪个 profile，都必须验证：

- 损失和梯度为有限值；
- 合法动作概率归一化；
- 非法动作概率为零；
- token step 与 committed step 的 GAE/掩码语义一致；
- checkpoint 保存、加载和恢复后可以继续更新；
- 规则、动作、环境和编码版本不兼容时拒绝加载。

## 命令

```powershell
python scripts/run_acceptance.py --stage A05 --profile local_fast
python scripts/run_acceptance.py --stage A05 --profile remote_full
```

自动报告不会设置监督层的 `accepted: true`。主代理必须检查差异、运行 profile 对应测试、运行完整测试并更新 `STATUS.yaml`。
