# 启动 A05：训练集成

以下 Prompt 可复制到新的 Codex 对话中，用于继续实施并本机运行 A05。它不是 A05 的验收报告，也不会自动上传 Kaggle。

```text
开始执行 D:\project\guandan 的 A05 训练集成。

先读取 project_status/STATUS.yaml、docs/MASTER_EXECUTION_PLAN.md、docs/AGENT_EXECUTION_PROTOCOL.md、docs/acceptance/A05_training_integration.md、configs/acceptance_profiles.json、当前 Git diff 和最近提交。A00-A04 不得重新实现或重新修改。

本机使用 C:\Users\yhx\.conda\envs\guandan_train\python.exe，profile 使用 local_fast。不要上传 Kaggle，不要使用 remote_full，不要修改 D:\project\doudizhu 或 D:\project\FableDan。

先检查当前 A05 实现和测试：
- PolicyValueNet 的输入/输出和合法 token mask；
- rollout/buffer；
- IPPO/VRPO 更新语义；
- token_step、committed_step、truncation、reset、GAE 和 terminal reward；
- checkpoint 保存、加载和 resume；
- actor/critic 的隐私边界和 player/team perspective。

主代理必须使用子代理分离实现，并在每个子代理完成后检查 diff、运行 scoped tests、运行 scripts/run_acceptance.py --stage A05 --profile local_fast，并运行目标依赖的全量测试。不要把 mean-pooled MLP 称为原始 MARVEL Transformer；VRPO 不能只是改名后的 PPO。

local_fast 的正式目标是：IPPO、VRPO 各 5 次更新；4 个环境；每次更新跨环境聚合 256 个 token steps（每环境 64 ticks）；每算法总计 1280 个 token steps；每 5 次更新保存 checkpoint；恢复后追加 1 次更新；总预算 0.5 小时。必须核对实际计数，不得只记录计划值。

本轮先完成 local_fast 并记录 local_ready；local_ready 不等于 accepted。只有完整测试、主代理审查、Git 提交和后续 remote_full 证据都满足后，才能考虑 accepted=true。

运行命令：
C:\Users\yhx\.conda\envs\guandan_train\python.exe -m guandan.train --algorithm ippo --profile local_fast
C:\Users\yhx\.conda\envs\guandan_train\python.exe -m guandan.train --algorithm vrpo --profile local_fast
C:\Users\yhx\.conda\envs\guandan_train\python.exe scripts\run_acceptance.py --stage A05 --profile local_fast

完成后报告：阶段、profile、每个算法实际更新数、实际 token steps、checkpoint 路径、resume 结果、loss/gradient、测试命令、报告路径、Git commit、未完成项。
```

## 当前明确限制

- 当前 A05 训练器是接线/冒烟实现，不代表模型已经达到可用强度。
- remote_full 必须在 Kaggle 或明确配置的远端 GPU 环境执行。
- 未完成 A06 评估之前，不得宣称模型能力。
