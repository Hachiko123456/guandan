# M03 MARVEL Training 来源审计

- 日期：2026-10-08
- 目标开始 commit：`6997cf8526f2852115f2df06ed279f1846682138`
- 只读参考：`D:\project\doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460`
- 参考许可证：MIT

## 文件映射

| 目标文件 | 参考来源 | 改写程度 | 说明 |
|---|---|---:|---|
| `training/ippo.py` | `marvel/training/ippo.py` | facade | canonical M03 IPPO entry point |
| `training/vrpo.py` | `marvel/training/vrpo.py` | facade | canonical M03 VRPO entry point |
| `training/local.py` | `ippo.py`, `vrpo.py`, `trainer.py`, `workflow.py` | 重写/合并 | 本地 fail-closed runtime；去除 evaluator/DDP/Kaggle 假设 |
| `training/checkpoints.py` | `runtime/checkpoints.py`, 本仓库 legacy compatibility 思路 | 重写 | V2 schema、semantic hash、resume 分类、RNG/sampler/replay |
| `training/validation.py` | 新写 | 新写 | 完整终局、on-policy、finite、actor privacy、replay gate |
| `training/metrics.py` | `runtime/orchestra.py` rollout logging | 重写 | 05 spec 本地 JSONL metrics |
| `training/smoke.py` | 新写 | 新写 | 显式 `T=512` Python reference smoke game identity |
| `scripts/run_marvel_training_smoke.py` | 新写 | 新写 | bounded IPPO/VRPO smoke + exact resume |
| `scripts/verify_marvel_training_artifacts.py` | 新写 | 新写 | checkpoint/schema/hash/classification verification |

继续使用 M01 已迁移并注明来源的 Agent、CentralCritic、GAE、loss、optimizer、sampler、buffer、logger 和 Transformer 模块。

## 未迁移范围

- reference evaluator / selection / champion：M06；
- DDP orchestra/Kaggle：M07；
- C++ game runtime：M04；
- 长时间训练与正式模型参数：M08/M09。

## 清理的参考假设

- 不解析 `nanopokerlib.doudizhu`；game module 由 Guandan Python adapter 注入；
- 不启用 reference W&B/evaluator；本地只写 JSONL；
- 不使用斗地主 role/landlord metrics；
- 不沿用 512/2048 dense 默认分配；
- 不允许 temperature/forced evaluation trajectory 进入 learner。

参考仓库只读，运行时代码不依赖其绝对路径。
