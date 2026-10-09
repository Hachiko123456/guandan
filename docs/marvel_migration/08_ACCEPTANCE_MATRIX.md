# 迁移验收矩阵

## 通用门槛

每阶段都必须：

- 前一阶段 `accepted`；
- Git 工作区在开始时可解释；
- scoped tests 真实执行；
- 目标依赖回归执行；
- 无新增 silent fallback；
- 阶段报告和机器可读证据；
- `git diff --check`；
- 一个原子 commit；
- `STATUS.yaml` 更新与证据一致。

## 阶段门槛

| 阶段 | 功能门槛 | 测试/证据 | 性能/远端 | 不得宣称 |
|---|---|---|---|---|
| M00 | 控制文档、状态、模板、Prompt 完整 | 链接、路径、YAML/JSON 模板检查 | 无 | 训练系统已迁移 |
| M01 | 通用 MARVEL 核心可导入，来源清楚 | 通用模块单元测试；旧套件不回退 | 无 | 掼蛋已接入 MARVEL |
| M02 | Python V2 adapter 完整终局 | V1/V2 动作等价；四玩家隐私；序列化 | 仅功能吞吐 | 模型变强 |
| M03 | MARVEL IPPO/VRPO 可更新/恢复 | GAE、loss、critic、replay、checkpoint、非有限值检查 | 本地 smoke | 可实战、GPU 打满 |
| M04 | C++ EnvBatch 与 pybind 可运行 | 牌型/动作/终局专项测试 | 初步 benchmark | 已替代 Python oracle |
| M05 | Python/C++ 语义一致 | 10k 完整对局；100k 中间状态；13 级牌；隐私、恢复与身份安全 | fixed-action single-env 与 batch 同机均 >=2×；真实 Transformer rollout/完整 iteration 为公开的非阻塞优化目标 | 未实测倍数、Kaggle 性能或模型强度 |
| M06 | selection/champion/pool 完整 | 固定 deals、四换座、clustered CI、晋升回滚 | 评估吞吐 | final holdout 结论 |
| M07 | 无需手改的 Kaggle DDP 包 | 本地 package/preflight/recovery 测试 | 仅 remote_ready | T4×2 verified |
| M08 | 真实 Kaggle pilot 完成 | 下载日志、checkpoint、manifest 审查 | 双卡目标 >=1.5×；显存/ETA 可接受 | 正式强度 |
| M09 | 正式训练和 final holdout | champion 历史、选择证据、最终报告 | 长跑完成/可恢复 | 超出评估范围的实战保证 |
| M10 | belief/search 合法且不泄露 | 世界采样约束、搜索预算、回退 | 推理延迟 | 用搜索掩盖基础策略错误 |

## M05 详细正确性建议

- 10,000 个完整随机终局：0 diff、0 illegal、0 conservation failure；
- 100,000 个采样中间状态：合法候选集合/commit 标志完全一致；
- 全 13 级牌；
- 自然/逢人牌声明；
- 炸弹排序；
- 贡、还贡、抗贡；
- 接风；
- 终局名次、结果类别和奖励；
- 四座位观测隐私；
- 序列化/恢复后的后续动作一致。

规模若因资源分阶段执行，只有全部完成后 M05 才能 `accepted`。

### M05 性能门槛修订（2026-10-10）

`M05_ACCEPTANCE_CRITERIA_REVISION.md` 将 fixed-action single-env 和 batch `>=2×` 保留为环境性能硬门槛；本机 CPU 真实 Transformer rollout、额外集合和完整 IPPO iteration 改为必须公开的非阻塞优化目标。正确性、隐私、合法动作、牌守恒、serialization/resume、checkpoint/backend identity 和 no-silent-fallback 门槛不变。Kaggle T4×2 实际吞吐与双卡效率仍由 M08 验证。旧报告中的 Transformer `<2×` 失败结论不得删除或改写。

## M08/M09 远端证据

远端状态必须保留：

- Kaggle notebook/kernel 标识（可获得时）；
- 启动和结束绝对时间；
- GPU 型号与数量；
- source/config/checkpoint hash；
- 实际 iteration/games/steps；
- 双 rank 日志；
- telemetry；
- durable checkpoint；
- evaluation summary；
- 下载后完整性校验。
