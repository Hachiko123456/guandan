# 斗地主参考源码映射与来源规则

## 1. 来源

- 参考仓库：`D:\project\doudizhu`
- 审计基线 commit：`acdcf19`（完整 SHA 必须由 M01 重新读取并写入阶段报告）
- 参考仓库许可证：MIT；M01 必须读取原 LICENSE，并在迁移文件中保留适当的来源和许可证说明。
- 参考仓库只读，任何阶段不得在其中写入、格式化、安装生成物或测试缓存。

## 2. 迁移分类

### A. 通用代码：优先迁移/泛化

- `marvel/model/transformer.py`
- `marvel/model/cache.py`
- `marvel/model/tensordict.py`
- `marvel/model/trainable.py`
- `marvel/model/checkpointable.py`
- `marvel/runtime/checkpoints.py`
- `marvel/runtime/compilation.py`
- `marvel/runtime/distributed.py`
- `marvel/runtime/intervals.py`
- `marvel/runtime/logger.py`
- `marvel/runtime/orchestra.py`
- `marvel/optim/*`
- `marvel/training/losses.py`
- `marvel/training/trainer.py`

迁移时要删除硬编码斗地主假设，并通过 game interface 获取玩家数、token 数、channels、horizon 和候选动作容量。

### B. 半通用代码：迁移后必须适配掼蛋

- `marvel/model/base.py`
- `marvel/model/agent.py`
- `marvel/model/critic.py`
- `marvel/buffer.py`
- `marvel/env_batch.py`
- `marvel/runtime/sampler.py`
- `marvel/training/gae.py`
- `marvel/training/ippo.py`
- `marvel/training/vrpo.py`
- `marvel/training/workflow.py`
- `marvel/eval/hooks.py`

重点适配：四人固定队伍、终局奖励、逐 token/COMMIT、贡还牌阶段、四玩家信息集和掼蛋候选动作结构。

### C. 游戏专用代码：不得把斗地主实现改名后使用

- `nanopokerlib/cpp/games/doudizhu/*`
- `evaluations/doudizhu/*`
- `kagami/games/doudizhu/*`
- 斗地主 role/landlord/peasants 指标
- 斗地主牌型、叫地主、底牌和炸弹倍数规则

这些部分必须按掼蛋规则重新实现。

### D. 后续阶段

- `yomi/*`：M10 参考；
- `kagami/search/*`：M10 参考；
- 外部 PerfectDou/DouZero/FPDou 连接器：不迁移为掼蛋基线。

## 3. 包命名决策

M01 必须评估以下两个方案并形成 ADR：

1. 保留顶级包名 `marvel`，但要求隔离环境；
2. 使用 `guandan_marvel` 或其他独立命名空间，避免与本地 `D:\project\doudizhu` 安装冲突。

在 ADR 通过前不得大规模机械替换 import。选择标准：Kaggle 打包、可复现、最小维护成本、与参考源码差异大小、同一 Python 环境下的冲突风险。

## 4. 不盲目复制的默认值

以下值只能作为实验基线，不能被描述为已验证最优：

- `rollout_batch_size=512`
- `collect_batch_size=2048`
- `total_iterations=2000`
- `agent_merging_beta=0.0`
- `evaluation_interval=0`
- `use_muon=True`
- `use_compile=True`
- replay buffer scale 64

M08 必须用真实 T4×2 pilot 决定最终资源配置。

## 5. 证据要求

M01 的阶段报告必须包含：

- 参考 commit 完整 SHA；
- 读取到的 LICENSE 摘要；
- 复制/改写/新写文件列表；
- 每个迁移文件的来源路径；
- 游戏专用依赖清理结果；
- 未迁移文件及原因。

## 6. M01 已解析映射（2026-10-08）

- 审计使用完整参考 commit：`acdcf199cf422e2b683cb22142b8e7cb33b14460`。
- D005/D006 已接受：目标包为 `guandan_marvel`，以许可证保留的仓库内派生子包交付。
- 完整逐文件来源、改写程度、残留假设和 M02 接口见 `M01_SOURCE_AUDIT.md`。
- `env_batch`、sampler、workflow、IPPO/VRPO trainer、evaluation hook 和 train CLI 未机械复制，分别等待 M02/M03/M06。
- 参考 MIT LICENSE 随包分发于 `guandan_marvel/LICENSE`，来源说明位于 `guandan_marvel/NOTICE.md`。
