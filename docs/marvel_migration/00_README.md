# Guandan MARVEL Migration 控制入口

- 项目：Guandan MARVEL Migration
- 建立日期：2026-10-08
- 工作仓库：`D:\project\guandan`
- 只读参考仓库：`D:\project\doudizhu`
- Kaggle 交付根目录：`D:\project\kaggle_train`
- 迁移分支：`codex/marvel-guandan-migration`
- 迁移状态唯一来源：`D:\project\guandan\docs\marvel_migration\STATUS.yaml`

## 目标

把 `D:\project\doudizhu` 中与游戏无关的 MARVEL 训练、运行时、模型、IPPO、VRPO、分布式、检查点和评估组织方式迁移到掼蛋项目；保留掼蛋规则语义，以新的版本化适配层和高性能 C++ 环境替换当前训练热路径。

本迁移不把斗地主规则复制成掼蛋规则，也不把已有烟雾训练称为强模型训练。

## 必读顺序

每个阶段开始前必须依次阅读：

1. 本文件；
2. `01_MASTER_MIGRATION_PLAN.md`；
3. `STATUS.yaml`；
4. `08_ACCEPTANCE_MATRIX.md`；
5. 当前阶段对应 Prompt；
6. 前一阶段的机器可读报告和提交差异。

需要修改接口、训练或评估时，再阅读对应专项文档：

- `02_SOURCE_MAPPING.md`
- `03_TARGET_ARCHITECTURE.md`
- `04_GAME_INTERFACE_V2.md`
- `M04_CPP_RUNTIME_PROTOCOL.md`
- `M04_CPP_RUNTIME_REPAIR_PROTOCOL.md`
- `M04_CPP_INTEGER_BOUNDS_REPAIR_PROTOCOL.md`
- `M05_DIFFERENTIAL_PERFORMANCE_PROTOCOL.md`
- `M05_DIFFERENTIAL_PERFORMANCE_REPAIR_PROTOCOL.md`
- `M05_PERFORMANCE_FINISH_PROTOCOL.md`
- `M05_ACCEPTANCE_CRITERIA_REVISION.md`
- `M06_EVALUATION_PROTOCOL.md`
- `M06_EVALUATION_REPAIR_PROTOCOL.md`
- `M06_FINAL_DATASET_BINDING_REPAIR_PROTOCOL.md`
- `M06_SOURCE_AUDIT.md`
- `05_TRAINING_SPEC.md`
- `06_EVALUATION_CHAMPION_SPEC.md`
- `07_KAGGLE_T4X2_RUNBOOK.md`
- `09_CHECKPOINT_COMPATIBILITY.md`
- `M07_KAGGLE_DDP_PROTOCOL.md`
- `M07_KAGGLE_DDP_REPAIR_PROTOCOL.md`
- `M07_KAGGLE_INPUT_LAYOUT_REPAIR_PROTOCOL.md`
- `M07_SOURCE_AUDIT.md`
- `10_RISK_AND_DECISIONS.md`

## 迁移边界

### 冻结为规则真值

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `guandan/state.py`
- `docs/rules.md`

M01-M09 不得主动改变上述文件的规则语义。若发现可复现规则缺陷，当前阶段必须停止扩散修改，先提交缺陷报告、最小复现和影响范围，由主审查决定是否建立独立修复阶段。

### 保留但允许增加 V2 接口

- A03 canonical 动作与 COMMIT 语义；
- A04 信息隔离和隐私边界；
- 当前 V1 序列化、检查点只读评估能力；
- 当前 Python 环境作为 reference oracle。

### 允许重建

- 模型结构；
- rollout/trajectory 数据组织；
- IPPO/VRPO 训练运行时；
- C++/pybind 高性能环境；
- 训练指标；
- champion/opponent-pool；
- T4×2 DDP 运行方式。

## 执行规则

1. 阶段严格串行；不得同时让两个对话修改同一工作区。
2. 新对话先检查 Git 状态、分支、HEAD 和前一阶段状态。
3. 前一阶段不是 `accepted` 时，不得实现下一阶段。
4. `D:\project\doudizhu` 始终只读。
5. 旧训练器和旧 checkpoint 不删除；迁移初期只做并行兼容。
6. 不以计划计数、启动日志或部分输出冒充实际完成。
7. smoke/local verification 不构成强度结论。
8. 本地测试不构成 Kaggle T4×2 证据。
9. 每阶段必须保存报告、测试命令、环境信息和 Git commit。
10. 只有 `STATUS.yaml` 记录迁移阶段状态，其他文档不得维护第二份动态状态。

## 阶段列表

- M00：控制文档与执行 Prompt
- M01：MARVEL 通用核心迁移
- M02：掼蛋 V2 游戏适配器
- M03：MARVEL IPPO/VRPO 本地训练冒烟
- M04：C++/pybind 掼蛋批量环境
- M05：Python/C++ 差分与性能门槛
- M06：评估、champion 与 opponent pool
- M07：Kaggle T4×2 DDP 交付
- M08：真实 Kaggle pilot 与参数定型
- M09：正式训练、晋升与最终验收
- M10：可选的 Yomi 信念与 Kagami 搜索

## 如何启动新对话

复制 `prompts\Mxx_*.md` 的全部内容到新对话。不要只复制其中一部分。每个阶段完成后，先人工检查：

- 修改文件；
- 测试是否真实执行；
- 是否有跳过或弱化验证；
- 阶段报告；
- Git commit；
- `STATUS.yaml`；
- blockers。

确认该阶段 `accepted` 后，再启动下一阶段。
