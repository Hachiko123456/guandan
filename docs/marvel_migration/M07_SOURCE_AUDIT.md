# M07 来源审计

- 日期：2026-10-10
- 参考仓库：`D:\project\doudizhu`（只读）
- 参考 commit：`acdcf199cf422e2b683cb22142b8e7cb33b14460`
- 许可证：MIT，随 `guandan_marvel/LICENSE` 和 `guandan_marvel/NOTICE.md` 分发

## 映射

| 目标 | 参考来源 | 分类 | 说明 |
|---|---|---|---|
| `guandan_marvel/runtime/distributed.py` | `marvel/runtime/distributed.py` | 延续迁移并加固 | 保留 DDP/reduce 结构；增加显式 DDPConfig、CPU/gloo 仅测试路径、无 silent fallback |
| `guandan_marvel/training/local.py` | `marvel/training/{trainer,workflow,ippo,vrpo}.py` | 掼蛋重写 | 在既有 M03 fail-closed trainer 上接入单模型 DDP、global/local batch、rank seed 和 rank-local exact state |
| `guandan_marvel/kaggle/checkpoints.py` | `marvel/runtime/{checkpoints,orchestra}.py` | 新写/概念参考 | DDP rank shard、原子 envelope、聚合 provenance、recovery binary bundle；不是源码复制 |
| `guandan_marvel/kaggle/runner.py` | `marvel/runtime/orchestra.py`, `marvel/train.py` | 新写/概念参考 | torchrun 入口、preflight、telemetry、partial bundle 和 mode orchestration |
| `guandan_marvel/kaggle/worker.py` | `marvel/training/workflow.py` | 新写/概念参考 | 两 rank 同一模型训练、global metric 汇总、checkpoint/evaluation 周期 |
| `guandan_marvel/kaggle/{discovery,package,telemetry,evaluation,profiles}.py` | 无直接文件复制 | 新写 | Kaggle fail-closed discovery、source ZIP/Notebook、系统遥测和有限评估 |

## 未迁移内容

- 未复制斗地主规则、landlord/peasants 指标或 evaluator agent；
- 未复制参考仓库未版本化的 Kaggle 路径假设；
- 未采用“两张卡各跑一个独立算法”的旧交付形态；
- 未引入网络上传或 Kaggle API 凭据处理。

## 第三方离线构建依赖

生成器将当前 `pybind11` Python/header 包和其 LICENSE 逐文件 hash 后放入 source ZIP，用于 Kaggle 离线 `setup.py build_ext --inplace`。该依赖不声称属于目标 Git commit；source manifest 的 `origin=bundled_pybind11` 与版本字段明确区分。
