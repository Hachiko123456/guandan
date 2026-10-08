# M02 Python Adapter 来源审计

- 日期：2026-10-08
- 目标仓库开始 HEAD：`d4638ab4eb7886335c8f438deadcaff12066f1bf`
- 只读参考仓库：`D:\project\doudizhu`
- 参考 commit：`acdcf199cf422e2b683cb22142b8e7cb33b14460`
- 参考许可证：MIT

## 文件映射

| 目标文件 | 来源 | 分类 | 说明 |
|---|---|---|---|
| `guandan_marvel/game.py` | 新写；包装本仓库 A03/A04 | GuanDan 专用 | V2 schema、四座位信息集、结构化 row、Python EnvBatch、序列化、privileged critic API |
| `guandan_marvel/env_batch.py` | `marvel/env_batch.py` | 轻改派生 | import 改为 `guandan_marvel`；NumPy 输入显式转 Torch/TensorDict |
| `guandan_marvel/runtime/sampler.py` | `marvel/runtime/sampler.py` | 轻改派生 | import 改为 `guandan_marvel`；游戏接口通过注入 module 获取 |
| `guandan_marvel/buffer.py` | M01 派生文件 | 小幅补充 | 增加经 TensorDict 验证的 slice assignment，供 complete-trajectory commit buffer 使用 |
| `tests/marvel_adapter/test_adapter.py` | 新写 | 测试 | V1/V2、隐私、overflow、终局、序列化、sampler |
| `scripts/verify_marvel_adapter_v2.py` | 新写 | 证据工具 | 13 级牌 + 贡还牌完整终局扫描 |

## 未复制范围

- 未复制 `nanopokerlib/cpp/games/doudizhu`；
- 未复制斗地主牌型、角色、叫地主、底牌或 evaluator；
- 未迁移 IPPO/VRPO trainer/workflow（M03）；
- 未实现 C++/pybind（M04）；
- 未改变 legacy V1 trainer/evaluator/default backend。

## 许可证与运行依赖

`env_batch.py` 和 `runtime/sampler.py` 保留 SPDX/commit/path 来源头；MIT LICENSE 已随 `guandan_marvel/LICENSE` 分发。运行时代码不导入参考 checkout，不包含对参考绝对路径的依赖。参考仓库审计前后 `git status --short` 为空。
