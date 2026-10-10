# M07 Kaggle 自动解包输入兼容性补修报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 基线实现：`77a3674993d14d8c87fcab3a5084b1cf0c399b67`
- 基线验收登记：`33e06507041f702e49f05e6ef47b60d7a8909d8f`
- 本轮实现 commit：`3934d1f70a8db728ac6c825bb70ce80d6fd60f9a`
- 验收登记 commit：本报告所在提交；精确 SHA 见最终答复
- 状态：`remote_ready / accepted`
- `remote_verified=false`

## 范围

本轮只修复 Kaggle Dataset 自动解包后的 source/recovery 输入兼容性。没有重做上轮 A–F 训练/恢复修复，没有修改 IPPO/VRPO、DDP 数学、规则、性能或模型强度，没有执行 M08、上传 Kaggle、正式长训练或 final holdout。

## Source 两种形态

均已通过：

1. 原始 `guandan-marvel-source.zip` 保留；
2. ZIP 被删除，只保留含 `source_manifest.json` 的解包目录。

支持额外包装目录和 ZIP 内容被平铺到 Dataset 目录。目录模式验证外部 manifest SHA、content manifest hash、source commit、逐文件 bytes/SHA-256、文件集合和路径安全；明确记录 `archive_sha256_verified=false`，不声称验证不存在的 ZIP，也不重新压缩硬凑 hash。

## Recovery 两种形态

均已通过：

1. 原始 `guandan-marvel-recovery.zip` 保留；
2. ZIP 被删除，只保留含 `recovery_manifest.json` 的解包目录。

Recovery schema 升级为 v3，绑定 source commit、原始 source archive identity 和 source content manifest identity。解包 recovery 完成了实际 CPU/gloo resume：iteration 1 -> 2，且 source/recovery 输入树前后 hash 不变。

## 一致解析和并存策略

新增 stdlib-only `guandan_marvel/kaggle/bootstrap.py`：

- 同一字节作为 source module 和 Dataset 外部 `guandan_marvel_bootstrap.py`；
- 生成 Notebook 初始化 cell 验证 bootstrap SHA 后调用它；
- runtime discovery 和 runner 复用同一 resolver/materializer。

ZIP 和目录并存时：两者都先验证；内容 manifest identity 相同则去重并优先目录，不同则 fail closed。

## Fail-closed 与路径安全

本地测试覆盖：

- 多个模式 Dataset；
- 多个不同 source/recovery；
- 篡改和缺失文件；
- 错误 source identity；
- ZIP 路径越界；
- ZIP symlink；
- 目录 symlink 防护；
- 未声明/额外文件；
- source/recovery/manifest/hash/schema 不匹配。

所有 input 只读，源码、恢复文件和 binary 只复制到 working 后使用。

## 实际 Notebook 初始化代码

测试直接读取生成的 `guandan_marvel_t4x2.ipynb` 第一段代码 cell，替换的只有本地临时 input/working root，并实际执行其 bootstrap：

- source ZIP 保留：通过；
- source 自动解包、无 ZIP：通过；
- source + recovery 均自动解包、无 ZIP：通过。

没有只测试独立 helper 后冒充 Notebook 通过。

## 测试结果

- `tests/marvel_kaggle`：32 passed；
- training/core/evaluation 目标回归：87 passed；
- A00：5 passed；
- 总计：124 passed，0 failed，0 errors，0 skipped。

额外执行：

- 精确 implementation commit package build/validation；
- source ZIP 解压、离线 pybind build/import；
- 解压源码 CLI preflight；
- 解包 source + 解包 recovery 的实际 CPU/gloo resume。

前次 A–F 的 112 项历史证据没有改写。

## 正式交付

- 路径：`D:\project\kaggle_train\guandan-marvel-t4x2-20261010`
- Implementation/source commit：`3934d1f70a8db728ac6c825bb70ce80d6fd60f9a`
- Source ZIP SHA-256：`9eb4c4040a3dafda040e6f1b19fd060f110c4137af551f3b489fe42a5f9b1b35`
- Delivery manifest hash：`720a7364041c07e761104cfdfcbdce5b1fa36f469cc3cb00b22d1efa335b7de1`
- Delivery schema：`guandan-marvel-kaggle-delivery-v2`
- Dataset schema：`guandan-marvel-kaggle-dataset-v2`
- Recovery schema：`guandan-marvel-recovery-manifest-v3`
- `workspace_snapshot=false`

每个模式 Dataset 目录包含四个应上传文件：Dataset manifest、bootstrap、source manifest、source ZIP。Kaggle 保留 ZIP 或自动解包均无需修改 Notebook。不得混合多个模式或旧版本 recovery。

## 证据

- `docs/marvel_migration/evidence/M07_kaggle_input_layout_repair/`
- `runs/M07_kaggle_input_layout_repair/20261010_acceptance/`
- `runs/M07_kaggle_input_layout_repair/20261010_reproduction/`

## 限制

所有布局均在本地临时目录模拟；没有伪造 `/kaggle` 或远端验证。CPU/gloo resume 只证明输入布局和 checkpoint 可用性。真实 Kaggle 自动解包行为和 T4×2 仍由 M08 确认。

## 结论

两种 source 和两种 recovery 形态、生成 Notebook 初始化 cell、严格内容身份与路径安全均通过。M07 可恢复 `remote_ready / accepted:true`，M08 可恢复 `ready`，但本轮不执行 M08。
