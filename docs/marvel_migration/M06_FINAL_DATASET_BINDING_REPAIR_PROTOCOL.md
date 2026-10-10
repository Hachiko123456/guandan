# M06 Final Dataset Binding 小范围补修协议

- Stage：`M06_final_dataset_binding_repair`
- 日期：2026-10-10
- 基线实现：`afcd8d9817a2e993b1fae5652bb339fce3294809`
- 基线登记：`597392519c776d65874cc646bc1787261f66a837`
- 范围：只加强 final audit 与本次实际 manifest 的 dataset identity 绑定，并修正一项历史 source hash 元数据

## 1. Dataset identity 传递

Final runner 对调用方传入的 manifest 执行一次深拷贝并验证，之后所有 identity、deal materialization 和 game loop 都只使用该冻结副本。

Final 分支计算：

- `actual_dataset_identity = dataset_identity(manifest, "final")`；
- `actual_generator_version = manifest.generator_version`；
- `actual_deal_count = len(final deals)`。

在加载 policy、materialize deal、创建输出或预约状态之前，调用 `FinalHoldoutAudit.validate_dataset_binding(...)` 做只读独占锁核验。

## 2. 锁内双重核验

`reserve()` 现在必须显式接收：

- `actual_dataset_identity`；
- `actual_generator_version`；
- `actual_deal_count`。

锁内同时比较：

1. 调用方传入的实际 identity；
2. audit 对象创建时预期的 identity；
3. 持久化 `holdout_identity`；
4. generator version；
5. deal count。

任何不一致都在 reservation 写入前失败，audit bytes 不变。

## 3. 兼容性

- final audit schema 仍为 v2；持久化格式和状态机不变；
- `reserve()` Python API 增加三个必填 identity 参数；仓库内调用方全部更新；
- 旧 reserved/running/consumed/failed/interrupted 文件不迁移、不重置；
- candidate、plan、config、evaluation_id、并发锁和一次性语义保持不变。

## 4. 证据勘误

前次 repair `source_manifest.json` 中 `test_hook_process.py` 的记录对应实际文件末尾额外一个换行的内容。基线实现提交中的真实文件为 8914 bytes。只修正该历史条目及其上层 artifact 引用，不重写其他历史源码 hash。

本轮新 source/artifact manifest 独立记录当前实现和本轮测试。
