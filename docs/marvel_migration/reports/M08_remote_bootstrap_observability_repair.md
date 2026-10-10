# M08 远端 bootstrap 空白挂起补修报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 状态：修复完成，等待用户用新交付重新执行真实 Kaggle preflight
- M08：仍为 `ready / accepted:false`

## 远端问题

用户真实 Kaggle T4×2 运行显示 `Running for 5581.8s`，同时 Logs 为空、Output 为 0 B。该运行没有进入 runner/训练阶段，不能作为 M08 证据，应停止且不得使用旧 Notebook 原样重跑。

代码审查确认旧 Notebook 在 `prepare_inputs()` 后通过无 timeout 的 `subprocess.run(... setup.py build_ext --inplace)` 阻塞；训练 `max_wall_seconds` 只在 runner/DDP 启动后生效。旧 bootstrap 在 source materialization/native build 前无 flush 日志、无 heartbeat、无失败诊断，因此 native compiler 或 Kaggle kernel 挂起时会无限占用 GPU 且页面空白。

## 修复

- bootstrap 从第一个语句开始创建 `/kaggle/working/guandan-marvel-bootstrap/`；
- Dataset 发现、校验、materialization、native build/import、runner 前后即时 flush 日志；
- native build 和 runner wrapper 每 5 秒 heartbeat；
- `PYTHONUNBUFFERED=1`；
- compiler 内层 timeout 600 秒，Notebook native-build 外层 timeout 660 秒，native import 120 秒；
- setup.py 的 gendef/dlltool/compiler 均使用带 timeout 的统一执行器并报告分段耗时；
- bootstrap 失败写 `bootstrap_progress.json`、`bootstrap.log`、`bootstrap_failure.json`；
- 失败时生成 `/kaggle/working/guandan-marvel-bootstrap-diagnostics.zip`；
- runner 也有基于模式 wall budget 的外层超时保护；
- Notebook validator 要求保留 timeout/heartbeat/diagnostics 标记；
- 增加 native-build timeout 故障注入测试。

## 本地实际验证

从生成的 dirty source ZIP 真实执行 Notebook bootstrap cell：

- Dataset 发现和校验立即输出；
- source materialization 后 native build 每 5 秒输出 heartbeat；
- Windows 离线 native build 完成；
- native import/build identity 通过；
- bootstrap 总耗时约 48.9 秒；
- 最终 progress 为 `ready_for_runner / complete`。

这不是 Kaggle/T4 远端通过证据，只证明新 Notebook 不再允许旧版无日志无限等待。

## 测试

- `tests/marvel_kaggle`：33 passed；
- training/core/evaluation 目标回归：87 passed；
- A00：5 passed；
- 合计 125 passed，0 failed，0 errors，0 skipped；
- `git diff --check`：通过。

## 重新运行要求

必须重新生成 clean-commit delivery，用户只上传新 `datasets/preflight` 四文件并重新导入新 Notebook。旧 Dataset/Notebook 不得复用。新运行若 native build 超过 660 秒会失败并产生可下载诊断，而不是继续消耗 GPU。
