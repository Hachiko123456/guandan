# Kaggle 通用一键上传器 v2 设计

日期：2026-10-24。状态：设计完成，未实施。本轮没有调用 Kaggle 上传、提交或训练。

## 结论

现有 `scripts/kaggle.ps1` 和 `scripts/kaggle_pipeline.py` 已经是通用编排器；`scripts/upload_kaggle_v12.ps1` 只是将某个交付目录硬编码进去的包装脚本。后续应保留一个稳定入口 `scripts/upload_kaggle.ps1`，每次打包只发布一份经过 QA 的“当前交付指针”，不再生成版本脚本。

当前日志只能满足基本恢复，尚不能称为完善：

- `run.json` 保存最新状态，但没有不可变的状态变化历史。
- 终端输出 JSON，没有自动保存完整控制台记录。
- 构建已有独立 stdout/stderr；上传、Dataset 处理、Notebook 提交、轮询、下载没有对应阶段日志。
- 为防 token 或服务端正文泄漏，SDK stdout/stderr 被导向空设备；这保护了凭据，也同时丢掉了上传进度和诊断。
- mutation 前已保存 `dataset_attempted` / `kernel_attempted`，因此可避免盲目重复提交；但异常本身没有统一写入 `failure.json` 和事件流水。
- watch 只在状态变化时输出，没有长时间运行心跳，用户难以区分“仍在等待”和“本地监控卡死”。
- 远端最终输出按固定 Notebook 版本下载并校验，这是现有实现中应保留的强项。

## 用户接口

固定文件：

```powershell
.\scripts\upload_kaggle.ps1
```

默认读取 `runs/kaggle_delivery/current.json`。该指针只在新交付完成实际 payload QA 后由构建/验收步骤原子更新，不能按目录修改时间猜“最新”。用户每次执行相同文件。

支持：

```powershell
# 使用已发布的当前交付：检查账号、上传、提交、监控、下载、校验
.\scripts\upload_kaggle.ps1

# 完全离线，显示将使用哪个交付及全部哈希
.\scripts\upload_kaggle.ps1 -DryRun

# 显式选择另一份已通过 QA 的交付；不改 current 指针
.\scripts\upload_kaggle.ps1 -Delivery 'D:\project\guandan\runs\...\delivery_v13'

# 提交后退出，本地以后用同一入口恢复监控
.\scripts\upload_kaggle.ps1 -NoWait
.\scripts\upload_kaggle.ps1 -Watch -RunId '<run-id>'

# 只查看本地快照和关键日志路径，不访问远端
.\scripts\upload_kaggle.ps1 -Status -RunId '<run-id>'
```

`-Delivery` 必须是通过项目实际 verifier 的目录。脚本不接受裸 ZIP、不自动选择最大版本号、不从失败 staging 推断目标。

## 当前交付指针

`runs/kaggle_delivery/current.json` 不含凭据，建议格式：

```json
{
  "schema": "guandan-kaggle-current-delivery-v1",
  "delivery_id": "serial-compare-v12",
  "delivery_dir": "runs/t4x2_serial_compare_delivery_20261023/delivery_v12",
  "payload": "guandan_serial_compare_payload_v1.bin",
  "payload_sha256": "...",
  "notebook": "guandan_serial_compare_v1.ipynb",
  "notebook_sha256": "...",
  "plan_sha256": "...",
  "manifest_hash": "...",
  "artifact_qa_path": "runs/.../ARTIFACT_QA_V12.json",
  "artifact_qa_hash": "...",
  "published_at": "2026-10-24T...+08:00"
}
```

发布动作必须再次读取交付文件和 QA，逐项核对哈希，然后先写临时文件、落盘、原子替换 current.json。旧 current 指针可以保存到 `runs/kaggle_delivery/history/<delivery_id>.json`，但不会被自动执行。

## 日志结构

每个 run 固定使用：

```text
runs/kaggle_pipeline/<run-id>/
  run.json                 最新可恢复状态快照
  events.jsonl             追加式结构化事件流水
  console.log              与终端一致的脱敏人类日志
  delivery.json            本轮绑定的交付身份和全部哈希
  validation.json          prepare 的 actual artifact QA
  failure.json             最近失败的安全摘要和恢复建议
  dataset/                  严格上传staging，仅payload和metadata
    dataset-metadata.json
  kernel/                   严格提交staging，仅Notebook和metadata
    kernel-metadata.json
  records/
   dataset/
    upload.json             bytes、开始/结束时间、耗时、结果引用
    processing.jsonl        Dataset 状态轮询历史
   kernel/
    submission.json         版本、URL、提交时间、绑定身份
    status.jsonl            Notebook 状态历史与心跳
  outputs-<id>/
    ...                     固定版本下载结果
    PIPELINE_VERIFICATION.json
```

`events.jsonl` 每行字段：

```json
{
  "schema": "guandan-kaggle-event-v1",
  "sequence": 17,
  "time": "2026-10-24T12:34:56.789+08:00",
  "run_id": "guandan-...",
  "stage": "dataset_processing",
  "event": "status_changed",
  "status": "ready",
  "attempt": 1,
  "elapsed_seconds": 42.3,
  "details": {"dataset_ref": "owner/slug", "version": 1}
}
```

事件写入应 flush；重要 mutation 边界按以下顺序记录：

```text
intent_persisted → remote_call_started → remote_call_returned/remote_outcome_unknown
                  → reconciliation_started → reconciled/blocked
```

因此即使进程在远端已经接收请求后崩溃，重启也会先对账，不重复开 GPU 作业。

## 日志内容和进度

上传阶段记录文件名、SHA-256、bytes、开始/完成时间、耗时和 Dataset 引用。官方 SDK 当前一次性上传文件，首版不伪造百分比；显示“已开始上传 48,422,334 bytes”和“上传请求完成，等待处理”。若以后接入支持回调的分块上传，再增加真实 `bytes_sent/total_bytes`。

Dataset 与 Notebook 状态只在状态变化时写 `status_changed`，并每 5 分钟写一次 `heartbeat`。终端示例：

```text
[12:03:11] VALIDATED delivery=serial-compare-v12 payload=d05a0e16…
[12:03:14] DATASET_UPLOADING bytes=48422334
[12:04:02] DATASET_READY version=1 elapsed=48s
[12:04:05] SUBMITTED kernel=hachiko123/guandan-... version=1
[12:09:05] REMOTE_RUNNING heartbeat elapsed=5m
[14:27:40] REMOTE_COMPLETE; downloading exact version 1
[14:29:03] VERIFIED verdict=INCONCLUSIVE output=...
```

远端状态接口无法可靠提供训练内部 phase 时，不推测“正在 candidate 训练”。完成后下载 Notebook 日志和输出，再从受信任的本项目结果 schema 提取 phase 状态。

## 失败日志与脱敏

SDK 的原始 response body、HTTP headers、环境变量和 traceback 不直接落盘。`failure.json` 只允许：

- stage、operation、exception type、HTTP status；
- run_id、远端资源引用和已知版本；
- outcome 是 `definite_failure`、`outcome_unknown` 或 `reconciled`；
- safe_message、recoverable、next_command；
- 对应事件 sequence。

统一 logger 在写盘前递归拒绝敏感字段名：`token`、`key`、`authorization`、`cookie`、`secret`、`password`；字符串中出现 `KGAT_` 或 Bearer 头模式则拒绝写入并改成 `[REDACTED]`。测试应将假 token 注入异常、stdout、stderr 和服务端正文，断言所有日志文件均不含该值。

保留后台 SDK 输出丢弃策略，但由适配器主动产生受控事件，例如 operation、HTTP status、耗时、bytes。这样不靠解析不可信 SDK 文本，也不牺牲必要诊断。

## 组件划分

- `scripts/upload_kaggle.ps1`：唯一稳定的人类入口，只做参数转发和退出码保留。
- `scripts/kaggle_pipeline.py`：CLI 路由、终端显示、顶层安全错误处理。
- `tools/kaggle_pipeline/logging.py`：事件、console、failure、脱敏、sequence。
- `tools/kaggle_pipeline/delivery_registry.py`：验证/发布/解析 current 指针。
- `tools/kaggle_pipeline/core.py`：状态机与幂等编排，通过 event sink 记录事件。
- `tools/kaggle_pipeline/backend.py`：Kaggle API；返回结构化安全结果，不写日志文件。
- `tools/kaggle_pipeline/results.py`：固定版本结果验证。

`upload_kaggle_v12.ps1` 在通用入口验收后删除；以后新版本只更新 current.json。原 `kaggle.ps1` 可保留为底层维护入口，日常用户只运行 `upload_kaggle.ps1`。

## Acceptance Criteria

1. 连续构建 v13/v14 时，仓库始终只有一个 `upload_kaggle.ps1`；current 指针分别原子更新。
2. 无参数 dry-run 显示明确交付目录、payload/Notebook/plan/QA 哈希，且无远端调用。
3. current 指针缺失、QA失败、文件哈希变化、多个候选包时，上传前失败。
4. 每个状态变化在 run.json 和 events.jsonl 中一致；事件 sequence 单调、时间有效。
5. 上传开始/完成、Dataset处理、提交、远端状态、下载和结果验证均有持久化事件。
6. 进程在 Dataset 创建或 Notebook 提交响应前后终止，重启后不会重复 mutation。
7. 终端中断后，`-Watch -RunId` 能恢复；心跳表明监控仍存活。
8. 假 token 出现在异常、SDK stdout/stderr/响应正文时，所有持久日志和终端输出均不含 token。
9. 远端失败保留版本固定的可用诊断；结果不完整仍是 FETCHED_INCOMPLETE。
10. 旧 `kaggle.ps1` 的现有流程测试继续通过，通用入口不改变训练配方或交付内容。

## 实施顺序与修改范围

1. 先实现 logging 与 delivery registry 及纯临时目录测试。
2. 改 core 注入 event sink；保留现有 run.json schema兼容读取，升级写入新schema。
3. 新增固定 `upload_kaggle.ps1`，迁移 v12 为 current 指针；仅执行 dry-run。
4. 做中断/重复提交/脱敏/锁/版本固定回归。
5. 验收后删除版本包装脚本。真实上传仍由用户明确执行。

允许修改 `tools/kaggle_pipeline/`、`scripts/kaggle_pipeline.py`、`scripts/kaggle.ps1`、新增固定入口、pipeline配置/文档/测试和delivery registry。禁止修改训练器、模型、实验plan、交付payload和已封存结果。

## 风险与取舍

- 首版只能报告上传开始/完成，不能显示虚假的逐字节百分比。
- Kaggle运行中可能只暴露排队/运行/完成，内部训练phase需等日志产出后确认。
- current指针让命令稳定，也增加“错误发布指针”的风险，所以发布必须依赖实际QA及哈希复核，不能手工只改路径。
- append-only事件日志在本地并非防篡改审计系统；可在最终结果中记录其SHA-256，足以服务本项目复现和诊断。
