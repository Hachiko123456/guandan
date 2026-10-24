# Kaggle 打包后自动上传并运行方案

日期：2026-10-24。状态：方案，未实施、未上传、未启动远端训练。

## 结论与范围

可以把现有交付接成「打包校验 → 上传私有 Dataset → 提交 GPU Notebook → 自动预检与训练 → 下载结果」的单入口流程。推荐以官方 Kaggle CLI/API 为执行后端，kaggle-tools/MCP 作为可选调用入口，避免必须依赖网页点击。

本次会话没有暴露 kaggle-tools 的工具或说明，不能确认用户所指的具体同名工具。接入前需核实它是否支持 Dataset 创建/版本更新、Notebook 提交运行、状态查询和产物下载；缺少的操作直接由官方 CLI 补齐。仅支持搜索/下载数据的工具不能独立完成此闭环。

## 当前项目依据

- 当前入口：`docs/marvel_v14_prompts/CURRENT_EXECUTION.md`，最新条目为 serial compare v12。
- 当前交付：`runs/t4x2_serial_compare_delivery_20261023/delivery_v12/`。
- 输入文件：`guandan_serial_compare_payload_v1.bin`。
- Notebook：`guandan_serial_compare_v1.ipynb`。
- 构建：`scripts/build_t4x2_serial_compare.py`。
- 交付验证：`scripts/verify_t4x2_serial_compare_delivery.py`。
- 已有编排：`tools/serial_compare/`，包含 control/candidate 预检、串行训练、评估、结果打包及恢复逻辑。

当前包已含业务编排；新层只负责云端运输和作业管理。不能用旧 Bootcamp/M09 入口拼接替代当前 v10，也不能把两臂比较改成另一套训练方案。本文仅确认文件与入口存在，没有重做交付验收。

## Spec

### 用户体验

首次配置本机 Kaggle 认证、账号和固定实验配置。凭据留在本机受支持的认证存储或环境中，不进入代码、交付包、Notebook 或日志。

以后用户说「打包并上传启动本轮实验」，代理调用同一个入口，自动完成上传与启动。运行配方、双卡要求、预算和恢复来源由配置保存，不再每次手填。作业返回可打开的 Kaggle 链接；排队、运行、失败和完成分别报告。

建议分两步交付：第一步直接消费已生成的 delivery_v10，实现「验证、上传、启动、下载」；第二步再把既有构建命令纳入同一入口，实现从源码构建到结果回收。

### 执行流程

1. 构建或接收明确指定的交付目录；禁止通过「最近修改目录」猜测版本。
2. 运行既有交付验证，检查 payload/Notebook/manifest 的绑定关系；本地验收失败则不上传。
3. 在独立 staging 目录仅放 payload 和 Dataset 元数据，创建私有 Dataset。首版建议每个实验使用唯一 slug，避免多个提交争抢 latest。相同实验重试复用已记录资源。
4. 等待 Dataset 处理完成；失败或超时则停止，不提前提交 Notebook。
5. 复制经过验证的 Notebook，生成 `kernel-metadata.json`，设置私有、输入 Dataset、GPU T4×2 和既定时间预算。保持已封装的离线依赖方式及 Internet 设置。
6. 使用官方提交运行接口。记录 Kaggle 返回的 Notebook 引用与版本，先报告「已提交」；只有实际远端证据表明进入训练才报告「训练已开始」。
7. 远端沿用现有 bootstrap 和双臂 preflight，检查真实 GPU 数量、设备、NCCL、native/运行时身份、更新/保存/恢复与评估；全部通过后在同一次作业内自动开始正式阶段。失败保留诊断，不绕过门槛。
8. 后台作业运行独立于本地客户端。监控器可停止并稍后按 run_id 恢复查询；本地关闭不会取消已提交作业，但本地监控和下载需要客户端重新运行或常驻执行器。
9. 完成后按固定 Notebook 版本下载结果到独立目录，验证结果清单、哈希、终止原因及业务阶段完成情况，生成简报。Kaggle COMPLETE 不自动等于训练目标或棋力验收通过。

### 拟新增接口与数据

以下是待实现接口，不是目前可执行命令：

```powershell
& '.\.venv-gpu\Scripts\python.exe' scripts/kaggle_pipeline.py run --config configs/kaggle_pipeline.json
& '.\.venv-gpu\Scripts\python.exe' scripts/kaggle_pipeline.py status --run-id <run-id>
& '.\.venv-gpu\Scripts\python.exe' scripts/kaggle_pipeline.py fetch --run-id <run-id>
```

配置保存：Kaggle owner、experiment、delivery_dir、构建输入、accelerator、timeout_seconds、output_root，以及可选恢复包路径。实验超参继续由当前交付 plan 管理，不复制一份容易漂移的训练参数。

`runs/kaggle_pipeline/<run-id>/run.json` 保存：schema、run_id、delivery/payload/Notebook/plan 哈希、Dataset 引用与版本、Kernel 引用与版本、各阶段时间及状态、最近错误、结果目录与校验结论。不得包含认证数据。

状态示例：VALIDATED → DATASET_UPLOADING → DATASET_READY → SUBMITTED → QUEUED/RUNNING → FETCHED → VERIFIED；失败记录发生阶段及证据。远端运行阶段只在能取得真实证据时细分为 preflight/training/evaluation。

### 失败与恢复边界

- 网络失败可以退避重试；上传/提交响应丢失时先对账，不能盲目再次创建 GPU 作业。
- 同一 run_id 只允许一个活跃远端版本；提交存在并发锁和恢复记录。
- 无 GPU 权限、调度资源不足或平台运行限制时如实报告，不能保证立即开训，也不切到 CPU。
- 软件或 native 身份不匹配时停止，不通过自动更新依赖或重编译绕过当前恢复协议。
- 首版只运行一个已确定预算的实验，不自动无限续跑。后续跨会话续训复用现有恢复协议，明确总预算与终止条件；仅使用实际保存且校验通过的恢复产物。
- 当前配置与首次远端授权完成后，同一轮流程自动推进，不在每个运输步骤重复索要确认。变更配方、扩大预算或启动另一个实验属于新范围。

## Acceptance Criteria

1. 一次 run 调用完成交付验证、私有上传和 Notebook 提交，不需要手动 Add Input 或 Run All。
2. 校验失败、Dataset 未就绪、认证失败时不会提交 GPU 作业。
3. 提交响应丢失/进程重启/重复命令不产生重复 GPU 作业；结果始终对应本轮记录的版本。
4. 远端真实双 T4/NCCL 预检通过后自动训练；预检失败时保留诊断且不开始正式阶段。
5. fetch 校验实际输出；缺失恢复包、哈希错误、阶段未完成时不标 VERIFIED。
6. 账号凭据不出现在交付包、生成元数据和日志中。
7. 首次真实端到端验证采用明确的小预算作业；通过后再验收既定正式实验，二者证据分开。

拟实施后的验证命令（相关新文件尚不存在，本次未运行）：

```powershell
& '.\.venv-gpu\Scripts\python.exe' -m pytest tests/unit/test_kaggle_pipeline.py -q
& '.\.venv-gpu\Scripts\python.exe' scripts/kaggle_pipeline.py run --config configs/kaggle_pipeline.json --dry-run
```

单测重点为失败中断、幂等对账、版本隔离、凭据脱敏和输出验证，不把 mock 成功作为远端验收。既有交付验证脚本的参数以其 `--help` 为准。

## 执行计划与允许修改范围

1. 主代理确认 Kaggle 后端、认证和当前交付身份，先保存 Spec/验收条件。
2. 独立实现任务：新增 `tools/kaggle_pipeline/`、`scripts/kaggle_pipeline.py`、`configs/kaggle_pipeline.json`，禁止修改训练器、奖励、实验 plan、checkpoint schema 和已封存交付。
3. 独立测试任务：新增 `tests/unit/test_kaggle_pipeline.py`，依赖稳定接口后开展；不与实现任务同时写同一文件。
4. 主代理独立审查 diff、运行测试和 dry-run，核对断言与失败分支，再做真实小预算端到端验证。
5. 验证通过后接入已有 builder，补充 `docs/kaggle_pipeline.md`；正式任务沿用明确的预算和终止条件。

当前只交付方案文档；自动化代码、账号连接和远端验证均未完成。

## 官方依据

- https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md：Dataset 创建、版本更新和状态查询。
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md：`kernels push` 上传后尝试执行，及 status/output 操作。
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md：私有、GPU、machine_shape 和 dataset_sources。
- https://www.kaggle.com/docs/mcp：官方 MCP 接入文档；具体工具能力和账号授权仍需接入时核实。
