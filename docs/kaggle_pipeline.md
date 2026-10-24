# Kaggle 自动上传与运行：只需填写一次账号

本机工具已安装在 `D:\project\guandan\.venv-kaggle`，不需要你安装插件、配置 MCP 或修改训练环境。调用官方 Kaggle 客户端完成上传与运行。

工具没有默认历史交付；下次由代理在新包验收后准备新任务。2026-10-24 初次配置未连接 Kaggle；随后按用户提供的临时新版 token 完成认证和只读列表验证，没有拿 v10 测试、上传或启动远端作业。

## 你现在只需要做这一步

1. 打开 [Kaggle 账号 API 设置](https://www.kaggle.com/settings/api)，创建 API Token。
2. 将 `KGAT_` 开头的 token 填入 **`D:\project\guandan\configs\kaggle_account.local.json`** 的 `api_token` 字段：

```json
{
  "api_token": "KGAT_这里替换为你的token"
}
```

用户名由首次认证自动识别并缓存在同一文件。测试用 token 撤销后，在同一位置换上正式 token，再运行下方 `check`。文件已经被 Git 忽略，不进入交付包、Notebook 或运行记录。

仍兼容 Legacy API Credentials 下载的 `kaggle.json`（仅有 `username` 和 `key`）。两种格式选择一种；不要将新版 `KGAT_...` 填入旧 `key` 字段，也不要同时填两种密钥。认证方式依据 [官方说明](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication)。

填写后可在项目目录运行下面的离线检查；它只确认文件格式和已安装依赖，不验证远端登录，也不会上传或消耗 GPU：

```powershell
.\scripts\kaggle.ps1 doctor
```

`READY` 表示本地配置齐全。`ACCOUNT_REQUIRED` 会指出填写位置。权限、凭据有效性和 GPU 是否可调度，要等下次实际连接时确认。

确认凭据有效性可以执行只读连接检查，不上传、不启动作业：

```powershell
.\scripts\kaggle.ps1 check
```

`CONNECTED` 表示账号认证和 Dataset/Notebook 列表读取均通过；仍不表示已经验证上传权限、GPU 调度或训练。该命令只输出账号名和检查结果，不输出 token。

## 下次如何使用

直接告诉代理「新包验收后直接上传并启动」。代理负责选择新交付目录并执行，无需你再配置 Dataset、Notebook、挂载输入或点击 Run All。

对应入口（`<新交付目录>` 必须换成下一轮实际生成的目录；下面不是要求你现在运行）：

```powershell
.\scripts\kaggle.ps1 run -Delivery '<新交付目录>'
```

一次调用将完成：校验真实交付 → 私有 Dataset 上传并等待就绪 → 绑定原 Notebook → 请求 T4×2 → 监控 → 下载指定版本输出 → 校验结果。远端继续执行已有双臂预检、训练和评估；不会绕过预检或改变训练配方。

如果代理已经在打包末尾准备任务：

```powershell
.\scripts\kaggle.ps1 prepare -Delivery '<新交付目录>'
.\scripts\kaggle.ps1 run
```

`prepare` 只做本地校验与 staging；`run` 才连接 Kaggle。相同文件与配置重复 prepare/run 会复用同一个 run_id；不会因为重复命令创建第二个 GPU 作业。

以下命令用于日后管理任务：

```powershell
.\scripts\kaggle.ps1 status
.\scripts\kaggle.ps1 watch
.\scripts\kaggle.ps1 fetch
```

默认管理最近一次由本工具准备的新任务。需要指定某次任务时添加 `-RunId '<run-id>'`。`run -NoWait` 提交后退出；默认 run 会持续监控并下载。关闭本地终端只停止监控，已提交的 Kaggle 作业继续运行；重新执行 watch 恢复。自动下载要求本地监控进程运行。

## 已配置的运行参数

`configs/kaggle_pipeline.json` 已设置私有 Dataset/Notebook、T4×2、Internet 关闭、15000 秒平台请求上限、30 秒查询间隔与 1800 秒 Dataset 处理等待。15000 秒覆盖现有 14400 秒实验预算并留余量，不改包内预算；具体设备与平台限制由 Kaggle 决定。

账号从唯一的 local 文件读取，不要求你配置系统环境变量。`runs/kaggle_pipeline/<run-id>/` 保存原始输入副本、验证证据、run.json、staging、结果与验证结论，不保存密钥。

首版适配**当前 serial_compare 构建器的交付格式**。新包如改用其他业务入口，需要代理增加对应适配，不能把任意 ZIP 当作兼容包。

从源码构建的可选 `-Build` 已接入既有 builder。`build` 默认 null，因为本次明确不绑定 v10/历史输入。下一轮代理根据明确任务填写该字段或直接传新交付目录，不需要用户填写模型路径。构建通过现有 `.venv-gpu` 独立进程加载模型；上传工具无需安装 Torch。配置示意：

```json
"build": {
  "output": "runs/下一轮新的交付目录",
  "model_dir": "下一轮模型输入目录",
  "base_inputs": "下一轮底包输入目录",
  "validation_record": "下一轮本地验收记录.json"
}
```

`-Build -DryRun` 不执行构建。若要刻意将完全相同的包作为另一轮实验重跑，应更换 `experiment`，而不是删除运行记录后反复提交。提交响应丢失时，工具先对账；身份无法确认则停止，不盲目重复开训。

## 结果含义与本地验收

- `SUBMITTED`：平台已接收版本，不代表正式训练已开始。
- `REMOTE_QUEUED/RUNNING`：平台排队/运行状态，不推测训练内部阶段。
- `VERIFIED`：指定版本的结果包完整、哈希及计划匹配，业务阶段完整且包含两臂恢复文件；不等于模型变强或正式晋级。
- `FETCHED_INCOMPLETE`：诊断/部分结果已保留，不能宣布本轮成功。

退出码：0 为当前操作成功；doctor 的 2 表示本地配置未齐；3 表示结果不完整；1 表示操作失败；130 表示用户中断本地客户端。

本地验收命令：

```powershell
& '.\.venv-kaggle\Scripts\python.exe' -m pytest tests/unit/test_kaggle_pipeline.py tests/unit/test_kaggle_pipeline_backend.py tests/unit/test_kaggle_pipeline_cli.py -q
.\scripts\kaggle.ps1 run -DryRun
& '.\.venv-kaggle\Scripts\python.exe' -m pip check
```

测试只用临时样例和假后端；真实账号连接、上传和双 T4/NCCL 验证留到下一轮新交付。完整结果见 `docs/kaggle_pipeline_acceptance_20261024.md`。

如以后迁移机器，重建上传环境即可：

```powershell
& '.\.venv-gpu\Scripts\python.exe' -m venv .venv-kaggle
& '.\.venv-kaggle\Scripts\python.exe' -m pip install -r requirements-kaggle.lock.txt
```
