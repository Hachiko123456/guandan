# A07：Kaggle 纯 Python 启动与源码包使用说明

本说明仅描述可移植入口及待验证操作；**不声称已经在 Kaggle 运行或通过 A07**。
入口不会上传、不会使用账号密钥，也不会自动安装依赖。训练仍采用当前实现：
固定 FIVE、独立首局；不是完整比赛升级流程，也不是 MARVEL Transformer，
不得由部署成功推导规则/训练缺陷已解决或策略强度已验证。

## 职责与公开接口

- `scripts/kaggle_entry.py`：基于自己的 `__file__` 验证仓库根目录，加入
  `sys.path`，委托 `guandan.deployment.session.main(argv)`。不复制 session 的
  argparse、定时保存、信号、恢复、评估或 provenance 逻辑。
- `scripts/kaggle_environment_check.py`：仅在显式调用时检测/复制/安装。
  公开函数：`inspect_environment() -> dict`、
  `require_remote_runtime(system) -> None`、
  `install_source(source_root, dest_root, install_deps=False, dry_run=False) -> dict`、
  `install_source_archive(archive_path, dest_root, install_deps=False, dry_run=False) -> dict`。
- `notebooks/kaggle_training.py`：`# %%` 格式的普通源码模板，可按单元使用，
  也可作为脚本传递同一组参数。示例训练/恢复/评估调用默认均为注释。
- 源码导出、摘要验证、运行证据、manifest、新挂载恢复与 trainer 生命周期由
  `guandan.deployment.package/provenance/session` 及 trainer 所有者负责。

环境报告包括 `python`、`python_version`、`python_version_info`、`platform`、
`torch`（版本或 null）、`torch_status`（ok/missing/error）、`torch_error`、
`cuda_version`、`cuda_available`、`gpu_count`、`gpus`（index/name/total_memory_bytes/
capability）以及 `kaggle` 子项。后者记录标准输入/输出目录是否存在、输出是否
可写，以及存在的 Kaggle marker **名称**，不导出 marker 值或任意环境变量。
`require_remote_runtime` 要求 Python >=3.12、可用 CUDA GPU、标准输入与可写
输出目录。目录和环境 marker 可以伪造，检查通过仍不是远端验收证明；返回
`accepted`、`kaggle_verified` 始终为 false。

## 1.0 Kaggle 单 ZIP 上传方式

Kaggle 输入可以只挂载一个源码 ZIP；不要求上传本地目录。推荐把
`guandan-source.zip` 作为私有 Dataset 的唯一文件上传，例如挂载为：

```text
/kaggle/input/guandan-source/guandan-source.zip
```

Notebook 先从 ZIP 中读取标准库环境检查入口，并调用
`install_source_archive()`。该函数会在写入目标前校验 ZIP 成员路径、特殊文件、
`SOURCE_MANIFEST.json`、完整文件集合和每个文件的 SHA-256，然后只把源码复制到
`/kaggle/working`。它不使用 `ZipFile.extractall()`，不写入 `/kaggle/input`，也不
上传、安装或启动训练。

因此上传时应使用源代码交付目录中的：

```text
D:\project\kaggle_train\guandan_a07_ready_20261004\guandan-source.zip
```

展开后的 `source_dataset` 仅用于本地检查或手工制作 Dataset，不是必须的 Kaggle
上传形式。

## 1. 三类目录必须分开

| 用途 | 示例 | 约束 |
| --- | --- | --- |
| 只读源码 export | `/kaggle/input/your-source/guandan` | 含 SOURCE_MANIFEST.json |
| 可写源码副本 | `/kaggle/working/guandan-source` | 必须是尚不存在的新目录 |
| 会话输出 | `/kaggle/working/guandan` | CLI 默认 output-root |
| 重新挂载的检查点 bundle | `/kaggle/input/your-checkpoints` | resume 指向这里的 manifest.json |

先由上游导出工具生成经过审阅的源码包，由用户自行上传/挂载并解压。
`install_source` 不解压、不 clone、不上传，只复制 manifest 列出的源码文件与
manifest 本身；不复制目录里的额外日志、缓存、检查点、密钥或 `.git`。
它复用 main-owned `source_provenance`，在复制前后验证摘要，拒绝目录重叠、
路径越界、链接文件、不完整 export、已有目标以及 `/kaggle/input` 内的目标。
失败时不递归清理或覆盖任何目标；需要检查失败的副本并选择新的目标目录。
不要将 checkpoint bundle 当作源码安装。

`SOURCE_MANIFEST.json` 使用严格的 `guandan-source-v1` schema：必须包含
`format`、40 位小写 commit、非空 `files`、`profile=remote_full`，以及
`uploaded=false`、`accepted=false`、`kaggle_verified=false`。manifest 中的路径
必须是规范的相对 POSIX 路径；绝对路径、`..`、反斜杠、符号链接、缺失文件和
未列出的额外文件都会被拒绝。文件集合和每个 SHA-256 摘要都必须逐项一致。
源码导出还必须包含 `pyproject.toml`、`configs/acceptance_profiles.json`、
`guandan` 包、provenance/package/session，以及两个 Kaggle 脚本入口；
`profiles-0.2` 的 `remote_full` 配置必须保持 remote/CUDA 的 canonical 设置。

```python
# 在一个明确选择路径的 notebook 单元中运行：
import runpy
api = runpy.run_path('/kaggle/input/your-source/guandan/scripts/kaggle_environment_check.py')
print(api['install_source']('/kaggle/input/your-source/guandan',
                            '/kaggle/working/guandan-source', dry_run=True))
# 审阅 dry-run 后再显式复制；默认不运行 pip：
print(api['install_source']('/kaggle/input/your-source/guandan',
                            '/kaggle/working/guandan-source'))
```

如果需要安装缺失依赖，在**复制时**明确设置 `install_deps=True`。
安装参数来自 `pyproject.toml` 的 dependencies；使用当前 Python 的
`-m pip install --only-binary=:all:`，没有 C++/源码构建回退，没有强制升级
已有 CUDA Torch。如果没有匹配 wheel 或网络不可用，安装明确失败，不降级。
`dry_run=True` 不写文件、不调用 pip。普通系统检查、plan、训练入口都不安装。

## 2. 无执行的检查

下列 CLI 假定 cwd 为可写源码副本；也可用脚本绝对路径从任意 cwd 调用。
不嵌入任何开发机盘符、Python 环境位置或旧 checkpoint 路径。

```text
python scripts/kaggle_entry.py --help
python scripts/kaggle_entry.py --action plan --profile remote_full
python scripts/kaggle_entry.py --action system --profile remote_full
python scripts/kaggle_environment_check.py --help
python scripts/kaggle_environment_check.py --show-system
python scripts/kaggle_environment_check.py --check-dependencies
python scripts/kaggle_environment_check.py --show-system --require-remote
```

`--help` 不做硬件探测/训练/依赖导入；系统检查会按需导入 Torch 以取得真实
CUDA/GPU 信息，但不导入 trainer。依赖检查单独显式执行，报告导入与已装版本，
不假装完成依赖版本约束解析。Torch 缺失时系统报告仍可打印；`--require-remote`
则失败。来自 `GUANDAN_PROFILE` 等环境变量的值不应绕过 session 的 CLI 执行门。

## 3. 仅显式执行 remote_full

会话 CLI 只接受 `remote_full`；smoke/train/evaluate 都要 `--execute`，
plan/system 不执行训练。不要添加另一组 `--train`、`--allow-remote-full` 等
并行开关；以 main-owned `--help` 为准。

```text
python scripts/kaggle_entry.py --action smoke --profile remote_full --execute
python scripts/kaggle_entry.py --action train --profile remote_full --execute --algorithm ippo --output-root /kaggle/working/guandan --session-hours 10 --save-margin-seconds 300 --checkpoint-seconds 600
python scripts/kaggle_entry.py --action train --profile remote_full --execute --algorithm vrpo --output-root /kaggle/working/guandan
```

`session-hours=10`、`save-margin-seconds=300`、`checkpoint-seconds=600` 是
CLI 默认预算，不是平台时限承诺。session/trainer 在安全更新边界执行保存与
退出，信号处理只请求停止；不可保存未消费的 rollout 作为完整更新。
软预算、信号停机或未完成运行必须在证据与退出码中保持非成功。SIGKILL、
断电及平台即时强杀没有保存保证；可恢复的是此前实际写成并校验的检查点。

## 4. 新会话与 A06 入口

用户负责保存可写输出，并在新会话中重新挂载 checkpoint bundle。重新复制
相同审阅源码后，把 resume 指向**新挂载** manifest，而不是旧绝对 `.pt` 路径：

```text
python scripts/kaggle_entry.py --action train --profile remote_full --execute --algorithm ippo --resume /kaggle/input/your-checkpoints/manifest.json --output-root /kaggle/working/guandan
python scripts/kaggle_entry.py --action evaluate --profile remote_full --execute --candidate-checkpoint /kaggle/input/your-candidate/candidate.pt --snapshot-checkpoint /kaggle/input/your-snapshot/snapshot.pt --output-root /kaggle/working/guandan
```

checkpoint/manifest 校验、相对路径重定位、算法与 profile 一致性、A06 输入
边界及失败退出码由 session 负责。示例本身不会代替任何 A05/A06 前置证据。
Notebook 模板里的 `run_session` 始终启动新的 Python 进程，避免混用旧挂载的
模块缓存；若直接在同一进程里载入另一份 guandan，入口拒绝并提示重启 kernel。

## 5. 证据与剩余验收

上游 session 应保存准确命令、运行时、配置及摘要、Git/source provenance、
checkpoint/manifest 路径与摘要、结果及停止原因。源码复制报告只证明本地复制
与校验，不能代替 session evidence。入口原样返回 session 退出码；checker
失败返回非零，pip 非零退出码不被伪装成成功。

本 worker 不运行远端或上传，不修改 STATUS，也不授予验收。
本地单测使用临时源码副本、缺失 Torch 模拟和模拟时钟/信号，不等于 Kaggle
GPU、完整 remote_full 训练、完整 A06 或跨真实会话恢复。测试会覆盖本地
`build_package → unzip → install_source` 的端到端边界，但该流程仍不是 Kaggle
上传、远程执行或验收证据。
`python scripts/run_acceptance.py --stage A07` 的验收套件由主任务负责；本 worker
不编辑该 runner 或 acceptance 文件。

## 单 Notebook 自动流水线（2026-10-04）

统一入口为 `python -B -m guandan.deployment.workflow --execute`。
`scripts/prepare_kaggle_unified.py --output <新的交付目录>` 从干净提交导出：

- `guandan-source.zip`：上传为一个私有 Dataset；
- `guandan_all_in_one.ipynb`：只需导入一份，点击 Run All；
- `PACKAGE_METADATA.json` 与使用说明。

默认顺序：环境检查和 A04 smoke → IPPO base100 → checkpoint 重载到102 →
VRPO base100 → 重载到102 → IPPO A06 3000局 → VRPO A06 3000局。
计数来自同一份 profiles-0.2，不能降采样后称为 remote_full。

全流程共用一个 CheckpointController，总预算不是每个阶段单独10小时。
Notebook 的源码准备时间也从总预算扣除；同一 kernel 重跑不重置起始时间。
预算不足为 incomplete，不承诺单次 Kaggle 会话一定跑完。

只有工作流显式提供、且与 output-root 解析后相等的 `/kaggle/working` 子目录，
才可作为已产出 checkpoint 的输入范围；普通单阶段 CLI 仍只接受 `/kaggle/input`。
这不是 runtime_check 绕过，也不改变 A00-A04 的规则、动作或环境。

同会话重载记录 `recovery_origin=current_working_output` 和 `new_session_resume=false`。
`genuine_new_kaggle_session_verified`、`accepted`、`kaggle_verified` 始终为 false；
本流水线不能凭一次 Run All 满足真实新 Kaggle 会话恢复验收。

每个完整阶段后及可捕获退出时，自动写本地 `guandan-unified-progress.zip`；
它包含相对路径、SHA256、update100 snapshot、update102 candidate、会话JSON报告与状态。
Notebook 的 stdout 日志另存于 `/kaggle/working/guandan-unified-logs`，请一并下载。
新的 Kaggle 会话挂载该进度包和**同一源码 ZIP**，仍用同一个 Notebook 运行。
可读取进度 ZIP 或已展开的进度 Dataset，不自动上传，不覆盖已有进度。
已完成阶段须验证实际 checkpoint/逐局记录再跳过；未知失败默认拒绝继续，
只有修复后显式 `--retry-failed` 才可重试。

当前不支持 A06 半份评估逐局续跑：保留其已完成局的诊断记录，但未完成算法的
评估从头重跑；不会合并两次局数充当一次完整评估。完整完成的另一算法评估可跳过。
强制断电/强杀时只能依赖此前已保存的文件，不能承诺最后的进度 ZIP 一定更新。
