# 本地 / Kaggle 限时性能验证（只短测，不改训练算法）

## 目的和范围

这三个入口只测量现有模型的推理、现有 IPPO/VRPO 的极短训练、完整终局的小样本评估。
不会启动 `guandan.experiment`，不会长训练、自动上传、自动接续旧训练、或承诺 GPU 100% 利用率。
默认 `plan_only`：不导入 Torch、不初始化 CUDA、不加载权重、不执行对局。

写入报告的 `samples/games/updates/token_steps` 是实际完成计数，超时未完成的批次不算成功。
推理是明确标记的合成输入微基准，不可冒充游戏吞吐或真实训练吞吐。训练重新按 seed 初始化；
可选训练 bundle 只提供推理和评估权重，不恢复优化器，不覆盖原 checkpoint。
评估固定 random 对手，仅用于完整链路计时，不宣称模型能力或正式座位平衡评估。

## 1. 本地先运行

现有环境（本机单 RTX 3050 Ti 4GB）：

```powershell
& C:\Users\yhx\.conda\envs\guandan_train\python.exe scripts/run_local_performance.py --output runs/performance-plan
```

检查 `plan.json`，确认后执行：

```powershell
& C:\Users\yhx\.conda\envs\guandan_train\python.exe scripts/run_local_performance.py --execute --budget-seconds 300 --output runs/performance-reference --bundle C:\Users\yhx\Downloads\guandan-dual-gpu-progress.zip
```

每次用新的 output 目录，拒绝覆盖。可省略 bundle，使用固定 seed 的初始模型；所有被比较 lane 必须使用同一模型来源。
显式 CPU vs `cuda:0`，CUDA 不可用会失败，不会偷偷回退 CPU。默认 batch=32，不盲目使用 T4 显存配置。
计时开始于 gate 初始化之前，Torch/CUDA 初始化、验证 ZIP/manifest 和加载均计入总预算。
仅有效完成的小样本结果用于比较；不足预算时报告 incomplete，不外推一个“成功”的加速比。

在参考版本完成一次报告后，集成优化，使用相同参数再测：

```powershell
& C:\Users\yhx\.conda\envs\guandan_train\python.exe scripts/run_local_performance.py --execute --budget-seconds 300 --output runs/performance-optimized --bundle C:\Users\yhx\Downloads\guandan-dual-gpu-progress.zip --reference-report runs/performance-reference/result.json
```

本地参考/优化要求同硬件、同软件版本、同 seeds、同模型权重和工作量。
默认每 lane：IPPO/VRPO 各 1 shard；每 shard 推理 16×32 samples、短训练 2 updates（8环境×128ticks/update）、评估1完整局。每算法2048训练tokens，全局4096；单双卡总量一样。
`--shards-per-algorithm` 对两 lane 同时生效，不允许双卡偷偷增加样本。
所有 workload 参数有小范围硬上限，不能把 gate 变成长训练入口。
每种 workload 按同批参数比较；各阶段平分剩余预算，防止前一项耗尽全部预算。

## 2. 判断报告

- `valid=true`：所有比较任务完整完成、数量一致；FP32推理探针通过容差检查；评估实际终局、训练有限 loss。
- `comparisons.inference/train/evaluation.speedup`：CPU wall / GPU wall 或单T4 wall / 双T4 wall。
- wall 包含对应子进程启动和初始化。另记同步后的 `steady_wall_seconds`、`steady_counts`、`initialization_wall_seconds`；不能混为精确 CUDA kernel 时间。训练 steady 从现有deadline的before_update边界计到最后update回调，不改变算法；包含全部rollout/update/checkpoint，不包含模型/环境初始化。lane steady为全部worker区间的时间并集，并发重叠只算一次，不能把双卡worker耗时相加冒充wall。
- `hardware` 记录真实设备名称、显存、CPU affinity、Torch/CUDA等，不假定本地是 T4。
- `recommended=false` 不等于代码报错：吞吐不足、没有参考证据、缺远端证据都会不推荐。
- 本地无论多快都是 `pending_remote`；不能以3050Ti成绩推断T4收益。
- 任意缺失、非有限时间、零时间、未完成或不同工作量：speedup=null。
- 本地 reference/optimized 目标按train/evaluation分别 >=2，比较各自CPU/GPU端到端；inference模型未改，不要求2倍，只验证数值一致且没有超过20%的明显回退。
- release按train/evaluation分别判断，要求同一优化源码对应的本地正确性及该项>=2目标，真实Kaggle该项单/双T4>=1.5。inference不要求双卡>=1.5，不因启动噪声否决所有任务。
- `release.per_workload`分别报告建议；本地GPU慢于CPU会明确`local_gpu_recommended=false`、`not_recommended`。本地报告仍不能代替T4验收。
- gate只是小样本有限性、合法性和数值检查，完整规则等价/算法回归仍由项目测试验收；不能用本脚本替代。

日志每5秒报告已完成任务、实际进度文件、耗时和剩余预算。没有足够数据时 ETA 为 null，不编造时间。
子进程日志和每个任务 `result.json/progress.json` 保留。环境中长单次操作可暂时没有新的token，但父进程仍显示预算。
失败不会变为成功；原始错误在 `worker.log` 和 `result.json` 中。

## 3. 打包（与旧packager独立）

完成本地验收、停止代码编辑后：

```powershell
& C:\Users\yhx\.conda\envs\guandan_train\python.exe scripts/prepare_kaggle_performance.py --output release-performance-20261007
```

新生成器会快照限定源码文件，不修改旧packager，不读取/打包 runs、checkpoint、密钥或整个Downloads。
允许未提交的已跟踪源码及本功能五个新增文件；metadata明确标注dirty，Git commit仅是来源，不假装代表全部内容。
Notebook 固定完整源码ZIP SHA256与SOURCE_MANIFEST.json SHA256；ZIP校验整包，展开目录校验manifest固定hash、文件集合及每个文件SHA256，拒绝路径逃逸和额外文件。源码和Notebook必须配套，任何改动必须重新生成。
尚未完成远端验证的包一律 `pending_remote`，不能称作正式长跑放行。

## 4. 上传 Kaggle（一次短测）

上传/添加：

1. 新生成 `guandan-performance-source.zip`，支持Kaggle自动解压（无需重新压缩）；
2. 可选已有 `guandan-dual-gpu-progress.zip` 或直接包含workers/的目录；
3. 可选本地优化 `result.json` 与参考 `result.json`，重命名便于区分；
4. 导入配套 `guandan_performance.ipynb`。

先保持 `EXECUTE=False` 并 Run All，只查看计划，无需开启GPU。
准备好后选择 GPU T4 x2。默认配置会自动在 `/kaggle/input` 下查找路径，不需要手填：

```python
EXECUTE = True
SOURCE_ZIP = ''       # 空字符串：按本 Notebook 固定 SHA 自动寻找唯一源码 ZIP/展开目录
PROGRESS_BUNDLE = ''  # 空字符串：自动寻找唯一 progress；None：明确使用 fresh seeded model
BUNDLE_SHA256 = ''    # 自动发现 ZIP 时可留空；填写后仍会校验原始 ZIP SHA256
LOCAL_REPORT = ''
REFERENCE_REPORT = ''
BUDGET_SECONDS = 600
```

自动发现规则：

- 源码 ZIP 必须匹配本 Notebook 固定的 `source_sha256`；展开源码必须匹配固定的 `SOURCE_MANIFEST.json` SHA256；
- progress 展开目录必须直接（或在 Dataset 根目录下）包含 `workers/ippo/` 和 `workers/vrpo/`；progress ZIP 必须包含这两个目录；
- 找不到源码、找到多个源码候选、或找到多个 progress 候选时都会停止并列出候选，绝不会按 Dataset 名称、修改时间或“最新”猜测；
- 找不到 progress 时会停止并提示；如果确实要做无 checkpoint 的 fresh seeded 短测，必须把 `PROGRESS_BUNDLE = None` 明确写出来，避免误把随机初始化结果当成训练权重性能；
- 也支持显式填写路径，但不需要为了避免填错而手动填写。

Notebook 输出中的 `INPUT_DISCOVERY` 会显示最终选择的绝对路径和 `source_mode/progress_mode`。

没有本地报告时可不填两项，但只诊断、绝不推荐release。
预算验证拒绝0、负数、NaN、Infinity和超过600秒；包括bootstrap与Torch初始化；为清理和写报告预留时间。
停止时间受操作系统终止/磁盘延迟影响，不承诺硬实时纳秒截止；到期立即取消计算，不通过重设阶段时钟延长预算。
远端单卡和双卡顺序在同一会话比较；每个算法相同任务和seed；单卡串行执行全部shards，双卡并发拆分同一集合。
没有两张真实T4或不在Kaggle，就不能通过remote gate。

Notebook不安装依赖、不启动长训练；最后只打包报告/日志，不重复压缩源码或权重。
下载显示的 `guandan-performance-<id>.zip` 后停止不用的GPU会话。
平台强制回收/硬杀可能来不及生成ZIP，不能保证中间文件被平台永久保存。
远端验证未通过就留在本地继续查原因，不能自动提升预算或转入长跑。

## 5. 限制

- 这次仅增加验证入口；环境优化和算法改动不在本写集。
- 每项只有短样本一次对照，包含冷启动；显著收益后还应检查多次波动，不能外推整个训练收敛速度。
- 单GPU评估比CPU快与否，以端到端wall为准，不以显存占用或利用率伪造收益。
- 训练用极短 fresh run，即使提供bundle也不续训旧checkpoint；原模型评估可用bundle。
- 不提供“跳过校验就通过”的开关，不凭空填写测试/远端结果。


## API 与手动 smoke

```python
from guandan.performance_gate import Config, run
# 默认不执行，先写计划
run(Config(mode="local"), "runs/my-plan", execute=False)
# 需要小smoke时显式缩小，不把结果当正式性能结论
run(Config(mode="local", budget_seconds=120, batch_size=8,
           inference_batches=4, rollout_envs=1, rollout_steps=2,
           updates_per_shard=1), "runs/my-smoke", execute=True)
```

```powershell
python -B -m guandan.performance_gate --mode local --execute --budget-seconds 300 --rollout-envs 8 --rollout-steps 128 --updates-per-shard 2 --output runs/local-gate
```

优化前后外部规则benchmark JSON（例如24局逐局trace对照）可以作为人工审查附件，但并非本入口result.json schema，不能把它当作自动release的完整硬件/seed/模型证据。自动门禁只接受本入口的同schema参考/优化报告，不猜字段补造缺失元数据。
本机有人正在做GPU基准时，不要同时启动本入口GPU测试，避免竞争污染结果。
