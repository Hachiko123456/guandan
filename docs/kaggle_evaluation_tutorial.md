# Kaggle 模型评估教程：先预跑估时，再决定正式局数

## 1. 这次不需要重新训练

你已经有 IPPO、VRPO 的 update100/update102 权重。本次交付的 `guandan_evaluation.ipynb` **只评估已有权重**，不调用训练器、不更新模型、不自动寻找旧 progress。

训练源码和评估源码可以是不同提交；**必须配套的是本次评估源码 ZIP 与本次评估 Notebook**。checkpoint 能否读取由评估加载器校验，不要为了匹配 Git commit 重新训练。

## 2. 哪些文件上传，哪些文件留在本地

本次最终交付目录为 `C:/Users/yhx/Downloads/guandan-evaluation-optimized-final-20261007`。请只上传该目录中的最终文件，不使用已经删除的旧 warmup/intermediate 目录。

| 文件 | 操作 |
|---|---|
| 本次评估交付目录的 `guandan-source.zip` | 上传为私有 Dataset（建议名 `guandan-source-evaluation`） |
| 已下载的 `guandan-dual-gpu-progress.zip` | 上传为另一个私有 Dataset（建议名 `guandan-training-progress`） |
| 本次 `guandan_evaluation.ipynb` | 作为 Notebook 导入，不作为训练数据 |
| `PACKAGE_METADATA.json` | 本地保留，用于核对来源与 SHA256 |
| `使用说明.txt` / `kaggle_evaluation_tutorial.md` | 本教程，本地阅读即可 |

不要挂载旧的 `guandan-unified-progress.zip`。也不要把评估输出 ZIP 当作训练 progress。

导入 Notebook 后，将两个 Dataset 添加到 Input，展开 Session options 并选择 **GPU T4 x2**。Notebook 明确分配 IPPO→cuda:0、VRPO→cuda:1；选择两张卡不代表保证高利用率，应以日志中的 telemetry 和对局吞吐为准。

## 3. PROGRESS_BUNDLE 路径怎么填

**修正旧说明：既支持 ZIP 文件，也支持已经展开、直接包含 `workers/` 的目录。** 不必为展开的 Dataset 再打 ZIP。

### 情况 A：Input 中能看到 ZIP

示例（实际路径以你的 Input 为准，不要照抄 Dataset 名）：

```python
PROGRESS_BUNDLE = '/kaggle/input/guandan-training-progress/guandan-dual-gpu-progress.zip'
```

### 情况 B：Input 中只有展开目录

找到这样的结构：

```text
某个根目录/
  workers/
    ippo/
    vrpo/
  result.json
```

填写“某个根目录”，不是 workers 本身，也不是 ippo 子目录：

```python
PROGRESS_BUNDLE = '/kaggle/input/guandan-training-progress/guandan-dual-gpu-progress'
```

### 不知道路径时

临时增加一个 Code 单元运行以下代码。它**只列举，不自动选择**，也不加载 checkpoint：

```python
from pathlib import Path
base = Path('/kaggle/input')
candidates = [p for p in base.rglob('guandan-dual-gpu-progress.zip') if p.is_file()]
candidates += [p.parent for p in base.rglob('workers')
               if (p / 'ippo').is_dir() and (p / 'vrpo').is_dir()]
if candidates:
    for p in sorted(set(candidates)):
        print(p)
else:
    print('没有找到训练 progress：请检查 Input 是否已添加训练结果 Dataset，或展开目录查看。')
```

复制你要评估的那一份的完整路径。路径必须位于 `/kaggle/input`，不能是 Windows 下载路径，也不能填源码 ZIP。多份候选时必须自行明确选择，Notebook 不会猜“最新”。

## 4. 第一次运行：只预跑，不自动继续

```python
EXECUTE = True
SOURCE_DATASET_HINT = ''
INSTALL_DEPS = False
PROGRESS_BUNDLE = '上一步确认的真实路径'
EVAL_PROFILE = 'local_fast'
WARMUP_GAMES_PER_OPPONENT = 10
TARGET_GAMES_PER_OPPONENT = 100
AUTO_CONTINUE_AFTER_WARMUP = False
MAX_ESTIMATED_MINUTES = 60
BATCH_SIZE = 16
```

点击 **Run All**，按顺序执行配置、源码验证、评估、结果展示。不要跳过源码验证单元。

### “10 局”和“100 局”具体是什么意思

这里的局数是**每个算法、每类对手**，不是全套合计：

| 算法 | 四类评估对手 | 预跑局数 | 正式目标 |
|---|---|---:|---:|
| IPPO102 | random、rule、IPPO100、VRPO102 | 4×10=40 | 4×100=400 |
| VRPO102 | random、rule、VRPO100、IPPO102 | 4×10=40 | 4×100=400 |
| 合计 | 8 个任务；random/rule 不重复跑 | **80** | **800** |

预跑使用独立 seed，**不计入正式胜率**。10 不是 4 的倍数也可以，因为它只用于估时；正式目标必须是 4 的倍数，以完整覆盖四个座位轮换。

`EVAL_PROFILE` 选择运行时兼容/检查方式；**不会暗中覆盖 TARGET**。从 local_fast 改成 remote_full，目标仍然是你填写的 100，不会自动变成 1000。

### 正常结束是什么样

`AUTO_CONTINUE_AFTER_WARMUP=False` 时，预跑完成且预估在预算内时结果为（超预算则为 stopped_by_budget，也没有正式对局）：

```text
status: warmup_complete
```

这是**主动正常退出**，不是卡住、不是正式 complete。Notebook 不会通过 `input()` 等待确认，也不会在后台继续正式局数。查看预跑速度、预估正式时间后，再决定下一次参数。

两张卡并行时，全套 ETA 按两个 worker 中**较慢者**的预计剩余墙钟时间计算，而不是把两个 worker 的用时相加。启动加载、对手类型、batch 尾部和 CPU 竞争会影响速度，ETA 会更新，不是固定承诺。

## 5. 确认时间可接受后再运行正式评估

先保持 100 局/对手，不要直接扩大到千局：

```python
EVAL_PROFILE = 'remote_full'
WARMUP_GAMES_PER_OPPONENT = 10
TARGET_GAMES_PER_OPPONENT = 100
AUTO_CONTINUE_AFTER_WARMUP = True
MAX_ESTIMATED_MINUTES = 60
BATCH_SIZE = 16
```

重新 **Run All**。每次生成新 run_id、新输出目录，不覆盖旧结果；新的运行会重新预跑，然后根据预算判断是否继续正式阶段。它不是“接着上次 warmup 的第11局”，也不复用旧正式局数。

如果正式预计耗时超过预算，结果会标记 `stopped_by_budget`，不自动扩大预算。请先查看原因与实际已完成数量，再决定减少 TARGET、优化运行或主动调整预算。

100 局/对手完成且速度可接受后，才考虑将 TARGET 改成 1000（全套 **8000 正式局**）。预算由你据预跑估计填写，不要为了开始运行就设置极大的值。MAX_ESTIMATED_MINUTES 同时用于预跑后的准入判断和本次评估 worker 阶段的墙钟保护：预跑已用时间加预计正式时间超过预算时不继续；运行中达到预算也会停止并保留已完成记录（清理退出有几秒延迟）。300 秒没有 token/game 更新会触发卡死保护。项目预算不是 Kaggle 平台时限保证，外部强制停止仍可能发生。

若只想查看计划，设置 `EXECUTE=False`：只写 plan/config，不加载模型、不启动评估 worker。源码验证单元仍会校验/准备源码；这不等于执行对局。

## 6. 输出、日志、ETA 在哪里

启动时 Notebook 会打印 `RUN_ID`、`COMMAND`、完整配置、预跑/正式总局数，以及对应路径。

```text
/kaggle/working/guandan-evaluation-<run_id>/
/kaggle/working/guandan-evaluation-<run_id>.zip
/kaggle/working/guandan-evaluation-logs-<run_id>/notebook.log
```

结果根目录重点查看：

```text
evaluation_config.json
plan.json
result.json
input_manifest.json
run.json
live/progress.json
live/metrics.jsonl
live/events.jsonl
```

逐局 games（正式）和 warmup_games（预跑）分目录保存。每个任务先后生成 warmup_summary.json、formal_summary.json，以及对应的完整 evaluation JSON。每张卡的系统采样在 ippo/live/system_metrics.jsonl、vrpo/live/system_metrics.jsonl，结束汇总为 system_summary.json。环境/推理耗时位于任务 summary 的 metadata.timings；这是主机墙钟时间，不是纯 GPU kernel 时间。

日志应能回答：当前算法/对手/阶段、完成局数与目标、耗时、games/s、剩余时间、预估结束时刻、预算停止原因。Notebook 每 30 秒展示 live progress；若进度文件超过 120 秒未更新，会警告。**heartbeat 只是监视器活着，不代表游戏局数有推进。** 还没完成首批时可能显示 ETA unknown，不应编造速度。

GPU/CPU 数据以实际 telemetry 为准；`unavailable` 表示采集不可用，不能当作 0% 或已充分利用两张卡。比较不同机器、配置的速度时必须同时看任务数、对手和 batch size。

## 7. 如何判断结果状态

| 状态 | 含义 | 下一步 |
|---|---|---|
| `plan_only` | 未运行对局 | 检查参数后改 EXECUTE=True |
| `warmup_complete` | 预跑完成，正式评估未开始 | 查看 ETA 后决定是否自动继续 |
| `stopped_by_budget` | 预算保护停止，正式目标可能未达成 | 看原因、估计和已完成记录，调整下一次配置 |
| `complete` | 本次配置的正式目标完成 | 核查计数与 summary，再分析胜率 |
| `failed` / `incomplete` | 异常或未完成 | 下载日志、逐局记录，查具体错误 |
| 没有 result.json | 没有可靠最终状态 | 看进度和错误日志，不能认定完成 |

`returncode=0` 也可能只是 warmup_complete 或 plan_only，**必须结合 status 判断**。`accepted=false`、`kaggle_verified=false` 是独立验收标记，不等同于失败，也不能被当作完成证据。

正式分析先核对每类对手数量、座位轮换、非法动作/异常/截断，再看 team win rate 和按独立牌局聚类的置信区间。100 局结果只是初步判断；不能要求 update102 必然显著优于只差两次更新的 update100，也不能凭 loss 或一个胜率宣称完整掼蛋能力。

## 8. 停止、重新运行与“恢复”

### 预跑正常退出 / 预算停止

Notebook 的 `finally` 会尝试打包已写出的记录与日志。下载这次 `<run_id>.zip`，再修改配置运行；不要把 warmup 拼入 formal。

### 手动中断 / 出错

可捕获的异常或中断同样会尝试退出子进程并在 finally 打包。检查 working 目录里的 ZIP 和原始文件，保存之后再关闭会话。**硬杀、断电、平台立即收回运行环境无法保证 finally 或最终 ZIP 执行**；若 working 仍可访问，及时下载已写入的 games、live 和日志。

### 新会话重新评估

重新挂载**本次评估源码**和**原始训练 progress**，设置 PROGRESS_BUNDLE，重新 Run All。当前交付不提供逐局评估 resume 参数：每次是新 run，重新预跑/正式任务，保留旧结果作诊断，不拼接两次结果冒充一次完整评估。

输入权重是只读的，中断评估不意味着 checkpoint 损坏，**不要因此重训**。

## 9. 常见问题

- **找路径的代码没有输出**：旧代码只遍历 ZIP；如果 Dataset 已展开自然找不到。使用本教程同时列举 ZIP 和 workers 父目录的代码。
- **只运行 CONFIG 没输出**：它只是赋值；运行后续源码和执行单元，或 Run All。
- **源码 SHA 不匹配**：重新挂载与本次 Notebook 同目录发布的源码 ZIP，必要时用 SOURCE_DATASET_HINT 缩小搜索范围。不要修改 EXPECTED_SHA 来绕过。
- **旧入口不认识 warmup 参数**：挂载的源码不是本次发布版本，不能删掉参数继续全量跑。
- **已有旧结果**：本版每次用独立 run_id，不覆盖；不要手动把 OUTPUT_ROOT 改回固定旧目录。
- **预跑完成后没有 formal**：默认 AUTO_CONTINUE=False 的设计行为，查看 ETA 后显式开启即可。
- **跑了很久但只见心跳**：查看 live 更新时间、真实局数、worker 状态与错误，不要仅凭进程活着继续盲等。

## 附录：以后确实需要训练时，使用独立训练入口

当前评估不需要这一步。将来要训练时，应使用配套的训练交付文件 `guandan-source.zip` 与 `guandan_all_in_one.ipynb`，而不是修改评估 Notebook。

已有训练入口是 `python -m guandan.experiment`。训练 Notebook 的首次配置：

```python
EXECUTE = True
RESUME_BUNDLE = ''
RESUME_IPPO = ''
RESUME_VRPO = ''
BASE_UPDATES = None
RESUME_UPDATES = None
```

源码 Dataset 挂载后选择 T4 x2，Run All；训练输出为 `guandan-dual-gpu-progress.zip`。需要恢复时，在新会话挂载可兼容的训练 progress，明确填写 RESUME_BUNDLE（ZIP 或展开目录），不要猜旧包。恢复是否允许由训练语义契约检查决定。

**训练入口和评估入口是分开的。** 当前训练包已经包含双 GPU worker、checkpoint 语义校验、阶段进度、system telemetry、预算保护和中断状态记录；评估包只读取已有权重，不更新模型。训练完成后，再回到本教程用评估 Notebook 测量模型能力。训练包与评估包必须分别使用各自目录中配套的源码 ZIP 和 Notebook。
