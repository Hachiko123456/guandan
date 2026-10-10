# Kaggle T4×2 DDP 运行手册规范

## 1. 交付根目录

所有新交付统一放在：

`D:\project\kaggle_train\guandan-marvel-t4x2-<YYYYMMDD>`

源仓库不得保存 Kaggle output 大文件。

## 2. Dataset 形态

推荐一个 Dataset 包含：

- source ZIP；
- 可选的恢复 bundle；
- deals/selection manifest；
- 不可变的输入 checkpoint。

Notebook 必须自动发现 Kaggle input mount，并按 manifest/hash 确认唯一输入；发现多个冲突候选时失败并给出列表，不猜路径。

## 3. 双 GPU

正式训练使用：

```text
torchrun --standalone --nproc_per_node=2 -m <training-entry> ...
```

要求：

- rank 0 -> cuda:0；
- rank 1 -> cuda:1；
- 两个 rank 训练同一个模型；
- 每 rank 独立 env seed；
- DDP 同步梯度；
- 只有 rank 0 进行 promotion、主日志和持久打包；
- rank 日志仍分别保存以便诊断。

## 4. Notebook 用户体验

用户只负责：

1. 上传 Dataset；
2. 导入 Notebook；
3. 打开 T4×2；
4. 点击 Run All。

不得留下必须手改的绝对路径、checkpoint 名、source hash 或设备参数。需要选择时由 manifest 或自动发现决定；无法唯一决定时明确失败。

## 5. 运行模式

Notebook 必须提供固定的模式入口，不要求用户改代码：

- `preflight`：安装/导入/设备/数据检查；
- `pilot`：短训练和短评估；
- `resume`：从输入 bundle 恢复；
- `full`：使用已由 pilot 确认的配置正式训练；
- `eval_only`：只评估冻结 checkpoint。

模式通过单一配置 cell 或 Dataset manifest 选择。最终交付 notebook 默认模式必须在说明中明确，避免误启动长跑。

## 6. 必需日志

```text
console.log
metrics.jsonl
progress.json
telemetry.jsonl
resolved_config.json
source_manifest.json
runtime_environment.json
checkpoint_manifest.json
champion_manifest.json
evaluation_summary.json
```

`progress.json` 至少包括：iteration、completed games、env/game steps、当前阶段、吞吐、ETA、最近 checkpoint、最近评估和 heartbeat 时间。

`telemetry.jsonl` 至少每 5-10 秒记录：GPU 利用率、显存、功耗（可用时）、CPU、RAM、磁盘和进程状态。

## 7. 恢复与打包

- checkpoint 只能在完整 learner update 边界标记 durable；
- 中途 rollout 不能计为完成；
- recovery manifest 使用相对 basename 和 SHA-256；
- `/kaggle/input` 只读；
- `/kaggle/working` 下使用唯一 output root；
- 正常结束、预算停止、异常退出都尽力生成 partial bundle；
- 平台硬杀不能承诺 finally 执行，应通过周期性 durable checkpoint 降低损失。

## 8. Pilot 门槛

M08 必须先测：

- 单卡/双卡相同全局工作量；
- DDP speedup；
- rollout 和 learner 分段耗时；
- CPU 环境供给；
- GPU 利用率分布；
- 显存峰值；
- 预计 full run 完成量；
- checkpoint/eval 开销。

双卡目标至少 1.5×；未达到时先诊断 CPU、通信、batch 和 compile，不通过制造无效计算提高利用率。

## 9. 输出验收

下载 output 后必须验证：

- ZIP 完整性；
- manifest hash；
- checkpoint 可加载；
- 两张 GPU 都真实参与；
- 日志无缺口或明确标出缺口；
- 实际完成计数与报告一致；
- 评估基于冻结 checkpoint；
- 任何 incomplete 不标 complete。
## 10. M07 实现绑定（2026-10-10）

M07 canonical delivery schema 为 `guandan-marvel-kaggle-delivery-v1`，Dataset/source/DDP checkpoint/recovery/progress/telemetry 均有独立版本。生成器、五模式 Dataset、Run-All Notebook、中文说明、fail-closed discovery、两 rank worker、周期 durable checkpoint 和 recovery binary 规则见 `M07_KAGGLE_DDP_PROTOCOL.md`。

阶段 `accepted` 的语义仅为 `remote_ready`：本地 package validation、Notebook 语法、离线 source build/import、CLI 和 CPU/gloo 双进程恢复通过。没有伪造 `/kaggle`，没有自动上传，真实 T4×2 仍由 M08 验证。
## 11. M07 集中补修绑定（2026-10-10）

补修后，bounded evaluation 使用 `Rank.TWO=15` 并在 spawn 子进程加载真实 C++ policy；durable checkpoint/index/recovery 必须先于评估发布。Resume 从已验证 rank shard 继承全部训练语义，profile 只覆盖追加迭代数和会话控制。预算停止由所有 rank collective 形成共同决定。Notebook 每 5–10 秒输出 progress/GPU/rank heartbeat，首个 iteration 前未知计数不得伪造为 0。完整协议见 `M07_KAGGLE_DDP_REPAIR_PROTOCOL.md`。
## 12. Kaggle 自动解包兼容性（2026-10-10）

Dataset source 和 recovery 均支持 ZIP 保留或 Kaggle 自动解包后的目录形态。Notebook、discovery 和 runner 复用同一个 stdlib bootstrap resolver，以 manifest/content identity 判定输入，不按目录名或第一个匹配项猜测。ZIP 存在时验证真实 archive bytes；ZIP 不存在时只声明并验证 content manifest/逐文件身份。完整协议见 `M07_KAGGLE_INPUT_LAYOUT_REPAIR_PROTOCOL.md`。
