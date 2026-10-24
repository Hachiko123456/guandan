# Kaggle T4×2：v1.4 第一轮学习效果实验

日期：2026-10-21。范围：上传交付、目标设备预检、最多绝对 50 iter/1000 局、训练前后开发评估。**本地验证不等于 Kaggle T4×2 已通过；第一次运行本身会验证远端。**

## 当前上传入口：v3 传输修复（2026-10-21）

v2 在输入校验阶段出现源码 ZIP 缺失/改变类错误，尚未启动训练；日志没有实际文件清单，具体平台处理方式未定。

**请使用 `runs/t4x2_pilot_delivery_20261021/delivery_v3/`：只上传 `pilot_payload_v3.bin` 作为 Dataset，另行导入 `guandan_t4x2_pilot_v3.ipynb`。不要上传ZIP或整个文件夹，不要继续用v2 Notebook。** v3采用有固定哈希的Base64传输容器，在工作目录中还原逐字节一致的v2输入，训练源码、超参、权重和数据身份均未变。

新增输入诊断保存在 `/kaggle/working/guandan_t4x2_pilot_v3_bootstrap/`。下面关于 ZIP 上传的旧步骤保留为 v2 历史说明，不再作为当前上传操作；训练/性能/恢复原理仍适用。

## 1. 上传什么？

构建输出目录里有：

- `pilot-upload.zip`：创建 Kaggle Dataset 时上传。包含冻结源码、离线 pybind11、M09 warm-start 权重、三段 profile、开发发牌和 manifest；不包含 FableDan 权重。
- `guandan_t4x2_pilot.ipynb`：导入为 Kaggle Notebook，Add Input 加入上述 Dataset。
- `README_KAGGLE.md`：本说明的交付副本。
- `LOCAL_VALIDATION.json`：本地检查和明确未运行的项目。

Notebook 的 Settings/Accelerator 必须选 **GPU T4 ×2**，不要只选单卡。Internet 可关闭，不需要 API key、W&B 或 pip 下载。运行时检查 Python>=3.12、两张 T4、CUDA/NCCL；不满足就失败，不改用 CPU。

如果有多个同名交付 Dataset，明确设置代码开头的 `DATASET_DIR`，不要让程序猜用哪一份。源码来自当前 dirty 工作区的内容快照，manifest 记录 source hash，不拿旧 commit 充当完整源码身份。

## 2. Run All 实际执行什么？

```text
输入 hash 与环境检查 → 离线构建当前 C++ 后端
  → 2 iter / 40 局，保存完整双 rank 状态
  → 从真实压缩恢复包恢复到第 3 iter / 60 局
  → 预检通过后继续到绝对第 50 iter / 1000 局
  → 释放训练进程，两个 GPU 分片执行开发评估
  → HTML、JSON、逐局结果、日志、恢复包、结果 ZIP
```

**2+1+47=50，不是先额外训练若干局再重头训练 50 次。** 总体安全预算 12600 秒（3.5 小时，包含源码准备阶段的已计时工作），训练前预留约 1 小时评估和 10 分钟保存/收尾。预算不足会保存已完成边界并报告未完成，不把计划数补成实际完成数。硬件卡死/外部强制终止不能保证最后一步保存，所以定期下载已有输出。

不自动继续 250/500/1500，不运行 final holdout，不启动 FableDan 对战，不把 rule 胜率当作超过 FableDan 的证明。

## 3. 冻结了什么学习配方？

- 从 M09 iter2000 **policy-only warm start**，optimizer/sampler/RNG/计数为新任务，不冒充旧训练 exact resume。
- v1.4-macro-fuse，history_dim=128、feature_dim=256、4 层/4 头，2783234 参数的目标模型。
- 宏决策 IPPO + 宏归一化 KL，沿用 base clip=0.02、reg=0.1、lr=4e-4 与现有 schedule；本交付不采用未经验证的 14 项批量改参提案。
- global collect=20，epochs=4、mini_batches=8，保留当前 52 上下文 7A schedule 与 warm-start curriculum。
- NTP/Belief/对称性/Pool/Search 关闭，贡还不在此次训练与评估分布内。
- 50 iter 只是早期信号实验；课程从旧上下文逐步混入目标分布，不能仅凭计划中有 52 行就声称每个上下文已充分训练。报告中保留每 rank 的实际覆盖。

## 4. 怎样利用两张 T4？

本交付使用显式、可追溯的性能 preset：

1. `torchrun --nproc_per_node=2`，一张卡一个 rank，NCCL 同步一个共享策略，不是两卡分别训练两个模型。
2. learner 启用 FP16 autocast + GradScaler，概率比/GAE/关键损失保持 FP32；没有因 T4 不支持某功能而静默改为 CPU。
3. global rollout 从先前 builder 的 4 改为 **16（每卡 8）**，global collect 仍 20，不增大每轮学习局数。并行数改变可能影响采样顺序，作为新版本性能 profile 明确记录。
4. **保留 4096 合法轨迹上限**，只在完整轨迹返回与 KV cache 重建时去掉无效尾部 padding。不是 horizon 改成 1024，也不丢弃长局。
5. 每 rank 默认 1 个 BLAS/OpenMP 线程，避免 CPU 超额争抢；不强行设置 NCCL P2P 开关，不假设存在 NVLink。
6. 编译关闭：短试验与动态有效长度不盲目支付 torch.compile 开销。
7. 训练退出后，两个 CUDA 评估 worker 各负责一半固定任务，避免只用一张卡评估。

### 显存预算不是“CPU RAM 充足”

T=4096、标准基础字段的单轨迹为 91,381,768 bytes：

```text
每 rank sampler 槽位=(2*16+20)/2=26
基础 sampler=2,375,925,968 bytes≈2.213 GiB / GPU
FP32 静态 KV cache 约 0.5 GiB / GPU（按当前模型、每 rank 8 环境）
```

这些不包括 learner 激活、参数/optimizer、临时副本和通信。sampler 预算守卫仍为 4 GiB，不为了跑通扩大守卫；远端 precheck 记录实际峰值。裁掉 padding 会减少大量无效前向与复制，但**没有实测前不承诺加速倍数或始终 100% GPU 利用率**。

自回归逐 token 环境交互、规则计算、小模型、checkpoint IO 都可能使利用率低于 100%。报告会显示 rollout/learner 分时与两卡利用率，下一轮据实际瓶颈再调，不能把把显存塞满当性能目标。

## 5. 训练结束怎么判断是否有效？

固定独立开发发牌 32 份，每份做四个座位轮换，三个 arm：

| arm | 局数 | 回答的问题 |
|---|---:|---|
| 训练前 control 对 rule | 128 | warm-start 起点表现 |
| 训练后 candidate 对 rule | 128 | 相同发牌上是否改善 |
| candidate 对 control | 128 | 是否直接优于未继续训练的策略 |

共 384 局，两个 GPU 各 192 局。control 是当前源码/当前 native 下迁移后、优化前冻结的策略，不绕过跨平台旧 checkpoint 的身份守卫。它与 candidate 绑定同源、同架构；所有 arm 的实际发牌 hash、rotation、权重 hash 和温度均核对。

数据域是 development/selection，final=0；评估发牌 seed 与实际训练来源分离，还比对物理 deal hash。此集被用于观察和决策，不冒充最终盲测。

统计按独立 deal 聚类，rule 前后比较使用 paired difference；样本不足时 `INCONCLUSIVE`，不把 50 次没显著提升当作架构失败。只有两项预先定义的区间条件都支持改善才报告 `EARLY_IMPROVEMENT_SUPPORTED`，仍**不能声称比 FableDan 强**。

## 6. 应该下载哪些结果？

首选下载：`guandan_t4x2_pilot_results.zip`，Notebook 提供链接。它不重复收录巨大 raw checkpoint shards，但包括最近的完整压缩恢复包。

主要内容：

- `report.html` / `REPORT.md` / `pilot_result.json`：结果、完整性、训练曲线、性能摘要与未验证范围。
- `training_history.jsonl`：跨三段的完整训练历史，正常应有 50 行有效 iteration。
- `telemetry.jsonl` / `performance_summary.json`：每 5 秒 GPU 利用率、显存、CPU/RAM/磁盘、各阶段信息。
- `*.console.log` / `events.jsonl`：原始启动命令、NCCL/进程输出、耗时、错误和超时。
- 各阶段 `ranks/rank00*/pilot_runtime.json`、`pilot_iterations.jsonl`、`diagnostic.jsonl`、`trainer/ippo_metrics.jsonl`：每卡设备/通信、有效更新、AMP scale/跳步、显存峰值、loss/梯度/宏统计。
- `evaluation/games.gpu*.jsonl`、worker 状态、summary、development manifest：可复算的逐局对照。
- `pilot_initial_policy.pt`、`candidate_iter000050.pt`：训练前和训练后评估策略。
- `latest_recovery.zip`：最近完整双 rank checkpoint、optimizer/sampler/RNG/scaler/native 与初始 control/manifest 绑定。**不是只有权重**。恢复包也保留此前的训练指标与 rank 日志，跨会话仍能重建完整学习曲线。

正常应看到 `status=complete`、迭代50、完成局数1000、两rank均有有效更新与非零峰值，评估384局完整；若不是，按 `failed/budget_stopped` 和实际计数分析，不能仅看 Notebook 返回了结果文件。

## 7. 会话中断后如何恢复？

1. 下载最近可用结果 ZIP，取出 `latest_recovery.zip`，上传到新的 Kaggle Dataset。
2. 保留并同时挂载**同一原始 pilot Dataset**，不能换成重新构建的新源码包。
3. 新 Notebook/新工作目录，设置 `RECOVERY_ZIP = Path('/kaggle/input/.../latest_recovery.zip')`。
4. 程序核对原 source/archive/config contract，恢复原 Linux native 二进制而不重新编译改变身份，继承同 world size=2 的状态。
5. 未到50只补足到绝对50；已经达到50则只恢复评估策略、补跑开发评估，**不再执行任何训练更新**。超过50的恢复点直接拒绝。

若训练已到50但评估中断，使用同一入口与同源 recovery，只补跑开发评估；新会话仍有预算上限，不重新训练，也不消耗 final holdout。

启动前构建失败：查看根目录的 `native_build.log` 与 Notebook 异常。训练中失败：先查对应 console/rank diagnostic，结果仍尽力保留最后原子 recovery。平台强制断电不等于所有最后写入都已持久化。

## 8. 本地验证与远端未运行范围

本地包含纯张量裁剪/GAE/安全与流程测试、CUDA 下 padded-vs-trimmed 输出/梯度/缓存后续输出比较、生产模型尺寸但小批量/horizon512 的两次实际更新与 exact resume。

**本地没有双 T4，也未远端运行此50次实验。** NCCL、完整 batch/horizon4096 的峰值、真实吞吐/利用率、棋力改善由上传后产生的证据决定。精确命令、通过数量和源码 hash 见交付中的 `LOCAL_VALIDATION.json`。
