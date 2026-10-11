# M08 单卡/双卡同工作量性能对照交付报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 状态：本地实现/验证完成，等待真实 Kaggle T4×2 单卡与双卡两次远端运行
- M08：仍为 `ready / accepted:false`

## 已有真实远端前置证据

- Kaggle T4×2 preflight 已通过；
- IPPO T4×2 pilot 两次 durable update 已通过；
- recovery iteration 2 -> 4 真实恢复已通过；
- 下载证据位于 `runs/M08_remote_pilot/remote/`。

## 本轮交付

新增两个 versioned Dataset 模式：

- `single_gpu_control`：在 T4×2 Kaggle 环境中只启动 world size 1 / cuda:0；
- `dual_gpu_control`：同机 world size 2 / cuda:0+cuda:1 / NCCL DDP。

二者固定相同：IPPO、正式 v1.3 128/256 模型、global rollout/collect batch=4、mini_batches=2、agent_epoch=4、seed、完整 trajectory、3 iterations、无周期 evaluation，仅末次 durable checkpoint。Iteration 1 为 warmup，2-3 为计时集合。唯一训练执行差异为 world size。

为了让 world-size 1/2 的初始全局环境流可对照，训练配置显式版本化 `environment_rank_seed_stride=2` 与 `guandan-marvel-m08-partitioned-prng-v1`：单卡四 slot 使用 base+0..3，双卡 rank0 使用 base+0..1、rank1 使用 base+2..3。

每次运行生成 `benchmark_summary.json`，记录 workload hash、warmup/measured iterations、training-core 时间、完整局数、env/game steps、吞吐和健康指标。`scripts/analyze_m08_benchmark.py` 对两个 summary 做 fail-closed 比较并计算 session/core/throughput speedup 与 scaling efficiency。

## 本地验证

- 单进程 CPU control launcher、checkpoint 和 benchmark summary：通过；
- 双进程 gloo control launcher、checkpoint 和 benchmark summary：通过；
- 两个 control 的 workload hash 一致；
- 本地 reduced profile 只证明 launch/checkpoint/metric 语义，不作为 GPU 性能证据。

## 测试

- `tests/marvel_kaggle`：37 passed；
- Kaggle + training/core/evaluation + A00 非重叠目标集合：129 passed；
- 0 failed、0 errors、0 skipped；
- `git diff --check`：通过。

## 远端执行要求

用户必须在相同 Kaggle T4×2 环境分别运行 single 和 dual Dataset，下载 output/log。M08 只有在两份 `benchmark_summary.json` 身份一致、健康指标为 0 且 speedup 门槛审查完成后才能 accepted。不得开始 full/M09。

## 远端双卡首次运行缺陷与补修

真实 single control 运行完成后发现正式 checkpoint 约 1.26 GiB。首次 dual control 仍按旧 benchmark contract 在末次迭代保存两 rank checkpoint，并在 run root、recovery、output ZIP 和 Kaggle Results 中重复压缩，导致训练完成后长时间 CPU 打包、Logs 页面无新输出且 Output 仍显示 0 B。性能对照已经有独立 pilot/resume 恢复证据，不需要再次保存训练 checkpoint。

补修后两个 benchmark contract 均声明 `persist_checkpoint=false`、`emit_recovery=false`；worker 不进入 checkpoint boundary，runner 不生成 recovery，仍生成 benchmark summary、metrics、telemetry、progress 和 output ZIP。Local single/dual reduced smoke 真实验证没有 `checkpoints/` 或 recovery ZIP，并正常结束。旧性能 Dataset/Notebook 不得继续重跑。
