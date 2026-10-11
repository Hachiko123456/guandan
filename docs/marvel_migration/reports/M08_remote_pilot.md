# M08 Remote Pilot 与参数定型最终验收报告

- 日期：2026-10-11
- 分支：`codex/marvel-guandan-migration`
- 最终实现 commit：待本次原子提交记录
- 阶段状态：`accepted`
- 真实远端：Kaggle T4×2

## 范围

本阶段完成真实 Kaggle preflight、IPPO pilot、durable checkpoint、iteration 2→4 recovery/resume、同 workload single/dual GPU control，并锁定 M09 基线配置。未运行 final holdout，未作模型强度结论。

## 真实远端证据

证据根目录：`runs/M08_remote_pilot/remote/`。

- `pilot_t4x2_20261010`：2 iterations、4 games、两 rank、evaluation/checkpoint/recovery 完成；
- `resume_t4x2_20261010`：从 iteration 2 恢复至 iteration 4，`resume_inherited=true`，native binary 从 recovery 恢复且跳过重编译；
- `single_gpu_control_d6319d2_20261011`：world size 1、只使用 cuda:0；
- `dual_gpu_control_20261011`：world size 2、NCCL、两张 T4 均有真实计算；
- `runs/M08_remote_pilot/benchmark_comparison_final.json`：严格同 workload benchmark 比较。

## 硬门槛结果

- 两张 Tesla T4、CUDA 12.8、PyTorch 2.11.0+cu128、NCCL：通过；
- DDP 同一模型：通过，rank state hash/gradient norm 一致；
- 完整 trajectories：通过；
- illegal/truncated/nonfinite/action/horizon health：单卡和双卡均为 0；
- durable checkpoint 与 recovery 下载/校验：通过；
- iteration 2→4 remote resume：通过；
- single/dual workload hash：一致；
- training-core speedup：`1.5165367203x`，达到 `>=1.5x`；
- session speedup：`1.5416364735x`，达到 `>=1.5x`；
- scaling efficiency：`0.7708182368`。

## 性能

Single control：

- measured games：8；
- training core：35.257936705s；
- core games/s：0.2268992672；
- GPU0 utilization mean/median/max：68.30%/77%/100%；
- GPU0 memory peak：13349 MiB；GPU1 idle。

Dual control：

- measured games：8；
- training core：23.248983182s；
- core games/s：0.3441010705；
- GPU0 utilization mean/median/max：52.05%/61%/100%；
- GPU1 utilization mean/median/max：52.49%/61%/99%；
- memory peaks：GPU0 6713 MiB、GPU1 6815 MiB。

## M09 锁定配置

配置文件：`configs/marvel_m08_locked_config.json`。

- 主线 baseline：IPPO；
- 参考模型：MARVEL v1.3，128/256，4 decoder layers；
- global rollout/collect：4/4；per-rank 2/2；
- mini-batches：2；agent epoch：4；
- AMP/compile/Muon：M09 初始仍关闭，后续需单独实验；
- checkpoint interval：25；evaluation interval：100；
- checkpoint retention：最近 2 个；evaluation retention：最近 2 个；
- full wall budget：42000s。

新增 retention 解决 full 长跑会永久累积大 rank checkpoint、耗尽 Kaggle 工作盘的问题；output ZIP 排除 checkpoint shard 和 recovery ZIP，recovery 仍单独发布。

M09 仍必须以相同完成对局预算比较 IPPO/VRPO，不能把 M08 的 2-iteration pilot 强度当作算法结论。

## 限制

- 训练数据当前仍需 M09 审查级牌/贡还覆盖；
- M08 bounded evaluation 不是 formal selection、champion promotion 或 final holdout；
- 模型强度尚未建立；
- M09 必须比较 IPPO/VRPO 并完成正式 selection/champion/final holdout。

## 结论

M08 所有真实远端硬门槛和双卡 speedup 门槛通过，本阶段可以 `accepted:true`；M09 可以启动，但必须使用本次锁定配置和 checkpoint retention 修复后的新 full 交付，不得复用旧 full Dataset。
