# M07 Kaggle T4×2 DDP 交付协议

- 协议日期：2026-10-10
- 阶段：`M07_kaggle_t4x2`
- 交付状态语义：`remote_ready`，不是 `remote_verified`
- 目标 GPU：Kaggle `T4 × 2`

## 1. 交付形态

生成器 `scripts/build_marvel_kaggle_delivery.py` 只读取仓库 source snapshot，不上传 Kaggle。正式输出位于：

`D:\project\kaggle_train\guandan-marvel-t4x2-20261010`

目录包含一个 Run-All Notebook、中文说明、总 manifest，以及五个互斥 Dataset 目录：`preflight`、`pilot`、`resume`、`full`、`eval_only`。用户只上传一个模式目录；同时添加多个模式时 Notebook 必须因多个 `guandan_marvel_dataset.json` fail closed。

`full` 不是默认模式。只有用户明确选择 `datasets/full` 创建 Dataset 后才会启动长训练。

## 2. Source ZIP

Schema：`guandan-marvel-source-manifest-v1`。

Source ZIP 必须记录：

- 完整 Git commit；
- workspace snapshot 标记；正式交付必须为 `false`；
- 每个文件的相对路径、字节数、SHA-256 和来源；
- `guandan_marvel`、`nanopokerlib` 与离线 pybind11 构建依赖的许可证；
- source manifest 自身 canonical hash；
- `uploaded=false`、`remote_verified=false`。

正式生成器从 Git commit 读取仓库文件，不从未提交工作区静默取文件。pybind11 作为版本化、逐文件 hash 的离线构建依赖打包，避免 Notebook 依赖网络安装。

## 3. 输入发现

Dataset schema：`guandan-marvel-kaggle-dataset-v1`。

Notebook 和 runner 必须：

1. 在 `/kaggle/input` 发现且只发现一个 Dataset manifest；
2. 使用 manifest 的相对路径定位 source ZIP 和外部 source manifest；
3. 验证 archive、manifest 和逐文件 SHA-256；
4. `resume`/`eval_only` 自动发现与 source commit/archive hash 同时匹配的唯一 recovery ZIP；
5. 多个匹配 recovery、路径越界、hash 不符、schema 不符时拒绝运行；
6. 不猜绝对路径、checkpoint 文件名、GPU 参数或输入优先级。

## 4. DDP 语义

远端训练固定使用：

```text
torchrun --standalone --nproc_per_node=2 -m guandan_marvel.kaggle.worker ...
```

硬约束：

- `WORLD_SIZE=2`，rank/local rank 只能为 0/1；
- rank 0 -> `cuda:0`，rank 1 -> `cuda:1`；
- backend=`nccl`，`require_cuda=true`；
- 两 rank 创建同架构模型，DDP 初始化广播并同步梯度；
- global rollout/collect batch 必须可被 2 整除，每 rank 只分配其本地份额；
- 环境 seed 为 `base_seed + rank * 1000003`，模型初始化 seed 保持一致；
- durable checkpoint 发布时验证 resume actor、merged actor（及 VRPO critic）跨 rank state hash 相同；
- rank 0 发布主 metrics/progress/checkpoint/evaluation/recovery；每 rank 保留诊断 JSONL 和精确恢复 shard。

本地 CPU/gloo 双进程检查只验证 DDP 同步和恢复语义，不是 Kaggle/GPU 证据。Windows 本机 PyTorch elastic store 缺 libuv，因此本地测试以等价 RANK/WORLD_SIZE 环境启动两个 worker；正式交付仍只使用 torchrun。

## 5. Durable checkpoint

DDP envelope schema：`guandan-marvel-ddp-checkpoint-v1`。

每个 durable 目录包含：

- `manifest.json`；
- `rank000.pt`、`rank001.pt`；
- 对应 rank summary。

rank shard 是 MARVEL training checkpoint v5，保存该 rank 的 sampler/RNG/replay 和训练状态；rank 0 原子发布 envelope manifest。Manifest 保存：world size、同 world size exact-resume 策略、rank shard hash、跨 rank 模型 hash、global counters、semantic/source/backend identity 和聚合训练 provenance。

World size 变化不得静默恢复；当前策略为 `exact_same_world_size_only`。

聚合 provenance schema：`guandan-marvel-ddp-training-data-provenance-v1`，保存所有 rank 的初始 seed 和 deal hash 并参与 selection/final 数据重叠检查。

## 6. Recovery bundle

Recovery schema：`guandan-marvel-recovery-manifest-v1`。

周期 checkpoint 后更新 `guandan-marvel-recovery.zip`，包含：

- 最近 durable DDP checkpoint；
- checkpoint/champion/evaluation/resolved config/progress/source/runtime manifests；
- 当前会话实际加载的 native extension binary 及 SHA-256；
- source commit、source archive hash 和 checkpoint manifest hash。

新 Kaggle 会话优先恢复 recovery 中的原 binary，保持 M05 backend/build identity exact；没有 recovery binary 时才从 source ZIP 离线构建。可捕获异常和正常结束会尽力重打 recovery；平台硬杀不承诺 finally，因此依靠周期 durable checkpoint 降低损失。

## 7. 模式

- `preflight`：source/C++/两张 T4/NCCL/输入检查，不训练；
- `pilot`：短 IPPO DDP、每 iteration checkpoint、有限 random/rule 管线评估；
- `resume`：唯一 recovery、相同 world size、兼容配置恢复后追加短训练；
- `full`：用户显式选择后使用基线 v1.3 配置长训练；参数仍待 M08 pilot 定型；
- `eval_only`：读取 recovery checkpoint，仅运行冻结策略有限评估。

M07 有限评估只证明 E0 管线，`formal_selection=false`、`final_holdout=false`、`promotion_eligible=false`，不得作为强度结论。

## 8. 必需输出

每个 run root 必须有：

- `console.log`
- `metrics.jsonl`
- `progress.json`
- `telemetry.jsonl`
- `resolved_config.json`
- `source_manifest.json`
- `runtime_environment.json`
- `checkpoint_manifest.json`
- `champion_manifest.json`
- `evaluation_summary.json`

训练模式还必须有 `ranks/rank000/diagnostic.jsonl` 和 `ranks/rank001/diagnostic.jsonl`。Telemetry 周期默认 5 秒，GPU/CPU/RAM/磁盘不可用字段必须显式标为 unavailable/partial，不以 0 冒充。

## 9. 验收边界

M07 接受只表示：

- source ZIP、Notebook、manifest、中文说明本地可生成并验证；
- CLI、离线 source build/import、CPU 双进程 DDP 和 durable resume 小测通过；
- 包已可由用户上传并在 T4×2 Run All。

它不表示真实 Kaggle kernel 已执行，不表示两卡 speedup、显存、吞吐、ETA 或模型强度通过。这些证据属于 M08。
