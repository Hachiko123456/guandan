# 03 · 操作手册：从只读检查到安全恢复

[训练原理](02_training.md) · [下一篇：评估说明](04_evaluation.md) · [术语表](glossary.md)

> 适用范围：本地 Windows / PowerShell 与当前 v1.4 主线的操作边界。核对日期：2026-10-21。当前阶段与授权要求见 [CURRENT_EXECUTION.md](../marvel_v14_prompts/CURRENT_EXECUTION.md)。本篇给出命令不代表授权执行训练。

快速定位：[只读检查](#checks) · [dry-run](#dry-run) · [未执行 CUDA 示例](#cuda-example) · [恢复区别](#resume) · [正式准备](#production) · [故障排查](#troubleshooting)

## 1. 先区分操作成本与验证状态

| 操作 | 是否执行模型/训练 | 本轮文档工作验证状态 |
|---|---|---|
| Git、文件、Python/torch/CUDA 可见性检查 | 否；CUDA 查询不是模型计算 | 已执行基础查询 |
| 显式 C++ 模块导入 | 否；不证明完整规则/构建身份通过 | 已执行 |
| 指定 CPU 牌/组合单测 | 否 | 已执行，25 passed |
| Gate help / 明确 dry-run | 否；不创建 Trainer | 已执行 |
| 交付 builder help | 否；不构建交付 | 已执行 |
| 新环境安装/C++ 编译 | 会改环境/二进制，可能下载 | 未执行 |
| 本文 CUDA 更新/保存/恢复示例 | **是，会训练并写文件** | 未执行；仅静态语法/API 核对 |
| 生产尺寸/AMP/双 GPU 验证 | **是** | 未执行 |
| 正式 Gate、Kaggle、外部对战、final holdout | **是，高成本或消耗独立数据** | 未执行，需单独授权 |

没有可以跳过准备直接一路长训的“一键命令”。小验证、正式配方和目标设备认证必须分开。

## 2. 路径与解释器

以下针对本机现有工作区。克隆到其他位置时修改 `$Repo` 和 `$Py`：

```powershell
$Repo = 'D:\project\guandan'
$Py = Join-Path $Repo '.venv-gpu\Scripts\python.exe'
Set-Location -LiteralPath $Repo
if (-not (Test-Path -LiteralPath $Py)) { throw "Python environment missing: $Py" }
```

不要找不到 GPU 环境就自动改用系统 Python/CPU。CPU 规则、统计、明确的单位/兼容测试可用 CPU；真实模型训练/probe 使用当前入口指定的解释器与显式 CUDA。

### 全新克隆怎样准备？

先看 [pyproject.toml](../../pyproject.toml)、[setup.py](../../setup.py)、[nanopokerlib/README.md](../../nanopokerlib/README.md)：

- 要求 Python `>=3.12`；现有 native 文件还受实际 Python ABI/平台限制。
- 构建依赖包括 setuptools、wheel、pybind11；运行依赖见 pyproject，测试需要 pytest。
- CUDA 版 torch 应与目标设备/驱动匹配，依赖声明没有自动冻结全部 CUDA 环境。
- Windows 当前 builder 使用 `g++`、`gendef`、`dlltool`；装有 MSVC 不等于满足该 builder。
- 编译超时由 `GUANDAN_BUILD_TIMEOUT_SECONDS` 控制，代码默认 600 秒。

下面是**未执行**的安装示例，会改环境、可能下载/编译，仅在独立环境已准备且获准后使用：

```powershell
& $Py -m pip install -e '.[dev]' --no-build-isolation
```

`--no-build-isolation` 要求构建依赖预先就绪。现有 `.pyd` 可导入不能代替新源码构建验证；不要覆盖全局环境或在文档任务里擅自重编译。

<a id="checks"></a>

## 3. 只读检查

```powershell
git status --short
git rev-parse HEAD
Get-Content -LiteralPath (Join-Path $Repo 'docs\marvel_v14_prompts\CURRENT_EXECUTION.md')
& $Py --version
& $Py -c "import sys,torch; print(sys.executable); print(torch.__version__); print('cuda=',torch.cuda.is_available()); print('devices=',torch.cuda.device_count())"
```

确认：工作区正确、未提交文件已识别、最新入口可读、解释器符合预期。不要 reset/clean 来“方便安装”，也不要用旧 `project_status/STATUS.yaml` 决定 v1.4 阶段。单卡可见不证明双 T4/NCCL 或生产显存足够。

### C++ 后端是否可导入？

```powershell
& $Py -c "from guandan_marvel import get_game_backend; g=get_game_backend('cpp'); print(g.BACKEND); print(g.identifier); print(g.protocol_version)"
```

预期打印 `cpp`、对应标识和 V2 协议。失败就排查 ABI/扩展/依赖，不能用 Python 参考环境替代报告通过。成功也**不证明 `.pyd` 内嵌源码身份与当前 C++ 一致**；正式交付仍要做 native 身份核对。

## 4. 最小 CPU 规则测试

```powershell
& $Py -m pytest tests/unit/test_cards.py tests/unit/test_combos.py -q -p no:cacheprovider
```

只验证牌与组合，不构建模型、不更新参数。本次结果为 **25 passed**，不是全量验收或 C++ 差分测试。更多测试见[开发指南](05_development.md)，运行前检查是否涉及 GPU/DDP/长程压力测试。

<a id="dry-run"></a>

## 5. 用 dry-run 理解预算，不训练

```powershell
& $Py scripts/run_phase6_training_gate.py --help
& $Py scripts/run_phase6_training_gate.py `
    --gate 1 --completed-iterations 0 --max-iterations 2 --dry-run
```

预期 JSON：`target_iterations=50`、`remaining_iterations=50`、`max_iterations=2`、`status="dry_run"`、`will_train=false`、`formal_gates_not_run=true`。

- 上限 2 不把 Gate1 的绝对目标改成 2。
- `completed-iterations` 是规划输入，不是读取真实 checkpoint 的证据。
- Gate 绝对目标为 50/250/500/1000/1500，不是每次再增加这些次数。
- **不添加 `--execute`。** 正式执行还要求确认与显式 trainer factory，工具不推断生产配方、warm start 或交付身份。

<a id="cuda-example"></a>

## 6. 最小 CUDA 更新、保存与恢复示例

> **未执行，会真实训练。** 供后续单独授权的小型 API 教程；非生产配方、非 AMP 验收、非双 GPU 认证，不能绕过当前阶段要求的正式交付准备。

本例显式选 CUDA/C++、IPPO、v1.4 架构、宏决策与正则。缩小模型，不改变游戏规则；使用完整局，horizon/非法/非有限错误必须失败。在第 2 节路径变量就绪后，运行形式如下：

```powershell
@"
from datetime import datetime
from pathlib import Path
import torch
from guandan_marvel import GameSpec, get_game_backend
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig
from guandan_marvel.training.local import IPPOTrainer

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required; no CPU fallback")
game = get_game_backend("cpp")
run = Path("runs") / ("manual_v14_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
run.mkdir(parents=True, exist_ok=False)
cfg = TrainConfig(
    game=GameSpec.from_game(game), device="cuda", algo_name="ippo",
    logging_dir=str(run), run_name="manual-v14-smoke", seed=20261021,
    rollout_batch_size=1, collect_batch_size=1, mini_batches=1,
    total_iterations=2, amp_enabled=False,
    max_sampler_bytes_per_rank=1024 * 2**20,
    ppo=PPOConfig(
        agent_epoch=1, training_semantics_profile="macro_decision_v1",
        policy_regularizer_mode="macro_normalized_kl_uniform_v1",
    ),
    model=ModelConfig(
        architecture="v1.4-macro-fuse", history_dim=32, feature_dim=64,
        decoder_layers=1, decoder_heads=4,
        state_layers=1, action_layers=1, fuse_layers=1,
    ),
)
trainer = IPPOTrainer(cfg, game=game, run_dir=run / "first")
try:
    before = {n: p.detach().clone() for n, p in trainer.agent_resume.named_parameters()}
    first = trainer.train_iteration()
    changed = any(not torch.equal(before[n], p.detach())
                  for n, p in trainer.agent_resume.named_parameters())
    if not changed:
        raise RuntimeError("No parameter change observed")
    checkpoint = trainer.save_checkpoint(run / "iter1.pt")
    print("first", first, "checkpoint", checkpoint)
finally:
    trainer.close()
del trainer, before

resumed = IPPOTrainer(cfg, game=game, run_dir=run / "resumed")
try:
    decision = resumed.load_checkpoint(checkpoint)
    if decision.kind not in {"exact_resume", "compatible_resume"}:
        raise RuntimeError(f"Unexpected resume decision: {decision}")
    if resumed.iteration != 1:
        raise RuntimeError("Restored iteration mismatch")
    second = resumed.train_iteration()
    if resumed.iteration != 2 or resumed.durable_updates != 2:
        raise RuntimeError("Durable counter mismatch")
    resumed.save_checkpoint(run / "iter2.pt")
    print("resume", decision.kind, "second", second)
finally:
    resumed.close()
"@ | & $Py -
```

验收时应检查：

- 真实参数改变、loss/梯度通过 Trainer 有限性检查；AMP 生产验证还要额外查 scaler 跳步。
- 真实完整终局，不把强制截断当 completed。
- `iter1.pt` 加载身份明确、计数正确；恢复后到 iteration 2，不是从 0 开始。
- 保留 `first/ippo_metrics.jsonl`、`resumed/ippo_metrics.jsonl` 和 checkpoint。
- 固定上下文、小模型、AMP off 不证明 52 上下文/贡还/生产尺寸/Pool/NCCL。

本例没有评估 hook，不运行外部对手或 final holdout；`total_iterations=2` 不是自动循环，两个 `train_iteration()` 调用才执行两轮。报错时保留现场，不修改规则、语义、身份校验和容差来跑通。

<a id="resume"></a>

## 7. Warm start 与 resume 不同

| 方式 | 取回什么 | 后续状态 |
|---|---|---|
| `load_warm_start_policy` | 经兼容/迁移规则检查的策略权重；新增分支按协议初始化 | optimizer、sampler、计数、schedule 保持新任务状态 |
| `load_checkpoint` / `load_checkpoint_payload` | 训练权重、优化器、RNG、采样器、计数及 schema 所需状态 | 仅接受 `exact_resume` / 允许的 `compatible_resume` |
| evaluation-only load | 推理权重与元信息 | 不获得续训资格 |
| migration / warm-restart 判定 | 无法原样恢复的差异 | 不是自动迁移授权 |

核对项包括训练语义、模型、奖励尺度、数据来源、源码、执行环境、native identity、optimizer/sampler/RNG 与计数；Pool 还要核对冻结策略与调度状态。

正常加载不能遇到 migration 就悄悄重置一部分状态。v1.3 → v1.4 有显式 warm-start 规则，任意尺寸/版本不兼容不能 `strict=False` 后称成功续训。

失败时保留 `decision.kind`、reasons、实际文件、配置/源码/native hash。路径变化是否允许，以分类器和交付协议为准。只加载可信来源 checkpoint，不把未知 `.pt` 当无执行风险的文本文件。

源码：[local.py](../../guandan_marvel/training/local.py)、[checkpoints.py](../../guandan_marvel/training/checkpoints.py)。

## 8. 日志与保存

- 直接 Trainer：`<run_dir>/ippo_metrics.jsonl` 或 `vrpo_metrics.jsonl`。
- Kaggle worker：还有聚合 `metrics.jsonl`、rank 日志、profile、遥测与 recovery，按具体 worker 输出。
- `save_checkpoint()` 默认保存到 `<run_dir>/checkpoints/`；直接 API 必须显式调用保存。

看真实 durable updates/完成局数，不把配置目标、启动日志或文件名当完成证据。正式运行前检查有效参数更新、AMP 跳步、显存峰值、宏样本统计与恢复连续性。

<a id="production"></a>

## 9. 正式交付与 Kaggle 准备顺序

1. 查当前入口，明确新授权范围与 Gate 预算。
2. 冻结源码、功能、模型、AMP、seed/数据域、局数/wall 预算、评估协议。
3. 明确 policy-only warm start 或严格 resume。
4. 用当前源码构建**新路径、新版本身份**的交付，保留旧包。
5. 核对 archive/profile/semantic/warm-start/native/manifest hash。
6. 在目标设备与真实配置上验证启动、有效更新、保存恢复、显存、DDP。
7. 准备通过并再次获准后执行相应 Gate，不自动一路跑满。

可先安全查看接口：

```powershell
& $Py scripts/build_marvel_v14_phase4_delivery.py --help
```

这是阶段 4 工具，**不是阶段 5/6 当前新交付已完成的证明**。默认 warm-start 文件在新克隆中可能不存在，不能用默认输出覆盖历史包。本轮没有 build、`--runtime-probe` 或双 T4 执行。

跨会话恢复使用匹配的 Dataset/source ZIP 与 recovery manifest，不是拷贝单个权重配最新代码。wall 预算应预留 checkpoint、原子 recovery 和 output 复制时间。历史细节见 [M07 Runbook](../marvel_migration/07_KAGGLE_T4X2_RUNBOOK.md)、[DDP 修复协议](../marvel_migration/M07_KAGGLE_DDP_REPAIR_PROTOCOL.md)，实际选择仍服从当前入口。

<a id="troubleshooting"></a>

## 10. 故障排查

| 现象 | 优先检查 | 不要这样处理 |
|---|---|---|
| CUDA 不可见 | 解释器、torch build、驱动与设备 | 改 CPU 后称 GPU 验证通过 |
| C++ 导入失败/身份漂移 | ABI、构建依赖、二进制与源码身份 | 复制旧二进制或禁用守卫 |
| OOM | 缓冲、KV、激活、优化器、per-rank 批量 | 仅凭参数小就断定放得下 |
| horizon/capacity 错误 | 完整局、编码容量、规则/协议 | 截断后标正常终局 |
| macro/概率契约失败 | 前缀、mask、温度、旧行为概率、owner | clamp/跳过制造通过 |
| resume 被拒绝 | 源码/语义/native/配置差异 | 删 metadata 或 `strict=False` |
| 参数没更新 | 梯度、scaler 跳步、真实 optimizer step | 仅看 loss 有限 |
| 评估失败但训练继续 | `evaluation/hook_status` 和 hook 错误 | 进程不崩溃就称评估成功 |
| 全量非全绿 | 完整日志与验收范围 | 删测试、改容差或隐瞒失败 |

## 11. 本轮验证的可追溯记录

2026-10-21 使用当前 `.venv-gpu/Scripts/python.exe`，实际命令与退出码见本地 `runs/documentation_20261021/commands.json`。这是文档 QA，不是生产认证：基础环境查询、显式 C++ 导入、25 项 CPU 规则单测、Gate help/dry-run、delivery help，以及示例的静态检查。

本轮**没有**安装/编译、模型前向/反向、GPU smoke、正式 Gate、Kaggle、外部对战、final holdout 或全量回归。历史全量状态看当前入口，不能用这 25 个测试代替。
