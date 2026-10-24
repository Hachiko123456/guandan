# 04A GPU 本地验证与 Claude Code 复核补充规范

更新日期：2026-10-19。与 `04A_phase4_delivery_phase3_integration_repair.md` 一并阅读。

## 当前状态与下一步

GPU 环境已安装，设备接线和交付集成已由 Codex 接续实施；不要重新从头实施 04/04A，也不要恢复读取旧 Prompt 的暂停任务。

下一步在全新的 Claude Code 项目会话使用：

```text
/review-phase docs/marvel_v14_prompts/04A_phase4_delivery_phase3_integration_repair.md
```

该入口只启动全新只读 Reviewer，不调用 Implementer、不自动修复。若 FAIL，输出定向修复 Prompt 后等待用户，不默认启动新的实施任务。

证据索引：`runs/gpu_runtime_completion/IMPLEMENTATION_REPORT.md`。不要把已有报告当作 Reviewer 自己运行的证据。

## 独立环境

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe
```

当前安装：Python 3.12，torch 2.13.0+cu126，CUDA build 12.6；本机 RTX 3050 Ti Laptop 4GB。每次报告仍须读取真实诊断，不依赖本段静态值。

- GPU 命令显式使用上述 Python；子进程用 `sys.executable`。
- 不使用裸 pip，不改全局 Python/Guance/NVIDIA 驱动，不重建 native 扩展。
- `.opencode/` 和 `opencode.jsonc` 必须原样保留。

## 设备及精度门禁

1. runtime 函数、build/verify CLI 支持 `--device auto|cpu|cuda`。auto 可按可用性选择，正式 GPU 验收必须明确 `cuda`；明确 CUDA 不可用时 fail-fast，OOM 原样失败，不静默 fallback。
2. 本机仅 `cuda:0`、world_size=1、batch=1、1 iteration、horizon=512；packaged T4x2/NCCL profile 和 4096 horizon 不变。
3. source/target 迁移前输出使用同设备 FP32，显式关闭 ambient autocast，必须 `torch.equal`；绝不比较 CPU 与 CUDA 逐位相等，不降低容差。
4. CUDA learner 必须 fp16 autocast + enabled GradScaler。CPU 专项明确 CPU BF16，不能从主机 CUDA 可用性推断测试设备。
5. local CUDA 初始 scale 默认 1024，用于单轮 smoke，记录在报告；正式生产 trainer 的 65536 默认 scale 不变。不能降低数学/精度标准换取通过。
6. 至少一次 effective optimizer update，learner_updates>=1 且参数真实变化；全部 skip/零更新直接失败。
7. 同设备 reload 必须逐位恢复模型/optimizer，恢复 iteration/learner/skip counters 和 scaler。严格继承原 fail-closed resume，不用旧 CPU checkpoint 冒充 CUDA exact resume。
8. probe scratch 必须在 delivery 目录之外，Reviewer 只能使用临时目录，不改正式 checkpoint、manifest、ZIP、验证 JSON 或已有证据。
9. 报告 Python 路径、torch/CUDA、GPU、parameter device、autocast dtype、scaler scale/skips、显存峰值、rollout/learner 时间和所有本地适配。

## 验证与预算

优先设备接线反例、GPU AMP 专项、一次 packaged-profile GPU probe。保留 CPU BF16 误差测试；CUDA 专项在缺 GPU 时明确 skip，而非删除。

最终源码冻结后重建 ZIP 并验证 source/archive/upload/warm-start/semantic/native hashes。旧源码 ZIP 的验证结果不能复用到新源码。

同一最终源码的全量回归只跑一次；保存完整日志和前后源码 hash（包括未跟踪文件）。Reviewer hash 核对后复用全量证据，但亲跑独立反例和决定性 GPU 路径。

已知 Guance dirty-tree/authority handoff failure 单列，不绕过、不混入本次修复。全量有该 failure 时必须如实报告，不能写“全绿”。

本地单卡 checkpoint 不是正式双 T4 recovery，本地通过不等于双 T4/NCCL 通过。不启动阶段 5/6、长训、Kaggle、final holdout。

## 本次修改范围

仅本地设备接线、delivery 集成/验证、对应测试、Prompt 与使用说明。PPO/GAE/reward/context/fuse/AMP 核心数学、全局配置构造器和 native 扩展不得为本次 GPU 接线而改变。
