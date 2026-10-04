# A07 Kaggle 远端会话与恢复验收

## 范围与结论（2026-10-04）

A07 只验证 **remote_full 的可审查会话接口、合作式停止、完整 checkpoint 恢复契约和离线打包边界**。本机结果只能是 `ready_package`/本地验收证据，不能写成 genuine Kaggle、GPU 训练完成或远端验收通过。

本轮明确：

- `configs/acceptance_profiles.json` 使用 `profiles-0.2`；`local_fast` 是本机测试 profile，`remote_full` 是唯一远端 profile。
- 本机恢复测试使用真实 `train(..., checkpoint_controller=CheckpointController(...))`，但只用显式生命周期 fixture：CPU、`rollout_envs=1`、每环境 `rollout_steps=1`、目标 2 updates。该覆盖用于验证生命周期，不代表任何 profile 数量或模型强度。
- 不启动真实 `remote_full`，不要求本机 CUDA，不访问 Kaggle API，不上传 source/checkpoint，不执行远程 notebook，不把路径外观或环境变量当作 Kaggle 证明。
- 本轮不修改 `project_status/STATUS.yaml`；验收 runner 的 `accepted` 保持 `false`。`scripts/run_acceptance.py` 产生的是一次性历史证据，不是 supervisory acceptance。
- 前置缺陷继续记录在 [`docs/prerequisite_defects_A05_A06.md`](../prerequisite_defects_A05_A06.md)，包括级别 2 强度、贡还/还贡、A02 随机完整对局覆盖和 A03 声明碰撞；本 A07 文档不删除、不重述为已修复。

## 主 API 与命令

入口是：

```text
guandan.deployment.session
```

命令行可通过 `python -m guandan.deployment.session` 调用，也可在 Kaggle source bundle 中调用等价入口。支持：

```text
--action plan|system|smoke|train|evaluate
--profile remote_full
--execute
--algorithm ippo|vrpo
--output-root /kaggle/working/guandan
--resume /kaggle/input/<bundle>/manifest.json
--session-hours 10
--save-margin-seconds 300
--checkpoint-seconds 600
--candidate-checkpoint /kaggle/input/<candidate>/candidate.pt
--snapshot-checkpoint /kaggle/input/<snapshot>/snapshot.pt
```

门禁约定：

1. `plan` 只解析 profile、恢复点和目标，不训练、不创建远端任务；默认目标是每算法 100 次 base updates。
2. `system` 只读取运行时事实。Python/CUDA/Kaggle 目录检查失败时必须显式失败；CPU fallback 不能冒充 remote_full。
3. `smoke`、`train`、`evaluate` 必须显式传 `--execute`；没有该选项不得自动执行。
4. 执行输出只能落在 `/kaggle/working` 下的命名子目录；`/kaggle/input` 只作为只读输入。
5. `evaluate` 必须显式提供 candidate 和 frozen snapshot；remote_full 的 canonical pair 是 candidate update 102 与 snapshot update 100，并且算法必须一致。
6. `training_plan` 不自动缩减目标：新会话是 `0 -> 100`；从 durable update 99 恢复是 `99 -> 100`；完成 base 后新会话是 `100 -> 102`。

本机验证时通过 `session.run(args, runtime_check=fake_check)` 注入测试 seam；这不是 CLI 的 runtime bypass，也不改变远端默认门禁。

## Kaggle 上传形态

Kaggle Dataset 可以只包含一个源码 ZIP；当前 A07 入口支持：

```text
/kaggle/input/<source-dataset>/guandan-source.zip
```

Notebook 会固定校验交付包的 ZIP SHA-256、manifest commit、manifest SHA-256 和
环境检查入口 SHA-256，再调用 `install_source_archive()` 在
`/kaggle/working` 创建新源码副本。ZIP 内部仍必须使用 package builder 生成的
`guandan/` 前缀和 `SOURCE_MANIFEST.json`，不能手工重打包或编辑文件。

## Source bundle / ready package

source bundle 必须来自离线、可验证的 source export，包含 `SOURCE_MANIFEST.json` 和 manifest 中的文件哈希；安装脚本只能把它复制到一个新的、非嵌套、可写目录。source manifest 的 `profile` 是 `remote_full`，但它的 `accepted`、`uploaded`、`kaggle_verified` 均必须保持 `false`，直到人工完成远端证据审查。

在本机，`ready_package` 的含义是：

- source 文件和 manifest 边界可检查；
- profile、命令和恢复协议可执行性已通过本地测试；
- 生成了本地 evidence / package 结果；
- **不是** genuine Kaggle session；
- **不是** CUDA availability 证明；
- **不是**训练、评估或模型强度证明。

本机测试断言 `accepted is false`、`kaggle_verified is false`、不上传，并比较 `project_status/STATUS.yaml` 前后字节完全不变。任何带 `/kaggle/...` 的字符串只是目标路径契约，不是执行地点证据。

## 合作式 session-stop 策略

`CheckpointController` 的接口和默认值：

```python
CheckpointController(
    session_hours=10,
    save_margin_seconds=300,
    checkpoint_seconds=600,
    clock=callable,
)
```

它提供：

- `before_update()`：启动 update 前检查请求停止、soft deadline 和基于历史完整 update 的预估余量；
- `after_update(elapsed)`：只记录已完成 update 的用时；
- `check()`：在 rollout/update 内传播 soft stop；
- `checkpoint_due()` / `checkpoint_saved()`：以最近一次成功保存为周期基准；
- `request_stop(reason)`：可重复观察的 cooperative stop；
- `signal_handlers()`：在上下文中接管 SIGINT/SIGTERM，退出时恢复旧 handler。

它不是 hard-kill 恢复保证。soft timeout 或 SIGTERM 只允许在安全边界保存；未完成的 rollout 或 optimizer update 不能伪装成完整 update。

## Durable checkpoint 与恢复不变量

`train(..., checkpoint_controller=controller)` 返回并写入 evidence 的核心字段：

- `recovery_manifest`：与 checkpoint 同目录的 sibling manifest；
- `durable_updates`、`durable_token_steps`：最近一次完整 update 的持久计数；
- `discarded_partial_steps`：中断 update 中已消费但回滚丢弃的真实 token steps；
- `discarded_optimizer_steps`：中断 optimizer update 中回滚丢弃的 optimizer steps；
- `status="incomplete"`：有 cooperative stop 时不能返回 `complete`。

manifest 格式是 `guandan-recovery-v1`，只存 `checkpoint_filename` 的 basename、checkpoint SHA-256、profile/algorithm、model config、compatibility、protocol versions、source commit 和 durable counters。它不能依赖原机器绝对路径。中断保存使用类似 `*_interrupted.pt` 的 checkpoint 和 sibling `.recovery.json`；即使在第 1 次或第 10 次 update 之前停止，也必须保存最近的完整状态。before-first-update 停止的 durable update 为 0。

恢复测试的强断言：

1. 对 before-first、before-second、after-first、rollout 中途 soft timeout、rollout 中途 SIGTERM、optimizer 中途 soft timeout 分别注入停止。
2. 回滚后的 checkpoint 中 weights、optimizer、环境序列化 bytes、Torch/NumPy/collector RNG、update/token counters 与最近完整边界一致。
3. `discarded_partial_steps` 与 `discarded_optimizer_steps` 只描述真实丢弃量；不得把部分 update 记成 100% 完成。
4. 把 checkpoint 和 manifest 复制到不同 input mount，删除旧目录后，仅用 manifest sibling basename 恢复；恢复不能读取旧绝对路径。
5. 恢复后的真实 continuation 与同 seed、同设置、不中断 baseline 逐项一致，包含后续 rollout digest 和最终 checkpoint 状态。
6. 伪造 digest、profile、algorithm、protocol、compatibility、source commit、counter、路径穿越或绝对路径均必须失败。

## 测试命令与计数范围

使用指定环境：

```powershell
C:\Users\yhx\.conda\envs\guandan_train\python.exe -B -m pytest `
  tests/acceptance/test_a07_kaggle.py `
  tests/unit/test_kaggle_recovery.py -q -p no:cacheprovider
```

上述两个文件的测试分为两类：

- `tests/acceptance/test_a07_kaggle.py`：session parser/plan/system/execute gate、runtime/output-root gate、mocked remote trainer/evaluator、A04 bounded smoke、runner evidence 和 STATUS 不变性；远端 trainer/evaluator 只允许 mock。
- `tests/unit/test_kaggle_recovery.py`：当前真实 trainer 的 local_fast CPU recovery lifecycle、CheckpointController fake clock/signal、manifest digest/path 防护，以及复制 checkpoint 后的实际 continuation。

local lifecycle 的计数是有意 scoped 的 `1 env × 1 token per update × 2 updates`，不是 `local_fast` canonical 计数（4 env、256 token steps、5 updates），更不是 `remote_full` 计数（8 env、1024 token steps、100 updates）。测试不会运行 `remote_full`。

阶段 runner：

```powershell
C:\Users\yhx\.conda\envs\guandan_train\python.exe -B scripts/run_acceptance.py `
  --stage A07 --profile local_fast
```

该 runner 只执行本地 A07 测试并保存 `project_status/history/<run-id>/A07/` 证据；它不会修改 `STATUS.yaml`。报告通过也只表示 ready package / local acceptance gate，不表示 genuine Kaggle。

## 已知限制与不宣称事项

- 当前日期为 **2026-10-04**；本机测试结果不能替代未来或其他日期的 Kaggle GPU 会话。
- 未执行 remote_full、未验证 CUDA GPU 训练吞吐、未验证 Kaggle notebook/kernel 生命周期、未验证网络/上传/凭据/断电硬杀恢复。
- session 的 cooperative policy 只覆盖可观测的 soft timeout 和 SIGINT/SIGTERM；不能对进程被强制杀死作无证据承诺。
- A05/A06 的已知前置缺陷仍适用，见 `docs/prerequisite_defects_A05_A06.md`；本轮不修改 A00-A04 或前置规则结论。
- `accepted=false` 是正确的当前状态；只有主代理在获得实际 remote_full 证据后，才能单独审查并更新阶段状态。

## 统一 Notebook 增量验收

- `guandan/deployment/workflow.py` 仅编排已有 trainer/evaluator，不替换训练算法。
- 一个共享 controller 覆盖 smoke、两个算法的100+2更新和两份3000局评估。
- 当前 working 的 checkpoint 必须限定在明确的 workflow output-root；standalone
  input-only 门禁保留；临时目录或符号链接逃逸必须失败。
- same-session reload 不能标成 genuine-new-Kaggle-session verified。
- durable update102 才允许跳过训练；update100 frozen snapshot 必须保留。
- completed evaluation 只有在 digest、checkpoint元数据、逐局排名/reward/终局、
  opponent/group/rotation 覆盖都通过时可跳过；不完整评估不累计充数。
- 进度包路径相对化，manifest与每个文件SHA256验证后写入全新working目录。
- 以前失败的 gate 不默认重试；不恢复自动 accepted 状态。
- 离线测试使用显式标注的 synthetic records，不是远端训练/对局证据。
- 增量测试：`tests/unit/test_kaggle_workflow.py` 和 A07 session workflow scoped tests。
- A02/A03 历史前置缺陷继续适用，不修改或移除既有缺陷记录。
