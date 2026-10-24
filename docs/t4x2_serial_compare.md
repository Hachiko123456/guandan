# Kaggle T4×2：一次上传、双配方串行训练与统一评估

当前使用 **delivery_v12（2026-10-24 candidate恢复预检窗口修复）**。v10真实T4×2已通过输入、control全预检和candidate首次两迭代训练；candidate精确恢复后的唯一迭代产生0次optimizer update（与其target-KL早停路径一致，外层日志不足以排除其他零更新来源），旧预检将这个单样本判为失败。正式两臂训练尚未开始；请勿再上传或混用v10及旧Notebook。

- 上传目录：`D:/project/guandan/runs/t4x2_serial_compare_delivery_20261023/delivery_v12/`。
- Dataset只上传该目录的 `guandan_serial_compare_payload_v1.bin`，Notebook导入同目录的 `guandan_serial_compare_v1.ipynb`；新建/重启干净的Kaggle会话，选择T4×2后Run All，首次不指定旧恢复路径。
- v12保留v10的逐episode候选池校验，只把精确恢复预检从单迭代判定改为最多3个迭代的完整窗口：窗口内必须至少产生一次真实optimizer update，三个迭代全为0或预算截断仍失败。训练配方、target-KL、batch、horizon、正式预算、模型、native、评估数据及plan hash不变；首次使用干净运行目录，不继承旧失败作业的半成品。
- 本机CUDA用生产结构batch=4连续训练2轮，保存恢复后再训练1轮并完成导出评估；实际覆盖跨局不同队伍/对手与12条轨迹。这是单GPU有界验证，不能冒充双T4全尺寸认证。
- 保留此前已修好的torchrun启动、日志、评估牌级内部编码（3..15，牌面2为15）、贡还排名语义及生成器v2。
- 所有168条评估配置在开训前先通过真实规则与native reset核验，防止非法评估数据等到训练之后才报错。本机GPU已完成生产尺寸模型的17场实际评估（含真实CLI和候选一次更新/保存/恢复后的评估），这只证明本地接线，不代表棋力提升或双T4认证。
- 评估记录必须绑定完整任务配置hash和自身hash，终局排名、赢家、结局、奖励必须一致；预检只接受全部任务完整结束，截断或重复记录不能放行正式训练。空间不足的降级结果ZIP会据实际成员重签清单并列出缺失证据。
- 每个子进程开始/结束会在Notebook打印阶段日志路径与耗时；详细训练输出仍写完整console.log。
- 若预检仍失败，错误输出会显示原始rank异常、完整日志路径；逐rank详情位于对应run目录的 `worker_errors/rankNNN.json`。请下载完整 `artifacts/phases/preflight_control/train.console.log` 或失败阶段的同名日志。
- 本地验证不代表目标双T4已实测；两臂目标预检继续执行，失败即停，禁止手动绕过。

## 先说结论边界

本包是 **2026-10-23 的新实验交付**，本会话只完成实现、单元/回归测试、交付包构建与离线 QA，**没有启动 Kaggle、没有正式训练、没有双 T4/NCCL 认证、没有 FableDan、没有 final holdout**。`formal_production_promotion` 恒为 `false`；本地单卡/Gloo 测试不能替代目标双 T4 预检。

## 上传：仍然是两个文件

1. 新建 Kaggle Dataset，上传唯一文件 `guandan_serial_compare_payload_v1.bin`（保持 `.bin`，不要手动解压）。
2. 导入 Notebook `guandan_serial_compare_v1.ipynb`。
3. Accelerator 选 **GPU T4×2**，Internet 可关闭，Run All。首次运行不需要改变量。

Notebook 与 payload 通过 SHA256 双向绑定；错包、旧包、多个候选 payload 会在任何训练前停止。包内已含三个原始模型（375 起点、250 池/参考、500 探针）、原 source、原 Linux native、新源码与驱动、固定发牌协议与 plan，不需要另外上传恢复包或模型。

## 两条臂到底差什么

| 项目 | control | candidate |
|---|---|---|
| 起点 | 同一 375 策略（显式迁移） | 同一 375 策略（显式迁移） |
| NTP | 保留，`.02` | 关闭，白名单移除 `ntp_head.*` |
| 训练对手 | 现有 self-play | 冻结 250、375 静态池，权重 1:1（工程默认） |
| target-KL | `None`（严格旧路径） | `.015`，全局宏样本 KL 早停 |
| optimizer/sampler/RNG | 新运行、显式初始化 | 同样的规则独立初始化 |
| 课程 | 冻结真实基线语义，逻辑起点 offset=375 | 完全相同 |
| 预算 | 相同墙钟配额 | 相同墙钟配额 |

两臂不是“同一模型先跑旧配方再改参数”：它们从同一 375 起点**独立**初始化，不继承对方的训练后权重、optimizer、采样器、cache 或随机状态。控制臂的最终模型永远不是候选臂的初始化来源。

配方按真实 v4 基线：`macro_decision_v1` + `macro_normalized_kl_uniform_v1`、lr `1e-4`、clip `.2`、reg `.01`、rollout 48 / collect 40、horizon 1024、AMP 开、Symmetry/Belief/search 关闭、贡还课程保留。不新增或改动奖励、GAE、层数、搜索、调度。

## 串行阶段与硬门

```
bootstrap/input_check
  -> preflight/control
  -> preflight/candidate
  -> freeze_execution_budget
  -> train/control
  -> train/candidate
  -> freeze_final_checkpoints
  -> evaluate/confirmation_primary
  -> evaluate/development_diagnostics
  -> summarize_and_package
```

- **两个预检都通过之前，任何正式训练都不会开始。** 预检在隔离目录完成真实双 rank 小量训练、保存、恢复后再更新、导出与少量真实对局（预检本身的评估），**不做结果打包**；预检不写结果 ZIP，其产物不进入正式臂。
- 每个阶段有前置条件、输出完整性、原子完成标记（绑定 plan/source/model/output hash）、失败记录与恢复位置。只看“目录存在”不算完成。
- 已完成的 control 不重训；两臂都完成后若评估中断，恢复只补评估/打包，零新增训练。
- 正式训练与评估使用独立目录和独立进程；只管理本任务创建的进程。

## 预算（默认总 14400 秒，从 bootstrap 起算）

| 阶段 | 默认上限/预留 |
|---|---:|
| 依赖、编译、校验与两臂预检 | 1200 秒 |
| control 训练 | 2400 秒 |
| candidate 训练 | 2400 秒 |
| 评估硬预留 | 6000 秒 |
| 最终保存/汇总/打包预留 | 600 秒 |
| 机动空间 | 1800 秒（不自动给任一训练臂） |

单臂新增迭代上限 256；首要比较口径是**相同训练墙钟配额**。只有两臂正式训练前才允许按公开公式对称缩减，并冻结不可变 resolved 预算与 subplan hash；一旦任一臂开训不得再缩减。累计预算跨会话保存、不重置，剩余额度取“累计剩余”与“本会话 deadline”的较小值。

## 评估：主确认 256 场 + 开发辅助 448 场 = 704 场

- 主确认：`candidate_final vs control_final`，128 副**新**发牌 × 2 个唯一整队分配 = 256 场。正式胜率与 95% 区间只看这一组。
- 开发辅助：另 32 副新发牌 × 2 分配 × 7 臂 = 448 场（candidate/control 对 control/375/rule/500）。
- 375 用过两个唯一整队分配，不再用重复的四轮换冒充独立样本；仍按**独立牌局聚类**估计不确定性。
- 正式判定：主区间下界 `> .5` 为支持；上界 `< .5` 为较差；跨 `.5` 为 `INCONCLUSIVE`；主确认不完整为 `EVALUATION_INCOMPLETE`，都不等于技术失败。
- 训练、预检、开发、确认使用不同命名空间：正式训练 seed 为 `570700`（两 rank 分别加 `1000003`），预检为 `1600000000`，开发 `1610000000`，确认 `1620000000`。`input_check` 不只比较 seed 名字：它要求各命名空间的 seed 不重叠，并用引擎真实生成**每个训练 rank seed 的第一副牌**与**每个评估 deal 的第一副牌**逐一比较 `base_deal_sha256`，任何直接碰撞会 fail-closed；预检第一副牌与正式训练第一副牌哈希不同。更强的一般性证明依赖 seed 命名空间互斥（训练 `570700`/`+1000003`，预检 `1600000000`，开发 `1610000000`，确认 `1620000000`），本检查额外覆盖第一副牌的物理身份。确认集不反馈训练、池权重、早停或选模。
- 无 FableDan、无 final holdout；rule 不是 FableDan 替身；500 不入训练池，只作未入池旧对手探针；375 在池内，只能看针对性收益。

## 失败、恢复与输出

- 技术失败（身份/非有限/非法/OOM/超时）不自动缩 batch/horizon、不切 CPU、不改规则、不跳过错误继续；失败阶段和可恢复位置写入 `RESULT.json`，不会写成 `complete`。
- 正常预算停止会安全保存：训练臂在 `arm_result.json` 中记录 `stop_reason=budget_stopped` 和实际迭代/更新数，完成的阶段不重做。单局评估超时会保留已落盘的逐局记录并把评估阶段记为 `incomplete`，而不是丢失整个 shard。
- 结果包：`guandan_serial_compare_results.zip`（在训练开始前先生成一个最小有效版本，结束时原子更新）。解压后是完整运行根目录：`artifacts/`（`REPORT.md`、`RESULT.json`、逐阶段耗时与预算、两臂 final 模型、初始化迁移证明、**每臂最新完整恢复**、逐局记录与聚类统计、发牌 manifest/provenance、KL 停止日志、架构诊断、per-rank GPU 遥测、错误 traceback）和 `input/`（plan、发牌 manifest、三个模型、source/native/driver/overlay）。另有 `recovery_index.json` 与包内 `RESULTS_MANIFEST.json` 列出每个恢复文件的 SHA256。
- 若要续跑：下载本次结果 ZIP 并解压得到目录 `R`（必须含 `R/input/plan.json`），用同一 Notebook，把顶部 `RESUME_RUN_ROOT` 设为 `R` 后 Run All。Notebook 会先校验 `input/delivery_manifest.json` 的内层 manifest、逐文件 SHA256 以及 `driver.zip`/`overlay_source.zip` 的成员哈希；任一失配都会在任何训练前停止。

## 交付目录与校验格式

- 交付目录含唯一 Dataset 文件 `guandan_serial_compare_payload_v1.bin`、Notebook、`plan.json`、`DELIVERY_MANIFEST.json`、`README_UPLOAD.md`、`LOCAL_VALIDATION.json`、`DELIVERY_VERIFICATION.json`。
- `SHA256SUMS` 是 JSON：`{"schema": "...", "files": [{"path", "bytes", "sha256"}, ...], "hash": "<对前面字段的 canonical sha256>"}`；`ARTIFACT_QA.json` 在构建后用离线脚本对最终 `.bin` 生成，故不在构建时的 `SHA256SUMS` 列表内。
- `ARTIFACT_QA.json` 是对最终 payload 的隔离解码验证记录（payload/inner manifest/逐文件 SHA/overlay 逐字节/Notebook 绑定）。

## 本地验证能证明什么、不能证明什么

- 能证明：配置语义、迁移白名单与逐位一致性、optimizer 覆盖、RNG 不被池加载消耗、target-KL 的全局 sum/count 数学（含不等宏数与空 rank 的真实双进程 Gloo 验证）、编排顺序/预算/恢复/标记、评估任务唯一性与冲突拒绝、聚类统计与判定规则、交付包解码/哈希/源码逐字节一致与驱动隔离导入。
- 不能证明：目标双 T4/NCCL 吞吐与显存、Kaggle 平台上的端到端 704 场耗时、真实棋力差异。这些都标 `NOT_RUN`，由目标机预检与正式运行回答。
