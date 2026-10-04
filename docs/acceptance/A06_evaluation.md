# A06 评估

## 范围

实现固定随机种子、固定 deal groups、四个座位轮换的队伍评估，以及 `random`、`rule`、`snapshot` 基线。评估必须读取 profile 并按 profile 执行实际完成局数；少量 smoke 结果不能解释成模型能力结论。

## Profile 计数契约

评估配置属于 `configs/acceptance_profiles.json` 的 `profiles-0.2`：

- `local_fast`：`deal_groups_per_pairing=4`，`seat_rotations=[0,1,2,3]`；对 `random`、`rule`、`snapshot` 每个对手实际完成 `4 * 4 = 16` 局。
- `remote_full`：`deal_groups_per_pairing=250`，`seat_rotations=[0,1,2,3]`；对每个对手实际完成 `250 * 4 = 1000` 局。
- 时间预算分别为 0.5 小时和 12 小时；超时未完成目标数量必须报告 `incomplete`，不得伪造通过、补写局数或由硬件自动缩放。
- 这些样本量只规定评估管线的可复现执行规模，不提供胜率、强度、泛化或排名保证。

`random`、`rule`、`snapshot` 三个对手都必须分别计数和报告，不能合并成一个“基线已运行”的布尔值。`seat_rotations` 不是可选提示，而是必须实际应用并在证据中列出。

## Runner 与实际证据

Runner 的 `--profile` 必须向测试转发 canonical `GUANDAN_PROFILE` 和 `GUANDAN_RESOLVED_PROFILE_JSON`。A06 测试必须实际加载配置并使用 `deal_groups_per_pairing`、`seat_rotations`、对手列表和时间预算；不能硬编码另一个局数，也不能只断言配置文件存在。

`--show-profile` 只打印解析配置、不运行测试、不创建验收通过结论。无效 profile/config 必须失败；硬件检测不能改变 profile。对局“完成”必须有真实终局记录、双方/队伍奖励、排名和所需元数据；计划启动但未终局的局不计入完成数。

## 最小正确性门槛

- 固定 seed 与固定输入可复现；
- `[0,1,2,3]` 四个座位/队伍轮换实际生效，且换座指标按轮换分组可追溯；
- 训练发牌/seed 与评估发牌/seed 分离；
- 队伍胜负、奖励、排名、终局标志和置信度元数据正确；
- `random`、`rule`、`snapshot` 基线均能实际运行并分别出现在报告中；
- 模型、环境或基线错误必须显式失败，不能静默回退到备用代理；
- 报告包含模型、规则、动作、环境、seed、profile、Git 元数据、实际完成局数、超时/截断/失败原因；
- 若评估依赖 A05 的 actor/critic 接口，必须继续保持私有牌不泄露和玩家/队伍视角，不得通过评估适配器绕过边界。

## 命令

```powershell
python scripts/run_acceptance.py --stage A06 --profile local_fast
python scripts/run_acceptance.py --stage A06 --profile remote_full
python scripts/run_acceptance.py --stage A06 --profile local_fast --show-profile
```

本机 `local_fast` 只能支持 `local_ready` 的管线正确性证据；远端 `remote_full` 才提供完整样本量证据。A06 不能将本机结果写成 `accepted: true`，也不能以本机结果替代 A07 的真实 Kaggle 会话。

## 测试与回归

每次 A06 变更先运行完整 scoped `local_fast` A06 及目标依赖测试；不要求每次改动都重跑 A02 既有 10,000 个随机种子对局。远端/release 阶段运行完整回归和发布测试，不得删除或弱化安全、牌守恒、终局/截断、reset、信息边界和错误不静默回退测试。

## 本机 local_fast 主代理复核（2026-10-04）

早期 `0e5c1a1` 的四个测试虽然通过，但其 snapshot 是固定规则、换座没有保持相同基础发牌、且 hash() 随机流不可跨进程复现；该报告已被主代理拒绝，不能作为本机通过依据。

最终实现以 A05 已审查证据中的 **IPPO update 6** 为候选策略，**IPPO update 5** 为冻结 snapshot 对手；加载实际权重，记录 checkpoint SHA256 和协议/模型版本。两个候选队伍座位都运行候选策略，对方两个座位运行指定基线，所有 agent 只接收 Observation。

每组发牌的物理分配和初始领出座位保持不变，轮换的是四个策略角色；每个候选主座位实际看到不同手牌。random/rule/snapshot 分别完成 4 组 × 4 轮换 = 16 个真实终局，总计 48 局。另有 1 局跨进程 Python hash seed 重放回归，不并入 48 局计数。

实际报告：`project_status/history/20261004T031552369870Z_39c781cb4c2740e4963f0b38765e3059/A06/report.json`；逐局 JSON 与汇总位于同目录 `execution/games/` 与 `execution/evaluation_summary.json`。Runner 包含 112 项 A06 scoped 测试，全部通过、无跳过；同代码全量回归为 708 passed。

统计包含真实局数、排名/reward、一致的候选队伍视角、按 rotation/deal 分组及 Wilson95 元数据。相同基础牌的轮换不是独立样本；区间仅用于展示统计管线，不是强度/显著性证明。缺模型、版本不兼容、非法动作、超时和截断都显式失败/不完整，不回退、不补计局数。

本机预算只运行 local_fast；remote_full 和 Kaggle 未运行，accepted 保持 false。已知前置规则缺陷见 `docs/prerequisite_defects_A05_A06.md`；本轮局限于级别 FIVE、无上局贡还的单副管线。
