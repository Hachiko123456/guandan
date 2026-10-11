# M09 正式训练准备与受控算法 Pilot 交付报告

- 日期：2026-10-11
- 分支：`codex/marvel-guandan-migration`
- 开始 commit：`4ef1937d58fa50027908b0cd7b4d822ee19c1961`
- 交付源码 commit：`18507bdaa8dd86b25817060d64852a7531cbd8e9`
- 准备实现 commits：`535780d981fc5b391fcfabe402ce485ba17b3b3b`、`18507bdaa8dd86b25817060d64852a7531cbd8e9`
- 执行环境：Windows / Python 3.12 / 本地 C++ backend 与 CPU smoke；远端仍需 Kaggle T4×2
- 阶段状态：`awaiting_remote_algorithm_pilots`
- M09 accepted：**否**

## 范围

本轮只完成 M09 正式远端运行前的安全边界：

1. 核验 M08 accepted commit、锁定配置、报告和机器证据；
2. 生成新的 M09 package id，明确禁止复用旧 `full` Dataset；
3. 把 IPPO/VRPO 比较锁定为相同 completed-game 预算；
4. 加固 checkpoint/evaluation retention；
5. 生成并验证 preflight、IPPO pilot、VRPO pilot 及各自 resume Dataset；
6. 运行本地预算/恢复边界 smoke 和 scoped regression。

本轮没有真实 Kaggle 运行，没有选择主算法，没有生成主算法 long-training Dataset，没有运行 final holdout，也没有模型强度结论。

## 开始状态核验

- 初始工作区干净；
- 当前分支正确；
- 初始 HEAD 与用户给出的 M08 最终 commit 一致：`4ef1937d58fa50027908b0cd7b4d822ee19c1961`；
- `STATUS.yaml`、M08 报告与 `verification.json` 均表明 M08 accepted；
- `configs/marvel_m08_locked_config.json` 与 accepted evidence copy 原始 SHA-256 均为 `6994f7f80eaeb8e44c695cac8d21c6bb03561e6a8615866d47e8dcc933f67d21`。

## M09 新交付

交付根目录：

`D:\project\kaggle_train\guandan-marvel-m09-algorithm-pilot-20261011`

身份：

- package id：`guandan-marvel-m09-algorithm-pilot-20261011`；
- source commit：`18507bdaa8dd86b25817060d64852a7531cbd8e9`；
- source ZIP SHA-256：`ca538e941f8cf671241b982a8ffae83271a04c2ab3503201f3c8b6b67644bec2`；
- plan hash：`923148386be4a8e2c102b913f26570b020b4e4cbfab14197d08a8f02cd5b93d9`；
- locked config SHA-256：`6994f7f80eaeb8e44c695cac8d21c6bb03561e6a8615866d47e8dcc933f67d21`；
- `workspace_snapshot=false`；
- local delivery validation：通过；
- `remote_verified=false`、`m09_accepted=false`。

新交付包含五个互斥 Dataset role：

- `preflight`；
- `ippo_pilot`；
- `ippo_resume`；
- `vrpo_pilot`；
- `vrpo_resume`。

每次 Notebook 只允许添加一个 role Dataset；resume 时再添加同算法最近下载的 recovery Dataset。IPPO/VRPO recovery 通过 `expected_algorithm` fail closed，不能交叉恢复。

## IPPO/VRPO 同完成对局预算

版本化计划：`configs/marvel_m09_full_plan.json`。

- comparison id：`m09-ippo-vrpo-equal-games-v1`；
- 每算法目标：**400 completed games**；
- global collect batch：4；
- 目标 iteration：100；
- 预算关系：`100 × 4 = 400`；
- 相同 seed：470700；
- 相同 environment rank stride：1,000,003；
- 相同模型、batch、epoch、replay、AMP/compile/Muon、DDP 与 wall safety limit；
- wall time 只作安全停止，提前停止的 run 为 incomplete，不能参与算法选择；
- shared workload hash：`6f9bd5c6d9d45fa2adab118be0c043181ab77d68e2d6c21572ec00062a3225ff`。

运行时新增 hash-protected `completed_game_budget` contract。每个 durable iteration 必须满足：

`global completed_games == iteration × global_collect_batch_size`

任何 counter skew、overshoot、算法不匹配或 target mismatch 都 fail closed。输出 `completed_game_budget.json` 明确标记 running/complete/incomplete/failed。

## Selection 数据与算法选择门槛

两个算法使用同一个 selection dataset identity：

`c8d11821a0a9cf8969c304fdbdfaa371c0addbc6883cf20e9617e526b6b5b583`

覆盖：

- 65 个独立 deal clusters；
- 13 个级牌；
- 四个 leader seats；
- `none`、`double_down`、`head_third`、`head_last`、`cancelled`；
- 四座位轮换；
- 相同 random/rule anchor。

这次 Kaggle bounded evaluation 只用于冻结候选和收集共享 selection anchor，不是 formal champion promotion，更不是 final holdout。收到两个算法的远端 checkpoint 后，按预先定义 gate 做 paired head-to-head clustered 95% CI；若仍 inconclusive，再比较 paired rule anchor；仍无法区分时保留 M08 IPPO baseline，并明确记录“VRPO 未证明更优”。禁止按 wall time、training loss 或 final holdout 选择算法。

## Checkpoint retention

- 正式 profile：`checkpoint_keep_last=2`、`evaluation_keep_last=2`；
- checkpoint interval：25；evaluation interval：100；
- retention schema 升级为 `guandan-marvel-artifact-retention-v2`；
- 只允许删除 `checkpoints/iterNNNNNN` 与 `evaluation_checkpoints/iterNNNNNN.pt`；
- symlink、未知文件/目录、path escape、latest/history mismatch 均拒绝；
- 先发布只引用 retained checkpoint 的安全 index，再删除旧文件，避免删除失败造成 manifest 指向已移除历史；
- output ZIP 排除训练 shard 和嵌套 recovery ZIP；durable recovery 仍单独输出。

本地 1-iteration budget smoke 实际得到 1/1 completed game、budget `complete=true`、retention v2，并生成可验证 checkpoint/recovery。该 smoke 只证明本地控制逻辑，不是 Kaggle 或强度证据。

## 训练分布审查边界

为了让本次 IPPO/VRPO 算法变量受控，pilot 两侧继续使用相同的 M08 `Rank.TWO/no previous hand` training distribution。现在改变训练发牌分布会同时改变算法和数据两个变量。

因此主算法 long training Dataset **尚未生成**。收到远端 pilot、完成算法选择后，必须单独记录级牌/贡还训练覆盖决定并冻结新的主训练 semantic config，之后才生成主算法 full Dataset。

## 测试命令与结果

1. `python -m pytest --import-mode=importlib tests/marvel_kaggle -q`
   - 47 passed；0 failed；0 error；0 skipped。
2. `python -m pytest --import-mode=importlib tests/marvel_training tests/marvel_core/test_core.py tests/marvel_evaluation tests/acceptance/test_a00_specification.py -q`
   - 92 passed；0 failed；0 error；0 skipped。
3. `python scripts/build_m09_full_delivery.py --safety-check`
   - exit 0；Git clean；M08 ancestor/config/package/budget/selection/retention gates 通过。
4. `python scripts/build_m09_full_delivery.py --output D:\project\kaggle_train\guandan-marvel-m09-algorithm-pilot-20261011 --commit HEAD`
   - exit 0；生成新的非 workspace snapshot 交付。
5. `python scripts/build_m09_full_delivery.py --validate D:\project\kaggle_train\guandan-marvel-m09-algorithm-pilot-20261011`
   - exit 0；delivery manifest 和五个 role Dataset 全部通过。
6. 本地 completed-game budget worker smoke
   - exit 0；1 iteration / 1 completed game；budget complete；retention v2 通过。

合计自动化测试：139 passed，0 failed/error/skipped。

## 机器可读证据

目录：`docs/marvel_migration/evidence/M09_full_training_preparation/`

- `delivery_safety_check.json`；
- `delivery_validation.json`；
- `algorithm_budget_plan.json`；
- `checkpoint_retention.json`；
- `local_budget_smoke.json`；
- `test_results.json`；
- `upload_inventory.json`；
- `source_manifest.json`；
- `artifact_manifest.json`；
- `environment.json`；
- `verification.json`。

大型测试日志、JUnit、local smoke checkpoint 和 delivery 本体均不提交 Git，分别位于 `runs/M09_full_training/` 和 `D:\project\kaggle_train\...`。

## 未修改的受保护范围

- 未修改 `guandan/cards.py`、`guandan/combos.py`、`guandan/round.py`、`guandan/state.py` 的规则语义；
- 未写入 `D:\project\doudizhu`；
- 未删除 legacy V1 trainer/model/checkpoint 只读评估能力；
- 未运行 final holdout；
- 未生成或晋升 champion。

## 已知限制与 Blockers

1. 尚无本次新交付的真实 Kaggle preflight 日志；
2. IPPO/VRPO 尚未各自完成 400 completed games；
3. 尚未下载并校验两份 remote output/recovery/console；
4. 尚未选择主算法；
5. 主算法 long-training Dataset 尚未生成；
6. 训练级牌/贡还覆盖决定尚未冻结；
7. champion history、正式 selection、long training、final candidate 和一次性 final holdout 均未完成；
8. 模型强度、E1/E2/E3 均未知；
9. 完整多副比赛管理器仍缺失。

## 是否允许启动下一阶段

**否。** 当前只允许用户运行本报告列出的 M09 preflight 和两个 equal-completed-game algorithm pilots。收到真实远端日志前，不得称 M09 accepted、正式训练完成或达到强度门槛；M10 继续 blocked。
