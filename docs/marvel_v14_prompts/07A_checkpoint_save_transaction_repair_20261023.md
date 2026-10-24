# Claude Code 定点修复 Prompt：真实双rank保存事务，限一次修复与验收

日期：2026-10-23（Asia/Shanghai）
项目：D:/project/guandan
任务：修复07阶段最终Reviewer指出的保存阻断，生成可供Codex复核的delivery_v6。

> 请执行本文件，不要从头重新实施07任务。用户把本Prompt交给你执行，即授权这一轮有明确边界的第三轮定点修复；不必再次询问“是否授权第三轮”。不是授权无限修复，也不是授权训练或远端执行。本文件明确收窄本轮范围和测试，不再沿用07中每次重跑全量的要求；工作区保护、独立验收、真实GPU验证要求继续有效。

## 1. 用户要求与完成边界

用户要尽快收尾，但不接受漏掉真实保存路径。唯一目标：

1. 修正DDP保存索引的相对/绝对路径契约。
2. 正确处理同一iteration的周期保存、最终保存和at-target恢复。
3. 将可捕获的保存/索引I/O错误同步给所有rank，有界失败，不单rank先抛错让另一rank久等。
4. 用真实生产保存函数的双rank集成测试证明上述行为，之后才构建一次v6。

本轮不改：网络、PPO/KL数学、lr/clip/reg、NTP设置、池内容、上下文/贡还课程、规则、C++、数据、统计阈值、原14400秒计划或704场未来Kaggle评估方案。

不启动Kaggle、正式训练、704场评估、FableDan或final holdout。不要继续修旧全量中的53 failed/58 errors或重新构建native/换全局Python/Torch。这些问题保留原始记录，不能宣称消失，也不能借它们跳过本轮必要测试。

保留dirty/untracked，不reset/clean/checkout/restore/stash/commit/push，不覆盖原模型、旧包与旧证据。不得绕过hook或删除失败锁。先确认旧实施者已停止写相关文件，不并行启动两个写同文件的Implementer。

## 2. 最少必读材料：不要重读整个项目历史

按此顺序阅读，其余文件仅按依赖补读：

- D:/project/guandan/CLAUDE.md及适用AGENTS.md。
- D:/project/guandan/runs/t4x2_serial_compare_codex_review_20261023/STATUS_REVIEW.md
- 同目录CHECKPOINT_PROBE.json、checkpoint_contract_probe.py。
- D:/project/guandan/runs/t4x2_serial_compare_implementation_20261023/CLAUDE_REVIEWER_REPORT_ROUND3.md
- 同目录HANDOFF_TO_CODEX.md顶部阻断、DEVICE_VALIDATION_CONSTRAINTS.md。
- tools/serial_compare/training.py的_save_recovery、周期/最终保存与at-target恢复调用；pipeline.py的最新恢复发现逻辑。
- guandan_marvel/kaggle/checkpoints.py中的save_distributed_checkpoint、load_ddp_manifest、update_checkpoint_index及实际恢复入口。
- tests/marvel_training/_serial_compare_ddp_loop_worker.py、tests/marvel_kaggle/test_t4x2_serial_compare_ddp_loop.py。

原07 Prompt只作未变合同参考，不重新执行其全套开发任务和验证矩阵。不要重写完整设计史、重复研究架构或重新核验所有旧失败归因。

本轮基线记录为git status、相关文件修改前快照/hash、本文SHA256即可。已有Codex核验基线：

- tools/serial_compare/training.py SHA256：0f92d83539863e543af178d15dad8bef0458dea58dc24e071047f2ed8a4052ce
- guandan_marvel/kaggle/checkpoints.py SHA256：52d509df4270827ba161a88484a11cf0288fdd0de7ebeccef96c818e7dfd5991
- v5 payload SHA256：bdabe3fbbc4660486a84f20b4fe252c034a7682619a381502c6b5bf7417496dd

若实际来源不同先说明，不能把新文件无声当作该基线。已有Codex最小失败证据可以引用，不必再运行长超时去“重新证明挂死”。旧probe本来断言缺陷存在，修好后不应再把该旧断言当最终通过条件。

编辑生产代码前，先在本轮证据目录保存简短REPAIR_SPEC_AND_ACCEPTANCE.md，写明协议、边界、测试节点、执行依赖与允许文件；不另做大篇幅研究报告。

## 3. 已确认根因，不再重新猜测

### A. 错误路径

training.py现有调用：

    update_checkpoint_index(run_dir,
        checkpoint_dir=destination.relative_to(run_dir), ddp_manifest=manifest)

而helper先resolve(output_root)，再checkpoint_dir.relative_to(root)，因此要求绝对checkpoint_dir。修正调用处，规范化run_dir/destination；最终索引内部仍保存run root下的相对路径，保证打包/恢复可移植。不要把路径校验全局放宽为接受任意目录，也不要存宿主绝对路径到交付恢复索引。

### B. 同iteration重复保存

DDP目录仅为checkpoints/iterXXXXXX，周期保存与最终保存可能同轮，at-target恢复也可能再次保存来源目录。真实save_distributed_checkpoint在primary上先检查目录存在并抛错，再进入barrier，导致另一rank等待。

必须定义受验证的幂等或generation方案，不能仅if path.exists(): return：

- 同一完整状态重复保存：全rank一致复用已验证恢复，不重复写或错误抛错。
- 同iteration不等于同状态。必须验证manifest/shards、每rank续训状态摘要、source/semantic/backend等适用身份；若不一致，要么全体拒绝并保留旧完整版本，要么采用显式新generation并正确更新索引。
- 已有目录只有半写/坏hash/错world size/错状态时，不能当成功或覆盖最后有效恢复。
- at-target恢复零训练，只复用/修复有效索引并进入后续评估；不能因为不应再保存却重写同目录。
- 不得只跳过最终保存：如果周期后仍有新状态，最终保存必须真正持久化。

### C. 保存事务与rank间失败收敛

把以下步骤作为完整协议设计：预检/选目标 → 创建临时目录 → 每rank写shard/summary → 汇合验证 → primary发布manifest → primary写索引 → 所有rank确认成功。

每个可能单rank发生的普通Python/I/O异常，应先形成该阶段错误状态，再由所有rank按相同顺序交换状态并共同失败。禁止rank0提前raise而rank1继续进入其他collective；索引更新也必须纳入错误传播。保留最初错误阶段和原始异常，不只给出后续barrier timeout。

写临时目录、manifest、索引须保持最后完整恢复可用，不删除唯一正确版本。索引失败不能标记本次保存完成；完整目录已发布但索引失败的状态，下次应能显式校验并修复索引或全体明确拒绝，不能误认为训练从未发生而重训。

不要在已有collective崩溃后无条件再发一个“错误广播”声称一定可恢复。rank进程死亡/传输故障仍依赖有界PG和launcher；区分可捕获I/O异常与通信进程失联，不承诺所有故障都能原地续跑。

## 4. 允许修改的文件

优先仅修改：

- tools/serial_compare/training.py。
- guandan_marvel/kaggle/checkpoints.py：确需完善通用保存事务时可改，保持旧调用者接口/manifest兼容，不破坏旧保存和恢复。
- 新测试tests/marvel_kaggle/test_serial_compare_checkpoint_transaction.py。
- 新worker tests/marvel_training/_serial_compare_checkpoint_transaction_worker.py。
- 必要的既有DDP loop测试少量接线，不能删除旧断言。
- 本轮证据、handoff/用户说明的状态更新，以及新delivery_v6构建输出。

pipeline.py仅在真实索引/幂等方案确需消费适配时允许小改并解释。不改budget/evaluation/model/算法等无关模块。新测试可使用独立有界launcher，避免更改全局launcher导致扩大回归。

若必须改其他核心文件，先在本轮说明具体依赖与必要性，不能自行扩成再次重构整个编排系统。旧Reviewer的P3按风险列表保留，不顺手全部修；新发现会阻断运行/破坏数据正确性的P1/P2则要明确报告，不能为了快速PASS藏起来。

## 5. 最小有效测试：这部分不能省

核心原则：可以替代模型计算，不可以替代被验证的保存/恢复链路。

- CPU/Gloo仅用于文件/collective控制流语义；用静态小tensor payload和最小schema完整fixture，不构造或训练CPU模型。
- 测试必须真的执行production _save_recovery、save_distributed_checkpoint、分片序列化、manifest、索引和真实rank恢复读取。不能再次把_save_recovery换成fake_save，也不能mock低层保存为“只写一个假文件”。
- 可在指定I/O边界注入失败，必须按文件/阶段精确限定；正向链路不能靠monkeypatch关闭hash/身份检查或全局断言。
- 若需模拟trainer计算以驱动真实run_training_arm，这是允许的，但checkpoint payload/状态需支持真实保存协议；报告说明它不是生产模型CUDA训练。

新增测试至少覆盖以下场景，可合并fixture/worker，避免每条都重新做昂贵初始化：

| 场景 | 必须观察到的结果 |
|---|---|
| 1. 两rank第一次保存 | 分片、manifest和索引真实存在且校验通过；两rank成功退出，索引路径相对且可解析 |
| 2. 周期保存恰在最后一轮，再最终保存 | 调用真实训练循环/保存路径；同状态幂等、无FileExistsError、无死锁 |
| 3. at-target恢复 | 新进程读取完整恢复及索引，零训练更新、无目录冲突，已有恢复hash不被无故改写 |
| 4. 周期后又有更新 | 最终恢复反映新iteration和真实新状态，不能“跳过final”丢进度 |
| 5. 同iteration但状态不同 | 按明确合同拒绝/新generation，不能静默复用旧权重 |
| 6. primary mkdir/manifest/index失败 | 有代表性的阶段分别注入；两rank都及时退出并保留原始错误；旧有效恢复可读 |
| 7. 非primary shard写失败 | 两rank共同失败，无假complete/索引指向不完整状态 |
| 8. 半写目录/坏hash/错world size | 全体有界拒绝，不能exists就当完成；保留最后有效恢复 |
| 9. 绝对/相对run_dir输入 | caller正确规范化，索引仍可移植，无法越出run root |
| 10. world_size=1旧行为 | 不新增无意义collective，旧单进程保存仍工作 |

注入错误的测试必须验证错误分类和两rank退出，不把“等进程超时被kill了”当正确同步处理。正常路径不能仅检查两rank迭代数相同，还要读取实际分片/manifest/index与最新状态。

测试PG超时建议10–15秒，单case进程总超时45–60秒；时间断言留足Windows启动抖动但明显短于PG超时。用事件/阶段标记证明是共同失败，不只比较墙钟。任何超时finally都回收本测试创建的全部子进程，不留后台孤儿。

本机已知torchrun/libuv问题不要重复探索或安装环境，沿用已可运行的显式双进程Gloo启动方式，必要时明确TCPStore use_libuv=False或已验证初始化方案。不要把Gloo语义测试报成双T4/NCCL认证。

## 6. 测试白名单与明确禁令：减少无谓耗时

### 本轮必须执行，顺序固定

1. 新保存事务测试：含真实双rank路径。先让关键测试通过，再做下面工作。
2. 一次相关快速回归：既有DDP loop + serial_compare repair消费者测试。
3. 如确实改了通用kaggle/checkpoints.py，再加相关checkpoint/index/recovery消费者的最小节点；至少检查test_progress_recovery.py中相关节点，列出为何选择，不扩成整个tests。
4. 一次独立只读Reviewer：针对本轮diff、真实保存事务、新增至少三类未由Implementer提供的边界反例，不重跑无关项目历史。
5. 通过后，构建一次v6，真实解码最终.bin做一次产物QA。

推荐命令（新文件名固定后可直接用）：

```powershell
& ./.venv-gpu/Scripts/python.exe -m pytest tests/marvel_kaggle/test_serial_compare_checkpoint_transaction.py -q -p no:cacheprovider --import-mode=importlib
& ./.venv-gpu/Scripts/python.exe -m pytest tests/marvel_kaggle/test_t4x2_serial_compare_ddp_loop.py tests/marvel_kaggle/test_t4x2_serial_compare_repair.py -q -p no:cacheprovider --import-mode=importlib
```

不要用tail管道吞掉pytest退出码。记录完整输出与真实exit code；失败后只重跑失败节点及其必要邻接项，源未变化且测试已通过则不机械重跑。

### 明确禁止的重复工作

- 不运行pytest tests -q或其他全量测试命令；本轮FULL_REGRESSION=NOT_RERUN_SCOPED_REPAIR，不能写成全量PASS。
- 不重跑53 failed/58 errors的归因脚本、cpp/native版本调查、缺失M09/M03工件测试、Guance dirty-tree验证。
- 不重复所有90项phase测试、不重新跑整套KL/NTP/架构/池数学测试，除非本轮确实修改其实现且说明直接影响。
- 不重跑704场、不做新的训练实验、不再次测试未改变的GPU计算路径、不改超参。
- 不在关键测试和审查通过前构建v6；不先造v6再用审查发现保存仍坏；不反复构建v6/v7/v8等待运气。
- 不为报告多次读取整个git历史/重复计算所有大文件hash，不另建一堆make_report/归因脚本；最小证据足够。

### GPU约束继续有效

本轮如果只改文件保存协议而未改模型、optimizer/scaler状态生成或载入语义，既有真实CUDA计算证据按原hash/范围引用，不需要为了展示GPU占用再训练一次。

若实际改到了生产trainer状态恢复/optimizer/scaler语义，才追加一次最小本机CUDA验证：生产网络结构、小batch，保存前至多1次有效更新 → 退出 → 新进程恢复 → 再1次有效更新；记录模型/输入cuda:0、状态恢复、有效更新和显存。禁止CPU模型训练替代，禁止缩网络替代。已有native身份障碍如实NOT_RUN/阻断说明，不伪造identity或关闭守卫制造PASS。

文件与静态tensor保存、CPU/Gloo通信测试不是模型训练，不要求强行占GPU。本轮真实双T4/NCCL、Kaggle仍NOT_RUN。

## 7. 速度与停止规则

- 本轮只有一个Implementer和一个全新只读Reviewer；任务之间存在同文件写冲突，不并行写。用户已选Claude Code，不为模型名重新改代理配置。
- 不从头重新启动07实施，不重新阅读所有旧Prompt，不再反复重建Agent上下文。实施者完成后交一次清楚的最小diff与测试索引。
- 新测试与相关回归预计应为分钟级；若一个命令超过10分钟，报告该命令、实际卡点与已用时间，停止无进展重试，不转为全量探索。此处是执行预算目标，不保证模型API推理耗时。
- API失败允许有限重试一次，不重新跑已完成且hash未变的测试。连续失败报告BLOCKED，不无限派发新Agent。
- 每15分钟至多一条有实际内容的进度：改了什么、通过哪一关、下一步是什么。不重复“正在工作”充数，也不因催进度重启任务。
- 一轮修复和独立验收后仍有P1/P2，保存最小失败证据并停止，不自行第四轮。不要把验收失败改名为INCONCLUSIVE以继续构建。

## 8. 构建v6：仅在保存协议验收通过后

继续使用已核验的base inputs与模型来源，不复制旧v5 payload然后只改外层hash。新源码必须真正进入overlay/driver，新的Notebook绑定新的payload/manifest。

新目录：D:/project/guandan/runs/t4x2_serial_compare_delivery_20261023/delivery_v6/。
若这个目录已存在，不覆盖或自动删它，先核对是否其他会话产物并说明。

使用现有scripts/build_t4x2_serial_compare.py的真实参数（--output、--model-dir、--base-inputs、--validation-record），沿用v5已经确认的输入路径，不凭猜测重建环境。模型目录是runs/t4x2_continue500_analysis_20261022；base-inputs从旧构建记录核实。

使用scripts/verify_t4x2_serial_compare_delivery.py --delivery <v6绝对目录> --output <本轮QA记录>对真实交付.bin运行一次。必须包含解码、文件hash、Notebook绑定、driver/overlay与最终源码一致、隔离导入；另检查包内保存入口确为本轮修复代码。QA不是训练可用的替代品，真实保存测试必须已经通过。

本轮若通用checkpoint helper有改动，必须核对它也进入实际overlay，不能只更新tools/serial_compare/training.py。旧v1–v5仍不可上传，不覆盖旧证据。

若构建后不得不再修核心源码，则该构建作废并明确原因；不得把改后源码的测试结果嫁接到改前包。不要反复自动构建新版本，报告偏离本轮“一次最终构建”的原因。

## 9. 最小证据与交接，完成后停止

新证据目录：D:/project/guandan/runs/t4x2_serial_compare_implementation_20261023/repair_round3/。

至少保留：

- REPAIR_SPEC_AND_ACCEPTANCE.md：很短即可，保存协议、选定幂等方案、各场景对应节点/命令、允许文件与顺序。
- REPAIR_REPORT.md：修改列表与增量diff/hash、实际命令/退出码/耗时、回归选择理由、未解决项。
- 新测试/相关回归/独立反例原始日志；不用复制数百KB旧失败日志。
- REVIEWER_REPORT.md：只读Reviewer真实结论；与Implementer测试证据区别。
- LOCAL_VALIDATION_R3.json：本轮测试及最终源码hash；未变CUDA计算证据以原路径/hash标记INHERITED_NOT_RERUN，注明并未证明新增双rank保存；全量和目标双T4明确NOT_RUN。可作为builder的--validation-record输入；不要重复执行旧make_local_validation/全量归因脚本来重建无关证据。
- ARTIFACT_QA.json：最终v6真实包验证。

独立Reviewer重点：路径契约、幂等验证是否真的比较状态、周期+final与at-target、单rankI/O错误的collective顺序、最后完整恢复不丢失、测试是否再次mock掉关键函数；至少三类新反例可选“索引发布失败后重试”“同iteration错RNG/optimizer摘要”“非primary分片写失败”。不重跑整个07验收，不将旧证据写成自己新跑。

更新原HANDOFF_TO_CODEX.md：本轮结果、v6路径/hash、测试日志与未运行边界；保留旧v5拒绝记录。docs说明可更新唯一候选版本，但标记待Codex复核，不能写成用户已获准上传。

最终中文回复必须有：

```text
REPAIR_ROUND: 3 (single scoped round)
CHECKPOINT_TRANSACTION_TESTS: PASS / FAIL
CLAUDE_REVIEW_VERDICT: PASS / FAIL
FULL_REGRESSION: NOT_RERUN_SCOPED_REPAIR
DELIVERY_V6: BUILT / NOT_BUILT
CODEX_REVIEW: PENDING
KAGGLE_EXECUTION: NOT_RUN
TARGET_DUAL_T4_VALIDATION: NOT_RUN
USER_UPLOAD_APPROVAL: PENDING_CODEX_REVIEW
```

同时给出实际修改文件、必要测试命令与结果、耗时、新包SHA256、HANDOFF路径和真实剩余风险。不得声称全部测试全绿或目标双T4已认证。停止，不启动远端训练、不再开启新修复循环，交给Codex检查。
