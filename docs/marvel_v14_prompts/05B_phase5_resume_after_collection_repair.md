# 阶段5断点续作：采集器修复后继续，失败即停

更新日期：2026-10-19。本文件是阶段5的续作入口，不是新增开发阶段，也不是阶段6。

## 0. 必须先读，不能从头探索

完整读取：

1. `05A_symmetry_belief_gpu_execution.md`：原阶段5目标与科学Gate。
2. `00_local_gpu_execution_contract.md`：GPU、预算、证据、冻结文件和禁止事项。
3. `runs/phase5_targeted_repair/REPAIR_REPORT.md`：本次修复、已通过命令、实际源码hash和未验收项。
4. `CLAUDE.md`、`.claude/agents/phase-implementer.md` 和当前真实phase5模块。

继承原05A全部工程/科学门禁，下面“非预期失败即停”优先于原来的自动修复表述。用户已停止旧Agent；用新的Claude Code会话加载本入口及最新角色hook，禁止复用旧Implementer，也不重新实施阶段1–4。

## 1. 已完成的修复与证据

当前代码已有阶段5部分模块，不是空白项目。Codex已修复：

- `__all__`误写`transform_candidates`，现导出真实`transform_candidate_rows`；编译/普通导入/星号导入均有回归。
- `start_index`非零时，后续自动发牌的absolute deal index和context统一；计数必须镜像每一个native auto-reset，包括不再记样本的槽位。
- 请求范围外的快槽位不再构造/校验虚假meta，finite context schedule绕回不能污染数据或触发假context mismatch；返回数据只含请求区间。
- 参数边界在policy/engine初始化之前拒绝；timeout返回unfinished元数据，不伪称采集完成。
- 已新增独立协议替身测试，刻意使快槽位反复reset、慢槽位延迟；不是只测deals=batch_slots的简单路径。

已有真实GPU pilot：16个完整deals、4槽位、起点48、跨入52之后的贡还context，取得128个样本。模型保持冻结，设备cuda:0、FP16 autocast、无CPU fallback；具体耗时/显存/indices看修复报告及JSON，不把静态文字当作亲跑证据。

保留已有修复和测试，不重做全仓接口测速。需要重跑pilot时仅限collector/engine接口相关变更，先跑最小回归。

## 2. 失败即停：不得再连续试错

Implementer的阶段5编译/测试/采集/神经probe/训练smoke，以及协调器启动的写产物任务，通过此运行器执行：

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase5_guarded.py run --label <unique-label> --timeout-seconds <bounded-seconds> -- <Python script or -m module and arguments>
```

运行器使用同一个GPU Python启动子进程，先做phase5源码编译/import/public-export预检，保存完整stdout/stderr、真实exit code与耗时。

**任一非预期失败、超时、CUDA assert/OOM或中断：一次就锁存，并停止等待用户。** 不自动改完重跑，不加timeout、换seed/batch、换Agent或更换state目录绕过。`runs/phase5/execution_guard/failure.json`存在时，Implementer的Bash/Edit/Write/NotebookEdit被hook拒绝，仍可Read失败记录。

此时输出：

```text
IMPLEMENTATION_STATUS: PARTIAL
STOP_REASON: FAILED_AWAITING_USER
failure_id / exact command / error / log path / modifications so far
NEXT_STAGE: DENY
```

只有该失败之后的直接用户确认，协调器才可以执行：

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase5_guarded.py acknowledge --failure-id <the exact reported id>
```

acknowledge仅解锁，不自动重跑；Implementer不能自行acknowledge/删除failure/修改guard。预期负例必须用`pytest.raises`/明确断言捕获，测试进程仍exit=0，不能将真实采集失败包装成“预期”。这是工作流防误操作层，不是OS沙箱，不能声称旧Agent已自动热加载。只读Reviewer例外：遵守原只读hook，用pytest临时目录亲跑验证，不直接启动这个显式写锁/日志的运行器；Reviewer遇到非预期失败须返回FAIL/INCONCLUSIVE并停止，不修复、不自动重试。

纯只读初始审计限15分钟；没有明确最小实现或可测结论时报告阻塞、等待用户，不继续一小时的无边界探索。

## 3. 准确的第一步：现有代码回归，不是重写或重训

先检查：

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase5_guarded.py status
```

若READY，以新基线记录源码/Prompt/hash和现有证据，然后执行：

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase5_guarded.py run --label phase5-resume-preflight --timeout-seconds 90 -- -m pytest tests/marvel_training/test_phase5_collection_import.py tests/marvel_training/test_phase5_collection_boundaries.py tests/marvel_training/test_phase5_execution_guard.py -q -p no:cacheprovider --import-mode=importlib
```

有一次非预期失败立刻按上一节停止。通过后复核已存的16局pilot source identity；若collector/engine未变化，不重复跑同一pilot。

需要重验真实GPU时使用有上限的现成CLI，不重写here-doc测速：

```text
D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase5_guarded.py run --label phase5-pilot16 --timeout-seconds 120 -- scripts/run_phase5_collection_pilot.py --device cuda --deals 16 --batch-slots 4 --start-index 48 --max-wall-seconds 90 --output D:/project/guandan/runs/phase5/pilot/resume_pilot16.json
```

这个CLI固定最多16局，不能当成完整B0 runner；它只做冻结策略采集，没有backward/optimizer。输出路径限定在pilot目录，不污染正式交付。

## 4. 在现有实现上继续完成原阶段5

在编译/导出/边界/pilot门禁通过后，先列出尚缺工程与反例，按原05A依次完成：

1. 对称性合法映射、可逆性、restricted/full macro概率比较及默认关闭等价。
2. B0可审计CLI、canonical deal group split、可见输入和隐藏标签隔离、四组基线与主要指标/CI；当前模块里的设计声明不是已完成实验。
3. 在锁定设计与预算下运行原05A授权的有界B0，神经probe与encoder使用CUDA，规则/环境/非神经统计可CPU。不要因为16局pilot成功就声称512局B0通过。
4. B0 PASS才实现B1；REJECT保持belief关闭，INCONCLUSIVE报告缺证据并停止该条件分支，不能换seed反复追求显著或自动扩预算。
5. 按实际改动补齐checkpoint/export/strict-resume/DDP/feature-off回归。不得为了赶进度绕过未知shape/key或修改历史冻结文件。
6. source/config/数据/模型变化后，旧phase4 ZIP只作历史交付，不能沿用旧hash冒充新功能可交付。

每次命令执行前说明目的、最大耗时和预期产物；超过30秒仍无结果时报告状态，不允许并发发起重复采集。不得在4GB GPU同时常驻全数据与多个生产模型。

任务内的小规模CUDA测试与有界B0遵循原05A预算，不等于授权50/250/500/1000/1500 iter正式训练、Kaggle、双T4/NCCL或final holdout。绝不实施阶段6或FableDan蒸馏。

## 5. 验收与停止

先最小测试，再相关回归；只有最终源码冻结后才全量一次。所有非预期失败仍立即停止，不通过“Reviewer最多两轮”规则继续自动试错。

Implementer报告完整的当前实现/未做项目、B0结论、source/dataset/checkpoint/hash、GPU实际设备/精度/显存/耗时与全部命令。全新只读Reviewer运行自己的反例；源码不变时复核完整全量日志，不能把旧报告当亲跑。

工程验收和科学Gate分开；pilot通过不等于阶段5通过，辅助loss下降不等于棋力提升。Reviewer必须输出`VERDICT: PASS|FAIL|INCONCLUSIVE`和`NEXT_STAGE: ALLOW|DENY`；只读Reviewer发现FAIL时输出修复Prompt并停下，用户确认之前不自动发回Implementer。

完成后停止，等待用户决定是否进入阶段6，不连续推进。
