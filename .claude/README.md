# Claude Code 多 Agent 工作流

此目录提供与现有 `.opencode/` 并存的 Claude Code 配置。不要删除或迁移 OpenCode 文件；两个客户端可以按需切换。

## 文件

```text
CLAUDE.md
.claude/settings.json
.claude/.gitignore
.claude/agents/phase-implementer.md
.claude/agents/phase-reviewer.md
.claude/skills/phase/SKILL.md
.claude/skills/review-phase/SKILL.md
.claude/skills/agent-smoke/SKILL.md
.claude/hooks/validate_reviewer_readonly.py
```

## 首次使用

从项目根目录启动或重新启动 Claude Code：

```powershell
Set-Location D:\project\guandan
claude
```

先运行只读冒烟测试：

```text
/agent-smoke
```

预期输出：

```text
ORCHESTRATOR_READY
IMPLEMENTER_READY
REVIEWER_READY
REVIEWER_HOOK_ALLOW_TEST: PASS
REVIEWER_HOOK_DENY_TEST: PASS
GIT_STATUS_UNCHANGED: YES
```

## 执行阶段

```text
/phase docs/marvel_v14_prompts/02_macro_sequence_ppo_and_gae.md
```

工作流为：

```text
主 Claude 会话
  → phase-implementer
  → 全新 phase-reviewer
  → FAIL: 原 implementer定向修复
  → 另一个全新 reviewer复验
  → 最多两轮修复后停止
```

Reviewer必须同时输出：

```text
VERDICT: PASS
NEXT_STAGE: ALLOW
```

才算阶段通过；工作流不会自动执行下一阶段。


## 核验已有实现

如果阶段已经由 OpenCode 或其他会话实施，不要重新运行 `/phase`。先停止其它客户端的写入，再执行：

```text
/review-phase docs/marvel_v14_prompts/02_macro_sequence_ppo_and_gae.md
```

该入口只启动全新的只读 Reviewer，不会再次调用 Implementer，也不会自动修复。
当前阶段 2 的 OpenCode 结果若为“空 rank”单一 blocker，应执行定向修复，而不是重跑完整阶段：

```text
/phase docs/marvel_v14_prompts/02A_phase2_empty_rank_repair.md
```

## 权限说明

- Implementer可编辑当前阶段文件，但项目级设置禁止危险 Git/删除命令。
- Reviewer没有 Edit/Write/NotebookEdit权限，并由 `validate_reviewer_readonly.py` 拦截常见 shell写入、删除、Git修改、依赖安装和正式构建命令。
- `.claude/settings.local.json` 是本机权限配置，已由 `.claude/.gitignore` 排除；不要把本机密钥或授权写入共享项目配置。
- Hook是防误操作层，不是操作系统级沙箱。Reviewer仍必须遵守 Agent中的只读要求。
- pytest可能产生忽略的 cache/temp文件；Reviewer不得运行会改写正式 manifest/checkpoint/交付物的构建命令。

## 与 OpenCode 并存

保留并继续支持：

```text
opencode.jsonc
.opencode/
```

Claude Code读取 `CLAUDE.md` 和 `.claude/`；OpenCode读取 `opencode.jsonc` 和 `.opencode/`。后续迁回 OpenCode时不需要恢复或重新生成配置。

## 当前入口：04A已通过，准备阶段5/6

04A Reviewer为PASS/ALLOW，Codex复查与原始Reviewer工具结果在 `runs/gpu_runtime_completion/review_audit_20261019/`。不要恢复旧暂停任务或从头重做04A。

在新的Claude Code项目会话执行阶段5：

```text
/phase docs/marvel_v14_prompts/05B_phase5_resume_after_collection_repair.md
```

完整读取主Prompt及 `00_local_gpu_execution_contract.md`。阶段5先对称性诊断/Belief B0，B0通过才B1；stage5 Reviewer通过后停止，不自动连续跑阶段6。修订06A已准备好，但现在不要执行。

所有真实神经训练/probe/模型推理显式使用 `D:/project/guandan/.venv-gpu/Scripts/python.exe` 与CUDA。规则、环境、统计及显式CPU兼容测试保留CPU。没有CUDA或OOM要失败报告，不偷换全局CPU环境。

旧README/05/06属于冻结历史，保留原件；当前索引见 `CURRENT_EXECUTION.md`。Implementer记录完整日志、实际pytest退出码和tracked/untracked源码hash；Reviewer独立新反例、只写测试临时文件，不覆盖正式产物。正式长训、Kaggle、双T4/NCCL和最终holdout仍未授权。


### 阶段5断点续作与停止护栏

采集器已修导出名、absolute start_index与并行auto-reset耗尽槽位问题，已有16局有界CUDA pilot。新会话先读05B和 `runs/phase5_targeted_repair/REPAIR_REPORT.md`，不重写已有模块、不重做接口探索。

所有phase5测试/采集/probe由 `scripts/run_phase5_guarded.py run` 启动；一次非预期失败即failure锁存，Implementer的Bash/Edit/Write被角色hook阻止，必须报告并等用户。只有协调器在用户确认后按failure_id acknowledge；不能换状态、换Agent或改guard绕过。运行器与hook不是OS沙箱；旧Agent不保证热加载，请新建会话。


## 最新状态（2026-10-20，优先于上文旧阶段5入口）

阶段5最新只读Reviewer已PASS/ALLOW，Codex复核当前源码/最终证据和9项actor/prefix反例通过。完整材料见 `runs/phase5/review_checks/final_acceptance_20261020/`。B0 REJECT，Belief/B1关闭；symmetry默认关闭，无棋力提升声明。

用户要开启下一阶段时，在新会话执行：

```text
/phase docs/marvel_v14_prompts/06A_pool_value_search_gpu_execution.md
```

本README更新不启动阶段6。不重采512局B0、不重训probe、不恢复旧05B/05C/05D/05E任务，不自动启动正式训练/Kaggle/双T4/final holdout。设备与当前限制以CURRENT_EXECUTION.md和06A为准。


## 最新状态补充：06B工程收尾通过（2026-10-20）

阶段6条件分支工程验收PASS/ALLOW；实际Reviewer与Codex复查在 `runs/phase6_closeout/codex_acceptance_20261020/`。旧“下一步执行06A”的段落是历史，当前以CURRENT_EXECUTION.md为准。

下一步是用户另行授权正式训练准备，不是再次实施阶段6或直接长训。Value INCONCLUSIVE/search关闭，Belief B0 REJECT/B1关闭；DELIVERY_PENDING。准备需重建当前源码新交付、核对warm-start/迁移、目标硬件CUDA启动/更新/保存恢复，再另行授权Gate1。不要把旧source GPU smoke或仅metadata的qualification execute当正式验证。
