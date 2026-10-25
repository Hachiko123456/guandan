# MARVEL Guandan — Claude Code 项目规则

> **当前主线（2026-10-22 起）为 DMC**：接手先读 `docs/dmc_prompts/CURRENT_STATE.md`（状态/环境/验证协议/地雷清单）。下文的 v1.4 阶段协调流程同样适用于 DMC 阶段 prompt 的实施。本文件描述的 `marvel_v14_prompts/` 内容已冻结为历史。

## 适用范围

本文件用于 Claude Code。现有 `.opencode/` 和 `opencode.jsonc` 是可独立使用的 OpenCode 配置，必须保留，两个系统不得互相覆盖。

## 默认工作方式

对于 `docs/marvel_v14_prompts/` 下的阶段任务，主 Claude Code 会话充当协调器：

1. 一次只执行一个阶段 Prompt。
2. 先记录实施前 Git 状态、tracked diff bytes/SHA256、未跟踪文件和阶段文件 SHA256。
3. 调用全新的 `phase-implementer` 子 Agent 实施并运行验证。
4. Implementer 完成后，调用全新的 `phase-reviewer` 子 Agent 独立只读核验。
5. Reviewer 不得复用 Implementer 的上下文；先独立检查代码和数学，再参考实施报告。
6. Reviewer FAIL 时必须输出完整修复 Prompt。修复发回原 Implementer，上线一个新的 Reviewer 复验。
7. 最多自动进行两轮修复；仍未 PASS 时停止并报告用户。
8. Reviewer 明确 `VERDICT: PASS` 且 `NEXT_STAGE: ALLOW` 前不得进入下一阶段。

优先通过以下 Skills 执行：

```text
/phase docs/marvel_v14_prompts/02_macro_sequence_ppo_and_gae.md
/review-phase docs/marvel_v14_prompts/02_macro_sequence_ppo_and_gae.md
/phase docs/marvel_v14_prompts/02A_phase2_empty_rank_repair.md
/agent-smoke
```

## 工作区安全

- 工作区存在未提交修改和未跟踪文件，不得假设 Git 是干净的。
- 禁止使用 `git reset`、`git clean`、`git checkout`、`git restore`、`git commit`、`git push`。
- 不得删除、覆盖、回滚或批量格式化与当前阶段无关的文件。
- 不得降低测试容差、删除测试、跳过失败或修改验收标准来制造通过。
- 未经用户明确授权，不得启动长程训练、Kaggle 作业、高成本正式评估或最终 holdout。
- FableDan 只能作为外部评估对手，不得作为蒸馏教师、监督标签或训练数据来源。

## Agent 职责

### phase-implementer

- 可以编辑当前阶段允许范围内的项目文件。
- 必须先审计现状和数学，再修改。
- 必须增加能失败于旧实现的反例测试并运行真实验证。
- 无权批准进入下一阶段。

### phase-reviewer

- 只读、对抗式、独立核验。
- 禁止 Edit、Write、NotebookEdit 和显式写文件的 shell 命令。
- 必须直接检查真实工作区，至少构造三类 Implementer 未提供的新反例。
- PASS/FAIL 必须基于自己运行的证据；未运行项目标记 `NOT RUN`。

## 阶段报告最低要求

最终报告必须包含：

- 阶段 Prompt 路径和 SHA256；
- 实施前后 Git 状态与 tracked diff hash；
- Implementer 完整报告；
- Reviewer 完整报告；
- 实际运行命令和结果；
- 自动修复轮次；
- 最终 `PASS / FAIL / INCONCLUSIVE`；
- `NEXT_STAGE: ALLOW / DENY`；
- 遗留风险和未实施的后续阶段。

## 当前入口：阶段6工程收尾已通过（2026-10-20）

阶段5已PASS；阶段6/06B最新只读Reviewer PASS / NEXT_STAGE ALLOW，Codex核对实际报告/source与产物hash并重跑22项tie-aware/planner测试通过。完整证据在 `runs/phase6_closeout/codex_acceptance_20261020/`。全量2037 passed / 2 skipped / 1 known Guance dirty-tree failure，不能说全绿。

当前索引见 `docs/marvel_v14_prompts/CURRENT_EXECUTION.md`。不重做05/06旧返修，不从头执行06A；下一步仅在用户另行授权后做正式训练准备：冻结配置、重建当前源码交付、明确warm-start/迁移、目标设备启动/有效更新/保存恢复验证，再由用户授权Gate1。不得自动长训。

Belief B0 REJECT、B1关闭；Value INCONCLUSIVE、search未实现且关闭；Symmetry/Pool不得因工程通过自行默认开启。DELIVERY_PENDING；既有旧source CUDA pool smoke不等于当前sourceGPU再认证，单机不证明双T4/NCCL。

正式qualification前必须补齐原始样本/ranking持久化和新metric schema/provenance，注意mixed-all-tie的偏严INCONCLUSIVE，不以只写metadata的execute报告当正式资格证据。具体P3以保存的Reviewer全文为准，未做项目不能写成已完成。

真实模型/probe执行显式独立GPU Python与CUDA；规则/统计/明确单位测试允许CPU，不静默fallback。保护全局环境、OpenCode和冻结历史；正式50+iter训练、Kaggle、双T4、FableDan/holdout仍需另行授权。

## 历史阶段5续作与失败即停（当前已完成，不是下一步入口）

以下保留为阶段5历史操作规则，仅在用户明确重新打开阶段5时适用；现在应按上文阶段6入口，不恢复旧Implementer或重复返修。

用户已停止旧实现。继续阶段5使用新会话入口 `docs/marvel_v14_prompts/05B_phase5_resume_after_collection_repair.md`，不是从头重做05A，也不复用旧Implementer。

非预期失败（含编译/导入/测试/采集、CUDA/OOM、超时）一律停止等待用户明确确认，不再自动“改一下再试”。Implementer/协调器的阶段5运行命令经 `scripts/run_phase5_guarded.py run`；只读Reviewer保持原只读hook，用pytest临时目录验证，失败只返回报告而不修复重试；一次失败锁存到 `runs/phase5/execution_guard/failure.json` 后Implementer的Bash/Edit/Write被hook阻止。用户确认前协调器不得acknowledge、清锁、换状态目录、换Agent或改guard绕过。预期反例以pytest.raises等正确捕获，测试进程仍必须exit=0。

必须读取 `runs/phase5_targeted_repair/REPAIR_REPORT.md`，先做导出/编译、非零start_index/并行auto-reset边界回归，再做16局有界GPU pilot。15分钟无最小可测进展就报告阻塞，不无限探索。细则见新的续作Prompt和phase-implementer角色定义。运行器+hook不是OS安全沙箱；新的角色配置需要在新会话加载，不能保证旧Agent热更新。

