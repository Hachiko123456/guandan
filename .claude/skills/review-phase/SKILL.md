---
name: review-phase
description: 使用全新的只读 Reviewer 核验已经存在的阶段实现，不重新调用 Implementer
argument-hint: <阶段 Prompt 路径>
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash, Agent
---

对 `$ARGUMENTS` 指定阶段的**现有工作区实现**执行独立只读核验。适用于该阶段已由 OpenCode、其他 Claude 会话或人工完成，不希望再次实施的情况。

## 参数与安全门禁

1. 参数必须是项目内单个 Markdown 阶段 Prompt；为空、不存在、非普通文件或逃出项目根目录时停止。
2. 完整读取阶段 Prompt 和项目 `CLAUDE.md`。
3. 本 Skill 禁止调用 `phase-implementer`，禁止编辑、修复和自动应用补丁。
4. 未经用户明确授权，不得启动长程训练、Kaggle任务或高成本正式评估。

## 冻结当前核验基线

记录：

- 阶段 Prompt 路径、bytes和SHA256；
- 当前 `git status --short`；
- `git diff --stat`；
- `git diff --binary HEAD` 的bytes和SHA256；
- 未跟踪文件；
- 已存在的实现报告、测试日志和smoke/recovery产物路径。

如果存在另一个客户端仍在写文件，先停止并要求用户暂停/关闭该任务；不得在变动中的工作区开始核验。

## 启动全新 Reviewer

使用 Agent 工具启动新的 `phase-reviewer`，传入：

1. 原始阶段 Prompt全文；
2. 当前核验基线；
3. 现有实现报告和日志仅作为待验证声明；
4. 要求 Reviewer先独立读代码、推导数学并设计反例；
5. 要求至少三类现有测试未覆盖的新反例；
6. 要求严格区分 `REVIEWER_RUN / EXISTING_REPORT / NOT_RUN`。

不得复用原 Implementer、OpenCode Reviewer或其他历史 Reviewer上下文。

## 输出

完整保留 Reviewer报告，至少包含：

- `VERDICT: PASS / FAIL / INCONCLUSIVE`；
- blockers及文件行号；
- independent derivation/evidence；
- new counterexamples；
- test results；
- legacy/scope/checkpoint/DDP/recovery audit；
- FAIL时完整 `REPAIR_PROMPT`；
- residual risks；
- `NEXT_STAGE: ALLOW / DENY`。

本 Skill 不自动执行修复。FAIL 时把完整修复 Prompt交给用户，由用户决定使用 Claude `/phase`、原 OpenCode Implementer或人工修复。