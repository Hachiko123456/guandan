---
description: 由实现 Agent 实施指定阶段，再由独立 Agent 核验
agent: orchestrator
subtask: false
---

请对以下阶段规范执行完整双 Agent 工作流：

$ARGUMENTS

要求：

1. 将参数解释为阶段规范文件路径并完整读取。
2. 冻结修改前 Git 状态和 diff hash。
3. 调用 implementer 实施。
4. implementer 完成后调用全新的 reviewer 独立核验。
5. reviewer 不得编辑文件。
6. reviewer 必须构造原测试之外的反例。
7. reviewer FAIL 时必须生成完整修复 Prompt。
8. 最多自动执行 2 轮修复，每轮使用新的 reviewer 会话。
9. 最终输出 implementer 和 reviewer 的完整报告。
10. reviewer 明确 PASS 前不得允许进入下一阶段。
