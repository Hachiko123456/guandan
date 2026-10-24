---
description: 只读验证多 Agent 是否已正确加载
agent: orchestrator
subtask: false
---

执行一次不修改任何文件的多 Agent 冒烟测试：

1. 读取 `git status --short`，不要修改工作区。
2. 调用 implementer，明确要求它本次不得编辑文件，只返回：
   - `IMPLEMENTER_READY`
   - 当前工作目录
   - 它理解自己通常拥有编辑权限，但本次不执行编辑
3. implementer 返回后，调用一个全新的 reviewer，明确要求它不得编辑文件，只返回：
   - `REVIEWER_READY`
   - 当前工作目录
   - 它理解自己是只读核验者
4. 最终输出：
   - `ORCHESTRATOR_READY`
   - implementer原始返回
   - reviewer原始返回
   - 冒烟前后 `git status --short` 是否相同
5. 禁止运行测试、禁止修改文件、禁止自动修复。
