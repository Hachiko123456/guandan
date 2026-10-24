---
name: agent-smoke
description: 只读验证 Claude Code 的 Implementer、Reviewer 和防写 Hook 是否加载
argument-hint: ""
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash, Agent
---

执行一次不会修改项目源文件的多 Agent 冒烟测试。

## 1. 基线

记录：

- 当前工作目录；
- `git status --short`；
- `.claude/agents/phase-implementer.md` SHA256；
- `.claude/agents/phase-reviewer.md` SHA256；
- `.claude/hooks/validate_reviewer_readonly.py` SHA256。

## 2. Implementer加载测试

启动新的 `phase-implementer`，明确告诉它本次不得编辑或运行测试，只返回：

```text
IMPLEMENTER_READY
CWD: <当前目录>
CAN_EDIT_NORMALLY: YES
EDITED_THIS_SMOKE: NO
```

如果格式不符，冒烟失败。

## 3. Reviewer加载和只读测试

启动全新的 `phase-reviewer`，要求：

1. 只读执行 `git status --short`；
2. 尝试执行一个不会真的写入的 Hook 反例命令描述，由主会话直接调用 Hook离线验证，不要让 Reviewer真正写文件；
3. 返回：

```text
REVIEWER_READY
CWD: <当前目录>
READ_ONLY_ROLE: YES
EDITED_THIS_SMOKE: NO
```

## 4. Hook离线测试

主会话直接向 Hook stdin发送两组模拟 `PreToolUse` JSON：

- 允许：`git status --short`，期望 exit 0；
- 拒绝：`Set-Content -LiteralPath forbidden.txt -Value x`，期望 exit 2且stderr包含拒绝原因。

不得实际执行拒绝命令。

## 5. 最终核对

再次读取 `git status --short`。Claude配置文件本身是预期的既有未跟踪文件；除这些已存在配置外，冒烟不得产生新源文件或修改任何文件。

最终输出：

```text
ORCHESTRATOR_READY
IMPLEMENTER_READY
REVIEWER_READY
REVIEWER_HOOK_ALLOW_TEST: PASS/FAIL
REVIEWER_HOOK_DENY_TEST: PASS/FAIL
GIT_STATUS_UNCHANGED: YES/NO
```

禁止运行训练、pytest、自动修复或任何阶段实现。