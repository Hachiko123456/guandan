---
name: phase
description: 使用 Implementer 与全新只读 Reviewer 执行一个 MARVEL 阶段 Prompt
argument-hint: <阶段 Prompt 路径>
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash, Agent
---

对 `$ARGUMENTS` 指定的单个阶段规范执行完整双 Agent 工作流。

## 参数门禁

1. 将参数解释为项目内一个 Markdown 阶段 Prompt 路径。
2. 参数为空、文件不存在、不是普通文件或解析后逃出项目根目录时停止。
3. 完整读取阶段文件，不得只读摘要。
4. 一次不得接受多个阶段文件。

## 第 1 步：冻结实施前状态

在不修改文件的前提下记录：

- 阶段 Prompt 绝对路径、字节数和 SHA256；
- `git status --short`；
- `git diff --stat`；
- `git diff --binary HEAD` 的 bytes 和 SHA256；
- 当前未跟踪文件清单；
- 阶段 Prompt 声明的前置门禁和禁止事项。

如果阶段 Prompt 的前置门禁失败，停止并报告，不得直接调用 Implementer 修复其他阶段。

## 第 2 步：调用 Implementer

使用 Agent 工具启动新的 `phase-implementer`，传入：

1. 阶段 Prompt 的完整内容与路径；
2. 实施前 Git 基线和阶段文件 SHA256；
3. 允许修改范围、禁止事项和后续阶段边界；
4. 要求其直接在当前工作区实施；
5. 要求实际运行 Prompt 指定的测试；
6. 要求严格使用 Agent 定义中的结构化报告格式。

等待 Implementer 完成。保存它的 agent id，以便 Reviewer FAIL 后将修复 Prompt发回同一个 Implementer上下文。

## 阶段5失败熔断与断点继续

- 阶段5执行前，读取 `scripts/run_phase5_guarded.py` 的status及续作Prompt。存在failure.json时不得派新Agent绕过、清锁或自动重试；先向用户展示failure_id/命令/日志/错误，并停止等明确确认。
- 用户确认后只由协调器按匹配failure_id acknowledge，然后启动加载最新规则的新Implementer；不得使用旧Agent上下文绕过新hook。Implementer/协调器的phase5运行命令经guarded runner，不直接发裸采集/训练命令。只读Reviewer继续用原只读hook与pytest临时目录，不显式运行写锁/日志的wrapper；Reviewer一旦非预期失败返回报告，不修复或重试。
- 所有当前阶段源码修改后必须通过编译/import/export预检再运行更大工作；预检失败也停止，不走自动修复循环。
- 本熔断独立于Reviewer两轮修复上限，且优先。负例测试应由pytest捕获并正常退出，不给真实错误加“预期失败”标签。
- 新失败没有用户确认时，只返回PARTIAL/INCONCLUSIVE、NEXT_STAGE DENY和准确日志，禁止自动继续。
- 这是一套运行器锁存、角色hook和指令的防误操作约束，不是OS沙箱；不得声称旧会话已经自动刷新hook。


## 第 3 步：启动全新 Reviewer

Implementer 完成后，使用 Agent 工具启动一个新的 `phase-reviewer`。不得复用 Implementer，也不得让 Implementer自审。

传给 Reviewer：

- 原始阶段 Prompt全文；
- 实施前 Git 基线；
- 当前 Git状态和diff；
- Implementer声称修改的文件及运行命令；
- 明确要求先独立检查再参考Implementer报告。

不要把 Implementer 的隐藏推理作为证据。Reviewer必须直接检查真实文件、运行自己的测试并构造至少三类新反例。

## 第 4 步：处理 Reviewer 结论

### PASS

只有同时满足：

```text
VERDICT: PASS
NEXT_STAGE: ALLOW
```

才停止修改并进入最终汇总。

### FAIL

Reviewer必须给出完整 `REPAIR_PROMPT`。

1. 将该 Prompt 原样发送给原 `phase-implementer` agent id；
2. Implementer只执行修复范围；
3. 修复后必须启动另一个全新的 `phase-reviewer`；
4. 不得复用前一个 Reviewer。

最多允许两轮自动修复。第二轮后仍非 PASS，停止并交给用户，不得继续试错或进入下一阶段。

### INCONCLUSIVE

停止并列出缺少的证据、未运行测试和需要用户处理的外部条件，不得视为通过。

## 测试预算控制

- 要求Implementer先跑最小反例，失败时立即修复，不要先跑全量。
- 同一源码diff上的全量回归最多运行一次；保存命令、wall time、日志和diff hash。
- Reviewer必须亲跑新反例，但可在最终diff未变时复核Implementer的全量日志，除非阶段Prompt明确要求Reviewer重新全跑。
- 不得为“更放心”连续重复A/B/C/D smoke、完整recovery和全量suite；只有相关源码变化才使旧证据失效。
## 第 5 步：最终输出

最终回复不得把子 Agent 报告压缩成一句话。至少包含：

1. 阶段文件路径和 SHA256；
2. 实施前 Git 基线；
3. Implementer完整报告；
4. 每一轮 Reviewer完整报告；
5. 自动修复轮数；
6. 实际测试结果及 `IMPLEMENTER_RUN / REVIEWER_RUN / NOT_RUN`；
7. 最终 Git状态、diff stat和tracked diff hash；
8. 最终 `PASS / FAIL / INCONCLUSIVE`；
9. `NEXT_STAGE: ALLOW / DENY`；
10. residual risks和未实施的后续内容。

禁止执行下一阶段、长程训练、Kaggle任务或最终评估，除非用户另行明确授权。