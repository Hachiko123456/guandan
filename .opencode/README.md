# OpenCode 多 Agent 配置

本项目配置三个角色：

- `orchestrator`：主编排 Agent，不编辑文件；
- `implementer`：实施子 Agent，可编辑当前阶段文件；
- `reviewer`：独立只读核验子 Agent。

## 首次加载

创建或更新配置后，请关闭当前 OpenCode 项目会话，再从 `D:\project\guandan` 重新打开。无需重启 Windows。

## 冒烟测试

重新打开后执行：

```text
/agent-smoke
```

预期最终输出包含：

```text
ORCHESTRATOR_READY
IMPLEMENTER_READY
REVIEWER_READY
```

并且冒烟前后 `git status --short` 相同。

## 执行阶段

```text
/phase docs/marvel_v14_prompts/02_macro_sequence_ppo_and_gae.md
```

实际执行前，应把路径替换成当前已获准执行的阶段或修复 Prompt。Reviewer明确PASS前不要执行下一阶段。

## 安全说明

不要使用 `--auto` 跳过 `ask` 权限确认。实现和核验使用同一个模型，但处于不同Agent上下文，Reviewer禁止使用编辑工具并被要求独立构造反例。
