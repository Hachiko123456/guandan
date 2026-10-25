# 多 Agent 模型、推理强度与跨客户端迁移评估

日期：2026-10-24。范围：现有项目配置静态审计，以及 WorkBuddy/CodeBuddy、ZCode 官方文档核对。本文件是迁移设计与验收依据，不是已经生效的客户端配置。

## 1. 当前配置事实

| 配置 | 模型声明 | 推理强度声明 | 其他运行参数 |
| --- | --- | --- | --- |
| `.claude/agents/phase-implementer.md` | 第6行 `model: inherit` | 未声明 | `maxTurns: 200`、`permissionMode: acceptEdits` |
| `.claude/agents/phase-reviewer.md` | 第6行 `model: inherit` | 未声明 | `maxTurns: 160`、`permissionMode: default` |
| `.opencode/agents/orchestrator.md` | 未声明 | 未声明 | `mode: primary`、角色权限 |
| `.opencode/agents/implementer.md` | 未声明 | 未声明 | `mode: subagent`、角色权限 |
| `.opencode/agents/reviewer.md` | 未声明 | 未声明 | `mode: subagent`、角色权限 |

- 项目内已审计的 Claude/OpenCode 设置和工作流没有固定具体供应商模型 ID。
- 当前聊天提供的“优先使用 deepseek-coder”是委派偏好，不等于项目内已配置了同名模型路由。Agent 名称、提示词正文、真正的 API 模型 ID 是不同内容。
- 主会话实际模型、用户级覆盖、运行进程环境、网关实际路由未在本次审计中认证。不能凭项目文件推断所有运行时参数。
- 根目录当前没有磁盘上的 `AGENTS.md`、`CODEBUDDY.md` 或 `.codebuddy/`；当前聊天注入的规范不能自动视为已写入磁盘。
- `maxTurns` 是代理执行轮数上限，不是思考 token 数、输出长度、上下文窗口或超时。

静态复核命令（PowerShell，项目根目录）：

```powershell
rg -n --hidden '^(model|effort|reasoningEffort|reasoning_effort|thoughtLevel|thinking|maxTurns|permissionMode)\s*:' .claude/agents .opencode/agents
rg -n --hidden 'deepseek-coder|reasoningEffort|reasoning_effort|thoughtLevel|MAX_THINKING_TOKENS|effort:|thinking:|model:' .claude .opencode CLAUDE.md docs/AGENT_EXECUTION_PROTOCOL.md opencode.jsonc
```

## 2. 主模型 A 与子模型 B

主模型 A、实施 B、审查 C 可以是正常的多模型协作。必要条件是客户端支持独立选择模型，模型 ID 能解析到正确供应商/端点，凭证和权限可用，模型支持角色需要的工具及上下文。

需要区分以下结果：

| 情况 | 可能结果 | 项目要求 |
| --- | --- | --- |
| B 可用且适配完整 | 按 B 执行，主模型仍为 A | 记录实际路由 |
| B 不在本地模型注册表或被禁用 | 客户端报错或回退 | 必须显示差异，不能把回退写成 B 已执行 |
| B 的 API 无权限、名称错误或参数非法 | 请求失败 | 保留原始错误；按当前任务失败规则处理 |
| B 名称被同名别名重新映射 | 可能调用另一模型/供应商 | 记录别名展开后的模型和供应商 |
| B 被高优先级设置覆盖 | 实际运行另一模型 | 检查配置来源和调用参数 |
| B 配置字段不被新客户端识别 | 可能忽略或加载失败 | 静态字段检查加运行验证 |

不要把代理正文中的“你使用某模型”当作运行控制，也不要把模型自己的身份声明当作证据。客户端解析结果、请求元数据和可用的服务端返回元数据才是依据；网关仍可能二次路由，证据不足时标为 UNKNOWN。

## 3. WorkBuddy/CodeBuddy 的核对点

WorkBuddy 官方项目文档明确支持 `.codebuddy/` 下的项目 agents、skills、commands、rules。具体执行行为仍需由目标 WorkBuddy 版本验证，不能仅凭最新 CLI 文档宣布桌面端已实测。

当前 CodeBuddy 子代理文档列出 `model` 和 `effort`：`effort` 可省略并继承会话强度，具体模型支持哪些档位仍需另查。`model: inherit` 不应解读为绕过其他配置的绝对锁定。

官方为内置子代理列出了环境变量、单次调用、项目设置、用户设置、产品默认和主模型兜底的路由顺序。特别检查：

- `CODEBUDDY_CODE_SUBAGENT_MODEL`：文档同时说明可统一覆盖子代理；对自定义角色须检查目标版本实际结果。
- Agent 工具本次调用的 model 参数。
- 项目及用户级 `subagents.agents.<角色名>.model`。
- 自定义角色 Markdown 中的 model 声明，以及与上述设置的冲突。
- 使用 `lite` / `reasoning` 时，检查 `variantModels` 及对应环境变量的最终展开。
- 子代理管理界面中的 `fallback-main` 等来源标签；该标签在文档中表示配置模型被禁用或本地未知后回到主模型，不能推断所有网络/API错误都会回退。

不要将内置代理的解析顺序未经验证推广为所有自定义角色字段的精确优先级。

WorkBuddy 的模型模式、推理强度、上下文窗口和 Max 模式是不同设置。官方模型文档中的 Max 模式控制上下文保留/压缩，不等同于 `effort: max`。

## 4. ZCode 的差异

当前官方子智能体文档说明：

- `model: inherit` 或省略 model 时，跟随主 Agent 当前模型。
- 推理档位字段为大小写敏感的 `thoughtLevel`。
- `thoughtLevel` 仅在显式指定具体 model 时生效；继承模式下跟随主 Agent 推理配置。
- `reasoningEffort` 等不认识的字段可能静默忽略。
- 修改定义文件后应新建会话；不要假设已有会话热更新。

因此，不能把 CodeBuddy 的 `effort` 原样复制到 ZCode，也不能假设“继承模型但单独提高 Reviewer 档位”的配置在各客户端都受支持。

## 5. 推理与能力适配

迁移时对每个角色分别验证：

1. **推理开关与档位**：同名 high/max 不是跨模型等量算力；未知档位可能报错、忽略或被客户端归一化，需验证实际行为。
2. **预算单位**：模型推理 token、输出 token、上下文窗口、代理轮数和墙钟超时分别设置。不能用增加 maxTurns 代替提高推理档位。
3. **协议转换**：客户端角色字段和 API 请求字段不是同一层；不要直接把供应商参数塞进任意 Agent Markdown。第三方端点是否支持目标参数要单独验证。
4. **工具能力**：至少验证文件读取、搜索、终端调用和结构化工具参数；实施者另需编辑/补丁能力。非原生工具模型即使文本回答正常也不代表能执行工作流。
5. **上下文与压缩**：给每个子代理明确原始需求、修改范围、当前源码基线和验收标准；切换模型不能替代交接，主会话上下文也不能假设自动全部注入子代理。
6. **独立审查**：Reviewer 应使用全新上下文且先形成独立检查，再看实施者说明；不要注入实施者的持久记忆。不同模型有助于形成不同视角，但不自动保证正确。
7. **记忆与权限耦合**：当前 CodeBuddy 文档提示某些 memory 设置会自动补充 Read/Write/Edit。迁移只读 Reviewer 时先不启用持久记忆，并检查最终工具集合与禁止项。
8. **权限与 Hook**：换模型不能改变角色权限；重点检查 frontmatter hooks 是否被信任闸门跳过，脚本入口、工具名和工作区变量是否适配。终端权限仍需命令拦截，Hook 不是 OS 沙箱。
9. **重试与修复**：统一管理 API重试、代理轮数和业务修复轮次，避免多层自动重试叠加。已有阶段5失败即停规则不可被迁移适配覆盖。

## 6. 建议的可移植设计

分离三层：

- 公共流程：规划、文件修改边界、独立核验、失败闭环、证据格式。写入项目公共规范和角色正文。
- 客户端适配：WorkBuddy/CodeBuddy、Claude Code、OpenCode、ZCode 各自支持的文件路径、字段、权限和 Hook 接口。
- 本机模型绑定：供应商、实际模型 ID/别名、端点、受支持推理档位；凭证只存客户端本地配置，不写共享文档。

建议定义语义角色 `coordinator`、`implementer`、`reviewer`，不要以具体模型命名角色。若需落实 deepseek-coder 偏好，应在本机把 implementer 映射到实际可用的 DeepSeek 模型 ID，而不是仅改角色名字。

提供两种明确模式：

| 模式 | 规则 | 适用情况 |
| --- | --- | --- |
| 跟随主会话 | 角色模型继承；推理继承或在客户端支持时单独适配 | 经常更换主模型，优先简化配置 |
| 固定角色绑定 | 各角色绑定具体模型及已验证档位 | 需要稳定分工、对比或可追溯结果 |

在一次阶段任务开始时记录最终模型配置并冻结本轮选择。中途切换主模型时，新派发子代理可能跟随新模型；旧子代理/恢复会话如何处理依客户端而定。需要换模型就明确记录边界并重新启动对应角色，不假设自动一致。

建议的角色质量要求（不是可直接执行的客户端配置）：

- coordinator：能理解阶段约束和复核证据，采用经过验证的推理档位。
- implementer：可靠工具调用与代码修改；数学、checkpoint、DDP 等任务使用已验证的较高推理档位。
- reviewer：独立上下文、可靠终端测试、较高推理档位，不为降成本自动换成未经验证的轻量模型。
- 模型或推理配置无法满足本轮已选策略时，报告 INCONCLUSIVE/配置不满足；不得静默变更后宣称按原策略完成。

## 7. 迁移验收条件

先用有界冒烟验证，业务源码不作迁移测试靶标。真实 API 调用会产生使用量，应在具体迁移任务明确范围后执行。

| 验收项 | 通过条件 | 本次状态 |
| --- | --- | --- |
| 静态模型审计 | 查清项目内所有角色模型与推理声明 | PASS：见第1节及复核命令 |
| 官方差异核对 | 查清目标字段、继承、覆盖、回退语义 | 文档核对完成；目标安装版本未验证 |
| 角色加载 | 实际加载目标角色，不由主会话模拟身份 | NOT RUN |
| 模型解析 | 记录请求绑定及实际可观测模型、供应商、配置来源 | NOT RUN |
| A/B 分工 | 主 A 子 B 的绑定符合选择；无未披露覆盖/回退 | NOT RUN |
| 继承模式 | 主模型改变后，新调用的继承行为符合目标版本 | NOT RUN |
| 推理档位 | 支持的档位确实传递；不支持时错误或归一化可见 | NOT RUN |
| 负例 | 无效模型/档位在临时配置中被发现，回退可见 | NOT RUN |
| 工具与只读 | 实施工具可用，Reviewer 无编辑能力，终端写入被阻止 | NOT RUN |
| 上下文独立 | Reviewer 实例全新，无实施者持久记忆污染 | NOT RUN |
| 工作区保护 | 比较源码内容/哈希和 Git状态，原改动不被覆盖 | NOT RUN |

建议记录：客户端版本、角色定义哈希、配置模式、角色实例 ID、模型声明、最终解析模型、供应商/端点标识（不含密钥）、推理声明与实际可观测值、配置来源、回退状态、工具与Hook加载结果。拿不到的值标 UNKNOWN，不能让模型自行编造。

当前 agent-smoke 只验证角色就绪、部分Hook与Git状态，没有模型/推理解析断言；迁移时需要扩展验收。

## 8. 执行计划与修改边界

1. 选择跟随模式或固定角色模式，并记录目标客户端版本和可用模型注册项。
2. 新建公共规范和 `.codebuddy/` 适配文件；保留 `.claude/`、`.opencode/`，不改业务源码。
3. 检查字段、模型别名、覆盖来源、记忆开关、权限及Hook注册；不要直接照抄不认识的字段。
4. 执行第7节有界冒烟，保存可脱敏证据；失败时修正适配后独立复验。
5. 全部通过后，才接续 CURRENT_EXECUTION.md 指定的当前任务；不因迁移自动重做历史阶段或启动训练/远端任务。

本轮只新增本文档；未修改模型、权限、Hook、用户全局设置或业务代码，未调用目标模型 API。静态核对不能代替运行验收。

## 9. 官方资料

- [WorkBuddy 项目配置](https://www.codebuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Function-Description/Project)
- [WorkBuddy 模型与推理设置](https://www.workbuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Function-Description/Model)
- [CodeBuddy 子代理、effort、模型解析、memory](https://www.codebuddy.cn/docs/cli/sub-agents)
- [CodeBuddy 模型环境覆盖](https://www.codebuddy.cn/docs/cli/env-vars)
- [CodeBuddy Hook 配置与信任闸门](https://cloud.tencent.cn/document/product/1831/137030)
- [ZCode 子智能体与 thoughtLevel](https://zcode.z.ai/cn/docs/subagents)

以上为2026-10-24查阅的在线文档；目标桌面客户端安装版本和网关行为仍待实测。
