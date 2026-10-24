# 术语表 · 先把几个“步”和“完成”分清

[项目全景](00_overview.md) · [训练原理](02_training.md) · [操作手册](03_runbook.md)

> 适用范围：当前项目文档用语，核对日期 2026-10-21。字段的精确定义以对应源码/schema 为准。

## 1. 游戏与数据计数

| 术语 | 通俗解释 | 容易混淆的点 |
|---|---|---|
| seat / player | 座位 0–3，队伍固定 0/2 对 1/3 | 一个共享策略可控制多个座位，不是四套独立模型 |
| hand / round / game / 一局 | 本训练语境通常指一次发牌到本局终止 | 不等于多副升级比赛；其他文件用词需结合上下文 |
| context | 级牌、leader、previous_result 等局面起始条件 | 相同 seed 不代表不同 context 下是相同任务 |
| deal | 基础发牌 | 可按座位轮换用于多次对战，不等于多份独立样本 |
| token | 事件/动作协议中的编号 | 不是聊天文本 token |
| token step / env step | 一次协议步骤推进 | 不等于打出一手牌；也可能是非玩家步骤 |
| prefix | 尚未提交的动作构造前缀 | 不是公开已经出牌的局面 |
| COMMIT | 提交构造好的完整决定 | 之前的 token 不能逐个当成已经出牌 |
| PASS | 不出，当前协议中是一 token 提交动作 | 仍是一次完整宏决定 |
| committed step / game step | 按采样器当前口径统计已提交的游戏决定，含 PASS | 不是整局；具体口径见 commit 标记与计数函数 |
| macro action / 宏动作 | 玩家从空前缀到 COMMIT/PASS 的完整策略段 | 长度是本段 token 数，不是到自己下次决定的距离 |
| Delta / tau | 同一玩家相邻宏起点之间的基础 token 距离 | 包括中间其他玩家/机会步骤，排除 padding |
| trajectory / rollout item | 记录一局交互的序列 | 本主线训练要求完整终局，不能把半局当完成 |
| padding | 为固定张量容量补出的无效位置 | 不能进入奖励、优势、均值、分段和计数 |
| horizon / capacity | 轨迹/候选/事件的容量上限 | 不是实际局长，也不是允许静默截断的许可 |
| terminal | 真实局终止 | 不等于 timeout 或数据不足 |
| truncation | 未完成就被截断 | 不能冒充正常 terminal |
| cluster | 一份独立 deal 及其相关轮换/样本的分组 | 不能把组内样本当全部独立 |

## 2. 训练与优化计数

| 术语 | 含义 | 注意 |
|---|---|---|
| rollout_batch_size | 活动采样并行环境批量 | 不必等于一轮学多少完整局 |
| collect_batch_size | 每轮收集的完整轨迹批量 | DDP 要分清 global 和 per-rank |
| iteration | 一次采集、目标计算、多次优化与完成记账 | 一轮可能有多局、多次 optimizer step |
| epoch | 在该轮采集的数据上再遍历一遍优化 | 不等于新采一批牌 |
| minibatch | 一次优化使用的轨迹子集 | 配置 mini_batches 是组织参数，不总等于实际非空批数 |
| optimizer step | 一次优化器更新 | AMP 溢出可能跳步，不能把调用次数当有效参数更新 |
| durable update | 完整成功迭代边界的更新记账 | 不是每个 token、minibatch 或半轮进度 |
| global batch | 全部 rank 的合计批量 | 不能复制到每个 GPU 后仍称相同全局预算 |
| rank / world_size | 分布式进程编号 / 总进程数 | 常见一 rank 对应一 GPU；以实际配置为准 |
| wall budget | 墙钟时间限额 | 应留出保存、recovery 与输出复制时间 |

**教学示例**：每轮采 20 局、做 4 个 epoch，不是完成了 80 局。若实际分成 5 个非空 minibatch，则每轮可能有 20 次 optimizer step，但仍是 1 个 iteration；是否有效更新还要查真实状态。

## 3. 算法与模型

| 术语 | 含义 |
|---|---|
| Actor / policy / π | 根据可见信息给出合法候选概率的策略 |
| Value / V | 对当前玩家局面后续收益的估计，不等于搜索器 |
| centralized critic / Q | VRPO 等路径可用的训练期集中式评价分支，不是 IPPO Actor 偷看手牌的理由 |
| self-play | 模型与自身策略进行对弈，通常共享参数但隔离信息 |
| IPPO | 当前项目独立 PPO 路径；四座位可共享同一个模型 |
| VRPO | 保留的 Q-critic、Q-boost/辅助训练路径，不是宏 IPPO 的别名 |
| on-policy | 轨迹行为分布与所声明的采样策略一致；本项目训练温度要求为 1 |
| GAE | 把 TD 残差沿轨迹组合成优势估计的方法 |
| advantage / A | 相对于价值预期，这个决定有多好 |
| gamma / γ | 环境折扣；当前仍按基础 token 时钟 |
| lambda / λ | trace 传播衰减；宏决策模式按同玩家宏决定计 |
| PPO ratio / clip | 新旧动作概率比及限制更新的裁剪目标，不是硬 KL 保证 |
| Sequence PPO | 宏内 log-prob 求和，对完整决定的 ratio 裁剪一次 |
| KL regularizer | 约束策略分布的附加损失；本项目有 token/宏归一化两种语义 |
| Actor Fuse | 状态/动作特征的可学习残差融合分支 |
| NTP | next-observation token prediction 辅助头，是否构建由权重配置决定 |
| Belief | 对未知信息的建模条件分支，必须先通过所需资格 |
| Symmetry | 利用座位/重复牌等对称关系的条件训练机制 |
| opponent pool | 历史/冻结对手集合；评估池和训练接线不能混为一谈 |
| policy owner / train mask | 标识谁作出动作、哪些样本可更新当前策略 |
| AMP | 混合精度；当前 CUDA 模型前向与关键 FP32 数学有明确边界 |
| DDP | 分布式数据并行，需正确同步梯度和全局有效样本统计 |
| KV cache | Transformer 按观察者维护的历史键值缓存 |
| EMA / merged policy | 可训练策略的合并输出路径；具体平滑规则与 beta 看实现/配置 |

## 4. 身份、恢复与结论

| 术语 | 含义 |
|---|---|
| checkpoint | 按 schema 保存的权重/训练状态文件，不一定都允许续训 |
| policy-only / warm start | 使用获准兼容权重初始化新任务，不恢复完整优化/采样进程 |
| exact_resume | 身份、语义、运行状态等满足精确恢复检查 |
| compatible_resume | 分类器允许的配置差异下继续；须保留差异原因 |
| migration | 存在需要显式迁移的变化，不是自动忽略不匹配 |
| fingerprint / hash | 对源码、配置、二进制、数据等内容的身份摘要；不是仅凭文件名识别 |
| manifest | 带版本和身份的文件/运行/数据说明清单 |
| provenance | 实际数据来源记录；不是只写计划采了什么 |
| resolved config | 完成 profile/默认值等解析后的实际运行配置 |
| schema / semantic contract | 数据结构版本 / 训练含义约束 |
| native identity | 已加载 C++ 扩展的构建来源/二进制身份 |
| recovery bundle | 跨会话恢复所需工件和 manifest 的集合，不只是单个权重 |
| candidate / champion | 待比较模型 / 经指定政策保留的代表模型 |
| holdout | 不参与训练和选模的独立最终评估数据 |
| CI / clustered CI | 不确定性区间 / 按独立 deal 分组的区间；不是训练 loss |
| EV | 解释方差；目标方差为零时未定义，不得硬填一个好分数 |
| Gate | 训练阶段、棋力晋升或 Value/Search 资格检查；必须说明哪一种 |
| PASS / FAIL | 在明确范围和协议下通过 / 失败，不代表其他范围 |
| INCONCLUSIVE | 证据不足或条件未满足，不能宣布通过 |
| NOT_RUN / OFF | 未执行 / 关闭；不是失败，也不是已认证 |
| dry-run | 只规划/检查，不执行训练；计划数量不是实际完成量 |

想看公式和代码如何对应，继续读[训练篇](02_training.md)；想知道什么证据可以支持棋力结论，读[评估篇](04_evaluation.md)。
