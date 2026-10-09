# M05 差分与性能补修协议

- 原始实现：`4ba65bfa20df2843fbc3cefb5299e885c14e2479`
- 补修阶段：`M05_differential_performance_repair`
- 原 10,000 局 / 100,000 状态证据保留，不覆盖、不改写

## 1. 公共 backend 导入契约

`guandan_marvel` 包初始化必须保持 extension-free，并通过延迟加载暴露公共 API：

- `guandan_marvel.default_game` 或 `get_game_backend()`：默认 C++ backend；
- `guandan_marvel.cpp_game` 或 `get_game_backend("cpp")`：显式 C++ backend；
- `guandan_marvel.python_game` 或 `get_game_backend("python_reference")`：显式 Python oracle；
- `guandan_marvel.game`：历史 Python-oracle 模块，必须与 `import guandan_marvel.game` 身份一致。

访问 config/model 基础模块不得加载原生扩展。显式 Python oracle 不依赖扩展存在；显式/默认 C++ 缺失时必须抛出 `BackendUnavailableError`，禁止 fallback。

## 2. Checkpoint backend 与构建身份

训练 checkpoint schema 升级为 `guandan-marvel-training-checkpoint-v4`，semantic contract 升级为 `guandan-marvel-training-contract-v3`。Checkpoint 必须保存：

- 实际 `backend`，不得只保存 `default`；
- hash-protected `backend_identity`；
- 与实际 backend 对应的 source fingerprint；
- C++ extension 实际路径、binary SHA-256、编译器、C++ 标准、Python ABI；
- extension 编译时嵌入的原生 source digest；
- checkout 中存在原生源码时，当前 source digest 与 embedded digest 的一致性。

Python backend 的身份生成不得导入 C++ extension。C++ 源码与加载二进制不匹配时拒绝建立可 exact-resume 的 trainer。Backend、binary、embedded source、wrapper、关键 header、semantic contract 或受保护语义发生变化时分类为 migration/reject，不允许 Python/C++ 自动跨 backend resume。

旧 v3 checkpoint 缺少 backend/build identity：训练恢复分类为 migration，显式评估可按既有 evaluation-only 策略处理；不得用当前 backend 自动补全并称为 exact。

## 3. 性能口径修正

原报告中 2.976× 的 lane 实际执行确定性 selector：

`position = (case.seed + 17 * step_index) % legal.numel()`

它覆盖观测转换、固定选牌规则和环境推进，但没有 Transformer forward。历史结果保留，统一改称 **deterministic selector / fixed-action-rule rollout**。

补修新增真实 frozen Transformer Agent rollout：相同冻结权重、eval/inference、相同物化 deal、greedy temperature=0、相同设备/线程、匹配动作/奖励 trajectory digest，至少两次且交替 backend 顺序。分别记录 cache 准备、Transformer forward with KV cache、sampling/action marshalling、environment step、observation conversion、trajectory buffer/commit 和总 rollout。

## 4. 保守重新验收决策

总体计划把 end-to-end rollout 列入 M05 的 >=2× 性能比较，而原始 2.976× lane 并非真实模型。为避免根据结果临时降级门槛，本轮采用保守口径：

- single-env 与 batch 环境结果继续作为历史 hard evidence；
- deterministic-selector 结果保留但不再代表神经模型；
- **真实 frozen Transformer rollout 必须达到 >=2× 才能重新 accepted**；
- learner 自身不要求 >=2×，但完整 IPPO iteration 的 rollout、learner 和 total 必须如实报告。

若真实 Agent lane 未达到 2×，M05 保持未接受，M06 保持 blocked；不得以环境 microbenchmark 替代完整 rollout 门槛。

## 5. 继承差分证据与本轮重放

若规则、动作、观测、奖励和环境执行语义文件相对原实现未变，可继承原 10k/100k 证据，但必须：

- 校验全部 corpus shard SHA-256、大小、记录数和动作数；
- 确认受保护规则与 Python/C++ 执行语义文件未变；
- 从原 corpus 完整回放全部 13 level；
- 重跑 directed wild、bomb、four-kings 和隐私场景；
- 将原 100,000 状态明确描述为已比较完整轨迹中的随机非终局子集。

## 6. 提交身份

补修使用两阶段身份：

1. implementation commit：代码、测试、规范与实际运行证据；
2. registration commit：只登记 implementation SHA、验收状态和下一阶段 blocker。

`STATUS.yaml` 的 M05 `commit` 在最终登记时必须指向 implementation commit，另存原始实现和 registration commit；不得使用 `null`、`HEAD` 或 self-reference 代替准确实现身份。
