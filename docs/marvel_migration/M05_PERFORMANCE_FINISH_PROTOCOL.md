# M05 真实性能收尾协议

- 基线实现：`758b2e97fd3bf0103fa1046a72bf6ed4a8cf3600`
- 阻塞登记：`3aee75fdeece3fc9f142f93c9fb971fe59954bd8`
- 当前硬门槛：真实冻结 Transformer Agent rollout 的 Python/C++ 配对中位数加速 `>=2.0×`
- 最大候选优化轮数：3

## 1. 冻结工作负载

主基准固定为此前连续可比较的 4 局：case index `50000..50003`。冻结 Agent 使用 v1.3、history 16、feature 24、1 decoder layer、4 heads、1 state/action/fuse layer，CPU、`torch.set_num_threads(1)`、fp32、`eval()`、`torch.inference_mode()`、greedy temperature=0、horizon=512、batch=1。

冻结权重首次按 seed `91001` 生成并保存，后续候选必须加载同一文件并校验 SHA-256。Deal manifest、模型配置、sampling、计时范围和依赖身份写入 benchmark manifest。主基准必须保持 1,027 个动作以及完整 action/reward trajectory digest；若变化则候选无效。

正式比较在双方预热后执行至少 5 次，顺序按 repetition 交替。初始化、首次 import 和预热不进入稳定 rollout 样本。Profiling run 不与非 profiling 样本相除。

## 2. 预先固定的泛化集合

额外集合固定为 case index `51000..51007`，共 8 局，使用同一冻结权重和完全相同的推理定义。用途是在主 4 局外确认优化没有只针对固定 deal。它要求：

- Python/C++ action/reward trajectory digest 完全一致；
- 工作计数每次一致；
- C++ 中位时间必须快于 Python；
- speedup 作为诊断完整报告，不替代主 `>=2×` gate。

## 3. Profiling 与候选顺序

修改功能代码前保存 baseline manifest、5 次主/泛化样本、分段时间、cProfile 与 PyTorch CPU operator summary。优先根据证据检查：

1. Agent 是否在 history decoder 后仍为四个玩家重复执行仅 active player 需要的 state/action/policy heads；
2. 候选动作是否被重复编码四次；
3. sampler TensorDict、buffer 和转换分配；
4. C++ environment copy/enumeration/validation。

候选最多三轮。通用 Agent/sampler 优化必须同时作用于 Python 和 C++。每轮先验证数学/trajectory 等价，再以相同冻结 manifest 测量。无收益候选必须撤回并记录。

## 4. 等价标准

数学等价预先固定：

- eval/inference policy probability：`rtol=1e-6, atol=1e-7`；
- value 输出：`rtol=1e-6, atol=1e-7`；
- training loss/gradient：`rtol=1e-5, atol=1e-6`；
- greedy action、reward、terminal 和 trajectory digest：完全一致。

不得删除 history/KV 更新、合法动作校验、隐私隔离或 buffer 写入。规则、动作、观测、奖励和 C++ 环境执行实现若未改，可继承原 10k/100k 并执行 hash、13-level、wild/贡还/炸弹回放；若改动则重跑全量差分。

## 5. 最终复测

最终候选必须复测：

- 主 4 局真实冻结 Agent rollout；
- 8 局泛化集合；
- single-env fixed-action；
- batch fixed-action；
- learner 不变完整 IPPO iteration；
- backend/checkpoint/build identity；
- 导入、C++、M01–M05、A00–A04 和隔离 wheel。

同时报告新 Python→新 C++（门槛）与旧 C++→新 C++（整体系统收益），不可混用。

若三轮后主基准仍 `<2×`，M05 保持 blocked，并报告每轮收益、剩余固定开销、理论上限和继续优化成本；不得降低门槛。
