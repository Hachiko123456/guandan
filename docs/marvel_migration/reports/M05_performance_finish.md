# M05 真实性能收尾报告

- 日期：2026-10-09
- 分支：`codex/marvel-guandan-migration`
- 基线 commit：`3aee75fdeece3fc9f142f93c9fb971fe59954bd8`
- 实现 commit：待第一笔实现提交后由登记提交固定
- 阶段状态：`incomplete`
- M05 accepted：false
- M06：blocked

## 1. 冻结基线

修改功能代码前固定并保存：

- 同一 v1.3 Transformer：history 16、feature 24、1 decoder layer、4 heads、1 state/action/fuse layer；
- frozen weight SHA-256：`e1f36b4d519e9cc74eb59d5792eb5061a5aae4fe5f818c87f4080ed504a9b393`；
- CPU、fp32、`torch.set_num_threads(1)`、eval + inference mode、greedy temperature=0；
- 主集合 case `50000..50003`，保持 4 局 / 1,027 actions；
- 泛化集合 case `51000..51007`，在运行结果前固定为 8 局诊断集合；
- import、权重加载和预热不计入稳定 rollout；每个 backend 5 次，顺序交替；
- 完整 action/reward trajectory digest 和工作计数必须一致。

基线主结果：Python 7.267330 s，C++ 4.551909 s，speedup **1.5965×**。泛化集合 speedup **1.5042×**。

## 2. 瓶颈分析

Baseline C++ 主基准代表性分段：

- model forward with KV cache：2.7010 s，约 59%；
- environment step：1.2489 s，约 27%；
- conversion、sampling、buffer、setup：约 13%。

PyTorch profiler 显示大量小 tensor operation、`to/copy/gather/reshape/linear` 调用。代码检查确认 history/KV 必须维护四个 observer，但 decoder 之后的 state/action/policy heads 仍把共享候选动作复制给四个玩家，最终又 mask 掉三个 inactive seat。

因此第一候选锁定为“保留四座位 history/KV，提前 gather active observer，仅执行一次 state/action/policy heads”。它不减少 Transformer 调用、不缩小模型、不删除历史或隐私信息。

## 3. 三轮候选

### Round 1：active-only post-decoder heads — 保留

修改 `Agent.forward()`：

- history decoder 和 KV cache 继续按 `B*P` 更新；
- decoder 后按 `player_ids` gather acting observer；
- state MLP 只处理 acting observer；
- 完全相同的候选动作只编码一次，不再 repeat 四份；
- value 输出仍恢复为原 `[B,T,P]` 形状，inactive seat 保持零；
- chance/padding 继续使用原 environment distribution。

预先固定的等价测试：

- policy probability/value：`rtol=1e-6, atol=1e-7`；
- loss/gradient：`rtol=1e-5, atol=1e-6`；
- greedy action、reward、terminal、trajectory digest：完全一致。

结果：主 speedup 1.6362×；C++ wall 4.5519 -> 4.2788 s。模型段 2.7010 -> 2.4686 s。

### Round 2：lightweight step-event snapshot — 撤回

尝试去掉只用于 event 构造的一次完整 `Game` copy。结果主 speedup 1.5965×，没有稳定配对收益；而保留会扩大 C++ event/serialization 差分范围。已完全撤回。

### Round 3：outer-transaction direct mutation — 撤回

尝试让已位于 `EnvBatch` trial copy 中的 game 直接执行 token，去掉内部第二次 transaction copy。C++ wall 有约 3–4% 小幅收益，但主 speedup 仅 1.6286×，仍明显低于 2×；保留需要重新执行正式 10k/100k C++ 环境差分。基于收益、风险和验证成本，已完全撤回。

### Tooling experiment：torch.compile — 未采用

Inductor 在本机先遇到默认 GBK 读取错误；使用 UTF-8 interpreter 后又因 Windows 缺少 MSVC `cl` 抛 `InvalidCxxCompiler`。没有产生可比较性能样本，也没有代码保留。

## 4. 最终公平配对结果

最终实现仅保留 Round 1。主集合所有样本保持 1,027 actions 和相同 digest。

| Backend | 5 次 wall 样本（秒） | Median | Stdev | actions/s |
|---|---|---:|---:|---:|
| Python | 6.592798 / 6.806015 / 6.710586 / 6.440370 / 6.435526 | 6.592798 | 0.146434 | 155.77 |
| C++ | 4.014001 / 4.068080 / 4.022979 / 4.069007 / 4.029885 | 4.029885 | 0.023215 | 254.85 |

最终主 speedup：**1.6360×**。

代表性分段：

- Python environment 3.5482 s；model forward 2.4465 s；
- C++ environment 1.1523 s；model forward 2.3295 s；
- C++ conversion 0.0759 s；sampling/marshalling 0.1697 s；buffer/commit 0.2202 s。

泛化 8 局集合 speedup：**1.5853×**，C++ 仍稳定更快，digest 和工作计数匹配。

## 5. 整体收益与可达上限

相对本轮修改前基线：

- 旧 C++ -> 新 C++：4.5519 / 4.0299 = **1.1295×**；
- 旧 Python -> 新 Python：7.2673 / 6.5928 = **1.1023×**；
- Python/C++ ratio：1.5965× -> 1.6360×。

当前 C++ 非环境固定工作约 2.878 s，environment 约 1.152 s。即使把 C++ environment 理想化为 0，按当前 Python wall 的上限约为 **2.29×**。达到 2× 需要 C++ total <=3.296 s，即再减少约 0.734 s：相当于消除约 64% 的现有 C++ environment 段，或让双方共同的非环境工作下降约 57%。

继续投入预计需要：

- 深入专用化 cached Transformer/KV inference；或
- 大幅重构 C++ transaction/state copy 数据结构，并承担完整 10k/100k 重新差分；或
- 配置可用的 Windows CPU compiler 后重新评估 Inductor。

这些均超出本轮三次有依据候选的有限预算。

## 6. 其他性能工作负载

最终复测：

- single-env fixed-action：**3.2026×**；
- batch fixed-action：**2.0505×**；
- deterministic selector：**2.7827×**，仍不表示 Transformer；
- 完整 IPPO iteration total：**1.0841×**；
- 完整 iteration rollout：**1.1307×**；
- learner ratio：**0.9387×**。

Learner 结果如实公开，未被纳入或伪装成环境加速。

## 7. 正确性与差分

最终保留的改动只在通用 Agent post-decoder heads；Python/C++ 使用同一优化。以下文件相对基线不变：

- A01/A02 规则真值；
- Python game adapter；
- C++ `guandan.cpp/.hpp`、bindings、动作/观测/transaction 实现。

因此继承原 10k/100k 正确性证据。本轮重新执行：

- 全部 corpus shard hash/记录/动作计数；
- 全 13 level 完整 trajectory replay；
- directed wild、贡还、炸弹、four kings；
- 四观察者隐私重排；
- 模型 probability/value/loss/gradient 等价；
- 主/泛化 benchmark 动作、reward、trajectory digest。

Corpus replay valid，10,000 records、3,243,977 actions 不变。无需因最终 Agent-only 改动重新运行完整 10k。

## 8. 测试、构建与恢复

Blocking tests：

- Agent 数学等价：2 passed；
- backend import/checkpoint identity：10 passed；
- M05 + C++ scoped：118 passed；
- M01–M03 regression：88 passed；
- A00–A04 protected：157 passed。

主要不重叠集合：363 passed，0 failed，0 skipped。

完整 `--import-mode=importlib`：1396 passed，10 skipped。默认 pytest 仍因两个历史 `test_training_smoke.py` basename 冲突而 collection exit 2，单独记录。

Checkpoint artifact verifier：exact/compatible/migration/evaluation/warm-restart 分类通过。C++ checkpoint：same build exact、logging-only compatible、cross-backend/build mismatch 拒绝并 poison。

最终 wheel SHA-256：`dc8f013cf7821815fb42afbb7b1978e7c58e88ae933bce3c993cead52f31e711`；隔离安装和 no-native 场景通过。Native source 未改，加载 binary 与 embedded source digest 匹配。

## 9. 修改文件

- `guandan_marvel/model/agent.py`
- `tests/marvel_training/test_agent_active_head_optimization.py`
- `scripts/benchmark_marvel_m05_finish.py`
- `scripts/profile_marvel_m05_finish.py`
- `docs/marvel_migration/M05_PERFORMANCE_FINISH_PROTOCOL.md`
- `docs/marvel_migration/00_README.md`
- `docs/marvel_migration/STATUS.yaml`
- 本报告、JSON 和 `evidence/M05_performance_finish/*`

所有 Round 2/3 C++ 候选均已撤回，最终 Git diff 不包含 C++ 规则/环境源修改。

## 10. 验收结论

当前硬门槛为真实冻结 Transformer 主基准 paired median `>=2×`。最终结果 **1.6360×**，未通过。

按照三轮停止条件：

- M05 保持 `accepted=false`；
- M06 保持 blocked；
- 不自动降低门槛；
- 不开始 M06、Kaggle、DDP 或正式训练。

下一步必须由用户决定：继续投入高成本 KV/Transformer 或 C++ transaction 重构，或明确批准调整真实 Agent 的验收口径。
