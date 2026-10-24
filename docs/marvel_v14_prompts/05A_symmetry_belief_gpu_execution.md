# 阶段 5 修订执行 Prompt：对称性诊断、Belief B0/B1 与显式 GPU 验收

更新日期：2026-10-19。此文件替代旧 05 的执行规范，不覆盖其冻结历史内容。05A 仍然是“阶段 5”，不是新的第七阶段。

## 0. 前提与执行边界

完整读取本文件、`00_local_gpu_execution_contract.md`、`CLAUDE.md` 和当前真实代码；共同构成验收规范。

核心阶段 1–4（含 2A/04A）已有独立验收。开工记录新基线，不复用历史 hash 冒充当前基线。只实施阶段 5，不实施 opponent pool、search 或最终正式训练。通过 `/phase` 运行 Implementer → 全新只读 Reviewer；本次执行命令只授权本文明确限定的本地有界实验。

以下三件事必须区分：

1. 工程是否实现并正确验收。
2. B0 信息量 gate 是否通过。
3. 某一增强是否获得足够对局证据，能默认开启。

不能把 B0 PASS 当作棋力提升，也不能把辅助 loss 下降当作默认启用依据。

## 1. 先审计，先冻结实验设计

审计 `config.py`、`model/agent.py`、宏动作 PPO/GAE、sampler、checkpoint、strict evaluation loader、C++ 观测与公开贡还信息。先说明真实观察字段和可用因果历史，不假定“channels”包含全部可见信息。

新增阶段 5 设计/运行配置，至少冻结：

- feature flags、模型/head/schema 版本、默认关闭值；
- 对称变换与宏动作候选比较定义；
- raw printed-rank 编码与 next/partner/previous 的相对座位约定；
- 数据集大小、contexts/贡还分层、group split seed/hash；
- 预定 probe 容量、seeds、epochs/early-stop、优化器、主要指标与 CI；
- B0 通过/否决/不确定规则、最小效应阈值、多个候选/指标的选择方法；
- 本地预算、设备、正式/本地适配、所有未运行项。

查看 test/final 结果之前冻结这些字段。禁止在看结果后更换指标、删样本、改 split 或增加试验直到显著。

允许最小扩展 config/model/training/sampler/checkpoints/evaluation，优先新增职责单一模块和新测试。不得改变已验收 PPO/GAE、reward=rank-points/3、52-context/curriculum、Fuse 零门和 AMP 核心数学。

## 2. GPU 优先的真实执行

所有真实神经策略、线性/浅层可训练 probe、完整 history probe、Belief head 的 forward/backward 在 CUDA；不把“probe 很小”当作改回 CPU 训练的理由。使用固定独立 GPU Python、显式 `--device cuda`；环境推进/规则/解析/统计仍可 CPU。

新 CLI 必须提供并记录 device、max_games、max_updates、max_wall_seconds、seed、output。明确 CUDA 不可用时在写正式产物前失败；OOM 不 fallback CPU。

先 16 个独立牌副的有界 pilot，报告预计采集与训练耗时、显存峰值；再执行下面的预定采集。默认每个 GPU smoke batch=1、1–2 PPO iteration，真实模型参数更新/scaler/保存恢复都必须验证。CPU BF16/纯数学/gloo 专项保留，不能替代 CUDA 验收。

## 3. 对称性：先诊断，后条件训练

### 3.1 合法变换与可逆性

A. 座位整体旋转 k=0,1,2,3：player/observer/leader/team/partner、历史、finished order、公开贡还、候选动作、宏动作起点和相对标签同步映射。

B. 同 rank/suit 的双副物理牌 copy-ID 交换：手牌、历史、current winning、候选动作、wild declaration 中所有对应引用一致映射。交换只是等价实体的重命名，不改变配牌或牌型规则。无法从当前公开编码唯一恢复的映射必须明确可定义域，不得偷读隐藏手牌补齐输入。

每个变换 T 和其动作置换 P_T 都需要逆映射，合法性 oracle、round-trip、终局 reward、两引擎可比 case。包含 PASS、wild/声明、出完/接风、贡还、padding、多 batch 和相同 token/不同物理牌边界；不能只测普通出牌。

### 3.2 对齐后的策略比较

正确比较：

```text
pi(P_T(a) | T(s))  vs  pi(a | s)
```

将变换后的分布映回原合法候选顺序：

```text
q_T(a | s) = pi(P_T(a) | T(s))
L_sym(s) = sum_{a in A(s)} pi(a | s) * [log pi(a | s) - log q_T(a | s)]
```

动作比较单位必须声明：完整合法宏动作/完整自回归序列，不能把下一 token 分布当作完整动作分布。序列 log-prob 是对应合法路径上各 token log-prob 之和；若完整枚举不可行，只在可证明一一对应的候选集合上条件归一化，明确标为 restricted-candidate KL，不声称 full-action KL。

padding/inactive/非法候选不参与，使用数值稳定 log-prob，零概率行为要明确。报告 max/mean probability difference、KL、argmax（含 tie 规则）一致率，按 phase/action kind/level/leader/贡还分桶。

### 3.3 条件训练与默认关闭

先完成原模型诊断。在训练前预注册“显著不一致”的阈值与样本数；无显著不一致则保留诊断工具，不强行加 loss。

若达到预注册阈值，才实现可配置 seat/copy-ID augmentation 及上面的对齐 KL。说明对称 KL 梯度作用于哪一/两侧分支；不允许目标分支的隐式 detach 或不一致 old/new policy 行为。

`symmetry_weight=0` 且 augmentation=false 时，不能增加 forward、模型参数、采样 RNG 消耗、推理延迟或改变默认轨迹。symmetry 默认关闭，只有独立 paired 棋力证据通过才建议开启。

## 4. Belief B0：先证明可见历史有额外信息

### 4.1 数据与无泄漏边界

- 真实 C++ 引擎采集；最终 B0 计划至少 512 个完整独立 canonical deals，多等级、leader、合法贡还/抗贡类型。512 是上限预算与计划样本量，不足则报告 INCONCLUSIVE，不把 pilot 当完整预检。
- 牌副 group ID 基于可审计的初始实体配牌/发牌身份，不能只用 step index 或 split seed。同一牌副的所有决策、旋转/copy-ID 增强与重复 episode 必须在同一 split；改变 context/命名也不能让同一实体牌副跨集合。
- 固定 train/validation/test，推荐约 70/15/15；在分层清单中确保支持的 level/leader/贡还分布不是单点。有意无法覆盖的 case 必须列为未覆盖，不使用 metadata 声称实测覆盖。
- 样本只取行为方宏动作起点，模型输入只用该时刻策略可见观测与因果历史。oracle 可读隐藏状态生成离线 label，但它必须在独立 label/审计通道，绝不能进入 policy/probe 输入或选样特征。
- 标签为 next/partner/previous 三家手牌的 15 printed ranks 剩余张数。13 普通 printed ranks 加大小王，不能把 wild 宣告牌型等级当成实体 printed rank。普通 rank 计数范围 0..8，joker 上限按真实两副实体牌数设置 mask，不允许预测不存在的计数。
- 记录 own+opponents+played 的实体守恒，公开贡还只计当前 owner、不得双计；Finished player 剩余张数为 0。
- 额外做泄漏反例：仅改变未公开对手手牌、保持可见 observation 完全相同，输入字节应相同；未来动作、终局结果、未来 private reveal 不得反向进入当前历史。

### 4.2 四组基线必须一致比较

A. 无条件先验：根据训练集与规则约束定义，不把任意高分辨率 oracle 叫“均匀先验”。

B. 公开信息强基线：自手牌、公开已出牌、各家公开剩余张数、公开贡还及可推导实体余额；不使用隐藏手牌或未来信息。

C. 当前 channels 线性/浅层 probe。

D. 完整因果 history+channels probe。

可训练 B/C/D 都在 CUDA，在同一 group split、固定 seeds 与可审计容量/优化预算下比较。使用冻结的项目自身 checkpoint encoder 时记录身份，不能训练时拿 oracle 特征、验证时换可见特征。每个候选至少 3 个预定 seeds，不能只报告最佳 seed。

B 可用纯规则计数 CPU 实现；若 B 有可训练部分，同样 CUDA。FableDan 的权重、策略、标签、动作轨迹或蒸馏均禁止。

### 4.3 指标与 Gate

每相对座位/阶段/context 报告 count L1、rank MAE、exact-count accuracy、守恒误差、校准。对 D 相对 B 的增量使用按 canonical deal 聚类的 paired bootstrap 95% CI，同一副牌多个 macro 不能当独立样本；预先说明各 deal/seat/rank 权重和 bootstrap seed/次数。

预注册主要指标为 D 相对 B 的 count-error 改善（B_error - D_error，越大越好），以及实际意义的最小效应阈值 epsilon，必须在训练前写入 config。允许 epsilon=0 仅作统计信息量验证，但必须明确“无最小实用效应要求”，不能借此宣传实用提升。

```text
B0 PASS：主要改善的 CI 下界 > epsilon，预定多个 seeds 稳定，且无输入泄漏/分组污染/严重分桶退化。
B0 REJECT：充分样本与已锁定设计下，主要改善的 CI 上界 <= epsilon；保持 belief 禁用并报告证据。
B0 INCONCLUSIVE：预算不足、CI 跨界、分组/覆盖不足或评估设计不充分；停止，不自动扩样或实现 B1。
```

只击败 A 不算通过。验证集用于容量/early-stop 选择，test 在设计锁定后只开一次作 B0 confirmatory 评估；此 test 仅是阶段5预检，不是最终棋力 holdout，二者 seed/hash 不得混用。

## 5. Belief B1：只有 B0 PASS 才实现

B0 REJECT 时不构建 B1，保留数据、工具、指标和否决报告；这可形成有效的阶段5条件分支结果。B0 INCONCLUSIVE 时停止并报告缺证据，不能把未跑当 REJECT。

B0 PASS 后才新增训练期 belief head：

```text
logits: [batch, macro_start, 3 relative seats, 15 printed ranks, count classes 0..8]
L_belief = masked count cross entropy
            + explicitly configured conservation penalties
```

joker/non-joker/finished-player 的非法类别 mask、label dtype、宏动作 sample mask 必须独立测试。守恒目标来自公开余额，不是把隐藏 oracle 总量喂入推理图。

head 从因果 player_state 构建；默认 `belief_weight=0` 且 head 不构建，推理路径无 head、无额外参数/forward/延迟。显式区分训练 checkpoint 与 inference/policy-only export，使用版本化 whitelist 处理训练专用 head，禁止 strict loader 任意忽略未知/缺失 key。

必须审计并测试 head/class-mask/weight/schema 对 config、optimizer、warm-start、evaluation、strict resume、DDP 的影响。全关闭时原路径 bitwise 等价，不多消耗 RNG。打开/关闭跨配置不能伪装 exact_resume。

梯度比例定义正确写为：

```text
rho_g = ||lambda_b * grad_theta_g L_belief||_2
        / max(||grad_theta_g L_RL||_2, eps)
```

分组 g 为 decoder/state_mlp/actor_fuse，独立算未污染的梯度后再组合；报告绝对 norm、eps 和零分母。若 head 取值在 Fuse 之前，actor_fuse 的辅助梯度为零可能是正确结构结果，应解释而不是虚构通路。候选权重按实测选择，不能照抄 0.05。此处梯度诊断在 CUDA 执行，保留 AMP 的实际 dtype/scaler 证据。

## 6. A/B、有限预算与启用标准

symmetry off/on 与 belief off/on 分开比较，必要时再组合，避免把两个特性同时改后无法归因。相同 warm start、seeds、训练局数、冻结 development deals 和 rotation，按 deal 配对/聚类。

默认实验预算：pilot 16 个独立 deals；完整 B0 最多 512 个完成 deals，采集上限 1800 秒；可训练 probe 合计上限 1200 秒；微型 RL A/B 每臂最多 10 PPO iterations、最多 4 个预注册臂，不启动 50+ iter 正式训练。每个循环均有单独预算参数，实施前按 pilot 核算；预计超预算则先报告，不自动扩容、换 CPU、缩减统计标准或自造 PASS。

微型 A/B 只证明工程健康。若不能在预算内得到足够 paired 棋力证据，报告 STRENGTH_NOT_ESTABLISHED，特性保持关闭；不能声称“无退化/棋力提升”。只有预注册 paired 棋力 CI 下界 > 0，且成本/合法性/分桶检查通过，才可建议默认启用；默认配置变更仍需用户选择。

报告 win-rate 和 rank-point delta，不把单次 loss 降低或小样本 argmax 变化当成棋力提升。final holdout/FableDan 正式评估禁止运行。

## 7. 必须的反例与测试

新增独立 tests/marvel_training/test_phase5_*.py、对应辅助 fixtures（按实际实现分模块），至少覆盖：

1. 旋转/等价实体置换的逆变换、合法候选一一映射、完整宏动作而非 next-token 偷换。
2. PASS/wild/贡还/出完/接风/finished player、不同 num_steps 和 hostile padding。
3. split group 泄漏（旋转/重放/同牌不同 context）、未来观测/隐藏牌输入泄漏。
4. 15 printed-rank 标签、joker 上限、wild 实体与宣告区分、公开守恒。
5. B0 未通过时 B1 不被构造，CI/预算不足时 INCONCLUSIVE，不伪造否决。
6. symmetry/belief 全关闭的参数/输出/RNG/轨迹零开销等价。
7. 若 B1 实施：mask、CE/守恒、梯度 norm、GPU 有效更新、导出 whitelist、缺 key/未知 key fail-closed、同配置恢复。
8. 显式 CUDA 不可用 fail-fast/无 CPU fallback；真实 CUDA FP16+scaler；CPU-only 专项不依赖主机无 GPU 假设。
9. 若改变 DDP 路径：current enabled trainable 参数的零样本 rank、collective 一致性和 weighted reduction；CPU 双 rank gloo 不冒充双 T4/NCCL。
10. 阶段2/3/4相关专项继续通过，不修改旧容差或历史 manifest。

先最小反例，再 GPU smoke，再相关回归，最后冻结源码后全量一轮。最小操作命令的标准形式（测试文件由本阶段实际实现）：

```powershell
& D:/project/guandan/.venv-gpu/Scripts/python.exe -m pytest tests/marvel_training -k phase5 -q -p no:cacheprovider
```

不能用 `-k phase5` 收集到 0 tests 当作通过；报告实际收集数和逐项命令。全量、日志和 Reviewer 规则见共用契约。

## 8. 报告与停止条件

输出至少包含：

```text
IMPLEMENTATION_STATUS
BEFORE/AFTER_GIT_AND_SOURCE_HASHES
SYMMETRY_DIAGNOSTIC / AUGMENTATION_STATUS / SYMMETRY_DEFAULT
B0_DATASET_AND_GROUP_SPLIT / BASELINES_A_B_C_D / METRICS_CI
B0_GATE: PASS | REJECT | INCONCLUSIVE
B1_STATUS: IMPLEMENTED | SKIPPED_B0_REJECTED | NOT_RUN_B0_INCONCLUSIVE
STRENGTH_STATUS / BELIEF_DEFAULT
GPU_DEVICE_AMP_UPDATES_MEMORY_TIMING
CHECKPOINT_EXPORT_RESUME / DELIVERY_PENDING
COUNTEREXAMPLES_AND_TEST_RESULTS
LEGACY_GUANCE_FAILURE / OTHER_FAILURES
UNIMPLEMENTED_STAGE6 / NO_LONG_TRAINING
```

全新 Reviewer 独立运行至少三类新反例，严格区分 REVIEWER_RUN / EXISTING_REPORT / NOT_RUN。完整数据链、反例、GPU和回归满足时，允许工程 PASS + B0_REJECTED/特性关闭；没有预算/统计证据时 INCONCLUSIVE，不得为了进入阶段6伪造 gate。

Reviewer 必须明确 `VERDICT: PASS|FAIL|INCONCLUSIVE`、`NEXT_STAGE: ALLOW|DENY`。完成后停止，阶段6需用户另行发命令；不启动长训、Kaggle 或最终评估。
