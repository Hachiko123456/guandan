# M06 评估、Champion 与 Opponent Pool 协议

- Stage：`M06_evaluation_champion`
- 日期：2026-10-10
- C++ backend：`GuandanMARVELV2Cpp`
- V2 training checkpoint：`guandan-marvel-training-checkpoint-v4`
- evaluation-only snapshot：`guandan-marvel-evaluation-checkpoint-v1`

## 1. 版本化协议

| 产物 | Schema |
|---|---|
| deals manifest | `guandan-marvel-deals-manifest-v1` |
| evaluation plan | `guandan-marvel-evaluation-plan-v1` |
| per-game record | `guandan-marvel-evaluation-game-v1` |
| clustered summary | `guandan-marvel-clustered-summary-v1` |
| promotion decision | `guandan-marvel-promotion-decision-v1` |
| champion manifest | `guandan-marvel-champion-manifest-v1` |
| opponent pool | `guandan-marvel-opponent-pool-v1` |
| pool sampling | `guandan-marvel-opponent-pool-weighted-sampling-v1` |
| final holdout audit | `guandan-marvel-final-holdout-audit-v1` |
| training evaluator hook | `guandan-marvel-training-evaluator-hook-v1` |

所有 JSON manifest 都使用 canonical JSON hash。Champion、opponent-pool、final audit 和小型报告使用同目录临时文件、flush/fsync 与 `os.replace` 原子更新。替换失败时旧文件保持不变。

## 2. Deals 隔离

`training`、`selection`、`final` 是同一 manifest 中的三个显式 domain。每条记录含：

- 全局唯一 `deal_id`；
- domain；
- seed；
- level rank；
- leader seat；
- previous result/贡还条件。

manifest 验证器拒绝跨 domain 重复 seed。默认 namespace 分别从 `100000`、`1000000000`、`1500000000` 开始。selection 只能进入 selection runner；final 只能进入独立 final audit runner。

Final holdout 必须先锁定 candidate SHA-256，然后只允许 `final_audit` 用途的一次使用。失败尝试也不能被改作模型选择；若需再次使用必须生成新版本。

## 3. 固定矩阵和四座位轮换

每个 selection deal 固定运行：

1. candidate vs current champion；
2. candidate vs rule；
3. candidate vs random；
4. candidate vs dedicated legacy V1 opponent adapter；
5. candidate vs weighted opponent-pool sample。

不包含 FableDan。

同一 C++ base deal 保持物理牌和初始条件不变。logical seats 0/2 为 candidate team，1/3 为 opponent team；rotation 0..3 将 logical role 轮换到 physical seats，保证 primary logical seat 访问四个座位。V2 模型使用相同 eval mode 和 action temperature，各 policy session 使用独立 KV/RNG 状态。

Legacy V1 adapter 在独立 Python V1 mirror 中读取 V1 `Observation`，其选择的 V1 token 必须唯一映射到当前 C++ V2 legal row；每步同步执行，终局 ranking/outcome 必须与 C++ host 相同。Legacy artifact 永远是 opponent-only，不能进入 V2 candidate/champion 或 training resume。

## 4. Clustered 95% CI

同一 base deal 的四个 rotations 组成一个 cluster。先计算每个 cluster 的四局平均胜率，再对 cluster means 使用双侧 95% t interval：

```text
mean(cluster_mean) ± t_(n-1, 0.975) * sd(cluster_mean) / sqrt(n)
```

区间裁剪到 `[0,1]`。少于两个完整 deal clusters 时区间为 `null` 且 `conclusive=false`；不得把四个 rotation 当作四个独立样本。

summary 同时报告规范要求的 win rate、CI、team02/team13、seat gap、team reward、三种终局结果率、illegal/truncated 和 games/s。

## 5. Promotion

预定义 gate：

- 固定矩阵完整；
- 四座位 coverage 完整；
- illegal/truncated 为 0；
- candidate 与 champion 身份完整且不是同一文件；
- candidate vs champion 点估计 `>50%`；
- deal cluster 数达到预定义最小值；
- clustered CI 下界达到阈值；
- rule/random anchor 不低于预定义容忍线。

样本不足或 CI 不足返回 `inconclusive`；不晋升。点估计 50.1% 不会自动晋升。相同 checkpoint 返回明确 `reject`。只有带 hash 的成功 promotion decision 才可原子更新 champion manifest。

初始 champion bootstrap 只表示管线初始状态，不是强度晋升。Manifest 明确记录 `kind=bootstrap`。

## 6. Opponent pool

Pool manifest 保存 current/recent/historical champion、rule 和 legacy anchors、active 标志与显式 weight。`sample_seed + manifest sampling seed` 决定可恢复的 weighted sample。恢复时重新验证 schema、manifest hash 和所有 checkpoint 文件 SHA-256。

selection 固定矩阵中的 `opponent_pool` lane 默认只从 current/historical champion kinds 抽样；rule/legacy 已有独立 lane，避免重复替代。

## 7. Training evaluator hook

`TrainingEvaluatorHook` 每 N 个完整 iteration：

1. 从 `agent_merged` 克隆 CPU state；
2. 写出 `guandan-marvel-evaluation-checkpoint-v1`；
3. 不包含 optimizer/sampler/replay/RNG resume state；
4. 将评估提交给独立单 worker executor；
5. 不把 live trainer 对象传入 evaluator。

Trainer 只在 durable iteration 边界调用 hook。调度失败被记录为 hook 状态，不 poison trainer，也不允许 evaluator 修改训练状态。若已有 pending evaluation，可配置为跳过本次，而不是阻塞训练。

## 8. Selection 与 final 分离

- `run_selection`：只读 selection deals，可产生 promotion decision 和 champion/pool 更新；
- `run_final_audit`：只读 final deals，需要 candidate lock，只记录 final audit use，永远不调用 promotion；
- training hook 调用 selection evaluator，不调用 final audit。

任何依据 final 结果选择 checkpoint、调参或重训的行为都会使 holdout 失效。

## 9. M06 本地演练边界

M06 小规模演练只证明 E0 管线可运行、身份/原子性/恢复/统计路径有效，不证明模型强度。Kaggle T4×2 吞吐和双卡效率仍由 M08 验证；本阶段不开始 Kaggle 正式训练。
