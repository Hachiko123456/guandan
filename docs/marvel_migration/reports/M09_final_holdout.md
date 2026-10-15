# M09 One-Shot Final Holdout 结果

- 日期：2026-10-16
- frozen candidate：IPPO iter2000
- candidate SHA-256：`af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8`
- final dataset hash：`06c8db261da5b8afbd33550080342ac7260bad31e613e2ec584822567743aeda`
- 状态：`consumed`
- 1300/1300 complete
- illegal/truncated：0/0
- M09 accepted：**否**

## 完整性

- final output ZIP SHA-256：`83722a8395f7cc5576b7bf66d2f645b1a68c64c8549b68a684ef7c5dcbce7721`；
- ZIP CRC：通过；
- evaluation hash：`d23132ba90e0f15207fe5cd23bc53a8807b54841d42b2d8787549bb99e11d73c`；
- plan hash：`4b99899dd776bd9ff77009b02fbd4f349b9ad2c152f87304691145b2b10b3a66`；
- summary hash：`45f713afa362d053d8d39da1a111ce6df97374409f09e8e2f25669d2f675fca2`；
- report hash：`c140e67e5c26b8959e01ac32b141d21393365def129d3e52c3cdcfb98902b694`；
- audit hash：`0fdfb19f6d9f42994dce75111f12bd058f5a66122e0a937ebf2afa0e8bf532af`；
- audit state：`consumed`；
- candidate/dataset/evaluation identities：一致。

## Final holdout 矩阵

65 independent clusters × 4 rotations × 5 lanes = 1,300 games。

| 对手 | 胜局 | 对局 | 胜率 | clustered CI95 low | high |
|---|---:|---:|---:|---:|---:|
| iter100 champion | 174 | 260 | 66.92% | 36.59% | 97.25% |
| rule | 144 | 260 | 55.38% | 25.12% | 85.65% |
| random | 223 | 260 | 85.77% | 60.04% | 100.00% |
| legacy V1 | 228 | 260 | 87.69% | 62.30% | 100.00% |
| opponent pool | 174 | 260 | 66.92% | 36.59% | 97.25% |
| overall | 943 | 1300 | 72.54% | 不跨相关 lanes 汇总 | 不跨相关 lanes 汇总 |

所有 lanes illegal/truncated=0。整体 seat rotation gap=12.92%；rule lane team02/team13 为 63.08%/47.69%，仍存在一定角色差异。

## 强度结论

- **E1 supported**：对 random 胜率 85.77%，clustered CI lower=60.04% > 50%。
- **E2 supported（competitive with rule）**：对 rule 点估计 55.38%，健康为 0；但 CI lower=25.12%，不构成稳定击败 rule 的结论。
- **Legacy superiority supported**：对 legacy 87.69%，CI lower=62.30% > 50%。
- **E3 not demonstrated**：E3 要求稳定击败 rule 且优于 legacy；legacy 部分通过，但 rule 的 CI lower 未超过 50%。
- 对 iter100 champion 点估计 66.92%，但 CI lower=36.59%，仍不能按保守 clustered gate 声明统计稳定优于 champion。

因此训练是有效的：独立 one-shot final 数据上稳定击败 random 和 legacy，且对 rule/champion 点估计为正；但没有达到原定 E3 发布目标，M09 不 accepted。

## One-shot 边界

该 final holdout 已消费，不得重跑，不得用其结果重新选择 checkpoint 或调参。若继续训练/改变统计协议，必须建立新版本训练计划并生成全新的 final holdout 数据集。
