# M09 r2 VRPO Controlled Pilot 验收报告

- 日期：2026-10-13
- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`
- package id：`guandan-marvel-m09-algorithm-pilot-20261011-r2`
- Kaggle run id：`20261013T062752Z-vrpo_pilot-7a0dab10`
- 环境：Kaggle Tesla T4×2 / NCCL DDP
- Pilot 状态：`accepted_as_controlled_algorithm_pilot`
- M09 阶段 accepted：**否**

## 训练预算

- VRPO 100/100 iterations；
- 100 durable updates；
- 400/400 completed games；
- 123,252 env steps；
- 48,709 game steps；
- session elapsed：4,478.77 秒；
- budget contract：`m09-ippo-vrpo-equal-games-v1`；
- wall-time stop：否。

100 条 metrics 连续完整；illegal、truncated、nonfinite loss/gradient、action capacity overflow、horizon overflow 全部为 0。

## Checkpoint 与 Recovery

- DDP checkpoint v2；world size 2；algorithm VRPO；iteration/durable updates 100；
- recovery manifest 14 个文件全部通过大小/hash 校验；
- rank shards 分别约 2.144 GB，ZIP 总大小 250,423,749 bytes；
- 两 rank 的 resume actor、merged actor、critic state hash 完全一致；
- continuation state 因每 rank 独立 sampler/RNG/replay 而不同，符合设计；
- output/recovery 的 7 份关键小文件逐字节一致；
- recovery status：complete。

冻结模型：

- `evaluation_checkpoints/iter000100.pt`；
- SHA-256：`db8aa71f22ec88091acb189e6c57df2ac9ac0c1d18e3a931f4d7a1ec2cca9853`；
- bytes：8,314,923；
- schema：`guandan-marvel-evaluation-checkpoint-v1`；
- algorithm/architecture：VRPO / v1.3；
- training provenance：complete；
- training resume：false。

## Bounded Evaluation

- 65 deals × 4 rotations × 2 opponents = 520/520 complete；
- selection dataset hash 与 IPPO 相同：`c8d11821a0a9cf8969c304fdbdfaa371c0addbc6883cf20e9617e526b6b5b583`；
- overall：305/520，58.65%；
- descriptive Wilson 95%：54.37%–62.81%；
- vs random：183/260，70.38%；
- vs rule：122/260，46.92%；
- illegal/truncated：0/0；
- evaluation elapsed：1,609.65 秒；
- rendezvous complete，runner exit 0。

该结果不是 formal promotion 或 final holdout，不能单独声明强度等级。

## IPPO/VRPO 当前描述性对比

相同预算和相同 selection anchors：

| 指标 | IPPO | VRPO |
|---|---:|---:|
| completed games | 400 | 400 |
| overall anchor | 59.62% | 58.65% |
| vs random | 73.85% | 70.38% |
| vs rule | 45.38% | 46.92% |
| health failures | 0 | 0 |
| session seconds | 2,800.01 | 4,478.77 |

两者 anchor 结果接近且方向不一致：IPPO 对 random/overall 略高，VRPO 对 rule 略高。不能仅凭这些独立 anchor 点估计跳过预先锁定的 paired direct head-to-head。

## 结论与下一步

VRPO controlled pilot 通过。现在 IPPO、VRPO 都有完整 400-game training、recovery、frozen checkpoint 和 shared anchor evaluation。

下一步必须在锁定的 65 deal clusters 和四座位轮换上运行 IPPO-vs-VRPO direct head-to-head，计算 clustered 95% CI；完成前不得选择主算法、启动 main long training 或运行 final holdout。
