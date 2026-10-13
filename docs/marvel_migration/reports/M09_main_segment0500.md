# M09 IPPO Main Training Segment 0500 报告

- 日期：2026-10-13
- run id：`20261013T135603Z-main_ippo_segment0500-dc3dfe4e`
- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`
- algorithm：IPPO
- 状态：`training_passed_evaluation_pending`
- M09 accepted：**否**

## 训练结果

从已验证 iter100 recovery exact resume：

- 起点：iteration 100 / 400 games；
- 终点：iteration 500 / 2,000 games；
- 本段新增：400 iterations / 1,600 games；
- env steps：599,154；
- game steps：229,509；
- session elapsed：4,795.41 秒，约 79.9 分钟；
- wall budget stop：否；
- runner exit：0。

Metrics 从 iteration 101 到 500 连续 400 条。illegal、truncated、nonfinite loss/gradient、action capacity overflow、horizon overflow 全部为 0。

## Checkpoint/Recovery

- 最新 durable checkpoint：iter500；
- retained：iter475、iter500；
- iter450 已按 keep-last 2 删除；
- recovery status：complete；
- recovery SHA-256：`2d395ae4293e42c0b07f28e9b6d8508aa3c07aa1419d11ce65cb53598a43cb44`；
- DDP world size 2、algorithm IPPO、iteration/durable updates 500；
- 两 rank shard hash 验证通过；
- 两 rank resume/merged actor state hash 一致；
- output/recovery 七份关键 artifacts 逐字节一致。

本段按计划不在 DDP 训练进程中运行长 evaluation，因此 output 没有 iter500 evaluation checkpoint，`evaluation_summary=not_run` 是预期状态，不是缺失或失败。

## 当前训练阶段

累计训练量已经从 Pilot 的 400 games 增加到 2,000 games，管线、吞吐、恢复和健康状态稳定。按本段实测，继续到 iteration2000/8,000 games 在分段方案下可行。

但“完整可用模型”不能只由训练量定义。当前仍必须完成 iter500 formal selection/champion evaluation，包括 current champion、rule、random、legacy V1 和 opponent pool，并根据 32-cluster gate 决定是否晋升。完成该审查前，segment1000 不授权运行。

此外主训练仍使用 Rank.TWO/no-previous-result 分布；全级牌/贡还能力由广覆盖评估约束。如果不能达到 rule/legacy 门槛，即使训练到 8,000 games，M09 也不能 accepted，需要独立训练分布迁移。

## 下一步

准备 iteration500 的独立 formal evaluation/champion Kaggle 交付。用户当前不应直接运行 segment1000。评估通过并记录 champion/pool 后，再继续增加训练量。
