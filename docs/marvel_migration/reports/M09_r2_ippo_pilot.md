# M09 r2 IPPO Controlled Pilot 验收报告

- 日期：2026-10-13
- 分支：`codex/marvel-guandan-migration`
- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`
- package id：`guandan-marvel-m09-algorithm-pilot-20261011-r2`
- Kaggle run id：`20261013T051332Z-ippo_pilot-09b28e94`
- 真实环境：Kaggle Tesla T4×2 / NCCL DDP
- Pilot 状态：`accepted_as_controlled_algorithm_pilot`
- M09 阶段 accepted：**否**

## 训练预算

- algorithm：IPPO；
- iteration：100/100；
- durable updates：100；
- completed games：400/400；
- env steps：122,426；
- game steps：47,893；
- wall-time stop：否；
- session elapsed：2,800.01 秒；
- budget contract：`m09-ippo-vrpo-equal-games-v1`。

100 条 iteration metrics 从 1 到 100 连续完整。所有 health 指标最大值均为 0：illegal、truncated、nonfinite loss/gradient、action capacity overflow、horizon overflow。

## 长评估与 Rendezvous 修复

- evaluation：520/520 games complete；
- 65 deals × 4 seat rotations × 2 opponents，矩阵无重复、无缺失；
- evaluation elapsed：1,590.65 秒；
- filesystem rendezvous schema：`guandan-marvel-ddp-evaluation-rendezvous-v1`；
- rank 0 发布 complete event；
- rank 1 未发生 NCCL watchdog timeout；
- 最终两 rank 均进入 `phase=complete`；
- runner exit 0。

这是真实远端对 M09 evaluation rendezvous 补修的通过证据。

## Bounded Evaluation 结果

总体：

- wins：310/520；
- descriptive win rate：59.62%；
- descriptive Wilson 95%：55.34%–63.75%；
- illegal/truncated：0/0。

分对手：

- vs random：192/260，73.85%；
- vs rule：118/260，45.38%。

该评估是 shared selection anchor 的 bounded pipeline evaluation，不是 formal promotion，也不是 final holdout。对 random 的结果可作为描述性 E1 候选信号；对 rule 尚未达到 50% 点估计。不得据此单独宣布 E1/E2/E3 或模型可实战。

## Checkpoint 与 Recovery

冻结模型：

- `evaluation_checkpoints/iter000100.pt`；
- SHA-256：`b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c`；
- bytes：8,316,529；
- schema：`guandan-marvel-evaluation-checkpoint-v1`；
- architecture：v1.3；
- training provenance：complete；
- training resume：false。

Recovery：

- ZIP SHA-256：`8b05b05f0d2f3715f2fb9701cbc4450a43d156a7cf2387eb916c2e9958c8b067`；
- manifest status：complete；
- iteration：100；
- DDP checkpoint v2：通过；
- world size：2；
- rank000/rank001 shard hash：通过；
- checkpoint manifest hash：`8b0cd55ace5214695c563b9faedfb21c1f7fd817d6da7840229c8cadba8655d9`；
- output/recovery 中 7 份关键小文件逐字节一致。

Retention 保留 iter75/iter100 和 iter100 evaluation checkpoint，符合 keep-last 2 配置。

## 结论

r2 IPPO 已完成相同完成对局预算、durable checkpoint、完整 recovery 和 shared bounded evaluation，作为 M09 controlled algorithm pilot 通过。

M09 整体仍 incomplete：VRPO 尚未运行，尚未完成 IPPO/VRPO head-to-head 和主算法选择，不能开始 main long training 或 final holdout。

## 下一步

运行 r2 VRPO pilot：

`D:\project\kaggle_train\guandan-marvel-m09-algorithm-pilot-20261011-r2\datasets\vrpo_pilot`

使用同一个 r2 Notebook、T4×2，只添加 VRPO Dataset。完成后返回完整 console log、`guandan-marvel-output.zip` 和 `guandan-marvel-recovery.zip`。
