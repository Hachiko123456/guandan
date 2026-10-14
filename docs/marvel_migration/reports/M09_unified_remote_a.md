# M09 Unified Pipeline Remote A 结果

- 日期：2026-10-14
- Kaggle：T4×2
- 模式：Remote A / unified small-budget smoke
- 状态：`accepted_complete`
- 正式 Segment1000：未运行
- final holdout：未运行
- M09 accepted：否

## 训练

从已验证 iter500 recovery exact resume：

- iteration：500 → 502；
- completed games：2,000 → 2,008；
- global env steps：601,674；
- global game steps：230,525；
- world size：2；
- 两个 rank shard 完整；
- training commit hash：`b72ac7fa982b182baa33b08d07f30e3182e359aeb5f78c08f83dac087d43af76`；
- committed recovery SHA-256：`1aea0ff39e4382e79d2532661bb4bd392ad7049ac501ae6b156d738c7f793a10`；
- checkpoint manifest hash：`015b8ecd216cd6dea48f0dbec9b1780f90617cce113a3b94dd3f67155a45c27f`。

DDP runner exit=0，训练 checkpoint/recovery commit 完成后 DDP 退出，随后才启动独立 evaluation worker。

## Evaluation

固定 smoke plan：2 clusters × 4 rotations × 5 lanes = 40 tasks。

- 40/40 complete；
- unique tasks：40；
- champion/rule/random/legacy/opponent-pool 各 8 局；
- illegal=0；
- truncated=0；
- worker status=`complete`；
- plan hash：`8f78e2fd7673aac143e5214bf42a92a617b3d234774c155c83f97faf516a6727`；
- evaluation hash：`2bf0967783c659c2eb3c10a1d81aa9c995b7ef0ee2055b1b86a1f24cbd59fe78`；
- records hash：`5b058811f6e214d208add889e5d5f66091861cd20ac0ddc1668289d850204826`；
- summary hash：`ffe5e079e340f5179b8f8594e54ec8f9c949caf3d7212ebaeb4b2917780de41e`。

总体点估计 win rate=75%，但该数据仅为 2-cluster smoke，不构成强度结论。

## Promotion/Champion

- policy：smoke；
- decision：`inconclusive`；
- promote=false；
- 原因：smoke policy 永不自动晋升；
- decision hash：`9b4fd0f6ecbedaecb08f665b75c7577b436dfef9ebf8b7ac099c80b5246a75c8`；
- champion 保持 generation 1 / iter100；
- champion SHA-256：`b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c`。

## Continuation 完整性

- ZIP bytes：76,296,774；
- ZIP SHA-256：`be6c07d267af388a922ce17da48e151742340b04ae68d9a6bea892b0c6a72016`；
- CRC：通过；
- 20 entries；
- manifest inventory/hash：通过；
- embedded recovery CRC/hash：通过；
- pipeline phase：`COMPLETE`；
- error：null；
- final_holdout_generated=false。

## 结论

Remote A 的训练、training commit、DDP/evaluation 进程隔离、formal-record binding、continuation packaging 和 heartbeat 均通过真实 Kaggle T4×2 验证。Remote A accepted。

下一步是 Remote B fault/resume；Remote B 和 Remote C 通过前，不得启用 production Segment1000。
