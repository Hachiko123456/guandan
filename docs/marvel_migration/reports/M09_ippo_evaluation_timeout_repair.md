# M09 IPPO Pilot 长评估 NCCL 超时补修报告

- 日期：2026-10-11
- 分支：`codex/marvel-guandan-migration`
- 失败远端 run：`20261011T065938Z-ippo_pilot-a62bd81e`
- 失败源码 commit：`18507bdaa8dd86b25817060d64852a7531cbd8e9`
- 补修源码 commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`
- 补修 package：`guandan-marvel-m09-algorithm-pilot-20261011-r2`
- 阶段状态：`remote_repair_ready`
- M09 accepted：**否**

## 用户返回日志

`C:\Users\yhx\Downloads\guandan-marvel-m09-t4x2 (1).log`

- bytes：367,593；
- SHA-256：`11269638f074aff8cd0535be399605afa9b74f442d3fb2b1a52e910775c61188`；
- 归档：`runs/M09_full_training/remote/ippo_pilot_failed_20261011/kaggle-console.log`。

本次仅收到 console log，未收到该失败会话的 output/recovery ZIP，因此日志中出现的 checkpoint 仍未完成下载后 hash/load 审查。

## 失败前实际完成量

日志明确显示：

- iteration：100/100；
- completed games：400/400；
- env steps：122,426；
- game steps：47,893；
- checkpoint 路径：`checkpoints/iter000100`；
- 两张 T4 在 rollout/learner 阶段均有真实计算；
- 失败发生在训练完成后的 `phase=evaluation`。

因此不是训练 loss、OOM、非法动作或训练预算未完成导致的失败。由于没有下载 checkpoint/recovery，仍不能把这 400 games 标记为已验收的 IPPO pilot。

## 根因

M09 selection anchor 为：

- 65 base deal clusters；
- 四座位轮换；
- random/rule 两个 opponent；
- 共 520 games。

M08 bounded evaluation 8 games 约需 16–20 秒，因此该评估自然可能超过 600 秒。

旧 worker 的顺序为：

1. rank 0 在独立子进程同步运行完整 evaluation；
2. rank 1 立即进入 `all_gather_object` 等待 rank 0；
3. rank 0 尚未进入 collective；
4. rank 1 的 NCCL `ALLGATHER` 等待 600,030 ms，超过默认 600 秒 watchdog；
5. rank 1 SIGABRT，torchrun 终止 rank 0，Notebook runner exit 1。

远端 stack 精确定位到旧源码：

`guandan_marvel/kaggle/worker.py:605`

NCCL 状态：rank 1 last enqueued 3634 / completed 3633；rank 0 last enqueued/completed 3633。这证明 collective 顺序在长评估窗口内不一致，而不是普通网络抖动。

## 补修

新增 `guandan-marvel-ddp-evaluation-rendezvous-v1`：

- rank 0 完成或失败后，原子发布 `evaluation_rendezvous_iterNNNNNN.json`；
- non-primary rank 在共享 run filesystem 轮询该文件；
- 等待上限为 evaluation timeout + 120 秒；
- 长评估期间不再挂起 NCCL collective；
- schema/iteration/event malformed 均 fail closed；
- rank 0 evaluation failure 通过 durable event 传播给所有 rank；
- evaluation 完成后各 rank 才进入最终短 barrier。

这不是简单提高 NCCL timeout，而是删除错误的“用 NCCL collective 等待单 rank 长时间 I/O/evaluation”设计。

## 补修交付

新交付根目录：

`D:\project\kaggle_train\guandan-marvel-m09-algorithm-pilot-20261011-r2`

身份：

- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`；
- package id：`guandan-marvel-m09-algorithm-pilot-20261011-r2`；
- plan hash：`5ca94bd3d930b86f99e5bdc329e0405fbaa0023ef64ff47cd5564b457b84f8f7`；
- source ZIP SHA-256：`d08b8743970b8cce38a5ffe7768a3169b117a76ba2c03c596acafadd7e7cd465`；
- locked config SHA-256：`6994f7f80eaeb8e44c695cac8d21c6bb03561e6a8615866d47e8dcc933f67d21`；
- delivery validation：通过；
- 旧 package id 已加入 forbidden reuse 列表。

## 测试

- `tests/marvel_kaggle`：49 passed；
- training/core/evaluation/acceptance target regression：92 passed；
- 合计：141 passed，0 failed/error/skipped；
- rendezvous delayed-publish 测试：通过；
- stale iteration rejection：通过；
- worker source 不再包含 `all_gather_object(eval_event...)`：通过；
- r2 delivery build/validation：通过。

## 恢复/重跑策略

### 优先保全旧训练结果

失败 runner 的 `finally` 设计会尝试生成：

- `/kaggle/working/guandan-marvel-output/guandan-marvel-output.zip`；
- `/kaggle/working/guandan-marvel-output/guandan-marvel-recovery.zip`。

如果失败会话仍可打开，应立即下载两者并返回。本地可审查 iter100 checkpoint，提取 Demo actor，并决定是否建立一次性 evaluation-only salvage；不要把旧 recovery 与 r2 Dataset 直接混用，因为 source identity 不同。

### 若旧 recovery 无法下载

使用 r2 重新运行。由于 source commit 已变化，必须先运行 r2 preflight，再运行 r2 IPPO pilot。旧 `ippo_pilot` Dataset 不得继续使用。

## 结论

本次失败是可复现的 post-training evaluation orchestration 缺陷。训练日志已达到计划 400 games，但因 checkpoint/recovery 尚未下载且 evaluation 未完成，IPPO pilot 保持 incomplete。补修已本地验证并生成 r2 交付；仍需真实 r2 Kaggle preflight/pilot 证据。
