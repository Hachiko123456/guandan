# M09 IPPO-vs-VRPO 算法选择最终报告

- 日期：2026-10-13
- package：`guandan-marvel-m09-algorithm-selection-20261013`
- selection manifest hash：`120b53ea104373fc0f0f26a146c005bc16e31a2dbc8992580d427d4cbd8a5811`
- decision hash：`f12421fd57b85fe3067974953382f59719da2fda4edbd2b35a2faa7dbda791bf`
- 状态：`complete`
- 选择算法：**IPPO**
- M09 accepted：**否**

## 完整性

`guandan-marvel-algorithm-selection.zip`：

- bytes：28,252；
- SHA-256：`65daedee28476f160448dba7b7e5a7915f5bacdf7a59aa479b1d42714b3c0aff`；
- CRC：通过；
- 5 个文件，无不安全路径；
- 单独下载的 JSON/JSONL 与 ZIP 内语义完全一致；逐局 JSONL 字节完全一致。

Manifest/decision canonical hash 均通过，summary 内嵌 decision 与独立 decision 完全一致。

## Direct Head-to-head

方向：VRPO candidate vs IPPO baseline。

- 65 独立 deal clusters；
- 每 cluster 四座位轮换；
- 260/260 games complete；
- VRPO wins：116；
- VRPO point win rate：44.62%；
- IPPO point win rate：55.38%；
- illegal/truncated：0/0；
- candidate/opponent checkpoint hash 与锁定 checkpoint 完全匹配。

按四轮先聚合后的 bounded empirical-Bernstein 95% CI：

- VRPO mean：44.62%；
- low/high：13.52%–75.71%。

该区间跨过 50%，因此 direct gate inconclusive。`conclusive=true` 只表示区间可计算，不表示胜负显著。

## Paired Rule Anchor

- 65 paired clusters；
- 260 paired rows；
- VRPO-IPPO point difference：+1.54pp；
- 95% CI：-50.62pp 至 +53.70pp。

区间跨过 0，因此 paired rule fallback 也 inconclusive。

## 决策

预定义 fallback 被触发：

```text
status = retain_ippo_inconclusive
selected_algorithm = ippo
reason = direct and paired rule-anchor intervals are inconclusive; retain the locked M08 IPPO baseline
```

选择 IPPO 不是“证明 IPPO 显著更强”，而是在受控证据无法区分时执行事先锁定的 baseline fallback。该选择还与资源事实一致：IPPO pilot 用时约 2,800 秒、每 rank 约 6.8 GB；VRPO 约 4,479 秒、每 rank 约 13.6 GB。

## 边界

- 未运行 final holdout；
- 未晋升 champion；
- 未作 E1/E2/E3 正式声明；
- selection 数据不得用于 final 无偏结论；
- VRPO 结果与 checkpoint 继续保留，不覆盖。

## 下一步

冻结 IPPO 为 M09 main algorithm，单独审查并冻结 long-training 的级牌/贡还 training distribution、session budget、checkpoint/evaluation/champion 计划，然后生成新的 main full/resume Kaggle 交付。只有完成 long training、champion history、锁定 final candidate 和一次 final holdout 后，M09 才可能 accepted。
