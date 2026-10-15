# M09 Unified Segment1000 结果

- 日期：2026-10-15
- 状态：`accepted_complete_no_promotion_segment1500_authorized`
- training：iter1000 / 4000 games
- formal evaluation：640/640
- promotion：inconclusive / false
- champion：仍为 iter100
- final holdout：未运行
- M09 accepted：否

## Training

- training commit hash：`4d5497289a97100f8482a0ba0f90c90586071105389bc1b25612c86f7a00f99e`；
- recovery SHA-256：`9d7f94c59258382794385c99a65178fee888d26b0e4a6a58ea77066ffb8b146d`；
- checkpoint manifest hash：`442fe60b032b2ba362415aa93a43764290397831663540343ea132aa6dedb5fe`；
- global env steps：1,189,344；
- global game steps：452,192；
- two rank shards 完整；
- health failures：0。

## Formal evaluation

双 worker 各 320 tasks，640/640 complete，illegal/truncated=0。

| 对手 | 胜局 | 对局 | 胜率 | CI95 low | high |
|---|---:|---:|---:|---:|---:|
| champion iter100 | 72 | 128 | 56.25% | 0.00% | 100.00% |
| rule | 72 | 128 | 56.25% | 3.86% | 100.00% |
| random | 103 | 128 | 80.47% | 32.59% | 100.00% |
| legacy | 104 | 128 | 81.25% | 33.23% | 100.00% |
| opponent pool | 72 | 128 | 56.25% | 0.00% | 100.00% |

- evaluation hash：`f533b4be34c03df10b6ed03609fc3121f8eedc96bced090a1971fd69f2753195`；
- records hash：`e1be3249cc484c41412c5500a6859c48bc204f5f7c7175dfb630d8ad30e53a88`；
- summary hash：`6fffe34a3ce828e0fd76f063bccb11a5d23f27c429d69dca37cdb1df06889016`。

## Promotion / gate

Candidate iter1000 SHA-256：`cc686624e62db74280ea0dce3e9e21ce87c09206be795a83c7bc1ec5f3496930`。

Candidate-vs-champion 点估计 56.25%，但 bounded CI lower bound 未达到 0.52，因此：

- status=`inconclusive`；
- promote=false；
- champion 保持 generation1 / iter100；
- rule/random floors 通过；
- health 通过；
- formal matrix 完整。

按冻结 continuation policy，Segment1500 可从 iter1000 training recovery exact resume。

## 边界

该结果不证明 E2/E3；主训练分布仍为 Rank.TWO/no-previous-result。final holdout 未运行，M09 仍 incomplete。
