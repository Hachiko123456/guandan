# M09 Unified Segment1500 结果

- 日期：2026-10-15
- 状态：`accepted_complete_no_promotion_segment2000_authorized`
- training：iter1500 / 6000 games
- formal evaluation：640/640
- promotion：inconclusive / false
- champion：仍为 iter100
- final holdout：未运行
- M09 accepted：否

## Training

- training commit hash：`b6bdddfc030727a55da8f1c73125e6eb42d343ef5a2df2d7fd9dbab2dbc59c6f`；
- recovery SHA-256：`79c9527cfba604b590a3b0caa2e88e76798c6ad3409f849a8890e8388bfe68f7`；
- checkpoint manifest hash：`1cda6b32c6100967c7ebd4983aa0bf087d2ab7e533361d13db1083bfbe6c1bed`；
- global env steps：1,773,060；
- global game steps：672,962；
- two rank shards 完整；
- health failures：0。

## Formal evaluation

双 worker 各 320 tasks，640/640 complete，illegal/truncated=0。

| 对手 | 胜局 | 对局 | 胜率 | CI95 low | high |
|---|---:|---:|---:|---:|---:|
| champion iter100 | 78 | 128 | 60.94% | 5.88% | 100.00% |
| rule | 64 | 128 | 50.00% | 0.00% | 100.00% |
| random | 112 | 128 | 87.50% | 40.62% | 100.00% |
| legacy | 108 | 128 | 84.38% | 36.49% | 100.00% |
| opponent pool | 78 | 128 | 60.94% | 5.88% | 100.00% |

- evaluation hash：`f92e4605d205de7b59635a8005dfddbcc0109480db22f42131c1fb5afb931d59`；
- records hash：`8df311b06714b169ed5935874932fc9ab126b22d74738ef73d6e6da338fad816`；
- summary hash：`f19bc1fa6ccb8774fcc0b67082f839ac81af455b3f431fdd160d1fea72a9cc0f`。

## Promotion / gate

Candidate iter1500 SHA-256：`c6e19323e369be1ff08432e4198236bc92bab785a2bfd619e2cd7c103f1f496b`。

Candidate-vs-champion 点估计 60.94%，但 conservative CI lower bound 仍未达到 0.52：

- status=`inconclusive`；
- promote=false；
- champion 保持 generation1 / iter100；
- rule/random floors 通过；
- health 通过；
- formal matrix 完整。

按冻结 policy，Segment2000 可从 iter1500 recovery exact resume。

## 边界

点估计继续提高，但 32-cluster bounded interval 仍很宽，不能声明显著优于 champion。final holdout 未运行，M09 仍 incomplete。
