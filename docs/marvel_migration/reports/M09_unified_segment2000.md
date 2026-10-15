# M09 Unified Segment2000 结果与 Final Candidate Freeze 准备

- 日期：2026-10-15
- 状态：`accepted_complete_no_promotion_training_plan_complete_final_candidate_freeze_authorized`
- training：iter2000 / 8000 games
- formal evaluation：640/640
- promotion：inconclusive / false
- champion：仍为 iter100
- final holdout：未运行
- M09 accepted：否

## Training

- training commit hash：`ca96a98c5315fb03a091d9c34448d4424f6ac6cd561022f98d6cc4dabe0576b9`；
- recovery SHA-256：`50ab0f413873119b49c0f86592bd773ef9f206531a4f6bb5394e05a18b503aae`；
- checkpoint manifest hash：`239da3f682714b22b0cc6b02efb2e2a2e1f4eff906811bfa040a41385b74c471`；
- global env steps：2,348,788；
- global game steps：886,810；
- two rank shards 完整；
- health failures：0。

## Formal evaluation

双 worker 各 320 tasks，640/640 complete，illegal/truncated=0。

| 对手 | 胜局 | 对局 | 胜率 | CI95 low | high |
|---|---:|---:|---:|---:|---:|
| champion iter100 | 80 | 128 | 62.50% | 8.38% | 100.00% |
| rule | 84 | 128 | 65.63% | 13.71% | 100.00% |
| random | 114 | 128 | 89.06% | 44.30% | 100.00% |
| legacy | 110 | 128 | 85.94% | 37.95% | 100.00% |
| opponent pool | 80 | 128 | 62.50% | 8.38% | 100.00% |

- evaluation hash：`314b472d87013837089d3aa8c976b7c3f48f09212d78bd1520ab4076d8cdd161`；
- records hash：`cf0492c2e27728a2e56096d8f803425fb2b4ee38f74358bfda5f1ee9e172ee99`；
- summary hash：`3f6342d69870b1621b17826b2a32c9bba2447b9d4f706d4dda0382ed03e14d22`。

## Promotion / gate

Candidate iter2000 SHA-256：`af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8`。

Candidate-vs-champion 点估计为 62.50%，是当前四个正式 candidate 中最高，但 conservative CI lower bound 仍只有 8.38%，未达到 0.52：

- status=`inconclusive`；
- promote=false；
- champion 保持 generation1 / iter100；
- rule/random floors 通过；
- health 通过；
- formal matrix 完整。

## 计划边界

iter2000 是当前 M09 主训练计划的 terminal checkpoint。continuation `next_target=null`，训练计划已完成，不能继续自动跑 Segment2500。

final candidate 现在可以按 M09 计划显式冻结，但必须先决定冻结规则，然后只运行一次 final holdout。final holdout 尚未运行，不能宣称 M09 accepted 或 E1/E2/E3。
