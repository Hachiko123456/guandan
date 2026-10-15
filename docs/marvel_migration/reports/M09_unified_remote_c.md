# M09 Unified Pipeline Remote C 结果

- 日期：2026-10-15
- 状态：`accepted_complete`
- 输入：Remote B2 completed smoke plan/candidate
- 单 worker：1×T4 evaluation
- 双 worker：2 个独立 T4 workers
- 正式 Segment1000：未运行
- final holdout：未运行
- M09 accepted：否

## 分片与完成度

- plan hash：`afa80afaecabc6e3e5df50ccac589778e11957294cb75efb1e314d30637d82cd`；
- task count：40；
- worker0：20/20；
- worker1：20/20；
- sharding：`task_index % 2`；
- 两个 worker 均 exit=0。

## 单/双 GPU 一致性

全部通过：

- records_identical=true；
- ordering_identical=true；
- records_hash_identical=true；
- summary_hash_identical=true；
- promotion_input_identical=true；
- field_mismatches=[]；
- complete=true。

Consistency hash：`59d3cfcc42f8edacc2710fc14b0d0eb96994fe004288818e7356e44f51968092`。

## 完整性

- review bundle SHA-256：`5f564217f9db0863c7173cf4e2d8f05c459aa76c6d5d0c883d7298f96f9b2da2`；
- bundle CRC：通过；
- Remote C log SHA-256：`b33ba7046b19f5ed776539daa4d12483ee6edc4552c0c352f47b5319878e9cd4`；
- final_holdout_generated=false。

## 结论

Remote C accepted。双 GPU formal evaluation 已证明与单 worker 对固定 plan 的 durable records、merge ordering、summary identity 和 promotion input 完全一致。Remote A/B/C 全部通过，可以准备 production-enabled Dataset control；正式训练仍须从原始 iter500 recovery 开始，先只运行 Segment1000，自动连续 Segment 保持关闭。
