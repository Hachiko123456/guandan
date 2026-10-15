# M09 Unified Pipeline Remote B 结果

- 日期：2026-10-15
- 状态：`accepted_complete`
- Remote A：accepted
- Remote B：accepted
- Remote C：pending
- production Segment1000：未运行
- final holdout：未运行
- M09 accepted：否

## B1 受控失败

- iteration=502 / completed games=2008；
- training commit hash：`d0233efa3ed20f2df009e67a01978cd14cae7fdf2b00f28ac728213d789a617f`；
- recovery SHA-256：`aa7669388c72806a83454c41309ad4fa9cfcd4e7059c25bab9f872afdb3a38c4`；
- plan hash：`afa80afaecabc6e3e5df50ccac589778e11957294cb75efb1e314d30637d82cd`；
- pipeline phase：`TRAINING_COMPLETE_EVALUATION_FAILED`；
- worker failure：`controlled formal evaluation failure`；
- 已保存 records：8，task index 0–7；
- illegal/truncated=0；
- continuation SHA-256：`902d11776aaeb9c8c3ee2132ded42651c6389ae187569ee6901aca28957a20a9`。

## B2 恢复

- 没有产生新的 training commit；
- training commit hash 与 B1 完全相同；
- recovery SHA-256 与 B1 完全相同；
- checkpoint/rank shards/global counters 与 B1 完全相同；
- plan hash 与 B1 完全相同；
- `completed_before=8`；
- `completed_now=32`；
- 最终 40/40 complete；
- B1 前 8 条 records 与 B2 前 8 条逐字节/内容一致；
- illegal/truncated=0；
- pipeline phase=`COMPLETE`；
- continuation SHA-256：`77c8f4e3ff2e6768f020f699b048a6bc1bfec31fa637ba29f5d95272c9085da2`。

## Evaluation/Promotion

- evaluation hash：`905be3fc425e5e3ba889f90cbaf1d97be292c210cf2f435d4854a7936b7eb5b8`；
- records hash：`cd61049b5c113adafd27cba16ca299d4f771352388ce645e0cea91188fe80ea8`；
- summary hash：`db106f3410080fc7765cb9031f53b285b1f1ab1459dd0484532eb7bf4fe63b2c`；
- promotion：`inconclusive` / `promote=false`；
- decision hash：`3864c490075687b10aa00ef6afeeef26550eadb8d497a0b8eb9e0a65d80746d4`；
- champion 保持 iter100。

## 结论

Remote B 证明 evaluation 失败不会损坏 committed training recovery；下一次运行能识别 pending segment、跳过训练、保留已有 records 并只补缺失 tasks。Remote B accepted。

下一步是 Remote C 单/双 GPU deterministic consistency。Remote C 通过前 production Segment1000 继续 fail-closed。
