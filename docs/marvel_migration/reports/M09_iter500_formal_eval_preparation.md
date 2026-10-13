# M09 iter500 Formal Evaluation/Champion 交付准备报告

- 日期：2026-10-14
- 状态：`remote_ready`
- 控制交付根目录：`D:\project\kaggle_train\guandan-marvel-m09-iter0500-formal-eval-20261014`
- M09 accepted：**否**

## 输入复用设计

本交付不会要求重新上传 segment0500 recovery、iter500 rank shards、source ZIP 或 iter100 champion。Kaggle Notebook 直接添加两个已有 Notebook Output 作为 Input：

1. `guandan_marvel_m09_main_training` 的 segment0500 Output：自动发现 iter500 complete recovery 和已编译 r2 source/runtime；
2. r2 IPPO Pilot Notebook Output：自动发现 SHA `b00a2c...` 的 iter100 evaluation checkpoint。

仅需一次性创建一个约 110 KB 的 control Dataset，包含 runner、legacy V1 anchor 和 hash-bound control manifest。后续 formal evaluation 可以继续复用该 control Dataset 和前序 Notebook Outputs。

## 评估计划

- Candidate：iter500 training checkpoint，运行时冻结为 evaluation-only checkpoint；
- Current champion：r2 IPPO iter100；
- Selection：32 independent deal clusters × 4 rotations；
- Fixed matrix：champion、rule、random、legacy V1、opponent pool；
- Promotion：formal，CI low threshold 0.52，min 32 clusters，rule ≥0.40，random ≥0.70；
- provenance：formal；
- final holdout：不使用。

评估结果、champion/pool manifests、候选/初始 champion checkpoint 和逐局记录会打包为：

`/kaggle/working/guandan-marvel-formal-eval-iter0500.zip`

## Control 身份

- control manifest hash：`75bd050d935d69bcb8ace7107c3c2088c3bdb806888d2a1f2513ef5b77d94800`；
- runner SHA-256：`b6f6d7378441582f1ac37acba41f7077fb4d857f75ffc0a34134dc3e69c25408`；
- legacy anchor SHA-256：`880e74c1c41a5be252c1319f9be3ccb1f19553c4e63fe4658198b57c850b1f1d`。

## 下一步

用户创建 control Dataset，导入 formal-eval Notebook，并添加 control Dataset、segment0500 Notebook Output、r2 IPPO Pilot Notebook Output 三个 Input。完成后返回 formal-eval ZIP 和完整日志。segment1000 在评估/晋升审查前继续 blocked。
