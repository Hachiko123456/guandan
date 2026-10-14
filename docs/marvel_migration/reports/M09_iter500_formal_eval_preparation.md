# M09 iter500 Formal Evaluation/Champion 交付准备报告

- 日期：2026-10-14
- 状态：`remote_ready_v4_with_backend_identity_legacy_profile_and_observability_repairs`
- 控制交付根目录：`D:\project\kaggle_train\guandan-marvel-m09-iter0500-formal-eval-20261014`
- M09 accepted：**否**

## 输入复用设计

本交付不会要求重新上传 segment0500 recovery、iter500 rank shards、source ZIP 或 iter100 champion。Kaggle Notebook 直接添加两个已有 Notebook Output 作为 Input：

1. `guandan_marvel_m09_main_training` 的 segment0500 Output：自动发现 iter500 complete recovery 和已编译 r2 source/runtime；
2. r2 IPPO Pilot Notebook Output：自动发现 SHA `b00a2c...` 的 iter100 evaluation checkpoint。

仅需一次性创建一个约 110 KB 的 control Dataset，包含 runner、legacy V1 anchor 和 hash-bound control manifest。后续 formal evaluation 可以继续复用该 control Dataset 和前序 Notebook Outputs。

## Backend identity 路径修复

首次远程运行已成功发现 iter500 recovery 和 iter100 champion，但 candidate 加载时报告 `checkpoint C++ backend/build identity mismatch`。原因是训练 checkpoint 记录的扩展路径位于 `/kaggle/working/guandan-marvel-source`，旧 runner 却直接从只读的 `/kaggle/input/notebooks/.../guandan-marvel-source` 导入同一份二进制；绝对路径参与 backend identity hash，造成内容相同但身份 hash 不同。

修复后的 runner 会先把 hash-bound source/runtime 完整复制到 `/kaggle/working/guandan-marvel-source`，再次核验 source manifest 和 runtime binary SHA-256 后才导入。不得使用符号链接，因为 `Path.resolve()` 会重新解析回 Input 路径。segment0500 recovery 和 iter100 champion 无需重新上传。

## Legacy V1 profile catalog 修复

第二次远程运行已经通过 iter500 candidate 与 iter100 champion 的 C++ backend identity 校验，证明 source/recovery/champion Inputs 均正确；随后在加载 legacy V1 anchor 时失败。legacy checkpoint 的 metadata 包含 `profile=local_fast`，通用 checkpoint loader 会从项目根目录读取 `configs/acceptance_profiles.json` 验证该 profile，但 M09 的最小 source manifest 没有包含这份 A07 配置。

修复后的 control Dataset 新增 hash-bound `acceptance_profiles.json`。runner 在导入 source 后将其复制到 `/kaggle/working/guandan-marvel-source/configs/acceptance_profiles.json`，再加载 legacy anchor。该文件只用于验证 legacy checkpoint metadata，不改变 MARVEL V2 规则、C++ runtime 或 iter500/iter100 模型。

## 持续进度与 GPU 可观测性修复

第三次远程尝试已进入 640 局 formal selection，但旧 runner 在 `run_selection` 返回前不向 stdout 输出任何进度，用户在约 500 秒时无法区分正常长评估与卡死并主动暂停。评估循环本身有固定的 5×32×4 对局上限且单局有 4096 token-step 上限；问题属于交付可观测性不足。

v4 runner 每 20 秒输出一条 `FORMAL_EVAL` JSON heartbeat，包含 phase、completed/total、当前 opponent lane、lane 内完成数、百分比、平均 games/s、ETA、无进展秒数、games.jsonl 字节数，以及每张 GPU 的利用率、显存和功耗。source discovery、materialization、recovery、candidate export、policy validation、evaluation、archive 也分别输出明确 phase 日志。

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

- control manifest hash：`5d67c99ae8c4d1179dab5b5156bd227fc35f12230f2ca55586887e84cb0921bd`；
- runner SHA-256：`fd9d3058844e613fc1a21f4d9519bbd4f3cfd7ddc1dc140dc5381e132b3b5780`；
- legacy anchor SHA-256：`880e74c1c41a5be252c1319f9be3ccb1f19553c4e63fe4658198b57c850b1f1d`；
- acceptance profiles SHA-256：`da0f42c5fd0798aabf01d1124da9f981f4c3a8ee9d2c35efda81825c1ed7b6e7`。

## 远程结果

combined-v4 已于 2026-10-14 在 Kaggle T4×2 完成：640/640 games、illegal/truncated=0、runner exit=0。iter500 candidate 对 iter100 champion 点估计为 57.81%，但 clustered CI lower bound 未达到 0.52，promotion decision 为 `inconclusive`，因此不晋升并保留 iter100 champion。完整结果见 `M09_iter500_formal_eval.md`；formal review 已完成，segment1000 现可按 iter500 recovery exact resume。
