# M09 两阶段训练 Kaggle 操作教程

日期：2026-10-16
状态：阶段一训练交付与阶段一批量 selection 交付已生成；阶段二构建器已实现，等待 selection 选出真实 anchor 后生成最终 Stage2 Dataset。

## 0. 重要边界

本方案不会修改已经完成的原 M09 历史，也不会复用 2026-10-16 已消费的 final holdout。

流程只有两个训练阶段：

1. 阶段一：iter2000 exact resume 到 iter5000；
2. 阶段二：从阶段一选出的最佳 policy warm-start，重置 optimizer/replay，使用新上下文分布训练2000 iterations。

阶段一与阶段二之间只有一次批量 selection 作业，它是模型选择步骤，不是新的训练阶段。

---

# 第一部分：上传阶段一训练

## 1. 找到交付目录

阶段一目录：

```text
D:\project\kaggle_train\guandan-marvel-m09-two-stage-stage1-iter5000-v1
```

关键文件：

```text
guandan-marvel-m09-two-stage-stage1-upload.zip
guandan_marvel_m09_two_stage_stage1.ipynb
README_STAGE1_UPLOAD.md
```

其中上传 ZIP 的 SHA-256 和准确文件大小可以用下面命令检查：

```powershell
Get-FileHash "D:\project\kaggle_train\guandan-marvel-m09-two-stage-stage1-iter5000-v1\guandan-marvel-m09-two-stage-stage1-upload.zip" -Algorithm SHA256
```

## 2. 在 Kaggle 创建私有 Dataset

1. 打开 Kaggle；
2. 进入 `Datasets`；
3. 点击 `New Dataset`；
4. Dataset 名称建议：

```text
guandan-marvel-m09-two-stage-stage1-iter5000
```

5. Visibility 选择 `Private`；
6. 上传：

```text
D:\project\kaggle_train\guandan-marvel-m09-two-stage-stage1-iter5000-v1\guandan-marvel-m09-two-stage-stage1-upload.zip
```

7. Kaggle 如果自动解压 ZIP，确认 Dataset 根目录直接出现：

```text
guandan_marvel_dataset.json
guandan_marvel_bootstrap.py
guandan-marvel-source.zip
source_manifest.json
guandan-marvel-recovery.zip
m09_two_stage_control.json
marvel_m09_two_stage_continuation_plan.json
```

如果 Kaggle 没有自动解压，就改为上传交付目录下 `dataset/` 中的全部文件。不要把 `dataset` 文件夹本身再套一层，不要改文件名。

## 3. 导入 Notebook

导入：

```text
D:\project\kaggle_train\guandan-marvel-m09-two-stage-stage1-iter5000-v1\guandan_marvel_m09_two_stage_stage1.ipynb
```

Notebook 建议名称：

```text
guandan-marvel-stage1-iter5000
```

## 4. Notebook 设置

右侧 Settings：

```text
Accelerator: GPU T4 x2
Internet: Off 即可
```

Add Input 只添加刚创建的 Stage1 Dataset。

不要添加：

- 旧 production Dataset；
- Segment2000 Notebook Output；
- final holdout Dataset；
- 额外 recovery Dataset。

iter2000 recovery 已经嵌入 Stage1 Dataset。

## 5. 运行

点击：

```text
Save Version
→ Save & Run All
```

正常启动后应看到：

```text
stage1_bootstrap_stage
source_materialization
native_import
training_iter2000_to_5000
```

训练 heartbeat 应从：

```text
iteration=2000/5000
games=8000
```

继续增加。

预计训练时间约9.5～10小时。安全 wall budget 为42000秒。如果 Kaggle 速度明显偏慢，可能在5000前触发 budget stop；此时下载最新 recovery，创建 Stage1 Dataset 的新版本并替换原 recovery，再只添加新版本重新 Run All。这属于同一阶段恢复，不是新增训练阶段。

## 6. 阶段一完成标志

应看到：

```text
iteration=5000/5000
games=20000
phase=complete
```

Notebook 会提供三个下载链接：

```text
stage1-milestone-candidates.zip
guandan-marvel-recovery.zip
guandan-marvel-output.zip
```

其中候选包必须包含：

```text
ippo_iter003000.pt
ippo_iter004000.pt
ippo_iter005000.pt
stage1_milestone_manifest.json
```

下载并保留全部三个文件。不要自行把 iter5000 宣布为最佳模型。

---

# 第二部分：阶段一批量 selection

## 7. 找到 selection 交付

目录：

```text
D:\project\kaggle_train\guandan-marvel-m09-two-stage-stage1-selection-v1
```

文件：

```text
guandan-marvel-m09-stage1-selection-upload.zip
guandan_marvel_m09_two_stage_selection.ipynb
README_SELECTION_UPLOAD.md
```

## 8. 创建 selection 私有 Dataset

Dataset 名称建议：

```text
guandan-marvel-m09-stage1-selection-v1
```

上传：

```text
D:\project\kaggle_train\guandan-marvel-m09-two-stage-stage1-selection-v1\guandan-marvel-m09-stage1-selection-upload.zip
```

如果 Kaggle 不自动解压，则上传该交付目录的 `dataset/` 全部文件。

## 9. 导入 selection Notebook

导入：

```text
D:\project\kaggle_train\guandan-marvel-m09-two-stage-stage1-selection-v1\guandan_marvel_m09_two_stage_selection.ipynb
```

设置：

```text
Accelerator: GPU T4 x2
Internet: Off
```

Add Input 添加两个输入：

1. Stage1 selection Dataset；
2. 刚刚完成的 Stage1 Notebook Output。

不要添加旧 final holdout Dataset。

## 10. selection 做什么

一次运行会在同一套新 selection-v2 数据上评估：

```text
iter2000
iter3000
iter4000
iter5000
```

每个 candidate 使用：

```text
65 independent clusters
4 seat rotations
champion / rule / random / legacy / opponent-pool lanes
```

选择顺序已冻结为：

1. rule clustered CI lower；
2. rule point estimate；
3. 更小的 rule seat rotation gap；
4. champion clustered CI lower；
5. champion point estimate；
6. 完全相同时选更早 iteration。

所有候选必须满足：

```text
illegal=0
truncated=0
random >= 70%
legacy >= 70%
```

## 11. selection 输出

下载：

```text
stage1-selection-results.zip
stage2_anchor.pt
stage1_selection_decision.json
```

把这三个文件发回当前 Codex 聊天。我会核验 evaluation hashes、summary hashes、候选身份和选择规则，然后用真实 `stage2_anchor.pt` 构建正式 Stage2 上传包。

不要上传本地目录：

```text
D:\project\kaggle_train\_TEMPLATE_DO_NOT_UPLOAD_stage2_iter2000_smoke
```

它只是构建器烟雾验证产物，不是正式 Stage2 模型。

---

# 第三部分：生成并上传正式 Stage2

## 12. Stage2 构建命令

收到并核验 selection 输出后，使用：

```powershell
python scripts/build_m09_two_stage_stage2_delivery.py `
  --output D:\project\kaggle_train\guandan-marvel-m09-two-stage-stage2-final-v1 `
  --source-dataset D:\project\kaggle_train\guandan-marvel-m09-unified-production-segment1000-v1\dataset `
  --candidate <stage2_anchor.pt的绝对路径>
```

正式 Stage2 builder 会：

- 把被选中的 policy 作为 warm-start；
- 重置 optimizer；
- 重置 replay/sampler/counters；
- 使用全 level-rank、全 leader-seat 和显式 previous-result/tribute context schedule；
- 使用新的 source identity 和 semantic contract；
- base LR 设为 `1e-4`；
- 训练2000 iterations / 8000 games；
- 不运行旧 final holdout。

## 13. Stage2 Kaggle 设置

创建新的 Private Dataset，上传正式 Stage2 `dataset/` 全部文件，并导入：

```text
guandan_marvel_m09_two_stage_stage2.ipynb
```

设置：

```text
GPU T4 x2
Internet Off
只添加 Stage2 Dataset
```

Run All 后下载：

```text
guandan-marvel-output.zip
guandan-marvel-recovery.zip
```

随后还需要对 Stage2 中点/终点做 selection，冻结唯一 final candidate，并生成全新的 one-shot final holdout v2。旧 final holdout 永远不能重跑或复用。

---

# 常见错误

## 700秒仍没有任何日志

如果：

```text
Logs 完全为空
Output 0 B
```

通常是 Kaggle GPU/Kernel 调度未真正启动，可以停止后重新 Save & Run All。

如果已经出现训练 heartbeat，则不要因为速度慢直接停止；先保存日志和截图。

## 找到多个 Dataset manifest

说明 Add Input 添加了多余 Dataset。只保留当前步骤要求的输入。

## recovery/source identity mismatch

不要混用旧 source Dataset、旧 recovery 或其他 Notebook Output。Stage1 只用嵌入的 iter2000 recovery；selection 只用 selection Dataset + Stage1 Output。

## Stage1 超过 wall time

下载最新 `guandan-marvel-recovery.zip`，创建 Stage1 Dataset 的新版本，用这个最新 recovery 替换 Dataset 中原来的 iter2000 recovery；然后只添加这个新 Dataset，再运行同一 Stage1 Notebook。不要同时添加旧 Dataset 和上一 Output，避免发现多个不同 recovery。
