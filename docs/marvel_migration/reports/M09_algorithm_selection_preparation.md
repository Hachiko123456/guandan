# M09 IPPO-vs-VRPO 算法选择远端交付准备报告

- 日期：2026-10-13
- source commit：`adfbbdbfb12c8d096f228af4661f10afdee458a4`
- package id：`guandan-marvel-m09-algorithm-selection-20261013`
- 状态：`remote_ready`
- M09 accepted：**否**

## 输入冻结

IPPO：

- checkpoint SHA-256：`b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c`；
- iteration/games：100 / 400；
- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`。

VRPO：

- checkpoint SHA-256：`db8aa71f22ec88091acb189e6c57df2ac9ac0c1d18e3a931f4d7a1ec2cca9853`；
- iteration/games：100 / 400；
- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`。

两者 shared workload hash、architecture、protocol、backend identity、training source 和 selection dataset 均一致；仅算法语义及其合法专属状态不同。

## 选择流程

1. 在锁定的 65 deal clusters × 4 seat rotations 上运行 260 局 VRPO candidate vs IPPO baseline direct head-to-head。
2. 每个 deal 的四个 rotations 先聚合为独立 cluster mean。
3. 使用 bounded empirical-Bernstein 95% CI：
   - CI low > 0.5：选 VRPO；
   - CI high < 0.5：选 IPPO。
4. direct 若 inconclusive，使用现有两份对 rule 的 65×4 精确配对记录，计算 VRPO-IPPO paired per-deal difference CI：
   - low > 0：选 VRPO；
   - high < 0：选 IPPO。
5. 两项仍 inconclusive：按预案保留 M08 IPPO baseline。

预检查显示 paired rule difference 点估计为 +1.54pp（VRPO-IPPO），但保守 95% CI 为约 -50.62pp 至 +53.70pp，明显 inconclusive；因此 direct head-to-head 是主要决定依据，若其也不显著将回退 IPPO。

本流程不训练、不晋升 champion、不运行 final holdout，也不作强度声明。

## 交付

根目录：

`D:\project\kaggle_train\guandan-marvel-m09-algorithm-selection-20261013`

Dataset 目录包含 source/bootstrap/manifests、两个冻结 checkpoint 和两份 anchor JSONL。Notebook：

`guandan_marvel_m09_algorithm_selection.ipynb`

身份：

- source archive SHA-256：`b73472491de67cf47fd9fb294467f7a86eb370775ea5b2e136da8b9c5c2a835f`；
- selection manifest hash：`120b53ea104373fc0f0f26a146c005bc16e31a2dbc8992580d427d4cbd8a5811`；
- selection dataset hash：`c8d11821a0a9cf8969c304fdbdfaa371c0addbc6883cf20e9617e526b6b5b583`；
- workspace snapshot：false；
- delivery validation：通过。

## 测试

- `tests/marvel_kaggle`：53 passed；
- target regression：92 passed；
- 合计：145 passed；
- paired gate/fallback 单元测试、真实 anchor exact coverage、checkpoint/manifest binding 和 delivery build/validation 均通过。

## 下一步

用户在 Kaggle 建立一个新的私有 Dataset，上传 `dataset` 目录全部文件；导入 selection Notebook，只添加该 Dataset，T4×2，Run All。完成后返回 `guandan-marvel-algorithm-selection.zip` 和完整 console log。
