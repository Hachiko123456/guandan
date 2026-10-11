# M09 Kaggle 远端 Preflight 审查报告

- 日期：2026-10-11
- 分支：`codex/marvel-guandan-migration`
- 当前仓库 HEAD：`0dc970194470b4a096e068f11d9211ae4a19b51f`
- Kaggle 交付源码 commit：`18507bdaa8dd86b25817060d64852a7531cbd8e9`
- 远端平台：Kaggle T4×2
- 远端 run id：`20261011T064106Z-preflight-88b7e7e6`
- 阶段状态：`remote_preflight_passed`
- M09 accepted：**否**

## 收到的产物

- `C:\Users\yhx\Downloads\results (4).zip`
  - bytes：2,725,760
  - SHA-256：`189966db0d67f2ecb88cea209cb00e4360b4f83bc0554ba531a7e85b5fcb2907`
- `C:\Users\yhx\Downloads\guandan-marvel-m09-t4x2.log`
  - bytes：9,798
  - SHA-256：`ffc95898d1644bf96e479851786dc75da18e27a1e78c194d8aec414d2f22beff`

归档副本和解包检查位于：

`runs/M09_full_training/remote/preflight_20261011T064106Z/`

## 完整性检查

- 外层 ZIP CRC：通过；
- 外层 ZIP 文件数：303；
- 非安全绝对路径/`..` 路径：0；
- 内层 `guandan-marvel-output.zip` CRC：通过；
- 内层文件数：16；
- 内层文件与实际 run root 大小/hash 不一致：0；
- source manifest 声明文件：187；
- source 文件大小/hash 不一致：0。

## 交付身份

- package id：`guandan-marvel-m09-algorithm-pilot-20261011`；
- role/mode：`preflight/preflight`；
- source commit：`18507bdaa8dd86b25817060d64852a7531cbd8e9`；
- source workspace snapshot：`false`；
- source content manifest hash：`f74b0991eadb99dd68a72ad78e2ef9151de5929601bed8ee386157eb9d6a3851`；
- declared source archive SHA-256：`ca538e941f8cf671241b982a8ffae83271a04c2ab3503201f3c8b6b67644bec2`；
- M09 plan hash：`923148386be4a8e2c102b913f26570b020b4e4cbfab14197d08a8f02cd5b93d9`；
- M08 locked config SHA-256：`6994f7f80eaeb8e44c695cac8d21c6bb03561e6a8615866d47e8dcc933f67d21`。

Kaggle 将 source ZIP 自动解包，因此远端报告 `source_input_kind=directory`、`source_archive_sha256_verified=false`。这不是失败：bootstrap 使用 source content manifest 对 187 个声明文件逐项验证；本地下载后复核也得到 0 mismatch。

## 远端环境与 Preflight 门槛

- Kaggle runtime：是；
- Python：3.13.15；
- PyTorch：2.11.0+cu128；
- CUDA runtime：12.8；
- GPU：Tesla T4 ×2；
- 每卡显存：15,360 MiB；
- NCCL available：是；
- torchrun available：是；
- logical/physical CPU：4/2；
- RAM：33,658,318,848 bytes。

所有 preflight checks 均通过：source root、CUDA、双卡数量/型号、distributed、NCCL、torchrun、C++ backend 和 source/binary identity。

## Bootstrap 与 Native Build

- native build：约 34.4 秒完成，exit 0；
- native import：约 43.8 秒完成；
- runner：约 53.5 秒完成，exit 0；
- C++ extension SHA-256：`5b9a1fd71361c27cdecafe8d6612165e8cfbdddd948b47679cfb6956f6e17a83`；
- embedded/current native source hash：`2c5a26c8ca8b645f22caa40f2ec25c7bf94555d83e06d4d6fb4bc843cb275b60`；
- `cpp_source_matches_binary=true`。

日志中的重复 build/native identity 行是 Kaggle/Notebook stdout 展示重复，没有出现两次不同 binary、失败或 identity mismatch。末尾两个 `SyntaxWarning` 来自 Kaggle 的 nbconvert/mistune，不影响训练 runner，且出现在 runner exit 0 之后。

## Run 结果

- `run_result.status=complete`；
- `exit_code=0`；
- `remote_preflight_passed=true`；
- `progress.phase=preflight_complete`；
- iteration/completed games：0/0，符合 preflight 语义；
- 未创建 checkpoint；
- 未运行 evaluation；
- 未晋升 champion；
- 未作强度声明。

## 结论与下一步

本次**新 M09 package 的真实 Kaggle T4×2 preflight 通过**。这只解除“远端环境/源码/native build/import/runner 是否可运行”的 blocker，不表示 IPPO/VRPO pilot、正式训练或 M09 验收完成。

下一步只运行：

`D:\project\kaggle_train\guandan-marvel-m09-algorithm-pilot-20261011\datasets\ippo_pilot`

要求：单独建立 Kaggle Dataset，只添加 IPPO pilot role，T4×2，Run All；完成后返回完整 console log、`guandan-marvel-output.zip` 和 `guandan-marvel-recovery.zip`。VRPO 在 IPPO 产物审查后按相同预算运行。

## 仍存在的 Blockers

- IPPO 尚未完成 400 completed games；
- VRPO 尚未完成 400 completed games；
- 尚未完成共享 selection/head-to-head 算法选择；
- 主训练分布和主算法 full Dataset 尚未冻结；
- long training、champion history、final candidate 和一次性 final holdout 均未完成；
- E1/E2/E3 未知；
- M09 仍不得标记 accepted，M10 继续 blocked。
