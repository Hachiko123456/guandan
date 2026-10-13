# M09 r2 Kaggle 远端 Preflight 审查报告

- 日期：2026-10-13
- 分支：`codex/marvel-guandan-migration`
- r2 source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`
- package id：`guandan-marvel-m09-algorithm-pilot-20261011-r2`
- Kaggle run id：`20261013T050124Z-preflight-cc1d459b`
- 运行环境：Kaggle Tesla T4×2
- 状态：`passed`
- M09 accepted：**否**

## 收到的产物

- `results (5).zip`
  - 2,727,047 bytes；
  - SHA-256：`1b52f1662ed8eca007584a2c70e28b1dc79d08ba6312147c5b68f01fa07393e1`。
- `guandan-marvel-ddp-t4x2.log`
  - 9,734 bytes；
  - SHA-256：`7079b371a5ade19e4dd61e7b4052c4c3b8ee80a2b854da5a6bfd817bb00d982c`。

归档位于：

`runs/M09_full_training/remote/r2_preflight_20261013T050124Z/`

## 完整性与身份

- 外层 ZIP CRC：通过；
- 外层文件数：303；
- 不安全路径：0；
- 内层 output ZIP CRC：通过；
- 内层 16 个文件与 run root 逐项一致；
- source manifest 声明文件：187；
- source 文件 mismatch：0；
- `workspace_snapshot=false`。

远端身份：

- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`；
- source archive SHA-256：`d08b8743970b8cce38a5ffe7768a3169b117a76ba2c03c596acafadd7e7cd465`；
- source content manifest hash：`fd8853e379b3bb2a6eefc00474bda9c200aa49139bcbea841a4ffa7c105e6fca`；
- plan hash：`5ca94bd3d930b86f99e5bdc329e0405fbaa0023ef64ff47cd5564b457b84f8f7`；
- M08 locked config SHA-256：`6994f7f80eaeb8e44c695cac8d21c6bb03561e6a8615866d47e8dcc933f67d21`。

以上均与本地 r2 delivery 完全一致。

## 环境与 Runner

- Python：3.13.15；
- PyTorch：2.11.0+cu128；
- CUDA：12.8；
- Tesla T4：2 张，每张 15,360 MiB；
- NCCL：可用；
- torchrun：可用；
- native build：约 30.6 秒，exit 0；
- native import：通过；
- C++ extension SHA-256：`5b9a1fd71361c27cdecafe8d6612165e8cfbdddd948b47679cfb6956f6e17a83`；
- `source_matches_loaded_binary=true`；
- runner：exit 0；
- `remote_preflight_passed=true`；
- `progress.phase=preflight_complete`；
- fatal error scan：0。

日志末尾的 nbconvert/mistune `SyntaxWarning` 是 Kaggle 生成 Notebook/HTML 时的非致命警告，发生在 runner 成功结束后。

## 结论

r2 新交付的真实 Kaggle T4×2 preflight 已通过，NCCL evaluation rendezvous 补修现在可以进入真实 IPPO pilot 验证。

本次没有训练，iteration/completed games 为 0/0，符合 preflight 语义。它不构成模型强度或 M09 accepted 结论。

## 下一步

运行 r2 `ippo_pilot`：

`D:\project\kaggle_train\guandan-marvel-m09-algorithm-pilot-20261011-r2\datasets\ippo_pilot`

要求：建立独立 Kaggle Dataset，只添加该 role；继续使用 r2 Notebook，T4×2，Run All。完成后返回完整 console log、`guandan-marvel-output.zip` 和 `guandan-marvel-recovery.zip`。
