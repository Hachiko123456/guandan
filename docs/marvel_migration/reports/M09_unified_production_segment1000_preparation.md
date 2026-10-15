# M09 Unified Production Segment1000 交付准备

- 日期：2026-10-15
- 状态：`remote_ready`
- Remote A/B/C：全部 accepted
- production_enabled：true
- 授权范围：仅 Segment1000
- auto-continuation：false
- final holdout：false
- M09 accepted：否

## 交付

根目录：

`D:\project\kaggle_train\guandan-marvel-m09-unified-production-segment1000-v1`

上传 ZIP：

`D:\project\kaggle_train\guandan-marvel-m09-unified-production-segment1000-v1\guandan-marvel-m09-unified-production-segment1000-upload.zip`

- bytes：69,294,081；
- SHA-256：`bcc0945b5763a0ad5d4b18ce655135ae948dfe55d59ec5f32bde2c3470a8802a`。

Notebook：

`D:\project\kaggle_train\guandan-marvel-m09-unified-production-segment1000-v1\guandan_marvel_m09_unified_pipeline.ipynb`

- SHA-256：`9f3bd254e8213468e0bf1a3a292fbfb693af1ab93249c02732d671e92ea2fc68`。

## Production control

- control manifest hash：`1e30157bae9ba26bf75857d932cd54597187c4aa6931aaf81812d9d5bbf77ec2`；
- delivery manifest hash：`278b05aa02af932c9163bc2198bab4fc0dc4b7773be88b8f0ecfbeaba5d6fd05`；
- source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`；
- initial recovery：iter500 / 2000 games；
- target：iter1000 / 4000 games；
- DDP world size：2；
- formal workers：2；
- deterministic sharding：task index modulo 2；
- dual formal Remote C validated：true；
- auto-continuation：false；
- final holdout：false。

## 合并后的用户流程

生产运行已合并为一次 Run All：

1. 从 iter500 recovery 运行 T4×2 DDP 到 iter1000；
2. durable training commit；
3. DDP 完全退出；
4. 两个独立 T4 formal workers 并行运行 640-task selection；
5. deterministic merge；
6. promotion/champion transaction；
7. continuation packaging；
8. 清理临时 runtime 大目录。

不再重复 Remote A/B/C，也不按 Segment 创建新 Dataset/Notebook。Segment1000 完成后只需下载一个 continuation ZIP 和日志进行 review；通过后使用同一 production Dataset + 上一 Output 运行下一段。自动进入 Segment1500 仍关闭。

## 验证

生产 upload ZIP 已执行打包后 artifact preflight：control files、模拟 Kaggle auto-unpack、source/recovery discovery、acceptance profile、runtime Dataset、packaged provenance 和真实 C++ deal hash 全部通过。

当前仅授权 Segment1000。未运行 final holdout，M09 仍 incomplete。
