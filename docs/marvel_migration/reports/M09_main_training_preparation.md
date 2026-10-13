# M09 IPPO 分段主训练交付准备报告

- 日期：2026-10-13
- 主算法：IPPO
- 选择 decision hash：`f12421fd57b85fe3067974953382f59719da2fda4edbd2b35a2faa7dbda791bf`
- main plan hash：`96f324f43ab391c3912a9284fd1dc862424b4c19d63ae9b375a0c15627095f88`
- 远端训练 source commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`
- 本地交付构建 commit：`a6416784a716be1c134e88993ddf9ad2a35c3799`
- 状态：`remote_ready_segment0500_only`
- M09 accepted：**否**

## Training distribution 决策

M09 主训练保留 M08/Pilot 的精确训练分布：

- continuous per-slot PRNG deals；
- `Rank.TWO`；
- `previous_result=None`；
- generator `guandan-marvel-env-batch-continuous-prng-v1`。

原因：这能在同一 source/backend/semantic contract 下从已验证的 r2 IPPO iter100 recovery 做 exact resume。当前不静默引入尚未经过差分、serialization/resume 和远端性能验证的动态多级牌/贡还 native scheduler。

限制：主训练不直接采样全部 13 级牌和贡还上下文。因此 formal selection/final 继续使用广覆盖数据。如果对 rule/legacy 的正式门槛未通过，M09 必须保持 incomplete，并建立单独版本化 scheduler migration，不能通过改写结论接受。

## 分段预算

累计目标：

| Segment | 目标 iteration | 累计 completed games |
|---|---:|---:|
| segment0500 | 500 | 2,000 |
| segment1000 | 1,000 | 4,000 |
| segment1500 | 1,500 | 6,000 |
| segment2000 | 2,000 | 8,000 |

起点为已验证的 IPPO iter100/400 games recovery。每段是绝对 target，不重复计算旧 iteration。每段 wall safety budget 为 15,000 秒，checkpoint interval 25，保留最近 2 个 durable checkpoints。

## Evaluation/Champion 计划

长评估不再嵌入 DDP 训练进程：

1. 训练到 500 iteration 边界；
2. 下载并验证 output/recovery；
3. 独立运行 formal 32-cluster × 4 rotations × 固定矩阵 selection/champion evaluation；
4. promotion policy：champion CI low ≥0.52、至少 32 clusters、rule ≥0.40、random ≥0.70、health=0；
5. 原子更新 champion generation 和 opponent pool；
6. 只有评估/晋升审查完成后才授权下一训练段。

固定矩阵规划：current champion、rule、random、legacy V1 anchor、opponent-pool sample。初始 champion 是 r2 IPPO iter100 checkpoint。Champion checkpoint 不覆盖，保留 generation history。

分离训练与评估避免单 rank 长评估占用 DDP/NCCL 资源，也便于每 500 iteration 强制人工证据审查。

## 交付

根目录：

`D:\project\kaggle_train\guandan-marvel-m09-main-ippo-segmented-20261013`

包含四个互斥 Dataset role 和一个共用 Notebook。`segment0500` 内嵌已验证的 r2 IPPO recovery；后续段不内嵌 recovery，必须添加上一段下载的 recovery 作为第二个 Kaggle Input。

Source archive 继续使用 r2 精确身份：

- SHA-256：`d08b8743970b8cce38a5ffe7768a3169b117a76ba2c03c596acafadd7e7cd465`；
- commit：`e5c82e7c27fb366800e9722c9e8e4720df7db01f`；
- initial recovery SHA-256：`8b05b05f0d2f3715f2fb9701cbc4450a43d156a7cf2387eb916c2e9958c8b067`。

Dataset/recovery auto-discovery 已本地验证：segment0500 识别 source ZIP、iter100 complete recovery、IPPO/world size 2。

## 测试

- Kaggle scoped：55 passed；
- target regression：92 passed；
- 合计：147 passed；
- 四段 profile、绝对 iteration/game budgets、首段 embedded recovery、后续无旧 recovery、Notebook 语法和 delivery validation 均通过。

## 边界与下一步

当前只授权运行 `segment0500`。不得预先运行 segment1000/1500/2000，不得跳过每段后的 formal evaluation/champion 步骤，不得运行 final holdout。

收到真实 segment0500 日志、output 和 recovery 前，不得宣称 main training 完成、champion 晋升完成、模型达到 E1/E2/E3 或 M09 accepted。
