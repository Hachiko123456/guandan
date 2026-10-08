# Guandan MARVEL Migration 总体计划

## 1. 目标

建立一套以 `D:\project\doudizhu` MARVEL 架构为参考、以掼蛋规则为唯一游戏语义来源的训练系统，使其支持：

- Transformer 历史编码和结构化候选动作编码；
- 完整轨迹采样；
- MARVEL 风格 IPPO；
- MARVEL 风格 VRPO、集中式 Q critic 和 trajectory replay；
- C++/pybind 高吞吐批量环境；
- 单模型双 T4 DDP；
- 训练中 selection/champion/opponent-pool；
- 完整日志、恢复、固定发牌评估和最终 holdout。

## 2. 非目标

M01-M09 不以以下内容为首要目标：

- 重写掼蛋规则；
- 立即实现完整多副升级比赛；
- 直接比较 FableDan；
- 在 C++ 差分通过前开始长训练；
- 把 legacy MLP checkpoint 转换为 Transformer resume；
- 在没有真实 Kaggle 证据时宣布双 T4 性能。

完整比赛管理器和 FableDan 对比可在 MARVEL V2 直接策略达到发布门槛后单独规划。Yomi/Kagami 属于 M10。

## 3. 总体依赖

```text
M00 -> M01 -> M02 -> M03 -> M04 -> M05 -> M06 -> M07 -> M08 -> M09
                                                               \
                                                                -> M10（可选）
```

M03 先通过 Python adapter 验证训练语义，M04/M05 再替换环境热路径；这样可以把“算法错误”和“C++ 规则错误”分离。

## 4. 分层策略

### A01/A02：规则参考层

保留 Python 实现作为规则 oracle。C++ 实现必须证明与其等价，不能反向用 C++ 输出改写 oracle 测试。

### A03/A04：协议兼容层

保留 `GD-ACTION-0.1`、`GD-ENV-0.1` 用于 legacy checkpoint 和已有评估；新增 `GD-ACTION-0.2-MARVEL`、`GD-ENV-0.2-MARVEL`。V2 是增量协议，不得静默改变 V1 的 wire semantics。

### A05：训练层重建

当前 mean-pooled MLP 和固定 token-transition collector 保留为 legacy。新实现按完整轨迹、Transformer、结构化动作、IPPO/VRPO replay 构建。

### A06：评估层升级

保留固定 deal group 和四座位轮换，增加 selection/final holdout、champion 晋升、历史对手池和训练中定期评估。

### A07：部署层升级

从“两张卡各跑一个独立算法/worker”升级为“两张卡通过 DDP 共同训练同一个模型”。

### A08：能力层

M10 才实现 belief/search。直接策略没有稳定达到门槛前，不用搜索掩盖基础策略不足。

## 5. 迁移原则

1. Correctness before performance：先 Python adapter，再 C++。
2. Differential before cutover：C++ 只有通过差分门槛才能成为默认训练后端。
3. Complete trajectories before scale：先保证轨迹、奖励和 GAE 正确，再扩大并行数。
4. Selection separate from final：模型选择数据不得污染 final holdout。
5. One model, two GPUs：正式 T4×2 使用 DDP，不以两个独立模型占满 GPU 冒充加速。
6. Evidence over labels：`remote_full`、`complete`、`accepted` 只由真实证据决定。
7. Backward-readable, not backward-resumable：旧 checkpoint 可评估，但不能伪装成 V2 resume。

## 6. 阶段概要

### M00 控制面

输出本文档集、状态文件、报告模板和 M01-M10 Prompt，不修改功能代码。

### M01 通用核心

迁移通用 MARVEL model/runtime/training/optimizer/checkpoint/logging/DDP 代码，建立来源和许可证记录，不接入掼蛋训练。

### M02 Python adapter

把现有 Python 规则适配到 MARVEL game API，输出四玩家信息集和结构化候选动作。

### M03 训练语义

在 Python adapter 上接通 Agent、CentralCritic、complete trajectory sampler、IPPO、VRPO、replay、checkpoint 和指标；只做小规模正确性验证。

### M04 C++ runtime

实现 `nanopokerlib.guandan` 批量环境和 pybind 接口，不改变训练目标。

### M05 差分与性能

验证 Python/C++ 规则、动作、观测、终局一致性；固定工作负载至少达到 2×，目标 5×。

### M06 模型选择

实现 training/selection/final 数据隔离、champion 晋升、opponent pool 和训练中评估。

### M07 Kaggle DDP

生成完整源码包、Dataset 说明、无需手改的 Notebook、双卡 DDP、遥测和恢复。

### M08 真实 pilot

由用户在 Kaggle T4×2 运行。依据真实吞吐、显存和 GPU 利用率确定长跑参数。

### M09 正式训练

相同预算比较 IPPO/VRPO，选择主算法，长训练，周期性晋升 champion，最后只运行一次 final holdout。

### M10 belief/search

移植 Yomi/Kagami 思路，新增合法世界采样、信念模型和预算约束搜索。

## 7. 发布定义

V2 模型只有同时满足以下条件才可称为可发布候选：

- 规则和观测差分门槛通过；
- C++ 训练后端无非法动作、牌守恒错误和信息泄露；
- 完整轨迹训练和 resume 验证通过；
- selection 上稳定超过现 champion；
- 对 rule/legacy 基线达到 `06_EVALUATION_CHAMPION_SPEC.md` 的门槛；
- final holdout 未被用于调参；
- checkpoint、源码、配置和日志可复现；
- 推理资源和延迟在部署预算内。
