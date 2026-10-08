# 风险登记与架构决策

## 1. 已确认原则

### D001 保留 Python 规则 oracle

决定：A01/A02 Python 实现不因 C++ 迁移而删除。

理由：需要独立差分参考、回放和调试。

### D002 V2 增量协议

决定：新增 MARVEL V2 action/env wire，不原地改变 V1。

理由：保护 legacy checkpoint 和已有评估证据。

### D003 双 GPU 训练同一模型

决定：正式 T4×2 使用 DDP。

理由：目标是提高单模型训练量，而不是并行占卡。

### D004 先直接策略，后 belief/search

决定：M10 在 M09 后。

理由：避免搜索掩盖基础模型和训练问题。

## 2. M01 必须决策

### D005 Python 包命名

候选：`marvel` 或独立命名空间。需评估本地安装冲突、Kaggle 打包、迁移 diff 和维护成本。

### D006 通用代码复制方式

候选：vendor、独立子包、共享本地依赖。不得依赖 `D:\project\doudizhu` 绝对路径运行。

## 3. M02/M04 必须决策

### D007 结构化动作 row 的最大长度和词表

必须通过合法动作全集扫描确定，不可猜测上界。

### D008 C++ 物理牌表示

需兼容双副 108 张物理牌、相同牌面不同 card_id、逢人配声明和确定性 canonicalization。

### D009 发牌和 chance step

决定 reset 内发牌或显式 chance node；必须与 sampler 统计、replay 和搜索需求一致。

## 4. M03/M08 必须决策

### D010 Muon

默认只作为可选实验；先以 AdamW 建立可复现基线，再决定是否启用。

### D011 torch.compile

必须测首次编译时间、图重编译和总吞吐；若动态 shape 导致退化可关闭。

### D012 AMP/TF32

T4 支持 fp16/TF32 相关路径，但正确性、稳定性和速度必须实测。不得牺牲非有限值检查。

### D013 Replay 设备

VRPO replay 放 GPU、CPU pinned memory 或分级缓存，由显存和吞吐决定。

### D014 模型尺寸

128/256、4-layer 是基线，不是已证明最优。pilot 后才能调整。

## 5. 主要风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| C++ 与 Python 规则漂移 | 训练错误且难发现 | M05 大规模差分、Python oracle 不删除 |
| 动作容量低估 | 合法动作被截断 | 扫描上界、overflow hard fail |
| 四玩家观测泄露 | 评估虚高 | 重排隐私测试、actor/critic tensor 分离 |
| 完整轨迹显存过高 | OOM | compact channels、分批 learner、pilot 定型 |
| CPU 仍供不上 GPU | 利用率低 | C++ batch、profile rollout/learner 分段 |
| DDP 通信收益不足 | 双卡变慢 | 相同全局工作量短测、>=1.5× gate |
| selection 过拟合 | champion 虚高 | final holdout 隔离、版本化 deals |
| 旧 checkpoint 混用 | 错误 resume | schema/contract hash、evaluation_only 分类 |
| 文档状态再次过期 | 错误推进 | 单一 STATUS、每阶段提交时更新 |
| 盲目照抄斗地主默认值 | 资源/算法不适配 | M08 实测、ADR 记录 |

## 6. ADR 格式

后续决策追加：

```text
### Dxxx 标题
日期：YYYY-MM-DD
状态：proposed/accepted/superseded
候选方案：...
决定：...
证据：...
影响：...
回滚方式：...
```
