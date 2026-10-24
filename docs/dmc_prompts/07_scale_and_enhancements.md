# DMC 阶段 7：规模化与增强（大纲 —— Gate A 通过后再细化为独立 prompt）

日期：2026-10-24 · 状态：大纲，不执行

Gate A PASS 后按以下优先级展开，每项单独成 prompt（沿用 00 的锁定表纪律）：

## 7.1 吞吐工程（决定日历长度，最优先）

- 用 Gate A 实测 games/sec 外推：追平 FableDan（≈2.5 亿样本）需要多久；
- 若 >5 个月：优先做 C++ 候选枚举导出（`nanopokerlib/cpp/guandan/` 加纯枚举函数 + pybind 导出，与 Python oracle 差分模糊测试对齐，走仓库 M04/M05 的差分验收模式）；
- 其次：决策批量推理跨环境拍平已做（阶段 4），再评估异步 actor/learner 分离是否值得（FableDan 式 1024 决策批）；
- 每项优化必须带"同 seed 前 1000 局行为等价"或"评估胜率不变"的回归证据。

## 7.2 belief 辅助头（对齐 FableDan 的核心增量）

- 基建已在 `guandan_marvel/phase5/`（belief_data/belief_collect/belief_probe，约 2500 行，当年 gate 判 REJECT 关闭）；
- 为 DMC 重新接入：oracle 三家手牌 rank 计数作监督（环境标签，不是 FableDan 蒸馏，合规）；权重从 0.05 起步做 A/B。

## 7.3 冻结冠军对手池

- `guandan_marvel/phase6/opponent_pool.py` 已实现（当前仅支持 ippo，`config.py:498`）；为 dmc 增加注册；
- 策略：50% 局面对冻结冠军（sha256 身份校验沿用），冠军动作不产生当前策略的训练样本。

## 7.4 周期性 FableDan h2h 监控

- 每 ~2500 万样本跑 256 局 h2h（复用 comparison harness）；只做监控不做选择，防过拟合对手；
- 目标轨迹参考：Gate A 后 ~15% → 5000 万样本 ~25% → 1.5 亿 ~40% → 2.5 亿+ 逼近 50%。

## 7.5 实战化（多副晋级制）

- 当前所有训练/评估都是单副、tribute=none；实战是多副晋级（2→A、进贡还贡影响发牌与级牌）；
- 需求：多副 match 模式的采样 curriculum + 评估 lane；引擎与 context_schedule 已支持贡还场景（combo run 覆盖了 tribute_modes），缺的是比赛编排器（M09 计划里"完整多副比赛管理器仍缺失"）；
- 时机：主强度进入 90%+ vs rule 之后再投入。

## 7.6 模型规模消融（可选）

- d_model 128→192/256、层数 4→6（T4 16GB 可容纳）；只在吞吐满足后做，一次只动一个变量。

## 里程碑与配额预算（30h/周 T4×2）

| 里程碑 | 累计样本 | 预计 |
|---|---:|---|
| Gate A（方向验证） | ~1000 万 | 1~2 周（含开发） |
| M1：vs rule ≥90% | ~5000 万 | +3~5 周 |
| M2：h2h FableDan ≥30% | ~1.5 亿 | +4~6 周 |
| M3：h2h 逼近/超过 50% | 2.5~4 亿 | +4~8 周，此后靠 7.2/7.3/7.5 拉开优势 |

每档预算都先看吞吐实测再冻结；M2 起每 2500 万样本存档一次全部评估原始数据。
