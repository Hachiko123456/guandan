# DMC 阶段 07B：观测窗口 head+tail 与梯度累积（Gate L 失败修复之二）

日期：2026-10-25 · 状态：待执行（依赖阶段 4 交付；与 07A 无代码冲突可并行开发，合并后一起重跑 Gate L）

## 依据（Gate L 实测）

- **884,337 次观测截断 / 1,043,104 个决策 ≈ 85% 决策点历史被尾部截断**——开局历史（含已解析贡还记录）对中后期决策全部丢失。FableDan 用 MAX_SEQ=512 的 head+tail 截断保留开局；
- qz_r2 中段回踩（cycle 25-29 跌至 0.07-0.10）与有效 batch=256 相关（锁定表 1024 因显存未达成）。

## 改动一：观测窗口 head+tail（版本化语义变更）

1. `dmc/sampler.py` 的流窗口改为：**保留头 128 token + 尾 896 token**（总 1024 不变，显存零影响）；中段丢弃计数独立上报（`obs_head_drops` / `obs_tail_drops`，替代单一 truncation 计数的模糊性）。
2. 语义版本升级：`SAMPLER_STATE_SCHEMA` 后缀 bump（如 `-v2`），旧恢复包拒绝加载并提示新身份；`GD-DMC-ACTION-1` 不变（动作特征不涉及）。
3. 设计锚点：BOS/级牌头部、首个出牌事件、贡还记录集中在头部；FableDan 同款策略已验证有效。
4. 测试：窗口组成单元测试（头保留、尾保留、中段丢弃、两个计数器）、确定性、与 07A 单元兼容。

## 改动二：梯度累积到有效 batch 1024

1. `dmc/learner.py` 增加 `accumulation_steps`（默认 4，即 4×256=1024，兑现锁定表 batch=1024）；
2. 正确配方：每个 micro-batch `scaler.scale(loss).backward()`；累积满后 `scaler.unscale_(optimizer)` → `clip_grad_norm_(1.0)` → `scaler.step` → `update`；溢出跳步语义保持（任一 micro-batch 溢出则整个 optimizer step 跳过并计数）；
3. metrics 按 optimizer step 口径不变（qz_r2 用 step 末的批次聚合）。

## 明确禁止

- 不动 ε 计划/梯度比/优化器/奖励语义；
- 不趁机调超参（窗口切分 128/896 与累积 4×256 是本阶段仅有的两个数字决策，写进 resolved config 与偏差声明）。

## 验收标准

1. `pytest tests/dmc -q` 全绿 + 新增：累积梯度与单次 1024 批等价性测试（FP32 下数值容差内一致）、溢出跳步语义测试、窗口组成测试；
2. 本机 200 局冒烟（同 stage-4 配置）：illegal/truncated/nonfinite=0、确定性保持；
3. 合成收敛测试仍过（R²>0.9）；
4. REPORT：两项改动各自的新旧语义说明 + 偏差声明（对照 00 锁定表）。

## 交付物

代码 + 测试 + `runs/dmc_stage07b_<date>/REPORT.md`。完成后与 07A 合并构建新 Gate L 交付（新 plan 身份，门槛数字不变）。
