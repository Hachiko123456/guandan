# DMC 阶段 3：Q 头、回放缓冲与学习器

日期：2026-10-24 · 状态：待执行（依赖阶段 1、2）

## 背景事实（已核实）

- `guandan_marvel/model/agent.py:24-31`：`Agent(BaseMarvelModule)`，`logit_head`（`agent.py:170` 对每个合法行输出标量再 softmax）；`value_head` 可选。主干 = Transformer 历史编码 + 结构化状态通道 + 候选-状态融合（ActorFuse，`model/base.py:116`）。
- `guandan_marvel/buffer.py` 是 PPO 的整轨迹存储；DMC 需要新的**扁平**缓冲，互不替代。
- 参考实现（只读，不复制依赖）：`d:/project/fabledan/fabledan/train_fast.py:390-427`（MSE 学习循环）、`ring.py`（样本结构 z=rewards/3）。

## 必读文件

1. `guandan_marvel/model/agent.py`、`guandan_marvel/model/base.py`（ActorFuse、打分路径）
2. `guandan_marvel/buffer.py`（内存预算风格、预分配风格）
3. `docs/dmc_prompts/00_DMC_MIGRATION_PLAN.md` 锁定表
4. `guandan_marvel/training/policy_import.py`（版本化迁移入口，DMC 需要拒绝隐式加载 ippo 权重）

## 任务范围（全部新文件，位于 `guandan_marvel/dmc/`）

1. `q_agent.py`：`DMCAgent`，包装现有主干：
   - `q_values(obs_tokens[B,T], obs_channels[B,256], action_feats[B,K,64], masks[B,K]) -> Tensor[B,K]`：一次前向对每个候选输出标量 Q（复用候选打分融合路径；批内 K 取最大候选数，短位用 mask 屏蔽）；
   - fresh init；`load_state_dict` 只接受 schema=="dmc-v1" 的 checkpoint；发现 ippo/vrpo 权重文件立即报错并提示走 `policy_import` 显式迁移；
   - NTP 辅助头沿用现有实现（可配置权重，默认 0）。
2. `replay.py`：`DMCReplay`：
   - 预分配定长环形缓冲，容量 = 阶段 1 配置；
   - 样本字段：`obs_tokens`（变长打包：int32 数组 + lengths）、`obs_channels(256)`、`action_feats(64)`（仅所选动作）、`z(float32)`、`seat(int8)`、`game_id(int64)`；
   - `push_batch(samples)`、`sample(batch_size, rng)` 均匀无放回；FIFO 淘汰语义；`__len__`、满员计数、seed 可复现；
   - 内存预算函数（参考 `buffer.py` 的 `trajectory_memory_budget` 风格），写进 REPORT。
3. `learner.py`：`DMCLearner`：
   - `train_step()`: 从 replay 抽 batch → 前向 Q(s, a_chosen) → `MSE(Q, z)`（+可选 NTP 项）→ AdamW(lr=1e-4, wd=0) → clip_grad_norm 1.0 → fp16 autocast + GradScaler（CPU/Gloo 路径自动 fp32）；
   - **无** target network、无 GAE、无 ratio/clip、无 KL 正则；
   - 指标（每步输出，jsonl 可写）：`loss`、`q_mean/q_std`、`z_mean/z_std`、`qz_r2`（Q 对 z 的解释方差）、`grad_norm`、`scaler_scale`、`nonfinite_count`（>0 即 fail-closed 抛错）。
4. 单测 `tests/dmc/test_q_learner.py`：
   - **合成收敛测试（核心）**：构造 z = 线性函数(某 64 维动作特征子集 + 某状态通道子集) 的数据集灌入 replay，训练 ≥2000 步后 `qz_r2 > 0.9`——证明梯度确实到达主干与动作特征通路；
   - replay：FIFO 淘汰、均匀采样分布（卡方或计数近似）、seed 复现两次采样序列相等；
   - 拒绝加载 ippo checkpoint（构造一个假 schema 文件断言报错）；
   - fp16：构造溢出场景断言 GradScaler 跳步而非 NaN 传播。

## 明确禁止

- 修改 `model/agent.py`、`model/base.py`、`buffer.py`、ippo/vrpo 训练代码。若主干接口不足以支持批式候选打分，允许**加法式**新增方法（新类包装优先），并在 REPORT 说明。
- 实现任何采样/环境交互逻辑（阶段 4 的事）。

## 验收标准

```
python -m pytest tests/dmc -q                  # 全绿
python -m pytest tests/unit/test_dmc_config.py -q
python -m pytest tests/marvel_adapter tests/marvel_cpp -q   # 主干未回归
```

## 交付物

三个模块 + 测试 + `runs/dmc_stage3_<date>/REPORT.md`（含合成收敛曲线数据、replay 内存预算计算）。
