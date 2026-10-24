# DMC 阶段 2：完整候选枚举与动作特征编码

日期：2026-10-24 · 状态：待执行（依赖阶段 1）

## 背景事实（已核实）

- `guandan/action_state.py:383` `_candidate_sequences()` 已枚举**完整合法动作**（返回 `(canonical_tokens, CommittedAction)` 列表，含去重与歧义校验）；`guandan/environment.py:25` 的包装带合法枚举记忆化缓存。
- `guandan_marvel/comparison/fabledan_guest.py:234-270` 是现成的使用范例（枚举→分类→逐候选决策→返回 token 序列提交）。
- 训练观测布局参考：`docs/encoding_layout.md`（新动作特征布局文档必须按同等精度写）。

## 必读文件

1. `guandan/action_state.py`（`_candidate_sequences`、`canonical_tokens`、`ActionPrefix`）
2. `guandan/environment.py`（缓存包装、`enumerate_legal_actions` 调用方式）
3. `guandan_marvel/comparison/fabledan_guest.py:230-300`（枚举消费方式）
4. `docs/encoding_layout.md`（文档精度标准）

## 任务范围

新建 `guandan_marvel/dmc/candidates.py`：

1. `@dataclass(frozen=True) CandidateAction`：`tokens: tuple[int,...]`、`family: int(0=PASS,1..10)`、`comparison_rank: int`、`card_ids: tuple[int,...]`、`wild_assignments: tuple`、`is_pass: bool`。
2. `enumerate_candidates(stepwise_state) -> tuple[CandidateAction, ...]`：包装 `_candidate_sequences`，输出稳定排序（按 tokens 升序），纯函数。
3. `encode_action_features(candidate, context) -> np.ndarray[float32]`，**布局锁定为 `GD-DMC-ACTION-1`，64 维**：

| 维度 | 内容 |
|---|---|
| 0 | is_pass |
| 1..11 | family one-hot（含 PASS 位；无用位补 0） |
| 12..27 | claim/comparison rank 计数直方图（15，÷4） |
| 28 | 出牌张数 ÷27 |
| 29 | 使用百搭张数 ÷2 |
| 30 | comparison_rank 标量 ÷15 |
| 31 | 是否压过当前轮面（有轮面且更高为 1；无轮面/过牌为 0） |
| 32..46 | 实体牌 printed-rank 直方图（15，÷4） |
| 47..51 | 实体牌花色计数（含 Joker，÷27） |
| 52 | 是否含红桃级牌（百搭）÷2 |
| 53..63 | 保留 0（版本对齐位） |

   `context` 至少含：`level_rank`、`current_winning`（无则 None）。所有除法在 float32 下可精确表示。
4. 新文档 `docs/dmc_action_encoding.md`：按 `encoding_layout.md` 的精度逐维写清，标注版本号。
5. 模糊/一致性测试 `tests/dmc/test_candidates.py`：
   - 用现有发牌生成器（参考 `tests/` 内任意构造 RoundState 的测试工具）随机 ≥200 个局面；
   - **可提交一致性**：每个 `CandidateAction.tokens` 逐 token 走 `legal_token_rows` 提交路径必须成功到达 COMMIT，且提交后的公共动作与 `CommittedAction` 等价；
   - **完备性**：枚举数量 == 从空前缀做 DFS 全部可提交序列的数量（即 token 视角无遗漏）；
   - **确定性**：同一局面两次枚举逐位相等；
   - PASS 当且仅当规则允许时出现；
   - 特征向量：有限、维度 64、同候选两次编码逐位相等。

## 明确禁止

- 不改 C++（`nanopokerlib/`）、不改 `guandan/` 规则代码（只读复用）；如发现 `_candidate_sequences` 有 bug，停下来写报告，不许就地修。
- 不做任何模型/训练相关改动。

## 验收标准

```
python -m pytest tests/dmc/test_candidates.py -q     # 全绿（含 ≥200 局面模糊）
python -m pytest tests/marvel_adapter -q             # 现有协议测试不回归
```

## 交付物

`guandan_marvel/dmc/candidates.py`、`docs/dmc_action_encoding.md`、测试、`runs/dmc_stage2_<date>/REPORT.md`（含一次 10 万随机局面枚举的耗时统计：每决策平均候选数、每决策平均耗时 μs）。
