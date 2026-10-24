# 阶段 2A 定向修复 Prompt：DDP 空 Rank 的有限零损失与完整梯度连图

> 本 Prompt 不是重新实施阶段 2。当前阶段 2 已完成绝大多数实现和验证，但 Reviewer #3 因空 rank 安全缺口给出 `VERDICT: FAIL / NEXT_STAGE: DENY`。只修复本文件指定的单一 blocker；修复完成后由全新的只读 Reviewer 复验。PASS 前不得进入阶段 3。

## 一、当前冻结状态

工作区：

```text
D:/project/guandan
```

阶段 2 主规范：

```text
docs/marvel_v14_prompts/02_macro_sequence_ppo_and_gae.md
SHA256=665ec6ccf64fa2e9cdd749f0174ad6796909675124e1998c0265fa1451c2b8a2
```

本修复开始前预期 tracked diff：

```text
bytes=79869
sha256=51170dcf636da714945a3893e6fae013e7734a7f9d55217a19d3573bbb954601
```

如果开始执行时 diff 已变化，先列出变化并确认是否来自用户/其他客户端；不得覆盖或回滚。必须暂停 OpenCode 或其他写入当前工作区的 Agent，避免并发修改。

当前已通过且不得破坏：

- Sequence PPO 数学与宏 clip；
- Profile B 因果隔离；
- Profile C macro GAE、Delta、continuation和无 successor；
- macro value、EV、normalization和regularizer消融；
- legacy逐位兼容；
- checkpoint/identity/exact resume/recovery；
- 非空 rank DDP按全局样本数加权；
- A/B/C/D真实 C++ smoke；
- 阶段1 34 tests；
- phase2专项141 tests；
- 全量回归无新增失败；
- 参数量2,061,314和state_dict 78 keys。

---

# 二、唯一 blocker

原规范 §9.3 要求：

```text
全局 N > 0、某个 rank 本地 n_k = 0 时：
该 rank 必须安全产生与模型计算图相连的有限零 loss，
不能造成 collective 分叉、NaN 或 unused-parameter 错误。
```

当前实现有两个已确认问题。

## 2.1 Profile B 空 rank 的 token value NaN

当前 `guandan_marvel/training/local.py` 在 Profile B 中无条件调用：

```python
compute_value_loss(...)
```

而 `guandan_marvel/training/losses.py::compute_value_loss` 对空 active token 集合执行：

```python
F.huber_loss(empty, empty, reduction="mean")
```

结果为 `NaN`，随后 `ensure_finite_tensor("macro_agent_loss", total)` 失败，整个 DDP iteration终止。

独立复现：

```text
compute_value_loss(empty).value_loss_mean = NaN
```

## 2.2 Profile C 空 rank 的 policy路径不连图

`compute_macro_pg_loss` 在本地 `max_macros=0` 时通过新建常量零 tensor形成 local loss；该 tensor `requires_grad=False`。

当前 fallback：

```python
connected_zero_like(pg.pg_loss_mean)
```

只等于：

```python
pg.pg_loss_mean * 0
```

当 reference 本身是常量且不连模型图时，结果仍不连图。因此空 rank只有 value路径参与 backward，policy decoder/state/action/logit参数在该 rank没有 gradient，`find_unused_parameters=False` 下会触发 DDP reduction错误。

已确认：

```text
pg constant requires_grad=False
connected_zero_like(pg constant) requires_grad=False
model_output.sum()*0 requires_grad=True
```

---

# 三、允许修改范围

只允许按需修改：

```text
guandan_marvel/training/local.py
guandan_marvel/training/macro_ppo.py
tests/marvel_training/test_macro_ddp.py
tests/marvel_training/_macro_ddp_worker.py
```

如确有必要新增一个只服务于本反例的测试 helper，可新增在：

```text
tests/marvel_training/
```

禁止修改：

- `find_unused_parameters=False`；
- 阶段1 manifest/report/probe；
- 阶段1宏分段和Legacy Oracle；
- 原阶段2 Prompt；
- reward、actor fuse、AMP、context、belief、opponent pool、search；
- 非空 rank的PPO/GAE/value/regularizer数学；
- checkpoint schema和训练语义枚举；
- FableDan相关代码。

禁止使用 `git reset/clean/checkout/restore/commit/push`。

---

# 四、精确修改要求

## 4.1 定义真正连接模型输出的零锚点

在训练接线层或合适的无参数 helper 中实现语义明确的零锚点，例如：

```python
def model_connected_zero(*tensors: torch.Tensor) -> torch.Tensor:
    anchor = None
    for tensor in tensors:
        term = tensor.float().sum() * 0.0
        anchor = term if anchor is None else anchor + term
    if anchor is None:
        raise ValueError("at least one model tensor is required")
    return anchor
```

具体命名允许按项目风格调整，但必须满足：

1. 输入来自本次 `agent_model(rollout=mb)` 的真实模型输出；
2. 至少覆盖 policy路径和value路径：
   ```text
   output.avail_log_probs 或等价policy输出
   + output.player_state_value
   ```
3. 如果NTP在当前配置启用且该 rank无有效NTP样本，现有NTP零项也必须保持连图；
4. anchor数值严格为0、finite、`requires_grad=True`；
5. backward后所有本次训练应参与的参数 `grad is not None`；空 rank的梯度值全部为0；
6. 禁止从 `pg.pg_loss_mean`、Python `0.0`、新建常量tensor或已经detach的tensor构造锚点。

## 4.2 统一识别本地空宏样本

在macro trainer mini-batch中明确：

```python
local_empty = pg.macro_count_local == 0
```

全局 `macro_count_global == 0` 仍按原规范 fail-fast；本修复只处理：

```text
local_count == 0
global_count > 0
```

所有 rank必须执行相同的collective序列，不能因本地为空跳过生产函数中的all-reduce。

## 4.3 Profile B

当 `local_empty=True`：

1. 不得调用会对空 active token集合求mean的 `compute_value_loss`；
2. token value loss使用模型连接零锚点；
3. legacy token KL使用模型连接零锚点；
4. PG、regularizer、value和最终total均finite；
5. 记录：
   ```text
   empty_local_batch=1
   token_value_loss=0.0
   value_loss=0.0
   value_ev=None/undefined
   ```
6. metrics sample count保持0，不得伪造本地样本；
7. 非空 rank路径逐位不变。

不得修改通用legacy `compute_value_loss` 的空集合语义来“顺手修复”，以免改变Profile A或其他算法路径；应在macro trainer接线层安全绕过。

## 4.4 Profile C

当 `local_empty=True`：

1. macro PG、macro regularizer和macro value的现有collective语义保持；
2. 最终 `total` 必须额外包含真实模型输出构造的零锚点，保证policy和value全部参数被autograd触达；
3. 不得只依赖 `connected_zero_like(pg.pg_loss_mean)`；
4. metrics保持本地count=0、全局count=N；
5. 非空 rank数值和梯度逐位不变。

## 4.5 DDP不变量

双 rank反例：

```text
rank0: n_0 = 0
rank1: n_1 = 8
global N = 8
```

必须满足：

- 两个rank均完成forward/backward/optimizer step；
- 无NaN/Inf；
- 无unused-parameter RuntimeError；
- 两rankcollective数量和顺序一致；
- rank0所有应训练参数 `grad is not None` 且数值全0；
- DDP平均梯度等于单进程只含rank1样本的reference；
- `macro_count_global=8`；
- 聚合后的loss/metrics等于reference；
- token/macro count与权重正确；
- Profile B与Profile C均覆盖。

---

# 五、必须新增的测试

## 5.1 快速unit反例

新增会失败于当前实现的测试：

1. `model_connected_zero` 对真实policy/value模型输出：
   - value=0；
   - finite；
   - requires_grad=True；
   - backward后policy/value相关参数grad非None且为0。
2. Profile B 空local batch不调用空集合Huber mean，total/value/KL finite。
3. 用常量 `pg_loss_mean` 调用旧 `connected_zero_like`不能满足模型连图，测试应证明新实现没有依赖该错误路径。
4. 非空rank原结果逐位不变。

## 5.2 双rank slow反例

增加：

```text
test_gloo_empty_rank_is_finite_connected_and_matches_reference
```

或语义等价名称，至少覆盖：

- rank0空、rank1八宏；
- Profile B和C；
- backward/optimizer完成；
- 每个应训练参数grad非None；
- 空rank梯度全0；
- 平均梯度和loss等于单进程reference；
- count/metrics正确；
- 测试在当前修复前必须失败。

不得通过设置：

```text
find_unused_parameters=True
```

规避问题。

---

# 六、分层验证计划（避免重复耗时）

## Tier 1：每次代码修改后

只运行新反例和相关DDP文件：

```text
python -m pytest tests/marvel_training/test_macro_ddp.py -q
```

若slow被默认过滤，显式运行空rank slow用例。Tier 1失败时不得继续扩大测试范围。

## Tier 2：实现稳定后只运行一次

```text
python -m pytest \
  tests/marvel_training/test_macro_ppo.py \
  tests/marvel_training/test_macro_gae.py \
  tests/marvel_training/test_macro_training_profiles.py \
  tests/marvel_training/test_macro_ddp.py \
  tests/marvel_training/test_macro_recovery.py \
  tests/marvel_training/test_macro_legacy_compat.py \
  -q
```

再运行阶段1门禁：

```text
python -m pytest \
  tests/marvel_training/test_macro_actions.py \
  tests/marvel_training/test_macro_actions_cpp_env.py \
  tests/marvel_training/test_macro_actions_padding_fuzz.py \
  tests/marvel_training/test_marvel_v14_phase1_baseline.py \
  -q
```

## Tier 3：最终源码冻结后只运行一次

全量回归只在所有定向测试通过且不再改源码后运行一次：

```text
python -m pytest tests -q --continue-on-collection-errors
```

记录运行前后的tracked diff bytes/SHA256；二者必须相同。若全量测试后源码未变，Reviewer不得无理由再次完整跑全量；Reviewer应复核日志、diff hash并独立重跑Tier 1关键反例。

已知既有基线：

- `test_guance_declaration_handoff.py` 1 failed；
- 两个同名 `test_training_smoke.py` collection error；
- 不得新增失败。

## Smoke/recovery

本修复不改变非空rank训练数学、checkpoint schema或recovery语义，因此：

- 不需要重复全部A/B/C/D多轮smoke和完整recovery；
- 必须运行新的双rank空rank真实trainer smoke；
- 至少运行一次非空Profile B或C的1-iter真实C++ smoke，确认普通路径无退化；
- 对原A/B/C/D、recovery证据复核文件hash即可；若相关源码超出允许范围发生变化，才重新运行对应验证。

---

# 七、Reviewer复验要求

修复完成后启动全新的只读 Reviewer，不得复用OpenCode的三个Reviewer上下文。

Reviewer必须：

1. 独立复现修复前NaN和不连图根因；
2. 检查新zero anchor确实来自真实模型输出；
3. 独立运行双rank空rank反例；
4. 检查每个参数grad非None且空rank梯度为0；
5. 对比单进程reference；
6. 检查非空rank逐位不变；
7. 复跑Tier 1；
8. 复核Tier 2/3日志对应最终diff hash；
9. 检查未修改阶段1、Prompt、manifest和阶段3内容。

Reviewer不要求在源码未变化时第二次运行约11分钟全量suite；但必须确认最终全量日志真实存在、命令正确、运行后diff hash未变化。

输出：

```text
VERDICT: PASS / FAIL / INCONCLUSIVE
NEXT_STAGE: ALLOW / DENY
```

FAIL时仍须给出完整修复Prompt。

---

# 八、硬验收标准

1. Profile B空rank不再产生NaN。
2. Profile C空rank不再产生unused-parameter错误。
3. 空rank所有应训练参数grad非None且为0。
4. 双rank梯度/loss/metrics等于单进程reference。
5. global N=0仍fail-fast。
6. 非空rank数学、日志和梯度逐位不变。
7. `find_unused_parameters`保持False。
8. test_macro_ddp全部通过且新增空rankslow反例。
9. phase2专项无新增失败。
10. 阶段1 34 tests继续通过。
11. 全量回归只有既有失败/collection error。
12. Prompt、manifest、macro_actions、legacy_gae哈希不变。
13. 未实施阶段3及以后内容。

---

# 九、Implementer报告格式

必须包含：

1. `IMPLEMENTATION_STATUS`；
2. 开工diff hash和并发写入检查；
3. 根因和代码映射；
4. 修改文件；
5. zero anchor覆盖哪些模型路径；
6. Profile B/C空rank处理；
7. 新反例为何失败于旧实现；
8. Tier 1/2/3实际命令、耗时和结果；
9. 双rank梯度与reference数字；
10. 非空rank逐位对账；
11. 最终diff bytes/SHA256；
12. 冻结文件hash；
13. 未运行项；
14. residual risks；
15. 明确声明未实施阶段3。

完成后停止，等待全新Reviewer复验。