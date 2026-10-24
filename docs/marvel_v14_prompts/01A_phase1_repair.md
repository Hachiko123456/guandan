# 阶段 1 修复 Prompt：Legacy GAE Oracle、宏分段鲁棒性与基线清单修复

> 本 Prompt 是对阶段 1 的定向修复，不是阶段 2。完整复制给 DeepSeek 执行。修复完成后停止，等待 Codex 独立复验；不得执行 `02_macro_sequence_ppo_and_gae.md`。

## 任务

修复工作区 `D:/project/guandan` 中阶段 1 验收发现的四类问题：

1. `gamma < 1` 时 legacy token-delta Oracle 折扣方向错误；
2. inactive/padding位置的垃圾 `action_taken` 会在 mask 生效前导致 gather 越界；
3. padding macro槽被错误标记为 `macro_is_last_for_player=True`；
4. baseline manifest 把自身、报告或 probe 生成物作为输入快照记录，导致声明哈希在写回后立即失效。

本次只能修复阶段 1，不得实施 sequence PPO、macro-decision GAE、actor fuse、AMP、奖励、全分布、belief、opponent pool 或 search。

## 工作区安全约束

- 工作区存在未提交修改和未跟踪文件。
- 不得使用 `git reset`、`git clean`、`git checkout` 覆盖文件。
- 不得删除、回滚或覆盖与本修复无关的修改。
- 不得提交 git commit，除非用户另行明确要求。
- 必须保留阶段 1 已正确实现的宏分段、真实 C++ 对账、probe 和 strict backend 检查。
- 不得通过放宽测试容差、删除反例或限制 `gamma=1` 来规避问题。

---

## 一、修复 Legacy Token Delta Oracle

### 1. 当前错误

文件：
`guandan_marvel/training/legacy_gae.py`

当前 forward window 采用近似：

`window = gamma * window + reward_t`

随后又乘 gap。这使从玩家当前行动 (t) 看未来奖励的折扣方向错误。

生产 token GAE 对同一玩家当前行动 (t) 和下一行动 (t_{next}) 的正确 delta 是：

[
delta_t^p
=
sum_{j=t}^{t_{next}-1}gamma^{j-t}r_j^p
+
gamma^{t_{next}-t}V^p(s_{t_{next}})
-
V^p(s_t)
]

若该玩家没有下一行动，则：

[
delta_t^p
=
sum_{j=t}^{n-1}gamma^{j-t}r_j^p
-
V^p(s_t)
]

其中 (n=	ext{num_steps}[b])，不得使用批次统一 horizon (T) 代替该轨迹实际长度。

### 2. 要求的实现方式

该模块是诊断 Oracle，不是高性能生产路径。优先采用最清晰、独立、易手算验证的 reference 实现：

- 对每个 batch；
- 对每个 player；
- 找到该玩家在 ([0,n)) 内的真实行动 token 索引；
- 对相邻行动窗口显式计算折扣 reward sum；
- 下一行动存在时使用同一玩家的 next value bootstrap；
- 最后行动窗口只累计到该轨迹 `num_steps`，bootstrap=0；
- 非行动位置 delta=0；
- padding完全忽略。

允许使用 Python 循环，因为这是测试 Oracle。不得复制生产 backward GAE 代码形成“同实现自证”。

参考伪代码：

```python
for b in range(B):
    n = int(num_steps[b])
    for p in range(P):
        times = [
            t for t in range(n)
            if int(player_ids[b, t]) == p
        ]
        for i, start in enumerate(times):
            has_next = i + 1 < len(times)
            stop = times[i + 1] if has_next else n

            reward_sum = 0
            for j in range(start, stop):
                reward_sum += (
                    gamma ** (j - start)
                    * rewards[b, j, p]
                )

            bootstrap = 0
            if has_next:
                bootstrap = (
                    gamma ** (stop - start)
                    * state_values[b, stop, p]
                )

            delta[b, start] = (
                reward_sum
                + bootstrap
                - state_values[b, start, p]
            )
```

必须保持 tensor device/dtype；不要把训练数据整体转换成 Python float。

### 3. gamma 合法范围

Oracle必须支持：

[
0legammale1
]

包括 `gamma=0`。

在 `gamma=0` 时采用数学约定：

[
gamma^0=1
]

当前行动同一步 reward 保留，之后奖励和bootstrap权重为0。

不得继续以“生产gamma=1”为理由拒绝gamma=0。

### 4. 独立反例测试

必须新增下列精确测试，不能只依赖随机值：

四玩家合法宏长1序列：

```text
t=0 player0 commit
t=1 player1 commit
t=2 player2 commit
t=3 player3 commit
t=4 player0 commit
```

value全部为0。

分别让 player0 reward=1 出现在：

- t=0：期望 delta0=1；
- t=1：期望 delta0=gamma；
- t=2：期望 delta0=gamma^2；
- t=3：期望 delta0=gamma^3；
- t=4：该reward属于player0下一决策窗口，不能进入t=0 delta。

至少测试：

- gamma=1.0；
- gamma=0.97；
- gamma=0.5；
- gamma=0.0。

Codex验收时已复现的失败值包括：

```text
gamma=0.97, reward at t=1:
production=0.97
old oracle=0.8587340257

gamma=0.5, reward at t=1:
production=0.5
old oracle=0.03125
```

修复后这些必须与生产实现匹配到：

- float64 max_abs_diff < 1e-10；
- float32 max_abs_diff < 1e-6。

### 5. 额外 Oracle 测试

必须新增：

1. `num_steps < T` 且 padding rewards/value为垃圾值；padding不得影响结果。
2. 一个 batch 中不同轨迹具有不同 `num_steps`。
3. 中间reward属于非当前行动玩家。
4. 下一同玩家行动前包含chance步骤。
5. 宏内部包含chance/environment token。
6. 随机合成至少100组，比较：
   - 生产 `compute_gae_from_value` 在macro_start处的advantage；
   - 独立delta＋legacy展开结果。
7. gamma/lambda组合至少：
   - gamma = 0, 0.5, 0.97, 1.0；
   - lambda = 0, 0.5, 0.95, 1.0。

注意：legacy宏起点展开仍是：

[
A_m^{old}
=
sum_{iin macro_m}
(gammalambda)^{t_i-start_m}delta_{t_i}
+
(gammalambda)^{	au_m}A_{m+1}^{old}
]

宏长大于1时，不得把宏内多个token delta错误替换成普通SMDP单delta。

---

## 二、修复 inactive/padding action index gather

文件：
`guandan_marvel/training/macro_actions.py`

### 当前问题

当前只验证 active位置的 `action_taken`，但随后 gather 使用所有位置。inactive/padding/chance位置即使按语义应被忽略，若含垃圾索引仍会越界。

Codex已复现：

```text
num_steps=1
padding player_id=-2
padding action_taken=999
→ RuntimeError index 999 out of bounds
```

### 修复要求

在任何 gather 之前：

```python
safe_action = torch.where(
    active,
    action_taken,
    torch.zeros_like(action_taken),
)
```

然后：

- 只验证 active位置的范围；
- inactive位置始终使用安全索引0；
- `avail_is_commit` gather使用safe_action；
- `avail_tokens` selected-row gather也使用safe_action；
- inactive位置最终chosen flag/kind仍应被mask掉。

先验证维度，再读取 `shape[-1]`；对非法输入给出明确ValueError，不得暴露无说明的底层gather RuntimeError。

### 测试

新增：

1. padding `action_taken=999` 被忽略且分段正确；
2. chance位置负索引/超大索引被忽略；
3. active位置超范围仍必须拒绝；
4. active位置负索引仍必须拒绝；
5. 同时覆盖有/无 `avail_tokens` 路径。

---

## 三、修复 padding macro 的 last-for-player mask

当前错误：

```python
macro_is_last_for_player = macro_next_index < 0
```

padding macro槽的 `start=-1,next=-1`，因此也被标记为True。

正确：

```python
valid_macro = macro_start_index >= 0
macro_is_last_for_player = (
    valid_macro
    & (macro_next_index < 0)
)
```

### 测试

构造batch：

- trajectory A：1个宏；
- trajectory B：多个宏。

要求：

- A的真实最后宏=True；
- A的所有padding macro=False；
- B每个玩家仅真实最后宏=True；
- padding所有相关布尔字段语义明确。

同步审计：

- `macro_is_terminal`；
- `macro_is_last_for_player`；
- future Stage2可能使用的其他macro mask；

保证padding不能被解释为真实宏。

---

## 四、增加长宏动作测试

不要把 `max_action_tokens=8` 误解为完整宏动作最长8。

真实C++阶段1 probe已经观察到：

- macro length max=9；
- macro length max=10。

`max_action_tokens=8` 是候选 action row 的固定编码宽度，不是完整宏序列上限。

新增合成分段测试：

- macro length 1；
- 8；
- 10；
- 16；
- 若协议允许，接近实际prefix上限的长度。

分段模块不得硬编码 `macro_length<=8`。

更新阶段1报告中任何“宏最长8”的错误或含混表述。

---

## 五、修复 baseline manifest 自引用哈希

文件：
`scripts/build_marvel_v14_phase1_baseline.py`

### 当前问题

`collect_git()` 在写最终manifest/report前扫描未跟踪源码。如果旧manifest/report已存在，会记录其旧hash，随后本次构建覆盖这些文件，导致清单声明立即失效。

Codex实际核验：

```text
manifest内声明自身 sha256=d28e9ae9...
当前实际                =1392b8a4...

manifest内声明report    =4e6098ee...
当前实际                =fcbbe984...
```

### 修复要求

冻结输入源码清单必须排除本次构建生成物，至少：

- 当前 `--manifest` 路径；
- 当前 `--report` 路径；
- 当前 `--probe-dir` 下所有文件；
- `baseline_build_result.json` 等本阶段生成结果。

不要尝试把manifest的完整文件sha256写进manifest自身。

建议将清单明确拆为：

- `snapshot_inputs`：真正冻结的源码/配置输入，可复算；
- `generated_outputs`：输出路径、生成后sha256在外部build result中记录；
- manifest内部canonical `hash`：基于去掉hash字段后的manifest body计算。

### 可重复构建

建议增加可选：

`--created-at <ISO8601>`

测试时使用同一created_at、相同输入构建到两个独立临时目录，要求：

- manifest bytes一致；
- report bytes一致；
- probe input hash一致；
- timestamp-free model output content hash一致；
- macro probe hash一致；
- snapshot_inputs清单一致；
- 所有声明 `path + sha256` 均可重算。

若保留动态created_at，则manifest文件hash允许变化，但snapshot_inputs和timestamp-free content identity必须稳定；测试中仍应提供固定created_at确定性模式。

### 哈希验证器

新增测试或脚本递归验证manifest中所有声明的输入文件：

- 路径存在；
- 字节数匹配；
- sha256匹配。

不得忽略失效条目。

---

## 六、不得破坏已通过部分

必须继续通过：

1. 真实C++多局 commit count对账；
2. PASS/COMMIT真实action row解码；
3. tau为同玩家宏起点全局token距离；
4. chance/padding不属于policy macro；
5. 60步四人全commit测试确认：
   - legacy使用 ((gammalambda)^	au)；
   - (gamma^	aulambda) 不与旧实现等价；
6. tracked diff hash不因本修复修改原有tracked训练文件；
7. 正式M09 strict loader继续因backend identity不匹配而拒绝；
8. diagnostic-only probe明确标记；
9. 参数量保持2,061,314；
10. 训练语义仍未改变；
11. 新模块仍未接入实际trainer/loss。

---

## 七、必须运行的验证

### 1. 阶段专项测试

```powershell
python -m pytest   tests/marvel_training/test_macro_actions.py   tests/marvel_training/test_macro_actions_cpp_env.py   -q
```

### 2. 新反例测试

必须在正式pytest测试中体现，不能只运行一次临时命令。报告每个gamma/reward位置的期望和实际值。

### 3. 随机/性质测试

至少100组合成轨迹，对生产token GAE与独立legacy Oracle比较。

### 4. Baseline builder重复构建

固定created_at、两个临时输出目录，证明确定性和所有声明hash可复算。

### 5. 全量回归

运行与阶段1报告相同的全量测试命令，并与开工前基线比较。既有collection error/既有失败可单列，但不得新增失败。

### 6. Git范围验证

报告：

- `git status --short`；
- `git diff --stat`；
- tracked diff bytes/hash；
- 本修复实际修改的阶段1文件；
- 确认未修改阶段2以后代码。

---

## 八、修复后的硬验收标准

1. 所有阶段1专项测试通过。
2. gamma=0/0.5/0.97/1.0下，独立Oracle与生产GAE匹配。
3. float64 max diff <1e-10；float32 <1e-6。
4. padding/chance垃圾action index不再影响gather。
5. active非法action index仍fail-fast。
6. padding macro的last-for-player=False。
7. 宏长10/16等分段正确。
8. 真实C++ commit/tau/边界验证继续通过。
9. baseline manifest不再包含失效的自引用hash。
10. manifest所有声明输入hash可复算。
11. 固定created_at重复构建结果稳定。
12. 旧训练路径、参数量、checkpoint正式检查不变。
13. 全量回归无新增失败。
14. 未实施任何阶段2内容。

---

## 九、交付报告格式

最终回复必须包含：

1. 原错误根因分析；
2. 修改文件清单；
3. 修正后的delta公式与实现对应；
4. gamma=0/0.5/0.97/1.0精确反例结果；
5. padding垃圾index测试；
6. padding macro mask测试；
7. 长宏测试；
8. baseline双构建/哈希验证；
9. 阶段专项测试实际输出；
10. 全量回归实际输出；
11. 新manifest/report/probe路径和sha256；
12. git范围证明；
13. 遗留风险；
14. 明确声明未实施阶段2及以后内容。

修复完成后停止，不要执行阶段2，等待Codex独立复验。
