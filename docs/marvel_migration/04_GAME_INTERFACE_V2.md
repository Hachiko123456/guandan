# Guandan MARVEL V2 游戏接口草案

## 1. 版本

- 动作协议：`GD-ACTION-0.2-MARVEL`
- 环境协议：`GD-ENV-0.2-MARVEL`
- 编码协议：`GD-ENCODING-0.2-MARVEL`

M02 已在 `guandan_marvel.game` 实现 Python reference adapter。规范化的常量、schema、容量依据和 channel/action 布局见 `M02_ADAPTER_PROTOCOL.md`；本文保留跨阶段接口原则。该实现不是 C++ 或高性能正式训练后端。

## 2. 游戏模块常量

V2 game module 至少暴露：

```text
num_players = 4
num_tokens
num_channels
pad_token
max_game_horizon
max_available_actions
max_observation_tokens
max_action_tokens
EnvBatch
actions_from_np
```

所有容量必须来自经过扫描和测试的上界，不允许对合法候选做 top-N 静默截断。

## 3. 环境 step 输出

单次批量 reset/step 输出：

```text
rewards                       float32 [B,4]
is_terminal                   bool    [B]
player_ids                    int64   [B]
observations.tokens           int64   [B,4,L_obs]
observations.channels         uint8/float [B,4,C]
available_actions.tokens      int64   [B,A,L_action]
available_actions.weights     float32 [B,A]
available_actions.is_commit   bool    [B,A]
```

`player_ids < 0` 仅用于环境控制的 chance/system step；掼蛋首版若发牌在 reset 内完成，则普通 play/tribute/return 决策应给出实际玩家。

## 4. 完整轨迹布局

```text
num_steps        int64   [B]
player_ids       int64   [B,T]
obs_tokens       int64   [B,T,4,L_obs]
obs_channels     compact [B,T,4,C]
avail_tokens     int64   [B,T,A,L_action]
avail_weights    float32 [B,T,A]
avail_is_commit  bool    [B,T,A]
policy_probs              float32 [B,T,A]
behavior_action_prob       float32 [B,T]
on_policy_valid            bool    [B,T]
action_taken     int64   [B,T]
rewards          float32 [B,T,4]
is_terminal      bool    [B,T]
```

B 表示完整终局轨迹数量；T 是环境 token step；COMMIT 通过 `avail_is_commit` 区分真实游戏动作与前缀动作。

## 5. 观测原则

每个时间步生成四个座位各自的信息集。对座位 i：

允许：

- 自己的物理手牌；
- 当前级牌、阶段、行动者；
- 已公开历史；
- 公开剩余牌数；
- 已完成交换后公开的贡还牌；
- 自己未提交的私有前缀/贡还牌选择；
- 自己的座位/队伍身份。

禁止：

- 其他玩家物理手牌；
- 未完成交换时其他玩家私有选择；
- 训练期完整牌归属；
- critic 专用标签。

M02 必须保留 A04 的隐私重排测试，并扩展到四玩家 V2 输出。

## 6. 单步 observation token

V2 不在每个时间步重复完整 4096-token 历史。推荐：

- token 表示本次新增事件、阶段和必要的离散语义；
- 当前手牌、剩余计数、完成名次、级牌和稳定状态放 channels；
- Transformer 沿 T 维累积历史；
- 单步 token 内部仍可采用 order-insensitive sum embedding，但跨时间顺序由 causal Transformer 保留。

M02 必须通过消融/测试确认没有遗漏 V1 中决策所需的公开信息。

## 7. 结构化候选动作

每个候选是固定长度 token row，而不是单一 token ID。候选可包含：

```text
PHASE
PREFIX_STAGE
CONTROL/PASS/COMMIT
FAMILY
RANKS
SUIT
LENGTH
CARD/GROUP
WILD_TARGET
COMMIT_FLAG
```

同时需要一个执行载荷或可逆 token row，使 `actions_from_np()` 能唯一恢复当前合法动作。不同语义动作不得共享同一 row；语义等价声明应 canonicalize。

## 8. critic 信息

IPPO value head 只使用 actor 信息。

VRPO centralized critic 可使用：

- 四座位信息集；
- 训练期完整 ownership channels；
- 当前候选动作；
- 队伍映射。

critic 不得修改 actor observation tensor，也不得通过共享缓存泄露完整状态。

## 9. 终局与截断

- 完整 trajectory 必须真实终局；
- 资源预算停止不能伪造成终局；
- horizon overflow 是错误或受控 incomplete，不得给出胜负奖励；
- 完整对局 collector 的训练 batch 不接受未标明来源的截断轨迹。
