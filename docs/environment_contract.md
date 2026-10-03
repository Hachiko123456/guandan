# 掼蛋环境契约

- **文档 ID：** GD-ENV-0.1
- **兼容规则：** GD-RULES-0.1
- **运行时：** Python 3.12+；使用 NumPy 数组进行传输；游戏引擎不要求 PyTorch。
- **范围：** 单副牌局环境，可选提供上一副牌局结果以设置进贡/还贡。
- **V1 固定维度：** `token_vocab_size=256`, `pad_token_id=0`, `max_observation_tokens=4096`, `max_action_tokens=32`, `max_legal_next_tokens=256`, `num_state_channels=256`。
- **V1 token 目标：** 每个环境步骤进行一次标量 token 决策；扩展前缀的奖励为零；PPO/GAE 使用 token 步作为时间轴。
- **V1 重置策略：** 基础 `GuandanEnv` 从不自动重置；`GuandanEnvBatch` 默认为 `auto_reset=False`；训练适配器在记录终局转移后显式重置终局槽位。

## 1. 设计原则

1. 在显式指定 seed 和初始状态时，环境具有确定性。
2. 环境只暴露当前行动玩家的合法下一 token 选择。
3. 环境区分 token 步和原子的已提交游戏步。
4. 环境绝不静默修复、截断或替换非法动作。
5. 对于相同的 seed、状态和 token 序列，单环境与批环境的行为完全一致。
6. 策略观测与完整信息状态是分离的类型。策略路径不得接收对手手牌。

## 2. 公共 Python 接口

包必须从 guandan.environment 提供以下名称，或通过文档说明的重新导出提供这些名称：

~~~python
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class GameConfig:
    level_rank: int
    seed: int | None = None
    leader_seat: int = 0
    previous_result: "PreviousHandResult | None" = None
    max_token_steps: int = 4096

@dataclass(frozen=True)
class PreviousHandResult:
    ranking: tuple[int, int, int, int]
    winner_team: int
    outcome_class: str
    tribute_cancelled: bool = False

@dataclass(frozen=True)
class ObservationSpec:
    token_vocab_size: int
    pad_token_id: int
    max_observation_tokens: int
    max_action_tokens: int
    max_legal_next_tokens: int
    num_state_channels: int

@dataclass(frozen=True)
class Observation:
    player_id: int
    team_id: int
    phase: str
    token_step: int
    committed_step: int
    observation_tokens: np.ndarray
    state_channels: np.ndarray
    legal_next_tokens: np.ndarray
    legal_next_mask: np.ndarray
    legal_next_is_commit: np.ndarray
    legal_next_kinds: np.ndarray
    legal_next_values: np.ndarray
    private_hand_card_ids: np.ndarray | None

@dataclass(frozen=True)
class StepResult:
    observation: Observation | None
    rewards: np.ndarray              # shape (4,), float32
    done: bool
    truncated: bool
    is_commit: bool
    committed_action: "CommittedAction | None"
    token_step: int
    committed_step: int
    info: dict

class GuandanEnv:
    observation_spec: ObservationSpec
    def __init__(self, config: GameConfig): ...
    def reset(self, *, seed: int | None = None,
              initial_state: bytes | None = None) -> Observation: ...
    def step(self, token_id: int) -> StepResult: ...
    def legal_tokens(self) -> np.ndarray: ...
    def clone(self) -> "GuandanEnv": ...
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, payload: bytes) -> "GuandanEnv": ...

class GuandanEnvBatch:
    def __init__(self, configs: Sequence[GameConfig]): ...
    def reset(self, *, seeds: Sequence[int] | None = None) -> "BatchObservation": ...
    def step(self, token_ids: np.ndarray) -> "BatchStepResult": ...
    def serialize(self) -> list[bytes]: ...
    def load_serialized(self, payloads: Sequence[bytes]) -> None: ...
~~~

只有当包重新导出相同的公共 API，并在版本化变更日志中记录该选择时，名称才可以放置在不同模块中。

## 3. 输入和输出约定

### 3.1 Token 数组

- Token ID 是非负的有符号 32 位整数。
- `PAD=0` 保留用于填充，且永远不合法。`BOS=1`。阶段 token 为 2..5（`LEAD_PLAY`、`FOLLOW_PLAY`、`TRIBUTE`、`RETURN`）。`PASS=6`，`COMMIT=7`。牌型 token 为 8..17。点数 token 为 18..32。花色 token 为 33..37。长度 token 为 38..48。牌 token 为 64..171，对应实体牌 ID 0..107。逢人牌指定 token 为 172..223，对应 13 个非王点数 × 4 种花色。49..63 和 224..255 保留。
- pad_token_id 位于合法 token-ID 集合之外。
- legal_next_tokens、legal_next_mask、legal_next_is_commit、legal_next_kinds 和 legal_next_values 共享首维 max_legal_next_tokens。
- 无效的填充行的 mask 为 false、is_commit 为 false，且元数据清零。
- 调用方每个环境步骤提交一个标量 token_id。

### 3.2 状态通道

- state_channels 是由 ObservationSpec 声明的固定大小 float32 向量或矩阵。
- 它包含公开状态、当前行动玩家的私有信息以及构造前缀。
- 完整发牌信息可以存在于引擎状态中，但不会进入 Observation，当前行动玩家的手牌除外。
- 集中式价值评估器的输入必须使用独立的完整状态 API；不能通过策略观测上的标志获得。

### 3.3 动作元数据

legal_next_kinds 和 legal_next_values 是用于日志记录/模型嵌入的语义元数据。环境按 token ID 和当前前缀进行验证，绝不信任调用方元数据。

## 4. 重置契约

reset()：

- 创建或恢复一场单副牌局；
- 除非提供显式的序列化状态或测试夹具状态，否则将全部 108 张牌发成四手各有 27 张牌的手牌；
- 将 token_step 和 committed_step 设为 0；
- 清除 done 和 truncated；
- 清除未完成前缀；
- 返回指定领出者的观测；若上一副牌局结果要求交换，则返回第一位进贡/还贡行动者的观测。

对于确定性测试，seed 和序列化的初始状态必须复现相同的发牌、领出者、公开历史和合法 token 顺序。

## 5. 步骤契约

step(token_id) 为当前行动玩家恰好消耗一个合法的下一 token。

### 5.1 非提交 token

- token_step 加一。
- committed_step 不增加。
- 不改变公开的牌归属、行动顺序、轮次领先者或终局名次。
- 只更新当前行动玩家的未完成构造。
- 返回零奖励、is_commit 为 false、committed_action 为 None。

### 5.2 提交 token

- token_step 加一。
- committed_step 加一。
- 解析由前缀表示的完整动作。
- 更新牌归属、公开历史、行动顺序、进贡/还贡交换、轮次状态或终局结果。
- 返回 is_commit 为 true 以及规范的 CommittedAction。
- 在下一次观测之前清除未完成前缀。
- 仅在本局结束时返回终局奖励；否则奖励全部为零。

### 5.3 过牌、进贡与还贡

- PASS 是一个单 token 的已提交动作，仅在存在当前领先出牌的跟牌决策期间可用。
- 进贡牌和还贡牌均使用方案 B 选择，并各提交一次。
- 非法 token 或提交会抛出 IllegalActionError，并保持状态不变。
- 在 done 或 truncated 之后调用 step 会抛出 EpisodeTerminatedError。

## 6. 批处理契约

GuandanEnvBatch 是相互独立的 GuandanEnv 实例集合，每个槽位有一个当前行动玩家。BatchObservation 和 BatchStepResult 的字段与单环境类型等价，只是增加首维 B。

必须保证：

- 槽位 i 在行为上等价于 config 和 token 序列相同的独立环境；
- 当其他槽位执行步骤时，done 槽位不发生变化；
- 槽位可以有不同的当前行动玩家和合法 token 数量，并通过掩码/填充表示；
- 不同的 seed 创建相互独立的随机流；
- 槽位之间不共享可变状态对象；
- rewards 的形状为 (B, 4)，且在非终局转移中全部为零。

## 7. 错误与不变量

必需的异常类型：

~~~python
class GuandanError(Exception): ...
class IllegalActionError(GuandanError): ...
class ProtocolError(GuandanError): ...
class EpisodeTerminatedError(GuandanError): ...
class StateInvariantError(GuandanError): ...
~~~

在调试/测试模式下，每次已提交转移后都断言：

- 每张实体牌恰好属于某位玩家的手牌、转移池或已出牌堆中的一处；
- 没有牌被重复或丢失；
- 手牌张数与可见的剩余牌数一致；
- 非终局状态下恰好存在一个当前行动玩家；
- 没有已完牌玩家被安排进行普通回合；
- 当前轮次领先出牌为空，或为一个已提交的合法出牌；
- 提交后前缀立即为空；
- 终局名次恰好包含每个名次一次；
- 终局奖励为队内共享的零和奖励，并与 winner_team 一致。

## 8. 序列化与可复现性

serialize() 包含规则版本、级牌、RNG 状态、实体牌分配、公开历史、私有手牌、前缀、阶段、当前行动座位、轮次状态、完牌名次以及终局/截断标志。

serialize/deserialize 往返过程会保留全部公开/私有状态、合法 token 集合及其顺序、两个计数器，以及对每个合法 token 序列的下一次已提交结果。

## 9. 训练集成边界

环境是顺序式的：每个槽位一次只暴露一位当前行动玩家。训练适配器将 player_id/team_id 映射到多智能体轨迹采样布局。

首个适配器暴露当前玩家观测、四玩家奖励向量、is_commit、终局/截断标志、两个计数器、动作掩码和合法 token 元数据。环境不计算 PPO/VRPO 优势、损失或经验回放索引。

## 10. V1 固定决策与延后范围

1. 文档开头的 `ObservationSpec` 数值和 token 范围在 v1 中固定。任何溢出均为 `ProtocolError`；不允许截断。
2. `private_hand_card_ids` 仅可通过受保护的调试/完整状态 API 使用，在策略观测中为 `None`。
3. 基础环境不自动重置。训练适配器在记录终局转移后显式重置终局槽位。
4. `truncated` 仅用于 `max_token_steps` 和调用方显式停止；每次截断都必须有 `info["termination_reason"]`。
5. 未来扩展词汇表需要新的观测协议版本，并使检查点不兼容。
