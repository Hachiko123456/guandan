# GuanDan Environment Contract

- **Document ID:** GD-ENV-0.1
- **Compatible rules:** GD-RULES-0.1
- **Runtime:** Python 3.12+; NumPy arrays for transport; PyTorch is not required by the game engine.
- **Scope:** single-hand environment with optional previous-hand outcome for tribute/return setup.
- **V1 fixed dimensions:** `token_vocab_size=256`, `pad_token_id=0`, `max_observation_tokens=4096`, `max_action_tokens=32`, `max_legal_next_tokens=256`, `num_state_channels=256`.
- **V1 token objective:** one scalar token decision per environment step; prefix extension has zero reward; PPO/GAE uses token steps as its time axis.
- **V1 reset policy:** base `GuandanEnv` never auto-resets; `GuandanEnvBatch` defaults to `auto_reset=False`; the training adapter explicitly resets terminal slots after recording their terminal transition.

## 1. Design principles

1. The environment is deterministic under an explicit seed and an explicit initial state.
2. It exposes only the acting player's legal next token choices.
3. It distinguishes a token step from an atomic committed game step.
4. It never silently repairs, truncates, or substitutes an illegal action.
5. Single-environment and batch-environment behavior are identical for the same seed, state, and token sequence.
6. Policy observations and full-information state are separate types. The policy path must not receive opponent hands.

## 2. Public Python surface

The package must provide these names from guandan.environment or a documented re-export:

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

Names may be placed in different modules only if the package re-exports the same public API and records that choice in a versioned changelog.

## 3. Input and output conventions

### 3.1 Token arrays

- Token IDs are non-negative signed 32-bit integers.
- `PAD=0` is reserved for padding and is never legal. `BOS=1`. Phase tokens are 2..5 (`LEAD_PLAY`, `FOLLOW_PLAY`, `TRIBUTE`, `RETURN`). `PASS=6`, `COMMIT=7`. Family tokens are 8..17. Rank tokens are 18..32. Suit tokens are 33..37. Length tokens are 38..48. Card tokens are 64..171 for physical card IDs 0..107. Wild-assignment tokens are 172..223 for 13 non-joker ranks × 4 suits. 49..63 and 224..255 are reserved.
- pad_token_id is outside the legal token-ID set.
- legal_next_tokens, legal_next_mask, legal_next_is_commit, legal_next_kinds, and legal_next_values share the first dimension max_legal_next_tokens.
- Invalid padded rows have mask false, is_commit false, and zeroed metadata.
- A caller submits one scalar token_id per environment per step.

### 3.2 State channels

- state_channels is a fixed-size float32 vector or matrix declared by ObservationSpec.
- It contains public state plus the acting player's private information and construction prefix.
- The full deal may exist in engine state but is not in Observation except through the acting player's hand.
- Centralized-critic input must use a separate full-state API; it cannot be obtained by a flag on the policy observation.

### 3.3 Action metadata

legal_next_kinds and legal_next_values are semantic metadata for logging/model embeddings. The environment validates by token ID and current prefix, never by trusting caller metadata.

## 4. Reset contract

reset():

- creates or restores one hand;
- deals all 108 cards into four 27-card hands unless an explicit serialized/fixture state is supplied;
- sets token_step and committed_step to 0;
- clears done and truncated;
- clears the unfinished prefix;
- returns an observation for the designated leader or the first tribute/return actor when a previous result requires exchange.

For deterministic tests, seed and a serialized initial state must reproduce the same deal, leader, public history, and legal-token ordering.

## 5. Step contract

step(token_id) consumes exactly one legal next token for the active player.

### 5.1 Non-commit token

- Increments token_step by one.
- Does not increment committed_step.
- Does not alter public card ownership, turn order, trick winner, or terminal ranking.
- Updates only the active player's unfinished construction.
- Returns zero reward, is_commit false, and committed_action None.

### 5.2 Commit token

- Increments token_step by one.
- Increments committed_step by one.
- Resolves the complete action represented by the prefix.
- Updates card ownership, public history, turn order, tribute/return exchange, trick state, or terminal result.
- Returns is_commit true and a canonical CommittedAction.
- Clears the unfinished prefix before the next observation.
- Returns terminal rewards only when the hand ends; otherwise rewards are all zero.

### 5.3 Pass, tribute, and return

- PASS is a one-token committed action available only during a follow decision with an existing winning play.
- A tribute card and a return card are each selected with Scheme B and committed once.
- An illegal token or commit raises IllegalActionError and leaves state unchanged.
- step after done or truncated raises EpisodeTerminatedError.

## 6. Batch contract

GuandanEnvBatch is a collection of independent GuandanEnv instances with one active player per slot. BatchObservation and BatchStepResult have fields equivalent to single-environment types with leading dimension B.

Required guarantees:

- slot i is behaviorally equivalent to a standalone environment with the same config and token sequence;
- a done slot does not mutate when other slots step;
- slots may have different active players and legal-token counts, represented by masks/padding;
- distinct seeds create independent random streams;
- no mutable state object is shared between slots;
- rewards has shape (B, 4) and is all zero on non-terminal transitions.

## 7. Errors and invariants

Required exception types:

~~~python
class GuandanError(Exception): ...
class IllegalActionError(GuandanError): ...
class ProtocolError(GuandanError): ...
class EpisodeTerminatedError(GuandanError): ...
class StateInvariantError(GuandanError): ...
~~~

In debug/test mode assert after every committed transition:

- every physical card belongs to exactly one hand, transfer pool, or played pile;
- no card is duplicated or lost;
- hand card counts agree with visible remaining counts;
- exactly one active player exists when non-terminal;
- no finished player is scheduled for a normal turn;
- current trick winner is absent or a committed legal play;
- prefix is empty immediately after commit;
- terminal ranking contains each rank exactly once;
- terminal reward is shared zero-sum and matches winner_team.

## 8. Serialization and reproducibility

serialize() includes rule version, level, RNG state, physical allocation, public history, private hands, prefix, phase, active seat, trick state, finishing ranks, and terminal/truncated flags.

A serialize/deserialize round trip preserves all public/private state, legal-token set and order, both counters, and the next committed result for every legal token sequence.

## 9. Training integration boundary

The environment is sequential: one slot exposes one active player at a time. The training adapter maps player_id/team_id to the multi-agent rollout layout.

The first adapter exposes current-player observation, four-player reward vector, is_commit, terminal/truncation flags, both counters, action mask, and legal-token metadata. The environment does not compute PPO/VRPO advantages, losses, or replay indices.

## 10. V1 fixed decisions and deferred scope

1. The numeric `ObservationSpec` and token ranges in the header are fixed for v1. Any overflow is a `ProtocolError`; no truncation is allowed.
2. `private_hand_card_ids` is available only from a protected debug/full-state API and is `None` in policy observations.
3. The base environment does not auto-reset. The training adapter resets terminal slots explicitly after recording the terminal transition.
4. `truncated` is used for `max_token_steps` and explicit caller stop only; `info["termination_reason"]` is mandatory for every truncation.
5. Future vocabulary expansion requires a new observation protocol version and checkpoint incompatibility.
