"""Versioned A04 transport types. Policy values contain no omniscient objects."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .action_state import ProtocolError
from .cards import Rank, validate_level_rank
from .state import PreviousHandResult

RULES_VERSION = "GD-RULES-0.1"
ACTION_VERSION = "GD-ACTION-0.1"
ENV_VERSION = "GD-ENV-0.1"
ENCODING_VERSION = "GD-ENCODING-0.1"
SERIALIZATION_VERSION = 1


def strict_int(value, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, not a coerced value")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return result


@dataclass(frozen=True, slots=True)
class GameConfig:
    level_rank: int = int(Rank.TWO)
    seed: int | None = None
    leader_seat: int = 0
    previous_result: PreviousHandResult | None = None
    max_token_steps: int = 4096

    def __post_init__(self) -> None:
        level = strict_int(self.level_rank, "level_rank")
        object.__setattr__(self, "level_rank", int(validate_level_rank(level)))
        if self.seed is not None:
            object.__setattr__(self, "seed", strict_int(self.seed, "seed"))
        object.__setattr__(self, "leader_seat", strict_int(self.leader_seat, "leader_seat", maximum=3))
        object.__setattr__(self, "max_token_steps", strict_int(self.max_token_steps, "max_token_steps", minimum=1))
        if self.previous_result is not None and not isinstance(self.previous_result, PreviousHandResult):
            raise TypeError("previous_result must be PreviousHandResult or None")


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    token_vocab_size: int = 256
    pad_token_id: int = 0
    max_observation_tokens: int = 4096
    max_action_tokens: int = 32
    max_legal_next_tokens: int = 256
    num_state_channels: int = 256

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, strict_int(getattr(self, name), name))
        if self.token_vocab_size != 256 or self.pad_token_id != 0 or self.num_state_channels != 256:
            raise ProtocolError("GD-ENV-0.1 requires vocabulary=256, PAD=0, channels=256")
        # Smaller capacities are explicit diagnostics, never a top-N filter.
        for name, ceiling in (("max_observation_tokens", 4096), ("max_action_tokens", 32), ("max_legal_next_tokens", 256)):
            if not 1 <= getattr(self, name) <= ceiling:
                raise ProtocolError(f"{name} must be in 1..{ceiling} for GD-ENV-0.1")


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
    private_hand_card_ids: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class CommittedAction:
    """Policy-facing commit receipt; no embedded A02 action or full state.

    Exchange card identities remain redacted until the whole exchange resolves.
    """
    actor: int
    phase: str
    kind: str
    family: str | None
    card_ids: tuple[int, ...]
    declared_ranks: tuple[int, ...]
    declared_suit: int | None
    wild_assignments: tuple[tuple[int, int, int | None], ...]
    public_index: int


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Observation | None
    rewards: np.ndarray
    done: bool
    truncated: bool
    is_commit: bool
    committed_action: CommittedAction | None
    token_step: int
    committed_step: int
    info: dict


@dataclass(frozen=True, slots=True)
class BatchObservation:
    player_id: np.ndarray
    team_id: np.ndarray
    phase: np.ndarray
    token_step: np.ndarray
    committed_step: np.ndarray
    observation_tokens: np.ndarray
    state_channels: np.ndarray
    legal_next_tokens: np.ndarray
    legal_next_mask: np.ndarray
    legal_next_is_commit: np.ndarray
    legal_next_kinds: np.ndarray
    legal_next_values: np.ndarray
    private_hand_card_ids: None = None
    present: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class BatchStepResult:
    observation: BatchObservation
    rewards: np.ndarray
    done: np.ndarray
    truncated: np.ndarray
    is_commit: np.ndarray
    committed_action: tuple[CommittedAction | None, ...]
    token_step: np.ndarray
    committed_step: np.ndarray
    info: tuple[dict, ...]
