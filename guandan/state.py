"""State primitives for the independent single-hand GuanDan engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np

from .cards import Card, Rank, Suit, encode_cards, full_deck, sort_cards, validate_level_rank
from .combos import Combination, CombinationKind, can_beat, require_combination


class GuandanError(Exception):
    pass


class IllegalActionError(GuandanError):
    pass


class EpisodeTerminatedError(GuandanError):
    pass


class StateInvariantError(GuandanError):
    pass


class Phase(str, Enum):
    PLAY = "play"
    TRIBUTE = "tribute"
    RETURN = "return"
    TERMINAL = "terminal"


TEAM_OF = (0, 1, 0, 1)


@dataclass(frozen=True, slots=True)
class PreviousHandResult:
    ranking: tuple[int, int, int, int]
    winner_team: int
    outcome_class: str


@dataclass(frozen=True, slots=True)
class CommittedAction:
    """One atomic action; Scheme B will construct this later."""

    player_id: int
    kind: CombinationKind | str
    cards: tuple[Card, ...] = ()
    declaration: Combination | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cards", tuple(self.cards))
        if isinstance(self.kind, str) and self.kind not in {"pass", "tribute", "return"}:
            object.__setattr__(self, "kind", CombinationKind(self.kind))

    @property
    def is_pass(self) -> bool:
        return self.kind == "pass"


@dataclass(slots=True)
class RoundState:
    """Mutable omniscient state for one atomic-action hand."""

    hands: list[list[Card]]
    level_rank: Rank = Rank.TWO
    active_seat: int = 0
    leader_seat: int = 0
    phase: Phase = Phase.PLAY
    current_winning: Combination | None = None
    current_winner_seat: int | None = None
    consecutive_passes: int = 0
    finished_ranks: list[int] = field(default_factory=list)
    done: bool = False
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    committed_step: int = 0
    history: list[CommittedAction] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.level_rank = validate_level_rank(self.level_rank)
        self.hands = [list(sort_cards(hand)) for hand in self.hands]
        if len(self.hands) != 4:
            raise ValueError("a GuanDan round requires four hands")
        if not 0 <= self.active_seat < 4 or not 0 <= self.leader_seat < 4:
            raise ValueError("seat must be in [0,4)")
        self.check_invariants()

    @classmethod
    def deal(cls, *, seed: int = 0, level_rank: Rank = Rank.TWO, leader_seat: int = 0) -> "RoundState":
        cards = list(full_deck())
        np.random.default_rng(seed).shuffle(cards)
        return cls([cards[i * 27:(i + 1) * 27] for i in range(4)], level_rank=level_rank,
                   active_seat=leader_seat, leader_seat=leader_seat)

    @classmethod
    def from_hands(cls, hands: Iterable[Iterable[Card]], *, level_rank: Rank = Rank.TWO,
                   leader_seat: int = 0) -> "RoundState":
        return cls([list(hand) for hand in hands], level_rank=level_rank,
                   active_seat=leader_seat, leader_seat=leader_seat)

    def clone(self) -> "RoundState":
        return RoundState(
            hands=[list(hand) for hand in self.hands], level_rank=self.level_rank,
            active_seat=self.active_seat, leader_seat=self.leader_seat,
            phase=self.phase, current_winning=self.current_winning,
            current_winner_seat=self.current_winner_seat,
            consecutive_passes=self.consecutive_passes,
            finished_ranks=list(self.finished_ranks), done=self.done,
            rewards=self.rewards.copy(), committed_step=self.committed_step,
            history=list(self.history),
        )

    def check_invariants(self) -> None:
        ids = [card.card_id for hand in self.hands for card in hand]
        played = [card.card_id for action in self.history for card in action.cards]
        if len(ids) != len(set(ids)) or len(played) != len(set(played)):
            raise StateInvariantError("a physical card is duplicated")
        if set(ids) | set(played) != set(card.card_id for card in full_deck()):
            raise StateInvariantError("physical card was lost or duplicated")
        if set(ids) & set(played):
            raise StateInvariantError("a played card remains in a hand")
        if len(self.finished_ranks) != len(set(self.finished_ranks)):
            raise StateInvariantError("finish ranks must be unique")
        if self.done and self.phase is not Phase.TERMINAL:
            raise StateInvariantError("done state must be terminal")
        if not self.done and self.phase is Phase.TERMINAL:
            raise StateInvariantError("non-done state cannot be terminal")

    def serialize(self) -> dict:
        return {
            "level_rank": int(self.level_rank), "active_seat": self.active_seat,
            "leader_seat": self.leader_seat, "phase": self.phase.value,
            "hands": [list(encode_cards(hand)) for hand in self.hands],
            "finished_ranks": list(self.finished_ranks), "done": self.done,
            "committed_step": self.committed_step,
            "history": [{"player_id": a.player_id, "kind": str(a.kind), "cards": list(encode_cards(a.cards))}
                        for a in self.history],
        }
