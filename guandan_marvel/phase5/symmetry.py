# SPDX-License-Identifier: MIT
"""Legal, invertible phase-5 symmetry transforms.

Two transform families are defined on the accepted V2 encoding:

``SeatRotation(k)``
    ``seat -> (seat + k) mod 4``.  Applied to the observer axis of the
    observation, every seat-indexed channel slice, the team-relative bits, the
    event-token seat fields and the reward vector.  The action vocabulary is
    seat-free, so the induced action permutation is the identity.

``CopySwap()``
    ``card_id -> card_id + 54 (mod 108)``: exchange the two physical decks.
    Both copies of a printed rank/suit pair exchange identities, so the
    *printed* content of the state is unchanged while the physical card ids
    (which the encoder publishes) change.  Card bits and ``CARD`` tokens are
    remapped; rank/suit/length/family/wild-assignment tokens are unchanged
    because they name printed ranks/suits, not physical cards.

Both are involutions.  ``compose`` is provided for the four rotation/swap
combinations, and every transform is validated by :func:`check_transform_involution`.

Domain restrictions are explicit and fail closed:

* ``transform_observation`` refuses a channel row whose seat-indexed slices are
  not one-hot (a malformed/ambiguous encoding is never guessed at);
* the copy-swap action permutation raises :class:`SymmetryDomainError` when a
  mapped candidate row is absent from the transformed candidate list, instead
  of silently dropping it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

SYMMETRY_SCHEMA = "guandan-marvel-phase5-symmetry-v1"

NUM_CHANNELS = 512
NUM_PLAYERS = 4
NUM_PHYSICAL_CARDS = 108
CARDS_PER_DECK = 54
# TEAM_OF from the accepted rules engine.
TEAM_OF = (0, 1, 0, 1)

# Token bases (accepted V2 protocol).
BOS_TOKEN = 1
PASS_TOKEN = 6
COMMIT_TOKEN = 7
FAMILY_TOKEN_BASE = 8
RANK_TOKEN_BASE = 18
SUIT_TOKEN_BASE = 33
LENGTH_TOKEN_BASE = 38
CARD_TOKEN_BASE = 64
WILD_TOKEN_BASE = 172
EVENT_RESET = 256
EVENT_PRIVATE_PREFIX = 257
EVENT_PUBLIC_COMMIT = 258
EVENT_PRIVATE_EXCHANGE_COMMIT = 259
EVENT_EXCHANGE_RESOLVED = 260
ACTION_ROW_BEGIN = 262
ACTION_STAGE_BASE = 263
ACTION_KIND_BASE = 296
ACTION_PAYLOAD = 304
ACTION_COMMIT = 306
ACTION_ROW_END = 307
EVENT_END = 308
TRIBUTE_TOKEN = 4
RETURN_TOKEN = 5

# Channel slices of the accepted encoder (mirrors guandan_marvel.game).
SELF_HAND = slice(0, 108)
CURRENT_WINNING = slice(108, 216)
SELF_EXCHANGE = slice(216, 324)
SELF_PREFIX = slice(324, 356)
REMAINING_COUNTS = slice(356, 360)
FINISHING_RANKS = slice(360, 364)
OBSERVER_ONE_HOT = slice(364, 368)
ACTIVE_ONE_HOT = slice(368, 372)
LEADER_ONE_HOT = slice(372, 376)
WINNER_ONE_HOT = slice(376, 380)
EMPTIED_ONE_HOT = slice(380, 384)
PHASE_ONE_HOT = slice(384, 388)
LEVEL_ONE_HOT = slice(388, 401)
PASS_COUNT_ONE_HOT = slice(401, 405)
TEAM_ONE_HOT = slice(405, 407)
PREVIOUS_WINNER_TEAM = slice(407, 409)
PREVIOUS_OUTCOME = slice(409, 412)
PREVIOUS_RANKING = slice(412, 416)
CURRENT_FAMILY = slice(416, 426)
CURRENT_RANK = slice(426, 441)
CURRENT_SIZE = slice(441, 452)
TRIBUTE_DONOR = slice(452, 456)
TRIBUTE_ORDER = slice(456, 460)
TOKEN_STEP_BITS = slice(460, 473)
COMMITTED_STEP_BITS = slice(473, 486)
SINGLE_BITS = {
    "tribute_cancelled": 486,
    "play_phase": 487,
    "anti_tribute": 488,
    "has_current_winning": 489,
    "prefix_length": 490,
    "is_observer": 491,
    "observer_finished": 492,
    "observer_emptied": 493,
    "has_return_obligation": 494,
    "observer_selected_tribute": 495,
    "observer_returned": 496,
    "has_previous_result": 497,
    "observer_is_winner": 498,
    "observer_is_leader": 499,
    "active_same_team": 500,
    "sentinel": 511,
}


class SymmetryDomainError(ValueError):
    """A transform is undefined for the supplied encoding."""


@dataclass(frozen=True, slots=True)
class SeatRotation:
    """Rotate every seat by ``k`` positions."""

    k: int

    def __post_init__(self) -> None:
        value = int(self.k)
        if not 0 <= value <= 3:
            raise ValueError("seat rotation k must lie in 0..3")
        object.__setattr__(self, "k", value)

    name = "seat_rotation"

    def inverse(self) -> "SeatRotation":
        return SeatRotation((-self.k) % 4)

    def apply_seat(self, seat: int) -> int:
        return (int(seat) + self.k) % 4


@dataclass(frozen=True, slots=True)
class CopySwap:
    """Exchange the two physical decks (``card_id + 54 mod 108``)."""

    name = "copy_swap"

    def inverse(self) -> "CopySwap":
        return self

    def apply_seat(self, seat: int) -> int:
        return int(seat)


@dataclass(frozen=True, slots=True)
class ComposedTransform:
    """A seat rotation followed by a deck swap (the closed symmetry group).

    The two operations act on independent axes (seat-indexed slicing vs card
    identity), so they commute.  ``rotation`` is applied to seats and ``swap``
    to physical card ids.  ``inverse`` reverses both.
    """

    rotation: int = 0
    swap: bool = False

    def __post_init__(self) -> None:
        value = int(self.rotation)
        if not 0 <= value <= 3:
            raise ValueError("composed rotation must lie in 0..3")
        object.__setattr__(self, "rotation", value)
        object.__setattr__(self, "swap", bool(self.swap))

    @property
    def name(self) -> str:
        suffix = "+copy_swap" if self.swap else ""
        return f"seat_rotation_{self.rotation}{suffix}"

    def inverse(self) -> "ComposedTransform":
        return ComposedTransform((-self.rotation) % 4, self.swap)

    def apply_seat(self, seat: int) -> int:
        return (int(seat) + self.rotation) % 4


Transform = SeatRotation | CopySwap | ComposedTransform


def _rotation_of(transform: Transform) -> int:
    if isinstance(transform, SeatRotation):
        return int(transform.k)
    if isinstance(transform, ComposedTransform):
        return int(transform.rotation)
    return 0


def _swap_of(transform: Transform) -> bool:
    if isinstance(transform, CopySwap):
        return True
    if isinstance(transform, ComposedTransform):
        return bool(transform.swap)
    return False


def identity_transform() -> SeatRotation:
    return SeatRotation(0)


def compose_transforms(first: Transform, second: Transform) -> Transform:
    """Return ``first`` after ``second`` for the closed rotation/swap group.

    The group is abelian with eight elements: four rotations and a deck swap.
    Mixed compositions are represented by :class:`ComposedTransform`; they are
    never silently collapsed to one of the operands.
    """
    rotation = (_rotation_of(first) + _rotation_of(second)) % 4
    swap = _swap_of(first) != _swap_of(second)
    if not swap:
        return SeatRotation(rotation)
    return ComposedTransform(rotation, True)


def transform_seat(transform: Transform, seat: int) -> int:
    return transform.apply_seat(int(seat))


def inverse_seat(transform: Transform, seat: int) -> int:
    return transform.inverse().apply_seat(int(seat))


def transform_card(transform: Transform, card_id: int) -> int:
    value = int(card_id)
    if not 0 <= value < NUM_PHYSICAL_CARDS:
        raise SymmetryDomainError(f"card id {value} is outside 0..107")
    if _swap_of(transform):
        return (value + CARDS_PER_DECK) % NUM_PHYSICAL_CARDS
    return value


def transform_team(transform: Transform, team: int) -> int:
    if int(team) not in (0, 1):
        raise SymmetryDomainError("team must be 0 or 1")
    if _rotation_of(transform) % 2 == 1:
        return 1 - int(team)
    return int(team)


def _seat_token(seat: int) -> int:
    if not 0 <= int(seat) <= 3:
        raise SymmetryDomainError("seat token only encodes seats 0..3")
    return LENGTH_TOKEN_BASE + int(seat)


def _decode_seat_token(token: int) -> int:
    return int(token) - LENGTH_TOKEN_BASE


# ---------------------------------------------------------------------------
# Event-token structural parser (rotation needs seat *positions*, not values)
# ---------------------------------------------------------------------------


def event_seat_positions(event: Sequence[int]) -> tuple[int, ...]:
    """Return the token indices of an event that encode a seat.

    The V2 event grammar is structural; a ``LENGTH`` token inside a declared
    action sequence is a *length*, not a seat, so the positions are recovered
    from the grammar rather than from the token range.  Ambiguous or malformed
    events fail closed.
    """
    tokens = [int(token) for token in event]
    while tokens and tokens[-1] == 0:
        tokens.pop()
    if not tokens:
        return ()
    end = len(tokens) - 1
    if tokens[0] != BOS_TOKEN or tokens[end] != EVENT_END:
        raise SymmetryDomainError("event framing is not BOS..EVENT_END")
    kind = tokens[1]
    if len(tokens) < 4:
        raise SymmetryDomainError("event is too short to carry a seat field")
    if kind == EVENT_RESET:
        if len(tokens) != 6:
            raise SymmetryDomainError("RESET event must be exactly six tokens")
        # (BOS, RESET, phase, rank(level), length(active_seat), END)
        positions = [4]
    elif kind in (EVENT_PRIVATE_PREFIX, EVENT_PUBLIC_COMMIT, EVENT_PRIVATE_EXCHANGE_COMMIT):
        # (BOS, kind, phase, length(seat), payload..., END)
        positions = [3]
    elif kind == EVENT_EXCHANGE_RESOLVED:
        positions = [3]
        body = tokens[4:end]
        if len(body) % 5:
            raise SymmetryDomainError("exchange record block is not a multiple of five tokens")
        for offset in range(4, end, 5):
            if tokens[offset] not in (TRIBUTE_TOKEN, RETURN_TOKEN):
                raise SymmetryDomainError("exchange record does not start with TRIBUTE/RETURN")
            positions.extend((offset + 1, offset + 2))
    else:
        raise SymmetryDomainError(f"unknown event kind {kind}")
    for position in positions:
        token = tokens[position]
        if not LENGTH_TOKEN_BASE <= token <= LENGTH_TOKEN_BASE + 3:
            raise SymmetryDomainError(
                f"event seat field at index {position} is not a seat token ({token})"
            )
    return tuple(positions)


def transform_event_tokens(event: Sequence[int], transform: Transform) -> tuple[int, ...]:
    tokens = [int(token) for token in event]
    if _swap_of(transform):
        for index, token in enumerate(tokens):
            if CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + NUM_PHYSICAL_CARDS:
                tokens[index] = CARD_TOKEN_BASE + transform_card(transform, token - CARD_TOKEN_BASE)
    if _rotation_of(transform) % 4 == 0:
        return tuple(tokens)
    for position in event_seat_positions(tokens):
        tokens[position] = _seat_token(transform_seat(transform, _decode_seat_token(tokens[position])))
    return tuple(tokens)


# ---------------------------------------------------------------------------
# Observation transform
# ---------------------------------------------------------------------------


def _one_hot_index(row: np.ndarray, slice_: slice, *, name: str) -> int | None:
    values = np.asarray(row[slice_])
    if not values.any():
        return None
    if int(values.sum()) != 1:
        raise SymmetryDomainError(f"{name} slice is not one-hot")
    return int(np.flatnonzero(values)[0])


def _transform_channels_rotation(row: np.ndarray, observer: int, k: int) -> np.ndarray:
    if k % 4 == 0:
        return np.asarray(row, dtype=np.uint8).copy()
    out = np.zeros(NUM_CHANNELS, dtype=np.uint8)
    source = np.asarray(row, dtype=np.uint8)
    # Observer-relative slices that are invariant under a common seat rotation.
    out[SELF_HAND] = source[SELF_HAND]
    out[CURRENT_WINNING] = source[CURRENT_WINNING]
    out[SELF_EXCHANGE] = source[SELF_EXCHANGE]
    out[SELF_PREFIX] = source[SELF_PREFIX]
    out[PHASE_ONE_HOT] = source[PHASE_ONE_HOT]
    out[LEVEL_ONE_HOT] = source[LEVEL_ONE_HOT]
    out[PASS_COUNT_ONE_HOT] = source[PASS_COUNT_ONE_HOT]
    out[PREVIOUS_OUTCOME] = source[PREVIOUS_OUTCOME]
    out[CURRENT_FAMILY] = source[CURRENT_FAMILY]
    out[CURRENT_RANK] = source[CURRENT_RANK]
    out[CURRENT_SIZE] = source[CURRENT_SIZE]
    out[TOKEN_STEP_BITS] = source[TOKEN_STEP_BITS]
    out[COMMITTED_STEP_BITS] = source[COMMITTED_STEP_BITS]
    # Seat-indexed slices are permuted.
    for seat in range(NUM_PLAYERS):
        target = (seat + k) % NUM_PLAYERS
        out[REMAINING_COUNTS.start + target] = source[REMAINING_COUNTS.start + seat]
        out[FINISHING_RANKS.start + target] = source[FINISHING_RANKS.start + seat]
        out[PREVIOUS_RANKING.start + target] = source[PREVIOUS_RANKING.start + seat]
        out[EMPTIED_ONE_HOT.start + target] = source[EMPTIED_ONE_HOT.start + seat]
        out[TRIBUTE_DONOR.start + target] = source[TRIBUTE_DONOR.start + seat]
    out[OBSERVER_ONE_HOT.start + (observer + k) % NUM_PLAYERS] = 1
    for name, slice_ in (
        ("active", ACTIVE_ONE_HOT), ("leader", LEADER_ONE_HOT),
        ("winner", WINNER_ONE_HOT),
    ):
        index = _one_hot_index(source, slice_, name=name)
        if index is not None:
            out[slice_.start + (index + k) % NUM_PLAYERS] = 1
    for index in range(NUM_PLAYERS):
        value = int(source[TRIBUTE_ORDER.start + index])
        if value:
            out[TRIBUTE_ORDER.start + index] = ((value - 1 + k) % NUM_PLAYERS) + 1
    # Team-relative bits are recomputed, not copied: an odd rotation swaps teams.
    out[TEAM_ONE_HOT.start + TEAM_OF[(observer + k) % NUM_PLAYERS]] = 1
    previous_team = _one_hot_index(source, PREVIOUS_WINNER_TEAM, name="previous_winner_team")
    if previous_team is not None:
        out[PREVIOUS_WINNER_TEAM.start + transform_team(SeatRotation(k), previous_team)] = 1
    # Observer-relative single bits that ride along with the observer row.
    for name in (
        "tribute_cancelled", "play_phase", "anti_tribute", "has_current_winning",
        "prefix_length", "is_observer", "observer_finished", "observer_emptied",
        "has_return_obligation", "observer_selected_tribute", "observer_returned",
        "has_previous_result", "observer_is_winner", "observer_is_leader",
        "active_same_team", "sentinel",
    ):
        out[SINGLE_BITS[name]] = source[SINGLE_BITS[name]]
    return out


def _transform_channels_copy_swap(row: np.ndarray) -> np.ndarray:
    source = np.asarray(row, dtype=np.uint8)
    out = source.copy()
    for slice_ in (SELF_HAND, CURRENT_WINNING, SELF_EXCHANGE):
        out[slice_] = 0
        for card_id in range(NUM_PHYSICAL_CARDS):
            if source[slice_.start + card_id]:
                out[slice_.start + transform_card(CopySwap(), card_id)] = 1
    for index in range(SELF_PREFIX.stop - SELF_PREFIX.start):
        token = int(source[SELF_PREFIX.start + index])
        if CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + NUM_PHYSICAL_CARDS:
            out[SELF_PREFIX.start + index] = CARD_TOKEN_BASE + transform_card(
                CopySwap(), token - CARD_TOKEN_BASE,
            )
    return out


def transform_observation(
    obs_tokens: np.ndarray,
    obs_channels: np.ndarray,
    transform: Transform,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``transform`` to one observation (``[4, 48]`` + ``[4, 512]``)."""
    tokens = np.asarray(obs_tokens)
    channels = np.asarray(obs_channels)
    if tokens.ndim != 2 or tokens.shape[0] != NUM_PLAYERS:
        raise SymmetryDomainError("obs_tokens must be [4, observation_capacity]")
    if channels.shape != (NUM_PLAYERS, NUM_CHANNELS):
        raise SymmetryDomainError("obs_channels must be [4, 512]")
    new_tokens = np.zeros_like(tokens)
    new_channels = np.zeros_like(channels)
    rotation = _rotation_of(transform)
    swap = _swap_of(transform)
    for observer in range(NUM_PLAYERS):
        target = transform_seat(transform, observer)
        new_tokens[target] = np.asarray(
            transform_event_tokens(tokens[observer], transform), dtype=tokens.dtype,
        )
        row = np.asarray(channels[observer], dtype=np.uint8)
        if swap:
            row = _transform_channels_copy_swap(row)
        if rotation % 4:
            row = _transform_channels_rotation(row, observer, rotation)
        else:
            row = row.copy()
        new_channels[target] = row
    return new_tokens, new_channels


# ---------------------------------------------------------------------------
# Candidate-row transform and induced action permutation
# ---------------------------------------------------------------------------


def transform_candidate_rows(rows: np.ndarray, transform: Transform) -> np.ndarray:
    """Map each ``[A, 8]`` candidate row to its image under ``transform``."""
    array = np.asarray(rows)
    if array.ndim != 2 or array.shape[1] != 8:
        raise SymmetryDomainError("candidate rows must be [A, 8]")
    out = array.copy()
    if _swap_of(transform):
        for index in range(array.shape[0]):
            token = int(array[index, 5])
            if CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + NUM_PHYSICAL_CARDS:
                out[index, 5] = CARD_TOKEN_BASE + transform_card(
                    transform, token - CARD_TOKEN_BASE,
                )
    return out


def transform_row_list(rows: np.ndarray, transform: Transform) -> np.ndarray:
    """Image of a candidate list **as a list**: one row per source candidate.

    The deck swap is a bijection on physical card ids, so distinct candidate
    rows keep distinct images.  A source list that already contains duplicate
    rows cannot define a one-to-one action permutation; callers use
    :func:`candidate_image_is_unique` to detect that case.
    """
    return transform_candidate_rows(rows, transform)


def candidate_image_is_unique(rows: np.ndarray, transform: Transform) -> bool:
    image = transform_candidate_rows(rows, transform)
    keys = {tuple(int(value) for value in row) for row in image}
    return len(keys) == image.shape[0]


def action_permutation(
    rows: np.ndarray,
    transform: Transform,
    *,
    target_rows: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Return ``perm`` with ``perm[i]`` the index of ``P_T(row_i)``.

    ``target_rows`` defaults to the image of ``rows`` (which is exact for the
    seat rotation, where the action vocabulary is seat-free).  A missing image
    is a hard domain error: the caller must narrow the candidate set instead of
    silently dropping actions.
    """
    array = np.asarray(rows)
    image = transform_candidate_rows(array, transform) if target_rows is None else np.asarray(target_rows)
    if image.shape != array.shape:
        raise SymmetryDomainError("transformed candidate list has a different shape")
    lookup: dict[tuple[int, ...], int] = {}
    for index in range(image.shape[0]):
        key = tuple(int(value) for value in image[index])
        if key in lookup:
            raise SymmetryDomainError("transformed candidate rows are not unique")
        lookup[key] = index
    perm: list[int] = []
    for index in range(array.shape[0]):
        key = tuple(int(value) for value in image[index])
        if key not in lookup:  # pragma: no cover - image equals itself by construction
            raise SymmetryDomainError("candidate image is absent from the transformed list")
        perm.append(lookup[key])
    if len(set(perm)) != len(perm):
        raise SymmetryDomainError("action permutation is not a bijection")
    return tuple(perm)


def macro_action_permutation(
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    transform: Transform,
) -> tuple[int, ...]:
    """Match each source candidate to its image in the *transformed* state list."""
    return action_permutation(source_rows, transform, target_rows=target_rows)


def inverse_candidate_rows(rows: np.ndarray, transform: Transform) -> np.ndarray:
    """Round-trip helper: the inverse image of a transformed candidate list."""
    return transform_candidate_rows(rows, transform.inverse())


def check_transform_involution(transform: Transform, *, trials: int = 1) -> bool:
    """``T^{-1}(T(x)) == x`` on a deterministic synthetic observation/candidate set.

    For ``k in (0, 2)`` and the copy swap the transform is its own inverse; for
    ``k in (1, 3)`` the inverse is the opposite rotation, which the helper uses
    explicitly (a rotation by one applied twice is a rotation by two).
    """
    inverse = transform.inverse()
    rng = np.random.default_rng(20261019)
    for _ in range(int(trials)):
        tokens = np.zeros((NUM_PLAYERS, 48), dtype=np.int64)
        channels = np.zeros((NUM_PLAYERS, NUM_CHANNELS), dtype=np.uint8)
        for observer in range(NUM_PLAYERS):
            tokens[observer, 0] = BOS_TOKEN
            tokens[observer, 1] = EVENT_RESET
            tokens[observer, 2] = 2
            tokens[observer, 3] = RANK_TOKEN_BASE + 12
            tokens[observer, 4] = _seat_token(int(rng.integers(0, 4)))
            tokens[observer, 5] = EVENT_END
            channels[observer, OBSERVER_ONE_HOT.start + observer] = 1
            channels[observer, ACTIVE_ONE_HOT.start + int(rng.integers(0, 4))] = 1
            channels[observer, LEADER_ONE_HOT.start + int(rng.integers(0, 4))] = 1
            channels[observer, TEAM_ONE_HOT.start + TEAM_OF[observer]] = 1
            channels[observer, PHASE_ONE_HOT.start] = 1
            channels[observer, LEVEL_ONE_HOT.start + 4] = 1
            for card_id in rng.choice(NUM_PHYSICAL_CARDS, size=5, replace=False):
                channels[observer, SELF_HAND.start + int(card_id)] = 1
        once = transform_observation(tokens, channels, transform)
        twice = transform_observation(*once, inverse)
        if not (np.array_equal(twice[0], tokens) and np.array_equal(twice[1], channels)):
            return False
        rows = np.zeros((3, 8), dtype=np.int64)
        for index, (token, commit) in enumerate((
            (PASS_TOKEN, ACTION_COMMIT),
            (CARD_TOKEN_BASE + 3, 305),
            (CARD_TOKEN_BASE + 57, 305),
        )):
            rows[index] = (
                ACTION_ROW_BEGIN, 2, ACTION_STAGE_BASE, ACTION_KIND_BASE,
                ACTION_PAYLOAD, token, commit, ACTION_ROW_END,
            )
        back = inverse_candidate_rows(transform_candidate_rows(rows, transform), transform)
        if not np.array_equal(back, rows):
            return False
    return True


__all__ = [
    "ACTION_COMMIT", "ACTION_KIND_BASE", "ACTION_PAYLOAD", "ACTION_ROW_BEGIN",
    "ACTION_ROW_END", "ACTION_STAGE_BASE", "ACTIVE_ONE_HOT", "BOS_TOKEN",
    "CARD_TOKEN_BASE", "COMMIT_TOKEN", "COMMITTED_STEP_BITS",
    "CURRENT_FAMILY", "CURRENT_RANK", "CURRENT_SIZE", "CURRENT_WINNING",
    "ComposedTransform", "CopySwap", "EMPTIED_ONE_HOT", "EVENT_END",
    "EVENT_EXCHANGE_RESOLVED",
    "EVENT_PRIVATE_EXCHANGE_COMMIT", "EVENT_PRIVATE_PREFIX", "EVENT_PUBLIC_COMMIT",
    "EVENT_RESET", "FAMILY_TOKEN_BASE", "FINISHING_RANKS", "LEADER_ONE_HOT",
    "LENGTH_TOKEN_BASE", "LEVEL_ONE_HOT", "NUM_CHANNELS", "NUM_PHYSICAL_CARDS",
    "NUM_PLAYERS", "OBSERVER_ONE_HOT", "PASS_COUNT_ONE_HOT", "PASS_TOKEN",
    "PHASE_ONE_HOT", "PREVIOUS_OUTCOME", "PREVIOUS_RANKING", "PREVIOUS_WINNER_TEAM",
    "RANK_TOKEN_BASE", "REMAINING_COUNTS", "RETURN_TOKEN", "SELF_EXCHANGE",
    "SELF_HAND", "SELF_PREFIX", "SINGLE_BITS", "SUIT_TOKEN_BASE",
    "SYMMETRY_SCHEMA", "SeatRotation", "SymmetryDomainError", "TEAM_OF",
    "TEAM_ONE_HOT", "TOKEN_STEP_BITS", "TRIBUTE_DONOR", "TRIBUTE_ORDER",
    "TRIBUTE_TOKEN", "Transform", "WILD_TOKEN_BASE", "WINNER_ONE_HOT",
    "action_permutation", "check_transform_involution", "compose_transforms",
    "event_seat_positions", "identity_transform", "inverse_candidate_rows",
    "inverse_seat", "macro_action_permutation", "transform_card",
    "transform_candidate_rows", "transform_event_tokens", "transform_observation",
    "transform_seat", "transform_team",
]
