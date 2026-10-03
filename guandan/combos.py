"""Independent GuanDan combination declarations and comparison.

This module implements the ten A01 families from the current project rule
profile.  It intentionally does not implement turns, passing, or a round
engine.  The rules represented here are:

* one single, pair, triple, full house, exactly-five straight, exactly three
  consecutive pairs, exactly two consecutive triples, four-to-ten-card rank
  bomb, exactly-five straight flush, and four kings;
* printed windows use 2,3,...,A in increasing order, plus the special low
  windows A2345, A23, and A2.  JQKA2 and KA2-style wraps are invalid;
* ordinary strength is computed separately from printed rank with
  ``2 < 3 < ... < A < level < small joker < big joker``;
* the two heart cards whose printed rank is the level are wild, but a wild can
  never represent a joker.  A standalone wild is only its natural printed
  card;
* declarations are explicit.  ``recognize_combinations`` returns every valid
  family/window/wild-assignment declaration, so a same-suit five-card run can
  be declared both STRAIGHT and STRAIGHT_FLUSH.  It never silently chooses the
  strongest family;
* ``require_combination(..., kind=None)`` is only a deterministic convenience:
  it chooses the first canonical declaration, not the strongest declaration.

The public ``Combination`` is frozen and retains canonical physical cards,
level context, comparison rank/key, and a mapping of every included wild's
physical ID to its represented rank/suit.  ``natural_wild_ids`` distinguishes
natural use from substitution when the represented rank/suit would otherwise
be identical.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .cards import (
    Card,
    NON_JOKER_RANKS,
    PRINTED_RANKS,
    Rank,
    Suit,
    WINDOW_RANKS,
    card_sort_key,
    effective_rank_value,
    rank_counts,
    sort_cards,
    validate_level_rank,
    validate_physical_cards,
)


class CombinationKind(str, Enum):
    SINGLE = "single"
    PAIR = "pair"
    TRIPLE = "triple"
    FULL_HOUSE = "full_house"
    STRAIGHT = "straight"
    PAIR_SEQUENCE = "pair_sequence"
    TRIPLE_SEQUENCE = "triple_sequence"
    RANK_BOMB = "rank_bomb"
    STRAIGHT_FLUSH = "straight_flush"
    FOUR_KINGS = "four_kings"

    # Compatibility aliases for the provisional API.  The canonical names
    # above are the names used in new declarations and reports.
    TRIPLE_WITH_PAIR = "full_house"
    CONSECUTIVE_PAIRS = "pair_sequence"
    STEEL_PLATE = "triple_sequence"
    BOMB = "rank_bomb"
    KING_BOMB = "four_kings"


class IncomparableCombinationsError(ValueError):
    """Raised when two declarations have no legal comparison relation."""


class LevelContextMismatchError(ValueError):
    """Raised when declarations from different levels are compared."""


_WILD_ASSIGNMENT = tuple[Rank, Suit]
_WILD_ITEMS = tuple[tuple[int, Rank, Suit], ...]


@dataclass(frozen=True, slots=True)
class Combination:
    """An immutable, explicit combination declaration.

    The constructor preserves enough metadata to support serialization and
    adversarial tests, but comparison validates the declaration against the
    recognizer before trusting any derived field.  A hand-built object with a
    forged kind, key, or wild mapping is therefore rejected by comparison.
    """

    kind: CombinationKind
    cards: tuple[Card, ...]
    level_rank: Rank
    comparison_rank: Rank
    comparison_key: tuple[int, ...]
    _wild_assignment_items: _WILD_ITEMS = field(default_factory=tuple, repr=False)
    _natural_wild_ids: tuple[int, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CombinationKind(self.kind))
        object.__setattr__(self, "cards", tuple(self.cards))
        object.__setattr__(self, "level_rank", validate_level_rank(self.level_rank))
        object.__setattr__(self, "comparison_rank", Rank(self.comparison_rank))
        object.__setattr__(self, "comparison_key", tuple(int(v) for v in self.comparison_key))
        items = tuple(
            (int(card_id), Rank(rank), Suit(suit))
            for card_id, rank, suit in self._wild_assignment_items
        )
        object.__setattr__(self, "_wild_assignment_items", items)
        object.__setattr__(self, "_natural_wild_ids", tuple(sorted(int(v) for v in self._natural_wild_ids)))

    @property
    def size(self) -> int:
        return len(self.cards)

    @property
    def wild_assignments(self) -> Mapping[int, _WILD_ASSIGNMENT]:
        """Return an immutable physical-ID to represented-rank/suit mapping."""

        return MappingProxyType({card_id: (rank, suit) for card_id, rank, suit in self._wild_assignment_items})

    @property
    def natural_wild_ids(self) -> tuple[int, ...]:
        """Physical wild IDs used naturally rather than substituted."""

        return self._natural_wild_ids

    @property
    def is_bomb(self) -> bool:
        return self.kind in {CombinationKind.RANK_BOMB, CombinationKind.STRAIGHT_FLUSH, CombinationKind.FOUR_KINGS}

    def physical_ids(self) -> tuple[int, ...]:
        return tuple(card.card_id for card in self.cards)


# Printed windows.  The first family uses exactly five; the other two use the
# minimum family length and may extend through the increasing natural order.
_STRAIGHT_WINDOWS: tuple[tuple[Rank, ...], ...] = tuple(
    tuple(WINDOW_RANKS[index : index + 5]) for index in range(len(WINDOW_RANKS) - 4)
) + ((Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE),)
_PAIR_WINDOWS: tuple[tuple[Rank, ...], ...] = tuple(
    tuple(WINDOW_RANKS[index : index + 3]) for index in range(len(WINDOW_RANKS) - 2)
) + ((Rank.ACE, Rank.TWO, Rank.THREE),)
_TRIPLE_WINDOWS: tuple[tuple[Rank, ...], ...] = tuple(
    tuple(WINDOW_RANKS[index : index + 2]) for index in range(len(WINDOW_RANKS) - 1)
) + ((Rank.ACE, Rank.TWO),)

_KIND_ORDER = {
    CombinationKind.SINGLE: 0,
    CombinationKind.PAIR: 1,
    CombinationKind.TRIPLE: 2,
    CombinationKind.FULL_HOUSE: 3,
    CombinationKind.STRAIGHT: 4,
    CombinationKind.PAIR_SEQUENCE: 5,
    CombinationKind.TRIPLE_SEQUENCE: 6,
    CombinationKind.RANK_BOMB: 7,
    CombinationKind.STRAIGHT_FLUSH: 8,
    CombinationKind.FOUR_KINGS: 9,
}


def _canonical_kind(kind: CombinationKind | str) -> CombinationKind:
    return CombinationKind(kind)


def _is_wild(card: Card, level_rank: Rank) -> bool:
    return card.is_level_wild(level_rank)


def _wild_and_natural(cards: tuple[Card, ...], level_rank: Rank) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    return (
        tuple(card for card in cards if _is_wild(card, level_rank)),
        tuple(card for card in cards if not _is_wild(card, level_rank)),
    )


def _assignment_variants(
    card: Card,
    target_rank: Rank,
    *,
    level_rank: Rank,
    target_suit: Suit | None = None,
) -> tuple[tuple[_WILD_ASSIGNMENT, bool], ...]:
    """Return deterministic natural/substitution choices for one wild."""

    if not _is_wild(card, level_rank):
        raise ValueError("assignment_variants requires a level wild")
    target_rank = Rank(target_rank)
    if target_rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER}:
        return ()
    variants: list[tuple[_WILD_ASSIGNMENT, bool]] = []
    if target_rank is level_rank and (target_suit is None or target_suit is Suit.HEARTS):
        variants.append(((target_rank, Suit.HEARTS), True))
    represented_suit = target_suit if target_suit is not None else Suit.CLUBS
    variants.append(((target_rank, represented_suit), False))
    return tuple(dict.fromkeys(variants))


def _assignment_products(
    wilds: tuple[Card, ...],
    targets: tuple[Rank, ...],
    *,
    level_rank: Rank,
    target_suit: Suit | None = None,
) -> tuple[tuple[_WILD_ITEMS, tuple[int, ...]], ...]:
    if len(wilds) != len(targets):
        return ()
    choices = [
        _assignment_variants(card, target, level_rank=level_rank, target_suit=target_suit)
        for card, target in zip(wilds, targets)
    ]
    if any(not choice for choice in choices):
        return ()
    results: list[tuple[_WILD_ITEMS, tuple[int, ...]]] = []
    for selected in product(*choices):
        items = tuple(
            (card.card_id, assignment[0], assignment[1])
            for card, (assignment, _natural) in zip(wilds, selected)
        )
        natural_ids = tuple(sorted(card.card_id for card, (_assignment, natural) in zip(wilds, selected) if natural))
        results.append((items, natural_ids))
    return tuple(dict.fromkeys(results))


def _ordinary_comparison_key(kind: CombinationKind, size: int, rank: Rank, level_rank: Rank) -> tuple[int, ...]:
    return (0, _KIND_ORDER[kind], size, effective_rank_value(rank, level_rank))


def _bomb_comparison_key(kind: CombinationKind, size: int, rank: Rank, level_rank: Rank) -> tuple[int, ...]:
    if kind is CombinationKind.FOUR_KINGS:
        return (9,)
    if kind is CombinationKind.STRAIGHT_FLUSH:
        return (3, effective_rank_value(rank, level_rank))
    if size == 4:
        bomb_class = 1
    elif size == 5:
        bomb_class = 2
    else:
        bomb_class = size - 2  # 6 -> 4, ..., 10 -> 8
    return (bomb_class, effective_rank_value(rank, level_rank))


def _build(
    kind: CombinationKind,
    cards: tuple[Card, ...],
    *,
    level_rank: Rank,
    comparison_rank: Rank,
    assignments: _WILD_ITEMS = (),
    natural_wild_ids: tuple[int, ...] = (),
) -> Combination:
    kind = _canonical_kind(kind)
    if kind in {CombinationKind.RANK_BOMB, CombinationKind.STRAIGHT_FLUSH, CombinationKind.FOUR_KINGS}:
        key = _bomb_comparison_key(kind, len(cards), comparison_rank, level_rank)
    else:
        key = _ordinary_comparison_key(kind, len(cards), comparison_rank, level_rank)
    return Combination(
        kind=kind,
        cards=cards,
        level_rank=level_rank,
        comparison_rank=comparison_rank,
        comparison_key=key,
        _wild_assignment_items=assignments,
        _natural_wild_ids=natural_wild_ids,
    )


def _group_declarations(
    kind: CombinationKind,
    cards: tuple[Card, ...],
    *,
    level_rank: Rank,
    target_rank: Rank,
    required_count: int,
) -> tuple[Combination, ...]:
    wilds, natural = _wild_and_natural(cards, level_rank)
    if len(cards) != required_count or len(wilds) > 2:
        return ()
    if any(card.is_joker or card.rank is not target_rank for card in natural):
        return ()
    targets = (target_rank,) * len(wilds)
    return tuple(
        _build(kind, cards, level_rank=level_rank, comparison_rank=target_rank, assignments=items, natural_wild_ids=natural_ids)
        for items, natural_ids in _assignment_products(wilds, targets, level_rank=level_rank)
    )


def _sequence_declarations(
    kind: CombinationKind,
    cards: tuple[Card, ...],
    *,
    level_rank: Rank,
    window: tuple[Rank, ...],
    copies_per_rank: int,
    declared_suit: Suit | None = None,
) -> tuple[Combination, ...]:
    wilds, natural = _wild_and_natural(cards, level_rank)
    if len(wilds) > 2 or any(card.is_joker for card in natural):
        return ()
    if declared_suit is not None and any(card.suit is not declared_suit for card in natural):
        return ()
    counts = Counter(card.rank for card in natural)
    if any(rank not in window or count > copies_per_rank for rank, count in counts.items()):
        return ()
    missing: list[Rank] = []
    for rank in window:
        missing.extend([rank] * (copies_per_rank - counts.get(rank, 0)))
    if len(missing) != len(wilds):
        return ()
    declarations: list[Combination] = []
    for assignments, natural_ids in _assignment_products(
        wilds,
        tuple(missing),
        level_rank=level_rank,
        target_suit=declared_suit,
    ):
        declarations.append(
            _build(
                kind,
                cards,
                level_rank=level_rank,
                comparison_rank=window[-1],
                assignments=assignments,
                natural_wild_ids=natural_ids,
            )
        )
    return tuple(declarations)


def _full_house_declarations(cards: tuple[Card, ...], *, level_rank: Rank) -> tuple[Combination, ...]:
    wilds, natural = _wild_and_natural(cards, level_rank)
    if len(cards) != 5 or len(wilds) > 2 or any(card.is_joker for card in natural):
        return ()
    natural_counts = Counter(card.rank for card in natural)
    declarations: list[Combination] = []
    for triple_rank in NON_JOKER_RANKS:
        for pair_rank in NON_JOKER_RANKS:
            if pair_rank is triple_rank:
                continue
            targets: list[Rank] = []
            valid = True
            for rank, count in natural_counts.items():
                if rank not in {triple_rank, pair_rank}:
                    valid = False
                    break
                if count > (3 if rank is triple_rank else 2):
                    valid = False
                    break
            if not valid:
                continue
            targets.extend([triple_rank] * (3 - natural_counts.get(triple_rank, 0)))
            targets.extend([pair_rank] * (2 - natural_counts.get(pair_rank, 0)))
            if len(targets) != len(wilds):
                continue
            for assignments, natural_ids in _assignment_products(
                wilds, tuple(targets), level_rank=level_rank
            ):
                declarations.append(
                    _build(
                        CombinationKind.FULL_HOUSE,
                        cards,
                        level_rank=level_rank,
                        comparison_rank=triple_rank,
                        assignments=assignments,
                        natural_wild_ids=natural_ids,
                    )
                )
    return tuple(declarations)


def _rank_bomb_declarations(cards: tuple[Card, ...], *, level_rank: Rank) -> tuple[Combination, ...]:
    if not 4 <= len(cards) <= 10:
        return ()
    wilds, natural = _wild_and_natural(cards, level_rank)
    if len(wilds) > 2 or any(card.is_joker or card.rank is not natural[0].rank for card in natural[1:]):
        return ()
    if natural and any(card.is_joker for card in natural):
        return ()
    target_ranks = NON_JOKER_RANKS if natural else NON_JOKER_RANKS
    declarations: list[Combination] = []
    for target_rank in target_ranks:
        if any(card.rank is not target_rank for card in natural):
            continue
        if not natural and not wilds:
            continue
        if len(natural) + len(wilds) != len(cards):
            continue
        for assignments, natural_ids in _assignment_products(
            wilds, (target_rank,) * len(wilds), level_rank=level_rank
        ):
            declarations.append(
                _build(
                    CombinationKind.RANK_BOMB,
                    cards,
                    level_rank=level_rank,
                    comparison_rank=target_rank,
                    assignments=assignments,
                    natural_wild_ids=natural_ids,
                )
            )
    return tuple(declarations)


def recognize_combinations(
    cards: Iterable[Card], *, level_rank: Rank = Rank.TWO
) -> tuple[Combination, ...]:
    """Return all valid declarations for the physical cards.

    Invalid physical-card collections return an empty tuple.  Results are
    canonical and deterministic; the function does not choose a strongest
    family when declarations overlap.
    """

    try:
        level_rank = validate_level_rank(level_rank)
        materialized = validate_physical_cards(cards)
    except (TypeError, ValueError):
        return ()
    if not materialized:
        return ()
    canonical = sort_cards(materialized)
    size = len(canonical)
    declarations: list[Combination] = []

    # Single wilds are natural-only; no substituted single is generated.
    if size == 1:
        card = canonical[0]
        if _is_wild(card, level_rank):
            assignments = ((card.card_id, card.rank, card.suit),)
            declarations.append(
                _build(
                    CombinationKind.SINGLE,
                    canonical,
                    level_rank=level_rank,
                    comparison_rank=card.rank,
                    assignments=assignments,
                    natural_wild_ids=(card.card_id,),
                )
            )
        else:
            declarations.append(
                _build(CombinationKind.SINGLE, canonical, level_rank=level_rank, comparison_rank=card.rank)
            )

    # Pair: same natural joker type is valid; mixed jokers are not.
    if size == 2:
        if all(card.rank is Rank.SMALL_JOKER for card in canonical) or all(card.rank is Rank.BIG_JOKER for card in canonical):
            declarations.append(_build(CombinationKind.PAIR, canonical, level_rank=level_rank, comparison_rank=canonical[0].rank))
        else:
            for rank in NON_JOKER_RANKS:
                declarations.extend(_group_declarations(CombinationKind.PAIR, canonical, level_rank=level_rank, target_rank=rank, required_count=2))

    if size == 3:
        for rank in NON_JOKER_RANKS:
            declarations.extend(_group_declarations(CombinationKind.TRIPLE, canonical, level_rank=level_rank, target_rank=rank, required_count=3))

    if size == 5:
        declarations.extend(_full_house_declarations(canonical, level_rank=level_rank))

    # Exactly four physical jokers: two small and two big.
    if size == 4:
        counts = Counter(card.rank for card in canonical)
        if counts == Counter({Rank.SMALL_JOKER: 2, Rank.BIG_JOKER: 2}):
            declarations.append(
                _build(
                    CombinationKind.FOUR_KINGS,
                    canonical,
                    level_rank=level_rank,
                    comparison_rank=Rank.BIG_JOKER,
                )
            )

    if 4 <= size <= 10:
        declarations.extend(_rank_bomb_declarations(canonical, level_rank=level_rank))

    if size == 5:
        for window in _STRAIGHT_WINDOWS:
            declarations.extend(
                _sequence_declarations(
                    CombinationKind.STRAIGHT,
                    canonical,
                    level_rank=level_rank,
                    window=window,
                    copies_per_rank=1,
                )
            )
            for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES):
                declarations.extend(
                    _sequence_declarations(
                        CombinationKind.STRAIGHT_FLUSH,
                        canonical,
                        level_rank=level_rank,
                        window=window,
                        copies_per_rank=1,
                        declared_suit=suit,
                    )
                )

    if size == 6:
        pair_count = 3
        if pair_count == 3:
            for start in range(len(WINDOW_RANKS) - pair_count + 1):
                window = tuple(WINDOW_RANKS[start : start + pair_count])
                declarations.extend(
                    _sequence_declarations(
                        CombinationKind.PAIR_SEQUENCE,
                        canonical,
                        level_rank=level_rank,
                        window=window,
                        copies_per_rank=2,
                    )
                )
            if pair_count == 3:
                declarations.extend(
                    _sequence_declarations(
                        CombinationKind.PAIR_SEQUENCE,
                        canonical,
                        level_rank=level_rank,
                        window=(Rank.ACE, Rank.TWO, Rank.THREE),
                        copies_per_rank=2,
                    )
                )

    if size == 6:
        triple_count = 2
        if triple_count == 2:
            for start in range(len(WINDOW_RANKS) - triple_count + 1):
                window = tuple(WINDOW_RANKS[start : start + triple_count])
                declarations.extend(
                    _sequence_declarations(
                        CombinationKind.TRIPLE_SEQUENCE,
                        canonical,
                        level_rank=level_rank,
                        window=window,
                        copies_per_rank=3,
                    )
                )
            if triple_count == 2:
                declarations.extend(
                    _sequence_declarations(
                        CombinationKind.TRIPLE_SEQUENCE,
                        canonical,
                        level_rank=level_rank,
                        window=(Rank.ACE, Rank.TWO),
                        copies_per_rank=3,
                    )
                )

    # Deduplicate only exact declarations; distinct families/windows remain.
    unique: dict[tuple[object, ...], Combination] = {}
    for declaration in declarations:
        key = (
            declaration.kind,
            declaration.physical_ids(),
            declaration.level_rank,
            declaration.comparison_rank,
            declaration._wild_assignment_items,
            declaration.natural_wild_ids,
        )
        unique[key] = declaration
    return tuple(sorted(unique.values(), key=_declaration_sort_key))


def _declaration_sort_key(declaration: Combination) -> tuple[object, ...]:
    return (
        _KIND_ORDER[declaration.kind],
        declaration.comparison_key,
        declaration.physical_ids(),
        declaration._wild_assignment_items,
        declaration.natural_wild_ids,
    )


def require_combination(
    cards: Iterable[Card], *, level_rank: Rank = Rank.TWO, kind: CombinationKind | str | None = None
) -> Combination:
    """Return one deterministic declaration or raise ``ValueError``.

    Passing ``kind`` is the recommended form when physical cards admit more
    than one family.  With ``kind=None`` the first canonical declaration is
    returned; this is not a strength-based selection.
    """

    declarations = recognize_combinations(cards, level_rank=level_rank)
    if kind is not None:
        requested = _canonical_kind(kind)
        declarations = tuple(declaration for declaration in declarations if declaration.kind is requested)
    if not declarations:
        raise ValueError("cards do not form the requested supported GuanDan declaration")
    return declarations[0]


def recognize_combination(
    cards: Iterable[Card], *, level_rank: Rank = Rank.TWO, kind: CombinationKind | str | None = None
) -> Combination | None:
    """Compatibility helper returning the first declaration, if any."""

    try:
        return require_combination(cards, level_rank=level_rank, kind=kind)
    except ValueError:
        return None


def _validate_declared_combination(combination: Combination) -> None:
    if not isinstance(combination, Combination):
        raise TypeError("expected Combination")
    candidates = recognize_combinations(combination.cards, level_rank=combination.level_rank)
    if not any(
        candidate.kind is combination.kind
        and candidate.comparison_rank is combination.comparison_rank
        and candidate.comparison_key == combination.comparison_key
        and candidate._wild_assignment_items == combination._wild_assignment_items
        and candidate.natural_wild_ids == combination.natural_wild_ids
        for candidate in candidates
    ):
        raise ValueError("combination declaration is not valid for its physical cards and level")


def compare_combinations(left: Combination, right: Combination) -> int:
    """Compare declarations and return -1, 0, or 1.

    Level contexts must match.  Ordinary families require the same family and
    same card count.  Bomb ordering is rank-bomb(4), rank-bomb(5), straight
    flush, rank-bomb(6..10), four kings.
    """

    _validate_declared_combination(left)
    _validate_declared_combination(right)
    if left.level_rank is not right.level_rank:
        raise LevelContextMismatchError("combinations use different level contexts")
    left_bomb = left.kind in {CombinationKind.RANK_BOMB, CombinationKind.STRAIGHT_FLUSH, CombinationKind.FOUR_KINGS}
    right_bomb = right.kind in {CombinationKind.RANK_BOMB, CombinationKind.STRAIGHT_FLUSH, CombinationKind.FOUR_KINGS}
    if left_bomb or right_bomb:
        if not left_bomb or not right_bomb:
            return (left_bomb > right_bomb) - (left_bomb < right_bomb)
        return (left.comparison_key > right.comparison_key) - (left.comparison_key < right.comparison_key)
    if left.kind is not right.kind or left.size != right.size:
        raise IncomparableCombinationsError(
            f"{left.kind.value}/{left.size} cannot be compared with {right.kind.value}/{right.size}"
        )
    left_value = effective_rank_value(left.comparison_rank, left.level_rank)
    right_value = effective_rank_value(right.comparison_rank, right.level_rank)
    return (left_value > right_value) - (left_value < right_value)


def can_beat(candidate: Combination, current: Combination) -> bool:
    """Return whether ``candidate`` strictly beats ``current``."""

    try:
        return compare_combinations(candidate, current) > 0
    except IncomparableCombinationsError:
        return False


__all__ = [
    "Combination",
    "CombinationKind",
    "IncomparableCombinationsError",
    "LevelContextMismatchError",
    "can_beat",
    "compare_combinations",
    "recognize_combination",
    "recognize_combinations",
    "require_combination",
]
