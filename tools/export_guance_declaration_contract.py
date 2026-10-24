#!/usr/bin/env python3
"""Export the fixed Guance Phase-2 complex-declaration contract.

This exporter is intentionally source-bound. It imports the current Python
rules/reference environment and the loaded native MARVEL backend, generates
only synthetic card fixtures, checks Python/C++/environment/token parity, and
writes a hash-verifiable handoff. It never reads runs, checkpoints, captures,
training data, or the Guance repository.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan import action_state as action_protocol  # noqa: E402
from guandan.action_state import (  # noqa: E402
    CARD_TOKEN_BASE, COMMIT_TOKEN, FAMILY_ORDER, FAMILY_TOKEN_BASE,
    LENGTH_TOKEN_BASE, RANK_ORDER, RANK_TOKEN_BASE, SUIT_TOKEN_BASE,
    WILD_TOKEN_BASE, StepwiseActionState, TokenCodec, canonical_tokens,
)
from guandan.cards import (  # noqa: E402
    ALL_CARDS, PRINTED_RANKS, WINDOW_RANKS, Card, Rank, Suit,
)
from guandan.combos import Combination, CombinationKind, recognize_combinations  # noqa: E402
from guandan.environment import GuandanEnv  # noqa: E402
from guandan.environment_types import GameConfig, RULES_VERSION  # noqa: E402
from guandan.round import apply_action, enumerate_plays  # noqa: E402
from guandan.state import CommittedAction, IllegalActionError, RoundState  # noqa: E402
from guandan_marvel import game as marvel_impl  # noqa: E402
from guandan_marvel import python_game  # noqa: E402
from nanopokerlib import guandan as cpp_game  # noqa: E402

OUTPUT_NAMES = (
    "guandan_declaration_rules_v1.json",
    "guandan_declaration_golden_vectors_v1.json",
    "guandan_declaration_rules_v1.md",
)
DEFAULT_OUTPUT_DIR = ROOT / "guance_phase2_declaration_handoff"
BASE_VECTOR_LEVEL = Rank.FIVE
LEGAL_LEVELS: tuple[Rank, ...] = tuple(PRINTED_RANKS)
INVALID_LEVELS: tuple[Rank, ...] = (Rank.SMALL_JOKER, Rank.BIG_JOKER)
FAMILIES = (
    CombinationKind.FULL_HOUSE, CombinationKind.STRAIGHT,
    CombinationKind.STRAIGHT_FLUSH, CombinationKind.PAIR_SEQUENCE,
    CombinationKind.TRIPLE_SEQUENCE, CombinationKind.FOUR_KINGS,
)
AUTHORITATIVE_SOURCES = (
    "guandan/cards.py", "guandan/combos.py", "guandan/round.py",
    "guandan/state.py", "guandan/action_state.py", "guandan/environment.py",
    "guandan/environment_types.py", "guandan_marvel/game.py",
    "guandan_marvel/python_game.py", "guandan_marvel/cpp_game.py",
    "nanopokerlib/guandan.py", "nanopokerlib/build_identity.py",
    "nanopokerlib/cpp/core/types.hpp", "nanopokerlib/cpp/guandan/guandan.hpp",
    "nanopokerlib/cpp/guandan/guandan.cpp", "nanopokerlib/cpp/bindings.cpp",
)
GENERATOR_PATHS = (
    "guandan/round.py::enumerate_plays",
    "guandan/round.py::enumerate_legal_actions",
    "nanopokerlib/cpp/guandan/guandan.cpp::enumerate_plays",
)
RECOGNIZER_PATHS = (
    "guandan/combos.py::recognize_combinations",
    "nanopokerlib/cpp/guandan/guandan.cpp::recognize_combinations",
)
SERIALIZER_PATHS = (
    "guandan/action_state.py::_declaration_fields",
    "guandan/action_state.py::_ordered_action_cards",
    "guandan/action_state.py::canonical_tokens",
    "nanopokerlib/cpp/guandan/guandan.cpp::declaration_ranks",
    "nanopokerlib/cpp/guandan/guandan.cpp::declaration_length",
    "nanopokerlib/cpp/guandan/guandan.cpp::declaration_suit",
    "nanopokerlib/cpp/guandan/guandan.cpp::ordered_action_cards",
    "nanopokerlib/cpp/guandan/guandan.cpp::canonical_tokens",
)
VALIDATOR_PATHS = (
    "guandan/round.py::_valid", "guandan/round.py::_play",
    "guandan/action_state.py::StepwiseActionState.step",
    "guandan/environment.py::GuandanEnv.step",
    "nanopokerlib/cpp/guandan/guandan.cpp::Game::step",
)
ENCODER_PATHS = (
    "guandan_marvel/game.py::_action_row",
    "guandan_marvel/game.py::_parse_action_row",
    "nanopokerlib/cpp/guandan/guandan.cpp::action_row",
    "nanopokerlib/cpp/bindings.cpp::actions_from_np",
)


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _semantic_digest(value: dict[str, Any]) -> str:
    normalized = copy.deepcopy(value)
    normalized.setdefault("integrity", {})["semantic_sha256"] = None
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _now_china() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def _rank(value: Rank | int) -> dict[str, Any]:
    rank = Rank(value)
    return {"name": rank.name, "value": int(rank)}


def _suit(value: Suit | int | None) -> dict[str, Any] | None:
    if value is None:
        return None
    suit = Suit(value)
    return {"name": suit.name, "value": int(suit)}


def _card_id(rank: Rank, suit: Suit, deck: int = 0) -> int:
    if rank in (Rank.SMALL_JOKER, Rank.BIG_JOKER):
        if suit is not Suit.JOKER:
            raise ValueError("joker cards must use Suit.JOKER")
        return deck * 54 + (52 if rank is Rank.SMALL_JOKER else 53)
    return deck * 54 + PRINTED_RANKS.index(rank) * 4 + int(suit)


def C(rank: Rank, suit: Suit, deck: int = 0) -> int:
    return _card_id(rank, suit, deck)


def _card_record(card_id: int, level_rank: Rank = BASE_VECTOR_LEVEL) -> dict[str, Any]:
    card = Card(card_id)
    return {
        "card_id": card.card_id, "deck_index": card.deck_index,
        "printed_rank": _rank(card.rank), "physical_suit": _suit(card.suit),
        "is_level_wild": card.is_level_wild(level_rank),
    }


@dataclass(frozen=True)
class PositiveSpec:
    vector_id: str
    family: CombinationKind
    card_ids: tuple[int, ...]
    coverage: tuple[str, ...]
    comparison_rank: Rank | None = None
    assignments: tuple[tuple[int, Rank, Suit, str], ...] = ()


def _positive_specs() -> tuple[PositiveSpec, ...]:
    H5_0 = C(Rank.FIVE, Suit.HEARTS, 0)
    H5_1 = C(Rank.FIVE, Suit.HEARTS, 1)
    return (
        PositiveSpec("full_house_natural", CombinationKind.FULL_HOUSE,
            (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS)),
            ("natural", "physical_length_min", "physical_length_max")),
        PositiveSpec("full_house_wild_natural", CombinationKind.FULL_HOUSE,
            (C(Rank.FIVE, Suit.CLUBS), C(Rank.FIVE, Suit.DIAMONDS), H5_0, C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS)),
            ("wildcard_natural",), Rank.FIVE, ((H5_0, Rank.FIVE, Suit.HEARTS, "natural"),)),
        PositiveSpec("full_house_wild_substituted_triple", CombinationKind.FULL_HOUSE,
            (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), H5_0, C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS)),
            ("wildcard_substituted_triple",), Rank.SEVEN, ((H5_0, Rank.SEVEN, Suit.CLUBS, "substituted"),)),
        PositiveSpec("full_house_wild_substituted_pair", CombinationKind.FULL_HOUSE,
            (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.CLUBS), H5_0),
            ("wildcard_substituted_pair",), Rank.SEVEN, ((H5_0, Rank.NINE, Suit.CLUBS, "substituted"),)),
        PositiveSpec("full_house_two_wildcards", CombinationKind.FULL_HOUSE,
            (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.NINE, Suit.CLUBS), H5_0, H5_1),
            ("wildcard_max_count", "wildcard_two_targets"), Rank.SEVEN,
            ((H5_0, Rank.SEVEN, Suit.CLUBS, "substituted"), (H5_1, Rank.NINE, Suit.CLUBS, "substituted"))),

        PositiveSpec("straight_natural", CombinationKind.STRAIGHT,
            (C(Rank.SIX, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.EIGHT, Suit.HEARTS), C(Rank.NINE, Suit.SPADES), C(Rank.TEN, Suit.CLUBS)),
            ("natural", "physical_length_min", "physical_length_max")),
        PositiveSpec("straight_lowest_window_a2345", CombinationKind.STRAIGHT,
            (C(Rank.ACE, Suit.CLUBS), C(Rank.TWO, Suit.DIAMONDS), C(Rank.THREE, Suit.HEARTS), C(Rank.FOUR, Suit.SPADES), C(Rank.FIVE, Suit.CLUBS)),
            ("lowest_rank_window", "special_ace_low_window"), Rank.FIVE),
        PositiveSpec("straight_highest_window_10jqka", CombinationKind.STRAIGHT,
            (C(Rank.TEN, Suit.CLUBS), C(Rank.JACK, Suit.DIAMONDS), C(Rank.QUEEN, Suit.HEARTS), C(Rank.KING, Suit.SPADES), C(Rank.ACE, Suit.CLUBS)),
            ("highest_rank_window",), Rank.ACE),
        PositiveSpec("straight_wild_natural", CombinationKind.STRAIGHT,
            (C(Rank.THREE, Suit.CLUBS), C(Rank.FOUR, Suit.DIAMONDS), H5_0, C(Rank.SIX, Suit.SPADES), C(Rank.SEVEN, Suit.CLUBS)),
            ("wildcard_natural",), Rank.SEVEN, ((H5_0, Rank.FIVE, Suit.HEARTS, "natural"),)),
        PositiveSpec("straight_wild_substituted", CombinationKind.STRAIGHT,
            (C(Rank.THREE, Suit.CLUBS), C(Rank.FIVE, Suit.CLUBS), C(Rank.SIX, Suit.DIAMONDS), C(Rank.SEVEN, Suit.SPADES), H5_0),
            ("wildcard_substituted",), Rank.SEVEN, ((H5_0, Rank.FOUR, Suit.CLUBS, "substituted"),)),
        PositiveSpec("straight_two_wildcards", CombinationKind.STRAIGHT,
            (C(Rank.THREE, Suit.CLUBS), C(Rank.FIVE, Suit.DIAMONDS), C(Rank.SEVEN, Suit.SPADES), H5_0, H5_1),
            ("wildcard_max_count",), Rank.SEVEN,
            ((H5_0, Rank.FOUR, Suit.CLUBS, "substituted"), (H5_1, Rank.SIX, Suit.CLUBS, "substituted"))),

        PositiveSpec("straight_flush_natural", CombinationKind.STRAIGHT_FLUSH,
            tuple(C(rank, Suit.CLUBS) for rank in (Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN)),
            ("natural", "physical_length_min", "physical_length_max")),
        PositiveSpec("straight_flush_lowest_window_a2345", CombinationKind.STRAIGHT_FLUSH,
            tuple(C(rank, Suit.CLUBS) for rank in (Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE)),
            ("lowest_rank_window", "special_ace_low_window"), Rank.FIVE),
        PositiveSpec("straight_flush_highest_window_10jqka", CombinationKind.STRAIGHT_FLUSH,
            tuple(C(rank, Suit.CLUBS) for rank in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE)),
            ("highest_rank_window",), Rank.ACE),
        PositiveSpec("straight_flush_wild_natural_hearts", CombinationKind.STRAIGHT_FLUSH,
            tuple(C(rank, Suit.HEARTS) for rank in (Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE)),
            ("wildcard_natural", "wildcard_target_suit_hearts"), Rank.FIVE,
            ((H5_0, Rank.FIVE, Suit.HEARTS, "natural"),)),
        PositiveSpec("straight_flush_wild_substituted_clubs", CombinationKind.STRAIGHT_FLUSH,
            (C(Rank.THREE, Suit.CLUBS), C(Rank.FIVE, Suit.CLUBS), C(Rank.SIX, Suit.CLUBS), C(Rank.SEVEN, Suit.CLUBS), H5_0),
            ("wildcard_substituted", "wildcard_target_suit_clubs"), Rank.SEVEN,
            ((H5_0, Rank.FOUR, Suit.CLUBS, "substituted"),)),
        PositiveSpec("straight_flush_wild_substituted_diamonds", CombinationKind.STRAIGHT_FLUSH,
            (C(Rank.THREE, Suit.DIAMONDS), C(Rank.FIVE, Suit.DIAMONDS), C(Rank.SIX, Suit.DIAMONDS), C(Rank.SEVEN, Suit.DIAMONDS), H5_0),
            ("wildcard_target_suit_diamonds",), Rank.SEVEN,
            ((H5_0, Rank.FOUR, Suit.DIAMONDS, "substituted"),)),
        PositiveSpec("straight_flush_wild_substituted_spades", CombinationKind.STRAIGHT_FLUSH,
            (C(Rank.THREE, Suit.SPADES), C(Rank.FIVE, Suit.SPADES), C(Rank.SIX, Suit.SPADES), C(Rank.SEVEN, Suit.SPADES), H5_0),
            ("wildcard_target_suit_spades",), Rank.SEVEN,
            ((H5_0, Rank.FOUR, Suit.SPADES, "substituted"),)),
        PositiveSpec("straight_flush_two_wildcards_hearts", CombinationKind.STRAIGHT_FLUSH,
            (C(Rank.THREE, Suit.HEARTS), C(Rank.SIX, Suit.HEARTS), C(Rank.SEVEN, Suit.HEARTS), H5_0, H5_1),
            ("wildcard_max_count", "wildcard_substituted_hearts", "wildcard_natural_and_substituted"), Rank.SEVEN,
            ((H5_0, Rank.FOUR, Suit.HEARTS, "substituted"), (H5_1, Rank.FIVE, Suit.HEARTS, "natural"))),

        PositiveSpec("pair_sequence_natural", CombinationKind.PAIR_SEQUENCE,
            (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.EIGHT, Suit.CLUBS), C(Rank.EIGHT, Suit.DIAMONDS), C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS)),
            ("natural", "physical_length_min", "physical_length_max")),
        PositiveSpec("pair_sequence_lowest_window_a23", CombinationKind.PAIR_SEQUENCE,
            tuple(C(rank, suit) for rank in (Rank.ACE, Rank.TWO, Rank.THREE) for suit in (Suit.CLUBS, Suit.DIAMONDS)),
            ("lowest_rank_window", "special_ace_low_window"), Rank.THREE),
        PositiveSpec("pair_sequence_highest_window_qka", CombinationKind.PAIR_SEQUENCE,
            tuple(C(rank, suit) for rank in (Rank.QUEEN, Rank.KING, Rank.ACE) for suit in (Suit.CLUBS, Suit.DIAMONDS)),
            ("highest_rank_window",), Rank.ACE),
        PositiveSpec("pair_sequence_wild_natural", CombinationKind.PAIR_SEQUENCE,
            (C(Rank.FOUR, Suit.CLUBS), C(Rank.FOUR, Suit.DIAMONDS), C(Rank.FIVE, Suit.CLUBS), H5_0, C(Rank.SIX, Suit.CLUBS), C(Rank.SIX, Suit.DIAMONDS)),
            ("wildcard_natural",), Rank.SIX, ((H5_0, Rank.FIVE, Suit.HEARTS, "natural"),)),
        PositiveSpec("pair_sequence_wild_substituted", CombinationKind.PAIR_SEQUENCE,
            (C(Rank.SIX, Suit.CLUBS), C(Rank.SIX, Suit.DIAMONDS), C(Rank.SEVEN, Suit.CLUBS), H5_0, C(Rank.EIGHT, Suit.CLUBS), C(Rank.EIGHT, Suit.DIAMONDS)),
            ("wildcard_substituted",), Rank.EIGHT, ((H5_0, Rank.SEVEN, Suit.CLUBS, "substituted"),)),
        PositiveSpec("pair_sequence_two_wildcards", CombinationKind.PAIR_SEQUENCE,
            (C(Rank.SIX, Suit.CLUBS), C(Rank.SIX, Suit.DIAMONDS), C(Rank.SEVEN, Suit.CLUBS), C(Rank.EIGHT, Suit.CLUBS), H5_0, H5_1),
            ("wildcard_max_count",), Rank.EIGHT,
            ((H5_0, Rank.SEVEN, Suit.CLUBS, "substituted"), (H5_1, Rank.EIGHT, Suit.CLUBS, "substituted"))),

        PositiveSpec("triple_sequence_natural", CombinationKind.TRIPLE_SEQUENCE,
            (C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS), C(Rank.NINE, Suit.HEARTS), C(Rank.TEN, Suit.CLUBS), C(Rank.TEN, Suit.DIAMONDS), C(Rank.TEN, Suit.SPADES)),
            ("natural", "physical_length_min", "physical_length_max")),
        PositiveSpec("triple_sequence_lowest_window_a2", CombinationKind.TRIPLE_SEQUENCE,
            tuple(C(rank, suit) for rank in (Rank.ACE, Rank.TWO) for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES)),
            ("lowest_rank_window", "special_ace_low_window"), Rank.TWO),
        PositiveSpec("triple_sequence_highest_window_ka", CombinationKind.TRIPLE_SEQUENCE,
            tuple(C(rank, suit) for rank in (Rank.KING, Rank.ACE) for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES)),
            ("highest_rank_window",), Rank.ACE),
        PositiveSpec("triple_sequence_wild_natural", CombinationKind.TRIPLE_SEQUENCE,
            (C(Rank.FIVE, Suit.CLUBS), C(Rank.FIVE, Suit.DIAMONDS), H5_0, C(Rank.SIX, Suit.CLUBS), C(Rank.SIX, Suit.DIAMONDS), C(Rank.SIX, Suit.SPADES)),
            ("wildcard_natural",), Rank.SIX, ((H5_0, Rank.FIVE, Suit.HEARTS, "natural"),)),
        PositiveSpec("triple_sequence_wild_substituted", CombinationKind.TRIPLE_SEQUENCE,
            (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), H5_0, C(Rank.EIGHT, Suit.CLUBS), C(Rank.EIGHT, Suit.DIAMONDS), C(Rank.EIGHT, Suit.SPADES)),
            ("wildcard_substituted",), Rank.EIGHT, ((H5_0, Rank.SEVEN, Suit.CLUBS, "substituted"),)),
        PositiveSpec("triple_sequence_two_wildcards", CombinationKind.TRIPLE_SEQUENCE,
            (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.EIGHT, Suit.CLUBS), C(Rank.EIGHT, Suit.DIAMONDS), H5_0, H5_1),
            ("wildcard_max_count",), Rank.EIGHT,
            ((H5_0, Rank.SEVEN, Suit.CLUBS, "substituted"), (H5_1, Rank.EIGHT, Suit.CLUBS, "substituted"))),

        PositiveSpec("four_kings_natural", CombinationKind.FOUR_KINGS,
            (C(Rank.SMALL_JOKER, Suit.JOKER, 0), C(Rank.SMALL_JOKER, Suit.JOKER, 1), C(Rank.BIG_JOKER, Suit.JOKER, 0), C(Rank.BIG_JOKER, Suit.JOKER, 1)),
            ("natural", "physical_length_min", "physical_length_max", "no_wildcards"), Rank.BIG_JOKER),
    )


def _python_decl_key(declaration: Combination) -> tuple[Any, ...]:
    return (
        declaration.kind.value, declaration.physical_ids(), int(declaration.level_rank),
        int(declaration.comparison_rank), tuple(declaration.comparison_key),
        tuple((card_id, int(rank), int(suit), card_id in declaration.natural_wild_ids)
              for card_id, (rank, suit) in sorted(declaration.wild_assignments.items())),
        tuple(declaration.natural_wild_ids),
    )


def _cpp_decl_key(raw: Any) -> tuple[Any, ...]:
    value = raw.to_dict()
    return (
        value["kind"], tuple(value["card_ids"]), int(value["level_rank"]),
        int(value["comparison_rank"]), tuple(value["comparison_key"]),
        tuple(tuple(item) for item in value["wild_assignments"]),
        tuple(value["natural_wild_ids"]),
    )


def _choose_declaration(spec: PositiveSpec, level_rank: Rank = BASE_VECTOR_LEVEL) -> tuple[Combination, Any]:
    cards = tuple(Card(card_id) for card_id in spec.card_ids)
    candidates = [item for item in recognize_combinations(cards, level_rank=level_rank) if item.kind is spec.family]
    expected_assignments = tuple((card_id, int(rank), int(suit), mode == "natural")
                                 for card_id, rank, suit, mode in spec.assignments)
    filtered = []
    for item in candidates:
        actual = tuple((card_id, int(rank), int(suit), card_id in item.natural_wild_ids)
                       for card_id, (rank, suit) in sorted(item.wild_assignments.items()))
        if expected_assignments and actual != expected_assignments:
            continue
        if not expected_assignments and actual:
            continue
        if spec.comparison_rank is not None and item.comparison_rank is not spec.comparison_rank:
            continue
        filtered.append(item)
    if len(filtered) != 1:
        raise AssertionError(f"{spec.vector_id}: expected one Python declaration, found {len(filtered)}")
    declaration = filtered[0]
    matches = [item for item in cpp_game.recognize_combinations(list(spec.card_ids), int(level_rank))
               if _cpp_decl_key(item) == _python_decl_key(declaration)]
    if len(matches) != 1:
        raise AssertionError(f"{spec.vector_id}: native recognizer parity failed")
    return declaration, matches[0]


def _synthetic_state(card_ids: Sequence[int], level_rank: Rank = BASE_VECTOR_LEVEL) -> RoundState:
    selected = set(int(value) for value in card_ids)
    if len(selected) != len(card_ids):
        raise ValueError("synthetic vector repeats a physical card")
    hands: list[list[Card]] = [[Card(value) for value in card_ids], [], [], []]
    for index, card in enumerate(card for card in ALL_CARDS if card.card_id not in selected):
        hands[1 + index % 3].append(card)
    return RoundState.from_hands(hands, level_rank=level_rank, leader_seat=0)


def _python_batch(state: RoundState):
    env = GuandanEnv(GameConfig(level_rank=int(state.level_rank), leader_seat=0, max_token_steps=4096))
    env.reset(initial_state=StepwiseActionState(state).serialize())
    return python_game.EnvBatch.from_envs([env], auto_reset=False)


def _cpp_batch(state: RoundState):
    batch = cpp_game.EnvBatch(1, auto_reset=False)
    step = batch.reset_from_deals(
        [[[card.card_id for card in hand] for hand in state.hands]],
        level_ranks=[int(state.level_rank)], leader_seats=[0],
    )
    return batch, step


def _row_for_token(step: Any, token_id: int) -> np.ndarray | None:
    live = step.available_actions.weights[0] > 0
    matches = np.flatnonzero(live & (step.available_actions.tokens[0, :, 5] == int(token_id)))
    if len(matches) == 0:
        return None
    if len(matches) != 1:
        raise AssertionError(f"token {token_id} has {len(matches)} current rows")
    return np.asarray(step.available_actions.tokens[0, matches[0]], dtype=np.int64)


def _normalize_declaration(declaration: Combination) -> dict[str, Any]:
    rank_tokens, length, suit = action_protocol._declaration_fields(declaration)
    action = CommittedAction(0, declaration.kind, declaration.cards, declaration)
    ordered = action_protocol._ordered_action_cards(action)
    assignments = []
    for card_id, (rank, represented_suit) in sorted(declaration.wild_assignments.items()):
        assignments.append({
            "card_id": int(card_id), "target_rank": _rank(rank),
            "target_suit": _suit(represented_suit),
            "mode": "natural" if card_id in declaration.natural_wild_ids else "substituted",
            "token": TokenCodec.wild(rank, represented_suit),
        })
    return {
        "family": declaration.kind.value.upper(), "family_wire_value": declaration.kind.value,
        "level_rank": _rank(declaration.level_rank),
        "declaration_card_ids": list(declaration.physical_ids()),
        "canonical_action_card_ids": [card.card_id for card in ordered],
        "comparison_rank": _rank(declaration.comparison_rank),
        "comparison_key": list(declaration.comparison_key),
        "declared_ranks": [_rank(TokenCodec.rank_from_token(token)) for token in rank_tokens],
        "declared_rank_count": len(rank_tokens), "declared_suit": _suit(suit),
        "declared_length": length, "wild_assignments": assignments,
        "natural_wild_ids": list(declaration.natural_wild_ids),
    }


def _token_record(token: int, level_rank: Rank = BASE_VECTOR_LEVEL) -> dict[str, Any]:
    token = int(token)
    if token == COMMIT_TOKEN:
        return {"token": token, "kind": "COMMIT", "value": None}
    if FAMILY_TOKEN_BASE <= token < FAMILY_TOKEN_BASE + len(FAMILY_ORDER):
        return {"token": token, "kind": "FAMILY", "value": TokenCodec.family_from_token(token).value.upper()}
    if RANK_TOKEN_BASE <= token < RANK_TOKEN_BASE + len(RANK_ORDER):
        return {"token": token, "kind": "RANK", "value": _rank(TokenCodec.rank_from_token(token))}
    if SUIT_TOKEN_BASE <= token <= SUIT_TOKEN_BASE + int(Suit.JOKER):
        return {"token": token, "kind": "SUIT", "value": _suit(TokenCodec.suit_from_token(token))}
    if LENGTH_TOKEN_BASE <= token <= LENGTH_TOKEN_BASE + 10:
        return {"token": token, "kind": "LENGTH", "value": TokenCodec.length_from_token(token)}
    if CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + 108:
        return {"token": token, "kind": "CARD", "value": _card_record(TokenCodec.card_from_token(token), level_rank)}
    if WILD_TOKEN_BASE <= token < WILD_TOKEN_BASE + 52:
        rank, suit = TokenCodec.wild_from_token(token)
        return {"token": token, "kind": "WILD_ASSIGNMENT", "value": {"rank": _rank(rank), "suit": _suit(suit)}}
    return {"token": token, "kind": "OTHER", "value": None}


def _execute_positive(declaration: Combination, cpp_declaration: Any) -> dict[str, Any]:
    recognizer_equal = _cpp_decl_key(cpp_declaration) == _python_decl_key(declaration)
    action = CommittedAction(0, declaration.kind, declaration.cards, declaration)
    tokens = tuple(int(value) for value in canonical_tokens(action))
    cpp_tokens = tuple(int(value) for value in cpp_game.canonical_tokens(cpp_declaration))
    tokens_equal = tokens == cpp_tokens
    generated = enumerate_plays(tuple(Card(value) for value in declaration.physical_ids()), level_rank=declaration.level_rank)
    generator_contains = _python_decl_key(declaration) in {_python_decl_key(item) for item in generated}
    if not (recognizer_equal and tokens_equal and generator_contains):
        raise AssertionError("recognizer/generator/token parity failed")

    state = _synthetic_state(declaration.physical_ids(), declaration.level_rank)
    py_batch = _python_batch(state)
    py_step = py_batch.observe()
    native_batch, native_step = _cpp_batch(state)
    rows = []
    rows_equal = True
    py_decode_equal = True
    native_decode_equal = True
    for expected_stage, token in enumerate(tokens):
        py_row = _row_for_token(py_step, token)
        native_row = _row_for_token(native_step, token)
        if py_row is None or native_row is None:
            raise AssertionError(f"canonical token {token} was not legal at stage {expected_stage}")
        current_rows_equal = bool(np.array_equal(py_row, native_row))
        rows_equal = rows_equal and current_rows_equal
        if not current_rows_equal:
            raise AssertionError("Python/native MARVEL rows differ")
        parsed_py = python_game.actions_from_np(py_row[None, :])[0]
        parsed_cpp = cpp_game.actions_from_np(py_row[None, :])[0]
        current_py_decode = parsed_py.token_id == token and parsed_py.prefix_stage == expected_stage
        current_native_decode = parsed_cpp.token_id == token and parsed_cpp.prefix_stage == expected_stage
        py_decode_equal = py_decode_equal and current_py_decode
        native_decode_equal = native_decode_equal and current_native_decode
        if not (current_py_decode and current_native_decode):
            raise AssertionError("MARVEL row decoder changed token id or stage")
        rows.append([int(value) for value in py_row.tolist()])
        py_step = py_batch.step(py_row[None, :])
        native_step = native_batch.step(py_row[None, :])

    py_history = py_batch.envs[0]._require_protocol().round_state.history[-1]
    py_commit_equal = py_history.declaration is not None and _python_decl_key(py_history.declaration) == _python_decl_key(declaration)
    native_history = native_batch.debug_state(0)["state"]["history"][-1]
    native_commit_equal = native_history["declaration"] == cpp_declaration.to_dict()
    parity = {
        "python_recognizer_equals_native": recognizer_equal,
        "legal_action_generator_contains_declaration": generator_contains,
        "python_tokens_equal_native_tokens": tokens_equal,
        "python_rows_equal_native_rows": rows_equal,
        "python_row_decode_equal": py_decode_equal,
        "native_row_decode_equal": native_decode_equal,
        "python_environment_commit_equal": py_commit_equal,
        "native_environment_commit_equal": native_commit_equal,
    }
    if not all(parity.values()):
        raise AssertionError("complete reference parity failed")
    return {
        "complete_token_sequence": list(tokens),
        "annotated_tokens": [_token_record(token, declaration.level_rank) for token in tokens],
        "marvel_v2_action_rows": rows, "parity": parity,
    }

def _build_positive_vectors() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    vectors: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for spec in _positive_specs():
        declaration, cpp_declaration = _choose_declaration(spec)
        execution = _execute_positive(declaration, cpp_declaration)
        value = {
            "id": spec.vector_id, "family": spec.family.value.upper(), "polarity": "positive",
            "coverage": list(spec.coverage), "level_rank": _rank(BASE_VECTOR_LEVEL),
            "physical_cards": [_card_record(card_id) for card_id in spec.card_ids],
            "expected": {
                "result": "legal", "reason_code": "LEGAL_CANONICAL_DECLARATION",
                "canonical_declaration": _normalize_declaration(declaration), **execution,
            },
        }
        vectors.append(value)
        by_id[spec.vector_id] = value
    return vectors, by_id


BASELINE = {
    CombinationKind.FULL_HOUSE: "full_house_natural",
    CombinationKind.STRAIGHT: "straight_natural",
    CombinationKind.STRAIGHT_FLUSH: "straight_flush_natural",
    CombinationKind.PAIR_SEQUENCE: "pair_sequence_natural",
    CombinationKind.TRIPLE_SEQUENCE: "triple_sequence_natural",
    CombinationKind.FOUR_KINGS: "four_kings_natural",
}
WILD_BASELINE = {
    CombinationKind.FULL_HOUSE: "full_house_wild_substituted_triple",
    CombinationKind.STRAIGHT: "straight_wild_substituted",
    CombinationKind.STRAIGHT_FLUSH: "straight_flush_wild_substituted_clubs",
    CombinationKind.PAIR_SEQUENCE: "pair_sequence_wild_substituted",
    CombinationKind.TRIPLE_SEQUENCE: "triple_sequence_wild_substituted",
    CombinationKind.FOUR_KINGS: "four_kings_natural",
}


def _vector_card_ids(vector: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(item["card_id"]) for item in vector["physical_cards"])


def _family_absent(card_ids: Sequence[int], family: CombinationKind, level_rank: Rank = BASE_VECTOR_LEVEL) -> dict[str, Any]:
    py = [item for item in recognize_combinations(tuple(Card(value) for value in card_ids), level_rank=level_rank) if item.kind is family]
    cpp = [item for item in cpp_game.recognize_combinations(list(card_ids), int(level_rank)) if item.kind == family.value]
    if py or cpp:
        raise AssertionError(f"invalid physical vector unexpectedly recognized as {family.value}")
    generated = [item for item in enumerate_plays(tuple(Card(value) for value in card_ids), level_rank=level_rank) if item.kind is family]
    if generated:
        raise AssertionError(f"invalid physical vector unexpectedly generated as {family.value}")
    return {"python_recognizer_rejected": True, "native_recognizer_rejected": True, "legal_generator_rejected": True}


def _mutated_row(protocol: StepwiseActionState, token: int) -> np.ndarray:
    return np.asarray(marvel_impl._action_row(protocol, int(token)).row, dtype=np.int64)


def _token_sequence_rejected(card_ids: Sequence[int], tokens: Sequence[int], level_rank: Rank = BASE_VECTOR_LEVEL) -> dict[str, Any]:
    state = _synthetic_state(card_ids)
    py_batch = _python_batch(state)
    py_step = py_batch.observe()
    native_batch, native_step = _cpp_batch(state)
    py_error = native_error = None
    rejected_at = None
    for stage, token in enumerate(tokens):
        py_row = _row_for_token(py_step, token)
        native_row = _row_for_token(native_step, token)
        if py_row is not None and native_row is not None:
            np.testing.assert_array_equal(py_row, native_row)
            row = py_row
        else:
            row = _mutated_row(py_batch.envs[0]._require_protocol(), int(token))
        try:
            next_py = py_batch.step(row[None, :])
        except Exception as exc:
            py_error = f"{type(exc).__name__}: {exc}"
        try:
            next_native = native_batch.step(row[None, :])
        except Exception as exc:
            native_error = f"{type(exc).__name__}: {exc}"
        if py_error or native_error:
            if not py_error or not native_error:
                raise AssertionError("Python/native disagreed on invalid token rejection")
            rejected_at = stage
            break
        py_step, native_step = next_py, next_native
    if rejected_at is None:
        raise AssertionError("mutated token sequence was not rejected")
    return {
        "python_environment_rejected": True, "native_environment_rejected": True,
        "rejected_at_token_index": rejected_at,
        "python_error": py_error, "native_error": native_error,
    }


def _forged_comparison_rejected(base: dict[str, Any]) -> dict[str, Any]:
    card_ids = _vector_card_ids(base)
    family = CombinationKind(base["family"].lower())
    declaration = next(
        item for item in recognize_combinations(tuple(Card(value) for value in card_ids), level_rank=BASE_VECTOR_LEVEL)
        if item.kind is family and _normalize_declaration(item) == base["expected"]["canonical_declaration"]
    )
    wrong_rank = Rank.SMALL_JOKER if declaration.comparison_rank is not Rank.SMALL_JOKER else Rank.THREE
    forged = Combination(
        kind=declaration.kind, cards=declaration.cards, level_rank=declaration.level_rank,
        comparison_rank=wrong_rank, comparison_key=declaration.comparison_key,
        _wild_assignment_items=tuple((card_id, rank, suit) for card_id, (rank, suit) in declaration.wild_assignments.items()),
        _natural_wild_ids=declaration.natural_wild_ids,
    )
    state = _synthetic_state(card_ids)
    try:
        apply_action(state, CommittedAction(0, family, declaration.cards, forged))
    except IllegalActionError as exc:
        error = f"{type(exc).__name__}: {exc}"
    else:
        raise AssertionError("forged comparison_rank was accepted")
    if any(int(item.comparison_rank) == int(wrong_rank) and item.kind == family.value
           for item in cpp_game.recognize_combinations(list(card_ids), int(BASE_VECTOR_LEVEL))):
        raise AssertionError("native recognizer produced forged comparison rank")
    return {
        "python_environment_rejected": True,
        "native_recognizer_has_no_forged_declaration": True,
        "python_error": error, "forged_comparison_rank": _rank(wrong_rank),
    }


def _replace_token(tokens: Sequence[int], index: int, replacement: int) -> list[int]:
    result = list(int(value) for value in tokens)
    result[index] = int(replacement)
    return result


def _different_rank_token(token: int) -> int:
    current = TokenCodec.rank_from_token(token)
    return TokenCodec.rank(Rank.THREE if current is not Rank.THREE else Rank.FOUR)


def _different_suit_token(token: int) -> int:
    current = TokenCodec.suit_from_token(token)
    return TokenCodec.suit(Suit.DIAMONDS if current is not Suit.DIAMONDS else Suit.SPADES)


def _different_wild_token(token: int) -> int:
    rank, suit = TokenCodec.wild_from_token(token)
    return TokenCodec.wild(Rank.THREE if rank is not Rank.THREE else Rank.FOUR, suit)


def _physical_negative_cards(family: CombinationKind, category: str, baseline_ids: tuple[int, ...]) -> tuple[int, ...]:
    if category == "wrong_length":
        return baseline_ids[:-1]
    if category == "non_consecutive":
        if family is CombinationKind.STRAIGHT:
            return (C(Rank.SIX, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.NINE, Suit.HEARTS), C(Rank.TEN, Suit.SPADES), C(Rank.JACK, Suit.CLUBS))
        if family is CombinationKind.STRAIGHT_FLUSH:
            return tuple(C(rank, Suit.CLUBS) for rank in (Rank.SIX, Rank.SEVEN, Rank.NINE, Rank.TEN, Rank.JACK))
        if family is CombinationKind.PAIR_SEQUENCE:
            return tuple(C(rank, suit) for rank in (Rank.SIX, Rank.SEVEN, Rank.NINE) for suit in (Suit.CLUBS, Suit.DIAMONDS))
        if family is CombinationKind.TRIPLE_SEQUENCE:
            return tuple(C(rank, suit) for rank in (Rank.SEVEN, Rank.NINE) for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES))
        raise ValueError("non-consecutive is schema-level for this family")
    if category == "wrong_multiplicity":
        if family is CombinationKind.FULL_HOUSE:
            return (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS), C(Rank.TEN, Suit.CLUBS))
        if family is CombinationKind.STRAIGHT:
            return (C(Rank.SIX, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.EIGHT, Suit.HEARTS), C(Rank.EIGHT, Suit.SPADES), C(Rank.TEN, Suit.CLUBS))
        if family is CombinationKind.STRAIGHT_FLUSH:
            return (C(Rank.SIX, Suit.CLUBS, 0), C(Rank.SEVEN, Suit.CLUBS, 0), C(Rank.EIGHT, Suit.CLUBS, 0), C(Rank.EIGHT, Suit.CLUBS, 1), C(Rank.TEN, Suit.CLUBS, 0))
        if family is CombinationKind.PAIR_SEQUENCE:
            return (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.SPADES), C(Rank.EIGHT, Suit.CLUBS), C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS))
        if family is CombinationKind.TRIPLE_SEQUENCE:
            return (C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS), C(Rank.TEN, Suit.CLUBS), C(Rank.TEN, Suit.DIAMONDS), C(Rank.TEN, Suit.HEARTS), C(Rank.TEN, Suit.SPADES))
        if family is CombinationKind.FOUR_KINGS:
            return (C(Rank.SMALL_JOKER, Suit.JOKER, 0), C(Rank.SMALL_JOKER, Suit.JOKER, 1), C(Rank.BIG_JOKER, Suit.JOKER, 0), C(Rank.ACE, Suit.CLUBS))
    if category == "wrong_suit" and family is CombinationKind.STRAIGHT_FLUSH:
        return (C(Rank.SIX, Suit.CLUBS), C(Rank.SEVEN, Suit.CLUBS), C(Rank.EIGHT, Suit.DIAMONDS), C(Rank.NINE, Suit.CLUBS), C(Rank.TEN, Suit.CLUBS))
    if category == "wrong_wildcard" and family is CombinationKind.FOUR_KINGS:
        return (C(Rank.SMALL_JOKER, Suit.JOKER, 0), C(Rank.SMALL_JOKER, Suit.JOKER, 1), C(Rank.BIG_JOKER, Suit.JOKER, 0), C(Rank.FIVE, Suit.HEARTS, 0))
    raise ValueError(f"no physical mutation for {family.value}/{category}")


def _negative_record(vector_id: str, family: CombinationKind, category: str,
                     reason_code: str, base: dict[str, Any], candidate: dict[str, Any],
                     checks: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": vector_id, "family": family.value.upper(), "polarity": "negative",
        "coverage": [category], "level_rank": _rank(BASE_VECTOR_LEVEL),
        "reference_positive_id": base["id"], "candidate": candidate,
        "expected": {
            "result": "invalid", "reason_code": reason_code,
            "reference_canonical_declaration": base["expected"]["canonical_declaration"],
            "reference_complete_token_sequence": base["expected"]["complete_token_sequence"],
            "authority_checks": checks,
        },
    }


def _build_negative_vectors(positive_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    negatives: list[dict[str, Any]] = []
    sequence_families = {
        CombinationKind.STRAIGHT, CombinationKind.STRAIGHT_FLUSH,
        CombinationKind.PAIR_SEQUENCE, CombinationKind.TRIPLE_SEQUENCE,
    }
    for family in FAMILIES:
        base = positive_by_id[BASELINE[family]]
        base_ids = _vector_card_ids(base)
        base_tokens = list(base["expected"]["complete_token_sequence"])

        if family in sequence_families:
            ids = _physical_negative_cards(family, "non_consecutive", base_ids)
            checks = _family_absent(ids, family)
            candidate = {"physical_cards": [_card_record(value) for value in ids], "canonical_declaration": None, "complete_token_sequence": None}
            reason = "PHYSICAL_RANK_WINDOW_NON_CONSECUTIVE"
        else:
            insertion = [TokenCodec.rank(Rank.THREE), TokenCodec.rank(Rank.FIVE)]
            mutated = (base_tokens[:-1] + insertion + [base_tokens[-1]]) if family is CombinationKind.FOUR_KINGS else (base_tokens[:3] + insertion + base_tokens[3:])
            checks = _token_sequence_rejected(base_ids, mutated)
            candidate = {"physical_cards": base["physical_cards"], "bogus_sequence_window": [_rank(Rank.THREE), _rank(Rank.FIVE)], "complete_token_sequence": mutated}
            reason = "DECL_SEQUENCE_WINDOW_FORBIDDEN"
        negatives.append(_negative_record(f"{family.value}_negative_non_consecutive", family, "non_consecutive", reason, base, candidate, checks))

        ids = _physical_negative_cards(family, "wrong_multiplicity", base_ids)
        checks = _family_absent(ids, family)
        negatives.append(_negative_record(
            f"{family.value}_negative_wrong_multiplicity", family, "wrong_multiplicity", "EFFECTIVE_MULTIPLICITY_MISMATCH", base,
            {"physical_cards": [_card_record(value) for value in ids], "canonical_declaration": None, "complete_token_sequence": None}, checks,
        ))

        ids = _physical_negative_cards(family, "wrong_length", base_ids)
        checks = _family_absent(ids, family)
        negatives.append(_negative_record(
            f"{family.value}_negative_wrong_length", family, "wrong_length", "PHYSICAL_LENGTH_MISMATCH", base,
            {"physical_cards": [_card_record(value) for value in ids], "canonical_declaration": None, "complete_token_sequence": None}, checks,
        ))

        if family is CombinationKind.STRAIGHT_FLUSH:
            ids = _physical_negative_cards(family, "wrong_suit", base_ids)
            checks = _family_absent(ids, family)
            candidate = {"physical_cards": [_card_record(value) for value in ids], "canonical_declaration": None, "complete_token_sequence": None}
            reason = "DECLARED_SUIT_MISMATCH"
        else:
            mutated = base_tokens[:1] + [TokenCodec.suit(Suit.CLUBS)] + base_tokens[1:]
            checks = _token_sequence_rejected(base_ids, mutated)
            candidate = {"physical_cards": base["physical_cards"], "declared_suit": _suit(Suit.CLUBS), "complete_token_sequence": mutated}
            reason = "DECLARED_SUIT_FORBIDDEN"
        negatives.append(_negative_record(f"{family.value}_negative_wrong_suit", family, "wrong_suit", reason, base, candidate, checks))

        rank_indices = [index for index, token in enumerate(base_tokens) if RANK_TOKEN_BASE <= token < RANK_TOKEN_BASE + len(RANK_ORDER)]
        if rank_indices:
            index = rank_indices[0]
            mutated = _replace_token(base_tokens, index, _different_rank_token(base_tokens[index]))
        else:
            mutated = base_tokens[:1] + [TokenCodec.rank(Rank.THREE)] + base_tokens[1:]
        checks = _token_sequence_rejected(base_ids, mutated)
        negatives.append(_negative_record(
            f"{family.value}_negative_wrong_declared_ranks", family, "wrong_declared_ranks", "DECLARED_RANKS_MISMATCH", base,
            {"physical_cards": base["physical_cards"], "complete_token_sequence": mutated}, checks,
        ))

        checks = _forged_comparison_rejected(base)
        negatives.append(_negative_record(
            f"{family.value}_negative_wrong_comparison_rank", family, "wrong_comparison_rank", "COMPARISON_RANK_MISMATCH", base,
            {"physical_cards": base["physical_cards"], "forged_comparison_rank": checks["forged_comparison_rank"], "complete_token_sequence": None}, checks,
        ))

        wild_base = positive_by_id[WILD_BASELINE[family]]
        wild_ids = _vector_card_ids(wild_base)
        wild_tokens = list(wild_base["expected"]["complete_token_sequence"])
        assignment_indices = [index for index, token in enumerate(wild_tokens) if WILD_TOKEN_BASE <= token < WILD_TOKEN_BASE + 52]
        if assignment_indices:
            index = assignment_indices[0]
            mutated = _replace_token(wild_tokens, index, _different_wild_token(wild_tokens[index]))
            checks = _token_sequence_rejected(wild_ids, mutated)
            candidate = {"physical_cards": wild_base["physical_cards"], "complete_token_sequence": mutated}
            reference = wild_base
        else:
            ids = _physical_negative_cards(family, "wrong_wildcard", base_ids)
            checks = _family_absent(ids, family)
            candidate = {"physical_cards": [_card_record(value) for value in ids], "canonical_declaration": None, "complete_token_sequence": None}
            reference = base
        negatives.append(_negative_record(
            f"{family.value}_negative_wrong_wildcard", family, "wrong_wildcard", "WILDCARD_ASSIGNMENT_MISMATCH", reference, candidate, checks,
        ))

        card_indices = [index for index, token in enumerate(base_tokens) if CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + 108]
        if len(card_indices) < 2:
            raise AssertionError("canonical order negative needs two card tokens")
        mutated = list(base_tokens)
        left, right = card_indices[0], card_indices[1]
        mutated[left], mutated[right] = mutated[right], mutated[left]
        checks = _token_sequence_rejected(base_ids, mutated)
        negatives.append(_negative_record(
            f"{family.value}_negative_wrong_canonical_order", family, "wrong_canonical_order", "CANONICAL_CARD_ORDER_MISMATCH", base,
            {"physical_cards": base["physical_cards"], "complete_token_sequence": mutated}, checks,
        ))
    return negatives


def _window_records(length: int) -> list[dict[str, Any]]:
    return [
        {
            "index": index, "ranks": [_rank(rank) for rank in window],
            "declared_rank": _rank(window[0]), "comparison_rank": _rank(window[-1]),
            "comparison_order_value": WINDOW_RANKS.index(window[-1]),
            "special_ace_low": window[0] is Rank.ACE and window[1] is Rank.TWO,
        }
        for index, window in enumerate(action_protocol._sequence_windows(length))
    ]


def _family_rules() -> dict[str, Any]:
    common_wild = {
        "physical_wild_definition": "printed rank equals level_rank AND physical suit is HEARTS",
        "maximum_physical_wildcards": 2,
        "target_rank_domain": [_rank(rank) for rank in PRINTED_RANKS],
        "joker_target_allowed": False,
        "assignment_token_order": "ascending wildcard physical card_id; token carries target rank/suit but not card_id",
        "natural_encoding": "target_rank=level_rank,target_suit=HEARTS; card_id is in natural_wild_ids; a WILD_ASSIGNMENT token is still emitted",
        "substituted_encoding_non_suit_family": "target_rank=represented rank,target_suit=CLUBS as the authoritative suitless placeholder",
        "straight_flush_encoding": "target_suit equals declared_suit; natural level-heart use in HEARTS is canonicalized to natural",
        "level_rank_dual_encoding": "For non-suit families, target_rank=level_rank permits distinct natural HEARTS and substituted CLUBS declarations. For STRAIGHT_FLUSH, target_rank=level_rank is natural only with declared HEARTS; CLUBS/DIAMONDS/SPADES are substituted.",
    }
    sequence_participation = {
        "ACE": "allowed at the natural high end and as the first rank of the single explicit special low window",
        "TWO": "allowed as the first rank of natural WINDOW_RANKS windows and as the second rank of the explicit ACE-low window",
        "SMALL_JOKER": "forbidden", "BIG_JOKER": "forbidden",
        "wrap_rule": "only explicit A-2 prefix windows listed here; J-Q-K-A-2 and K-A-2 wraps are forbidden",
    }
    return {
        "FULL_HOUSE": {
            "wire_value": "full_house", "physical_lengths": [5],
            "effective_rank_multiplicity": [{"role": "triple", "count": 3}, {"role": "pair", "count": 2, "must_differ_from": "triple"}],
            "rank_participation": {"ACE": "allowed", "TWO": "allowed", "SMALL_JOKER": "forbidden", "BIG_JOKER": "forbidden"},
            "sequence_windows": None, "suit_rule": "physical suits are ignored; declared_suit must be null",
            "declared_ranks": {"count": 2, "order": ["triple_rank", "pair_rank"]},
            "comparison_rank": "triple_rank", "comparison_formula": "comparison_key=[0,FAMILY_ORDER.index(FULL_HOUSE),5,effective_rank_value(triple_rank,level_rank)]", "length_token": {"present": False, "value": None},
            "canonical_physical_card_order": "represented triple group first, represented pair group second; ascending physical card_id inside each group",
            "declaration_storage_card_order": "global printed-rank/suit/deck/card_id order from guandan.cards.sort_cards",
            "wildcard": {**common_wild, "allowed": True, "legal_target_ranks": "only declared triple_rank or pair_rank", "legal_target_suits": {"natural": "HEARTS iff target_rank=level_rank", "substituted": ["CLUBS"]}, "family_specific": ["at most two wildcards", "triple contains at least one natural card", "wildcards fill triple and/or pair deficits"]},
        },
        "STRAIGHT": {
            "wire_value": "straight", "physical_lengths": [5],
            "effective_rank_multiplicity": [{"role": "each rank in window", "count": 1}],
            "rank_participation": sequence_participation, "sequence_windows": _window_records(5),
            "suit_rule": "unrestricted; same-suit cards also admit a distinct STRAIGHT_FLUSH declaration",
            "declared_ranks": {"count": 1, "order": ["window_first_rank"]},
            "comparison_rank": "window_last_rank", "comparison_formula": "WINDOW_RANKS.index(window_last_rank); level strength is not used",
            "length_token": {"present": False, "value": None},
            "canonical_physical_card_order": "declared window order, then ascending physical card_id within represented rank",
            "declaration_storage_card_order": "global printed-rank/suit/deck/card_id order from guandan.cards.sort_cards",
            "wildcard": {**common_wild, "allowed": True, "legal_target_ranks": "only missing ranks in the selected 5-rank window", "legal_target_suits": {"natural": "HEARTS iff target_rank=level_rank", "substituted": ["CLUBS"]}, "family_specific": ["wildcards fill missing window ranks only", "no represented rank occurs more than once"]},
        },
        "STRAIGHT_FLUSH": {
            "wire_value": "straight_flush", "physical_lengths": [5],
            "effective_rank_multiplicity": [{"role": "each rank in window", "count": 1}],
            "rank_participation": sequence_participation, "sequence_windows": _window_records(5),
            "suit_rule": "all non-wild physical cards and represented wildcard targets share exactly one declared standard suit",
            "declared_ranks": {"count": 1, "order": ["window_first_rank"]},
            "declared_suit_order": "FAMILY, SUIT, RANK", "comparison_rank": "window_last_rank",
            "comparison_formula": "comparison_key=[3,WINDOW_RANKS.index(window_last_rank)]",
            "length_token": {"present": False, "value": None},
            "canonical_physical_card_order": "declared window order, then ascending physical card_id within represented rank",
            "declaration_storage_card_order": "global printed-rank/suit/deck/card_id order from guandan.cards.sort_cards",
            "wildcard": {**common_wild, "allowed": True, "legal_target_ranks": "only missing ranks in the selected 5-rank window", "legal_target_suits": "exactly declared_suit; declared_suit is one of CLUBS,DIAMONDS,HEARTS,SPADES", "family_specific": ["target_suit equals declared_suit", "all four standard target suits are legal when naturals permit"]},
        },
        "PAIR_SEQUENCE": {
            "wire_value": "pair_sequence", "physical_lengths": [6],
            "effective_rank_multiplicity": [{"role": "each rank in 3-rank window", "count": 2}],
            "rank_participation": sequence_participation, "sequence_windows": _window_records(3),
            "suit_rule": "unrestricted; declared_suit must be null",
            "declared_ranks": {"count": 1, "order": ["window_first_rank"]},
            "comparison_rank": "window_last_rank", "comparison_formula": "WINDOW_RANKS.index(window_last_rank); level strength is not used",
            "length_token": {"present": True, "value": 3, "meaning": "represented rank count, not physical card count"},
            "canonical_physical_card_order": "declared window order, then ascending physical card_id within each pair",
            "declaration_storage_card_order": "global printed-rank/suit/deck/card_id order from guandan.cards.sort_cards",
            "wildcard": {**common_wild, "allowed": True, "legal_target_ranks": "only deficit ranks in the selected 3-rank window", "legal_target_suits": {"natural": "HEARTS iff target_rank=level_rank", "substituted": ["CLUBS"]}, "family_specific": ["wildcards fill pair deficits", "each represented rank totals exactly two"]},
        },
        "TRIPLE_SEQUENCE": {
            "wire_value": "triple_sequence", "physical_lengths": [6],
            "effective_rank_multiplicity": [{"role": "each rank in 2-rank window", "count": 3}],
            "rank_participation": sequence_participation, "sequence_windows": _window_records(2),
            "suit_rule": "unrestricted; declared_suit must be null",
            "declared_ranks": {"count": 1, "order": ["window_first_rank"]},
            "comparison_rank": "window_last_rank", "comparison_formula": "WINDOW_RANKS.index(window_last_rank); level strength is not used",
            "length_token": {"present": True, "value": 2, "meaning": "represented rank count, not physical card count"},
            "canonical_physical_card_order": "declared window order, then ascending physical card_id within each triple",
            "declaration_storage_card_order": "global printed-rank/suit/deck/card_id order from guandan.cards.sort_cards",
            "wildcard": {**common_wild, "allowed": True, "legal_target_ranks": "only deficit ranks in the selected 2-rank window", "legal_target_suits": {"natural": "HEARTS iff target_rank=level_rank", "substituted": ["CLUBS"]}, "family_specific": ["wildcards fill triple deficits", "each represented rank totals exactly three"]},
        },
        "FOUR_KINGS": {
            "wire_value": "four_kings", "physical_lengths": [4],
            "effective_rank_multiplicity": [{"rank": _rank(Rank.SMALL_JOKER), "count": 2}, {"rank": _rank(Rank.BIG_JOKER), "count": 2}],
            "rank_participation": {"ACE": "forbidden", "TWO": "forbidden", "SMALL_JOKER": "exactly two physical copies", "BIG_JOKER": "exactly two physical copies"},
            "sequence_windows": None, "suit_rule": "physical joker suit marker only; declared_suit must be null",
            "declared_ranks": {"count": 0, "order": []}, "comparison_rank": "BIG_JOKER",
            "comparison_formula": "comparison_key=[9]", "length_token": {"present": False, "value": None},
            "canonical_physical_card_order": "ascending physical card_id in action tokens; canonical deck [52,53,106,107]",
            "declaration_storage_card_order": "global printed-rank/suit/deck/card_id order; canonical deck [52,106,53,107]",
            "wildcard": {**common_wild, "allowed": False, "maximum_physical_wildcards": 0, "legal_target_ranks": [], "legal_target_suits": [], "family_specific": ["no non-joker card", "no level-heart substitution"]},
            "family_token": TokenCodec.family(CombinationKind.FOUR_KINGS), "complete_encoding_formula": "FAMILY(FOUR_KINGS), CARD ids ascending, COMMIT; no RANK, SUIT, LENGTH, or WILD_ASSIGNMENT tokens",
        },
    }


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout


def _line_ending_kind(payload: bytes) -> str:
    crlf = payload.count(b"\r\n")
    without_crlf = payload.replace(b"\r\n", b"")
    bare_cr = without_crlf.count(b"\r")
    bare_lf = without_crlf.count(b"\n")
    if crlf == 0 and bare_cr == 0 and bare_lf == 0:
        return "NONE"
    if crlf and not bare_cr and not bare_lf:
        return "CRLF"
    if bare_lf and not crlf and not bare_cr:
        return "LF"
    return "MIXED"


def _normalize_lf(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _assert_authority_clean() -> None:
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *AUTHORITATIVE_SOURCES], cwd=ROOT)
    if result.returncode != 0:
        raise AssertionError("authoritative source files differ from HEAD")


def _source_records(source_commit: str) -> list[dict[str, Any]]:
    _assert_authority_clean()
    records = []
    for relative in AUTHORITATIVE_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        working = path.read_bytes()
        normalized = _normalize_lf(working)
        object_spec = f"{source_commit}:{relative.replace(chr(92), '/')}"
        oid = _run_git("rev-parse", object_spec)
        blob = _git_bytes("show", object_spec)
        blob_size = int(_run_git("cat-file", "-s", oid))
        if blob_size != len(blob):
            raise AssertionError(f"git blob size mismatch: {relative}")
        matches = normalized == blob
        if not matches:
            raise AssertionError(f"normalized working tree does not match authority blob: {relative}")
        records.append({
            "path": relative.replace("\\", "/"),
            "working_tree": {
                "hash_scope": "raw_checkout_bytes", "line_endings": _line_ending_kind(working),
                "bytes": len(working), "sha256": _sha256_bytes(working),
            },
            "normalized_lf": {
                "normalization": "CRLF and bare CR converted to LF", "bytes": len(normalized),
                "sha256": _sha256_bytes(normalized),
            },
            "git_blob": {
                "commit": source_commit, "oid": oid, "bytes": len(blob), "sha256": _sha256_bytes(blob),
            },
            "matches_head_after_normalization": matches,
        })
    return records


def _artifact_git_identity(output_dir: Path) -> dict[str, Any]:
    paths = [Path(__file__).resolve(), *(output_dir / name for name in (*OUTPUT_NAMES, "SHA256SUMS"))]
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    tracked = []
    for item in relative:
        probe = subprocess.run(["git", "ls-files", "--error-unmatch", "--", item], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tracked.append(probe.returncode == 0)
    if not any(tracked):
        status, commit = "untracked", None
    else:
        status_text = _run_git("status", "--porcelain=v1", "--untracked-files=all", "--", *relative)
        status, commit = ("tracked_clean_unbound", None) if not status_text else ("tracked_or_index_modified", None)
    return {"handoff_artifact_commit": commit, "handoff_artifact_git_status": status, "paths": relative}

def _build_rules(generated_at: str, source_commit: str, level_parity: dict[str, Any], artifact_git: dict[str, Any], exporter_sha256: str) -> dict[str, Any]:
    native_identity = copy.deepcopy(cpp_game.BACKEND_BUILD_IDENTITY)
    if not native_identity.get("source_matches_loaded_binary"):
        raise AssertionError("loaded native backend does not match current native sources")
    rules = {
        "schema": "guandan-declaration-rules-v1", "status": "fixed-authoritative-handoff",
        "ruleset": RULES_VERSION, "protocol_version": python_game.protocol_version,
        "protocol_components": {"action": python_game.ACTION_VERSION, "environment": python_game.ENV_VERSION, "encoding": python_game.ENCODING_VERSION},
        "game_identifiers": {"python_reference": python_game.GAME_IDENTIFIER, "native_runtime": cpp_game.GAME_IDENTIFIER},
        "authority_source_commit": source_commit,
        "guandan_source_commit": source_commit,
        "handoff_artifact_commit": artifact_git["handoff_artifact_commit"],
        "handoff_artifact_git_status": artifact_git["handoff_artifact_git_status"],
        "exporter_sha256": exporter_sha256,
        "authoritative_source_files_clean_at_export": True,
        "generated_at": generated_at, "generation_timezone": "Asia/Shanghai (+08:00)", "cwd": str(ROOT),
        "authority": {
            "primary_runtime": "native C++ MARVEL backend loaded from nanopokerlib._guandan_cpp",
            "reference_runtime": "Python rules/environment oracle",
            "recognizer_paths": list(RECOGNIZER_PATHS), "generator_paths": list(GENERATOR_PATHS),
            "serializer_paths": list(SERIALIZER_PATHS), "validator_paths": list(VALIDATOR_PATHS),
            "marvel_encoder_paths": list(ENCODER_PATHS), "source_files": _source_records(source_commit),
            "native_build_identity": native_identity,
        },
        "token_contract": {
            "semantic_token_vocabulary": 256, "marvel_v2_vocabulary": python_game.num_tokens,
            "complete_declaration_sequence": "V1 semantic tokens from guandan.action_state.canonical_tokens, terminated by COMMIT",
            "marvel_v2_transport": "each current V1 token is wrapped in one fixed eight-token structured action row",
            "marvel_action_row_fields": list(python_game.ACTION_SCHEMA["row"]),
            "family_token_base": FAMILY_TOKEN_BASE, "rank_token_base": RANK_TOKEN_BASE,
            "suit_token_base": SUIT_TOKEN_BASE, "length_token_base": LENGTH_TOKEN_BASE,
            "card_token_base": CARD_TOKEN_BASE, "wild_token_base": WILD_TOKEN_BASE,
            "commit_token": COMMIT_TOKEN, "family_order": [kind.value.upper() for kind in FAMILY_ORDER],
            "rank_order": [_rank(rank) for rank in RANK_ORDER], "suit_order": [_suit(suit) for suit in Suit],
            "wild_token_formula": "172 + printed_rank_index(THREE..TWO)*4 + suit_value(CLUBS..SPADES)",
            "card_token_formula": "64 + physical_card_id",
        },
        "global_rank_orders": {
            "printed_rank_order": [_rank(rank) for rank in PRINTED_RANKS],
            "sequence_window_order": [_rank(rank) for rank in WINDOW_RANKS],
            "sequence_comparison_note": "sequence comparison uses WINDOW_RANKS index of comparison_rank, not level-adjusted strength",
        },
        "families": _family_rules(),
        "level_parity": level_parity,
        "artifact_git": artifact_git,
        "reason_codes": {
            "LEGAL_CANONICAL_DECLARATION": "authoritative recognizer/generator/serializer/environment/native parity accepted",
            "PHYSICAL_RANK_WINDOW_NON_CONSECUTIVE": "represented ranks match no authoritative window",
            "DECL_SEQUENCE_WINDOW_FORBIDDEN": "non-sequence family was given sequence fields/tokens",
            "EFFECTIVE_MULTIPLICITY_MISMATCH": "represented-rank counts do not match the family",
            "PHYSICAL_LENGTH_MISMATCH": "physical card count is outside the family length set",
            "DECLARED_SUIT_MISMATCH": "straight-flush represented suit disagrees",
            "DECLARED_SUIT_FORBIDDEN": "suit token/field supplied to a suitless family",
            "DECLARED_RANKS_MISMATCH": "declared rank token content/order differs from canonical serialization",
            "COMPARISON_RANK_MISMATCH": "forged comparison_rank has no authoritative declaration",
            "WILDCARD_ASSIGNMENT_MISMATCH": "wild target/mode/suit is not authoritative",
            "CANONICAL_CARD_ORDER_MISMATCH": "card tokens are not in canonical family order",
        },
        "unresolved": [],
        "not_verified": ["Guance-side ingestion and closure of GMV2-S2-DECL-003; D:/project/guance was intentionally not accessed"],
        "scope_guards": {
            "checkpoint_read": False, "real_capture_read": False,
            "complete_real_game_read": False, "hidden_hand_data_read": False,
            "training_executed": False, "model_inference_executed": False,
            "guance_repository_accessed": False, "fixtures": "synthetic physical-card vectors only",
        },
        "integrity": {"semantic_hash_rule": "SHA256 of canonical compact JSON with integrity.semantic_sha256=null", "semantic_sha256": None},
    }
    rules["integrity"]["semantic_sha256"] = _semantic_digest(rules)
    return rules


def _coverage_summary(vectors: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in FAMILIES:
        name = family.value.upper()
        family_vectors = [value for value in vectors if value["family"] == name]
        positives = [value for value in family_vectors if value["polarity"] == "positive"]
        negatives = [value for value in family_vectors if value["polarity"] == "negative"]
        result[name] = {
            "positive": len(positives), "negative": len(negatives),
            "coverage_tags": sorted({tag for value in family_vectors for tag in value["coverage"]}),
        }
    return result


def _build_vectors(generated_at: str, source_commit: str, rules_hash: str,
                   level_parity: dict[str, Any], artifact_git: dict[str, Any],
                   exporter_sha256: str) -> dict[str, Any]:
    base_positives, by_id = _build_positive_vectors()
    base_negatives = _build_negative_vectors(by_id)
    special = _special_boundary_vectors()
    vectors = base_positives + base_negatives + special
    positives = [vector for vector in vectors if vector["polarity"] == "positive"]
    negatives = [vector for vector in vectors if vector["polarity"] == "negative"]
    positive_parity = [vector["expected"]["parity"] for vector in positives]
    reference_parity = {
        "pipeline": ["physical_cards", "legal_action/declaration", "canonical_declaration", "V1 semantic tokens", "MARVEL V2 action rows", "decode", "same action/declaration"],
        "all_positive_vectors_passed": bool(positive_parity) and all(all(item.values()) for item in positive_parity),
        "python_native_recognizer_parity": all(item["python_recognizer_equals_native"] for item in positive_parity) and level_parity["python_native_recognizer_parity"],
        "python_native_token_parity": all(item["python_tokens_equal_native_tokens"] for item in positive_parity) and level_parity["python_native_token_parity"],
        "python_native_action_row_parity": all(item["python_rows_equal_native_rows"] for item in positive_parity) and level_parity["python_native_action_row_parity"],
        "python_native_decode_parity": all(item["python_row_decode_equal"] and item["native_row_decode_equal"] for item in positive_parity) and level_parity["python_native_decode_parity"],
        "python_native_environment_commit_parity": all(item["python_environment_commit_equal"] and item["native_environment_commit_equal"] for item in positive_parity) and level_parity["python_native_environment_commit_parity"],
    }
    if not all(value for key, value in reference_parity.items() if key != "pipeline"):
        raise AssertionError("aggregate reference parity failed")
    payload = {
        "schema": "guandan-declaration-golden-vectors-v1",
        "rules_schema": "guandan-declaration-rules-v1", "rules_file_sha256": rules_hash,
        "authority_source_commit": source_commit, "guandan_source_commit": source_commit,
        "handoff_artifact_commit": artifact_git["handoff_artifact_commit"],
        "handoff_artifact_git_status": artifact_git["handoff_artifact_git_status"],
        "exporter_sha256": exporter_sha256, "protocol_version": python_game.protocol_version,
        "generated_at": generated_at, "base_detailed_vector_level": _rank(BASE_VECTOR_LEVEL),
        "stable_reason_codes_are": "exporter contract classifications; raw Python/native exception text is diagnostic only",
        "vector_count": {"positive": len(positives), "negative": len(negatives), "total": len(vectors),
                         "base_positive": len(base_positives), "base_negative": len(base_negatives),
                         "special_boundary": len(special)},
        "coverage_by_family": _coverage_summary(vectors),
        "special_boundary_vector_ids": [vector["id"] for vector in special],
        "level_parity": level_parity, "reference_parity": reference_parity,
        "artifact_git": artifact_git, "vectors": vectors,
        "integrity": {"semantic_hash_rule": "SHA256 of canonical compact JSON with integrity.semantic_sha256=null", "semantic_sha256": None},
    }
    payload["integrity"]["semantic_sha256"] = _semantic_digest(payload)
    return payload

def _markdown(rules: dict[str, Any], vectors: dict[str, Any], rules_hash: str, vectors_hash: str) -> str:
    matrix = vectors["level_parity"]
    lines = [
        "# Guandan complex declaration contract v1", "",
        "> Guance Phase 2 / GMV2-S2-DECL-003 权威输入候选。所有规则证据来自当前 Python reference 与源码匹配的 native C++ MARVEL backend；未访问 Guance。", "",
        "## 身份与 durability", "",
        f"- Ruleset: `{rules['ruleset']}`", f"- Protocol: `{rules['protocol_version']}`",
        f"- Authority source commit: `{rules['authority_source_commit']}`",
        f"- Handoff artifact commit: `{'null' if rules['handoff_artifact_commit'] is None else rules['handoff_artifact_commit']}`",
        f"- Handoff artifact git status: `{rules['handoff_artifact_git_status']}`",
        f"- Exporter SHA256: `{rules['exporter_sha256']}`", f"- Generated at: `{rules['generated_at']}`",
        f"- CWD: `{rules['cwd']}`", f"- Rules JSON SHA256: `{rules_hash}`",
        f"- Rules semantic SHA256: `{rules['integrity']['semantic_sha256']}`",
        f"- Golden vectors JSON SHA256: `{vectors_hash}`",
        f"- Vectors semantic SHA256: `{vectors['integrity']['semantic_sha256']}`",
        f"- Native extension SHA256: `{rules['authority']['native_build_identity']['extension_sha256']}`",
        f"- Native source matches loaded binary: `{str(rules['authority']['native_build_identity']['source_matches_loaded_binary']).lower()}`", "",
        "`authority_source_commit` 只绑定权威规则源码；当前 handoff/exporter 仍是 untracked，`handoff_artifact_commit=null`，未把源码 commit 冒充 artifact commit。", "",
        "## Authority source hash scope", "",
        "每个 authority source record 同时固定三种身份：", "",
        "1. `working_tree`: 当前 checkout 原始字节、换行类型、字节数和 SHA256，仅表示本机执行环境；",
        "2. `normalized_lf`: 将 CRLF 和 bare CR 转换为 LF 后的字节数和 SHA256；",
        "3. `git_blob`: `authority_source_commit:path` 的 blob OID、blob bytes 和内容 SHA256，作为跨平台 authority identity。", "",
        "Exporter 要求每个文件 `normalized_lf == git_blob bytes` 且 authority paths 相对 HEAD clean，否则 fail closed / exit 1。", "",
        "## 权威路径", "",
        "- Recognizer: " + ", ".join(f"`{item}`" for item in rules["authority"]["recognizer_paths"]),
        "- Legal generator: " + ", ".join(f"`{item}`" for item in rules["authority"]["generator_paths"]),
        "- Serializer: " + ", ".join(f"`{item}`" for item in rules["authority"]["serializer_paths"]),
        "- Validator: " + ", ".join(f"`{item}`" for item in rules["authority"]["validator_paths"]),
        "- MARVEL encoder/decoder: " + ", ".join(f"`{item}`" for item in rules["authority"]["marvel_encoder_paths"]), "",
        "## 13-level parity matrix", "",
        f"- Levels: `{', '.join(matrix['levels_tested'])}`",
        f"- Invalid levels rejected: `{', '.join(matrix['invalid_levels_rejected'])}`",
        f"- Matrix cases: `{matrix['matrix_case_count']}` (`{matrix['cases_per_level']}` per level)",
        "- 每个 level：6 个 family natural baseline；5 个 wildcard family × natural/substituted-other-rank/substituted-level-rank/max-two；FOUR_KINGS wildcard rejection 与无 rank/suit/length/wild token。", "",
        "| Level | cases | natural | natural wild | substituted-other | substituted-level | max-two wild | result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for level in matrix["levels"]:
        lines.append(
            f"| {level['level']['name']} | {level['case_count']} | {level['natural_family_count']} | {level['natural_wildcard_family_count']} | {level['substituted_wildcard_family_count']} | {level['substituted_level_rank_wildcard_family_count']} | {level['maximum_wildcard_family_count']} | {'PASS' if level['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "", "聚合 parity 值全部由逐 case 实际执行结果计算，不从已有 JSON 读取，也不直接硬编码。`--check` 会重新运行完整 matrix。", "",
        "## THREE / ACE / TWO 固定边界 vectors", "",
        "- THREE：level-heart natural/substituted、缺 THREE 的 sequence、FULL_HOUSE 中 THREE 作为 triple 与 pair。",
        "- ACE：heart ACE natural/substituted、A2345、10JQKA、非法 JQKA2、ACE 作为 full-house triple/pair、hearts straight flush。",
        "- TWO：heart TWO natural/substituted、A2345、pair-sequence A23、triple-sequence A2、非法 KA2、TWO 作为 full-house triple/pair、hearts straight flush。",
        f"- Detailed special vectors: `{vectors['vector_count']['special_boundary']}`。", "",
        "## 关键序列语义", "",
        "序列基础顺序是 `2,3,4,5,6,7,8,9,10,J,Q,K,A`。只额外允许 A-low 窗口：`A2345`、`A23`、`A2`。不允许 `JQKA2`、`KA2` 或其他绕回；大小王不参加序列；比较使用窗口末位在 WINDOW_RANKS 中的索引。", "",
        "## Family 摘要", "",
        "| Family | physical length | multiplicity | declared_ranks | comparison_rank | LENGTH | suit | wild |",
        "|---|---:|---|---|---|---|---|---|",
    ])
    for name, family in rules["families"].items():
        multiplicity = "; ".join(f"{item.get('role', item.get('rank', {}).get('name', 'rank'))}×{item['count']}" for item in family["effective_rank_multiplicity"])
        declared = ",".join(family["declared_ranks"]["order"]) or "[]"
        length = str(family["length_token"]["value"]) if family["length_token"]["present"] else "none"
        lines.append(f"| {name} | {','.join(map(str, family['physical_lengths']))} | {multiplicity} | {declared} | {family['comparison_rank']} | {length} | {family['suit_rule']} | {'yes' if family['wildcard']['allowed'] else 'no'} / max {family['wildcard']['maximum_physical_wildcards']} |")
    lines.extend([
        "", "## Encoding 要点", "",
        "- FULL_HOUSE: `declared_ranks=[triple_rank,pair_rank]`; card token 先 triple 后 pair。",
        "- STRAIGHT/STRAIGHT_FLUSH: `declared_ranks=[window_first_rank]`; comparison rank 是 `window_last_rank`。",
        "- STRAIGHT_FLUSH 字段顺序为 `FAMILY,SUIT,RANK`。",
        "- PAIR_SEQUENCE: physical length 6，`LENGTH=3`；TRIPLE_SEQUENCE: physical length 6，`LENGTH=2`。",
        "- FOUR_KINGS: `declared_ranks=[]`, `comparison_rank=BIG_JOKER`, `comparison_key=[9]`，无 SUIT/LENGTH/WILD；action order `[52,53,106,107]`，storage order `[52,106,53,107]`。",
        "- 无 suit family 的 substituted wildcard 使用 `CLUBS` 作为权威占位 suit；`target_rank=level_rank` 时 natural HEARTS 与 substituted CLUBS 是两个可区分的合法 declaration。",
        "- STRAIGHT_FLUSH 的 target suit 必须等于 declared suit；level rank 在 HEARTS 中 canonicalize 为 natural，在 CLUBS/DIAMONDS/SPADES 中为 substituted。", "",
        "## Golden vectors", "",
        f"- Positive: `{vectors['vector_count']['positive']}`（base `{vectors['vector_count']['base_positive']}`）",
        f"- Negative: `{vectors['vector_count']['negative']}`（base `{vectors['vector_count']['base_negative']}`）",
        f"- Special boundary: `{vectors['vector_count']['special_boundary']}`",
        f"- Total: `{vectors['vector_count']['total']}`", "",
        "## 生成和验证", "", "```powershell", f"cd {rules['cwd']}",
        f"py -3.12 -B tools/export_guance_declaration_contract.py --output-dir guance_phase2_declaration_handoff --generated-at {rules['generated_at']}",
        "py -3.12 -B tools/export_guance_declaration_contract.py --output-dir guance_phase2_declaration_handoff --check",
        "py -3.12 -B -m pytest -q -p no:cacheprovider tests/unit/test_combos.py tests/unit/test_action_protocol.py tests/acceptance/test_a02_rules.py tests/marvel_cpp/test_round_protocol.py tests/acceptance/test_guance_declaration_handoff.py",
        "git diff --check", "```", "",
        "## 停止点 / 未验证", "",
        "- Handoff final acceptance 候选可由本仓独立验收；不等于 Guance DECL-003 或 Guance Phase 2 已通过。",
        "- 未访问或修改 `D:\\project\\guance`；未进入 Phase 3。",
        "- 未读取 checkpoint、真实 capture、完整真实牌局或隐藏牌；未执行训练或模型推理。",
        "- 未 commit，未 push；artifact durability 尚未由 Git commit 提供。", "",
    ])
    return "\n".join(lines)

def _manifest_bytes(paths: Sequence[Path]) -> bytes:
    return ("\n".join(f"{_sha256_file(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths) + "\n").encode("ascii")


def _build_artifacts(generated_at: str, output_dir: Path) -> dict[str, bytes]:
    _assert_authority_clean()
    source_commit = _run_git("rev-parse", "HEAD")
    exporter_sha256 = _sha256_file(Path(__file__).resolve())
    artifact_git = _artifact_git_identity(output_dir)
    level_parity = _build_level_parity_matrix()
    rules = _build_rules(generated_at, source_commit, level_parity, artifact_git, exporter_sha256)
    rules_bytes = _json_bytes(rules)
    rules_hash = _sha256_bytes(rules_bytes)
    vectors = _build_vectors(generated_at, source_commit, rules_hash, level_parity, artifact_git, exporter_sha256)
    vectors_bytes = _json_bytes(vectors)
    vectors_hash = _sha256_bytes(vectors_bytes)
    markdown_bytes = (_markdown(rules, vectors, rules_hash, vectors_hash) + "\n").encode("utf-8")
    return {OUTPUT_NAMES[0]: rules_bytes, OUTPUT_NAMES[1]: vectors_bytes, OUTPUT_NAMES[2]: markdown_bytes}

def _write(output_dir: Path, generated_at: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _build_artifacts(generated_at, output_dir)
    for name, payload in artifacts.items():
        (output_dir / name).write_bytes(payload)
    manifest_paths = [output_dir / name for name in OUTPUT_NAMES] + [Path(__file__).resolve()]
    (output_dir / "SHA256SUMS").write_bytes(_manifest_bytes(manifest_paths))
    vectors = json.loads(artifacts[OUTPUT_NAMES[1]])
    return {
        "mode": "write", "output_dir": str(output_dir), "source_commit": _run_git("rev-parse", "HEAD"),
        "positive_vectors": vectors["vector_count"]["positive"], "negative_vectors": vectors["vector_count"]["negative"],
        "levels_tested": vectors["level_parity"]["levels_tested"], "level_matrix_cases": vectors["level_parity"]["matrix_case_count"],
        "handoff_artifact_git_status": vectors["handoff_artifact_git_status"],
        "parity": vectors["reference_parity"],
        "files": {name: _sha256_file(output_dir / name) for name in (*OUTPUT_NAMES, "SHA256SUMS")}, "exit_code": 0,
    }



def _assert_semantic_hash(payload: dict[str, Any], label: str) -> None:
    actual = payload.get("integrity", {}).get("semantic_sha256")
    expected = _semantic_digest(payload)
    if actual != expected:
        raise AssertionError(f"{label} semantic SHA256 mismatch")

def _check(output_dir: Path) -> dict[str, Any]:
    rules_path = output_dir / OUTPUT_NAMES[0]
    if not rules_path.is_file():
        raise FileNotFoundError(rules_path)
    existing_rules = json.loads(rules_path.read_text(encoding="utf-8"))
    vectors_path = output_dir / OUTPUT_NAMES[1]
    existing_vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    _assert_semantic_hash(existing_rules, "rules")
    _assert_semantic_hash(existing_vectors, "vectors")
    generated_at = str(existing_rules["generated_at"])
    expected = _build_artifacts(generated_at, output_dir)
    for name, payload in expected.items():
        if (output_dir / name).read_bytes() != payload:
            raise AssertionError(f"artifact differs from authoritative regeneration: {name}")
    expected_manifest = _manifest_bytes([output_dir / name for name in OUTPUT_NAMES] + [Path(__file__).resolve()])
    if (output_dir / "SHA256SUMS").read_bytes() != expected_manifest:
        raise AssertionError("SHA256SUMS differs from current artifact/script hashes")
    vectors = json.loads((output_dir / OUTPUT_NAMES[1]).read_text(encoding="utf-8"))
    return {
        "mode": "check", "output_dir": str(output_dir), "source_commit": _run_git("rev-parse", "HEAD"),
        "positive_vectors": vectors["vector_count"]["positive"], "negative_vectors": vectors["vector_count"]["negative"],
        "levels_tested": vectors["level_parity"]["levels_tested"], "level_matrix_cases": vectors["level_parity"]["matrix_case_count"],
        "handoff_artifact_git_status": vectors["handoff_artifact_git_status"],
        "parity": vectors["reference_parity"],
        "files": {name: _sha256_file(output_dir / name) for name in (*OUTPUT_NAMES, "SHA256SUMS")}, "exit_code": 0,
    }





def _other_ranks(level_rank: Rank, count: int, *, excluded: Sequence[Rank] = ()) -> tuple[Rank, ...]:
    blocked = {Rank(level_rank), *(Rank(value) for value in excluded)}
    result = tuple(rank for rank in PRINTED_RANKS if rank not in blocked)
    if len(result) < count:
        raise AssertionError("not enough non-level ranks for synthetic fixture")
    return result[:count]


def _window_containing(level_rank: Rank, length: int) -> tuple[Rank, ...]:
    return next(window for window in action_protocol._sequence_windows(length) if level_rank in window)


def _window_not_containing(level_rank: Rank, length: int) -> tuple[Rank, ...]:
    return next(window for window in action_protocol._sequence_windows(length) if level_rank not in window)


def _matrix_specs(level_rank: Rank) -> tuple[PositiveSpec, ...]:
    h0 = C(level_rank, Suit.HEARTS, 0)
    h1 = C(level_rank, Suit.HEARTS, 1)
    other0, other1 = _other_ranks(level_rank, 2)
    straight_window = _window_containing(level_rank, 5)
    pair_window = _window_containing(level_rank, 3)
    triple_window = _window_containing(level_rank, 2)

    def mixed_single_cards(window: Sequence[Rank], skipped: set[Rank] | None = None) -> tuple[int, ...]:
        skipped = skipped or set()
        suits = (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES, Suit.CLUBS, Suit.DIAMONDS)
        return tuple(C(rank, suits[index]) for index, rank in enumerate(window) if rank not in skipped)

    specs: list[PositiveSpec] = [
        PositiveSpec(f"matrix_{level_rank.name.lower()}_full_house_natural", CombinationKind.FULL_HOUSE,
            (C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS)),
            ("matrix_natural",)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_natural", CombinationKind.STRAIGHT,
            (C(Rank.SIX, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.EIGHT, Suit.SPADES), C(Rank.NINE, Suit.CLUBS), C(Rank.TEN, Suit.DIAMONDS)),
            ("matrix_natural",)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_flush_natural", CombinationKind.STRAIGHT_FLUSH,
            tuple(C(rank, Suit.CLUBS) for rank in (Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN)),
            ("matrix_natural",)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_pair_sequence_natural", CombinationKind.PAIR_SEQUENCE,
            tuple(C(rank, suit) for rank in (Rank.SEVEN, Rank.EIGHT, Rank.NINE) for suit in (Suit.CLUBS, Suit.DIAMONDS)),
            ("matrix_natural",)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_triple_sequence_natural", CombinationKind.TRIPLE_SEQUENCE,
            tuple(C(rank, suit) for rank in (Rank.NINE, Rank.TEN) for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES)),
            ("matrix_natural",)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_four_kings_natural", CombinationKind.FOUR_KINGS,
            (C(Rank.SMALL_JOKER, Suit.JOKER, 0), C(Rank.SMALL_JOKER, Suit.JOKER, 1), C(Rank.BIG_JOKER, Suit.JOKER, 0), C(Rank.BIG_JOKER, Suit.JOKER, 1)),
            ("matrix_natural",), Rank.BIG_JOKER),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_full_house_wild_natural", CombinationKind.FULL_HOUSE,
            (C(level_rank, Suit.CLUBS), C(level_rank, Suit.DIAMONDS), h0, C(other0, Suit.CLUBS), C(other0, Suit.DIAMONDS)),
            ("matrix_wild_natural",), level_rank, ((h0, level_rank, Suit.HEARTS, "natural"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_full_house_wild_substituted", CombinationKind.FULL_HOUSE,
            (C(other0, Suit.CLUBS), C(other0, Suit.DIAMONDS), h0, C(other1, Suit.CLUBS), C(other1, Suit.DIAMONDS)),
            ("matrix_wild_substituted",), other0, ((h0, other0, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_full_house_wild_substituted_level_rank", CombinationKind.FULL_HOUSE,
            (C(level_rank, Suit.CLUBS), C(level_rank, Suit.DIAMONDS), h0, C(other0, Suit.CLUBS), C(other0, Suit.DIAMONDS)),
            ("matrix_wild_substituted_level_rank",), level_rank, ((h0, level_rank, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_full_house_wild_max", CombinationKind.FULL_HOUSE,
            (C(other0, Suit.CLUBS), C(other0, Suit.DIAMONDS), C(other1, Suit.CLUBS), h0, h1),
            ("matrix_wild_max",), other0,
            ((h0, other0, Suit.CLUBS, "substituted"), (h1, other1, Suit.CLUBS, "substituted"))),
    ]

    # STRAIGHT: natural level use, one substitution, and two substitutions.
    straight_nat_cards = tuple(h0 if rank is level_rank else C(rank, (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES)[index % 3])
                               for index, rank in enumerate(straight_window))
    straight_sub_target = next(rank for rank in straight_window if rank is not level_rank)
    straight_sub_cards = mixed_single_cards(straight_window, {straight_sub_target}) + (h0,)
    straight_max_targets = tuple(rank for rank in straight_window if rank is not level_rank)[:2]
    straight_max_cards = mixed_single_cards(straight_window, set(straight_max_targets)) + (h0, h1)
    specs.extend([
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_wild_natural", CombinationKind.STRAIGHT,
            straight_nat_cards, ("matrix_wild_natural",), straight_window[-1], ((h0, level_rank, Suit.HEARTS, "natural"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_wild_substituted_level_rank", CombinationKind.STRAIGHT,
            straight_nat_cards, ("matrix_wild_substituted_level_rank",), straight_window[-1], ((h0, level_rank, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_wild_substituted", CombinationKind.STRAIGHT,
            straight_sub_cards, ("matrix_wild_substituted",), straight_window[-1], ((h0, straight_sub_target, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_wild_max", CombinationKind.STRAIGHT,
            straight_max_cards, ("matrix_wild_max",), straight_window[-1],
            ((h0, straight_max_targets[0], Suit.CLUBS, "substituted"), (h1, straight_max_targets[1], Suit.CLUBS, "substituted"))),
    ])

    sf_nat_cards = tuple(h0 if rank is level_rank else C(rank, Suit.HEARTS) for rank in straight_window)
    sf_sub_cards = tuple(C(rank, Suit.CLUBS) for rank in straight_window if rank is not straight_sub_target) + (h0,)
    sf_max_cards = tuple(C(rank, Suit.SPADES) for rank in straight_window if rank not in straight_max_targets) + (h0, h1)
    specs.extend([
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_flush_wild_natural", CombinationKind.STRAIGHT_FLUSH,
            sf_nat_cards, ("matrix_wild_natural", "matrix_wild_target_suit"), straight_window[-1], ((h0, level_rank, Suit.HEARTS, "natural"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_flush_wild_substituted_level_rank", CombinationKind.STRAIGHT_FLUSH,
            tuple(h0 if rank is level_rank else C(rank, Suit.CLUBS) for rank in straight_window),
            ("matrix_wild_substituted_level_rank", "matrix_wild_target_suit"), straight_window[-1], ((h0, level_rank, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_flush_wild_substituted", CombinationKind.STRAIGHT_FLUSH,
            sf_sub_cards, ("matrix_wild_substituted", "matrix_wild_target_suit"), straight_window[-1], ((h0, straight_sub_target, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_straight_flush_wild_max", CombinationKind.STRAIGHT_FLUSH,
            sf_max_cards, ("matrix_wild_max", "matrix_wild_target_suit"), straight_window[-1],
            ((h0, straight_max_targets[0], Suit.SPADES, "substituted"), (h1, straight_max_targets[1], Suit.SPADES, "substituted"))),
    ])

    pair_sub_target = next(rank for rank in pair_window if rank is not level_rank)
    pair_max_targets = tuple(rank for rank in pair_window if rank is not level_rank)[:2]
    pair_nat_cards = tuple(card for rank in pair_window for card in
                           ((C(rank, Suit.CLUBS), h0) if rank is level_rank else (C(rank, Suit.CLUBS), C(rank, Suit.DIAMONDS))))
    pair_sub_cards = tuple(card for rank in pair_window for card in
                           ((C(rank, Suit.CLUBS),) if rank is pair_sub_target else (C(rank, Suit.CLUBS), C(rank, Suit.DIAMONDS)))) + (h0,)
    pair_max_cards = tuple(card for rank in pair_window for card in
                           ((C(rank, Suit.CLUBS),) if rank in pair_max_targets else (C(rank, Suit.CLUBS), C(rank, Suit.DIAMONDS)))) + (h0, h1)
    specs.extend([
        PositiveSpec(f"matrix_{level_rank.name.lower()}_pair_sequence_wild_natural", CombinationKind.PAIR_SEQUENCE,
            pair_nat_cards, ("matrix_wild_natural",), pair_window[-1], ((h0, level_rank, Suit.HEARTS, "natural"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_pair_sequence_wild_substituted_level_rank", CombinationKind.PAIR_SEQUENCE,
            pair_nat_cards, ("matrix_wild_substituted_level_rank",), pair_window[-1], ((h0, level_rank, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_pair_sequence_wild_substituted", CombinationKind.PAIR_SEQUENCE,
            pair_sub_cards, ("matrix_wild_substituted",), pair_window[-1], ((h0, pair_sub_target, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_pair_sequence_wild_max", CombinationKind.PAIR_SEQUENCE,
            pair_max_cards, ("matrix_wild_max",), pair_window[-1],
            ((h0, pair_max_targets[0], Suit.CLUBS, "substituted"), (h1, pair_max_targets[1], Suit.CLUBS, "substituted"))),
    ])

    triple_sub_target = next(rank for rank in triple_window if rank is not level_rank)
    max_triple_window = _window_not_containing(level_rank, 2)
    triple_nat_cards = tuple(card for rank in triple_window for card in
                             ((C(rank, Suit.CLUBS), C(rank, Suit.DIAMONDS), h0) if rank is level_rank else
                              (C(rank, Suit.CLUBS), C(rank, Suit.DIAMONDS), C(rank, Suit.SPADES))))
    triple_sub_cards = tuple(card for rank in triple_window for card in
                             ((C(rank, Suit.CLUBS), C(rank, Suit.DIAMONDS)) if rank is triple_sub_target else
                              (C(rank, Suit.CLUBS), C(rank, Suit.DIAMONDS), C(rank, Suit.SPADES)))) + (h0,)
    triple_max_cards = tuple(C(rank, suit) for rank in max_triple_window for suit in (Suit.CLUBS, Suit.DIAMONDS)) + (h0, h1)
    specs.extend([
        PositiveSpec(f"matrix_{level_rank.name.lower()}_triple_sequence_wild_natural", CombinationKind.TRIPLE_SEQUENCE,
            triple_nat_cards, ("matrix_wild_natural",), triple_window[-1], ((h0, level_rank, Suit.HEARTS, "natural"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_triple_sequence_wild_substituted_level_rank", CombinationKind.TRIPLE_SEQUENCE,
            triple_nat_cards, ("matrix_wild_substituted_level_rank",), triple_window[-1], ((h0, level_rank, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_triple_sequence_wild_substituted", CombinationKind.TRIPLE_SEQUENCE,
            triple_sub_cards, ("matrix_wild_substituted",), triple_window[-1], ((h0, triple_sub_target, Suit.CLUBS, "substituted"),)),
        PositiveSpec(f"matrix_{level_rank.name.lower()}_triple_sequence_wild_max", CombinationKind.TRIPLE_SEQUENCE,
            triple_max_cards, ("matrix_wild_max",), max_triple_window[-1],
            ((h0, max_triple_window[0], Suit.CLUBS, "substituted"), (h1, max_triple_window[1], Suit.CLUBS, "substituted"))),
    ])
    return tuple(specs)


def _matrix_case(level_rank: Rank, spec: PositiveSpec) -> dict[str, Any]:
    declaration, cpp_declaration = _choose_declaration(spec, level_rank)
    execution = _execute_positive(declaration, cpp_declaration)
    parity = execution["parity"]
    if not all(parity.values()):
        raise AssertionError(f"matrix parity flag failed for {spec.vector_id}")
    canonical = _normalize_declaration(declaration)
    token_payload = json.dumps(execution["complete_token_sequence"], separators=(",", ":")).encode("ascii")
    row_payload = json.dumps(execution["marvel_v2_action_rows"], separators=(",", ":")).encode("ascii")
    return {
        "id": spec.vector_id, "family": spec.family.value.upper(),
        "case_type": spec.coverage[0], "coverage": list(spec.coverage),
        "passed": all(parity.values()), "physical_card_ids": list(spec.card_ids),
        "comparison_rank": canonical["comparison_rank"], "comparison_key": canonical["comparison_key"],
        "wildcard_count": len(canonical["wild_assignments"]),
        "wild_assignments": canonical["wild_assignments"],
        "complete_token_sequence_sha256": _sha256_bytes(token_payload),
        "marvel_action_rows_sha256": _sha256_bytes(row_payload),
        "canonical_declaration_sha256": _sha256_bytes(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "checks": parity,
    }


def _invalid_level_results() -> list[dict[str, Any]]:
    results = []
    probe = [C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.CLUBS), C(Rank.NINE, Suit.DIAMONDS)]
    for level_rank in INVALID_LEVELS:
        py_rejected = not recognize_combinations(tuple(Card(value) for value in probe), level_rank=level_rank)
        native_rejected = not cpp_game.recognize_combinations(probe, int(level_rank))
        try:
            GameConfig(level_rank=int(level_rank))
        except ValueError:
            env_rejected = True
        else:
            env_rejected = False
        if not (py_rejected and native_rejected and env_rejected):
            raise AssertionError(f"invalid joker level was accepted: {level_rank.name}")
        results.append({"level": level_rank.name, "python_recognizer_rejected": py_rejected,
                        "native_recognizer_rejected": native_rejected, "environment_config_rejected": env_rejected})
    return results


def _build_level_parity_matrix() -> dict[str, Any]:
    levels = []
    all_cases: list[dict[str, Any]] = []
    for level_rank in LEGAL_LEVELS:
        cases = [_matrix_case(level_rank, spec) for spec in _matrix_specs(level_rank)]
        # FOUR_KINGS wildcard rejection and encoding shape are checked per level.
        four = next(case for case in cases if case["family"] == "FOUR_KINGS")
        four_spec = next(spec for spec in _matrix_specs(level_rank) if spec.family is CombinationKind.FOUR_KINGS)
        four_decl, four_cpp = _choose_declaration(four_spec, level_rank)
        four_exec = _execute_positive(four_decl, four_cpp)
        forbidden_kinds = {"RANK", "SUIT", "LENGTH", "WILD_ASSIGNMENT"}
        if any(item["kind"] in forbidden_kinds for item in four_exec["annotated_tokens"]):
            raise AssertionError("FOUR_KINGS emitted forbidden declaration token")
        if _normalize_declaration(four_decl)["canonical_action_card_ids"] != [52, 53, 106, 107]:
            raise AssertionError("FOUR_KINGS canonical action card order changed")
        if list(four_decl.physical_ids()) != [52, 106, 53, 107]:
            raise AssertionError("FOUR_KINGS declaration storage order changed")
        invalid_four_ids = (52, 106, 53, C(level_rank, Suit.HEARTS, 0))
        four_reject = _family_absent(invalid_four_ids, CombinationKind.FOUR_KINGS, level_rank)
        four["four_kings_contract"] = {
            "wildcard_rejected": all(four_reject.values()),
            "forbidden_declaration_tokens_absent": True,
            "canonical_action_card_ids": [52, 53, 106, 107],
            "declaration_storage_card_ids": [52, 106, 53, 107],
        }
        level_passed = all(case["passed"] for case in cases) and all(four_reject.values())
        if not level_passed:
            raise AssertionError(f"level matrix failed for {level_rank.name}")
        levels.append({
            "level": _rank(level_rank), "passed": level_passed, "case_count": len(cases),
            "natural_family_count": sum("matrix_natural" in case["coverage"] for case in cases),
            "natural_wildcard_family_count": sum("matrix_wild_natural" in case["coverage"] for case in cases),
            "substituted_wildcard_family_count": sum("matrix_wild_substituted" in case["coverage"] for case in cases),
            "substituted_level_rank_wildcard_family_count": sum("matrix_wild_substituted_level_rank" in case["coverage"] for case in cases),
            "maximum_wildcard_family_count": sum("matrix_wild_max" in case["coverage"] for case in cases),
            "cases": cases,
        })
        all_cases.extend(cases)
    invalid = _invalid_level_results()
    natural = [case for case in all_cases if "matrix_natural" in case["coverage"]]
    natural_wild = [case for case in all_cases if "matrix_wild_natural" in case["coverage"]]
    substituted = [case for case in all_cases if "matrix_wild_substituted" in case["coverage"]]
    substituted_level = [case for case in all_cases if "matrix_wild_substituted_level_rank" in case["coverage"]]
    maximum = [case for case in all_cases if "matrix_wild_max" in case["coverage"]]
    checks = [check for case in all_cases for check in [case["checks"]]]
    return {
        "levels_tested": [rank.name for rank in LEGAL_LEVELS],
        "invalid_levels_rejected": [item["level"] for item in invalid],
        "families_tested": [family.value.upper() for family in FAMILIES],
        "matrix_case_count": len(all_cases), "cases_per_level": len(all_cases) // len(LEGAL_LEVELS),
        "natural_cases_passed": bool(natural) and all(case["passed"] for case in natural),
        "natural_wildcard_cases_passed": bool(natural_wild) and all(case["passed"] for case in natural_wild),
        "substituted_wildcard_cases_passed": bool(substituted) and all(case["passed"] for case in substituted),
        "substituted_level_rank_wildcard_cases_passed": bool(substituted_level) and all(case["passed"] for case in substituted_level),
        "maximum_wildcard_cases_passed": bool(maximum) and all(case["passed"] for case in maximum),
        "python_native_recognizer_parity": all(check["python_recognizer_equals_native"] for check in checks),
        "python_native_token_parity": all(check["python_tokens_equal_native_tokens"] for check in checks),
        "python_native_action_row_parity": all(check["python_rows_equal_native_rows"] for check in checks),
        "python_native_decode_parity": all(check["python_row_decode_equal"] and check["native_row_decode_equal"] for check in checks),
        "python_native_environment_commit_parity": all(check["python_environment_commit_equal"] and check["native_environment_commit_equal"] for check in checks),
        "invalid_level_results": invalid, "levels": levels,
    }


def _leveled_positive_vector(level_rank: Rank, spec: PositiveSpec) -> dict[str, Any]:
    declaration, cpp_declaration = _choose_declaration(spec, level_rank)
    execution = _execute_positive(declaration, cpp_declaration)
    return {
        "id": spec.vector_id, "family": spec.family.value.upper(), "polarity": "positive",
        "coverage": list(spec.coverage), "level_rank": _rank(level_rank),
        "physical_cards": [_card_record(card_id, level_rank) for card_id in spec.card_ids],
        "expected": {"result": "legal", "reason_code": "LEGAL_CANONICAL_DECLARATION",
                     "canonical_declaration": _normalize_declaration(declaration), **execution},
    }


def _special_boundary_vectors() -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    for level_rank in (Rank.THREE, Rank.ACE, Rank.TWO):
        h0 = C(level_rank, Suit.HEARTS, 0)
        other0, other1 = _other_ranks(level_rank, 2)
        matrix = {spec.vector_id: spec for spec in _matrix_specs(level_rank)}
        wanted = [
            f"matrix_{level_rank.name.lower()}_full_house_wild_natural",
            f"matrix_{level_rank.name.lower()}_full_house_wild_substituted",
            f"matrix_{level_rank.name.lower()}_full_house_wild_substituted_level_rank",
            f"matrix_{level_rank.name.lower()}_straight_wild_natural",
            f"matrix_{level_rank.name.lower()}_straight_wild_substituted",
            f"matrix_{level_rank.name.lower()}_straight_wild_substituted_level_rank",
            f"matrix_{level_rank.name.lower()}_straight_flush_wild_natural",
            f"matrix_{level_rank.name.lower()}_straight_flush_wild_substituted_level_rank",
            f"matrix_{level_rank.name.lower()}_pair_sequence_wild_substituted_level_rank",
            f"matrix_{level_rank.name.lower()}_triple_sequence_wild_substituted_level_rank",
        ]
        for key in wanted:
            original = matrix[key]
            tagged = PositiveSpec(
                f"special_{key.removeprefix('matrix_')}", original.family, original.card_ids,
                (f"special_level_{level_rank.name}", *original.coverage), original.comparison_rank, original.assignments,
            )
            vectors.append(_leveled_positive_vector(level_rank, tagged))

        # FULL_HOUSE with level rank specifically in the pair role.
        pair_spec = PositiveSpec(
            f"special_{level_rank.name.lower()}_full_house_level_pair", CombinationKind.FULL_HOUSE,
            (C(other0, Suit.CLUBS), C(other0, Suit.DIAMONDS), C(other0, Suit.SPADES), C(level_rank, Suit.CLUBS), h0),
            (f"special_level_{level_rank.name}", "level_rank_as_full_house_pair"), other0,
            ((h0, level_rank, Suit.HEARTS, "natural"),),
        )
        vectors.append(_leveled_positive_vector(level_rank, pair_spec))

    # Explicit low/high windows and hearts straight flushes for ACE and TWO.
    for level_rank in (Rank.ACE, Rank.TWO):
        h0 = C(level_rank, Suit.HEARTS, 0)
        low_cards = tuple(h0 if rank is level_rank else C(rank, Suit.HEARTS)
                          for rank in (Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE))
        low_assignment = ((h0, level_rank, Suit.HEARTS, "natural"),)
        low_mixed_cards = tuple(h0 if rank is level_rank else C(rank, (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES)[index % 3])
                                for index, rank in enumerate((Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE)))
        vectors.append(_leveled_positive_vector(level_rank, PositiveSpec(
            f"special_{level_rank.name.lower()}_straight_a2345", CombinationKind.STRAIGHT,
            low_mixed_cards, (f"special_level_{level_rank.name}", "a2345"), Rank.FIVE, low_assignment)))
        vectors.append(_leveled_positive_vector(level_rank, PositiveSpec(
            f"special_{level_rank.name.lower()}_straight_flush_hearts_a2345", CombinationKind.STRAIGHT_FLUSH,
            low_cards, (f"special_level_{level_rank.name}", "a2345", "straight_flush_hearts"), Rank.FIVE, low_assignment)))
        high_cards = tuple(C(rank, Suit.CLUBS) for rank in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE))
        vectors.append(_leveled_positive_vector(level_rank, PositiveSpec(
            f"special_{level_rank.name.lower()}_straight_high_10jqka", CombinationKind.STRAIGHT,
            high_cards, (f"special_level_{level_rank.name}", "10jqka"), Rank.ACE)))

    # TWO-specific A23 and A2 windows.
    vectors.append(_leveled_positive_vector(Rank.TWO, PositiveSpec(
        "special_two_pair_sequence_a23", CombinationKind.PAIR_SEQUENCE,
        tuple(C(rank, suit) for rank in (Rank.ACE, Rank.TWO, Rank.THREE) for suit in (Suit.CLUBS, Suit.DIAMONDS)),
        ("special_level_TWO", "pair_sequence_a23"), Rank.THREE)))
    vectors.append(_leveled_positive_vector(Rank.TWO, PositiveSpec(
        "special_two_triple_sequence_a2", CombinationKind.TRIPLE_SEQUENCE,
        tuple(C(rank, suit) for rank in (Rank.ACE, Rank.TWO) for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES)),
        ("special_level_TWO", "triple_sequence_a2"), Rank.TWO)))

    # Explicit invalid wrap evidence is physical recognizer/generator rejection.
    invalids = [
        (Rank.ACE, "special_ace_invalid_jqka2", CombinationKind.STRAIGHT,
         tuple(C(rank, (Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES, Suit.CLUBS, Suit.DIAMONDS)[index])
               for index, rank in enumerate((Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE, Rank.TWO))), "J_Q_K_A_2_WRAP_FORBIDDEN"),
        (Rank.TWO, "special_two_invalid_pair_sequence_ka2", CombinationKind.PAIR_SEQUENCE,
         tuple(C(rank, suit) for rank in (Rank.KING, Rank.ACE, Rank.TWO) for suit in (Suit.CLUBS, Suit.DIAMONDS)), "K_A_2_WRAP_FORBIDDEN"),
    ]
    for level_rank, vector_id, family, card_ids, reason in invalids:
        checks = _family_absent(card_ids, family, level_rank)
        vectors.append({
            "id": vector_id, "family": family.value.upper(), "polarity": "negative",
            "coverage": [f"special_level_{level_rank.name}", "invalid_wrap"], "level_rank": _rank(level_rank),
            "candidate": {"physical_cards": [_card_record(card_id, level_rank) for card_id in card_ids],
                          "canonical_declaration": None, "complete_token_sequence": None},
            "expected": {"result": "invalid", "reason_code": reason, "authority_checks": checks,
                         "reference_canonical_declaration": None, "reference_complete_token_sequence": None},
        })
    return vectors

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-at", help="fixed ISO-8601 timestamp; defaults to current Asia/Shanghai time")
    parser.add_argument("--check", action="store_true", help="regenerate in memory and verify files and SHA256SUMS")
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        result = _check(output_dir) if args.check else _write(output_dir, args.generated_at or _now_china())
    except Exception as exc:
        print(json.dumps({"exit_code": 1, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())































