"""Directed GuanDan action cases for M02 adapter execution verification."""
from __future__ import annotations

from dataclasses import dataclass

from guandan.cards import Card, Rank, Suit, full_deck
from guandan.action_state import StepwiseActionState, TokenCodec, canonical_tokens
from guandan.combos import CombinationKind, recognize_combinations
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan.state import CommittedAction, RoundState


@dataclass(frozen=True, slots=True)
class DirectedActionCase:
    name: str
    family: CombinationKind
    card_ids: tuple[int, ...]
    comparison_rank: Rank
    assignments: tuple[tuple[int, Rank, Suit], ...] = ()
    natural_wild_ids: tuple[int, ...] = ()

    @property
    def wild_mode(self) -> str:
        assigned = len(self.assignments)
        natural = len(self.natural_wild_ids)
        if assigned > 1 and natural:
            return "mixed_multiple"
        if assigned > 1:
            return "substitute_multiple"
        if assigned and natural:
            return "natural"
        if assigned:
            return "substitute"
        return "none"


# All cases use level FOUR; physical level-wild cards are 6 and 60.
DIRECTED_ACTION_CASES = (
    DirectedActionCase("single", CombinationKind.SINGLE, (0,), Rank.THREE),
    DirectedActionCase(
        "single-natural-wild", CombinationKind.SINGLE, (6,), Rank.FOUR,
        ((6, Rank.FOUR, Suit.HEARTS),), (6,),
    ),
    DirectedActionCase("pair", CombinationKind.PAIR, (54, 0), Rank.THREE),
    DirectedActionCase("triple", CombinationKind.TRIPLE, (2, 0, 1), Rank.THREE),
    DirectedActionCase(
        "full-house-two-wild-targets", CombinationKind.FULL_HOUSE,
        (60, 0, 2, 6, 1), Rank.THREE,
        ((6, Rank.FIVE, Suit.CLUBS), (60, Rank.FIVE, Suit.CLUBS)),
    ),
    DirectedActionCase(
        "full-house-natural-and-substituted", CombinationKind.FULL_HOUSE,
        (60, 0, 2, 6, 1), Rank.THREE,
        ((6, Rank.FOUR, Suit.HEARTS), (60, Rank.FOUR, Suit.CLUBS)), (6,),
    ),
    DirectedActionCase(
        "straight", CombinationKind.STRAIGHT, (0, 4, 8, 12, 16), Rank.SEVEN,
    ),
    DirectedActionCase(
        "pair-sequence", CombinationKind.PAIR_SEQUENCE,
        (0, 54, 4, 58, 8, 62), Rank.FIVE,
    ),
    DirectedActionCase(
        "triple-sequence", CombinationKind.TRIPLE_SEQUENCE,
        (0, 1, 2, 4, 5, 7), Rank.FOUR,
    ),
    DirectedActionCase(
        "rank-bomb-ten-two-wilds", CombinationKind.RANK_BOMB,
        (0, 1, 2, 3, 54, 55, 56, 57, 6, 60), Rank.THREE,
        ((6, Rank.THREE, Suit.CLUBS), (60, Rank.THREE, Suit.CLUBS)),
    ),
    DirectedActionCase(
        "straight-flush-substituted-wild", CombinationKind.STRAIGHT_FLUSH,
        (3, 6, 11, 15, 19), Rank.SEVEN,
        ((6, Rank.FOUR, Suit.SPADES),),
    ),
    DirectedActionCase(
        "straight-flush-natural-wild", CombinationKind.STRAIGHT_FLUSH,
        (2, 6, 10, 14, 18), Rank.SEVEN,
        ((6, Rank.FOUR, Suit.HEARTS),), (6,),
    ),
    DirectedActionCase(
        "four-kings", CombinationKind.FOUR_KINGS, (107, 52, 106, 53), Rank.BIG_JOKER,
    ),
)


def build_directed_action(case: DirectedActionCase) -> tuple[RoundState, CommittedAction]:
    if not isinstance(case, DirectedActionCase):
        raise TypeError("case must be DirectedActionCase")
    cards = tuple(Card(card_id) for card_id in case.card_ids)
    assignments = {card_id: (rank, suit) for card_id, rank, suit in case.assignments}
    declarations = [
        declaration
        for declaration in recognize_combinations(cards, level_rank=Rank.FOUR)
        if declaration.kind is case.family
        and declaration.comparison_rank is case.comparison_rank
        and declaration.wild_assignments == assignments
        and declaration.natural_wild_ids == case.natural_wild_ids
    ]
    if len(declarations) != 1:
        raise AssertionError(f"directed case {case.name} resolved to {len(declarations)} declarations")
    used = set(case.card_ids)
    remaining = [card for card in full_deck() if card.card_id not in used]
    extra = remaining.pop(0)
    hands = [list(cards) + [extra], [], [], []]
    for index, card in enumerate(remaining):
        hands[1 + index % 3].append(card)
    state = RoundState.from_hands(hands, level_rank=Rank.FOUR, leader_seat=0)
    action = CommittedAction(0, case.family, cards, declarations[0])
    return state, action


def execute_directed_case(case: DirectedActionCase, *, verify_serialization: bool = True) -> dict[str, object]:
    """Execute a complete V2 prefix/COMMIT path against an independent V1 clone."""
    import numpy as np
    from guandan_marvel import python_game as game

    state, action = build_directed_action(case)
    config = GameConfig(level_rank=state.level_rank, leader_seat=0, max_token_steps=game.max_game_horizon)
    env = GuandanEnv(config)
    env.reset(initial_state=StepwiseActionState(state.clone(), max_token_steps=game.max_game_horizon).serialize())
    oracle = GuandanEnv.deserialize(env.serialize())
    batch = game.EnvBatch.from_envs([env], auto_reset=False)
    step = batch.observe()
    sequence = canonical_tokens(action)
    max_candidates = 0
    serialized_at_prefix = False
    for index, token in enumerate(sequence):
        mask = step.available_actions.weights[0] > 0
        rows = step.available_actions.tokens[0, mask]
        decoded = game.actions_from_np(rows)
        matches = [row_index for row_index, candidate in enumerate(decoded) if candidate.token_id == token]
        if len(matches) != 1:
            raise AssertionError(f"case {case.name} token {token} does not have exactly one V2 row")
        max_candidates = max(max_candidates, len(decoded))
        chosen = decoded[matches[0]]
        expected = oracle.step(int(token))
        step = batch.step((chosen,))
        if batch.envs[0].serialize() != oracle.serialize():
            raise AssertionError(f"case {case.name} diverged from V1 at token index {index}")
        if bool(step.is_terminal[0]) != expected.done:
            raise AssertionError("V1/V2 terminal mismatch")
        np.testing.assert_array_equal(step.rewards[0], expected.rewards)
        if verify_serialization and not serialized_at_prefix and 0 < index < len(sequence) - 1:
            restored = game.EnvBatch.deserialize(batch.serialize())
            current = batch.observe()
            resumed = restored.observe()
            np.testing.assert_array_equal(current.observations.tokens, resumed.observations.tokens)
            np.testing.assert_array_equal(current.observations.channels, resumed.observations.channels)
            np.testing.assert_array_equal(current.available_actions.tokens, resumed.available_actions.tokens)
            np.testing.assert_array_equal(current.available_actions.weights, resumed.available_actions.weights)
            batch = restored
            step = resumed
            serialized_at_prefix = True

    if not chosen.is_commit or sequence[-1] != 7:
        raise AssertionError("directed action did not execute COMMIT")
    event_rows = step.observations.tokens[0]
    if not np.all(event_rows == event_rows[0]):
        raise AssertionError("completed play event must be public to all four seats")
    event = event_rows[0]
    nonzero = event[event != 0]
    if len(nonzero) > game.max_observation_tokens or not np.all((0 < nonzero) & (nonzero < game.num_tokens)):
        raise AssertionError("public event violates token vocabulary/capacity")
    expected_event = np.asarray(
        [1, game.EVENT_PUBLIC_COMMIT, 2, TokenCodec.length(0), *sequence, game.EVENT_END],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(nonzero, expected_event)
    final_state = batch.envs[0]._require_protocol().round_state
    final_state.check_invariants()
    if final_state.history[-1].kind is not case.family:
        raise AssertionError("committed family mismatch")
    remaining = [card.card_id for hand in final_state.hands for card in hand]
    played = [
        card.card_id for item in final_state.history
        if item.kind not in ("pass", "tribute", "return") for card in item.cards
    ]
    if len(remaining) + len(played) != 108 or len(set(remaining + played)) != 108:
        raise AssertionError("physical card conservation failed")
    return {
        "name": case.name,
        "family": case.family.value,
        "level_rank": int(Rank.FOUR),
        "wild_mode": case.wild_mode,
        "wild_assignments": len(case.assignments),
        "natural_wilds": len(case.natural_wild_ids),
        "prefix_tokens": len(sequence) - 1,
        "canonical_tokens": list(sequence),
        "max_candidates": max_candidates,
        "commit_executed": True,
        "serialization_resume": serialized_at_prefix,
        "v1_transition_equal": True,
        "public_event_tokens": len(nonzero),
        "real_terminal": bool(step.is_terminal[0]),
        "card_conservation": True,
    }


if {case.family for case in DIRECTED_ACTION_CASES} != set(CombinationKind):
    raise AssertionError("directed action inventory must cover every current CombinationKind")


__all__ = ["DIRECTED_ACTION_CASES", "DirectedActionCase", "build_directed_action", "execute_directed_case"]
