from __future__ import annotations

import numpy as np
import pytest

from guandan.action_state import (
    CARD_TOKEN_BASE,
    COMMIT_TOKEN,
    FAMILY_TOKEN_BASE,
    MAX_ACTION_TOKENS,
    MAX_LEGAL_NEXT_TOKENS,
    PASS_TOKEN,
    ProtocolError,
    StepwiseActionState,
    TokenCodec,
    canonical_tokens,
)
from guandan.cards import Rank, Suit, cards_for_rank, full_deck
from guandan.combos import CombinationKind
from guandan.round import enumerate_legal_actions
from guandan.state import CommittedAction, EpisodeTerminatedError, IllegalActionError, PreviousHandResult, RoundState


def partition(hand: list) -> list[list]:
    used = {card.card_id for card in hand}
    rest = [card for card in full_deck() if card.card_id not in used]
    hands = [list(hand), [], [], []]
    for index, card in enumerate(rest):
        hands[1 + index % 3].append(card)
    return hands


def reduced_state() -> RoundState:
    hand = list(cards_for_rank(Rank.THREE)[:2]) + list(cards_for_rank(Rank.FOUR)[:3]) + list(cards_for_rank(Rank.FIVE)[:2])
    return RoundState.from_hands(partition(hand), level_rank=Rank.SIX)


def action_key(action: CommittedAction) -> tuple:
    declaration = action.declaration
    return (
        action.kind.value if isinstance(action.kind, CombinationKind) else str(action.kind),
        tuple(card.card_id for card in action.cards),
        None if declaration is None else (declaration.kind.value, declaration.physical_ids(), declaration.comparison_key, tuple(sorted(declaration.wild_assignments.items()))),
    )


def commit_path(protocol: StepwiseActionState, action: CommittedAction):
    result = None
    for token in canonical_tokens(action):
        result = protocol.step(token)
    assert result is not None and result.is_commit
    return result


def test_public_api_and_v1_token_ranges() -> None:
    assert TokenCodec.card(0) == CARD_TOKEN_BASE
    assert TokenCodec.card(107) == 171
    assert TokenCodec.family(CombinationKind.SINGLE) == FAMILY_TOKEN_BASE
    assert TokenCodec.rank(Rank.THREE) == 18
    assert TokenCodec.wild(Rank.THREE, Suit.CLUBS) == 172
    state = StepwiseActionState(reduced_state())
    assert all(0 < int(token) < 256 for token in state.legal_tokens())
    assert 0 not in set(int(token) for token in state.legal_tokens())
    assert state.max_action_tokens == MAX_ACTION_TOKENS
    assert state.max_legal_next_tokens == MAX_LEGAL_NEXT_TOKENS


def test_each_legal_action_has_a_reachable_prefix_path() -> None:
    state = reduced_state()
    actions = enumerate_legal_actions(state)
    assert actions
    for action in actions:
        result = commit_path(StepwiseActionState(state.clone()), action)
        assert action_key(result.committed_action.atomic_action) == action_key(action)


def test_no_dead_prefix_is_exposed() -> None:
    protocol = StepwiseActionState(reduced_state())
    frontier = [protocol]
    seen = set()
    for _depth in range(4):
        next_frontier = []
        for current in frontier:
            key = current.prefix.tokens
            if key in seen:
                continue
            seen.add(key)
            tokens = current.legal_tokens()
            assert len(tokens) > 0
            for token in tokens[: min(12, len(tokens))]:
                child = current.clone()
                result = child.step(int(token))
                if not result.is_commit and not child.truncated:
                    next_frontier.append(child)
        frontier = next_frontier


def test_each_exposed_commit_is_one_legal_canonical_action() -> None:
    state = reduced_state()
    expected = {action_key(action) for action in enumerate_legal_actions(state)}
    observed = set()
    for action in enumerate_legal_actions(state):
        result = commit_path(StepwiseActionState(state.clone()), action)
        observed.add(action_key(result.committed_action.atomic_action))
    assert observed == expected


def test_stepwise_set_matches_independent_complete_action_enumerator() -> None:
    state = reduced_state()
    expected = {action_key(action) for action in enumerate_legal_actions(state)}
    observed = {action_key(commit_path(StepwiseActionState(state.clone()), action).committed_action.atomic_action) for action in enumerate_legal_actions(state)}
    assert observed == expected


def test_canonicalization_removes_card_order_and_full_house_permutations() -> None:
    state = reduced_state()
    sequences = [canonical_tokens(action) for action in enumerate_legal_actions(state)]
    assert len(sequences) == len(set(sequences))
    pairs = [sequence for sequence in sequences if sequence[0] == TokenCodec.family(CombinationKind.PAIR)]
    assert pairs and len(pairs) == len(set(pairs))


def test_wild_assignments_are_canonical_and_joker_assignments_rejected() -> None:
    wilds = [card for card in cards_for_rank(Rank.FIVE) if card.suit is Suit.HEARTS]
    hand = list(cards_for_rank(Rank.TWO)[:1]) + list(cards_for_rank(Rank.THREE)[:1]) + list(cards_for_rank(Rank.FOUR)[:1]) + list(cards_for_rank(Rank.SIX)[:1]) + [wilds[0]]
    state = RoundState.from_hands(partition(hand), level_rank=Rank.FIVE)
    actions = enumerate_legal_actions(state)
    sequences = [canonical_tokens(action) for action in actions if action.declaration is not None and action.declaration.kind is CombinationKind.STRAIGHT]
    assert sequences
    assert len(sequences) == len(set(sequences))
    assert all(not (172 <= token <= 223 and TokenCodec.wild_from_token(token)[0] in {Rank.SMALL_JOKER, Rank.BIG_JOKER}) for sequence in sequences for token in sequence)


def test_phase_specific_grammar_and_tribute_return_eligibility() -> None:
    previous = PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD")
    protocol = StepwiseActionState(StepwiseActionState.deal(seed=0, level_rank=Rank.FIVE, previous_result=previous).round_state)
    assert protocol.round_state.phase.value == "tribute"
    assert all(64 <= int(token) <= 171 for token in protocol.legal_tokens())
    tribute = enumerate_legal_actions(protocol.round_state)[0]
    commit_path(protocol, tribute)
    assert all(64 <= int(token) <= 171 for token in protocol.legal_tokens())


def test_prefix_tokens_are_private_and_transactional() -> None:
    protocol = StepwiseActionState(reduced_state())
    before = protocol.round_state.serialize()
    token = int(protocol.legal_tokens()[0])
    result = protocol.step(token)
    assert not result.is_commit
    assert protocol.round_state.serialize() == before
    assert protocol.token_step == 1 and protocol.committed_step == 0
    with pytest.raises(IllegalActionError):
        protocol.step(0)
    assert protocol.token_step == 1 and protocol.round_state.serialize() == before


def test_commit_mutates_once_and_advances_both_counters_once() -> None:
    protocol = StepwiseActionState(reduced_state())
    action = next(action for action in enumerate_legal_actions(protocol.round_state) if action.kind is CombinationKind.SINGLE)
    result = commit_path(protocol, action)
    assert result.is_commit and result.committed_step == 1
    assert protocol.token_step == len(canonical_tokens(action))
    assert len(protocol.round_state.history) == 1


def test_pass_is_one_token_commit_only_in_follow_phase() -> None:
    state = reduced_state()
    action = next(action for action in enumerate_legal_actions(state) if action.kind is CombinationKind.SINGLE)
    from guandan.round import apply_action
    apply_action(state, action)
    protocol = StepwiseActionState(state)
    assert PASS_TOKEN in set(int(token) for token in protocol.legal_tokens())
    result = protocol.step(PASS_TOKEN)
    assert result.is_commit and result.token_step == 1 and result.committed_step == state.committed_step


def test_prefix_reward_is_zero_and_terminal_reward_matches_environment_contract() -> None:
    protocol = StepwiseActionState(reduced_state())
    result = protocol.step(int(protocol.legal_tokens()[0]))
    np.testing.assert_array_equal(result.rewards, np.zeros(4, dtype=np.float32))
    assert result.committed_action is None


def test_illegal_tokens_and_incomplete_commit_are_atomic() -> None:
    protocol = StepwiseActionState(reduced_state())
    before = protocol.serialize()
    with pytest.raises(IllegalActionError):
        protocol.step(0)
    assert protocol.serialize() == before
    with pytest.raises(IllegalActionError):
        protocol.step(COMMIT_TOKEN)
    assert protocol.serialize() == before


def test_action_token_overflow_raises_protocol_error_without_truncation() -> None:
    protocol = StepwiseActionState(reduced_state(), max_action_tokens=1)
    with pytest.raises(ProtocolError):
        protocol.legal_tokens()


def test_legal_next_token_overflow_raises_protocol_error_without_truncation() -> None:
    protocol = StepwiseActionState(reduced_state(), max_legal_next_tokens=1)
    with pytest.raises(ProtocolError):
        protocol.legal_tokens()


def test_max_token_steps_truncates_with_reason_and_no_win_loss_reward() -> None:
    protocol = StepwiseActionState(reduced_state(), max_token_steps=1)
    result = protocol.step(int(protocol.legal_tokens()[0]))
    assert result.truncated and result.info["termination_reason"] == "max_token_steps"
    np.testing.assert_array_equal(result.rewards, np.zeros(4, dtype=np.float32))
    with pytest.raises(EpisodeTerminatedError):
        protocol.step(0)


def test_serialize_clone_round_trip_preserves_prefix_and_legal_tokens() -> None:
    protocol = StepwiseActionState(reduced_state())
    protocol.step(int(protocol.legal_tokens()[0]))
    restored = StepwiseActionState.deserialize(protocol.serialize())
    clone = protocol.clone()
    assert restored.prefix == protocol.prefix == clone.prefix
    assert restored.legal_tokens().tolist() == protocol.legal_tokens().tolist() == clone.legal_tokens().tolist()
    assert restored.token_step == protocol.token_step == clone.token_step


def test_policy_observation_does_not_expose_opponent_hands() -> None:
    protocol = StepwiseActionState(reduced_state())
    prefix = protocol.prefix
    assert not hasattr(prefix, "hands")
    assert not hasattr(prefix, "opponent_hands")
    assert not any(card_id in prefix.selected_card_ids for card_id in [card.card_id for card in protocol.round_state.hands[1]])
