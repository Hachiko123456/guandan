from __future__ import annotations

import numpy as np
import pytest

from guandan.action_state import (
    CARD_TOKEN_BASE, COMMIT_TOKEN, FAMILY_TOKEN_BASE, PASS_TOKEN,
    ProtocolError, StepwiseActionState, TokenCodec, canonical_tokens,
)
from guandan.cards import Rank, Suit, cards_for_rank, full_deck
from guandan.combos import CombinationKind
from guandan.round import apply_action, enumerate_legal_actions
from guandan.state import CommittedAction, EpisodeTerminatedError, IllegalActionError, PreviousHandResult, RoundState


def _partition(hand):
    used = {card.card_id for card in hand}
    rest = [card for card in full_deck() if card.card_id not in used]
    hands = [list(hand), [], [], []]
    for index, card in enumerate(rest):
        hands[1 + index % 3].append(card)
    return hands


def _state():
    hand = list(cards_for_rank(Rank.THREE)[:2]) + list(cards_for_rank(Rank.FOUR)[:3]) + list(cards_for_rank(Rank.FIVE)[:2])
    return RoundState.from_hands(_partition(hand), level_rank=Rank.SIX)


def _commit(protocol, action):
    result = None
    for token in canonical_tokens(action):
        result = protocol.step(token)
    assert result is not None and result.is_commit
    return result


def _key(action):
    return (str(action.kind), tuple(card.card_id for card in action.cards), None if action.declaration is None else action.declaration.comparison_key)


def test_public_api_and_v1_token_ranges():
    assert TokenCodec.card(0) == CARD_TOKEN_BASE and TokenCodec.card(107) == 171
    assert TokenCodec.family(CombinationKind.SINGLE) == FAMILY_TOKEN_BASE
    protocol = StepwiseActionState(_state())
    assert all(0 < int(token) < 256 for token in protocol.legal_tokens())


def test_each_legal_action_has_a_reachable_prefix_path():
    protocol_state = _state()
    actions = enumerate_legal_actions(protocol_state)
    assert actions
    for action in actions[:10]:
        result = _commit(StepwiseActionState(protocol_state.clone()), action)
        assert _key(result.committed_action.atomic_action) == _key(action)


def test_no_dead_prefix_is_exposed():
    protocol = StepwiseActionState(_state())
    for _ in range(3):
        tokens = protocol.legal_tokens()
        assert len(tokens) > 0
        protocol.step(int(tokens[0]))
        if protocol.done or protocol.truncated:
            break


def test_each_exposed_commit_is_one_legal_canonical_action():
    state = _state()
    expected = {_key(action) for action in enumerate_legal_actions(state)}
    observed = {_key(_commit(StepwiseActionState(state.clone()), action).committed_action.atomic_action) for action in enumerate_legal_actions(state)}
    assert observed == expected


def test_stepwise_set_matches_independent_complete_action_enumerator():
    state = _state()
    expected = {_key(action) for action in enumerate_legal_actions(state)}
    observed = set()
    for action in enumerate_legal_actions(state):
        result = _commit(StepwiseActionState(state.clone()), action)
        observed.add(_key(result.committed_action.atomic_action))
    assert observed == expected


def test_canonicalization_removes_card_order_and_full_house_permutations():
    sequences = [canonical_tokens(action) for action in enumerate_legal_actions(_state())]
    assert len(sequences) == len(set(sequences))


def test_wild_assignments_are_canonical_and_joker_assignments_rejected():
    wild = next(card for card in cards_for_rank(Rank.FIVE) if card.suit is Suit.HEARTS)
    hand = [cards_for_rank(rank)[0] for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.SIX)] + [wild]
    state = RoundState.from_hands(_partition(hand), level_rank=Rank.FIVE)
    sequences = [canonical_tokens(action) for action in enumerate_legal_actions(state) if action.declaration is not None]
    assert sequences and all(not (TokenCodec.wild_from_token(token)[0] in {Rank.SMALL_JOKER, Rank.BIG_JOKER}) for sequence in sequences for token in sequence if 172 <= token <= 223)


def test_phase_specific_grammar_and_tribute_return_eligibility():
    previous = PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD")
    state = RoundState.deal(seed=0, level_rank=Rank.FIVE, previous_result=previous)
    protocol = StepwiseActionState(state)
    assert all(CARD_TOKEN_BASE <= int(token) <= 171 for token in protocol.legal_tokens())
    _commit(protocol, enumerate_legal_actions(state)[0])
    assert all(CARD_TOKEN_BASE <= int(token) <= 171 for token in protocol.legal_tokens())


def test_prefix_tokens_are_private_and_transactional():
    protocol = StepwiseActionState(_state())
    before = protocol.round_state.serialize()
    protocol.step(int(protocol.legal_tokens()[0]))
    assert protocol.round_state.serialize() == before
    assert protocol.committed_step == 0 and protocol.token_step == 1
    with pytest.raises(IllegalActionError):
        protocol.step(0)
    assert protocol.round_state.serialize() == before


def test_commit_mutates_once_and_advances_both_counters_once():
    protocol = StepwiseActionState(_state())
    action = next(action for action in enumerate_legal_actions(protocol.round_state) if action.kind is CombinationKind.SINGLE)
    result = _commit(protocol, action)
    assert result.is_commit and result.committed_step == 1 and len(protocol.round_state.history) == 1


def test_pass_is_one_token_commit_only_in_follow_phase():
    state = _state()
    action = next(action for action in enumerate_legal_actions(state) if action.kind is CombinationKind.SINGLE)
    apply_action(state, action)
    protocol = StepwiseActionState(state)
    assert PASS_TOKEN in set(int(token) for token in protocol.legal_tokens())
    result = protocol.step(PASS_TOKEN)
    assert result.is_commit and result.committed_step == state.committed_step


def test_prefix_reward_is_zero_and_terminal_reward_matches_environment_contract():
    result = StepwiseActionState(_state()).step(int(StepwiseActionState(_state()).legal_tokens()[0]))
    np.testing.assert_array_equal(result.rewards, np.zeros(4, dtype=np.float32))
    assert result.committed_action is None


def test_illegal_tokens_and_incomplete_commit_are_atomic():
    protocol = StepwiseActionState(_state())
    before = protocol.serialize()
    with pytest.raises(IllegalActionError):
        protocol.step(0)
    with pytest.raises(IllegalActionError):
        protocol.step(COMMIT_TOKEN)
    assert protocol.serialize() == before


def test_action_token_overflow_raises_protocol_error_without_truncation():
    with pytest.raises(ProtocolError):
        StepwiseActionState(_state(), max_action_tokens=1).legal_tokens()


def test_legal_next_token_overflow_raises_protocol_error_without_truncation():
    with pytest.raises(ProtocolError):
        StepwiseActionState(_state(), max_legal_next_tokens=1).legal_tokens()


def test_max_token_steps_truncates_with_reason_and_no_win_loss_reward():
    protocol = StepwiseActionState(_state(), max_token_steps=1)
    result = protocol.step(int(protocol.legal_tokens()[0]))
    assert result.truncated and result.info["termination_reason"] == "max_token_steps"
    np.testing.assert_array_equal(result.rewards, np.zeros(4, dtype=np.float32))
    with pytest.raises(EpisodeTerminatedError):
        protocol.step(0)


def test_serialize_clone_round_trip_preserves_prefix_and_legal_tokens():
    protocol = StepwiseActionState(_state())
    protocol.step(int(protocol.legal_tokens()[0]))
    restored = StepwiseActionState.deserialize(protocol.serialize())
    clone = protocol.clone()
    assert restored.prefix == protocol.prefix == clone.prefix
    assert restored.legal_tokens().tolist() == protocol.legal_tokens().tolist() == clone.legal_tokens().tolist()


def test_policy_observation_does_not_expose_opponent_hands():
    protocol = StepwiseActionState(_state())
    assert not hasattr(protocol.prefix, "hands")
    assert not hasattr(protocol.prefix, "opponent_hands")


def test_legal_token_arrays_are_padded_and_masked():
    protocol = StepwiseActionState(_state())
    tokens, mask, is_commit, kinds, values = protocol.legal_token_arrays()
    assert tokens.shape == mask.shape == is_commit.shape == kinds.shape == values.shape == (protocol.max_legal_next_tokens,)
    assert int(mask.sum()) == len(protocol.legal_tokens())
    assert np.all(tokens[~mask] == 0)
    assert np.all(~is_commit[~mask])
    assert np.all(kinds[~mask] == 0)
    assert np.all(values[~mask] == 0)


def test_commit_rolls_back_when_next_phase_token_capacity_overflows():
    previous = PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD")
    protocol = StepwiseActionState(StepwiseActionState.deal(seed=0, level_rank=Rank.FIVE, previous_result=previous).round_state, max_legal_next_tokens=1)
    tribute_card = int(protocol.legal_tokens()[0])
    protocol.step(tribute_card)
    before = protocol.serialize()
    with pytest.raises(ProtocolError):
        protocol.step(COMMIT_TOKEN)
    assert protocol.serialize() == before
    assert protocol.committed_step == 0
