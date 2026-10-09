from __future__ import annotations

import numpy as np
import pytest

from guandan.action_state import StepwiseActionState, canonical_tokens
from guandan.cards import Card, Rank, full_deck
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan.state import PreviousHandResult, RoundState
from guandan_marvel import python_game
from guandan_marvel.testing.complex_actions import DIRECTED_ACTION_CASES, build_directed_action
from nanopokerlib import guandan as cpp


def _row_for_token(step, token_id: int, batch: int = 0) -> np.ndarray:
    live = step.available_actions.weights[batch] > 0
    matches = np.flatnonzero(live & (step.available_actions.tokens[batch, :, 5] == token_id))
    assert len(matches) == 1, (token_id, matches.tolist())
    return step.available_actions.tokens[batch, matches[0]]


def _oracle_batch(state: RoundState):
    env = GuandanEnv(GameConfig(
        level_rank=int(state.level_rank),
        leader_seat=state.leader_seat,
        previous_result=state.previous_result,
        max_token_steps=4096,
    ))
    env.reset(initial_state=StepwiseActionState(state).serialize())
    return python_game.EnvBatch.from_envs([env], auto_reset=False)


def _deal_ids(state: RoundState):
    return [[card.card_id for card in hand] for hand in state.hands]


@pytest.mark.parametrize("case", DIRECTED_ACTION_CASES, ids=lambda case: case.name)
def test_directed_prefix_commit_rows_events_and_channels_match_python(case):
    state, action = build_directed_action(case)
    oracle = _oracle_batch(state)
    expected = oracle.observe()
    batch = cpp.EnvBatch(1, auto_reset=False)
    actual = batch.reset_from_deals([_deal_ids(state)], level_ranks=[int(state.level_rank)], leader_seats=[0])

    for token_id in canonical_tokens(action):
        np.testing.assert_array_equal(actual.available_actions.tokens, expected.available_actions.tokens)
        np.testing.assert_array_equal(actual.available_actions.weights, expected.available_actions.weights)
        np.testing.assert_array_equal(actual.available_actions.is_commit, expected.available_actions.is_commit)
        np.testing.assert_array_equal(actual.observations.tokens, expected.observations.tokens)
        np.testing.assert_array_equal(actual.observations.channels, expected.observations.channels)
        row = _row_for_token(actual, token_id)[None, :]
        expected = oracle.step(row)
        actual = batch.step(row)

    np.testing.assert_array_equal(actual.rewards, expected.rewards)
    np.testing.assert_array_equal(actual.is_terminal, expected.is_terminal)
    assert batch.debug_state(0)["state"]["history"][-1]["kind"] == case.family.value
    critic = batch.critic_state()
    np.testing.assert_array_equal(
        critic.ownership.sum(axis=1) + critic.escrow + critic.played,
        np.ones((1, 108), dtype=np.uint8),
    )


def test_double_down_tribute_return_privacy_and_public_resolution_match_oracle():
    previous = PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN")
    state = RoundState.deal(seed=17, level_rank=Rank.SIX, previous_result=previous)
    oracle = _oracle_batch(state)
    expected = oracle.observe()
    batch = cpp.EnvBatch(1, auto_reset=False)
    actual = batch.reset_from_deals(
        [_deal_ids(state)],
        level_ranks=[int(Rank.SIX)],
        leader_seats=[0],
        previous_results=[{"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN"}],
    )

    private_commits = 0
    for _ in range(32):
        np.testing.assert_array_equal(actual.available_actions.tokens, expected.available_actions.tokens)
        np.testing.assert_array_equal(actual.observations.channels, expected.observations.channels)
        phase = batch.debug_state(0)["state"]["phase"]
        if phase == 0:
            break
        row = actual.available_actions.tokens[0, np.flatnonzero(actual.available_actions.weights[0] > 0)[0]][None, :]
        expected = oracle.step(row)
        actual = batch.step(row)
        nonzero = [tuple(values[values != 0].tolist()) for values in actual.observations.tokens[0]]
        if phase in (1, 2) and batch.debug_state(0)["state"]["phase"] in (1, 2):
            private_commits += 1
            assert sum(bool(values) for values in nonzero) == 1
        np.testing.assert_array_equal(actual.observations.tokens, expected.observations.tokens)
    else:
        raise AssertionError("exchange did not resolve")

    assert private_commits >= 3
    event_rows = actual.observations.tokens[0]
    np.testing.assert_array_equal(event_rows, np.repeat(event_rows[:1], 4, axis=0))
    assert cpp.EVENT_EXCHANGE_RESOLVED in event_rows[0]
    restored = cpp.EnvBatch.deserialize(batch.serialize())
    assert restored.serialize() == batch.serialize()


def test_anti_tribute_starts_public_play_without_exchange():
    previous = {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN"}
    hands = [[], [], [], []]
    for card in full_deck():
        hands[card.card_id % 4].append(card.card_id)
    # Donors are seats 1 and 3. Move both big jokers into their union.
    for joker, donor in ((53, 1), (107, 3)):
        owner = next(seat for seat in range(4) if joker in hands[seat])
        if owner != donor:
            swap = next(card for card in hands[donor] if card not in (53, 107))
            hands[owner].remove(joker)
            hands[donor].remove(swap)
            hands[owner].append(swap)
            hands[donor].append(joker)
    batch = cpp.EnvBatch(1, auto_reset=False)
    step = batch.reset_from_deals([hands], level_ranks=[int(Rank.SIX)], previous_results=[previous])
    state = batch.debug_state(0)["state"]
    assert state["phase"] == 0
    assert state["anti_tribute"] is True
    assert state["active_seat"] == 0
    assert np.all(step.observations.channels[0, :, 488] == 1)
    assert not np.any(step.observations.channels[0, :, 216:324])


def test_pass_chain_contact_and_double_down_terminal_reward():
    # Team 0 seats each have one card. The remaining 106 cards are split across
    # seats 1 and 3 so seat 0 can finish, all opponents pass, and seat 2 takes
    # contact before finishing second.
    hands = [[0], [], [4], []]
    for card_id in range(108):
        if card_id in (0, 4):
            continue
        hands[1 if card_id % 2 else 3].append(card_id)
    batch = cpp.EnvBatch(1, auto_reset=False)
    step = batch.reset_from_deals([hands], level_ranks=[int(Rank.SIX)], leader_seats=[0])

    for token in (cpp.FAMILY_TOKEN_BASE, cpp.CARD_TOKEN_BASE + 0, cpp.COMMIT_TOKEN):
        step = batch.step(_row_for_token(step, token)[None, :])
    assert batch.debug_state(0)["state"]["finished_ranks"] == [0]
    for _ in range(3):
        step = batch.step(_row_for_token(step, cpp.PASS_TOKEN)[None, :])
    assert batch.debug_state(0)["state"]["active_seat"] == 2
    for token in (cpp.FAMILY_TOKEN_BASE, cpp.CARD_TOKEN_BASE + 4, cpp.COMMIT_TOKEN):
        step = batch.step(_row_for_token(step, token)[None, :])

    assert bool(step.is_terminal[0])
    np.testing.assert_array_equal(step.rewards[0], np.asarray([1, -1, 1, -1], np.float32))
    info = step.infos[0]
    assert info["outcome_class"] == "DOUBLE_DOWN"
    assert info["ranking"][0] == 1 and info["ranking"][2] == 2  # ranking is indexed by seat
    state = batch.debug_state(0)["state"]
    assert state["done"] is True and sorted(state["finished_ranks"]) == [0, 1, 2, 3]


def test_stale_or_forged_rows_fail_without_mutating_serialized_state():
    batch = cpp.EnvBatch(1, auto_reset=False)
    step = batch.reset(11)
    before = batch.serialize()
    row = step.available_actions.tokens[0, np.flatnonzero(step.available_actions.weights[0] > 0)[0]].copy()
    row[2] += 1
    with pytest.raises(cpp.ProtocolError):
        batch.step(row[None, :])
    assert batch.serialize() == before

    live = step.available_actions.tokens[0, np.flatnonzero(step.available_actions.weights[0] > 0)[0]][None, :]
    batch.step(live)
    with pytest.raises(cpp.ProtocolError):
        batch.step(live)


def test_capacity_overflow_fails_instead_of_truncating():
    batch = cpp.EnvBatch(1, auto_reset=False, action_capacity=1)
    with pytest.raises(cpp.AdapterCapacityError):
        batch.reset(3)
