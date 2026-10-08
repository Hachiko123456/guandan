from __future__ import annotations

from dataclasses import fields, is_dataclass
import json

import numpy as np
import pytest
import torch
from torch import nn

from guandan.action_state import COMMIT_TOKEN, StepwiseActionState, TokenCodec
from guandan.cards import Rank
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan.round import return_cards_for, tribute_card_for
from guandan.state import Phase, PreviousHandResult, RoundState
from guandan_marvel import game
from guandan_marvel.buffer import BufferConfig, TrajectoryBuffer
from guandan_marvel.model.types import AgentOutput
from guandan_marvel.runtime.sampler import Sampler, SamplerConfig

PREVIOUS = PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN")


def _same(left, right) -> None:
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        assert type(left) is type(right)
        for field in fields(left):
            _same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def _env_from_state(state: RoundState) -> GuandanEnv:
    config = GameConfig(
        level_rank=state.level_rank, seed=0, leader_seat=state.active_seat,
        previous_result=state.previous_result, max_token_steps=game.max_game_horizon,
    )
    env = GuandanEnv(config)
    protocol = StepwiseActionState(state.clone(), max_token_steps=game.max_game_horizon)
    if state.committed_step == 0:
        env.reset(initial_state=protocol.serialize())
    else:
        # Test-only construction of an accepted V1 intermediate oracle state.
        env.reset(seed=0)
        protocol.token_step = max(protocol.token_step, state.committed_step * 2)
        env._protocol = protocol
        env._reset_count = 1
        env.observe()
    return env


def _valid_rows(step, slot: int = 0) -> np.ndarray:
    return step.available_actions.tokens[slot, step.available_actions.weights[slot] > 0]


def _choose_row(step, slot: int = 0) -> np.ndarray:
    mask = step.available_actions.weights[slot] > 0
    rows = step.available_actions.tokens[slot, mask]
    commits = step.available_actions.is_commit[slot, mask]
    return rows[int(np.flatnonzero(commits)[0])] if commits.any() else rows[0]




def _row_for_token(step, token_id: int, slot: int = 0) -> np.ndarray:
    rows = _valid_rows(step, slot)
    decoded = game.actions_from_np(rows)
    matches = [index for index, action in enumerate(decoded) if action.token_id == token_id]
    assert len(matches) == 1
    return rows[matches[0]]

def _play_to_terminal(batch: game.EnvBatch, step):
    count = 0
    while not bool(step.is_terminal[0]):
        step = batch.step(game.actions_from_np(_choose_row(step)))
        count += 1
        assert count <= batch.max_horizon
    return step, count


def test_protocol_constants_schemas_and_game_spec_are_versioned() -> None:
    assert game.ACTION_VERSION == "GD-ACTION-0.2-MARVEL"
    assert game.ENV_VERSION == "GD-ENV-0.2-MARVEL"
    assert game.ENCODING_VERSION == "GD-ENCODING-0.2-MARVEL"
    assert game.BACKEND == "python_reference"
    assert game.ACTION_SCHEMA["version"] == game.ACTION_VERSION
    assert game.ENV_SCHEMA["version"] == game.ENV_VERSION
    assert game.ENCODING_SCHEMA["version"] == game.ENCODING_VERSION
    assert game.GAME_SPEC.identifier == game.GAME_IDENTIFIER
    assert game.GAME_SPEC.protocol_version == game.protocol_version
    assert game.max_observation_tokens < 4096


def test_reset_shapes_four_actor_information_sets_and_structured_rows() -> None:
    batch = game.EnvBatch(2, auto_reset=False)
    step = batch.reset(11)
    assert step.rewards.shape == (2, 4) and step.rewards.dtype == np.float32
    assert step.is_terminal.shape == (2,) and not step.is_terminal.any()
    assert step.player_ids.shape == (2,)
    assert step.observations.tokens.shape == (2, 4, game.max_observation_tokens)
    assert step.observations.channels.shape == (2, 4, game.num_channels)
    assert step.observations.channels.dtype == np.uint8
    assert step.available_actions.tokens.shape == (2, game.max_available_actions, game.max_action_tokens)
    assert step.available_actions.weights.shape == (2, game.max_available_actions)
    for slot in range(2):
        rows = _valid_rows(step, slot)
        assert len(rows) == len({tuple(row) for row in rows})
        decoded = game.actions_from_np(rows)
        assert [action.token_id for action in decoded] == batch.envs[slot].legal_tokens().tolist()
        assert all(action.row[0] == game.ACTION_ROW_BEGIN and action.row[-1] == game.ACTION_ROW_END for action in decoded)


def test_privileged_critic_state_is_explicit_and_absent_from_actor_wire() -> None:
    batch = game.EnvBatch(1, auto_reset=False)
    step = batch.reset(19)
    assert not hasattr(step, "ownership") and not hasattr(step.observations, "ownership")
    critic = batch.critic_state()
    assert critic.ownership.shape == (1, 4, 108)
    assert critic.escrow.shape == critic.played.shape == (1, 108)
    np.testing.assert_array_equal(
        critic.ownership.sum(axis=1) + critic.escrow + critic.played,
        np.ones((1, 108), dtype=np.uint8),
    )


def test_each_v1_legal_next_action_has_exactly_one_v2_executable_candidate() -> None:
    batch = game.EnvBatch(1, auto_reset=False)
    step = batch.reset(7)
    for _ in range(90):
        v1 = batch.envs[0]
        legal = tuple(int(token) for token in v1.legal_tokens())
        decoded = game.actions_from_np(_valid_rows(step))
        assert sorted(action.token_id for action in decoded) == sorted(legal)
        assert len(decoded) == len(legal) == len({action.row for action in decoded})
        step = batch.step((decoded[0],))
        if step.is_terminal[0]:
            break


def test_v2_selection_matches_v1_transition_commit_and_serialization() -> None:
    batch = game.EnvBatch(1, auto_reset=False)
    step = batch.reset(23)
    reached_terminal = False
    for _ in range(game.max_game_horizon):
        reference = GuandanEnv.deserialize(batch.envs[0].serialize())
        row = _choose_row(step)
        action = game.actions_from_np(row)[0]
        expected = reference.step(action.token_id)
        actual = batch.step((action,))
        assert batch.envs[0].serialize() == reference.serialize()
        np.testing.assert_array_equal(actual.rewards[0], expected.rewards)
        assert bool(actual.is_terminal[0]) is expected.done
        if expected.done:
            reached_terminal = True
            break
        step = actual
    assert reached_terminal
    restored = game.EnvBatch.deserialize(batch.serialize())
    assert restored.serialize() == batch.serialize()


def test_serialized_restore_has_identical_future_trajectory() -> None:
    left = game.EnvBatch(1, auto_reset=False)
    step_left = left.reset(31)
    for _ in range(37):
        step_left = left.step(game.actions_from_np(_choose_row(step_left)))
    right = game.EnvBatch.deserialize(left.serialize())
    step_right = right.observe()
    _same(step_left.observations, step_right.observations)
    _same(step_left.available_actions, step_right.available_actions)
    reached_terminal = False
    for _ in range(game.max_game_horizon):
        action_rows = _choose_row(step_left)
        out_left = left.step(game.actions_from_np(action_rows))
        out_right = right.step(game.actions_from_np(action_rows.copy()))
        _same(out_left, out_right)
        assert left.serialize() == right.serialize()
        step_left = out_left
        if out_left.is_terminal[0]:
            reached_terminal = True
            break
    assert reached_terminal


def test_four_player_privacy_under_other_hand_swaps() -> None:
    base = RoundState.deal(seed=301, level_rank=Rank.SIX, leader_seat=0)
    for observer in range(4):
        hidden = [seat for seat in range(4) if seat != observer]
        left, right = hidden[0], hidden[1]
        alternate = base.clone()
        alternate.hands[left][0], alternate.hands[right][0] = alternate.hands[right][0], alternate.hands[left][0]
        alternate.check_invariants()
        a = game.EnvBatch.from_envs([_env_from_state(base)], auto_reset=False)
        b = game.EnvBatch.from_envs([_env_from_state(alternate)], auto_reset=False)
        obs_a = a._build_output(a.envs, a._events).observations
        obs_b = b._build_output(b.envs, b._events).observations
        np.testing.assert_array_equal(obs_a.tokens[0, observer], obs_b.tokens[0, observer])
        np.testing.assert_array_equal(obs_a.channels[0, observer], obs_b.channels[0, observer])


def test_unfinished_tribute_and_return_keep_other_selections_private() -> None:
    state = RoundState.deal(seed=2, level_rank=Rank.SIX, leader_seat=0, previous_result=PREVIOUS)
    env = _env_from_state(state)
    batch = game.EnvBatch.from_envs([env], auto_reset=False)
    step = batch.observe()
    first = batch.envs[0]._require_protocol().round_state.active_seat
    assert batch.envs[0]._require_protocol().round_state.phase is Phase.TRIBUTE
    step = batch.step(game.actions_from_np(_choose_row(step)))
    assert step.observations.tokens[0, first, 1] == game.EVENT_PRIVATE_PREFIX
    step = batch.step(game.actions_from_np(_choose_row(step)))
    event_rows = step.observations.tokens[0]
    assert event_rows[first, 1] == game.EVENT_PRIVATE_EXCHANGE_COMMIT
    for observer in range(4):
        if observer != first:
            assert not np.any(event_rows[observer])
            assert not np.any(event_rows[observer] == TokenCodec.card(
                batch.envs[0]._require_protocol().round_state.tribute_selections[first].card_id
            ))

    # Complete tribute selections and inspect the first unresolved return.
    while batch.envs[0]._require_protocol().round_state.phase is Phase.TRIBUTE:
        step = batch.step(game.actions_from_np(_choose_row(step)))
    assert batch.envs[0]._require_protocol().round_state.phase is Phase.RETURN
    return_actor = batch.envs[0]._require_protocol().round_state.active_seat
    step = batch.step(game.actions_from_np(_choose_row(step)))
    assert step.observations.tokens[0, return_actor, 1] == game.EVENT_PRIVATE_PREFIX
    step = batch.step(game.actions_from_np(_choose_row(step)))
    assert step.observations.tokens[0, return_actor, 1] == game.EVENT_PRIVATE_EXCHANGE_COMMIT
    for observer in range(4):
        if observer != return_actor:
            assert not np.any(step.observations.tokens[0, observer])




def test_hidden_exchange_worlds_are_identical_for_every_uninvolved_observer() -> None:
    # Tribute: change which physical highest-rank card the donor owns. Only the
    # donor may see the selected identity while tribute remains incomplete.
    state = RoundState.deal(seed=2, level_rank=Rank.SIX, leader_seat=0, previous_result=PREVIOUS)
    donor = state.active_seat
    selected = tribute_card_for(state, donor)
    assert selected is not None
    owner = next(seat for seat in range(4) if seat != donor and any(
        card.rank is selected.rank and not card.is_level_wild(Rank.SIX) for card in state.hands[seat]
    ))
    alternate_card = next(card for card in state.hands[owner]
                          if card.rank is selected.rank and not card.is_level_wild(Rank.SIX))
    alternate = state.clone()
    donor_index = alternate.hands[donor].index(selected)
    owner_index = alternate.hands[owner].index(alternate_card)
    alternate.hands[donor][donor_index], alternate.hands[owner][owner_index] = alternate_card, selected
    alternate.check_invariants()

    a = game.EnvBatch.from_envs([_env_from_state(state)], auto_reset=False)
    b = game.EnvBatch.from_envs([_env_from_state(alternate)], auto_reset=False)
    sa, sb = a.observe(), b.observe()
    for batch, step, card in ((a, sa, selected), (b, sb, alternate_card)):
        step = batch.step(game.actions_from_np(_row_for_token(step, TokenCodec.card(card.card_id))))
        step = batch.step(game.actions_from_np(_row_for_token(step, COMMIT_TOKEN)))
        if batch is a:
            out_a = step
        else:
            out_b = step
    for observer in range(4):
        if observer in (donor, owner):
            continue
        np.testing.assert_array_equal(out_a.observations.tokens[0, observer], out_b.observations.tokens[0, observer])
        np.testing.assert_array_equal(out_a.observations.channels[0, observer], out_b.observations.channels[0, observer])

    # Return: branch on two physical cards with the same rank. The source may
    # see its choice; neither the target donor nor other seats may see identity
    # before the final return resolves the exchange.
    return_state = RoundState.deal(seed=0, level_rank=Rank.SIX, leader_seat=0, previous_result=PREVIOUS)
    while return_state.phase is Phase.TRIBUTE:
        from guandan.round import apply_action, enumerate_legal_actions
        apply_action(return_state, enumerate_legal_actions(return_state)[0])
    actor = return_state.active_seat
    choices = return_cards_for(return_state, actor)
    pair = next((left, right) for i, left in enumerate(choices) for right in choices[i + 1:] if left.rank is right.rank)
    branches = []
    for choice in pair:
        batch = game.EnvBatch.from_envs([_env_from_state(return_state)], auto_reset=False)
        step = batch.observe()
        step = batch.step(game.actions_from_np(_row_for_token(step, TokenCodec.card(choice.card_id))))
        step = batch.step(game.actions_from_np(_row_for_token(step, COMMIT_TOKEN)))
        assert batch.envs[0]._require_protocol().round_state.phase is Phase.RETURN
        branches.append(step)
    for observer in range(4):
        if observer == actor:
            continue
        np.testing.assert_array_equal(branches[0].observations.tokens[0, observer], branches[1].observations.tokens[0, observer])
        np.testing.assert_array_equal(branches[0].observations.channels[0, observer], branches[1].observations.channels[0, observer])


def test_incremental_events_do_not_repeat_full_v1_history() -> None:
    batch = game.EnvBatch(1, auto_reset=False)
    step = batch.reset(5)
    nonzero_lengths = []
    for _ in range(100):
        nonzero_lengths.append(int(np.count_nonzero(step.observations.tokens[0, 0])))
        step = batch.step(game.actions_from_np(_choose_row(step)))
        if step.is_terminal[0]:
            break
    assert max(nonzero_lengths) <= game.max_observation_tokens < 4096
    assert len(batch.envs[0]._require_protocol().round_state.history) > 10


def test_capacity_and_horizon_overflow_fail_closed_without_state_publish() -> None:
    too_small = game.EnvBatch(1, auto_reset=False, action_capacity=1)
    with pytest.raises(game.AdapterCapacityError, match="legal action count"):
        too_small.reset(0)
    assert not too_small._is_reset[0]

    obs_small = game.EnvBatch(1, auto_reset=False, observation_capacity=2)
    with pytest.raises(game.AdapterCapacityError, match="event requires"):
        obs_small.reset(0)
    assert not obs_small._is_reset[0]

    horizon = game.EnvBatch(1, auto_reset=False, max_horizon=1)
    step = horizon.reset(0)
    before = horizon._envs[0].serialize()
    with pytest.raises(game.TrajectoryHorizonError):
        horizon.step(game.actions_from_np(_choose_row(step)))
    assert horizon._envs[0].serialize() == before


def test_complete_games_have_real_terminal_rewards_ranking_and_card_conservation() -> None:
    cases = [
        (0, None), (1, None), (2, None),
        (3, PREVIOUS),
    ]
    for seed, previous in cases:
        configs = [GameConfig(level_rank=Rank.SIX, previous_result=previous, max_token_steps=game.max_game_horizon)]
        batch = game.EnvBatch(1, auto_reset=False, configs=configs)
        terminal, steps = _play_to_terminal(batch, batch.reset(seed))
        assert 0 < steps < game.max_game_horizon
        assert terminal.is_terminal.tolist() == [True]
        assert set(terminal.rewards[0].tolist()) == {-1.0, 1.0}
        state = batch.envs[0]._require_protocol().round_state
        assert state.done and state.phase is Phase.TERMINAL
        assert set(state.ranking) == {1, 2, 3, 4}
        state.check_invariants()
        remaining = [card.card_id for hand in state.hands for card in hand]
        played = [card.card_id for action in state.history if action.kind not in ("pass", "tribute", "return") for card in action.cards]
        assert len(remaining) + len(played) == 108
        assert len(set(remaining + played)) == 108
        assert terminal.infos[0]["ranking"] == tuple(state.ranking)


class _DecoderConfig:
    num_key_value_heads = 1
    num_attention_heads = 1
    hidden_size = 4


class _Decoder:
    config = _DecoderConfig()


class _CommitFirstAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.decoder = _Decoder()
        self.decoder_n_layers = 1

    def forward(self, *, rollout, **_kwargs) -> AgentOutput:
        weights = rollout["avail_weights"]
        rows = rollout["avail_tokens"]
        legal = weights > 0
        commits = (rows[..., 6] == game.ACTION_COMMIT) & legal
        chosen = torch.where(commits.any(-1), commits.float().argmax(-1), legal.float().argmax(-1))
        probs = torch.zeros_like(weights).scatter_(-1, chosen.unsqueeze(-1), 1.0)
        return AgentOutput(probs)


class _SmallHorizonGame:
    identifier = game.identifier
    GAME_IDENTIFIER = game.GAME_IDENTIFIER
    num_players = game.num_players
    num_tokens = game.num_tokens
    pad_token = game.pad_token
    num_channels = game.num_channels
    max_game_horizon = 512
    max_observation_tokens = game.max_observation_tokens
    max_action_tokens = game.max_action_tokens
    max_available_actions = game.max_available_actions
    protocol_version = game.protocol_version
    actions_from_np = staticmethod(game.actions_from_np)

    @staticmethod
    def EnvBatch(batch_size: int):
        return game.EnvBatch(batch_size, max_horizon=512)


def test_marvel_sampler_consumes_minimal_complete_python_adapter_trajectory() -> None:
    sampler = Sampler(config=SamplerConfig(
        game=_SmallHorizonGame, device="cpu", rollout_batch_size=1,
        collect_batch_size=1, sampling_temperature=0.0, env_seed=0,
    ))
    trajectory = sampler.collect(agent_model=_CommitFirstAgent())
    steps = int(trajectory["num_steps"][0])
    assert 0 < steps < _SmallHorizonGame.max_game_horizon
    assert int(trajectory["is_terminal"][0, :steps].sum()) == 1
    assert trajectory["is_terminal"][0, steps - 1]
    assert set(trajectory["rewards"][0, steps - 1].tolist()) == {-1.0, 1.0}
    assert trajectory["obs_tokens"].shape[2:] == (4, game.max_observation_tokens)
    assert trajectory["avail_tokens"].shape[2:] == (game.max_available_actions, game.max_action_tokens)

    replay = TrajectoryBuffer(1, BufferConfig(game=_SmallHorizonGame, device="cpu"))
    replay.add(trajectory)
    sampled = replay.sample(1)
    assert int(sampled["num_steps"][0]) == steps
