from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from guandan_marvel import game as python_game
from guandan_marvel.env_batch import EnvStepBatch
from nanopokerlib import guandan as cpp


def _random_rows(step, rng):
    rows = []
    for batch in range(step.rewards.shape[0]):
        legal = np.flatnonzero(step.available_actions.weights[batch] > 0)
        rows.append(step.available_actions.tokens[batch, int(rng.choice(legal))])
    return np.asarray(rows, dtype=np.int64)


def test_import_is_explicit_and_python_reference_remains_default():
    assert python_game.BACKEND == "python_reference"
    assert cpp.BACKEND == "cpp"
    assert cpp.GAME_SPEC.identifier == cpp.GAME_IDENTIFIER
    assert cpp.GAME_SPEC.protocol_version == python_game.protocol_version


def test_batch_shapes_independence_and_marvel_tensor_conversion():
    batch = cpp.EnvBatch(3, auto_reset=True)
    step = batch.reset(101)
    assert step.rewards.shape == (3, 4)
    assert step.is_terminal.shape == (3,)
    assert step.player_ids.shape == (3,)
    assert step.observations.tokens.shape == (3, 4, 48)
    assert step.observations.channels.shape == (3, 4, 512)
    assert step.available_actions.tokens.shape == (3, 256, 8)
    assert step.available_actions.weights.shape == (3, 256)
    assert step.available_actions.is_commit.shape == (3, 256)

    cfg = SimpleNamespace(device="cpu")
    parsed = EnvStepBatch.from_env_step(step, cfg)
    assert tuple(parsed["obs_tokens"].shape) == (3, 4, 48)
    assert tuple(parsed["avail_tokens"].shape) == (3, 256, 8)

    rows = []
    for slot in range(3):
        legal = np.flatnonzero(step.available_actions.weights[slot] > 0)
        rows.append(step.available_actions.tokens[slot, legal[min(slot, len(legal) - 1)]])
    next_step = batch.step(np.asarray(rows, dtype=np.int64))
    states = [batch.debug_state(index) for index in range(3)]
    assert len({tuple(state["prefix_tokens"]) for state in states}) >= 2
    assert next_step.player_ids.shape == (3,)


def test_four_seat_actor_channels_only_contain_observer_hand_identity():
    batch = cpp.EnvBatch(1, auto_reset=False)
    step = batch.reset(33)
    state = batch.debug_state(0)["state"]
    for observer in range(4):
        encoded = set(np.flatnonzero(step.observations.channels[0, observer, :108]))
        assert encoded == set(state["hands"][observer])
        for other in range(4):
            if other != observer:
                assert not set(state["hands"][other]).issubset(encoded)
    # Full ownership is available only through the explicit critic API.
    critic = batch.critic_state()
    assert critic.ownership.shape == (1, 4, 108)
    assert not hasattr(step, "ownership")


def test_real_batch_games_reach_terminal_independently_and_auto_reset():
    batch_size = 4
    batch = cpp.EnvBatch(batch_size, auto_reset=True)
    step = batch.reset(901)
    rng = np.random.default_rng(20261009)
    terminal_counts = np.zeros(batch_size, dtype=np.int64)
    terminal_records = []
    for _ in range(4096):
        step = batch.step(_random_rows(step, rng))
        for slot in np.flatnonzero(step.is_terminal):
            terminal_counts[slot] += 1
            info = step.infos[int(slot)]
            terminal_records.append((int(slot), tuple(info["ranking"]), info["outcome_class"], tuple(step.rewards[slot].tolist())))
        if np.all(terminal_counts > 0):
            break
    else:
        raise AssertionError("batch did not produce a real terminal for every slot")

    assert len(terminal_records) >= batch_size
    for _slot, ranking, outcome, rewards in terminal_records:
        assert sorted(ranking) == [1, 2, 3, 4]
        assert outcome in {"DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST"}
        assert set(rewards) == {-1.0, 1.0}
    # Auto-reset slots expose a new playable frame while preserving terminal
    # rewards/is_terminal in the returned outcome.
    assert np.all(step.player_ids >= 0)
    assert np.all(step.available_actions.weights.sum(axis=1) > 0)


def test_batch_serialization_preserves_prefix_rows_and_future_transition():
    batch = cpp.EnvBatch(2, auto_reset=False)
    step = batch.reset(44)
    rows = _random_rows(step, np.random.default_rng(5))
    step = batch.step(rows)
    payload = batch.serialize()
    restored = cpp.EnvBatch.deserialize(payload)
    assert restored.serialize() == payload
    observed = restored.observe()
    np.testing.assert_array_equal(observed.observations.tokens, step.observations.tokens)
    np.testing.assert_array_equal(observed.observations.channels, step.observations.channels)
    np.testing.assert_array_equal(observed.available_actions.tokens, step.available_actions.tokens)

    next_rows = _random_rows(step, np.random.default_rng(6))
    left = batch.step(next_rows)
    right = restored.step(next_rows)
    np.testing.assert_array_equal(left.rewards, right.rewards)
    np.testing.assert_array_equal(left.observations.tokens, right.observations.tokens)
    np.testing.assert_array_equal(left.observations.channels, right.observations.channels)
    np.testing.assert_array_equal(left.available_actions.tokens, right.available_actions.tokens)

def test_actions_from_np_validates_complete_structured_rows():
    batch = cpp.EnvBatch(1, auto_reset=False)
    step = batch.reset(19)
    row = step.available_actions.tokens[0, np.flatnonzero(step.available_actions.weights[0] > 0)[0]]
    decoded = cpp.actions_from_np(row)
    assert len(decoded) == 1
    assert tuple(decoded[0].row) == tuple(row.tolist())
    assert decoded[0].prefix_stage == 0
    forged = row.copy()
    forged[6] = cpp.ACTION_COMMIT if forged[6] == cpp.ACTION_NON_COMMIT else cpp.ACTION_NON_COMMIT
    try:
        cpp.actions_from_np(forged)
    except cpp.ProtocolError:
        pass
    else:
        raise AssertionError("forged COMMIT metadata was accepted")
