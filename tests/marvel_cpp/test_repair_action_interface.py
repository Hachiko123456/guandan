from __future__ import annotations

import numpy as np
import pytest

from nanopokerlib import guandan as cpp


def _first_rows(step) -> np.ndarray:
    rows = []
    for slot in range(step.rewards.shape[0]):
        index = int(np.flatnonzero(step.available_actions.weights[slot] > 0)[0])
        rows.append(step.available_actions.tokens[slot, index])
    return np.asarray(rows, dtype=np.int64)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_raw_rows_and_decoded_actions_produce_identical_transition(batch_size: int) -> None:
    original = cpp.EnvBatch(batch_size, auto_reset=False)
    initial = original.reset(71)
    payload = original.serialize()
    raw_env = cpp.EnvBatch.deserialize(payload)
    decoded_env = cpp.EnvBatch.deserialize(payload)
    rows = _first_rows(initial)

    raw = raw_env.step(rows)
    decoded = decoded_env.step(cpp.actions_from_np(rows))

    np.testing.assert_array_equal(decoded.rewards, raw.rewards)
    np.testing.assert_array_equal(decoded.is_terminal, raw.is_terminal)
    np.testing.assert_array_equal(decoded.player_ids, raw.player_ids)
    np.testing.assert_array_equal(decoded.observations.tokens, raw.observations.tokens)
    np.testing.assert_array_equal(decoded.observations.channels, raw.observations.channels)
    np.testing.assert_array_equal(decoded.available_actions.tokens, raw.available_actions.tokens)
    np.testing.assert_array_equal(decoded.available_actions.weights, raw.available_actions.weights)
    np.testing.assert_array_equal(decoded.available_actions.is_commit, raw.available_actions.is_commit)
    assert decoded_env.serialize() == raw_env.serialize()


@pytest.mark.parametrize(
    "rows",
    [
        np.zeros((1, 7), dtype=np.int64),
        np.zeros((1, 8, 1), dtype=np.int64),
        np.zeros((1, 8), dtype=np.float32),
    ],
)
def test_raw_row_shape_and_dtype_rejection_preserves_state(rows: np.ndarray) -> None:
    env = cpp.EnvBatch(1, auto_reset=False)
    env.reset(9)
    before = env.serialize()
    with pytest.raises((ValueError, cpp.ProtocolError)):
        env.step(rows)
    assert env.serialize() == before


def test_decoded_action_count_mismatch_and_non_action_sequence_are_rejected() -> None:
    env = cpp.EnvBatch(2, auto_reset=False)
    step = env.reset(12)
    rows = _first_rows(step)
    before = env.serialize()
    decoded = cpp.actions_from_np(rows)
    with pytest.raises(ValueError, match="one action row per slot"):
        env.step(decoded[:1])
    assert env.serialize() == before
    with pytest.raises(ValueError, match="V2Action"):
        env.step((object(), object()))
    assert env.serialize() == before


def test_decoded_stale_prefix_is_rejected_without_mutation() -> None:
    env = cpp.EnvBatch(1, auto_reset=False)
    step = env.reset(17)
    rows = _first_rows(step)
    stale = cpp.actions_from_np(rows)
    env.step(stale)
    before = env.serialize()
    with pytest.raises(cpp.ProtocolError, match="stale|forged|uniquely"):
        env.step(stale)
    assert env.serialize() == before


def test_forged_metadata_and_one_invalid_multi_slot_are_atomic() -> None:
    env = cpp.EnvBatch(3, auto_reset=False)
    step = env.reset(21)
    rows = _first_rows(step)
    before = env.serialize()
    forged = rows.copy()
    forged[1, 2] += 1  # prefix-stage metadata for only one slot
    with pytest.raises(cpp.ProtocolError):
        env.step(forged)
    assert env.serialize() == before

    forged = rows.copy()
    forged[2, 6] = cpp.ACTION_COMMIT if forged[2, 6] == cpp.ACTION_NON_COMMIT else cpp.ACTION_NON_COMMIT
    with pytest.raises(cpp.ProtocolError):
        cpp.actions_from_np(forged)
    assert env.serialize() == before
