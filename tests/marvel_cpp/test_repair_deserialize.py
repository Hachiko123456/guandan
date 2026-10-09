from __future__ import annotations

import json

import numpy as np
import pytest

from nanopokerlib import guandan as cpp


def _payload(env) -> dict:
    return json.loads(env.serialize())


def _bytes(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _commit_first_row(step) -> np.ndarray:
    legal = step.available_actions.weights[0] > 0
    commits = legal & step.available_actions.is_commit[0]
    index = int(np.flatnonzero(commits if commits.any() else legal)[0])
    return step.available_actions.tokens[[0], [index]]


@pytest.mark.parametrize("replacement", ["", "NOT_A_VALID_RNG_STATE"])
def test_empty_and_invalid_rng_text_are_rejected(replacement: str) -> None:
    env = cpp.EnvBatch(1, auto_reset=False)
    env.reset(3)
    data = _payload(env)
    data["games"][0]["rng_state"] = replacement
    with pytest.raises(cpp.ProtocolError, match="RNG state"):
        cpp.EnvBatch.deserialize(_bytes(data))


def test_truncated_and_non_whitespace_trailing_rng_text_are_rejected() -> None:
    env = cpp.EnvBatch(1, auto_reset=False)
    env.reset(4)
    data = _payload(env)
    valid = data["games"][0]["rng_state"]
    for damaged in (valid[: len(valid) // 2], valid + " TRAILING_GARBAGE"):
        mutated = json.loads(json.dumps(data))
        mutated["games"][0]["rng_state"] = damaged
        with pytest.raises(cpp.ProtocolError, match="RNG state"):
            cpp.EnvBatch.deserialize(_bytes(mutated))


def test_rng_trailing_whitespace_is_accepted_and_canonicalized() -> None:
    env = cpp.EnvBatch(1, auto_reset=False)
    env.reset(5)
    original = env.serialize()
    data = json.loads(original)
    data["games"][0]["rng_state"] += " \n\t  "
    restored = cpp.EnvBatch.deserialize(_bytes(data))
    assert restored.serialize() == original


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.__setitem__("trajectory_steps", [-1]), "legal range"),
        (lambda data: data.__setitem__("trajectory_steps", [4097]), "legal range"),
        (lambda data: data.__setitem__("trajectory_steps", [123]), "game token_step"),
        (lambda data: data["games"][0]["config"].__setitem__("max_token_steps", 512), "horizon"),
        (lambda data: data.__setitem__("trajectory_steps", []), "shape"),
        (lambda data: data.__setitem__("events", []), "shape"),
    ],
)
def test_inconsistent_batch_counters_and_lengths_are_rejected(mutator, message: str) -> None:
    env = cpp.EnvBatch(1, auto_reset=False)
    env.reset(6)
    data = _payload(env)
    mutator(data)
    with pytest.raises(cpp.ProtocolError, match=message):
        cpp.EnvBatch.deserialize(_bytes(data))


def test_game_token_commit_history_terminal_and_truncated_invariants_are_rejected() -> None:
    env = cpp.EnvBatch(1, auto_reset=False)
    env.reset(7)
    base = _payload(env)
    mutations = []

    token = json.loads(json.dumps(base))
    token["games"][0]["token_step"] = 1
    token["trajectory_steps"] = [1]
    mutations.append(token)

    committed = json.loads(json.dumps(base))
    committed["games"][0]["state"]["committed_step"] = 1
    mutations.append(committed)

    truncated = json.loads(json.dumps(base))
    truncated["games"][0]["truncated"] = True
    mutations.append(truncated)

    terminal = json.loads(json.dumps(base))
    terminal["games"][0]["state"]["done"] = True
    mutations.append(terminal)

    for data in mutations:
        with pytest.raises((cpp.ProtocolError, cpp.StateInvariantError)):
            cpp.EnvBatch.deserialize(_bytes(data))


def test_mid_prefix_roundtrip_continues_to_same_real_terminal() -> None:
    left = cpp.EnvBatch(1, auto_reset=False)
    first = left.reset(31)
    first = left.step(_commit_first_row(first))
    assert left.debug_state(0)["prefix_tokens"]
    payload = left.serialize()
    right = cpp.EnvBatch.deserialize(payload)
    assert right.serialize() == payload

    left_step = left.observe()
    right_step = right.observe()
    for _ in range(4096):
        row = _commit_first_row(left_step)
        left_step = left.step(row)
        right_step = right.step(cpp.actions_from_np(row))
        np.testing.assert_array_equal(right_step.rewards, left_step.rewards)
        np.testing.assert_array_equal(right_step.is_terminal, left_step.is_terminal)
        np.testing.assert_array_equal(right_step.observations.tokens, left_step.observations.tokens)
        np.testing.assert_array_equal(right_step.observations.channels, left_step.observations.channels)
        np.testing.assert_array_equal(right_step.available_actions.tokens, left_step.available_actions.tokens)
        if bool(left_step.is_terminal[0]):
            break
    else:
        raise AssertionError("restored game did not reach terminal")
    assert right.serialize() == left.serialize()


def test_rng_continuation_matches_through_actual_auto_reset_new_deal() -> None:
    left = cpp.EnvBatch(1, auto_reset=True)
    step = left.reset(41)
    # Save after RNG-consuming deal and after one protocol step.
    step = left.step(_commit_first_row(step))
    right = cpp.EnvBatch.deserialize(left.serialize())
    right_step = right.observe()

    for _ in range(4096):
        row = _commit_first_row(step)
        step = left.step(row)
        right_step = right.step(cpp.actions_from_np(row))
        np.testing.assert_array_equal(right_step.rewards, step.rewards)
        np.testing.assert_array_equal(right_step.is_terminal, step.is_terminal)
        np.testing.assert_array_equal(right_step.observations.tokens, step.observations.tokens)
        np.testing.assert_array_equal(right_step.observations.channels, step.observations.channels)
        if bool(step.is_terminal[0]):
            break
    else:
        raise AssertionError("auto-reset continuation did not reach terminal")

    # The terminal step has already called reset_auto() internally. This hand
    # identity therefore depends on the restored mt19937_64 continuation.
    left_state = left.debug_state(0)
    right_state = right.debug_state(0)
    assert left_state["state"]["hands"] == right_state["state"]["hands"]
    assert left_state["token_step"] == right_state["token_step"] == 0
    assert left.serialize() == right.serialize()
