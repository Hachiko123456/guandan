from __future__ import annotations

import numpy as np
import pytest

from guandan_marvel import cpp_game, game as python_game


def _reset_first_rows(game, batch_size: int = 1, seed: int = 333):
    env = game.EnvBatch(batch_size, auto_reset=False)
    step = env.reset(seed)
    rows = []
    for slot in range(batch_size):
        index = int(np.flatnonzero(step.available_actions.weights[slot] > 0)[0])
        rows.append(step.available_actions.tokens[slot, index])
    return env, np.asarray(rows)


def _assert_cpp_decode_and_raw_reject(rows: np.ndarray, *, batch_size: int = 1) -> None:
    decoded_env, _ = _reset_first_rows(cpp_game, batch_size=batch_size)
    decoded_before = decoded_env.serialize()
    with pytest.raises(cpp_game.ProtocolError):
        cpp_game.actions_from_np(rows)
    assert decoded_env.serialize() == decoded_before

    raw_env, _ = _reset_first_rows(cpp_game, batch_size=batch_size)
    raw_before = raw_env.serialize()
    with pytest.raises(cpp_game.ProtocolError):
        raw_env.step(rows)
    assert raw_env.serialize() == raw_before


@pytest.mark.parametrize("field", range(cpp_game.max_action_tokens))
@pytest.mark.parametrize(
    ("dtype", "delta"),
    [(np.int64, 2**32), (np.int64, -(2**32)), (np.uint64, 2**32)],
    ids=["signed-plus-2p32", "signed-minus-2p32", "unsigned-plus-2p32"],
)
def test_all_eight_fields_reject_32bit_wraparound_before_narrowing(field: int, dtype, delta: int) -> None:
    _env, legal = _reset_first_rows(cpp_game)
    bad = legal.astype(dtype, copy=True)
    expected = int(legal[0, field]) + delta
    bad[0, field] = expected
    assert int(bad[0, field]) == expected  # test construction must preserve the value
    _assert_cpp_decode_and_raw_reject(bad)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (np.int32, -1),
        (np.int32, cpp_game.num_tokens),
        (np.int32, np.iinfo(np.int32).min),
        (np.int32, np.iinfo(np.int32).max),
        (np.int64, np.iinfo(np.int64).min),
        (np.int64, np.iinfo(np.int64).max),
        (np.int64, 2**32 + cpp_game.FAMILY_TOKEN_BASE),
        (np.int64, cpp_game.FAMILY_TOKEN_BASE - 2**32),
        (np.uint32, cpp_game.num_tokens),
        (np.uint32, np.iinfo(np.uint32).max),
        (np.uint64, np.iinfo(np.int64).max + 1),
        (np.uint64, np.iinfo(np.uint64).max),
        (np.uint64, 2**32 + cpp_game.FAMILY_TOKEN_BASE),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_signed_and_unsigned_integer_boundaries_are_rejected_without_preoverflow(dtype, value) -> None:
    _env, legal = _reset_first_rows(cpp_game)
    bad = legal.astype(dtype, copy=True)
    bad[0, 5] = value
    assert int(bad[0, 5]) == int(value)
    _assert_cpp_decode_and_raw_reject(bad)


def test_multi_slot_out_of_range_uint64_is_atomic_for_every_slot_and_rng() -> None:
    env, rows = _reset_first_rows(cpp_game, batch_size=2, seed=444)
    bad = rows.astype(np.uint64, copy=True)
    bad[1, 5] = np.uint64(2**32 + int(rows[1, 5]))
    assert int(bad[1, 5]) == 2**32 + int(rows[1, 5])
    before = env.serialize()
    with pytest.raises(cpp_game.ProtocolError):
        cpp_game.actions_from_np(bad)
    assert env.serialize() == before
    with pytest.raises(cpp_game.ProtocolError):
        env.step(bad)
    assert env.serialize() == before


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_legal_integer_dtypes_keep_identical_decode_and_transition(dtype) -> None:
    source, rows = _reset_first_rows(cpp_game, batch_size=2, seed=555)
    payload = source.serialize()
    typed = rows.astype(dtype, copy=True)
    np.testing.assert_array_equal(typed.astype(np.uint64), rows.astype(np.uint64))

    raw_env = cpp_game.EnvBatch.deserialize(payload)
    decoded_env = cpp_game.EnvBatch.deserialize(payload)
    raw = raw_env.step(typed)
    decoded = decoded_env.step(cpp_game.actions_from_np(typed))
    assert raw_env.serialize() == decoded_env.serialize()
    np.testing.assert_array_equal(raw.observations.tokens, decoded.observations.tokens)
    np.testing.assert_array_equal(raw.observations.channels, decoded.observations.channels)
    np.testing.assert_array_equal(raw.available_actions.tokens, decoded.available_actions.tokens)


def test_non_contiguous_and_non_native_endian_legal_rows_are_read_by_value() -> None:
    source, rows = _reset_first_rows(cpp_game, seed=666)
    payload = source.serialize()
    backing = np.zeros((1, 16), dtype=np.int64)
    backing[:, ::2] = rows
    non_contiguous = backing[:, ::2]
    assert not non_contiguous.flags.c_contiguous
    big_endian = rows.astype(np.dtype(">i8"), copy=True)
    assert not big_endian.dtype.isnative

    reference = cpp_game.EnvBatch.deserialize(payload)
    expected = reference.step(rows)
    for candidate in (non_contiguous, big_endian):
        raw_env = cpp_game.EnvBatch.deserialize(payload)
        decoded_env = cpp_game.EnvBatch.deserialize(payload)
        raw = raw_env.step(candidate)
        decoded = decoded_env.step(cpp_game.actions_from_np(candidate))
        assert raw_env.serialize() == decoded_env.serialize() == reference.serialize()
        np.testing.assert_array_equal(raw.observations.channels, expected.observations.channels)
        np.testing.assert_array_equal(decoded.available_actions.tokens, expected.available_actions.tokens)


@pytest.mark.parametrize("delta", [2**32, -(2**32)])
def test_python_and_cpp_both_reject_same_wraparound_payload_without_state_change(delta: int) -> None:
    for game in (python_game, cpp_game):
        env, rows = _reset_first_rows(game, seed=777)
        bad = rows.astype(np.int64, copy=True)
        bad[0, 5] = int(rows[0, 5]) + delta
        before = env.serialize()
        with pytest.raises(Exception):
            game.actions_from_np(bad)
        assert env.serialize() == before
        with pytest.raises(Exception):
            env.step(bad)
        assert env.serialize() == before
