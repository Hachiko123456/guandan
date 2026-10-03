"""A04 batch contracts, including failures after trial slots have advanced."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace

import numpy as np
import pytest

from guandan.action_state import ProtocolError, StepwiseActionState
from guandan.cards import Rank, full_deck
from guandan.env_batch import BatchObservation, BatchStepResult, GuandanEnvBatch
from guandan.environment import GuandanEnv
from guandan.environment_types import (
    BatchObservation as SharedBatchObservation,
    BatchStepResult as SharedBatchStepResult,
    GameConfig,
    Observation,
    ObservationSpec,
)
from guandan.state import EpisodeTerminatedError, IllegalActionError, RoundState


CONFIGS = (
    GameConfig(level_rank=Rank.TWO, seed=5),
    GameConfig(level_rank=Rank.SIX, seed=17, leader_seat=3),
)
ARRAY_FIELDS = (
    "observation_tokens", "state_channels", "legal_next_tokens",
    "legal_next_mask", "legal_next_is_commit", "legal_next_kinds", "legal_next_values",
)


def assert_observations_equal(actual: Observation, expected: Observation) -> None:
    for field in fields(Observation):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def assert_row(actual: BatchObservation, index: int, expected: Observation) -> None:
    assert actual.present[index]
    assert actual.private_hand_card_ids is None
    for field in fields(Observation):
        if field.name == "private_hand_card_ids":
            assert getattr(expected, field.name) is None
            continue
        left, right = getattr(actual, field.name)[index], getattr(expected, field.name)
        if isinstance(right, np.ndarray):
            assert left.dtype == right.dtype
            np.testing.assert_array_equal(left, right)
            assert not np.shares_memory(left, right)
        else:
            assert left == right


def next_tokens(batch: GuandanEnvBatch) -> np.ndarray:
    observation = batch.observe()
    return np.array([
        observation.legal_next_tokens[i, observation.legal_next_mask[i]][0]
        if observation.present[i] else 0
        for i in range(len(batch.envs))
    ], dtype=np.int32)


def assert_unchanged(batch: GuandanEnvBatch, slots: tuple, payloads: list[bytes]) -> None:
    assert batch.envs is slots
    assert batch.serialize() == payloads
    assert [env.serialize() for env in slots] == payloads


def assert_inactive(observation: BatchObservation, index: int, phase: str) -> None:
    assert not observation.present[index]
    assert observation.phase[index] == phase
    assert observation.player_id[index] == observation.team_id[index] == -1
    for name in ARRAY_FIELDS:
        assert not np.any(getattr(observation, name)[index])


@pytest.fixture
def batch() -> GuandanEnvBatch:
    result = GuandanEnvBatch(CONFIGS)
    result.reset()
    return result


@pytest.fixture
def terminal_fixture() -> bytes:
    # A conserved fixture with partner seat 2 already finished.  Seat 0 can
    # finish the winning team with one natural single; no losing hand needs
    # enumeration.  Fixtures may have non-27-card hands under the contract.
    deck = list(full_deck())
    card = next(card for card in deck if card.rank == Rank.THREE)
    remaining = [other for other in deck if other != card]
    state = RoundState(
        [[card], remaining[::2], [], remaining[1::2]],
        level_rank=Rank.TWO, active_seat=0, leader_seat=0,
        finished_ranks=[2], emptied_seats=[2],
    )
    return StepwiseActionState(state).serialize()


def test_shared_types_and_environment_reexport() -> None:
    from guandan.environment import GuandanEnvBatch as PublicBatch

    assert BatchObservation is SharedBatchObservation
    assert BatchStepResult is SharedBatchStepResult
    assert PublicBatch is GuandanEnvBatch


def test_constructor_never_resets_and_every_slot_needs_reset() -> None:
    batch = GuandanEnvBatch(CONFIGS)
    assert len(batch.envs) == 2 and batch.envs[0] is not batch.envs[1]
    assert all(env.observe() is None for env in batch.envs)
    observation = batch.observe()
    assert not np.any(observation.present)
    with pytest.raises(RuntimeError, match="reset"):
        batch.step(np.array([8, 8], dtype=np.int32))
    batch.reset_at(0, seed=5)
    saved, slots = batch.envs[0].serialize(), batch.envs
    with pytest.raises(RuntimeError, match="reset"):
        batch.step(np.array([8, 8], dtype=np.int32))
    assert batch.envs is slots and batch.envs[0].serialize() == saved
    assert batch.envs[1].observe() is None
    batch.reset_at(1, seed=17)
    assert np.all(batch.step(next_tokens(batch)).observation.present)


def test_batch_shapes_dtypes_and_masks(batch: GuandanEnvBatch) -> None:
    observation = batch.observe()
    spec = batch.observation_spec
    expected = {
        "player_id": ((2,), np.int32), "team_id": ((2,), np.int32),
        "token_step": ((2,), np.int64), "committed_step": ((2,), np.int64),
        "present": ((2,), np.bool_),
        "observation_tokens": ((2, spec.max_observation_tokens), np.int32),
        "state_channels": ((2, spec.num_state_channels), np.float32),
        **{name: ((2, spec.max_legal_next_tokens), np.bool_ if name in
                   {"legal_next_mask", "legal_next_is_commit"} else np.int32)
           for name in ARRAY_FIELDS[2:]},
    }
    for name, (shape, dtype) in expected.items():
        array = getattr(observation, name)
        assert array.shape == shape and array.dtype == dtype
    assert observation.phase.shape == (2,) and observation.phase.dtype.kind == "U"
    for name in ("legal_next_tokens", "legal_next_is_commit", "legal_next_kinds", "legal_next_values"):
        assert not np.any(getattr(observation, name)[~observation.legal_next_mask])
    result = batch.step(next_tokens(batch))
    assert result.rewards.shape == (2, 4) and result.rewards.dtype == np.float32
    for name in ("done", "truncated", "is_commit"):
        assert getattr(result, name).shape == (2,) and getattr(result, name).dtype == np.bool_
    for name in ("token_step", "committed_step"):
        assert getattr(result, name).shape == (2,) and getattr(result, name).dtype == np.int64
    assert isinstance(result.committed_action, tuple) and isinstance(result.info, tuple)


def test_seeded_standalone_and_batch_identical_prefixes_and_commits() -> None:
    batch = GuandanEnvBatch(CONFIGS)
    standalone = [GuandanEnv(config) for config in CONFIGS]
    observation = batch.reset(seeds=[41, 53])
    for index, env in enumerate(standalone):
        assert_row(observation, index, env.reset(seed=[41, 53][index]))
    saw_prefix = saw_commit = saw_pass = False
    for _ in range(12):
        tokens = next_tokens(batch)
        before = [env.committed_step for env in standalone]
        expected = [env.step(int(token)) for env, token in zip(standalone, tokens)]
        actual = batch.step(tokens)
        for index, result in enumerate(expected):
            np.testing.assert_array_equal(actual.rewards[index], result.rewards)
            for name in ("done", "truncated", "is_commit", "token_step", "committed_step"):
                assert getattr(actual, name)[index] == getattr(result, name)
            assert actual.info[index] == result.info
            assert actual.committed_action[index] == result.committed_action
            assert_row(actual.observation, index, result.observation)
            assert batch.envs[index].serialize() == standalone[index].serialize()
            if result.is_commit:
                saw_commit = True
                assert result.committed_step == before[index] + 1
                saw_pass |= int(tokens[index]) == 6
            else:
                saw_prefix = True
                assert result.committed_step == before[index]
                assert result.committed_action is None and not np.any(result.rewards)
    assert saw_prefix and saw_commit and saw_pass


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_signed_integer_vector_widths(batch: GuandanEnvBatch, dtype) -> None:
    result = batch.step(next_tokens(batch).astype(dtype))
    np.testing.assert_array_equal(result.token_step, [1, 1])


@pytest.mark.parametrize("bad", [
    [8, 8], (8, 8), 8, np.array(8, dtype=np.int32),
    np.array([[8, 8]], dtype=np.int32), np.array([[8], [8]], dtype=np.int32),
    np.array([8], dtype=np.int32), np.array([8, 8, 8], dtype=np.int32),
    np.array([8.0, 8.0]), np.array([True, False]),
    np.array([8, 8], dtype=np.uint32), np.array([8, 8], dtype=object),
    np.array(["8", "8"]), np.array([8j, 8j]),
])
def test_invalid_shape_or_dtype_never_coerced_and_atomic(batch: GuandanEnvBatch, bad) -> None:
    slots, saved = batch.envs, batch.serialize()
    with pytest.raises((TypeError, ValueError)):
        batch.step(bad)
    assert_unchanged(batch, slots, saved)


@pytest.mark.parametrize("bad", [0, -1, 7, 255, 256, 2**32 + 8])
def test_illegal_later_slot_token_rolls_back_entire_batch(batch: GuandanEnvBatch, bad: int) -> None:
    slots, saved = batch.envs, batch.serialize()
    tokens = next_tokens(batch).astype(np.int64)
    tokens[1] = bad
    with pytest.raises(IllegalActionError):
        batch.step(tokens)
    assert_unchanged(batch, slots, saved)


def test_reset_at_only_changes_selected_slot_and_returns_single_observation(batch: GuandanEnvBatch) -> None:
    batch.step(next_tokens(batch))
    other, saved = batch.envs[1], batch.envs[1].serialize()
    actual = batch.reset_at(np.int64(0), seed=71)
    standalone = GuandanEnv(CONFIGS[0])
    assert isinstance(actual, Observation)
    assert_observations_equal(actual, standalone.reset(seed=71))
    assert batch.envs[1] is other and other.serialize() == saved
    assert batch.envs[0].token_step == batch.envs[0].committed_step == 0


@pytest.mark.parametrize("index", [True, np.bool_(False), 0.0, "0", -1, 2])
def test_reset_at_rejects_bad_index_without_changes(batch: GuandanEnvBatch, index) -> None:
    slots, saved = batch.envs, batch.serialize()
    with pytest.raises((TypeError, ValueError, IndexError)):
        batch.reset_at(index, seed=123)
    assert_unchanged(batch, slots, saved)


def test_reset_seed_and_initial_state_lengths_and_bad_values_are_atomic(batch: GuandanEnvBatch) -> None:
    slots, saved = batch.envs, batch.serialize()
    for kwargs in (
        {"seeds": [1]}, {"seeds": [1, 2, 3]}, {"seeds": [1, True]},
        {"seeds": [1, 2.5]}, {"seeds": [1, -2]},
        {"initial_states": [None]}, {"initial_states": [None, "not bytes"]},
        {"seeds": [1, None], "initial_states": [saved[0], None]},
    ):
        with pytest.raises((TypeError, ValueError)):
            batch.reset(**kwargs)
        assert_unchanged(batch, slots, saved)
    with pytest.raises((ValueError, ProtocolError)):
        batch.reset(initial_states=[saved[0], b"not serialized"])
    assert_unchanged(batch, slots, saved)
    with pytest.raises((ValueError, ProtocolError)):
        batch.reset_at(1, initial_state=b"not serialized")
    assert_unchanged(batch, slots, saved)


def test_reset_accepts_fresh_initial_states_and_mixed_slot_seed(batch: GuandanEnvBatch) -> None:
    fresh = batch.serialize()
    batch.step(next_tokens(batch))
    observation = batch.reset(initial_states=fresh)
    for index in range(2):
        assert_row(observation, index, GuandanEnv.deserialize(fresh[index]).observe())
    mixed = batch.reset(seeds=[103, None], initial_states=[None, fresh[1]])
    assert_row(mixed, 0, GuandanEnv(CONFIGS[0]).reset(seed=103))
    assert_row(mixed, 1, GuandanEnv.deserialize(fresh[1]).observe())


@pytest.mark.parametrize("operation", ["reset", "reset_at", "step", "load"])
def test_final_stack_failure_keeps_every_original_slot(batch: GuandanEnvBatch, monkeypatch, operation: str) -> None:
    slots, saved = batch.envs, batch.serialize()
    tokens = next_tokens(batch)
    calls = []

    def fail_stack(trials, observations):
        calls.append(tuple(trials))
        assert all(trial is not original for trial, original in zip(trials, slots))
        if operation == "step":
            assert [env.token_step for env in trials] == [1, 1]
        raise ProtocolError("late stacking failure")

    monkeypatch.setattr(batch, "_stack", fail_stack)
    with pytest.raises(ProtocolError, match="late stacking"):
        if operation == "reset":
            batch.reset(seeds=[103, 107])
        elif operation == "reset_at":
            batch.reset_at(0, seed=103)
        elif operation == "step":
            batch.step(tokens)
        else:
            batch.load_serialized(saved)
    assert calls
    assert_unchanged(batch, slots, saved)


@pytest.mark.parametrize("failure", ["raise", "wrong_shape"])
def test_later_slot_encoder_failure_after_steps_is_atomic(batch: GuandanEnvBatch, monkeypatch, failure: str) -> None:
    import guandan.environment as environment

    original_encoder = environment.encode_observation
    slots, saved = batch.envs, batch.serialize()
    tokens, encoded_after_step = next_tokens(batch), []

    def broken_encoder(protocol, spec):
        observation = original_encoder(protocol, spec)
        if protocol.token_step > 0:
            encoded_after_step.append(observation.player_id)
            if observation.player_id == CONFIGS[1].leader_seat:
                if failure == "raise":
                    raise ProtocolError("later slot encoder failure")
                return replace(observation, observation_tokens=np.zeros(1, dtype=np.int32))
        return observation

    monkeypatch.setattr(environment, "encode_observation", broken_encoder)
    with pytest.raises(ProtocolError):
        batch.step(tokens)
    assert CONFIGS[0].leader_seat in encoded_after_step
    assert CONFIGS[1].leader_seat in encoded_after_step
    assert_unchanged(batch, slots, saved)


def test_later_slot_reset_encoder_failure_keeps_rng_and_originals(batch: GuandanEnvBatch, monkeypatch) -> None:
    import guandan.environment as environment

    slots, saved = batch.envs, batch.serialize()
    original_encoder = environment.encode_observation

    def broken_encoder(protocol, spec):
        observation = original_encoder(protocol, spec)
        if observation.player_id == CONFIGS[1].leader_seat:
            raise ProtocolError("later reset encoder failure")
        return observation

    with monkeypatch.context() as patch:
        patch.setattr(environment, "encode_observation", broken_encoder)
        with pytest.raises(ProtocolError, match="later reset"):
            batch.reset()
    assert_unchanged(batch, slots, saved)
    expected = [GuandanEnv.deserialize(payload) for payload in saved]
    actual = batch.reset()
    for index, env in enumerate(expected):
        assert_row(actual, index, env.reset())
        assert batch.envs[index].serialize() == env.serialize()


def test_no_array_alias_between_observations_results_and_slots(batch: GuandanEnvBatch) -> None:
    first, second = batch.observe(), batch.observe()
    saved = batch.serialize()
    for name in ARRAY_FIELDS:
        a, b = getattr(first, name), getattr(second, name)
        assert not np.shares_memory(a, b)
        assert not np.shares_memory(a[0], a[1])
        a[...] = 0
    first.player_id[:] = -1
    first.token_step[:] = 999
    assert batch.serialize() == saved
    for index, env in enumerate(batch.envs):
        assert_row(second, index, env.observe())
    original = batch.envs[0]
    clone = original.clone()
    clone.step(int(clone.legal_tokens()[0]))
    assert original.serialize() == saved[0]
    result = batch.step(next_tokens(batch))
    after = batch.serialize()
    result.rewards[:] = 123
    result.token_step[:] = 456
    result.observation.state_channels[:] = 99
    result.info[0]["user_added"] = [1]
    assert batch.serialize() == after and batch.envs[0].token_step == 1


def test_reset_seed_streams_and_clones_are_independent() -> None:
    batch = GuandanEnvBatch(CONFIGS)
    singles = [GuandanEnv(config) for config in CONFIGS]
    seen = []
    for _ in range(3):
        actual = batch.reset()
        for index, env in enumerate(singles):
            assert_row(actual, index, env.reset())
            assert batch.envs[index].serialize() == env.serialize()
        seen.append(actual.observation_tokens.copy())
    assert not np.array_equal(seen[0], seen[1])
    assert not np.array_equal(seen[1], seen[2])
    other, saved = batch.envs[1], batch.envs[1].serialize()
    clone = other.clone()
    clone.reset()
    batch.reset_at(0)
    assert_observations_equal(batch.envs[0].observe(), singles[0].reset())
    assert batch.envs[1] is other and other.serialize() == saved
    batch.reset_at(1)
    assert_observations_equal(batch.envs[1].observe(), singles[1].reset())
    assert_observations_equal(batch.envs[1].observe(), clone.observe())
    first = batch.reset(seeds=[211, 223])
    batch.reset()
    again = batch.reset(seeds=[211, 223])
    np.testing.assert_array_equal(first.observation_tokens, again.observation_tokens)


def test_serialization_preserves_prefix_and_next_committed_result(batch: GuandanEnvBatch) -> None:
    batch.step(next_tokens(batch))
    saved = batch.serialize()
    restored = GuandanEnvBatch([GameConfig(), GameConfig()], auto_reset=True)
    restored.load_serialized(saved)
    assert restored.auto_reset is True and restored.serialize() == saved
    assert all(left is not right for left, right in zip(batch.envs, restored.envs))
    for _ in range(4):
        tokens = next_tokens(batch)
        actual, expected = restored.step(tokens), batch.step(tokens)
        assert restored.serialize() == batch.serialize()
        np.testing.assert_array_equal(actual.rewards, expected.rewards)
        np.testing.assert_array_equal(actual.is_commit, expected.is_commit)
        assert actual.committed_action == expected.committed_action
    original = batch.serialize()
    restored.reset_at(0)
    assert batch.serialize() == original


def test_load_payload_length_types_corruption_and_spec_are_atomic(batch: GuandanEnvBatch) -> None:
    slots, saved = batch.envs, batch.serialize()
    for payloads in (saved[:1], saved + saved[:1], saved[0], [saved[0], None], [saved[0], b"{"]):
        with pytest.raises((TypeError, ValueError, ProtocolError)):
            batch.load_serialized(payloads)
        assert_unchanged(batch, slots, saved)
    other = GuandanEnv(CONFIGS[1], observation_spec=ObservationSpec(max_observation_tokens=4095))
    other.reset()
    with pytest.raises(ProtocolError, match="ObservationSpec"):
        batch.load_serialized([saved[0], other.serialize()])
    assert_unchanged(batch, slots, saved)


def test_loading_duplicate_payloads_creates_independent_rngs_and_slots(batch: GuandanEnvBatch) -> None:
    payload = batch.serialize()[0]
    batch.load_serialized([payload, payload])
    assert batch.envs[0] is not batch.envs[1]
    assert batch.serialize() == [payload, payload]
    other = batch.envs[1]
    batch.reset_at(0)
    assert batch.envs[1] is other and other.serialize() == payload
    first_reset = batch.envs[0].serialize()
    batch.reset_at(1)
    assert batch.serialize() == [first_reset, first_reset]


def test_truncated_slot_pad_is_frozen_zero_noop() -> None:
    batch = GuandanEnvBatch([GameConfig(seed=5, max_token_steps=1), CONFIGS[1]])
    batch.reset()
    first = batch.step(next_tokens(batch))
    assert first.truncated.tolist() == [True, False]
    assert first.info[0]["termination_reason"] == "max_token_steps"
    ended, saved = batch.envs[0], batch.envs[0].serialize()
    for _ in range(3):
        result = batch.step(next_tokens(batch))
        assert result.truncated.tolist() == [True, False] and not result.done[0]
        assert not result.is_commit[0] and result.committed_action[0] is None
        assert not np.any(result.rewards[0])
        assert result.token_step[0] == result.observation.token_step[0] == 1
        assert result.committed_step[0] == result.observation.committed_step[0] == 0
        assert result.info[0]["inactive"] is True
        assert result.info[0]["termination_reason"] == "max_token_steps"
        assert_inactive(result.observation, 0, "TRUNCATED")
        assert batch.envs[0] is ended and ended.serialize() == saved
    slots, before = batch.envs, batch.serialize()
    tokens = next_tokens(batch)
    tokens[0] = 8
    with pytest.raises(EpisodeTerminatedError):
        batch.step(tokens)
    assert_unchanged(batch, slots, before)


def test_terminal_slot_noop_and_terminal_rewards(terminal_fixture: bytes) -> None:
    batch = GuandanEnvBatch([CONFIGS[0], CONFIGS[1]])
    batch.reset(initial_states=[terminal_fixture, None])
    for _ in range(8):
        result = batch.step(next_tokens(batch))
        if result.done[0]:
            break
    assert result.done[0] and result.is_commit[0]
    assert result.committed_action[0] is not None
    np.testing.assert_array_equal(result.rewards[0], [1, -1, 1, -1])
    saved, ended = batch.envs[0].serialize(), batch.envs[0]
    counters = (result.token_step[0], result.committed_step[0])
    for _ in range(3):
        result = batch.step(next_tokens(batch))
        assert result.done[0] and not result.truncated[0] and not result.is_commit[0]
        assert result.committed_action[0] is None and not np.any(result.rewards[0])
        assert (result.token_step[0], result.committed_step[0]) == counters
        assert (result.observation.token_step[0], result.observation.committed_step[0]) == counters
        assert_inactive(result.observation, 0, "TERMINAL")
        assert batch.envs[0] is ended and ended.serialize() == saved
    assert batch.reset_at(0, seed=5).token_step == 0
    assert not batch.envs[0].done


@pytest.mark.parametrize("ended_kind", ["truncated", "terminal"])
def test_auto_reset_preserves_final_results_but_returns_new_observation(terminal_fixture: bytes, ended_kind: str) -> None:
    config = CONFIGS[0] if ended_kind == "terminal" else GameConfig(seed=5, max_token_steps=1)
    batch = GuandanEnvBatch([config], auto_reset=True)
    standalone = GuandanEnv(config)
    state = terminal_fixture if ended_kind == "terminal" else None
    batch.reset(initial_states=[state])
    standalone.reset(initial_state=state)
    for _ in range(8):
        tokens = next_tokens(batch)
        expected = standalone.step(int(tokens[0]))
        actual = batch.step(tokens)
        if expected.done or expected.truncated:
            break
    assert expected.done or expected.truncated
    assert actual.done[0] == expected.done and actual.truncated[0] == expected.truncated
    assert actual.is_commit[0] == expected.is_commit
    assert actual.committed_action[0] == expected.committed_action
    assert actual.token_step[0] == expected.token_step > 0
    assert actual.committed_step[0] == expected.committed_step
    np.testing.assert_array_equal(actual.rewards[0], expected.rewards)
    assert actual.info[0]["auto_reset"] is True
    assert actual.info[0]["final_info"] == expected.info
    assert_row(actual.observation, 0, standalone.reset())
    assert actual.observation.token_step[0] == actual.observation.committed_step[0] == 0
    assert not batch.envs[0].done and not batch.envs[0].truncated
    assert batch.envs[0].serialize() == standalone.serialize()


def test_auto_reset_does_not_reset_previously_ended_loaded_slots(terminal_fixture: bytes) -> None:
    ended = GuandanEnvBatch([CONFIGS[0], GameConfig(seed=17, max_token_steps=1)])
    ended.reset(initial_states=[terminal_fixture, None])
    for _ in range(8):
        ended.step(next_tokens(ended))
        if ended.envs[0].done:
            break
    assert ended.envs[0].done and ended.envs[1].truncated
    saved = ended.serialize()
    restored = GuandanEnvBatch([CONFIGS[0], CONFIGS[1]], auto_reset=True)
    restored.load_serialized(saved)
    assert restored.serialize() == saved and restored.auto_reset is True
    slots = restored.envs
    for _ in range(2):
        result = restored.step(np.zeros(2, dtype=np.int32))
        assert restored.serialize() == saved
        assert all(left is right for left, right in zip(slots, restored.envs))
        assert result.done.tolist() == [True, False]
        assert result.truncated.tolist() == [False, True]
        for info in result.info:
            assert info["inactive"] is True
            assert "auto_reset" not in info and "final_info" not in info


def test_auto_reset_deepcopies_final_info_without_full_state(monkeypatch) -> None:
    batch = GuandanEnvBatch([GameConfig(seed=5, max_token_steps=1)], auto_reset=True)
    batch.reset()
    step = GuandanEnv.step
    nested = {"events": [1]}

    def tagged_step(self, token):
        result = step(self, token)
        return replace(result, info={**result.info, "nested": nested})

    def forbidden_full_state(self):
        raise AssertionError("batch must never attach a full-state dump")

    monkeypatch.setattr(GuandanEnv, "step", tagged_step)
    monkeypatch.setattr(GuandanEnv, "full_state", forbidden_full_state)
    result = batch.step(next_tokens(batch))
    info, final_info = result.info[0], result.info[0]["final_info"]
    assert info["nested"] == final_info["nested"] == nested
    info["nested"]["events"].append(2)
    final_info["nested"]["events"].append(3)
    assert nested == {"events": [1]}
    assert info["nested"] == {"events": [1, 2]}
    assert final_info["nested"] == {"events": [1, 3]}
    assert "full_state" not in info and "full_state" not in final_info


def test_later_auto_reset_failure_rolls_back_all_slots_and_rng(monkeypatch) -> None:
    configs = [GameConfig(seed=5, max_token_steps=1), GameConfig(seed=17, max_token_steps=1)]
    batch = GuandanEnvBatch(configs, auto_reset=True)
    batch.reset()
    slots, saved = batch.envs, batch.serialize()
    original_reset = GuandanEnv.reset
    calls = []

    def broken_reset(self, **kwargs):
        calls.append(self.config.seed)
        if self.config.seed == 17:
            raise ProtocolError("later auto-reset failed")
        return original_reset(self, **kwargs)

    tokens = next_tokens(batch)
    with monkeypatch.context() as patch:
        patch.setattr(GuandanEnv, "reset", broken_reset)
        with pytest.raises(ProtocolError, match="later auto-reset"):
            batch.step(tokens)
    assert calls == [5, 17]
    assert_unchanged(batch, slots, saved)
    assert np.all(batch.step(tokens).truncated)


@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("max_token_steps", [6, 4096], ids=["truncate-on-first-return", "live-return"])
def test_first_return_redaction_matches_single_before_and_after_optional_auto_reset(
    auto_reset: bool, max_token_steps: int
) -> None:
    from guandan.action_state import COMMIT_TOKEN
    from guandan.state import PreviousHandResult

    config = GameConfig(
        level_rank=Rank.SIX,
        seed=0,
        previous_result=PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN"),
        max_token_steps=max_token_steps,
    )
    standalone = GuandanEnv(config)
    batch = GuandanEnvBatch([config], auto_reset=auto_reset)
    assert_row(batch.reset(), 0, standalone.reset())

    for phase in ("TRIBUTE", "TRIBUTE", "RETURN"):
        assert standalone.observe().phase == phase
        for token in (int(standalone.legal_tokens()[0]), COMMIT_TOKEN):
            expected = standalone.step(token)
            actual = batch.step(np.array([token], dtype=np.int32))
            assert actual.done[0] == expected.done
            assert actual.truncated[0] == expected.truncated
            assert actual.is_commit[0] == expected.is_commit
            assert actual.token_step[0] == expected.token_step
            assert actual.committed_step[0] == expected.committed_step
            assert actual.committed_action[0] == expected.committed_action
            np.testing.assert_array_equal(actual.rewards[0], expected.rewards)
            if expected.truncated and auto_reset:
                assert actual.info[0]["auto_reset"] is True
                assert actual.info[0]["final_info"] == expected.info
                assert_row(actual.observation, 0, standalone.reset())
            else:
                assert actual.info[0] == expected.info
                assert "auto_reset" not in actual.info[0]
                if expected.observation is None:
                    assert_inactive(actual.observation, 0, "TRUNCATED")
                else:
                    assert_row(actual.observation, 0, expected.observation)
            assert batch.serialize() == [standalone.serialize()]
            assert "resolved_transfers" not in actual.info[0]
            assert "full_state" not in actual.info[0]

        receipt = actual.committed_action[0]
        assert receipt.phase == receipt.kind == phase
        assert receipt.card_ids == receipt.declared_ranks == receipt.wild_assignments == ()
        assert receipt.family is None and receipt.declared_suit is None

    assert actual.token_step[0] == 6 and actual.committed_step[0] == 3
    assert receipt.public_index == 2
    if max_token_steps == 6:
        assert actual.truncated[0] and not actual.done[0]
        assert actual.info[0]["termination_reason"] == "max_token_steps"
        if auto_reset:
            assert actual.info[0]["final_info"]["termination_reason"] == "max_token_steps"
        else:
            saved = batch.serialize()
            noop = batch.step(np.array([0], dtype=np.int32))
            assert noop.truncated[0] and not noop.is_commit[0]
            assert noop.info[0]["termination_reason"] == "max_token_steps"
            assert "auto_reset" not in noop.info[0]
            assert batch.serialize() == saved
    else:
        # Redaction ends only when the second return resolves the exchange.
        assert actual.observation.phase[0] == "RETURN"
        for token in (int(standalone.legal_tokens()[0]), COMMIT_TOKEN):
            expected = standalone.step(token)
            actual = batch.step(np.array([token], dtype=np.int32))
            assert actual.committed_action[0] == expected.committed_action
            assert actual.info[0] == expected.info
            assert_row(actual.observation, 0, expected.observation)
        assert actual.observation.phase[0] == "LEAD_PLAY"
        assert actual.info[0]["resolved_transfers"]["return"]
        assert batch.serialize() == [standalone.serialize()]


@pytest.mark.parametrize("corruption", ["version", "rng", "prefix", "observation_overflow"])
def test_corrupted_later_snapshot_preserves_originals_and_future_rng(
    batch: GuandanEnvBatch, corruption: str
) -> None:
    import json

    batch.step(next_tokens(batch))
    slots, saved = batch.envs, batch.serialize()
    controls = [GuandanEnv.deserialize(payload) for payload in saved]
    earlier = controls[0].clone()
    earlier.step(int(earlier.legal_tokens()[0]))
    earlier_payload = earlier.serialize()
    assert earlier_payload != saved[0]
    data = json.loads(saved[1])
    if corruption == "version":
        data["serialization_version"] = -1
    elif corruption == "rng":
        data["rng_state"]["bit_generator"] = "not-a-supported-bit-generator"
    elif corruption == "prefix":
        data["protocol"]["prefix_tokens"] = [0]
    else:
        data["observation_spec"]["max_observation_tokens"] = 1
    corrupted = json.dumps(data).encode("utf-8")
    with pytest.raises((ValueError, ProtocolError)):
        batch.load_serialized([earlier_payload, corrupted])
    assert_unchanged(batch, slots, saved)
    # Equal serialized bytes include RNG; a subsequent unseeded reset also
    # checks the behavioral stream, rather than just snapshot equality.
    actual = batch.reset()
    for index, control in enumerate(controls):
        assert_row(actual, index, control.reset())
        assert batch.envs[index].serialize() == control.serialize()


def test_terminal_auto_reset_matches_single_with_already_truncated_neighbor(
    terminal_fixture: bytes, monkeypatch
) -> None:
    standalone = GuandanEnv(CONFIGS[0])
    standalone.reset(initial_state=terminal_fixture)
    inactive_config = GameConfig(seed=17, max_token_steps=1)
    inactive = GuandanEnv(inactive_config)
    inactive.reset()
    inactive.step(int(inactive.legal_tokens()[0]))
    inactive_payload = inactive.serialize()
    batch = GuandanEnvBatch([CONFIGS[0], inactive_config], auto_reset=True)
    batch.load_serialized([standalone.serialize(), inactive_payload])
    inactive_slot = batch.envs[1]
    original_reset = GuandanEnv.reset
    reset_calls = []

    def reset_live_only(self, **kwargs):
        assert self.config != inactive_config, "an inactive PAD slot must never reset"
        reset_calls.append(self.config)
        return original_reset(self, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(GuandanEnv, "reset", reset_live_only)
        for _ in range(8):
            token = int(standalone.legal_tokens()[0])
            expected = standalone.step(token)
            actual = batch.step(np.array([token, 0], dtype=np.int32))
            assert actual.done[0] == expected.done
            assert actual.truncated[0] == expected.truncated
            assert actual.is_commit[0] == expected.is_commit
            assert actual.committed_action[0] == expected.committed_action
            assert actual.token_step[0] == expected.token_step
            assert actual.committed_step[0] == expected.committed_step
            np.testing.assert_array_equal(actual.rewards[0], expected.rewards)
            assert actual.truncated[1] and not actual.done[1] and not actual.is_commit[1]
            assert actual.committed_action[1] is None and not np.any(actual.rewards[1])
            assert actual.token_step[1] == actual.observation.token_step[1] == 1
            assert actual.committed_step[1] == actual.observation.committed_step[1] == 0
            assert actual.info[1] == {"inactive": True, "termination_reason": "max_token_steps"}
            assert_inactive(actual.observation, 1, "TRUNCATED")
            assert batch.envs[1] is inactive_slot
            assert batch.serialize()[1] == inactive_payload
            if expected.done:
                assert actual.info[0]["auto_reset"] is True
                assert actual.info[0]["final_info"] == expected.info
                break
            assert actual.info[0] == expected.info
            assert_row(actual.observation, 0, expected.observation)
    assert expected.done and reset_calls == [CONFIGS[0]]
    np.testing.assert_array_equal(actual.rewards[0], [1, -1, 1, -1])
    assert_row(actual.observation, 0, standalone.reset())
    assert batch.serialize() == [standalone.serialize(), inactive_payload]
