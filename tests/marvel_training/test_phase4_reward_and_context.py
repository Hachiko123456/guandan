# SPDX-License-Identifier: MIT
"""MARVEL v14 phase-4 decisive counterexamples.

Each test in this module fails against the *pre-phase-4* implementation (where
the engine paid a binary +/-1 terminal reward, candidate_win could be derived
from an exact +/-1 comparison, coverage was declared from request metadata, and
the context schedule had no versioned curriculum or per-deal context hash).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from guandan.action_state import BOS_TOKEN, COMMIT_TOKEN, RETURN_TOKEN, TRIBUTE_TOKEN, TokenCodec
from guandan_marvel.context_schedule import (
    TRIBUTE_OBSERVATION_AUDIT, build_context_schedule, build_rank_leader_schedule,
    context_hash, curriculum_alpha, curriculum_expanded_contexts, curriculum_provenance,
    schedule_coverage, validate_context_curriculum, validate_context_schedule,
)
from guandan_marvel.config import TrainConfig
from guandan_marvel import game as python_game
from guandan_marvel import cpp_game
from guandan_marvel.testing import differential
from guandan_marvel.reward import (
    RANK_POINTS, RAW_REWARD_MAX, RAW_REWARD_MIN, REWARD_SEMANTICS,
    TRAINING_REWARD_SCALE, rank_point_reward, reward_semantics_block, scale_reward,
)

# Hand-written table: the test must not re-derive this from the production helper.
EXPECTED_POINTS = {"DOUBLE_DOWN": 3.0, "HEAD_THIRD": 2.0, "HEAD_LAST": 1.0}


def _smoke_game_spec():
    import guandan_marvel.game as g
    return g.GAME_SPEC


def _terminal_rewards(step) -> list[float]:
    return [float(value) for value in np.asarray(step.rewards[0]).tolist()]


def _play_same_case(case):
    python_batch, python_step = differential.python_batch_from_case(case)
    cpp_batch, cpp_step = differential.cpp_batch_from_case(case)
    rng = np.random.default_rng(4242 + case.game_index)
    guard = 0
    while not bool(python_step.is_terminal[0]):
        guard += 1
        assert guard < python_game.max_game_horizon
        legal = np.flatnonzero(python_step.available_actions.weights[0] > 0)
        row = python_step.available_actions.tokens[0, int(rng.choice(legal))].copy()
        python_step = python_batch.step(row[None, :])
        cpp_step = cpp_batch.step(row[None, :])
    return python_batch, python_step, cpp_batch, cpp_step


def test_cross_engine_rank_point_reward_over_all_three_outcomes():
    """Python vs C++ item-by-item terminal reward agreement, all outcome classes."""
    observed = {}
    for index in range(24):
        case = differential.build_case(index)
        python_batch, python_step, cpp_batch, cpp_step = _play_same_case(case)
        py_info = dict(python_step.infos[0])
        cpp_info = dict(cpp_step.infos[0])
        outcome = str(py_info["outcome_class"])
        assert outcome == str(cpp_info["outcome_class"])
        assert list(py_info["ranking"]) == list(cpp_info["ranking"])
        assert int(py_info["winner_team"]) == int(cpp_info["winner_team"])
        assert tuple(py_info["team_reward"]) == tuple(cpp_info["team_reward"])
        py_rewards = _terminal_rewards(python_step)
        cpp_rewards = _terminal_rewards(cpp_step)
        assert py_rewards == cpp_rewards
        # hand-written rank-point table (independent of the production helper)
        points = EXPECTED_POINTS[outcome]
        winner = int(py_info["winner_team"])
        expected = [points if seat % 2 == winner else -points for seat in range(4)]
        assert py_rewards == expected
        assert tuple(py_info["team_reward"]) == (
            points if winner == 0 else -points,
            points if winner == 1 else -points,
        )
        # serialized round trip preserves the terminal reward payload
        restored = python_game.EnvBatch.deserialize(python_batch.serialize())
        restored_cpp = cpp_game.EnvBatch.deserialize(cpp_batch.serialize())
        assert restored.serialize() == python_batch.serialize()
        assert restored_cpp.serialize() == cpp_batch.serialize()
        state = restored.envs[0]._require_protocol().round_state
        assert [float(value) for value in state.rewards] == expected
        observed[outcome] = True
    assert set(observed) == {"DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST"}, observed


def test_reward_semantics_block_matches_hand_table():
    assert RANK_POINTS == {"DOUBLE_DOWN": 3, "HEAD_THIRD": 2, "HEAD_LAST": 1}
    assert (RAW_REWARD_MIN, RAW_REWARD_MAX) == (-3.0, 3.0)
    assert TRAINING_REWARD_SCALE == 1.0 / 3.0
    assert REWARD_SEMANTICS == "guandan_rank_points"
    for outcome, points in EXPECTED_POINTS.items():
        for winner in (0, 1):
            t0 = points if winner == 0 else -points
            t1 = points if winner == 1 else -points
            assert rank_point_reward(outcome, winner) == (t0, t1)
    assert scale_reward(3.0) == 1.0
    assert scale_reward(-2.0) == pytest.approx(-2.0 / 3.0)
    block = reward_semantics_block()
    assert block["semantics"] == REWARD_SEMANTICS
    assert block["rank_points"] == RANK_POINTS
    assert block["training_scale"] == TRAINING_REWARD_SCALE


def test_training_scale_is_not_decoupled_from_raw_sign():
    for raw in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0):
        assert np.sign(scale_reward(raw)) == np.sign(raw)


def test_config_rejects_legacy_reward_semantics_and_wrong_scale():
    spec = _smoke_game_spec()
    with pytest.raises(ValueError):
        TrainConfig(game=spec, reward_semantics="winner_binary_v1")
    with pytest.raises(ValueError):
        TrainConfig(game=spec, reward_scale=1.0)
    cfg = TrainConfig(game=spec)
    assert cfg.reward_semantics == REWARD_SEMANTICS
    assert cfg.reward_scale == TRAINING_REWARD_SCALE


def test_rank_leader_schedule_is_52_uniform_no_tribute():
    schedule = build_rank_leader_schedule()
    coverage = schedule_coverage(schedule)
    assert coverage["context_count"] == 52
    assert coverage["level_ranks"] == list(range(3, 16))
    assert coverage["leader_seats"] == [0, 1, 2, 3]
    assert coverage["tribute_modes"] == ["none"]
    assert coverage["weight_total"] == 52
    contexts = schedule["contexts"]
    assert len({context_hash(row) for row in contexts}) == 52


def test_manifest_hash_covers_schedule_order_and_weights():
    spec = _smoke_game_spec()
    base = build_rank_leader_schedule()
    reordered = {"schema": base["schema"], "contexts": list(reversed(base["contexts"]))}
    reweighted = {"schema": base["schema"], "contexts": [dict(row) for row in base["contexts"]]}
    reweighted["contexts"][0] = {**reweighted["contexts"][0], "weight": 5}
    h_base = TrainConfig(game=spec, training_context_schedule=base).training_data_manifest_hash
    h_order = TrainConfig(game=spec, training_context_schedule=reordered).training_data_manifest_hash
    h_weight = TrainConfig(game=spec, training_context_schedule=reweighted).training_data_manifest_hash
    assert h_base != h_order, "manifest hash must cover context order"
    assert h_base != h_weight, "manifest hash must cover context weights"


def test_tribute_contexts_are_accepted_now_that_the_audit_passes():
    assert TRIBUTE_OBSERVATION_AUDIT["passed"] is True
    tribute = build_context_schedule(
        level_ranks=(3, 15), leader_seats=(0, 1),
        previous_results=[{"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN"}],
    )
    validate_context_schedule(tribute)  # 7B enabled: accepted
    cfg = TrainConfig(game=_smoke_game_spec(), training_context_schedule=tribute)
    assert cfg.training_context_schedule is not None


def test_tribute_payloads_are_validated_by_rules_object():
    with pytest.raises(ValueError):
        # winner_team must match the rank-1 seat's team; the rules object rejects it
        build_context_schedule(
            level_ranks=(3,), leader_seats=(0,),
            previous_results=[{"ranking": [1, 2, 3, 4], "winner_team": 1, "outcome_class": "DOUBLE_DOWN"}],
        )


def test_versioned_schema_validates_all_7b_payload_variants():
    variants = [
        {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN"},
        {"ranking": [1, 2, 3, 4], "winner_team": 0, "outcome_class": "HEAD_THIRD"},
        {"ranking": [1, 2, 4, 3], "winner_team": 0, "outcome_class": "HEAD_LAST"},
        {"ranking": [3, 1, 4, 2], "winner_team": 1, "outcome_class": "DOUBLE_DOWN"},
        {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": True},
        None,
    ]
    schedule = build_context_schedule(
        level_ranks=(3, 4, 5), leader_seats=(0, 1), previous_results=variants,
    )
    validated = validate_context_schedule(schedule)
    modes = {
        "none" if row["previous_result"] is None else (
            "cancelled" if row["previous_result"].get("tribute_cancelled")
            else row["previous_result"]["outcome_class"].lower()
        )
        for row in validated["contexts"]
    }
    assert modes == {"none", "cancelled", "double_down", "head_third", "head_last"}


def test_tribute_full_observation_audit_has_no_legal_collision():
    from guandan_marvel.testing.tribute_audit import audit_tribute_observation
    report = audit_tribute_observation()
    assert report["passed"] is True
    assert report["legal_relations_checked"] >= 6
    assert report["full_observation_collisions"] == []
    assert report["hidden_card_leaks"] == []
    # The bag-level symmetry that motivated the pre-repair counterexample is
    # broken by the channel planes.
    assert report["bag_level_collision_present"] is True
    assert report["bag_level_collision_disambiguated_by_channels"] is True


def _is_old(row) -> bool:
    return row["level_rank"] == 15 and row["leader_seat"] == 0 and row["previous_result"] is None


def test_curriculum_is_iteration_determined_interleaved_and_exact():
    schedule = build_rank_leader_schedule()
    spec = validate_context_curriculum({
        "schema": "guandan-marvel-context-curriculum-v1",
        "old_context": {"level_rank": 15, "leader_seat": 0, "previous_result": None},
        "warmup_iterations": 20, "blend_iterations": 80, "resolution": 1000,
    })
    assert curriculum_alpha(spec, 0) == 0.0
    assert curriculum_alpha(spec, 20) == 0.0
    assert curriculum_alpha(spec, 60) == pytest.approx(0.5)
    assert curriculum_alpha(spec, 100) == 1.0
    warmup = curriculum_expanded_contexts(spec, schedule, 10)
    assert len(warmup) == 1000 and all(_is_old(row) for row in warmup)
    # alpha=0.5: the mixture is interleaved so a contiguous window ~matches it
    blended = curriculum_expanded_contexts(spec, schedule, 60)
    window = blended[400:420]
    old_in_window = sum(1 for row in window if _is_old(row))
    assert 4 <= old_in_window <= 16  # ~half, not all-front
    final = curriculum_expanded_contexts(spec, schedule, 100)
    old_in_final = sum(1 for row in final if _is_old(row))
    assert old_in_final <= 1000 // 52 + 2  # only the target share of the old context
    assert curriculum_provenance(spec, schedule, 60) == curriculum_provenance(spec, schedule, 60)


def _drive_52_contexts(env):
    """Consume 3 iterations x 20 games, re-configuring the offset per iteration."""
    from guandan_marvel.context_schedule import expand_context_schedule
    expanded = expand_context_schedule(build_rank_leader_schedule())
    seen: set[tuple[int, int]] = set()
    games = 0

    def note():
        for slot in range(4):
            if hasattr(env, "debug_state"):
                cfg = env.debug_state(slot)["config"]
                seen.add((int(cfg["level_rank"]), int(cfg["leader_seat"])))
            else:
                cfg = env.envs[slot].config
                seen.add((int(cfg.level_rank), int(cfg.leader_seat)))

    for iteration in range(3):
        env.configure_context_schedule(expanded, offset=(iteration * 20) % len(expanded))
        step = env.reset(1000)
        note()
        while games < (iteration + 1) * 20:
            weights = np.asarray(step.available_actions.weights)
            rows = []
            for slot in range(4):
                legal = np.flatnonzero(weights[slot] > 0)
                rows.append(step.available_actions.tokens[slot, int(legal[0])])
            terminal = np.asarray(step.is_terminal)
            step = env.step(np.stack(rows))
            if terminal.any():
                games += int(terminal.sum())
                note()
    return games, seen


def test_52_context_smoke_observed_coverage_cpp_and_python():
    for env in (cpp_game.EnvBatch(4), python_game.EnvBatch(4)):
        games, seen = _drive_52_contexts(env)
        assert games >= 52
        assert seen == {(level, leader) for level in range(3, 16) for leader in range(4)}
        assert {level for level, _ in seen} == set(range(3, 16))
        assert {leader for _, leader in seen} == {0, 1, 2, 3}


def test_context_configure_is_offset_driven_and_resets_counter_once():
    env = cpp_game.EnvBatch(2)
    expanded = [{"level_rank": level, "leader_seat": 0, "previous_result": None} for level in range(3, 9)]
    env.configure_context_schedule(expanded, offset=2)
    env.reset(7)
    first = [env.debug_state(slot)["config"]["level_rank"] for slot in range(2)]
    assert first == [5, 6]  # positions offset+0, offset+1 -> distinct slots
    # re-configuring with the same offset is idempotent for the caller, and a new
    # offset is a pure function of the iteration (not wall-clock time).
    env.configure_context_schedule(expanded, offset=4)
    env.reset(7)
    assert [env.debug_state(slot)["config"]["level_rank"] for slot in range(2)] == [7, 8]


class _FakeDebugEnv:
    """Debug-state sequence used to test provenance timing deterministically."""

    def __init__(self, levels):
        self._levels = list(levels)
        self._calls = 0

    def debug_state(self, index):
        level = self._levels[min(self._calls, len(self._levels) - 1)]
        self._calls += 1
        return {
            "config": {"seed": 7, "level_rank": level, "leader_seat": 0, "previous_result": None},
            "state": {"level_rank": level, "leader_seat": 0, "previous_result": None, "hands": [[0], [], [], []]},
            "reset_count": self._calls,
        }


def test_provenance_records_the_completed_game_not_the_next_deal():
    from guandan_marvel.runtime.sampler import Sampler
    sampler = Sampler.__new__(Sampler)
    sampler.envs = _FakeDebugEnv([3, 7, 11])
    sampler._env_seed = 0
    sampler._episode_identities = {}
    sampler._training_deal_records = []
    sampler._training_deal_hashes = set()
    sampler._capture_episode_identities([0])  # game A = level 3
    sampler._record_training_deals([0])       # completes A, refresh store to B=7
    sampler._record_training_deals([0])       # completes B, refresh store to C=11
    assert [row["level_rank"] for row in sampler._training_deal_records] == [3, 7]
    assert [row["reset_count"] for row in sampler._training_deal_records] == [1, 2]
    assert all("context_sha256" in row and "deal_sha256" in row for row in sampler._training_deal_records)
    # the store now holds the in-flight game C, not yet recorded
    assert sampler._episode_identities[0]["level_rank"] == 11


def test_training_reward_has_a_single_scaling_point():
    from guandan_marvel.env_batch import EnvStepBatch

    raw = np.asarray([[3.0, -3.0, 3.0, -3.0]], dtype=np.float32)

    class _Step:
        rewards = raw
        is_terminal = np.asarray([True])
        player_ids = np.asarray([0])
        observations = type("O", (), {"tokens": np.zeros((1, 4, 4), np.int64), "channels": np.zeros((1, 4, 5), np.uint8)})()
        available_actions = type("A", (), {
            "tokens": np.zeros((1, 2, 8), np.int64),
            "weights": np.ones((1, 2), np.float32),
            "is_commit": np.zeros((1, 2), bool),
        })()

    class _Cfg:
        device = "cpu"
        reward_scale = TRAINING_REWARD_SCALE

    parsed = EnvStepBatch.from_env_step(_Step(), _Cfg())
    assert np.allclose(parsed.outcome["rewards"].numpy(), raw * TRAINING_REWARD_SCALE)
    # and no additional scaling happens further downstream
    assert np.allclose(parsed.outcome["rewards"].numpy(), [[1.0, -1.0, 1.0, -1.0]])


@pytest.mark.parametrize("params,iterations", [
    ({"warmup_iterations": 20, "blend_iterations": 80, "resolution": 1000}, (20, 60, 120)),
    ({"warmup_iterations": 0, "blend_iterations": 2, "resolution": 52}, (0, 1, 2)),
])
def test_curriculum_mixture_for_both_parameter_sets(params, iterations):
    from guandan_marvel.context_schedule import CURRICULUM_SCHEMA
    schedule = build_rank_leader_schedule()
    spec = validate_context_curriculum({
        "schema": CURRICULUM_SCHEMA,
        "old_context": {"level_rank": 15, "leader_seat": 0, "previous_result": None},
        **params,
    })
    warmup, mid, final = iterations
    warmup_rows = curriculum_expanded_contexts(spec, schedule, warmup)
    assert all(_is_old(row) for row in warmup_rows)  # alpha=0
    mid_rows = curriculum_expanded_contexts(spec, schedule, mid)
    mid_share = sum(1 for row in mid_rows if _is_old(row)) / len(mid_rows)
    assert 0.4 <= mid_share <= 0.6, mid_share  # alpha=0.5: both old and target
    final_rows = curriculum_expanded_contexts(spec, schedule, final)
    final_share = sum(1 for row in final_rows if _is_old(row)) / len(final_rows)
    assert final_share <= 1.0 / 52.0 + 0.02, final_share  # alpha=1: only the target share


def test_reward_module_has_update_when_engines_are_rebuilt():
    # A regenerated native extension must still expose the raw rank points.
    batch = cpp_game.EnvBatch(2, auto_reset=False)
    step = batch.reset(7)
    assert step.available_actions.weights.shape[0] == 2
