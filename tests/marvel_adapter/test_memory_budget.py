from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from guandan_marvel import game
from guandan_marvel.buffer import BufferConfig, TrajectoryBuffer, require_trajectory_memory_budget, trajectory_memory_budget


def test_default_dense_budget_matches_actual_field_specs_without_large_allocation(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("large tensor allocation is forbidden in the memory estimator")

    monkeypatch.setattr(torch, "zeros", forbidden)
    budget = trajectory_memory_budget(
        game.GAME_SPEC,
        rollout_batch_size=512,
        collect_batch_size=2048,
        world_size=2,
    )
    assert budget["per_trajectory_bytes"] == sum(field["bytes"] for field in budget["fields"])
    assert budget["per_trajectory_mib"] == pytest.approx(87.14844512939453)
    assert budget["sampler_gib_global"] == pytest.approx(261.4453353881836)
    assert budget["sampler_gib_per_rank"] == pytest.approx(130.7226676940918)
    assert {field["name"] for field in budget["fields"]} >= {
        "policy_probs", "behavior_action_prob", "on_policy_valid",
    }


def test_memory_preflight_rejects_before_dense_buffer_allocation() -> None:
    required = trajectory_memory_budget(
        game.GAME_SPEC,
        rollout_batch_size=512,
        collect_batch_size=2048,
        world_size=2,
    )["sampler_bytes_per_rank"]
    with pytest.raises(MemoryError, match="exceeds"):
        require_trajectory_memory_budget(
            game.GAME_SPEC,
            rollout_batch_size=512,
            collect_batch_size=2048,
            world_size=2,
            max_sampler_bytes_per_rank=int(required) - 1,
        )
    accepted = require_trajectory_memory_budget(
        game.GAME_SPEC,
        rollout_batch_size=512,
        collect_batch_size=2048,
        world_size=2,
        max_sampler_bytes_per_rank=int(required),
    )
    assert accepted["sampler_bytes_per_rank"] == required


def test_explicit_local_smoke_profile_is_small_and_not_the_default_protocol_shape() -> None:
    smoke = replace(
        game.GAME_SPEC,
        identifier=f"{game.GAME_IDENTIFIER}-local-smoke-T512",
        max_game_horizon=512,
    )
    budget = trajectory_memory_budget(
        smoke,
        rollout_batch_size=1,
        collect_batch_size=1,
        world_size=1,
    )
    assert smoke.max_game_horizon != game.GAME_SPEC.max_game_horizon
    assert budget["per_trajectory_mib"] == pytest.approx(10.893562316894531)
    assert budget["sampler_bytes_global"] / 2**20 == pytest.approx(32.680686950683594)


def test_pre_repair_buffer_schema_is_explicitly_incompatible() -> None:
    current = TrajectoryBuffer(1, BufferConfig(game=game.GAME_SPEC, device="cpu"))
    state = current.state_dict()
    state["schema"] = "guandan-marvel-buffer-v1"
    with pytest.raises(ValueError, match="Unsupported"):
        current.load_state_dict(state)
