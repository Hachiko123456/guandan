"""Phase-6 repair: frozen opponent caches advance every step and resume exactly."""
from __future__ import annotations

import copy

import pytest
import torch

from guandan_marvel.config import OpponentPoolConfig
from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.phase6.opponent_pool import (
    FrozenPolicySession, OpponentPool, PoolPolicyRouter,
)

from tests.marvel_training._phase6_pool_fixtures import make_tiny_eval_checkpoint

P = 4
A = 8
L = 48
AL = 8


def _step(player: int | list[int], seed: int) -> TensorDict:
    players = torch.as_tensor([player] if isinstance(player, int) else player, dtype=torch.long)
    batch = players.numel()
    generator = torch.Generator().manual_seed(int(seed))
    return TensorDict({
        "num_steps": torch.ones(batch, dtype=torch.long),
        "player_ids": players.unsqueeze(1),
        "obs_tokens": torch.randint(1, 309, (batch, 1, P, L), generator=generator, dtype=torch.long),
        "obs_channels": torch.randint(0, 2, (batch, 1, P, 512), generator=generator, dtype=torch.uint8),
        "avail_tokens": torch.randint(1, 309, (batch, 1, A, AL), generator=generator, dtype=torch.long),
        "avail_weights": torch.ones(batch, 1, A),
    }, batch_size=[batch])


def _common(fixture):
    return dict(
        device="cpu",
        expected_architecture="v1.4-macro-fuse",
        expected_protocol_version=str(fixture["game"].protocol_version),
        expected_backend="cpp",
        expected_backend_identity_hash=None,
        model_feature_identity={
            "history_dim": 16, "feature_dim": 24, "decoder_layers": 1,
            "decoder_heads": 4, "state_layers": 1, "action_layers": 1,
            "fuse_layers": 1,
        },
        batch_slots=1,
    )


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    return make_tiny_eval_checkpoint(tmp_path_factory.mktemp("phase6-history"))


def _pool(fixture, seed=5):
    return OpponentPool.from_config(
        OpponentPoolConfig(
            enabled=True, entries=(fixture["entry"],), sampling_seed=seed,
        ),
        **_common(fixture),
    )


def _router(pool, *, batch=1):
    return PoolPolicyRouter(
        pool=pool, batch_size=batch, num_players=P, max_ctx_len=64,
        device="cpu", max_resident_policies=1,
    )


def _current_probs(batch: int) -> torch.Tensor:
    return torch.full((batch, A), 1.0 / A)


def test_frozen_cache_advances_on_every_step_for_owner_sequence():
    import tempfile
    from pathlib import Path

    fixture = make_tiny_eval_checkpoint(Path(tempfile.mkdtemp(prefix="phase6-history-")))
    pool = _pool(fixture)
    router = _router(pool)
    router.prepare(rollout_buffer=None)
    session = router.sessions["opponent-a"]
    assert session.slot_sequence_lengths() == [0]
    players = [0, 2, 1, 3]  # current, current, opponent, opponent
    for index, player in enumerate(players, start=1):
        step = _step(player, 100 + index)
        router.policy_probabilities(
            step_rollout=step,
            parsed_envs_step={"player_ids": torch.tensor([player])},
            current_probabilities=_current_probs(1),
        )
        assert session.slot_sequence_lengths() == [index]
    # The frozen decision at step 4 must equal a fresh full-history session.
    reference = FrozenPolicySession(
        entry=pool.entries[0], batch_size=1, num_players=P, max_ctx_len=64, device="cpu",
    )
    for index, player in enumerate(players, start=1):
        reference.probabilities(_step(player, 100 + index))
    step5 = _step(1, 200)
    with torch.no_grad():
        expected = reference.probabilities(step5)
        actual = session.probabilities(step5)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_populate_and_mid_episode_resume_match_live_cache():
    import tempfile
    from pathlib import Path

    fixture = make_tiny_eval_checkpoint(Path(tempfile.mkdtemp(prefix="phase6-history2-")))
    pool = _pool(fixture)
    router = _router(pool)
    router.prepare(rollout_buffer=None)
    steps = [_step(player, 300 + index) for index, player in enumerate([0, 2, 1])]
    for step, player in zip(steps, [0, 2, 1]):
        router.policy_probabilities(
            step_rollout=step,
            parsed_envs_step={"player_ids": torch.tensor([player])},
            current_probabilities=_current_probs(1),
        )
    next_step = _step(3, 400)
    with torch.no_grad():
        control = router.sessions["opponent-a"].probabilities(next_step)

    # Mid-episode resume: state only, then rebuild from the recorded history.
    state = router.state_dict()
    restored_pool = _pool(fixture)
    restored = _router(restored_pool)
    restored.load_state_dict(copy.deepcopy(state))
    history = TensorDict({
        "num_steps": torch.tensor([3], dtype=torch.long),
        "player_ids": torch.tensor([[0, 2, 1]], dtype=torch.long),
        "obs_tokens": torch.cat([step["obs_tokens"] for step in steps], dim=1),
        "obs_channels": torch.cat([step["obs_channels"] for step in steps], dim=1),
        "avail_tokens": torch.cat([step["avail_tokens"] for step in steps], dim=1),
        "avail_weights": torch.cat([step["avail_weights"] for step in steps], dim=1),
    }, batch_size=[1])
    restored.prepare(rollout_buffer=history)
    with torch.no_grad():
        resumed = restored.sessions["opponent-a"].probabilities(next_step)
    torch.testing.assert_close(resumed, control, rtol=1e-5, atol=1e-6)


def test_on_commit_resets_only_the_finished_slot(fixture):
    pool = _pool(fixture)
    router = _router(pool)
    router.prepare(rollout_buffer=None)
    step = _step(0, 700)
    router.policy_probabilities(
        step_rollout=step,
        parsed_envs_step={"player_ids": torch.tensor([0])},
        current_probabilities=_current_probs(1),
    )
    session = router.sessions["opponent-a"]
    assert session.slot_sequence_lengths() == [1]
    router.on_commit(torch.tensor([True]))
    assert session.slot_sequence_lengths() == [0]
    assignment = pool.assignment(slot=0)
    assert assignment.episode_index == 1


def test_two_slots_do_not_pollute_each_other_history():
    import tempfile
    from pathlib import Path

    fixture = make_tiny_eval_checkpoint(Path(tempfile.mkdtemp(prefix="phase6-history3-")))
    pool_a = _pool(fixture)
    pool_b = _pool(fixture)
    router_a = _router(pool_a, batch=2)
    router_b = _router(pool_b, batch=2)
    router_a.prepare(rollout_buffer=None)
    router_b.prepare(rollout_buffer=None)
    step_a = _step([0, 1], 500)
    step_b = step_a.clone()
    step_b["obs_tokens"][1] = _step([0, 1], 999)["obs_tokens"][1]
    router_a.policy_probabilities(
        step_rollout=step_a,
        parsed_envs_step={"player_ids": torch.tensor([0, 1])},
        current_probabilities=_current_probs(2),
    )
    router_b.policy_probabilities(
        step_rollout=step_b,
        parsed_envs_step={"player_ids": torch.tensor([0, 1])},
        current_probabilities=_current_probs(2),
    )
    next_step_a = _step([0, 1], 600)
    next_step_b = next_step_a.clone()
    next_step_b["obs_tokens"][1] = _step([0, 1], 998)["obs_tokens"][1]
    with torch.no_grad():
        probs_a = router_a.sessions["opponent-a"].probabilities(next_step_a)[0]
        probs_b = router_b.sessions["opponent-a"].probabilities(next_step_b)[0]
    torch.testing.assert_close(probs_a, probs_b, rtol=1e-6, atol=1e-7)
