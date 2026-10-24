"""Stage-3 DMC Q head, replay buffer and learner contract tests.

The decisive test is ``test_synthetic_q_learner_converges``: a target that is a
linear function of a 64-d action-feature subset *and* a 256-d channel subset is
regressed through the real trunk.  A learner that fails to route gradients
through the history decoder, the state MLP or the new action encoder cannot
reach the required held-out ``qz_r2 > 0.9``.  The companion shuffled-target run
is a negative control proving the metric is not trivially high.

No test mutates ``model/``, ``buffer.py`` or the IPPO/VRPO training code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig
from guandan_marvel.dmc.q_agent import (
    ACTION_FEATURE_DIM,
    DMCAgent,
    DMCCheckpointError,
    INVALID_Q_SENTINEL,
)
from guandan_marvel.dmc.replay import (
    DMCReplay,
    DMCReplayBatch,
    DMCTransition,
    dmc_replay_memory_budget,
)
from guandan_marvel.dmc.learner import DMCLearner, DMCNonFiniteError, qz_r2

CHANNEL_DIM = 256
MAX_OBS_TOKENS = 8
NUM_TOKENS = 32
A_IDX = np.arange(0, 16)
C_IDX = np.arange(0, 16)


def _game(
    *,
    channel_dim: int = CHANNEL_DIM,
    num_tokens: int = NUM_TOKENS,
    max_obs_tokens: int = MAX_OBS_TOKENS,
) -> GameSpec:
    return GameSpec(
        identifier="dmc-stage3-test-only",
        num_players=4,
        num_tokens=num_tokens,
        pad_token=0,
        num_channels=channel_dim,
        max_game_horizon=64,
        max_observation_tokens=max_obs_tokens,
        max_action_tokens=8,
        max_available_actions=256,
    )


def _cfg(**game_kwargs) -> TrainConfig:
    return TrainConfig(
        game=_game(**game_kwargs),
        device="cpu",
        model=ModelConfig(
            architecture="v1.3",
            history_dim=16,
            feature_dim=24,
            decoder_layers=1,
            decoder_heads=4,
            state_layers=1,
            action_layers=1,
            fuse_layers=1,
        ),
    )


def _synthetic_dataset(n: int, rng: np.random.Generator):
    """Action + channel target: z = w_a . a_sel + w_c . c_sel (standardised)."""
    w_a = rng.standard_normal(len(A_IDX)).astype(np.float32)
    w_c = rng.standard_normal(len(C_IDX)).astype(np.float32)
    action = np.zeros((n, ACTION_FEATURE_DIM), np.float32)
    action[:, A_IDX] = rng.random((n, len(A_IDX))) < 0.5
    channels = np.zeros((n, CHANNEL_DIM), np.uint8)
    channels[:, C_IDX] = rng.random((n, len(C_IDX))) < 0.5
    raw = (action[:, A_IDX] * w_a).sum(1) + (
        channels[:, C_IDX].astype(np.float32) * w_c
    ).sum(1)
    mean, std = float(raw.mean()), float(raw.std())
    z = ((raw - mean) / std).astype(np.float32)
    return action, channels, z, (w_a, w_c, mean, std)


def _heldout(rng: np.random.Generator, count: int, stats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w_a, w_c, mean, std = stats
    action = np.zeros((count, ACTION_FEATURE_DIM), np.float32)
    action[:, A_IDX] = rng.random((count, len(A_IDX))) < 0.5
    channels = np.zeros((count, CHANNEL_DIM), np.uint8)
    channels[:, C_IDX] = rng.random((count, len(C_IDX))) < 0.5
    raw = (action[:, A_IDX] * w_a).sum(1) + (
        channels[:, C_IDX].astype(np.float32) * w_c
    ).sum(1)
    z = ((raw - mean) / std).astype(np.float32)
    return action, channels, z


def _fill_replay(action, channels, z, *, capacity=None) -> DMCReplay:
    n = int(z.shape[0])
    replay = DMCReplay(
        capacity or n,
        max_obs_tokens=MAX_OBS_TOKENS,
        channel_dim=CHANNEL_DIM,
        num_tokens=NUM_TOKENS,
        seed=7,
    )
    tokens = np.full((n, 2), 1, dtype=np.int32)  # constant history; no padding
    replay.push_batch({
        "obs_tokens": tokens,
        "obs_token_lengths": np.full(n, 2, dtype=np.int32),
        "obs_channels": channels,
        "action_feats": action,
        "z": z,
        "seat": (np.arange(n) % 4).astype(np.int8),
        "game_id": np.arange(n, dtype=np.int64),
    })
    return replay


def _heldout_r2(agent: DMCAgent, action, channels, z) -> float:
    n = int(z.shape[0])
    tokens = torch.ones(n, 2, dtype=torch.long)
    with torch.no_grad():
        q = agent.q_values(
            tokens,
            torch.from_numpy(channels),
            torch.from_numpy(action).unsqueeze(1),
            torch.ones(n, 1, dtype=torch.bool),
            token_lengths=torch.full((n,), 2),
        )[:, 0]
    return qz_r2(q, torch.from_numpy(z))


# ---------------------------------------------------------------------------
# The core convergence test
# ---------------------------------------------------------------------------
def test_synthetic_q_learner_converges():
    torch.manual_seed(1234)
    rng = np.random.default_rng(20261024)
    N = 8192
    action, channels, z, stats = _synthetic_dataset(N, rng)
    replay = _fill_replay(action, channels, z)
    agent = DMCAgent(_cfg())
    learner = DMCLearner(agent, replay, batch_size=128, seed=2)

    ho_action, ho_channels, ho_z = _heldout(np.random.default_rng(99), 512, stats)
    curve = []
    for step in range(2200):
        metrics = learner.train_step()
        assert metrics["nonfinite_count"] == 0
        assert math.isfinite(metrics["loss"])
        assert math.isfinite(metrics["grad_norm"])
        if step % 200 == 0:
            curve.append((step, metrics["qz_r2"], metrics["loss"], metrics["grad_norm"]))

    heldout_r2 = _heldout_r2(agent, ho_action, ho_channels, ho_z)
    assert heldout_r2 > 0.9, f"synthetic convergence failed: held-out qz_r2={heldout_r2:.4f}"
    # The training-batch metric at the end of the run must agree.
    assert metrics["qz_r2"] > 0.9, (curve[-5:], metrics)
    assert curve[0][1] < heldout_r2, "the curve must show learning, not a constant metric"
    print("convergence curve (step, qz_r2, loss, grad_norm):", curve)


def test_shuffled_target_negative_control_does_not_converge():
    """The convergence assertion is meaningful: broken pairing gives ~0 R^2."""

    torch.manual_seed(1234)
    rng = np.random.default_rng(20261024)
    N = 8192
    action, channels, z, stats = _synthetic_dataset(N, rng)
    shuffled = z.copy()
    np.random.default_rng(5).shuffle(shuffled)
    replay = _fill_replay(action, channels, shuffled)
    agent = DMCAgent(_cfg())
    learner = DMCLearner(agent, replay, batch_size=128, seed=2)
    for _ in range(600):
        learner.train_step()

    ho_action, ho_channels, ho_z = _heldout(np.random.default_rng(99), 512, stats)
    heldout_r2 = _heldout_r2(agent, ho_action, ho_channels, ho_z)
    assert heldout_r2 < 0.5, f"shuffled target unexpectedly explains variance: {heldout_r2:.4f}"


def test_gradients_reach_history_action_and_state_paths():
    torch.manual_seed(321)
    rng = np.random.default_rng(11)
    action, channels, z, _ = _synthetic_dataset(64, rng)
    replay = _fill_replay(action, channels, z)
    agent = DMCAgent(_cfg())
    learner = DMCLearner(agent, replay, batch_size=16, seed=3)
    learner.train_step()

    grads = {name: param.grad for name, param in agent.named_parameters()}
    for prefix in (
        "backbone.state_tokens_embed",
        "backbone.decoder",
        "backbone.state_mlp",
        "backbone.logit_head",
        "action_input_proj",
        "action_trunk",
    ):
        matched = [
            grad for name, grad in grads.items()
            if name.startswith(prefix) and grad is not None and grad.abs().sum() > 0
        ]
        assert matched, f"no non-zero gradient reached {prefix}"


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
def _transition(game_id: int, **overrides) -> DMCTransition:
    values = {
        "obs_tokens": (game_id % 4 + 1, 1),
        "obs_channels": np.zeros(4, dtype=np.uint8),
        "action_feats": np.full(ACTION_FEATURE_DIM, game_id / 10.0, dtype=np.float32),
        "z": float(game_id) / 3.0,
        "seat": game_id % 4,
        "game_id": game_id,
    }
    values.update(overrides)
    return DMCTransition(**values)


def _small_replay(capacity: int = 4) -> DMCReplay:
    return DMCReplay(
        capacity, max_obs_tokens=4, channel_dim=4, num_tokens=8, seed=0,
    )


def test_replay_fifo_eviction_and_counters():
    replay = _small_replay(4)
    assert replay.push_batch([_transition(i) for i in range(4)]) == 4
    assert len(replay) == 4 and replay.is_full
    assert replay.total_pushed == 4 and replay.write_position == 0

    replay.push_batch([_transition(4), _transition(5)])
    assert len(replay) == 4 and replay.is_full
    assert replay.total_pushed == 6 and replay.write_position == 2
    batch = replay.sample(4, np.random.default_rng(0))
    assert set(batch.game_id.tolist()) == {2, 3, 4, 5}
    # The evicted rows are gone, not merely hidden.
    assert 0 not in batch.game_id and 1 not in batch.game_id


def test_replay_sample_is_uniform_without_replacement_and_seed_reproducible():
    replay = _small_replay(10)
    replay.push_batch([_transition(i) for i in range(10)])

    draw = replay.sample(10, np.random.default_rng(4))
    assert len(set(draw.game_id.tolist())) == 10, "sample must be without replacement"

    rng = np.random.default_rng(123)
    counts = np.zeros(10, dtype=np.int64)
    draws = 50_000
    for _ in range(draws):
        sample = replay.sample(1, rng)
        counts[int(sample.game_id[0])] += 1
    expected = draws / 10
    sigma = math.sqrt(draws * 0.1 * 0.9)
    assert np.all(np.abs(counts - expected) < 6 * sigma), counts

    first = [replay.sample(5, np.random.default_rng(seed)).game_id.tolist() for seed in (9, 9, 9)]
    second = [replay.sample(5, np.random.default_rng(9)).game_id.tolist() for _ in range(3)]
    assert first == second
    # Two replays with identical content and the supplied Generator agree.
    twin = _small_replay(10)
    twin.push_batch([_transition(i) for i in range(10)])
    assert replay.sample(6, np.random.default_rng(77)).game_id.tolist() == (
        twin.sample(6, np.random.default_rng(77)).game_id.tolist()
    )


def test_replay_dtypes_and_packing():
    replay = _small_replay(4)
    replay.push_batch([_transition(1, obs_tokens=(3,))])
    batch = replay.sample(1, np.random.default_rng(0))
    assert isinstance(batch, DMCReplayBatch)
    assert batch.obs_tokens.dtype == np.int32
    assert batch.obs_token_lengths.dtype == np.int32
    assert batch.obs_channels.dtype == np.uint8
    assert batch.action_feats.dtype == np.float32
    assert batch.z.dtype == np.float32
    assert batch.seat.dtype == np.int8
    assert batch.game_id.dtype == np.int64
    assert batch.obs_token_lengths.tolist() == [1]
    # A padded token row is small; sample() trims to the batch max length.
    assert batch.obs_tokens.shape == (1, 1)

    replay.push_batch([_transition(2, obs_tokens=(2, 3, 4))])
    mixed = replay.sample(2, np.random.default_rng(0))
    assert mixed.obs_tokens.shape[1] == 3
    for row, length in zip(mixed.obs_tokens, mixed.obs_token_lengths):
        assert np.all(row[int(length):] == replay.pad_token)


def test_replay_validation_is_atomic():
    replay = _small_replay(4)
    replay.push_batch([_transition(0)])
    bad_cases = [
        _transition(1, z=float("nan")),
        _transition(2, seat=4),
        _transition(3, obs_tokens=(1, 1, 1, 1, 1)),
        _transition(4, action_feats=np.full(ACTION_FEATURE_DIM, np.inf, dtype=np.float32)),
        _transition(5, action_feats=np.zeros(ACTION_FEATURE_DIM - 1, dtype=np.float32)),
        _transition(6, obs_channels=np.zeros(5, dtype=np.uint8)),
        _transition(7, obs_channels=np.full(4, 256, dtype=np.int16)),
    ]
    for bad in bad_cases:
        with pytest.raises((ValueError, KeyError)):
            replay.push_batch([_transition(99), bad])
        assert len(replay) == 1, "a rejected batch must not write a prefix"
    with pytest.raises(ValueError, match="without replacement"):
        replay.sample(2)
    with pytest.raises(ValueError, match="empty"):
        _small_replay(4).sample(1)


def _buffer_state(replay: DMCReplay) -> tuple[int, int, int]:
    return len(replay), replay.write_position, replay.total_pushed


def test_push_batch_atomic_when_game_id_exceeds_int64():
    replay = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=8)
    replay.push_batch([_transition(0)])
    before = _buffer_state(replay)
    for bad_game_id in (2**63, 10**20):
        # The first row is valid; the second overflows int64 only when written.
        with pytest.raises(ValueError, match="game_id"):
            replay.push_batch([_transition(100), _transition(bad_game_id)])
        assert _buffer_state(replay) == before, "a rejected batch mutated buffer state"
        visible = replay.sample(len(replay), np.random.default_rng(0)).game_id.tolist()
        assert visible == [0], f"a rejected prefix row leaked into the buffer: {visible}"


def test_push_batch_atomic_when_seat_exceeds_int8():
    # Chosen implementation: num_players is capped at int8's 128 values, so an
    # int8-overflowing seat can no longer be configured.
    with pytest.raises(ValueError, match="num_players"):
        DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_players=129, num_tokens=8)
    with pytest.raises(ValueError, match="num_players"):
        DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_players=1000, num_tokens=8)

    replay = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=8)
    replay.push_batch([_transition(0)])
    before = _buffer_state(replay)
    with pytest.raises(ValueError, match="seat"):
        replay.push_batch([_transition(100), _transition(101, seat=300)])
    assert _buffer_state(replay) == before, "a rejected batch mutated buffer state"
    visible = replay.sample(len(replay), np.random.default_rng(0)).game_id.tolist()
    assert visible == [0], f"a rejected prefix row leaked into the buffer: {visible}"


def test_num_tokens_cannot_exceed_int32():
    with pytest.raises(ValueError, match="num_tokens"):
        DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=2**31 + 1)
    with pytest.raises(ValueError, match="num_tokens"):
        DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=2**32)

    bounded = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=2**31)
    bounded.push_batch([_transition(0, obs_tokens=(2**31 - 1,))])
    assert len(bounded) == 1

    unbounded = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=None)
    with pytest.raises(ValueError, match="obs_tokens"):
        unbounded.push_batch([_transition(0, obs_tokens=(2**31,))])
    assert len(unbounded) == 0
    unbounded.push_batch([_transition(0, obs_tokens=(2**31 - 1,))])
    assert len(unbounded) == 1


def test_push_batch_atomic_for_batched_uint64_game_ids():
    replay = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=8)
    replay.push_batch([_transition(0)])
    before = _buffer_state(replay)
    batch = {
        "obs_tokens": np.array([[1, 1], [2, 2]], dtype=np.int32),
        "obs_token_lengths": np.array([2, 2], dtype=np.int32),
        "obs_channels": np.zeros((2, 4), dtype=np.uint8),
        "action_feats": np.zeros((2, ACTION_FEATURE_DIM), dtype=np.float32),
        "z": np.array([0.1, 0.2], dtype=np.float32),
        "seat": np.array([1, 2], dtype=np.int8),
        "game_id": np.array([100, 2**63], dtype=np.uint64),
    }
    with pytest.raises(ValueError, match="game_id"):
        replay.push_batch(batch)
    assert _buffer_state(replay) == before, "a rejected batch mutated buffer state"
    assert replay.sample(1, np.random.default_rng(0)).game_id.tolist() == [0]


def test_replay_memory_budget_matches_hand_arithmetic():
    budget = dmc_replay_memory_budget(500_000, max_obs_tokens=48, channel_dim=256)
    expected = (
        500_000 * 48 * 4       # obs_tokens int32
        + 500_000 * 4          # obs_token_lengths int32
        + 500_000 * 256 * 1    # obs_channels uint8
        + 500_000 * 64 * 4     # action_feats float32
        + 500_000 * 4          # z float32
        + 500_000 * 1          # seat int8
        + 500_000 * 8          # game_id int64
    )
    assert budget["total_bytes"] == expected == 360_500_000
    assert sum(field["bytes"] for field in budget["fields"]) == expected
    assert budget["total_gib"] == pytest.approx(expected / 2**30)
    assert budget["channel_dim"] == 256 and budget["action_feature_dim"] == 64

    replay = DMCReplay(7, max_obs_tokens=3, channel_dim=5)
    small = dmc_replay_memory_budget(7, max_obs_tokens=3, channel_dim=5)
    assert replay.nbytes() == small["total_bytes"]


# ---------------------------------------------------------------------------
# Q values / masks
# ---------------------------------------------------------------------------
def test_q_values_mask_short_candidates_and_token_padding():
    torch.manual_seed(5)
    agent = DMCAgent(_cfg())
    B, K = 3, 5
    tokens = torch.randint(1, NUM_TOKENS, (B, 6))
    lengths = torch.tensor([6, 3, 5])
    channels = torch.randint(0, 2, (B, CHANNEL_DIM), dtype=torch.uint8)
    action = torch.randn(B, K, ACTION_FEATURE_DIM)
    masks = torch.ones(B, K, dtype=torch.bool)
    masks[0, 3:] = False
    q = agent.q_values(tokens, channels, action, masks, token_lengths=lengths)
    assert q.shape == (B, K)
    assert torch.equal(q[0, 3:], torch.full((2,), INVALID_Q_SENTINEL))
    assert torch.isfinite(q[masks]).all()

    # Invalid candidates do not leak into valid scores.
    changed = action.clone()
    changed[0, 3:] += 1000.0
    q2 = agent.q_values(tokens, channels, changed, masks, token_lengths=lengths)
    assert torch.allclose(q2[0, :3], q[0, :3], atol=1e-6)

    # Tokens beyond the true length are ignored (attention mask + last index).
    padded = tokens.clone()
    padded[1, 3:] = 7
    q3 = agent.q_values(padded, channels, action, masks, token_lengths=lengths)
    assert torch.equal(q3, q)

    with pytest.raises(ValueError, match="token_lengths"):
        agent.q_values(tokens, channels, action, masks, token_lengths=torch.tensor([7, 1, 1]))
    with pytest.raises(ValueError, match="channel width"):
        agent.q_values(tokens, channels[:, :-1], action, masks)
    with pytest.raises(ValueError, match="masks"):
        agent.q_values(tokens, channels, action, torch.ones(B, K + 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="player_ids"):
        agent.q_values(tokens, channels, action, masks, player_ids=torch.tensor([0, 1, 4]))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def test_checkpoint_rejects_ippo_and_accepts_dmc_v1():
    torch.manual_seed(6)
    agent = DMCAgent(_cfg())
    payload = agent.checkpoint_payload()
    assert payload["schema"] == "dmc-v1"
    clone = DMCAgent(_cfg())
    clone.load_state_dict(payload)
    for key, value in agent.state_dict().items():
        assert torch.equal(clone.state_dict()[key], value), key

    ippo = {
        "schema": "guandan-marvel-training-checkpoint-v1",
        "algorithm": "ippo",
        "agent_state_dict": agent.state_dict(),
    }
    with pytest.raises(DMCCheckpointError, match="policy_import"):
        clone.load_state_dict(ippo)
    with pytest.raises(DMCCheckpointError, match="policy_import"):
        clone.load_state_dict(agent.state_dict())  # raw weights carry no DMC schema

    bad_version = dict(payload)
    bad_version["action_feature_version"] = "GD-DMC-ACTION-0"
    with pytest.raises(DMCCheckpointError, match="action_feature_version"):
        clone.load_state_dict(bad_version)


# ---------------------------------------------------------------------------
# AMP / GradScaler fail-safe semantics
# ---------------------------------------------------------------------------
class _InfBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * float("inf")


class _OverflowAgent(nn.Module):
    action_feature_dim = ACTION_FEATURE_DIM
    enable_ntp = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def q_values(self, obs_tokens, obs_channels, action_feats, masks, *, token_lengths=None, return_ntp=False):
        total = action_feats.sum(dim=(1, 2))  # [B]
        base = (self.weight * total).unsqueeze(1)  # [B, 1]
        return _InfBackward.apply(base)


class _NaNAgent(_OverflowAgent):
    def q_values(self, obs_tokens, obs_channels, action_feats, masks, *, token_lengths=None, return_ntp=False):
        return torch.full((obs_tokens.shape[0], 1), float("nan"))


def _learner_for(agent: nn.Module, **kwargs) -> DMCLearner:
    replay = _small_replay(32)
    replay.push_batch([
        _transition(i, obs_channels=np.zeros(4, dtype=np.uint8),
                    action_feats=np.full(ACTION_FEATURE_DIM, (i + 1) / 10.0, dtype=np.float32))
        for i in range(32)
    ])
    return DMCLearner(agent, replay, batch_size=8, seed=0, **kwargs)


def test_grad_scaler_overflow_skips_step_without_nan_propagation(tmp_path):
    import json

    torch.manual_seed(0)
    agent = _OverflowAgent()
    before = {name: value.detach().clone() for name, value in agent.state_dict().items()}
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=65536.0)
    log = tmp_path / "overflow.jsonl"
    learner = _learner_for(agent, grad_scaler=scaler, log_path=log)
    metrics = learner.train_step()

    assert metrics["step_skipped"] is True
    assert metrics["scaler_scale"] == 32768.0
    assert metrics["nonfinite_count"] == 0  # an overflow is not a forward fault
    assert all(torch.isfinite(p).all() for p in agent.parameters())
    for name, value in agent.state_dict().items():
        assert torch.equal(value, before[name]), f"skipped step changed {name}"
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["step_skipped"] is True
    assert record["grad_norm"] is None, "non-finite grad_norm must not emit invalid JSON"


def test_fp32_nonfinite_gradient_fails_closed_without_a_scaler():
    torch.manual_seed(0)
    agent = _OverflowAgent()
    before = {name: value.detach().clone() for name, value in agent.state_dict().items()}
    learner = _learner_for(agent)  # CPU: no autocast, no scaler
    with pytest.raises(DMCNonFiniteError, match="gradient norm"):
        learner.train_step()
    for name, value in agent.state_dict().items():
        assert torch.equal(value, before[name])


def test_nonfinite_forward_fails_closed():
    torch.manual_seed(0)
    agent = _NaNAgent()
    before = {name: value.detach().clone() for name, value in agent.state_dict().items()}
    learner = _learner_for(agent)
    with pytest.raises(DMCNonFiniteError, match="non-finite"):
        learner.train_step()
    for name, value in agent.state_dict().items():
        assert torch.equal(value, before[name])


# ---------------------------------------------------------------------------
# NTP auxiliary head (default off)
# ---------------------------------------------------------------------------
def test_ntp_head_is_optional_and_default_off():
    torch.manual_seed(8)
    base = DMCAgent(_cfg())
    assert base.enable_ntp is False
    B, K = 2, 3
    args = (
        torch.ones(B, 2, dtype=torch.long),
        torch.zeros(B, CHANNEL_DIM, dtype=torch.uint8),
        torch.zeros(B, K, ACTION_FEATURE_DIM),
        torch.ones(B, K, dtype=torch.bool),
    )
    q, ntp = base.q_values(*args, return_ntp=True)
    assert q.shape == (B, K) and ntp is None

    with_ntp = DMCAgent(_cfg(), enable_ntp_head=True)
    assert with_ntp.enable_ntp is True
    q, ntp = with_ntp.q_values(*args, return_ntp=True)
    assert q.shape == (B, K) and ntp.shape == (B, NUM_TOKENS)

    with pytest.raises(ValueError, match="enable_ntp_head"):
        DMCAgent(_cfg(), ntp_weight=0.1)


def test_jsonl_metrics_and_amp_cpu_rejection(tmp_path):
    import json

    torch.manual_seed(10)
    agent = DMCAgent(_cfg())
    replay = DMCReplay(
        32, max_obs_tokens=4, channel_dim=CHANNEL_DIM, num_tokens=NUM_TOKENS, seed=0,
    )
    replay.push_batch([
        _transition(i, obs_channels=np.zeros(CHANNEL_DIM, dtype=np.uint8))
        for i in range(32)
    ])
    with pytest.raises(ValueError, match="only available on CUDA"):
        DMCLearner(agent, replay, batch_size=8, amp=True)

    log = tmp_path / "metrics.jsonl"
    learner = DMCLearner(agent, replay, batch_size=8, seed=0, log_path=log)
    metrics = learner.train_step()
    expected = {
        "step", "loss", "q_mean", "q_std", "z_mean", "z_std", "qz_r2",
        "grad_norm", "scaler_scale", "nonfinite_count", "step_skipped",
    }
    assert expected <= set(metrics)
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["step"] == 0 and record["scaler_scale"] == 1.0
    assert record["nonfinite_count"] == 0 and record["step_skipped"] is False


def test_replay_rejects_out_of_range_tokens_without_wraparound():
    replay = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=8)
    replay.push_batch([_transition(0)])
    for token in (2**40, 2**32 + 3, -2**32 + 1):
        with pytest.raises(ValueError, match="obs_tokens"):
            replay.push_batch([_transition(1, obs_tokens=(token,))])
        assert len(replay) == 1, "a rejected token must not be written"

    unbounded = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=None)
    with pytest.raises(ValueError, match="obs_tokens"):
        unbounded.push_batch([_transition(1, obs_tokens=(2**31,))])
    assert len(unbounded) == 0
    unbounded.push_batch([_transition(2, obs_tokens=(0, 2**31 - 1))])
    assert len(unbounded) == 1

    replay.push_batch([_transition(2, obs_tokens=(0, 1, 7))])
    assert len(replay) == 2


def test_replay_rejects_non_integral_mapping_fields():
    replay = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=8)
    replay.push_batch([_transition(0)])
    base = {
        "obs_tokens": (2,),
        "obs_channels": np.zeros(4, dtype=np.uint8),
        "action_feats": np.zeros(ACTION_FEATURE_DIM, dtype=np.float32),
        "z": 0.1,
        "seat": 1,
        "game_id": 2,
    }
    for field, value in (("obs_tokens", (2.9,)), ("seat", 1.9), ("game_id", 2.9)):
        bad = dict(base)
        bad[field] = value
        with pytest.raises(ValueError, match=field):
            replay.push_batch([bad])
        assert len(replay) == 1, "a rejected mapping must not be written"

    batched = {
        "obs_tokens": np.array([[2.9]], dtype=np.float64),
        "obs_channels": np.zeros((1, 4), dtype=np.uint8),
        "action_feats": np.zeros((1, ACTION_FEATURE_DIM), dtype=np.float32),
        "z": np.array([0.1]),
        "seat": np.array([1]),
        "game_id": np.array([2]),
    }
    with pytest.raises(ValueError, match="obs_tokens"):
        replay.push_batch(batched)
    assert len(replay) == 1


def test_replay_rejects_z_not_representable_in_float32():
    replay = DMCReplay(4, max_obs_tokens=4, channel_dim=4, num_tokens=8)
    replay.push_batch([_transition(0)])
    for z in (1e300, 1e39, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="z"):
            replay.push_batch([_transition(1, z=z)])
        assert len(replay) == 1
    replay.push_batch([_transition(2, z=0.1)])
    assert len(replay) == 2


def test_learner_rejects_channel_or_player_mismatch():
    torch.manual_seed(12)
    agent = DMCAgent(_cfg())
    wide = DMCReplay(8, max_obs_tokens=4, channel_dim=128, num_tokens=NUM_TOKENS)
    with pytest.raises(ValueError, match="channel_dim"):
        DMCLearner(agent, wide)
    two_players = DMCReplay(
        8, max_obs_tokens=4, channel_dim=CHANNEL_DIM, num_players=2, num_tokens=NUM_TOKENS,
    )
    with pytest.raises(ValueError, match="num_players"):
        DMCLearner(agent, two_players)
    matching = DMCReplay(8, max_obs_tokens=4, channel_dim=CHANNEL_DIM, num_tokens=NUM_TOKENS)
    assert DMCLearner(agent, matching).batch_size == 1024


def test_memory_budget_matches_replay_nbytes_for_all_supported_dtypes():
    for channel_dim in (1, 4, CHANNEL_DIM):
        for max_tokens in (1, 4, 48):
            replay = DMCReplay(11, max_obs_tokens=max_tokens, channel_dim=channel_dim)
            budget = dmc_replay_memory_budget(
                11, max_obs_tokens=max_tokens, channel_dim=channel_dim,
            )
            assert replay.nbytes() == budget["total_bytes"]
    default = dmc_replay_memory_budget(500_000, max_obs_tokens=48, channel_dim=256)
    assert default["total_bytes"] == 360_500_000
    with pytest.raises(TypeError):
        dmc_replay_memory_budget(4, max_obs_tokens=4, channel_dim=4, channel_dtype=np.float32)


def test_history_encoding_participates_in_q():
    torch.manual_seed(13)
    agent = DMCAgent(_cfg())
    B, K, T = 2, 2, 5
    channels = torch.zeros(B, CHANNEL_DIM, dtype=torch.uint8)
    action = torch.randn(B, K, ACTION_FEATURE_DIM) * 0.5  # non-zero: state path contributes
    action[action.abs() < 1e-3] = 0.5
    masks = torch.ones(B, K, dtype=torch.bool)
    lengths = torch.tensor([5, 1])
    base = torch.tensor([[3, 4, 5, 6, 7], [8, 9, 9, 9, 9]])
    q0 = agent.q_values(base, channels, action, masks, token_lengths=lengths)

    changed_last = base.clone()
    changed_last[0, 4] = 11  # row 0: last valid token
    changed_last[1, 0] = 12  # row 1: only token (length 1)
    q1 = agent.q_values(changed_last, channels, action, masks, token_lengths=lengths)
    assert (q1[0] - q0[0]).abs().max() > 1e-6, "the last valid token must change Q"
    assert (q1[1] - q0[1]).abs().max() > 1e-6, "with length=1 only token 0 can change Q"

    changed_padding = base.clone()
    changed_padding[1, 1:] = 13  # all padding for the length-1 row
    q2 = agent.q_values(changed_padding, channels, action, masks, token_lengths=lengths)
    assert torch.equal(q2[1], q0[1]), "padding positions must not affect Q"


def test_ntp_weight_requires_targets_and_adds_the_loss():
    torch.manual_seed(9)
    agent = DMCAgent(_cfg(), enable_ntp_head=True)
    replay = DMCReplay(
        32, max_obs_tokens=4, channel_dim=CHANNEL_DIM, num_tokens=NUM_TOKENS, seed=0,
    )
    replay.push_batch([
        _transition(i, obs_channels=np.zeros(CHANNEL_DIM, dtype=np.uint8),
                    action_feats=np.zeros(ACTION_FEATURE_DIM, dtype=np.float32))
        for i in range(32)
    ])
    learner = DMCLearner(agent, replay, batch_size=8, ntp_weight=0.1, seed=0)
    with pytest.raises(ValueError, match="ntp_targets"):
        learner.train_step()
    metrics = learner.train_step(ntp_targets=torch.zeros(8, NUM_TOKENS))
    assert "ntp_loss" in metrics and metrics["nonfinite_count"] == 0
    assert math.isfinite(metrics["ntp_loss"])
