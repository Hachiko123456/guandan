"""Stage 07B contract tests: head+tail observation window and accumulation.

Two stage-07B changes are pinned here:

* **Observation window** — ``trim_obs_window`` keeps ``head 128 + tail 896``
  of the 1024-token stream, so the opening history survives.  The unit test
  asserts the exact composition (head preserved, tail preserved, middle
  dropped) and both split counters; the sampler test drives the real
  ``_append_events`` path and the health counters; the state test proves the
  ``-v2`` schema and the head split are part of the resume identity.
* **Gradient accumulation** — one optimizer step covers
  ``batch_size * accumulation_steps`` rows drawn as a single batch and sliced
  into micro-batches.  The decisive test compares it against a plain single
  batch of the same size on identical rows (same RNG stream): FP32 gradients
  must agree to numerical tolerance, which is what makes 4 x 256 a faithful
  realisation of the locked effective batch 1024.  A per-row inf-gradient
  agent proves that one overflowing micro-batch skips the whole step.

No test edits ``model/``, ``runtime/``, ``training/`` or any IPPO/VRPO path.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from guandan_marvel.backends import get_game_backend
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig
from guandan_marvel.dmc.candidates import ACTION_FEATURE_DIM
from guandan_marvel.dmc.learner import (
    DMCLearner,
    optimizer_steps_for_samples,
)
from guandan_marvel.dmc.q_agent import DMCAgent
from guandan_marvel.dmc.replay import DMCReplay
from guandan_marvel.dmc.sampler import (
    MIN_OBS_WINDOW,
    SAMPLER_STATE_SCHEMA,
    DMCSampler,
    DMCSamplerConfig,
    trim_obs_window,
)
from guandan_marvel.dmc.train import DMCRunConfig, DMCTrainer, load_run_config

GAME = get_game_backend("python_reference")
CHANNEL_DIM = int(GAME.num_channels)  # 512
NUM_TOKENS = int(GAME.num_tokens)

# A small window keeps the sampler tests fast; 8 + 56 = 64 >= head + one row.
WINDOW = 64
HEAD = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_cfg() -> TrainConfig:
    return TrainConfig(
        game=GAME,
        algo_name="dmc",
        device="cpu",
        ppo=PPOConfig(training_semantics_profile="dmc_v1"),
        model=ModelConfig(
            history_dim=16,
            feature_dim=24,
            decoder_layers=1,
            decoder_heads=4,
            state_layers=1,
            action_layers=1,
            fuse_layers=1,
        ),
    )


def _make_sampler(
    *,
    num_envs: int = 1,
    max_obs_tokens: int = WINDOW,
    obs_head_tokens: int = HEAD,
    seed: int = 7,
) -> DMCSampler:
    torch.manual_seed(0)
    agent = DMCAgent(_tiny_cfg())
    replay = DMCReplay(
        4096,
        max_obs_tokens=max_obs_tokens,
        channel_dim=CHANNEL_DIM,
        num_tokens=NUM_TOKENS,
        pad_token=int(GAME.pad_token),
        num_players=4,
        seed=0,
    )
    return DMCSampler(
        q_agent=agent,
        replay=replay,
        config=DMCSamplerConfig(
            num_envs=num_envs,
            device="cpu",
            env_seed=0,
            seed=seed,
            max_obs_tokens=max_obs_tokens,
            obs_head_tokens=obs_head_tokens,
        ),
    )


def _fill_replay(rows: int, *, capacity: int | None = None, marker_rows=()) -> DMCReplay:
    """Replay of ``rows`` distinct samples; ``marker_rows`` get action[0] = 9."""
    replay = DMCReplay(
        capacity or rows,
        max_obs_tokens=4,
        channel_dim=CHANNEL_DIM,
        num_tokens=NUM_TOKENS,
        pad_token=int(GAME.pad_token),
        num_players=4,
        seed=0,
    )
    rng = np.random.default_rng(11)
    action = rng.random((rows, ACTION_FEATURE_DIM)).astype(np.float32)
    marked = np.zeros(rows, dtype=bool)
    marked[list(marker_rows)] = True
    action[marked, 0] = 9.0
    replay.push_batch({
        "obs_tokens": np.ones((rows, 2), dtype=np.int32),
        "obs_token_lengths": np.full(rows, 2, dtype=np.int32),
        "obs_channels": rng.integers(0, 2, size=(rows, CHANNEL_DIM), dtype=np.uint8),
        "action_feats": action,
        "z": rng.standard_normal(rows).astype(np.float32),
        "seat": (np.arange(rows) % 4).astype(np.int8),
        "game_id": np.arange(rows, dtype=np.int64),
    })
    return replay


class _RowInfBackward(torch.autograd.Function):
    """Forward identity, backward multiplies marked rows by +inf."""

    @staticmethod
    def forward(ctx, value, marker):
        ctx.save_for_backward(marker)
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        (marker,) = ctx.saved_tensors
        factor = torch.where(
            marker, torch.full_like(grad, float("inf")), torch.ones_like(grad),
        )
        return grad * factor, None


class _ChunkOverflowAgent(nn.Module):
    """Finite Q; the backward pass is inf for rows whose first action feature is 9."""

    action_feature_dim = ACTION_FEATURE_DIM
    enable_ntp = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def q_values(self, obs_tokens, obs_channels, action_feats, masks, *, token_lengths=None, return_ntp=False):
        base = (self.weight * action_feats.sum(dim=(1, 2))).unsqueeze(1)
        marker = (action_feats[:, 0, 0] > 8.0).reshape(-1, 1)
        return _RowInfBackward.apply(base, marker)


# ---------------------------------------------------------------------------
# Change 1: head+tail window composition
# ---------------------------------------------------------------------------
def test_trim_window_keeps_head_and_tail_and_drops_the_middle():
    max_tokens, head = 1024, 128
    tail = max_tokens - head

    # A stream that fits is untouched (no counters move).
    short = list(range(700))
    assert trim_obs_window(short, max_tokens=max_tokens, head_tokens=head) == (0, 0)
    assert short == list(range(700))

    exact = list(range(max_tokens))
    assert trim_obs_window(exact, max_tokens=max_tokens, head_tokens=head) == (0, 0)
    assert len(exact) == max_tokens

    # One token over: the single dropped token is the first post-head token.
    one_over = list(range(max_tokens + 1))
    assert trim_obs_window(one_over, max_tokens=max_tokens, head_tokens=head) == (0, 1)
    assert len(one_over) == max_tokens
    assert one_over[:head] == list(range(head)), "the head must be pinned"
    assert one_over[head:] == list(range(head + 1, max_tokens + 1))

    # Far over: head 128 + tail 896, middle 976 dropped.
    total = 3000
    long_stream = list(range(total))
    drops = trim_obs_window(long_stream, max_tokens=max_tokens, head_tokens=head)
    assert drops == (0, total - max_tokens)
    assert len(long_stream) == max_tokens
    assert long_stream[:head] == list(range(head))
    assert long_stream[head:] == list(range(total - tail, total))

    # Growing the stream never changes the pinned head (the 07B fix).
    stream: list[int] = []
    for token in range(2000):
        stream.append(token)
        trim_obs_window(stream, max_tokens=max_tokens, head_tokens=head)
        assert len(stream) <= max_tokens
        assert stream[:head] == list(range(min(len(stream), head)))


def test_trim_window_head_zero_reproduces_the_legacy_tail_window():
    max_tokens = 1024
    stream = list(range(3000))
    drops = trim_obs_window(stream, max_tokens=max_tokens, head_tokens=0)
    assert drops == (3000 - max_tokens, 0), "no head => every loss is opening history"
    assert stream == list(range(3000 - max_tokens, 3000))


def test_trim_window_and_config_validation():
    with pytest.raises(ValueError, match="head_tokens"):
        trim_obs_window([1, 2, 3], max_tokens=2, head_tokens=3)
    with pytest.raises(ValueError, match="head_tokens"):
        trim_obs_window([1, 2, 3], max_tokens=2, head_tokens=-1)
    with pytest.raises(ValueError, match="max_tokens"):
        trim_obs_window([1], max_tokens=0, head_tokens=0)

    # The tail must always be able to hold one event row.
    with pytest.raises(ValueError, match="obs_head_tokens"):
        DMCSamplerConfig(num_envs=1, max_obs_tokens=1024,
                         obs_head_tokens=1024 - MIN_OBS_WINDOW + 1)
    DMCSamplerConfig(num_envs=1, max_obs_tokens=1024,
                     obs_head_tokens=1024 - MIN_OBS_WINDOW)


def test_sampler_window_pins_the_head_and_counts_the_middle():
    sampler = _make_sampler()
    sampler.reset()
    slot = sampler._slots[0]
    slot.streams[0] = []
    rows = np.zeros((4, 1), dtype=np.int64)
    for token in range(1000, 1200):
        rows[0, 0] = token
        sampler._append_events(slot, rows)

    stream = slot.streams[0]
    assert len(stream) == WINDOW
    assert stream[:HEAD] == list(range(1000, 1000 + HEAD))
    assert stream[HEAD:] == list(range(1200 - (WINDOW - HEAD), 1200))
    health = sampler.health()
    assert health["obs_head_drops"] == 0, "the opening must never be dropped"
    assert health["obs_tail_drops"] == 200 - WINDOW
    assert health["obs_head_tokens"] == HEAD
    assert health["obs_window_tokens"] == WINDOW
    assert "obs_window_truncations" not in health, "the ambiguous counter is gone"
    # Untouched seats keep their own streams.
    for seat in (1, 2, 3):
        assert len(slot.streams[seat]) <= WINDOW


def test_sampler_window_end_to_end_health_and_sample_width():
    sampler = _make_sampler(num_envs=2)
    sampler.collect_games(3)
    health = sampler.health()
    assert health["illegal"] == 0 and health["nonfinite"] == 0
    assert health["obs_head_drops"] == 0
    assert health["obs_tail_drops"] > 0, "a 64-token window must overflow in real games"
    replay = sampler._replay
    lengths = replay._obs_token_lengths[: len(replay)]
    assert int(lengths.max()) <= WINDOW
    assert int(lengths.min()) >= 1


def test_sampler_state_is_v2_and_binds_the_head_split():
    sampler = _make_sampler(obs_head_tokens=HEAD)
    state = sampler.state_dict()
    assert SAMPLER_STATE_SCHEMA.endswith("-v2")
    assert state["schema"] == SAMPLER_STATE_SCHEMA
    assert state["obs_head_tokens"] == HEAD

    other = _make_sampler(obs_head_tokens=HEAD + 8)
    with pytest.raises(ValueError, match="obs_head_tokens"):
        other.load_state_dict(state)

    legacy = dict(state)
    legacy["schema"] = "guandan-marvel-dmc-sampler-state-v1"
    with pytest.raises(ValueError, match="schema"):
        other.load_state_dict(legacy)


# ---------------------------------------------------------------------------
# Change 2: gradient accumulation
# ---------------------------------------------------------------------------
def test_accumulation_equals_a_single_effective_batch():
    torch.manual_seed(5)
    agent_a = DMCAgent(_tiny_cfg())
    agent_b = copy.deepcopy(agent_a)
    replay = _fill_replay(256)

    accumulated = DMCLearner(
        agent_a, replay, batch_size=256, accumulation_steps=4, seed=1,
    )
    single = DMCLearner(
        agent_b, replay, batch_size=256, accumulation_steps=1, seed=1,
    )
    assert accumulated.batch_size == single.batch_size == 256
    assert accumulated.micro_batch_size == 64
    assert single.micro_batch_size == 256

    metrics_a = accumulated.train_step()
    metrics_b = single.train_step()

    # Both learners drew the same 256 rows from the same RNG state, so the only
    # difference is the micro-batch split.
    assert metrics_a["loss"] == pytest.approx(metrics_b["loss"], rel=1e-6)
    assert metrics_a["qz_r2"] == pytest.approx(metrics_b["qz_r2"], abs=1e-6)
    assert metrics_a["accumulation_steps"] == 4
    assert metrics_a["micro_batch_size"] == 64
    assert metrics_b["effective_batch_size"] == 256

    worst = 0.0
    compared = 0
    for (name, param_a), (_, param_b) in zip(
        agent_a.named_parameters(), agent_b.named_parameters(),
    ):
        assert (param_a.grad is None) == (param_b.grad is None), name
        if param_a.grad is None:
            continue  # parameters this forward never touches (e.g. the NTP head)
        compared += 1
        worst = max(worst, float((param_a.grad - param_b.grad).abs().max()))
        torch.testing.assert_close(
            param_a.grad, param_b.grad, rtol=1e-4, atol=1e-6,
            msg=f"gradient mismatch for {name}",
        )
    assert compared > 0, "no gradient was produced; the comparison would be vacuous"
    assert worst < 1e-5, f"accumulated vs single-batch gradient max diff {worst}"
    # AdamW applied the same update.
    for (name, param_a), (_, param_b) in zip(
        agent_a.named_parameters(), agent_b.named_parameters(),
    ):
        torch.testing.assert_close(
            param_a, param_b, rtol=1e-4, atol=1e-6,
            msg=f"parameter mismatch for {name}",
        )


def test_one_overflowing_micro_batch_skips_the_whole_step():
    torch.manual_seed(0)
    agent = _ChunkOverflowAgent()
    before = {name: value.detach().clone() for name, value in agent.state_dict().items()}
    # Only rows 64..127 (micro-batch 1 of 4) overflow.
    replay = _fill_replay(256, marker_rows=range(64, 128))
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    learner = DMCLearner(
        agent, replay, batch_size=64, accumulation_steps=4, seed=0, grad_scaler=scaler,
    )
    metrics = learner.train_step()
    assert metrics["step_skipped"] is True
    assert metrics["scaler_scale"] == 512.0
    assert metrics["nonfinite_count"] == 0, "an overflow is not a forward fault"
    for name, value in agent.state_dict().items():
        assert torch.equal(value, before[name]), f"skipped step changed {name}"

    # Positive control: the same shape with no overflowing row steps normally.
    torch.manual_seed(0)
    clean_agent = _ChunkOverflowAgent()
    clean_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    clean = DMCLearner(
        clean_agent, _fill_replay(256), batch_size=64, accumulation_steps=4,
        seed=0, grad_scaler=clean_scaler,
    )
    clean_metrics = clean.train_step()
    assert clean_metrics["step_skipped"] is False
    assert math.isfinite(clean_metrics["grad_norm"])


def test_accumulation_is_deterministic():
    torch.manual_seed(9)
    base = DMCAgent(_tiny_cfg())
    replay = _fill_replay(128)
    first = DMCLearner(copy.deepcopy(base), replay, batch_size=32,
                       accumulation_steps=4, seed=3)
    second = DMCLearner(copy.deepcopy(base), replay, batch_size=32,
                        accumulation_steps=4, seed=3)
    metrics_first = first.train_step()
    metrics_second = second.train_step()
    for key in ("loss", "qz_r2", "grad_norm", "q_mean", "q_std"):
        assert metrics_first[key] == metrics_second[key], key
    for (name, param_first), (_, param_second) in zip(
        first.q_agent.named_parameters(), second.q_agent.named_parameters(),
    ):
        assert torch.equal(param_first, param_second), f"non-deterministic update: {name}"


def test_optimizer_step_arithmetic_uses_the_effective_batch():
    assert optimizer_steps_for_samples(25_600, ratio=1.0, effective_batch_size=1024) == 25
    assert optimizer_steps_for_samples(30_000, ratio=0.5, effective_batch_size=1024) == 15
    assert optimizer_steps_for_samples(20_000, ratio=1.0, effective_batch_size=1024) == 20
    # The divisor is the optimizer-step batch, never a micro-batch: dividing by
    # 256 here would silently multiply the gradient/data ratio by 4.
    assert optimizer_steps_for_samples(
        25_600, ratio=1.0, effective_batch_size=256,
    ) == 100
    with pytest.raises(ValueError):
        optimizer_steps_for_samples(10, ratio=1.0, effective_batch_size=0)


# ---------------------------------------------------------------------------
# Config plumbing (resolved config carries both 07B numbers)
# ---------------------------------------------------------------------------
def _run_config(**overrides) -> DMCRunConfig:
    payload = {
        "schema": "guandan-marvel-dmc-train-config-v1",
        "run_name": "dmc_stage07b_test",
        "device": "cpu",
        "seed": 0,
        "env_seed": 0,
        "num_envs": 1,
        "total_games": 4,
        "games_per_cycle": 2,
        "gradient_replay_ratio": 1.0,
        "batch_size": 256,
        "accumulation_steps": 4,
        "obs_head_tokens": 128,
        "buffer_capacity": 4096,
        "max_obs_tokens": 1024,
        "heartbeat_seconds": 3600.0,
        "model": {
            "history_dim": 16, "feature_dim": 24, "decoder_layers": 1,
            "decoder_heads": 4, "state_layers": 1, "action_layers": 1, "fuse_layers": 1,
        },
    }
    payload.update(overrides)
    return load_run_config(payload)


def test_run_config_carries_effective_batch_and_window_split():
    config = _run_config()
    assert config.accumulation_steps == 4
    assert config.obs_head_tokens == 128
    trainer = DMCTrainer(config)
    assert trainer.learner.batch_size == 256
    assert trainer.learner.accumulation_steps == 4
    assert trainer.learner.effective_batch_size == 256
    assert trainer.learner.micro_batch_size == 64
    health = trainer.sampler.health()
    assert health["obs_head_tokens"] == 128
    assert health["obs_window_tokens"] == 1024
    resolved = trainer.train_config.to_dict()
    assert resolved["dmc"]["batch_size"] == 256
    assert resolved["dmc"]["accumulation_steps"] == 4
    assert resolved["dmc"]["obs_head_tokens"] == 128


def test_gate_l_config_keeps_the_locked_effective_batch():
    payload = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "dmc_gate_l.json")
        .read_text(encoding="utf-8")
    )
    config = load_run_config(payload)
    assert config.batch_size == 1024, "the locked optimizer-step batch"
    assert config.accumulation_steps == 4
    assert config.obs_head_tokens == 128
    assert config.max_obs_tokens == 1024


def test_run_config_rejects_bad_07b_values():
    with pytest.raises(ValueError, match="accumulation_steps"):
        _run_config(accumulation_steps=0)
    with pytest.raises(ValueError, match="obs_head_tokens"):
        _run_config(obs_head_tokens=1024)
    with pytest.raises(ValueError, match="buffer_capacity"):
        _run_config(buffer_capacity=128)
    with pytest.raises(ValueError, match="accumulation_steps"):
        _run_config(accumulation_steps=512)
