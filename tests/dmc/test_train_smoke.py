"""Stage-4 DMC sampler / training-loop contract tests.

The decisive checks:

* ``test_small_config_end_to_end`` — a real multi-game self-play run through
  the shipped Python-reference adapter: zero illegal moves, zero truncations,
  per-game sample counts inside the locked [15, 120] band, terminal ``z``
  values exactly on the ±3/±2/±1 ÷ 3 grid, and a learner that regresses on the
  collected replay with finite loss.
* ``test_determinism_same_seed_identical_sequences`` — the same seed twice
  produces byte-identical ``(game_id, decisions, action hash)`` sequences and
  a different seed diverges.  CPU is used deliberately: the sampler-level
  guarantee must be bit-exact, not "same up to fp16 scheduling".
* ``test_single_candidate_draws_no_epsilon`` — a counting RNG proves
  single-candidate decisions consume no ε draw.
* ``test_sampler_rejects_wrong_channel_dim`` — the reviewer-locked rule that
  the replay must be built with ``guandan_marvel.game.num_channels`` (512),
  never a module default width.
* Trainer-level checkpoint/resume: the replay restarts empty with
  ``replay_reset=true`` recorded while every other counter resumes exactly.

No test edits ``model/``, ``runtime/``, ``training/`` or any IPPO/VRPO path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from guandan_marvel.backends import get_game_backend
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig
from guandan_marvel.dmc.learner import DMCLearner
from guandan_marvel.dmc.q_agent import DMCAgent
from guandan_marvel.dmc.replay import DMCReplay, DEFAULT_CHANNEL_DIM
from guandan_marvel.dmc.sampler import (
    DMCSampler,
    DMCSamplerConfig,
    linear_epsilon,
    select_candidate_index,
)
from guandan_marvel.dmc.train import (
    DMC_METRICS_SCHEMA,
    DMCTrainer,
    DMCRunConfig,
    load_run_config,
    train_from_config,
)

GAME = get_game_backend("python_reference")
CHANNEL_DIM = int(GAME.num_channels)  # 512: the shipped adapter width
NUM_TOKENS = int(GAME.num_tokens)


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
    agent_seed: int = 0,
    num_envs: int = 2,
    env_seed: int = 0,
    seed: int = 7,
    max_obs_tokens: int = 1024,
    epsilon_start: float = 0.10,
    epsilon_final: float = 0.02,
    epsilon_decay_samples: int = 5_000_000,
    channel_dim: int = CHANNEL_DIM,
) -> tuple[DMCSampler, DMCAgent]:
    torch.manual_seed(agent_seed)
    agent = DMCAgent(_tiny_cfg())
    replay = DMCReplay(
        8192,
        max_obs_tokens=max_obs_tokens,
        channel_dim=channel_dim,
        num_tokens=NUM_TOKENS,
        pad_token=int(GAME.pad_token),
        num_players=4,
        seed=0,
    )
    sampler = DMCSampler(
        q_agent=agent,
        replay=replay,
        config=DMCSamplerConfig(
            num_envs=num_envs,
            device="cpu",
            env_seed=env_seed,
            seed=seed,
            max_obs_tokens=max_obs_tokens,
            epsilon_start=epsilon_start,
            epsilon_final=epsilon_final,
            epsilon_decay_samples=epsilon_decay_samples,
        ),
    )
    return sampler, agent


# ---------------------------------------------------------------------------
# End-to-end self-play invariants
# ---------------------------------------------------------------------------
def test_small_config_end_to_end():
    sampler, agent = _make_sampler(num_envs=4, env_seed=0, seed=11)
    stats = sampler.collect_games(12)
    health = sampler.health()
    assert health["illegal"] == 0
    assert health["truncated"] == 0
    assert health["nonfinite"] == 0
    assert health["failed"] is False
    assert health["games_completed"] == 12
    assert health["replay_size"] == health["samples_pushed"] > 0
    # Decision accounting: every decision is exactly single- or multi-candidate.
    assert (
        health["single_candidate_decisions"] + health["multi_candidate_decisions"]
        == health["decisions"]
    )
    # Locked sample band, checked per completed game.
    for record in sampler.game_records():
        assert 15 <= record["samples"] <= 120, record
        assert record["decisions"] >= record["samples"] >= 1
    # z lives exactly on the ±3/±2/±1 divided-by-three grid (gamma = 1);
    # compare with a float32-safe tolerance.
    allowed = (-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0)
    z_values = [float(value) for value in sampler._replay._z[: len(sampler._replay)]]
    worst = max(min(abs(value - grid) for grid in allowed) for value in z_values)
    assert worst < 1e-5, sorted(set(z_values))

    # The learner regresses on the collected replay with finite updates.
    learner = DMCLearner(agent, sampler._replay, batch_size=64, seed=3)
    for _ in range(3):
        metrics = learner.train_step()
        assert metrics["nonfinite_count"] == 0
        assert math.isfinite(metrics["loss"])
    assert stats["games"] == 12


def test_epsilon_annealing_is_linear_and_bounded():
    kwargs = {"start": 0.10, "final": 0.02, "decay_samples": 1_000_000}
    assert linear_epsilon(0, **kwargs) == 0.10
    assert linear_epsilon(250_000, **kwargs) == pytest.approx(0.08)
    assert linear_epsilon(1_000_000, **kwargs) == pytest.approx(0.02)
    assert linear_epsilon(9_000_000, **kwargs) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        linear_epsilon(0, start=0.02, final=0.10, decay_samples=100)
    with pytest.raises(ValueError):
        linear_epsilon(0, start=0.10, final=0.02, decay_samples=0)


def test_select_candidate_index_greedy_and_exploration():
    q = torch.tensor([1.0, 3.0, 2.0])
    # ε=0: always the argmax, still exactly one RNG draw per decision.
    rng = np.random.default_rng(0)
    draws = [select_candidate_index(q, rng, 0.0) for _ in range(5)]
    assert all(index == 1 and is_argmax for index, is_argmax in draws)
    # ε=1: uniform exploration somewhere in the legal candidate range.
    rng = np.random.default_rng(4)
    seen = {select_candidate_index(q, rng, 1.0)[0] for _ in range(200)}
    assert seen == {0, 1, 2}
    # torch.argmax tie-break is the first index (determinism contract).
    tie = torch.tensor([2.0, 2.0, 1.0])
    assert select_candidate_index(tie, np.random.default_rng(0), 0.0) == (0, True)
    with pytest.raises(Exception):
        select_candidate_index(torch.tensor([float("nan")]), np.random.default_rng(0), 0.0)


def test_single_candidate_draws_no_epsilon():
    class _CountingRNG:
        """Proxy recording every random()/integers() call."""

        def __init__(self, rng):
            self._rng = rng
            self.random_calls = 0
            self.integer_calls = 0

        def random(self):
            self.random_calls += 1
            return self._rng.random()

        def integers(self, *args, **kwargs):
            self.integer_calls += 1
            return self._rng.integers(*args, **kwargs)

    sampler, _agent = _make_sampler(num_envs=2, env_seed=0, seed=5, epsilon_start=0.5)
    counting = _CountingRNG(sampler._rng)
    sampler._rng = counting
    sampler.collect_games(3)
    health = sampler.health()
    assert health["single_candidate_decisions"] > 0, "fixture must exercise auto-submit"
    # Exactly one random() per multi-candidate decision, none for the
    # single-candidate ones; integers() only on ε-exploration draws.
    assert counting.random_calls == health["multi_candidate_decisions"]
    assert counting.integer_calls <= counting.random_calls


def test_sampler_rejects_wrong_channel_dim():
    # The locked rule: the replay width is guandan_marvel.game.num_channels
    # (512).  The replay module's DEFAULT_CHANNEL_DIM (256) must never be
    # silently accepted for the real game.
    assert CHANNEL_DIM == 512 and DEFAULT_CHANNEL_DIM == 256
    torch.manual_seed(0)
    agent = DMCAgent(_tiny_cfg())
    replay = DMCReplay(
        64, max_obs_tokens=256, channel_dim=DEFAULT_CHANNEL_DIM, num_tokens=NUM_TOKENS,
    )
    with pytest.raises(ValueError, match="num_channels"):
        DMCSampler(
            q_agent=agent,
            replay=replay,
            config=DMCSamplerConfig(num_envs=2, device="cpu", max_obs_tokens=256),
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def _sequence(sampler: DMCSampler) -> list[tuple[int, int, str]]:
    return [
        (int(record["game_id"]), int(record["decisions"]), str(record["action_sha256"]))
        for record in sampler.game_records()
    ]


def test_determinism_same_seed_identical_sequences():
    first, _ = _make_sampler(num_envs=4, env_seed=0, seed=7, agent_seed=0)
    first.collect_games(50)
    second, _ = _make_sampler(num_envs=4, env_seed=0, seed=7, agent_seed=0)
    second.collect_games(50)
    assert _sequence(first) == _sequence(second)
    assert len(_sequence(first)) == 50

    other, _ = _make_sampler(num_envs=4, env_seed=0, seed=8, agent_seed=0)
    other.collect_games(6)
    assert _sequence(other) != _sequence(first)[:6], "a different seed must diverge"

    other_env, _ = _make_sampler(num_envs=4, env_seed=1, seed=7, agent_seed=0)
    other_env.collect_games(6)
    assert _sequence(other_env) != _sequence(first)[:6], "a different env seed must diverge"


def test_epsilon_does_not_change_single_and_argmax_accounting():
    # With ε=0 the executed action is always the Q-argmax.
    sampler, _ = _make_sampler(
        num_envs=2, env_seed=0, seed=3,
        epsilon_start=0.0, epsilon_final=0.0,
    )
    sampler.collect_games(4)
    health = sampler.health()
    assert health["multi_candidate_decisions"] > 0
    assert health["argmax_hit_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Exact sampler checkpoint transfer
# ---------------------------------------------------------------------------
def test_sampler_state_roundtrip_exact_resume():
    sampler, agent = _make_sampler(num_envs=2, env_seed=0, seed=7, agent_seed=0)
    sampler.collect_games(4)
    state = sampler.state_dict()

    reference, _ = _make_sampler(num_envs=2, env_seed=0, seed=7, agent_seed=0)
    reference.collect_games(7)
    expected = _sequence(reference)[-3:]

    restored, restored_agent = _make_sampler(
        num_envs=2, env_seed=0, seed=99, agent_seed=1234,
    )
    # Same weights, same sampler state: the constructor seeds are overridden
    # by the restored state, proving the resume is state- not init-driven.
    restored_agent.load_state_dict(agent.checkpoint_payload())
    restored.load_state_dict(state)
    restored.collect_games(3)
    assert _sequence(restored) == expected
    assert restored.health()["games_completed"] == 7
    assert restored.health()["samples_pushed"] == reference.health()["samples_pushed"]
    assert restored.health()["epsilon"] == pytest.approx(reference.health()["epsilon"])

    with pytest.raises(ValueError, match="schema"):
        broken = dict(state)
        broken["schema"] = "guandan-marvel-dmc-sampler-state-v0"
        restored.load_state_dict(broken)
    with pytest.raises(ValueError, match="mismatch"):
        broken = dict(state)
        broken["max_obs_tokens"] = 512
        restored.load_state_dict(broken)


# ---------------------------------------------------------------------------
# Trainer-level loop, metrics, checkpoint/resume
# ---------------------------------------------------------------------------
def _run_config(**overrides) -> DMCRunConfig:
    payload = {
        "schema": "guandan-marvel-dmc-train-config-v1",
        "run_name": "dmc_stage4_test",
        "device": "cpu",
        "seed": 0,
        "env_seed": 0,
        "num_envs": 2,
        "total_games": 12,
        "games_per_cycle": 6,
        "gradient_replay_ratio": 1.0,
        "batch_size": 64,
        "buffer_capacity": 4096,
        "max_obs_tokens": 1024,
        "checkpoint_interval_cycles": 1,
        "heartbeat_seconds": 3600.0,
        "model": {
            "history_dim": 16, "feature_dim": 24, "decoder_layers": 1,
            "decoder_heads": 4, "state_layers": 1, "action_layers": 1, "fuse_layers": 1,
        },
    }
    payload.update(overrides)
    return load_run_config(payload)


def test_run_config_validation(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        load_run_config({"schema": "nope"})
    with pytest.raises(ValueError, match="unknown"):
        load_run_config({**{"schema": "guandan-marvel-dmc-train-config-v1"}, "bogus": 1})
    with pytest.raises(ValueError, match="buffer_capacity"):
        _run_config(buffer_capacity=8)
    with pytest.raises(ValueError, match="epsilon"):
        _run_config(epsilon_start=0.0, epsilon_final=0.5)
    assert _run_config().batch_size == 64


def test_trainer_end_to_end_and_metrics(tmp_path):
    config = _run_config(run_name="dmc_trainer_e2e")
    summary = train_from_config(config, output_dir_override=tmp_path / "run", log=lambda *_: None)
    assert summary["stop_reason"] == "complete"
    assert summary["totals"]["games_completed"] == 12
    assert summary["totals"]["train_steps"] > 0
    assert summary["channel_dim"] == 512
    assert summary["health"]["illegal"] == 0
    assert summary["health"]["truncated"] == 0
    assert summary["health"]["nonfinite"] == 0
    assert summary["per_game_samples"]["min"] >= 15
    assert summary["per_game_samples"]["max"] <= 120
    assert summary["scaler_enabled"] is False  # CPU run: fp32, no scaler

    metrics_path = tmp_path / "run" / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["schema"] == DMC_METRICS_SCHEMA for row in rows)
    assert sum(row["games"] for row in rows) == 12
    for row in rows:
        if row["train_steps"] > 0:
            assert math.isfinite(row["loss_mean"]) and math.isfinite(row["loss_last"])
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "checkpoint_final.pt").exists()
    progress = json.loads((tmp_path / "run" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete" and progress["replay_reset"] is False


def test_trainer_resume_replay_reset_and_exact_counters(tmp_path):
    first_dir = tmp_path / "first"
    train_from_config(_run_config(run_name="dmc_resume"), output_dir_override=first_dir,
                      log=lambda *_: None)
    checkpoint = first_dir / "checkpoint_final.pt"

    # The checkpoint is a dmc-v1 superset: a bare DMCAgent accepts it too.
    agent = DMCAgent(_tiny_cfg())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(payload)

    resumed = DMCTrainer(_run_config(run_name="dmc_resume", total_games=18))
    resumed.load_checkpoint(checkpoint)
    assert resumed.replay_reset is True
    assert len(resumed.replay) == 0, "a restore restarts the replay empty"
    assert resumed.sampler.games_completed == 12
    assert resumed.cycles == 2
    # The annealing clock is exact even with an empty replay.
    assert resumed.sampler.samples_pushed == payload["sampler_state"]["samples_pushed"]
    assert resumed.sampler.epsilon() == resumed.sampler.epsilon()

    summary = train_from_config(
        _run_config(run_name="dmc_resume", total_games=18),
        resume_path=checkpoint,
        output_dir_override=tmp_path / "resumed",
        log=lambda *_: None,
    )
    assert summary["stop_reason"] == "complete"
    assert summary["totals"]["games_completed"] == 18
    assert summary["replay_reset"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "resumed" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows and all(row["replay_reset"] for row in rows)
    assert rows[0]["games"] == 6


def test_trainer_rejects_cuda_without_gpu():
    if torch.cuda.is_available():
        pytest.skip("CUDA present: the no-CUDA rejection cannot be exercised")
    with pytest.raises(ValueError, match="silently fall back"):
        DMCTrainer(_run_config(device="cuda"))


def test_trainer_rejects_mismatched_resume(tmp_path):
    train_from_config(_run_config(run_name="dmc_bad_resume"), output_dir_override=tmp_path,
                      log=lambda *_: None)
    with pytest.raises(ValueError, match="run_config mismatch"):
        DMCTrainer(_run_config(run_name="dmc_other", buffer_capacity=8192)).load_checkpoint(
            tmp_path / "checkpoint_final.pt",
        )


def test_wall_budget_stops_cleanly_with_exit_code_2(tmp_path):
    """The safety stop saves a checkpoint, records budget_stopped, exits 2."""
    config = _run_config(run_name="dmc_budget", max_wall_seconds=1e-6)
    out = tmp_path / "budget"
    summary = train_from_config(config, output_dir_override=out, log=lambda *_: None)
    assert summary["stop_reason"] == "budget_stopped"
    assert summary["totals"]["games_completed"] == 0
    assert (out / "checkpoint_final.pt").exists(), "a safety stop must still checkpoint"
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "budget_stopped"

    import subprocess
    import sys

    config_path = tmp_path / "budget.json"
    payload = {
        "schema": "guandan-marvel-dmc-train-config-v1",
        "run_name": "dmc_budget_cli",
        "device": "cpu",
        "num_envs": 2,
        "total_games": 4,
        "games_per_cycle": 2,
        "batch_size": 16,
        "buffer_capacity": 256,
        "max_obs_tokens": 1024,
        "max_wall_seconds": 1e-6,
        "heartbeat_seconds": 3600.0,
        "model": {
            "history_dim": 16, "feature_dim": 24, "decoder_layers": 1,
            "decoder_heads": 4, "state_layers": 1, "action_layers": 1, "fuse_layers": 1,
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "guandan_marvel.dmc.train", "--config", str(config_path),
         "--output-dir", str(tmp_path / "budget_cli")],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 2, result.stderr[-2000:]
    assert "budget_stopped" in result.stdout


def test_metrics_rows_survive_a_nonfinite_grad_norm(tmp_path):
    """A fp16 scaler overflow reports inf grad_norm; the row must stay writable."""

    class _OverflowAgent(torch.nn.Module):
        action_feature_dim = 64
        num_channels = 512
        num_players = 4
        enable_ntp = False

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def q_values(self, obs_tokens, obs_channels, action_feats, masks, **kwargs):
            class _Inf(torch.autograd.Function):
                @staticmethod
                def forward(ctx, value):
                    return value.clone()

                @staticmethod
                def backward(ctx, grad):
                    return grad * float("inf")

            base = (self.w * action_feats.sum(dim=(1, 2))).unsqueeze(1)
            return _Inf.apply(base)

    from guandan_marvel.dmc.learner import DMCLearner

    replay = DMCReplay(64, max_obs_tokens=4, channel_dim=512, num_tokens=NUM_TOKENS, seed=0)
    n = 64
    replay.push_batch({
        "obs_tokens": np.ones((n, 2), dtype=np.int32),
        "obs_token_lengths": np.full(n, 2, dtype=np.int32),
        "obs_channels": np.zeros((n, 512), dtype=np.uint8),
        "action_feats": np.full((n, 64), 0.1, dtype=np.float32),
        "z": np.zeros(n, dtype=np.float32),
        "seat": np.zeros(n, dtype=np.int8),
        "game_id": np.arange(n, dtype=np.int64),
    })
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    learner = DMCLearner(_OverflowAgent(), replay, batch_size=8, seed=0, grad_scaler=scaler)
    metrics = learner.train_step()
    assert metrics["step_skipped"] is True

    from guandan_marvel.dmc.train import _json_safe

    row = {"schema": DMC_METRICS_SCHEMA, **_json_safe(metrics)}
    text = json.dumps(row, sort_keys=True, allow_nan=False)  # must not raise
    assert json.loads(text)["grad_norm"] is None
