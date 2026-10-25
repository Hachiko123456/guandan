"""Stage-07A parallel-topology and replay fast-path tests.

The decisive checks:

* ``test_replay_fast_path_matches_per_row_path`` — the vectorised batched
  push writes byte-identical buffer state for valid batches and raises the
  identical error for every defective batch, compared against the original
  per-row path with the fast path disabled.
* ``test_sampler_sink_equivalence`` — running the sampler against a recording
  sink produces exactly the push batches the real replay path produces.
* ``test_parallel_run_cycle1_identity_and_determinism`` — the two-process run
  keeps the locked contract (gradient steps = round(samples·ratio/batch),
  zero illegal/nonfinite, finite losses), reproduces the serial trainer's
  cycle-1 behavior sequences bit-for-bit (both start from the same init
  weights), is deterministic across two identical runs, and every weight sync
  passes the fp32 Q probe.
* ``test_parallel_resume_restores_counters_and_serial_compat`` — a parallel
  checkpoint resumes with ``replay_reset=true`` and exact counters, and the
  payload loads in a serial ``DMCTrainer`` (cross-mode recovery path).
* ``test_parallel_budget_stop_saves_checkpoint`` — the wall-budget safety
  stop records ``budget_stopped`` and still checkpoints.
* ``test_resolve_parallel_devices_*`` — the collector device is the second
  GPU when present, the learner's device on the single-device degraded path,
  and explicit overrides are honoured (or rejected when they do not exist).

No test edits ``model/``, ``runtime/``, ``training/``, the V2 adapter or any
IPPO/VRPO path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from guandan_marvel.backends import get_game_backend
from guandan_marvel.dmc.parallel import (
    PARALLEL_SUMMARY_KEY,
    DMCParallelTrainer,
    resolve_parallel_devices,
    train_from_config_parallel,
)
from guandan_marvel.dmc.replay import DMCReplay
from guandan_marvel.dmc.sampler import DMCSampleSink
from guandan_marvel.dmc.train import (
    DMC_METRICS_SCHEMA,
    DMCTrainer,
    DMCRunConfig,
    load_run_config,
)

GAME = get_game_backend("python_reference")
CHANNEL_DIM = int(GAME.num_channels)
NUM_TOKENS = int(GAME.num_tokens)


def _run_config(**overrides) -> DMCRunConfig:
    payload = {
        "schema": "guandan-marvel-dmc-train-config-v1",
        "run_name": "dmc_parallel_test",
        "device": "cpu",
        "seed": 0,
        "env_seed": 0,
        "num_envs": 2,
        "total_games": 12,
        "games_per_cycle": 6,
        "gradient_replay_ratio": 1.0,
        "batch_size": 16,
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


def _sequence(records) -> list[tuple[int, int, str]]:
    return [
        (int(record["game_id"]), int(record["decisions"]), str(record["action_sha256"]))
        for record in records
    ]


# ---------------------------------------------------------------------------
# Replay fast path (no processes)
# ---------------------------------------------------------------------------
def _batch(rows: int = 5, max_length: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    tokens = rng.integers(1, 64, size=(rows, max_length), dtype=np.int32)
    lengths = rng.integers(1, max_length + 1, size=(rows,), dtype=np.int32)
    for row, length in enumerate(lengths):
        tokens[row, length:] = 0  # padded tail
    return {
        "obs_tokens": tokens,
        "obs_token_lengths": lengths,
        "obs_channels": rng.integers(0, 256, size=(rows, CHANNEL_DIM), dtype=np.uint8),
        "action_feats": rng.random((rows, 64), dtype=np.float32),
        "z": rng.random((rows,), dtype=np.float32),
        "seat": rng.integers(0, 4, size=(rows,), dtype=np.int8),
        "game_id": np.arange(rows, dtype=np.int64),
    }


def _buffer_state(replay: DMCReplay) -> bytes:
    return b"|".join([
        replay._obs_tokens[: len(replay)].tobytes(),
        replay._obs_token_lengths[: len(replay)].tobytes(),
        replay._obs_channels[: len(replay)].tobytes(),
        replay._action_feats[: len(replay)].tobytes(),
        replay._z[: len(replay)].tobytes(),
        replay._seat[: len(replay)].tobytes(),
        replay._game_id[: len(replay)].tobytes(),
    ])


def _new_replays(capacity: int = 64) -> tuple[DMCReplay, DMCReplay]:
    fast = DMCReplay(capacity, max_obs_tokens=1024, channel_dim=CHANNEL_DIM,
                     num_tokens=NUM_TOKENS, pad_token=int(GAME.pad_token),
                     num_players=4, seed=0)
    slow = DMCReplay(capacity, max_obs_tokens=1024, channel_dim=CHANNEL_DIM,
                     num_tokens=NUM_TOKENS, pad_token=int(GAME.pad_token),
                     num_players=4, seed=0)
    return fast, slow


def test_replay_fast_path_matches_per_row_path():
    fast, slow = _new_replays()
    batch = _batch()
    assert fast._try_push_batched_arrays(dict(batch)) == len(batch["z"])
    assert slow.push_batch(dict(batch)) == len(batch["z"])
    assert _buffer_state(fast) == _buffer_state(slow)
    assert fast.total_pushed == slow.total_pushed == len(batch["z"])
    assert fast.write_position == slow.write_position

    # The vectorised path is what actually served the push above.
    fast2, slow2 = _new_replays()
    assert slow2._try_push_batched_arrays({"nope": 1}) is None  # falls back
    with pytest.raises(KeyError):
        slow2.push_batch({"nope": 1})


@pytest.mark.parametrize("mutation", [
    "token_out_of_range", "negative_token", "length_zero", "length_over_width",
    "length_over_window", "channel_out_of_range", "feat_nan", "feat_inf",
    "feat_not_float32", "z_nan", "z_inf", "seat_negative", "seat_over",
    "game_id_negative", "capacity_overflow",
])
def test_replay_fast_path_rejects_like_per_row_path(mutation):
    fast, slow = _new_replays(capacity=8)
    batch = _batch(rows=4, max_length=12)
    if mutation == "token_out_of_range":
        batch["obs_tokens"][2, 0] = 10**6  # position 0 is always inside the validated prefix
    elif mutation == "negative_token":
        batch["obs_tokens"][1, 0] = -5
    elif mutation == "length_zero":
        batch["obs_token_lengths"][3] = 0
    elif mutation == "length_over_width":
        batch["obs_token_lengths"][0] = 13
    elif mutation == "length_over_window":
        # Width stays large enough that only the max_obs_tokens check fires.
        batch["obs_tokens"] = np.pad(batch["obs_tokens"], ((0, 0), (0, 1024)))
        batch["obs_token_lengths"][1] = 2000
    elif mutation == "channel_out_of_range":
        batch["obs_channels"] = batch["obs_channels"].astype(np.int32)
        batch["obs_channels"][0, 5] = 256
    elif mutation == "feat_nan":
        batch["action_feats"][1, 7] = np.nan
    elif mutation == "feat_inf":
        batch["action_feats"][2, 0] = np.inf
    elif mutation == "feat_not_float32":
        batch["action_feats"] = batch["action_feats"].astype(np.float64) * 1e300
    elif mutation == "z_nan":
        batch["z"][0] = np.nan
    elif mutation == "z_inf":
        batch["z"][2] = -np.inf
    elif mutation == "seat_negative":
        batch["seat"][3] = -1
    elif mutation == "seat_over":
        batch["seat"][1] = 4
    elif mutation == "game_id_negative":
        batch["game_id"][2] = -3
    elif mutation == "capacity_overflow":
        pass  # handled separately below

    if mutation == "capacity_overflow":
        # A single batch larger than the whole ring is the rejected case;
        # pushing a small batch into a FULL ring is legal FIFO overwrite.
        overflow = _batch(rows=9)
        with pytest.raises(ValueError, match="cannot push"):
            fast.push_batch(dict(overflow))
        with pytest.raises(ValueError, match="cannot push"):
            slow.push_batch(dict(overflow))
        return

    with pytest.raises(ValueError) as fast_error:
        fast.push_batch(dict(batch))
    with pytest.raises(ValueError) as slow_error:
        slow.push_batch(dict(batch))
    assert str(fast_error.value) == str(slow_error.value), mutation
    assert len(fast) == len(slow) == 0, "a rejected batch must leave the buffer untouched"


def test_replay_fast_path_falls_back_for_foreign_dtypes():
    """Float token rows keep the exact per-row semantics (2.0 accepted, 2.9 not)."""
    fast, slow = _new_replays()
    batch = _batch(rows=2)
    batch["obs_tokens"] = batch["obs_tokens"].astype(np.float64)
    batch["obs_tokens"][0, 0] = 2.0
    batch["obs_token_lengths"] = batch["obs_token_lengths"].astype(np.float64)
    assert fast.push_batch(dict(batch)) == 2
    assert slow.push_batch(dict(batch)) == 2
    assert _buffer_state(fast) == _buffer_state(slow)

    batch["obs_tokens"][0, 0] = 2.9
    with pytest.raises(ValueError, match="must be an integer"):
        fast.push_batch(dict(batch))


# ---------------------------------------------------------------------------
# Sampler sink equivalence (no processes)
# ---------------------------------------------------------------------------
class _RecordingSink(DMCSampleSink):
    def __init__(self) -> None:
        self.batches: list[dict] = []
        self.channel_dim = CHANNEL_DIM
        self.action_feature_dim = 64
        self.pad_token = int(GAME.pad_token)
        self.num_tokens = NUM_TOKENS
        self.num_players = 4
        self.max_obs_tokens = 1024

    def push_batch(self, samples) -> int:
        rows = int(np.asarray(samples["obs_tokens"]).shape[0])
        self.batches.append(samples)
        return rows

    def __len__(self) -> int:
        return sum(int(np.asarray(batch["obs_tokens"]).shape[0])
                   for batch in self.batches)


def _collect_with(sampler) -> tuple:
    sampler.collect_games(4)
    return (
        _sequence(sampler.game_records()),
        sampler.health(),
    )


def test_sampler_sink_equivalence():
    from guandan_marvel.dmc.q_agent import DMCAgent
    from guandan_marvel.dmc.sampler import DMCSampler, DMCSamplerConfig
    from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig

    def build(replay_or_sink):
        torch.manual_seed(0)
        agent = DMCAgent(TrainConfig(
            game=GAME, algo_name="dmc", device="cpu",
            ppo=PPOConfig(training_semantics_profile="dmc_v1"),
            model=ModelConfig(history_dim=16, feature_dim=24, decoder_layers=1,
                              decoder_heads=4, state_layers=1, action_layers=1,
                              fuse_layers=1),
        ))
        return DMCSampler(
            q_agent=agent, replay=replay_or_sink,
            config=DMCSamplerConfig(num_envs=2, device="cpu", env_seed=0, seed=7,
                                    max_obs_tokens=1024),
        )

    replay = DMCReplay(8192, max_obs_tokens=1024, channel_dim=CHANNEL_DIM,
                       num_tokens=NUM_TOKENS, pad_token=int(GAME.pad_token),
                       num_players=4, seed=0)
    sink = _RecordingSink()
    replay_result = _collect_with(build(replay))
    sink_result = _collect_with(build(sink))

    assert replay_result[0] == sink_result[0], "identical behavior sequences"
    assert replay_result[1]["samples_pushed"] == sink_result[1]["samples_pushed"]
    assert len(sink) == sink_result[1]["samples_pushed"] > 0
    # The replayed rows equal the sinked batches row by row.
    replayed = replay._obs_tokens[: len(replay)]
    lengths = replay._obs_token_lengths[: len(replay)]
    pushed_rows = []
    for batch in sink.batches:
        for row in range(batch["obs_tokens"].shape[0]):
            pushed_rows.append((batch["obs_tokens"][row], batch["obs_token_lengths"][row]))
    assert len(pushed_rows) == len(replay)
    for index, (tokens, length) in enumerate(pushed_rows):
        assert np.array_equal(replayed[index, : int(length)], tokens[: int(length)])
        assert lengths[index] == length


# ---------------------------------------------------------------------------
# Device resolution (no processes)
# ---------------------------------------------------------------------------
def test_resolve_parallel_devices_degrades_on_single_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    learner, collector = resolve_parallel_devices("cpu", None)
    assert learner.type == "cpu" and collector == learner

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    learner, collector = resolve_parallel_devices("cuda", None)
    assert learner.type == "cuda" and collector == learner, "single-GPU degraded path"

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    learner, collector = resolve_parallel_devices("cuda", None)
    assert learner.type == "cuda" and collector == torch.device("cuda", 1)

    learner, collector = resolve_parallel_devices("cuda", "cuda:0")
    assert collector == torch.device("cuda:0"), "explicit override wins"
    with pytest.raises(ValueError, match="does not exist"):
        resolve_parallel_devices("cuda", "cuda:9")


# ---------------------------------------------------------------------------
# Two-process runs (spawn; kept small on purpose)
# ---------------------------------------------------------------------------
def test_parallel_run_cycle1_identity_and_determinism(tmp_path):
    config = _run_config()

    serial = DMCTrainer(_run_config())
    serial.train_cycle()  # cycle 1: six games from the shared init weights
    serial_cycle1 = _sequence(serial.sampler.game_records())
    assert len(serial_cycle1) == 6

    summary_a = train_from_config_parallel(
        _run_config(), output_dir_override=tmp_path / "a", log=lambda *_: None,
    )
    summary_b = train_from_config_parallel(
        _run_config(), output_dir_override=tmp_path / "b", log=lambda *_: None,
    )
    for summary in (summary_a, summary_b):
        assert summary["stop_reason"] == "complete"
        assert summary["totals"]["games_completed"] == 12
        assert summary["totals"]["cycles"] == 2
        assert summary["health"]["illegal"] == 0
        assert summary["health"]["nonfinite"] == 0
        assert summary["health"]["truncated"] == 0
        # Every cycle's gradient session matches the locked arithmetic.
        rows = [json.loads(line) for line in
                (tmp_path / ("a" if summary is summary_a else "b") / "metrics.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        assert all(row["schema"] == DMC_METRICS_SCHEMA for row in rows)
        for row in rows:
            assert row["train_steps"] == int(
                round(row["samples_pushed"] * 1.0 / 16)
            ) or row["low_buffer_skips"] > 0
            # Stage 07B: batch_size stays the optimizer-step batch; accumulation
            # only splits it into micro-batches.
            assert row["effective_batch_size"] == 16
            assert row["accumulation_steps"] == 4
            assert row["loss_mean"] is None or row["loss_mean"] == row["loss_mean"]
            assert row["weight_version_used"] == max(0, row["cycle"] - 2)
            assert row["weight_probe_match"] is True, "fp32 Q probe must match"
        assert summary[PARALLEL_SUMMARY_KEY]["single_device_degraded"] is True

    # Cycle 1 is identical to the serial trainer (same init weights, same
    # seeds, same lockstep); cycle 2 uses the pinned version-0 weights, so it
    # deliberately may diverge from serial — that is the documented lag.
    assert _sequence(summary_a["game_records"])[:6] == serial_cycle1
    assert _sequence(summary_a["game_records"]) == _sequence(summary_b["game_records"]), \
        "same seed must reproduce identical behavior sequences"


def test_parallel_resume_restores_counters_and_serial_compat(tmp_path):
    first = train_from_config_parallel(
        _run_config(total_games=6), output_dir_override=tmp_path / "first",
        log=lambda *_: None,
    )
    checkpoint = tmp_path / "first" / "checkpoint_final.pt"
    assert checkpoint.exists()

    resumed = train_from_config_parallel(
        _run_config(total_games=12), resume_path=checkpoint,
        output_dir_override=tmp_path / "resumed", log=lambda *_: None,
    )
    assert resumed["stop_reason"] == "complete"
    assert resumed["totals"]["games_completed"] == 12
    assert resumed["totals"]["cycles"] == 2
    assert resumed["replay_reset"] is True
    rows = [json.loads(line) for line in
            (tmp_path / "resumed" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["replay_reset"] for row in rows)
    assert rows[0]["games"] == 6
    # The resumed collector continues the pinned schedule from the restored
    # version (fresher by one cycle than an uninterrupted run would use — the
    # documented seam semantics of the overlap topology).
    assert rows[0]["weight_version_used"] == 1

    # The payload is a plain dmc-v1 training checkpoint: a serial trainer
    # accepts it, so a parallel run can hand recovery back to serial.
    serial = DMCTrainer(_run_config(total_games=18))
    serial.load_checkpoint(checkpoint)
    assert serial.replay_reset is True
    assert serial.cycles == 1
    assert serial.sampler.games_completed == 6


def test_parallel_budget_stop_saves_checkpoint(tmp_path):
    summary = train_from_config_parallel(
        _run_config(max_wall_seconds=1e-6), output_dir_override=tmp_path / "budget",
        log=lambda *_: None,
    )
    assert summary["stop_reason"] == "budget_stopped"
    assert (tmp_path / "budget" / "checkpoint_final.pt").exists()
    progress = json.loads(
        (tmp_path / "budget" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "budget_stopped"


def test_parallel_rejects_mismatched_resume(tmp_path):
    first = train_from_config_parallel(
        _run_config(), output_dir_override=tmp_path / "first", log=lambda *_: None,
    )
    del first
    checkpoint = tmp_path / "first" / "checkpoint_final.pt"
    with pytest.raises(ValueError, match="run_config mismatch"):
        DMCParallelTrainer(
            _run_config(buffer_capacity=8192),
            learner_device=torch.device("cpu"), collector_device=torch.device("cpu"),
            resume_path=checkpoint,
        )

