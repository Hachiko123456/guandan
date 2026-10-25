# SPDX-License-Identifier: MIT
"""DMC training loop (stage 4).

One cycle is exactly the locked contract::

    collect ``games_per_cycle`` ε-greedy games  ->  push replay (sampler)
    gradient steps = round(new_samples * gradient_replay_ratio / batch_size)
    AdamW(lr=1e-4, weight_decay=0), MSE(Q, z), fp16 autocast + GradScaler on CUDA

Everything else in this module is bookkeeping around that contract:

* ``dmc-v1`` checkpoints (model + optimizer + GradScaler + sampler state + RNG
  states + epsilon/counter progress).  A restore rebuilds the replay **empty**
  and records ``replay_reset=true``; every other quantity resumes exactly.
* ``guandan-marvel-dmc-metrics-v1`` jsonl rows, one per cycle.
* Heartbeat file + log lines on the safe (cycle / collect-iteration)
  boundaries, never touching any RNG.
* ``max_wall_seconds`` is a safety stop only: the budget is checked at cycle
  boundaries, the checkpoint is saved, progress records ``budget_stopped`` and
  the process exits with code 2 — mirroring the existing worker convention.

CLI::

    python -m guandan_marvel.dmc.train --config configs/dmc_smoke.json

The behavioural data contract (ε-greedy, single-candidate auto-submit,
multi-candidate samples only) lives in :mod:`guandan_marvel.dmc.sampler`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from guandan_marvel.backends import get_game_backend
from guandan_marvel.config import (
    ModelConfig,
    PPOConfig,
    TrainConfig,
)
from guandan_marvel.dmc import (
    DMC_ACTION_FEATURE_VERSION,
    DMC_CHECKPOINT_SCHEMA,
    DMC_TRAINING_SEMANTICS_PROFILE,
    DMCRuntimeConfig,
)
from guandan_marvel.dmc.learner import DMCLearner
from guandan_marvel.dmc.q_agent import DMCAgent
from guandan_marvel.dmc.replay import DMCReplay
from guandan_marvel.dmc.sampler import DMCSampler, DMCSamplerConfig

DMC_TRAIN_CONFIG_SCHEMA = "guandan-marvel-dmc-train-config-v1"
DMC_METRICS_SCHEMA = "guandan-marvel-dmc-metrics-v1"
DMC_PROGRESS_SCHEMA = "guandan-marvel-dmc-progress-v1"
DMC_RUN_SUMMARY_SCHEMA = "guandan-marvel-dmc-run-summary-v1"
DMC_HEARTBEAT_SCHEMA = "guandan-marvel-dmc-heartbeat-v1"
#: The full train checkpoint is a *superset* of the stage-3 ``dmc-v1`` policy
#: payload (extra training state next to ``model_state``), so it still loads
#: through :meth:`DMCAgent.load_state_dict`.
DMC_TRAIN_PAYLOAD_KIND = "dmc_train_full_v1"


class DMCConfigError(ValueError):
    """Raised for an invalid or unknown DMC training configuration."""


@dataclass
class DMCRunConfig:
    """Validated, schema-checked view of the JSON training config."""

    run_name: str = "dmc_run"
    output_dir: str | None = None
    device: str = "auto"
    seed: int = 0
    env_seed: int = 0
    num_envs: int = 8
    total_games: int = 200
    games_per_cycle: int = 256
    gradient_replay_ratio: float = 1.0
    batch_size: int = 1024
    buffer_capacity: int = 500_000
    max_obs_tokens: int = 1024
    epsilon_start: float = 0.10
    epsilon_final: float = 0.02
    epsilon_decay_samples: int = 5_000_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    checkpoint_interval_cycles: int = 0
    max_wall_seconds: float | None = None
    heartbeat_seconds: float = 30.0
    context_schedule: Mapping[str, Any] | None = None
    model: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "num_envs", "total_games", "games_per_cycle", "batch_size",
            "buffer_capacity", "max_obs_tokens", "epsilon_decay_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise DMCConfigError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        # 0 disables intermediate checkpoints (the final one is always saved).
        if isinstance(self.checkpoint_interval_cycles, bool) or int(self.checkpoint_interval_cycles) < 0:
            raise DMCConfigError("checkpoint_interval_cycles must be a non-negative integer")
        object.__setattr__(self, "checkpoint_interval_cycles", int(self.checkpoint_interval_cycles))
        if int(self.buffer_capacity) < int(self.batch_size):
            raise DMCConfigError("buffer_capacity must be >= batch_size")
        ratio = float(self.gradient_replay_ratio)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise DMCConfigError("gradient_replay_ratio must be finite and positive")
        for name in ("learning_rate",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise DMCConfigError(f"{name} must be finite and positive")
        for name in ("weight_decay", "grad_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise DMCConfigError(f"{name} must be finite and non-negative")
        if float(self.grad_clip_norm) <= 0.0:
            raise DMCConfigError("grad_clip_norm must be positive")
        if not 0.0 <= float(self.epsilon_final) <= float(self.epsilon_start) <= 1.0:
            raise DMCConfigError("epsilon must satisfy 0 <= epsilon_final <= epsilon_start <= 1")
        device = str(self.device).strip().lower()
        if device not in ("auto", "cuda", "cpu"):
            raise DMCConfigError("device must be one of auto/cuda/cpu")
        self.device = device
        if self.max_wall_seconds is not None:
            seconds = float(self.max_wall_seconds)
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise DMCConfigError("max_wall_seconds must be a finite positive number or None")
            self.max_wall_seconds = seconds
        heartbeat = float(self.heartbeat_seconds)
        if not math.isfinite(heartbeat) or heartbeat <= 0.0:
            raise DMCConfigError("heartbeat_seconds must be finite and positive")
        self.heartbeat_seconds = heartbeat
        if self.context_schedule is not None and not isinstance(self.context_schedule, Mapping):
            raise DMCConfigError("context_schedule must be a mapping or None")
        if not isinstance(self.model, Mapping):
            raise DMCConfigError("model must be a mapping of ModelConfig fields")


_KNOWN_CONFIG_FIELDS = frozenset(field.name for field in dataclasses.fields(DMCRunConfig))


def load_run_config(payload: Mapping[str, Any]) -> DMCRunConfig:
    """Parse and validate the JSON config mapping (unknown keys rejected)."""
    if not isinstance(payload, Mapping):
        raise DMCConfigError("training config must be a JSON mapping")
    schema = payload.get("schema")
    if schema != DMC_TRAIN_CONFIG_SCHEMA:
        raise DMCConfigError(
            f"unsupported training config schema {schema!r}; expected "
            f"{DMC_TRAIN_CONFIG_SCHEMA!r}"
        )
    unknown = sorted(
        str(key) for key in payload
        if key != "schema" and key not in _KNOWN_CONFIG_FIELDS
    )
    if unknown:
        raise DMCConfigError(f"unknown training config keys: {', '.join(unknown)}")
    fields = {
        field.name: payload[field.name]
        for field in dataclasses.fields(DMCRunConfig)
        if field.name in payload
    }
    return DMCRunConfig(**fields)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cuda/cpu explicitly; never silently downgrade CUDA."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise DMCConfigError(
            "device='cuda' was requested but CUDA is unavailable; refusing to "
            "silently fall back to CPU"
        )
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _json_safe(value: Any) -> Any:
    """Map non-finite floats to null so strict JSON rows stay writable.

    A fp16 GradScaler overflow reports ``grad_norm=inf`` for the skipped step
    (the learner's own log does the same); the metrics row must survive it.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DMCTrainer:
    """Owns the agent, replay, learner and sampler for one training run."""

    def __init__(self, run_config: DMCRunConfig, *, device_override: str | None = None) -> None:
        self.run_config = run_config
        requested = (device_override or run_config.device).strip().lower()
        self.device = resolve_device(requested)
        self._game = get_game_backend("python_reference")

        # The replay is built with the game adapter's own dimensions — the
        # shipped Guandan adapter is 512-wide and this is the checkpoint the
        # reviewer locked: never a module default width.
        channel_dim = int(self._game.num_channels)
        self.replay = DMCReplay(
            run_config.buffer_capacity,
            max_obs_tokens=run_config.max_obs_tokens,
            channel_dim=channel_dim,
            num_tokens=int(self._game.num_tokens),
            pad_token=int(self._game.pad_token),
            num_players=int(self._game.num_players),
            seed=run_config.seed,
        )

        torch.manual_seed(run_config.seed)
        self.train_config = TrainConfig(
            game=self._game,
            algo_name="dmc",
            ppo=PPOConfig(training_semantics_profile=DMC_TRAINING_SEMANTICS_PROFILE),
            dmc=DMCRuntimeConfig(
                buffer_capacity=run_config.buffer_capacity,
                batch_size=run_config.batch_size,
                games_per_cycle=run_config.games_per_cycle,
                gradient_replay_ratio=run_config.gradient_replay_ratio,
                epsilon_start=run_config.epsilon_start,
                epsilon_final=run_config.epsilon_final,
                epsilon_decay_samples=run_config.epsilon_decay_samples,
                learning_rate=run_config.learning_rate,
                weight_decay=run_config.weight_decay,
                grad_clip_norm=run_config.grad_clip_norm,
            ),
            model=ModelConfig(**dict(run_config.model)),
            device=str(self.device),
            seed=run_config.seed,
        )
        self.q_agent = DMCAgent(self.train_config)

        self.learner = DMCLearner(
            self.q_agent,
            self.replay,
            device=self.device,
            config=self.train_config.dmc,
            seed=run_config.seed + 1,
        )
        self.sampler = DMCSampler(
            q_agent=self.q_agent,
            replay=self.replay,
            config=DMCSamplerConfig(
                num_envs=run_config.num_envs,
                device=str(self.device),
                env_seed=run_config.env_seed,
                seed=run_config.seed,
                max_obs_tokens=run_config.max_obs_tokens,
                epsilon_start=run_config.epsilon_start,
                epsilon_final=run_config.epsilon_final,
                epsilon_decay_samples=run_config.epsilon_decay_samples,
                context_schedule=(
                    None if run_config.context_schedule is None
                    else dict(run_config.context_schedule)
                ),
            ),
        )
        self.cycles = 0
        self.total_train_steps = 0
        self.low_buffer_skips = 0
        self.step_skipped_total = 0
        self.nonfinite_count = 0
        self.wall_seconds = 0.0
        self.replay_reset = False

    # ------------------------------------------------------------------
    # Checkpoint transfer
    # ------------------------------------------------------------------
    def checkpoint_payload(self) -> dict[str, Any]:
        cuda_states = []
        if self.device.type == "cuda":
            cuda_states = [
                torch.cuda.get_rng_state(index).tolist()
                for index in range(torch.cuda.device_count())
            ]
        return {
            "schema": DMC_CHECKPOINT_SCHEMA,
            "action_feature_version": DMC_ACTION_FEATURE_VERSION,
            "payload_kind": DMC_TRAIN_PAYLOAD_KIND,
            "model_state": self.q_agent.state_dict(),
            "optimizer_state": self.learner.optimizer.state_dict(),
            "scaler_state": self.learner.scaler.state_dict(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": cuda_states,
            "sampler_state": self.sampler.state_dict(),
            "learner_rng_state": self.learner.rng.bit_generator.state,
            "replay_stats": {
                "capacity": self.replay.capacity,
                "max_obs_tokens": self.replay.max_obs_tokens,
                "channel_dim": self.replay.channel_dim,
                "action_feature_dim": self.replay.action_feature_dim,
                "num_tokens": self.replay.num_tokens,
                "pad_token": self.replay.pad_token,
                "num_players": self.replay.num_players,
                "size": len(self.replay),
                "write_position": self.replay.write_position,
                "total_pushed": self.replay.total_pushed,
                "nbytes": self.replay.nbytes(),
            },
            "counters": {
                "cycles": self.cycles,
                "games_completed": self.sampler.games_completed,
                "samples_pushed": self.sampler.samples_pushed,
                "total_train_steps": self.total_train_steps,
                "low_buffer_skips": self.low_buffer_skips,
                "step_skipped_total": self.step_skipped_total,
                "nonfinite_count": self.nonfinite_count,
                "wall_seconds": self.wall_seconds,
                "replay_reset": self.replay_reset,
            },
            "run_config": dataclasses.asdict(self.run_config),
        }

    def save_checkpoint(self, path: Path) -> None:
        _atomic_torch_save(self.checkpoint_payload(), path)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != DMC_CHECKPOINT_SCHEMA:
            raise DMCConfigError(
                f"checkpoint schema {payload.get('schema')!r} is not {DMC_CHECKPOINT_SCHEMA!r}"
            )
        if payload.get("payload_kind") != DMC_TRAIN_PAYLOAD_KIND:
            raise DMCConfigError("checkpoint is not a full DMC training checkpoint")
        saved_config = payload.get("run_config")
        if not isinstance(saved_config, Mapping):
            raise DMCConfigError("checkpoint is missing its run_config")
        current = dataclasses.asdict(self.run_config)
        # output_dir/total_games are operational stop targets, not training
        # semantics: a resume may move the output directory or extend the
        # total.  Everything else must match exactly.
        for operational in ("output_dir", "total_games"):
            saved_config = {k: v for k, v in saved_config.items() if k != operational}
            current = {k: v for k, v in current.items() if k != operational}
        if dict(saved_config) != current:
            raise DMCConfigError(
                "checkpoint run_config mismatch; a DMC resume must replay the exact "
                "training configuration"
            )
        self.q_agent.load_state_dict(payload)  # validates schema + feature version
        self.q_agent.to(self.device)
        self.learner.optimizer.load_state_dict(payload["optimizer_state"])
        self.learner.scaler.load_state_dict(payload["scaler_state"])
        torch.set_rng_state(payload["torch_cpu_rng_state"])
        restored_cuda = list(payload.get("torch_cuda_rng_states", []))
        if self.device.type == "cuda":
            if len(restored_cuda) != torch.cuda.device_count():
                raise DMCConfigError(
                    "checkpoint CUDA RNG state count does not match this machine"
                )
            for index, state in enumerate(restored_cuda):
                torch.cuda.set_rng_state(
                    torch.as_tensor(state, dtype=torch.uint8), torch.device("cuda", index),
                )
        elif restored_cuda:
            raise DMCConfigError("checkpoint carries CUDA RNG state but this run is on CPU")
        self.sampler.load_state_dict(payload["sampler_state"])
        self.learner.rng.bit_generator.state = payload["learner_rng_state"]

        counters = payload["counters"]
        self.cycles = int(counters["cycles"])
        self.total_train_steps = int(counters["total_train_steps"])
        self.low_buffer_skips = int(counters["low_buffer_skips"])
        self.step_skipped_total = int(counters["step_skipped_total"])
        self.nonfinite_count = int(counters["nonfinite_count"])
        self.wall_seconds = float(counters["wall_seconds"])
        # Locked restore semantics: the replay itself restarts empty (its
        # contents are not checkpointed) and the reset is recorded; the
        # epsilon/annealing clock lives in the sampler counters and is exact.
        stats = payload["replay_stats"]
        for key, value in (
            ("capacity", self.replay.capacity),
            ("max_obs_tokens", self.replay.max_obs_tokens),
            ("channel_dim", self.replay.channel_dim),
            ("action_feature_dim", self.replay.action_feature_dim),
            ("num_tokens", self.replay.num_tokens),
            ("pad_token", self.replay.pad_token),
            ("num_players", self.replay.num_players),
        ):
            if stats.get(key) != value:
                raise DMCConfigError(f"checkpoint replay_stats.{key} mismatch")
        self.replay_reset = True

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train_cycle(self, progress_callback=None) -> dict[str, Any]:
        """Collect one cycle, run the locked gradient steps, return metrics."""
        remaining = self.run_config.total_games - self.sampler.games_completed
        cycle_games = min(int(self.run_config.games_per_cycle), remaining)
        stats = self.sampler.collect_games(cycle_games, progress_callback=progress_callback)

        new_samples = int(stats["samples_pushed"])
        desired_steps = int(round(new_samples * self.run_config.gradient_replay_ratio
                                  / self.run_config.batch_size))
        train_steps = 0
        losses: list[float] = []
        qz_r2_last = None
        grad_norms: list[float] = []
        scaler_scale_last = None
        if desired_steps > 0:
            if len(self.replay) < self.run_config.batch_size:
                # The learner cannot sample without replacement below one
                # batch; the skipped updates are counted, never silent.
                self.low_buffer_skips += desired_steps
            else:
                for _ in range(desired_steps):
                    metrics = self.learner.train_step()
                    train_steps += 1
                    losses.append(float(metrics["loss"]))
                    grad_norms.append(float(metrics["grad_norm"]))
                    qz_r2_last = float(metrics["qz_r2"])
                    scaler_scale_last = float(metrics["scaler_scale"])
                    self.step_skipped_total += int(metrics["step_skipped"])
                    self.nonfinite_count += int(metrics["nonfinite_count"])
        self.cycles += 1
        self.total_train_steps += train_steps
        return {
            **stats,
            "cycle": self.cycles,
            "desired_train_steps": desired_steps,
            "train_steps": train_steps,
            "low_buffer_skips": self.low_buffer_skips,
            "loss_mean": (sum(losses) / len(losses)) if losses else None,
            "loss_last": losses[-1] if losses else None,
            "qz_r2_last": qz_r2_last,
            "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
            "scaler_scale_last": scaler_scale_last,
            "step_skipped_total": self.step_skipped_total,
            "nonfinite_count": self.nonfinite_count,
            "replay_size": len(self.replay),
            "replay_total_pushed": self.replay.total_pushed,
            "replay_reset": self.replay_reset,
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def train_from_config(
    run_config: DMCRunConfig,
    *,
    device_override: str | None = None,
    resume_path: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    total_games_override: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Run the full DMC training loop; returns the run summary mapping."""
    if total_games_override is not None:
        if int(total_games_override) <= 0:
            raise DMCConfigError("--total-games must be positive")
        run_config.total_games = int(total_games_override)
    output_dir = Path(
        output_dir_override if output_dir_override is not None
        else (run_config.output_dir or f"runs/{run_config.run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    progress_path = output_dir / "progress.json"
    started = time.perf_counter()
    started_clock = time.time()

    trainer = DMCTrainer(run_config, device_override=device_override)
    device = trainer.device
    if device_override is not None:
        log(f"[DMC] device override: {device_override} -> {device}")
    log(f"[DMC] device={device} envs={run_config.num_envs} games={run_config.total_games} "
        f"buffer={run_config.buffer_capacity} batch={run_config.batch_size} "
        f"channel_dim={trainer.replay.channel_dim}")

    if resume_path is not None:
        trainer.load_checkpoint(Path(resume_path))
        log(f"[DMC] resumed from {resume_path} at cycle={trainer.cycles} "
            f"games={trainer.sampler.games_completed} (replay_reset=true)")
    elif trainer.sampler.games_completed >= run_config.total_games:
        raise DMCConfigError("total_games must exceed the games already completed")

    _atomic_write_json(progress_path, {
        "schema": DMC_PROGRESS_SCHEMA,
        "status": "running",
        "run_name": run_config.run_name,
        "cycle": trainer.cycles,
        "games_completed": trainer.sampler.games_completed,
        "total_games": run_config.total_games,
        "replay_reset": trainer.replay_reset,
        "latest_checkpoint": None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    })

    def heartbeat(phase: str) -> None:
        _atomic_write_json(output_dir / "heartbeat.json", {
            "schema": DMC_HEARTBEAT_SCHEMA,
            "phase": phase,
            "cycle": trainer.cycles,
            "games_completed": trainer.sampler.games_completed,
            "total_games": run_config.total_games,
            "samples_pushed": trainer.sampler.samples_pushed,
            "replay_size": len(trainer.replay),
            "epsilon": trainer.sampler.epsilon(),
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        })

    last_heartbeat = [time.perf_counter()]

    def collect_progress(progress: Mapping[str, Any]) -> None:
        now = time.perf_counter()
        if now - last_heartbeat[0] < run_config.heartbeat_seconds:
            return
        last_heartbeat[0] = now
        heartbeat("collect")
        log(f"[DMC] collect progress: games={progress['games_completed']}/"
            f"{run_config.total_games} samples={progress['samples_pushed']} "
            f"epsilon={trainer.sampler.epsilon():.4f}")

    stop_reason = "complete"
    while trainer.sampler.games_completed < run_config.total_games:
        if (
            run_config.max_wall_seconds is not None
            and time.perf_counter() - started >= run_config.max_wall_seconds
        ):
            stop_reason = "budget_stopped"
            log("[DMC] wall budget reached at a cycle boundary; stopping cleanly")
            break
        trainer.wall_seconds = time.perf_counter() - started
        metrics = trainer.train_cycle(collect_progress)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"schema": DMC_METRICS_SCHEMA, **_json_safe(metrics)},
                sort_keys=True, allow_nan=False,
            ) + "\n")
        heartbeat("train")
        loss_text = "n/a" if metrics["loss_mean"] is None else f"{metrics['loss_mean']:.5f}"
        gps = metrics["games_per_second"]
        gps_text = "n/a" if gps is None else f"{gps:.3f}"
        log(f"[DMC] cycle {metrics['cycle']}: games={metrics['games']} "
            f"samples={metrics['samples_pushed']} steps={metrics['train_steps']} "
            f"loss={loss_text} gps={gps_text}")
        if (
            run_config.checkpoint_interval_cycles > 0
            and trainer.cycles % run_config.checkpoint_interval_cycles == 0
        ):
            checkpoint = output_dir / f"checkpoint_cycle_{trainer.cycles:06d}.pt"
            trainer.save_checkpoint(checkpoint)
            _atomic_write_json(progress_path, {
                "schema": DMC_PROGRESS_SCHEMA,
                "status": "running",
                "run_name": run_config.run_name,
                "cycle": trainer.cycles,
                "games_completed": trainer.sampler.games_completed,
                "total_games": run_config.total_games,
                "replay_reset": trainer.replay_reset,
                "latest_checkpoint": str(checkpoint),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            })

    trainer.wall_seconds = time.perf_counter() - started
    final_checkpoint = output_dir / "checkpoint_final.pt"
    trainer.save_checkpoint(final_checkpoint)

    records = trainer.sampler.game_records()
    sample_counts = [int(record["samples"]) for record in records]
    per_game = {
        "games_observed": len(sample_counts),
        "min": min(sample_counts) if sample_counts else None,
        "p50": _percentile(sample_counts, 0.50) if sample_counts else None,
        "p90": _percentile(sample_counts, 0.90) if sample_counts else None,
        "max": max(sample_counts) if sample_counts else None,
        "mean": (sum(sample_counts) / len(sample_counts)) if sample_counts else None,
    }
    summary = {
        "schema": DMC_RUN_SUMMARY_SCHEMA,
        "run_name": run_config.run_name,
        "stop_reason": stop_reason,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp_fp16": device.type == "cuda",
        "scaler_enabled": bool(trainer.learner.scaler.is_enabled()),
        "torch_version": torch.__version__,
        "channel_dim": trainer.replay.channel_dim,
        "max_obs_tokens": trainer.replay.max_obs_tokens,
        "replay_reset": trainer.replay_reset,
        "totals": {
            "cycles": trainer.cycles,
            "games_completed": trainer.sampler.games_completed,
            "samples_pushed": trainer.sampler.samples_pushed,
            "train_steps": trainer.total_train_steps,
            "low_buffer_skips": trainer.low_buffer_skips,
            "step_skipped_total": trainer.step_skipped_total,
            "nonfinite_count": trainer.nonfinite_count,
            "wall_seconds": trainer.wall_seconds,
        },
        "throughput": {
            "games_per_second": (
                trainer.sampler.games_completed / trainer.wall_seconds
                if trainer.wall_seconds > 0 else None
            ),
            "samples_per_second": (
                trainer.sampler.samples_pushed / trainer.wall_seconds
                if trainer.wall_seconds > 0 else None
            ),
        },
        "per_game_samples": per_game,
        "health": trainer.sampler.health(),
        "resolved_train_config": trainer.train_config.to_dict(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started_clock)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }
    _atomic_write_json(output_dir / "summary.json", _json_safe(summary))
    _atomic_write_json(progress_path, {
        "schema": DMC_PROGRESS_SCHEMA,
        "status": stop_reason,
        "run_name": run_config.run_name,
        "cycle": trainer.cycles,
        "games_completed": trainer.sampler.games_completed,
        "total_games": run_config.total_games,
        "replay_reset": trainer.replay_reset,
        "latest_checkpoint": str(final_checkpoint),
        "summary": "summary.json",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    })
    log(f"[DMC] done: {stop_reason} games={trainer.sampler.games_completed} "
        f"samples={trainer.sampler.samples_pushed} "
        f"gps={summary['throughput']['games_per_second']}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DMC stage-4 training loop")
    parser.add_argument("--config", required=True, help="path to the JSON training config")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None,
                        help="override the config device (explicit, logged)")
    parser.add_argument("--output-dir", default=None,
                        help="override the config output directory (explicit, logged)")
    parser.add_argument("--total-games", type=int, default=None,
                        help="override the config total game count (explicit, logged)")
    parser.add_argument("--resume", default=None,
                        help="path to a dmc-v1 full training checkpoint to resume")
    parser.add_argument("--parallel", action="store_true",
                        help="run the 07A two-process collector/learner topology "
                             "(on a single device the collector shares it, logged)")
    parser.add_argument("--collector-device", default=None,
                        help="explicit collector inference device for --parallel "
                             "(for example cuda:1); default: second GPU when present, "
                             "else the learner's device")
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_config = load_run_config(raw)
    if args.parallel:
        # Imported lazily so the serial path never loads the module.
        from guandan_marvel.dmc.parallel import train_from_config_parallel

        summary = train_from_config_parallel(
            run_config,
            device_override=args.device,
            collector_device_override=args.collector_device,
            resume_path=args.resume,
            output_dir_override=args.output_dir,
            total_games_override=args.total_games,
        )
    else:
        summary = train_from_config(
            run_config,
            device_override=args.device,
            resume_path=args.resume,
            output_dir_override=args.output_dir,
            total_games_override=args.total_games,
        )
    return 0 if summary["stop_reason"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
