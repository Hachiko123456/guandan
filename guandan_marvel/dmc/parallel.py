# SPDX-License-Identifier: MIT
"""DMC parallel trainer (stage 07A): collector/learner process overlap.

Two processes instead of the serial :class:`~guandan_marvel.dmc.train.DMCTrainer`
topology, so GPU training overlaps CPU-bound collection:

* **learner process** — owns the replay buffer, the training agent and the
  optimizer on the training device.  It drains sample batches pushed by the
  collector, and at every cycle close runs exactly the locked stage-4
  arithmetic (``desired = round(new_samples * gradient_replay_ratio /
  batch_size)`` steps of AdamW(lr=1e-4, wd=0, clip 1.0, CUDA fp16 autocast +
  GradScaler)).  While a training session runs it deliberately does **not**
  read the push queue, so the replay content at every gradient step is
  exactly cycles 1..k in push order — identical to the serial trainer.  The
  accepted row count is cross-checked against the collector's declared count
  at every close and any divergence is fatal.  After training it snapshots
  the updated weights (version ``k``) to the collector and pauses for a
  coordinator control message, which makes its post-training state race-free
  for checkpoint assembly.
* **collector process** — owns :class:`~guandan_marvel.dmc.sampler.DMCSampler`
  (full lockstep, ε-greedy, single-candidate auto-submit) with an
  :class:`~guandan_marvel.dmc.sampler.DMCSampleSink` that forwards finished
  games' batches to the learner instead of holding a second buffer.  Inference
  runs on a private agent copy: on a 2-GPU machine the collector's device is
  ``cuda:1`` (learner ``cuda:0``); on a single device both share it (the
  logged single-GPU degraded path — never a silent switch).

Weight sync is **per-cycle and versioned**: after training cycle ``k`` the
learner broadcasts version ``k``.  Before starting cycle ``c`` the collector
blocks until version ``pin(c) = max(base, c - 2)`` is loaded, where ``base``
is 0 for a fresh run or the restored cycle count on resume.  Because the
learner can only train cycle ``k`` after the collector closes cycle ``k``,
genuine overlap requires collecting cycle ``c`` with version ``c - 2`` — one
cycle of behavioral weight lag versus the serial contract (serial: ``c - 1``).
This is the FableDan ``infer_server`` broadcast pattern the 07A prompt
authorises; it touches no locked learning-semantics field (ε plan, gradient
ratio, optimizer, buffer, z and the sampling rules are all unchanged), and
the regression trajectory itself matches the serial trainer step for step.

Determinism: the collector's behaviour is a pure function of (pinned weight
version, env seed, sampler RNG, ε from pushed-sample count) — the pin makes
the version choice race-free, so the same configuration and seed reproduce
identical behavior sequences run to run.  Every weight load is verified by a
deterministic fp32 Q probe computed by both processes and compared by the
coordinator; a mismatch is recorded in the metrics row and logged (the two
devices of one machine are expected to agree bit-for-bit).
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import queue as queue_module
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from guandan_marvel.backends import get_game_backend
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig
from guandan_marvel.dmc import (
    DMC_ACTION_FEATURE_VERSION,
    DMC_CHECKPOINT_SCHEMA,
    DMC_TRAINING_SEMANTICS_PROFILE,
    DMCRuntimeConfig,
)
from guandan_marvel.dmc.learner import DMCLearner, optimizer_steps_for_samples
from guandan_marvel.dmc.q_agent import DMCAgent
from guandan_marvel.dmc.replay import DMCReplay
from guandan_marvel.dmc.sampler import DMCSampleSink, DMCSampler, DMCSamplerConfig
from guandan_marvel.dmc.train import (
    DMC_HEARTBEAT_SCHEMA,
    DMC_METRICS_SCHEMA,
    DMC_PROGRESS_SCHEMA,
    DMC_RUN_SUMMARY_SCHEMA,
    DMC_TRAIN_PAYLOAD_KIND,
    DMCConfigError,
    DMCRunConfig,
    _atomic_torch_save,
    _atomic_write_json,
    _json_safe,
    _percentile,
    resolve_device,
)

#: Extra summary key carrying the parallel topology evidence (additive; every
#: other summary field mirrors the serial run summary one for one).
PARALLEL_SUMMARY_KEY = "parallel"

_WATCHDOG_SECONDS = 30.0
_PROGRESS_SECONDS = 5.0

_PROBE_SEED = 20261025
_PROBE_TOKENS = 16
_PROBE_CANDIDATES = 3


def _weight_payload(model_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Wrap a raw state dict as the ``dmc-v1`` payload DMCAgent accepts."""
    return {
        "schema": DMC_CHECKPOINT_SCHEMA,
        "action_feature_version": DMC_ACTION_FEATURE_VERSION,
        "model_state": model_state,
    }


def _build_train_config(run_config: DMCRunConfig, device: str) -> TrainConfig:
    game = get_game_backend("python_reference")
    return TrainConfig(
        game=game,
        algo_name="dmc",
        ppo=PPOConfig(training_semantics_profile=DMC_TRAINING_SEMANTICS_PROFILE),
        dmc=DMCRuntimeConfig(
            buffer_capacity=run_config.buffer_capacity,
            batch_size=run_config.batch_size,
            accumulation_steps=run_config.accumulation_steps,
            obs_head_tokens=run_config.obs_head_tokens,
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
        device=device,
        seed=run_config.seed,
    )


def _probe_inputs(num_tokens: int, num_channels: int) -> dict[str, torch.Tensor]:
    """Deterministic Q-probe input, identical in every process."""
    generator = torch.Generator().manual_seed(_PROBE_SEED)
    tokens = torch.randint(
        1, max(2, int(num_tokens) - 1), (2, _PROBE_TOKENS), generator=generator,
    )
    lengths = torch.tensor([_PROBE_TOKENS, 9], dtype=torch.long)
    channels = (torch.rand(2, int(num_channels), generator=generator) * 255.0).to(torch.uint8)
    feats = torch.rand(2, _PROBE_CANDIDATES, 64, generator=generator)
    masks = torch.tensor(
        [[True] * _PROBE_CANDIDATES, [True, True, False]], dtype=torch.bool,
    )
    return {"tokens": tokens, "lengths": lengths, "channels": channels,
            "feats": feats, "masks": masks}


def _weight_probe(agent: DMCAgent, inputs: dict[str, torch.Tensor]) -> list[float]:
    """fp32 Q values on the fixed probe input (no autocast; deterministic)."""
    device = agent.backbone.state_tokens_embed.weight.device
    with torch.inference_mode():
        q = agent.q_values(
            inputs["tokens"].to(device),
            inputs["channels"].to(device),
            inputs["feats"].to(device),
            inputs["masks"].to(device),
            token_lengths=inputs["lengths"].to(device),
        )
    return [float(value) for value in q.detach().to("cpu").reshape(-1).tolist()]


def _state_to_cpu(value: Any) -> Any:
    """Recursively move tensors to CPU so IPC payloads stay portable."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_state_to_cpu(item) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    return value


def resolve_parallel_devices(
    requested: str, collector_override: str | None,
) -> tuple[torch.device, torch.device]:
    """Learner and collector devices; the single-device path is explicit.

    With no override and more than one CUDA device the collector takes the
    second GPU; otherwise it shares the learner's device (the logged
    single-GPU degraded path used for local verification).
    """
    learner = resolve_device(requested)
    if collector_override is not None:
        collector = torch.device(collector_override.strip().lower())
        if collector.type == "cuda":
            if collector.index is None:
                collector = torch.device("cuda", 0)
            if collector.index >= torch.cuda.device_count():
                raise DMCConfigError(
                    f"collector device {collector} does not exist "
                    f"(torch.cuda.device_count()={torch.cuda.device_count()})"
                )
        return learner, collector
    if learner.type == "cuda" and torch.cuda.device_count() > 1:
        return learner, torch.device("cuda", 1)
    return learner, learner


class _IPCSampleSink(DMCSampleSink):
    """Forwards finished games' batches to the learner process.

    The returned row count is the batch's own row count; the learner
    cross-checks it against its accepted count at the cycle close and any
    divergence is fatal, so an optimistic count can never silently desync the
    ε-clock from the replay.
    """

    def __init__(
        self,
        push_queue: Any,
        *,
        channel_dim: int,
        action_feature_dim: int,
        pad_token: int,
        num_tokens: int | None,
        num_players: int,
        max_obs_tokens: int,
    ) -> None:
        self.push_queue = push_queue
        self.channel_dim = int(channel_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.pad_token = int(pad_token)
        self.num_tokens = None if num_tokens is None else int(num_tokens)
        self.num_players = int(num_players)
        self.max_obs_tokens = int(max_obs_tokens)
        self._accepted = 0

    def push_batch(self, samples) -> int:
        rows = int(np.asarray(samples["obs_tokens"]).shape[0])
        self.push_queue.put(("push", dict(samples)))
        self._accepted += rows
        return rows

    def __len__(self) -> int:
        return self._accepted


def _child_get(mailbox: Any) -> tuple:
    """Blocking get for children; abort if the parent process disappears."""
    while True:
        try:
            return mailbox.get(timeout=_WATCHDOG_SECONDS)
        except queue_module.Empty:
            parent = mp.parent_process()
            if parent is not None and not parent.is_alive():
                sys.exit(1)


# ---------------------------------------------------------------------------
# Learner process
# ---------------------------------------------------------------------------
def _learner_process_main(
    run_config: DMCRunConfig,
    device: str,
    *,
    pushes: Any,
    weights: Any,
    events: Any,
    control: Any,
    initial: Mapping[str, Any] | None,
) -> None:
    try:
        torch.manual_seed(run_config.seed)
        train_config = _build_train_config(run_config, device)
        game = train_config.game
        replay = DMCReplay(
            run_config.buffer_capacity,
            max_obs_tokens=run_config.max_obs_tokens,
            channel_dim=int(game.num_channels),
            num_tokens=int(game.num_tokens),
            pad_token=int(game.pad_token),
            num_players=int(game.num_players),
            seed=run_config.seed,
        )
        agent = DMCAgent(train_config)
        learner = DMCLearner(
            agent, replay, device=device, config=train_config.dmc,
            accumulation_steps=run_config.accumulation_steps,
            seed=run_config.seed + 1,
        )

        base_cycle = 0
        low_buffer_skips = 0
        step_skipped_total = 0
        nonfinite_total = 0
        if initial is not None:
            agent.load_state_dict(_weight_payload(initial["model_state"]))
            learner.optimizer.load_state_dict(initial["optimizer_state"])
            learner.scaler.load_state_dict(initial["scaler_state"])
            learner.rng.bit_generator.state = initial["learner_rng_state"]
            torch.set_rng_state(initial["torch_cpu_rng_state"])
            cuda_states = list(initial.get("torch_cuda_rng_states", []))
            if learner.device.type == "cuda":
                if len(cuda_states) != torch.cuda.device_count():
                    raise DMCConfigError(
                        "checkpoint CUDA RNG state count does not match this machine"
                    )
                for index, state in enumerate(cuda_states):
                    torch.cuda.set_rng_state(
                        torch.as_tensor(state, dtype=torch.uint8),
                        torch.device("cuda", index),
                    )
            elif cuda_states:
                raise DMCConfigError(
                    "checkpoint carries CUDA RNG state but this learner is on CPU"
                )
            stats = initial["replay_stats"]
            for key, value in (
                ("capacity", replay.capacity),
                ("max_obs_tokens", replay.max_obs_tokens),
                ("channel_dim", replay.channel_dim),
                ("action_feature_dim", replay.action_feature_dim),
                ("num_tokens", replay.num_tokens),
                ("pad_token", replay.pad_token),
                ("num_players", replay.num_players),
            ):
                if stats.get(key) != value:
                    raise DMCConfigError(f"checkpoint replay_stats.{key} mismatch")
            base_cycle = int(initial["counters"]["cycles"])
            low_buffer_skips = int(initial["counters"]["low_buffer_skips"])
            step_skipped_total = int(initial["counters"]["step_skipped_total"])
            nonfinite_total = int(initial["counters"]["nonfinite_count"])

        probe_inputs = _probe_inputs(int(game.num_tokens), int(game.num_channels))
        events.put(("learner_ready", base_cycle, _weight_probe(agent, probe_inputs)))

        def checkpoint_part(cycle: int) -> dict[str, Any]:
            return {
                "model_state": {
                    name_: tensor.detach().to("cpu", copy=True)
                    for name_, tensor in agent.state_dict().items()
                },
                "optimizer_state": _state_to_cpu(learner.optimizer.state_dict()),
                "scaler_state": learner.scaler.state_dict(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_states": [
                    torch.cuda.get_rng_state(index).tolist()
                    for index in range(torch.cuda.device_count())
                ] if learner.device.type == "cuda" else [],
                "learner_rng_state": learner.rng.bit_generator.state,
                "replay_stats": {
                    "capacity": replay.capacity,
                    "max_obs_tokens": replay.max_obs_tokens,
                    "channel_dim": replay.channel_dim,
                    "action_feature_dim": replay.action_feature_dim,
                    "num_tokens": replay.num_tokens,
                    "pad_token": replay.pad_token,
                    "num_players": replay.num_players,
                    "size": len(replay),
                    "write_position": replay.write_position,
                    "total_pushed": replay.total_pushed,
                    "nbytes": replay.nbytes(),
                },
            }

        received = 0
        running = True
        while running:
            # Idle wait: pushes carry the collection stream, but the learner
            # also services control here so the coordinator can take a final
            # checkpoint snapshot or stop the run while no pushes are pending
            # (post-drain the learner parks here, not at the barrier below).
            message = None
            while message is None and running:
                try:
                    message = pushes.get(timeout=0.1)
                except queue_module.Empty:
                    try:
                        command = control.get_nowait()
                    except queue_module.Empty:
                        continue
                    if command[0] == "stop":
                        running = False
                    elif command[0] == "checkpoint":
                        events.put(("checkpoint_part", None, checkpoint_part(-1)))
                    else:
                        raise DMCConfigError(
                            f"unknown learner control {command[0]!r} while idle"
                        )
            if not running:
                break
            kind = message[0]
            if kind == "stop":
                break
            if kind == "push":
                received += int(replay.push_batch(message[1]))
                continue
            if kind != "close":  # pragma: no cover - protocol guard
                raise DMCConfigError(f"unknown learner message {kind!r}")

            cycle = int(message[1])
            declared = int(message[2])
            if declared != received:
                raise DMCConfigError(
                    f"DMC parallel push accounting mismatch at cycle {cycle}: "
                    f"collector declared {declared} rows, learner accepted {received}"
                )
            received = 0
            # Locked stage-4 gradient arithmetic (mirror of train.py); stage
            # 07B keeps dividing by the optimizer-step batch, never a micro-batch.
            desired_steps = optimizer_steps_for_samples(
                declared,
                ratio=run_config.gradient_replay_ratio,
                effective_batch_size=learner.effective_batch_size,
            )
            train_steps = 0
            losses: list[float] = []
            qz_r2_last = None
            grad_norms: list[float] = []
            scaler_scale_last = None
            if desired_steps > 0:
                if len(replay) < learner.effective_batch_size:
                    low_buffer_skips += desired_steps
                else:
                    for _ in range(desired_steps):
                        metrics = learner.train_step()
                        train_steps += 1
                        losses.append(float(metrics["loss"]))
                        grad_norms.append(float(metrics["grad_norm"]))
                        qz_r2_last = float(metrics["qz_r2"])
                        scaler_scale_last = float(metrics["scaler_scale"])
                        step_skipped_total += int(metrics["step_skipped"])
                        nonfinite_total += int(metrics["nonfinite_count"])

            weights.put((
                "weights", cycle,
                {name: tensor.detach().to("cpu", copy=True)
                 for name, tensor in agent.state_dict().items()},
            ))
            events.put(("cycle_trained", cycle, {
                "cycle": cycle,
                "desired_train_steps": desired_steps,
                "train_steps": train_steps,
                "effective_batch_size": learner.effective_batch_size,
                "accumulation_steps": run_config.accumulation_steps,
                "low_buffer_skips": low_buffer_skips,
                "loss_mean": (sum(losses) / len(losses)) if losses else None,
                "loss_last": losses[-1] if losses else None,
                "qz_r2_last": qz_r2_last,
                "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
                "scaler_scale_last": scaler_scale_last,
                "step_skipped_total": step_skipped_total,
                "nonfinite_count": nonfinite_total,
                "replay_size": len(replay),
                "replay_total_pushed": replay.total_pushed,
                "weight_version": cycle,
                "train_probe": _weight_probe(agent, probe_inputs),
            }))

            # Post-training barrier: the coordinator decides whether to take a
            # race-free checkpoint snapshot before the next cycle's pushes are
            # drained, keeping the replay at cycles 1..k during every gradient
            # step exactly like the serial trainer.
            while True:
                command = _child_get(control)
                name = command[0]
                if name == "continue":
                    break
                if name == "checkpoint":
                    events.put(("checkpoint_part", cycle, checkpoint_part(cycle)))
                    continue
                if name == "stop":
                    running = False
                    break
                raise DMCConfigError(f"unknown learner control {name!r}")
    except BaseException:  # noqa: BLE001 - forward every failure to the coordinator
        try:
            events.put(("fatal", "learner", traceback.format_exc()))
        except Exception:  # pragma: no cover - queue already torn down
            pass
        sys.exit(1)


# ---------------------------------------------------------------------------
# Collector process
# ---------------------------------------------------------------------------
def _actor_process_main(
    run_config: DMCRunConfig,
    collector_device: str,
    *,
    pushes: Any,
    weights: Any,
    events: Any,
    control: Any,
    initial: Mapping[str, Any] | None,
) -> None:
    try:
        torch.manual_seed(run_config.seed)
        train_config = _build_train_config(run_config, collector_device)
        game = train_config.game
        agent = DMCAgent(train_config)

        base_cycle = 0
        if initial is not None:
            agent.load_state_dict(_weight_payload(initial["model_state"]))
            base_cycle = int(initial["base_cycle"])
        device = torch.device(collector_device)
        agent.to(device)
        agent.eval()

        sink = _IPCSampleSink(
            pushes,
            channel_dim=int(game.num_channels),
            action_feature_dim=64,
            pad_token=int(game.pad_token),
            num_tokens=int(game.num_tokens),
            num_players=int(game.num_players),
            max_obs_tokens=run_config.max_obs_tokens,
        )
        sampler = DMCSampler(
            q_agent=agent,
            replay=sink,
            config=DMCSamplerConfig(
                num_envs=run_config.num_envs,
                device=collector_device,
                env_seed=run_config.env_seed,
                seed=run_config.seed,
                max_obs_tokens=run_config.max_obs_tokens,
                obs_head_tokens=run_config.obs_head_tokens,
                epsilon_start=run_config.epsilon_start,
                epsilon_final=run_config.epsilon_final,
                epsilon_decay_samples=run_config.epsilon_decay_samples,
                context_schedule=(
                    None if run_config.context_schedule is None
                    else dict(run_config.context_schedule)
                ),
            ),
        )
        if initial is not None:
            sampler.load_state_dict(initial["sampler_state"])

        probe_inputs = _probe_inputs(int(game.num_tokens), int(game.num_channels))
        loaded_version = base_cycle
        cycle = base_cycle
        while True:
            remaining = run_config.total_games - sampler.games_completed
            cycle_games = min(int(run_config.games_per_cycle), remaining)
            last_progress = [time.perf_counter()]

            def progress(progress_state: Mapping[str, Any]) -> None:
                now = time.perf_counter()
                if now - last_progress[0] < _PROGRESS_SECONDS:
                    return
                last_progress[0] = now
                events.put(("progress", int(progress_state["games_completed"]),
                            int(progress_state["samples_pushed"]),
                            float(sampler.epsilon())))

            stats = sampler.collect_games(cycle_games, progress_callback=progress)
            # The close marker follows every cycle-k push on the same FIFO
            # queue, so the learner's accepted-count cross-check is exact.
            pushes.put(("close", cycle, int(stats["samples_pushed"])))
            probe = _weight_probe(agent, probe_inputs)
            events.put(("cycle_done", cycle, {
                "games": int(stats["games"]),
                "samples_pushed": int(stats["samples_pushed"]),
                "decisions": int(stats["decisions"]),
                "epsilon_start": float(stats["epsilon_start"]),
                "epsilon_end": float(stats["epsilon_end"]),
                "wall_seconds": float(stats["wall_seconds"]),
                "games_per_second": stats["games_per_second"],
                "samples_per_second": stats["samples_per_second"],
                "health": stats["health"],
            }, [dict(record) for record in sampler.game_records()],
                loaded_version, probe))

            command = _child_get(control)
            name = command[0]
            if name == "stop":
                events.put(("final_state", sampler.state_dict()))
                break
            if name != "next":  # pragma: no cover - protocol guard
                raise DMCConfigError(f"unknown actor control {name!r}")
            if len(command) > 1 and bool(command[1]):
                events.put(("boundary_state", sampler.state_dict()))

            cycle += 1
            target = max(base_cycle, cycle - 2)
            while loaded_version < target:
                message = _child_get(weights)
                if message[0] != "weights":  # pragma: no cover - protocol guard
                    raise DMCConfigError(f"unknown weight message {message[0]!r}")
                version, state = int(message[1]), message[2]
                if version != loaded_version + 1:
                    raise DMCConfigError(
                        f"weight versions must arrive in order: got {version}, "
                        f"expected {loaded_version + 1}"
                    )
                agent.load_state_dict(_weight_payload(state))
                loaded_version = version
            # Versions past the pin stay queued; the learner is at most one
            # cycle ahead, so at most one message ever waits here.
    except BaseException:  # noqa: BLE001 - forward every failure to the coordinator
        try:
            events.put(("fatal", "collector", traceback.format_exc()))
        except Exception:  # pragma: no cover - queue already torn down
            pass
        sys.exit(1)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
class _ParallelProcesses:
    """Spawn context for the two children plus their mailboxes."""

    def __init__(
        self,
        run_config: DMCRunConfig,
        *,
        learner_device: str,
        collector_device: str,
        learner_initial: Mapping[str, Any] | None,
        actor_initial: Mapping[str, Any] | None,
    ) -> None:
        context = mp.get_context("spawn")
        self.pushes = context.Queue()
        self.weights = context.Queue()
        self.actor_events = context.Queue()
        self.learner_events = context.Queue()
        self.learner_control = context.Queue()
        self.actor_control = context.Queue()
        self.learner = context.Process(
            target=_learner_process_main,
            args=(run_config, learner_device),
            kwargs={
                "pushes": self.pushes, "weights": self.weights,
                "events": self.learner_events, "control": self.learner_control,
                "initial": learner_initial,
            },
            daemon=True,
        )
        self.actor = context.Process(
            target=_actor_process_main,
            args=(run_config, collector_device),
            kwargs={
                "pushes": self.pushes, "weights": self.weights,
                "events": self.actor_events, "control": self.actor_control,
                "initial": actor_initial,
            },
            daemon=True,
        )
        self.learner.start()
        self.actor.start()

    def alive(self) -> tuple[bool, bool]:
        return self.learner.is_alive(), self.actor.is_alive()

    def shutdown(self) -> None:
        for process in (self.actor, self.learner):
            if process.is_alive():
                process.terminate()
        for process in (self.actor, self.learner):
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for mailbox in (self.pushes, self.weights, self.actor_events,
                        self.learner_events, self.learner_control, self.actor_control):
            try:
                mailbox.close()
            except Exception:  # pragma: no cover - already closed
                pass


class DMCParallelTrainer:
    """Coordinator mirroring :class:`~guandan_marvel.dmc.train.DMCTrainer`."""

    def __init__(
        self,
        run_config: DMCRunConfig,
        *,
        learner_device: torch.device,
        collector_device: torch.device,
        resume_path: str | Path | None = None,
    ) -> None:
        self.run_config = run_config
        self.learner_device = learner_device
        self.collector_device = collector_device
        self.cycles = 0
        self.total_train_steps = 0
        self.low_buffer_skips = 0
        self.step_skipped_total = 0
        self.nonfinite_count = 0
        self.wall_seconds = 0.0
        self.replay_reset = False
        self.game_records: list[dict[str, Any]] = []
        self._processes: _ParallelProcesses | None = None
        self._resume_payload: Mapping[str, Any] | None = None
        self._learner_probes: dict[int, list[float]] = {}
        self._last_checkpoint_state: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
        if resume_path is not None:
            self._resume_payload = self._load_resume(Path(resume_path))

    # -- resume validation (mirror of the serial load_checkpoint checks) -----
    def _load_resume(self, path: Path) -> Mapping[str, Any]:
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
        for operational in ("output_dir", "total_games"):
            saved_config = {k: v for k, v in saved_config.items() if k != operational}
            current = {k: v for k, v in current.items() if k != operational}
        if dict(saved_config) != current:
            raise DMCConfigError(
                "checkpoint run_config mismatch; a DMC resume must replay the exact "
                "training configuration"
            )
        cuda_states = list(payload.get("torch_cuda_rng_states", []))
        if self.learner_device.type == "cuda":
            if len(cuda_states) != torch.cuda.device_count():
                raise DMCConfigError(
                    "checkpoint CUDA RNG state count does not match this machine"
                )
        elif cuda_states:
            raise DMCConfigError(
                "checkpoint carries CUDA RNG state but this run is on CPU"
            )
        stats = payload.get("replay_stats")
        if not isinstance(stats, Mapping):
            raise DMCConfigError("checkpoint is missing its replay_stats")
        game = get_game_backend("python_reference")
        for key, value in (
            ("capacity", self.run_config.buffer_capacity),
            ("max_obs_tokens", self.run_config.max_obs_tokens),
            ("channel_dim", int(game.num_channels)),
            ("num_tokens", int(game.num_tokens)),
            ("pad_token", int(game.pad_token)),
            ("num_players", int(game.num_players)),
        ):
            if stats.get(key) != value:
                raise DMCConfigError(f"checkpoint replay_stats.{key} mismatch")
        return payload

    # -- child bootstrap -----------------------------------------------------
    def spawn(self) -> _ParallelProcesses:
        if self._processes is not None:
            return self._processes
        payload = self._resume_payload
        learner_initial = None
        actor_initial = None
        if payload is not None:
            counters = payload["counters"]
            base_cycle = int(counters["cycles"])
            learner_initial = {
                "model_state": payload["model_state"],
                "optimizer_state": payload["optimizer_state"],
                "scaler_state": payload["scaler_state"],
                "torch_cpu_rng_state": payload["torch_cpu_rng_state"],
                "torch_cuda_rng_states": payload.get("torch_cuda_rng_states", []),
                "learner_rng_state": payload["learner_rng_state"],
                "replay_stats": payload["replay_stats"],
                "counters": counters,
            }
            actor_initial = {
                "model_state": payload["model_state"],
                "sampler_state": payload["sampler_state"],
                "base_cycle": base_cycle,
            }
            self.cycles = base_cycle
            self.total_train_steps = int(counters["total_train_steps"])
            self.low_buffer_skips = int(counters["low_buffer_skips"])
            self.step_skipped_total = int(counters["step_skipped_total"])
            self.nonfinite_count = int(counters["nonfinite_count"])
            self.wall_seconds = float(counters["wall_seconds"])
            self.replay_reset = True
        self._processes = _ParallelProcesses(
            self.run_config,
            learner_device=str(self.learner_device),
            collector_device=str(self.collector_device),
            learner_initial=learner_initial,
            actor_initial=actor_initial,
        )
        return self._processes

    # -- cycle orchestration -------------------------------------------------
    def train_cycle(self, *, checkpoint_due: bool) -> dict[str, Any]:
        """Wait for one collector cycle and its locked gradient session."""
        processes = self.spawn()
        actor_message = None
        learner_message = None
        while actor_message is None or learner_message is None:
            if actor_message is None:
                message = self._wait(processes.actor_events, "collector")
                kind = message[0]
                if kind == "fatal":
                    raise RuntimeError(f"DMC parallel collector failed:\n{message[2]}")
                if kind == "cycle_done":
                    actor_message = message
                # "progress" rows only feed the watchdog; dropped here.
            if learner_message is None:
                message = self._wait(processes.learner_events, "learner")
                kind = message[0]
                if kind == "fatal":
                    raise RuntimeError(f"DMC parallel learner failed:\n{message[2]}")
                if kind == "learner_ready":
                    self._learner_probes[int(message[1])] = list(message[2])
                elif kind == "cycle_trained":
                    learner_message = message

        _, actor_cycle, actor_stats, records, used_version, actor_probe = actor_message
        _, learner_cycle, learner_stats = learner_message
        if actor_cycle != self.cycles or learner_cycle != self.cycles:
            raise DMCConfigError(
                f"DMC parallel cycle desync: coordinator at {self.cycles}, "
                f"collector reported {actor_cycle}, learner reported {learner_cycle}"
            )
        self.game_records = [dict(record) for record in records]
        learner_probe = self._learner_probes.get(int(used_version))
        probe_match = (
            None if learner_probe is None else learner_probe == list(actor_probe)
        )

        # Coordinator decisions at the cycle boundary.
        self.cycles += 1
        self.total_train_steps += int(learner_stats["train_steps"])
        self.low_buffer_skips = int(learner_stats["low_buffer_skips"])
        self.step_skipped_total = int(learner_stats["step_skipped_total"])
        self.nonfinite_count = int(learner_stats["nonfinite_count"])
        row = {
            **actor_stats,
            "cycle": self.cycles,
            "desired_train_steps": int(learner_stats["desired_train_steps"]),
            "train_steps": int(learner_stats["train_steps"]),
            "effective_batch_size": int(learner_stats["effective_batch_size"]),
            "accumulation_steps": int(learner_stats["accumulation_steps"]),
            "low_buffer_skips": self.low_buffer_skips,
            "loss_mean": learner_stats["loss_mean"],
            "loss_last": learner_stats["loss_last"],
            "qz_r2_last": learner_stats["qz_r2_last"],
            "grad_norm_mean": learner_stats["grad_norm_mean"],
            "scaler_scale_last": learner_stats["scaler_scale_last"],
            "step_skipped_total": self.step_skipped_total,
            "nonfinite_count": self.nonfinite_count,
            "replay_size": int(learner_stats["replay_size"]),
            "replay_total_pushed": int(learner_stats["replay_total_pushed"]),
            "replay_reset": self.replay_reset,
            "weight_version_used": int(used_version),
            "weight_probe_match": probe_match,
        }

        self._last_checkpoint_state = None
        if checkpoint_due:
            # The learner part is race-free here: it is paused at its
            # post-training barrier until ("continue",) below.
            processes.learner_control.put(("checkpoint",))
            part = None
            while part is None:
                message = self._wait(processes.learner_events, "learner")
                if message[0] == "checkpoint_part":
                    part = message[2]
                elif message[0] == "fatal":
                    raise RuntimeError(f"DMC parallel learner failed:\n{message[2]}")
            self._last_checkpoint_state = part
            processes.learner_control.put(("continue",))
        else:
            processes.learner_control.put(("continue",))
        return row

    def advance_collector(self, *, need_state: bool) -> Mapping[str, Any] | None:
        """Let the collector start its next cycle (coordinator's decision).

        The entry loop owns this decision so a run whose game target was just
        reached (or whose budget expired) stops the collector at its boundary
        instead of letting it start a zero-game cycle.  With ``need_state``
        the collector answers with its race-free boundary sampler state for
        the interval checkpoint before it moves on.
        """
        processes = self._processes
        assert processes is not None
        processes.actor_control.put(("next", need_state))
        if not need_state:
            return None
        boundary = self._wait(processes.actor_events, "collector")
        if boundary[0] != "boundary_state":
            raise DMCConfigError(
                f"expected collector boundary state, got {boundary[0]!r}"
            )
        return boundary[1]

    def _wait(self, mailbox: Any, who: str) -> tuple:
        """Blocking get with a liveness watchdog; a silent dead child is fatal."""
        processes = self._processes
        assert processes is not None
        while True:
            try:
                return mailbox.get(timeout=_WATCHDOG_SECONDS)
            except queue_module.Empty:
                learner_alive, actor_alive = processes.alive()
                if not learner_alive or not actor_alive:
                    raise DMCConfigError(
                        f"DMC parallel {who} wait aborted: learner alive={learner_alive} "
                        f"collector alive={actor_alive} "
                        f"(learner exitcode={processes.learner.exitcode}, "
                        f"collector exitcode={processes.actor.exitcode})"
                    )

    # -- shutdown / checkpoint ------------------------------------------------
    def stop_children(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Final boundary: collector state + learner checkpoint part."""
        processes = self._processes
        assert processes is not None
        processes.actor_control.put(("stop",))
        final_state = None
        while final_state is None:
            message = self._wait(processes.actor_events, "collector")
            if message[0] == "final_state":
                final_state = message[1]
            elif message[0] == "fatal":
                raise RuntimeError(f"DMC parallel collector failed:\n{message[2]}")
        processes.learner_control.put(("checkpoint",))
        part = None
        while part is None:
            message = self._wait(processes.learner_events, "learner")
            if message[0] == "checkpoint_part":
                part = message[2]
            elif message[0] == "fatal":
                raise RuntimeError(f"DMC parallel learner failed:\n{message[2]}")
        processes.learner_control.put(("stop",))
        for process in (processes.actor, processes.learner):
            process.join(timeout=30)
        return final_state, part

    def close(self) -> None:
        if self._processes is not None:
            self._processes.shutdown()
            self._processes = None

    def checkpoint_payload(
        self,
        *,
        sampler_state: Mapping[str, Any],
        learner_part: Mapping[str, Any],
        games_completed: int,
        samples_pushed: int,
    ) -> dict[str, Any]:
        return {
            "schema": DMC_CHECKPOINT_SCHEMA,
            "action_feature_version": DMC_ACTION_FEATURE_VERSION,
            "payload_kind": DMC_TRAIN_PAYLOAD_KIND,
            "model_state": learner_part["model_state"],
            "optimizer_state": learner_part["optimizer_state"],
            "scaler_state": learner_part["scaler_state"],
            "torch_cpu_rng_state": learner_part["torch_cpu_rng_state"],
            "torch_cuda_rng_states": learner_part["torch_cuda_rng_states"],
            "sampler_state": sampler_state,
            "learner_rng_state": learner_part["learner_rng_state"],
            "replay_stats": learner_part["replay_stats"],
            "counters": {
                "cycles": self.cycles,
                "games_completed": int(games_completed),
                "samples_pushed": int(samples_pushed),
                "total_train_steps": self.total_train_steps,
                "low_buffer_skips": self.low_buffer_skips,
                "step_skipped_total": self.step_skipped_total,
                "nonfinite_count": self.nonfinite_count,
                "wall_seconds": self.wall_seconds,
                "replay_reset": self.replay_reset,
            },
            "run_config": dataclasses.asdict(self.run_config),
        }


# ---------------------------------------------------------------------------
# Entry point (mirrors train_from_config)
# ---------------------------------------------------------------------------
def train_from_config_parallel(
    run_config: DMCRunConfig,
    *,
    device_override: str | None = None,
    collector_device_override: str | None = None,
    resume_path: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    total_games_override: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Run the DMC training loop on the two-process 07A topology."""
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

    learner_device, collector_device = resolve_parallel_devices(
        (device_override or run_config.device), collector_device_override,
    )
    degraded = learner_device == collector_device
    log(f"[DMC] parallel topology: learner={learner_device} collector={collector_device}"
        + (" (single-device degraded path: collector shares the learner device)"
           if degraded else ""))
    trainer = DMCParallelTrainer(
        run_config,
        learner_device=learner_device,
        collector_device=collector_device,
        resume_path=resume_path,
    )
    try:
        if resume_path is not None:
            log(f"[DMC] parallel resume from {resume_path} at cycle={trainer.cycles} "
                f"(replay_reset=true)")
        games_completed = 0
        samples_pushed = 0
        if trainer._resume_payload is not None:
            counters = trainer._resume_payload["counters"]
            games_completed = int(counters["games_completed"])
            samples_pushed = int(counters["samples_pushed"])
        if games_completed >= run_config.total_games:
            raise DMCConfigError("total_games must exceed the games already completed")

        _atomic_write_json(progress_path, {
            "schema": DMC_PROGRESS_SCHEMA,
            "status": "running",
            "run_name": run_config.run_name,
            "cycle": trainer.cycles,
            "games_completed": games_completed,
            "total_games": run_config.total_games,
            "replay_reset": trainer.replay_reset,
            "latest_checkpoint": None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        })

        def heartbeat(phase: str, games: int, samples: int, replay_size: int,
                      epsilon: float | None) -> None:
            _atomic_write_json(output_dir / "heartbeat.json", {
                "schema": DMC_HEARTBEAT_SCHEMA,
                "phase": phase,
                "cycle": trainer.cycles,
                "games_completed": games,
                "total_games": run_config.total_games,
                "samples_pushed": samples,
                "replay_size": replay_size,
                "epsilon": epsilon,
                "elapsed_seconds": time.perf_counter() - started,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            })

        trainer.spawn()
        row = None
        stop_reason = "complete"
        while games_completed < run_config.total_games:
            elapsed = time.perf_counter() - started
            if (
                run_config.max_wall_seconds is not None
                and elapsed >= run_config.max_wall_seconds
            ):
                stop_reason = "budget_stopped"
                log("[DMC] wall budget reached at a cycle boundary; stopping cleanly")
                break
            trainer.wall_seconds = time.perf_counter() - started
            interval = int(run_config.checkpoint_interval_cycles)
            checkpoint_due = interval > 0 and (trainer.cycles + 1) % interval == 0
            row = trainer.train_cycle(checkpoint_due=checkpoint_due)
            games_completed = int(row["health"]["games_completed"])
            samples_pushed = int(row["health"]["samples_pushed"])
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"schema": DMC_METRICS_SCHEMA, **_json_safe(row)},
                    sort_keys=True, allow_nan=False,
                ) + "\n")
            heartbeat("train", games_completed, samples_pushed,
                      int(row["replay_size"]), float(row["epsilon_end"]))
            loss_text = "n/a" if row["loss_mean"] is None else f"{row['loss_mean']:.5f}"
            gps = row["games_per_second"]
            log(f"[DMC] cycle {row['cycle']}: games={row['games']} "
                f"samples={row['samples_pushed']} steps={row['train_steps']} "
                f"loss={loss_text} gps={gps if gps is None else round(gps, 3)} "
                f"weight_version={row['weight_version_used']} "
                f"probe_match={row['weight_probe_match']}")
            if checkpoint_due and games_completed < run_config.total_games:
                boundary_state = trainer.advance_collector(need_state=True)
                checkpoint = output_dir / f"checkpoint_cycle_{trainer.cycles:06d}.pt"
                _atomic_torch_save(
                    trainer.checkpoint_payload(
                        sampler_state=boundary_state,
                        learner_part=trainer._last_checkpoint_state,
                        games_completed=games_completed,
                        samples_pushed=samples_pushed,
                    ),
                    checkpoint,
                )
                _atomic_write_json(progress_path, {
                    "schema": DMC_PROGRESS_SCHEMA,
                    "status": "running",
                    "run_name": run_config.run_name,
                    "cycle": trainer.cycles,
                    "games_completed": games_completed,
                    "total_games": run_config.total_games,
                    "replay_reset": trainer.replay_reset,
                    "latest_checkpoint": str(checkpoint),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                })
            elif games_completed < run_config.total_games:
                # The run continues: let the collector leave its boundary.  A
                # reached target or expired budget leaves it waiting here for
                # the stop_children stop instead.
                trainer.advance_collector(need_state=False)

        trainer.wall_seconds = time.perf_counter() - started
        final_sampler_state, final_learner_part = trainer.stop_children()
        # The sampler's own counters are authoritative: a budget stop can end
        # the run while a cycle was still in flight, and those completed games
        # are in the final state even though the coordinator loop never saw
        # their metrics row.
        games_completed = int(final_sampler_state["games_completed"])
        samples_pushed = int(final_sampler_state["samples_pushed"])
        final_checkpoint = output_dir / "checkpoint_final.pt"
        _atomic_torch_save(
            trainer.checkpoint_payload(
                sampler_state=final_sampler_state,
                learner_part=final_learner_part,
                games_completed=games_completed,
                samples_pushed=samples_pushed,
            ),
            final_checkpoint,
        )

        records = trainer.game_records
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
            "device": str(learner_device),
            "cuda_available": bool(torch.cuda.is_available()),
            "amp_fp16": learner_device.type == "cuda",
            "scaler_enabled": learner_device.type == "cuda",
            "torch_version": torch.__version__,
            "channel_dim": int(final_learner_part["replay_stats"]["channel_dim"]),
            "max_obs_tokens": int(final_learner_part["replay_stats"]["max_obs_tokens"]),
            "replay_reset": trainer.replay_reset,
            "totals": {
                "cycles": trainer.cycles,
                "games_completed": games_completed,
                "samples_pushed": samples_pushed,
                "train_steps": trainer.total_train_steps,
                "low_buffer_skips": trainer.low_buffer_skips,
                "step_skipped_total": trainer.step_skipped_total,
                "nonfinite_count": trainer.nonfinite_count,
                "wall_seconds": trainer.wall_seconds,
            },
            "throughput": {
                "games_per_second": (
                    games_completed / trainer.wall_seconds if trainer.wall_seconds > 0 else None
                ),
                "samples_per_second": (
                    samples_pushed / trainer.wall_seconds if trainer.wall_seconds > 0 else None
                ),
            },
            "per_game_samples": per_game,
            "health": dict(row["health"]) if row is not None else {},
            "game_records": [dict(record) for record in records],
            PARALLEL_SUMMARY_KEY: {
                "learner_device": str(learner_device),
                "collector_device": str(collector_device),
                "single_device_degraded": degraded,
                "weight_sync": (
                    "per-cycle versioned broadcast; collector pins "
                    "max(base, cycle-2) and blocks until loaded"
                ),
                "behavioral_weight_lag_cycles": 1,
                "weight_probe": (
                    "deterministic fp32 Q probe computed by both processes and "
                    "compared per weight version (None = init version before the "
                    "learner's first probe)"
                ),
            },
            "resolved_train_config": _build_train_config(
                run_config, str(learner_device),
            ).to_dict(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started_clock)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        }
        _atomic_write_json(output_dir / "summary.json", _json_safe(summary))
        _atomic_write_json(progress_path, {
            "schema": DMC_PROGRESS_SCHEMA,
            "status": stop_reason,
            "run_name": run_config.run_name,
            "cycle": trainer.cycles,
            "games_completed": games_completed,
            "total_games": run_config.total_games,
            "replay_reset": trainer.replay_reset,
            "latest_checkpoint": str(final_checkpoint),
            "summary": "summary.json",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        })
        log(f"[DMC] done: {stop_reason} games={games_completed} "
            f"samples={samples_pushed} "
            f"gps={summary['throughput']['games_per_second']}")
        return summary
    finally:
        trainer.close()


__all__ = [
    "PARALLEL_SUMMARY_KEY",
    "DMCParallelTrainer",
    "resolve_parallel_devices",
    "train_from_config_parallel",
]
