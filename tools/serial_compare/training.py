# SPDX-License-Identifier: MIT
"""Real per-arm training wiring for the serial comparison.

The orchestration contract is injectable, so the state machine can be tested
with fake actions; this module is the production action implementation that the
Kaggle notebook runs on two ranks.  It is *not* exercised locally without the
original Linux native and two T4 GPUs, and any such run must be reported as
NOT_RUN rather than claimed as target certification.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .common import PlanError, atomic_json, canonical, read_json, sha

ARM_RUNTIME_SCHEMA = "guandan-serial-compare-arm-runtime-v1"


def _source_config(source_payload: Mapping[str, object]) -> Mapping[str, object]:
    resolved = source_payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise PlanError("the 375 source checkpoint has no resolved_config")
    return resolved


def build_train_config(
    plan: Mapping[str, object],
    arm: str,
    *,
    game,
    source_payload: Mapping[str, object],
    device: str,
    ddp_enabled: bool,
    world_size: int,
    opponent_entries: tuple[Mapping[str, object], ...] = (),
    seed_offset: int = 0,
):
    """Build the arm's ``TrainConfig`` from the frozen plan and real source config."""
    from guandan_marvel.config import (
        DDPConfig, ModelConfig, OptimizerConfig, PPOConfig, ScheduleConfig, TrainConfig,
    )
    from guandan_marvel.phase6.config_types import OpponentPoolConfig

    spec = dict((plan["arms"] or {})[arm])
    source = _source_config(source_payload)
    source_model = dict(source["model"])
    source_ppo = dict(source["ppo"])
    model_spec = dict(spec["model"])
    for name, expected in model_spec.items():
        if name == "architecture":
            continue
        if int(source_model.get(name, -1)) != int(expected):
            raise PlanError(f"source model {name!r} disagrees with the frozen arm model")
    if str(source_model.get("architecture")) != str(model_spec["architecture"]):
        raise PlanError("source architecture disagrees with the frozen arm model")
    if str(source_ppo.get("training_semantics_profile")) != str(spec["ppo"]["training_semantics_profile"]):
        raise PlanError("source training semantics disagree with the frozen arm")
    if str(source_ppo.get("policy_regularizer_mode")) != str(spec["ppo"]["policy_regularizer_mode"]):
        raise PlanError("source regularizer mode disagrees with the frozen arm")

    ppo = PPOConfig(
        base_lr=float(spec["ppo"]["base_lr"]),
        agent_epoch=int(spec["ppo"]["agent_epoch"]),
        gae_lambda=float(spec["ppo"]["gae_lambda"]),
        base_clip_coef=float(spec["ppo"]["base_clip_coef"]),
        base_reg_coef=float(spec["ppo"]["base_reg_coef"]),
        training_semantics_profile=str(spec["ppo"]["training_semantics_profile"]),
        policy_regularizer_mode=str(spec["ppo"]["policy_regularizer_mode"]),
        macro_gae_lambda=spec["ppo"].get("macro_gae_lambda"),
        target_kl=None if spec.get("target_kl") is None else float(spec["target_kl"]),
    )
    pool_spec = dict(spec["opponent_pool"])
    pool = OpponentPoolConfig(
        enabled=bool(pool_spec["enabled"]),
        entries=tuple(dict(entry) for entry in opponent_entries),
        sampling_seed=int(pool_spec.get("sampling_seed", 0)),
        sampling_scheme=str(pool_spec.get("sampling_scheme", "guandan-marvel-phase6-weighted-active-sampling-v1")),
        assignment_scheme=str(pool_spec.get("assignment_scheme", "team_rotation_v1")),
        max_resident_policies=int(pool_spec.get("max_resident_policies", 1)),
    )
    schedule_source = source.get("schedule")
    schedule = (
        ScheduleConfig(**{str(k): float(v) for k, v in dict(schedule_source).items()})
        if isinstance(schedule_source, Mapping) and schedule_source
        else ScheduleConfig()
    )
    config = TrainConfig(
        game=game.spec if hasattr(game, "spec") else game.GAME_SPEC,
        algo_name="ippo",
        device=str(device),
        # Preflight runs in its own seed namespace; the formal arms use the
        # plan's training base unchanged (seed_offset == 0).
        seed=int(spec["seed"]) + int(seed_offset),
        run_name=f"serial-compare-{arm}",
        rollout_batch_size=int(spec["rollout_batch_size"]),
        collect_batch_size=int(spec["collect_batch_size"]),
        mini_batches=int(spec["mini_batches"]),
        value_loss_coef=float(spec["value_loss_coef"]),
        discount_gamma=float(spec["discount_gamma"]),
        environment_rank_seed_stride=int(spec["environment_rank_seed_stride"]),
        reward_semantics=str(source["reward_semantics"]),
        reward_scale=float(source["reward_scale"]),
        training_data_generator_version=str(source["training_data_generator_version"]),
        training_context_schedule=source.get("training_context_schedule"),
        training_context_curriculum=source.get("training_context_curriculum"),
        ntp_weight=float(spec["ntp_weight"]),
        ntp_pos_weight=float(spec["ntp_pos_weight"]),
        amp_enabled=bool(spec["amp_enabled"]),
        trim_completed_padding=bool(spec["trim_completed_padding"]),
        max_sampler_bytes_per_rank=int(source.get("max_sampler_bytes_per_rank") or 6 * 1024**3),
        # Inherit the source's non-arm semantics instead of silently reverting
        # them to library defaults (a resolved config must not pretend to be
        # the real baseline while differing in schedule/replay/debug flags).
        debug=bool(source.get("debug", True)),
        replay_buffer_scale=int(source.get("replay_buffer_scale", 64)),
        total_iterations=int(source.get("total_iterations", 2000)),
        checkpoint_replay_buffer=bool(source.get("checkpoint_replay_buffer", True)),
        schedule=schedule,
        model=ModelConfig(**{**model_spec}),
        ppo=ppo,
        optimizer=OptimizerConfig(
            use_muon=bool(dict(source.get("optimizer") or {}).get("use_muon", False)),
            adam_beta1=float(dict(source.get("optimizer") or {}).get("adam_beta1", 0.9)),
            adam_beta2=float(dict(source.get("optimizer") or {}).get("adam_beta2", 0.999)),
            muon_beta1=float(dict(source.get("optimizer") or {}).get("muon_beta1", 0.95)),
            weight_decay=float(dict(source.get("optimizer") or {}).get("weight_decay", 1e-2)),
            grad_max_norm=float(dict(source.get("optimizer") or {}).get("grad_max_norm", 1.0)),
        ),
        ddp=DDPConfig(enabled=bool(ddp_enabled), backend="nccl", require_cuda=bool(ddp_enabled),
                      find_unused_parameters=bool(dict(source.get("ddp") or {}).get("find_unused_parameters", False))),
        opponent_pool=pool,
    )
    # The two arms may not silently diverge in the inherited non-arm settings.
    resolved = config.to_dict()
    for name in ("training_context_schedule", "training_context_curriculum"):
        if resolved.get(name) != source.get(name):
            raise PlanError(f"{name} must be inherited verbatim from the 375 source")
    expected_schedule = (
        {str(k): float(v) for k, v in dict(schedule_source).items()}
        if isinstance(schedule_source, Mapping) and schedule_source else {}
    )
    if resolved.get("schedule", {}) != expected_schedule:
        raise PlanError("the learning-rate/clip/reg schedule must be inherited from the 375 source")
    for name, default, cast in (
        ("debug", True, bool),
        ("replay_buffer_scale", 64, int),
        ("total_iterations", 2000, int),
        ("checkpoint_replay_buffer", True, bool),
    ):
        if resolved.get(name) != cast(source.get(name, default)):
            raise PlanError(f"{name} must be inherited from the 375 source")
    return config


def build_pool_entries(
    plan: Mapping[str, object], arm: str, model_paths: Mapping[str, str | Path], *,
    load_policy=None,
) -> tuple[dict[str, object], ...]:
    """Pool entries describing the *actual* frozen 250/375 artifacts."""
    spec = dict((plan["arms"] or {})[arm])
    pool = dict(spec["opponent_pool"])
    if not pool.get("enabled"):
        return ()
    if load_policy is None:
        from guandan_marvel.evaluation.checkpoints import load_v2_policy as load_policy  # type: ignore[assignment]
    weights = dict(pool.get("weights") or {})
    entries: list[dict[str, object]] = []
    for name in pool["entries"]:
        path = Path(model_paths[name]).resolve()
        artifact = load_policy(str(path), device="cpu", expected_algorithm="ippo")
        if artifact.metadata.get("training_resume_allowed") is not False:
            raise PlanError(f"pool entry {name!r} is not evaluation-only")
        if artifact.architecture != str(plan["arms"][arm]["model"]["architecture"]):
            raise PlanError(f"pool entry {name!r} architecture differs from the arm")
        entries.append({
            "id": name,
            "kind": "v2_evaluation",
            "checkpoint": str(path),
            "checkpoint_sha256": str(plan["models"][name]["sha256"]),
            "architecture": artifact.architecture,
            "protocol_version": artifact.protocol_version,
            "backend": artifact.backend,
            "resolved_config_hash": artifact.resolved_config_hash,
            "source_hash": str(artifact.metadata.get("source_hash") or ""),
            "policy_version": int(plan["models"][name].get("iteration", 0)),
            "weight": float(weights.get(name, 1.0)),
            "active": True,
            "tags": ["recent"],
        })
    return tuple(entries)


def _save_recovery(trainer, run_dir: Path) -> Path:
    """Persist one recovery under ``run_dir`` and publish its portable index.

    ``run_dir`` is normalized to an absolute path so a relative caller and an
    absolute caller produce the same on-disk layout; the index keeps a path
    *relative* to the run root so the recovery bundle stays portable.  For DDP
    the whole save (shards, manifest, index) is one all-rank transaction: see
    ``guandan_marvel.kaggle.checkpoints.save_distributed_checkpoint``.
    """
    from guandan_marvel.kaggle.checkpoints import save_distributed_checkpoint

    run_dir = Path(run_dir).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = f"iter{int(trainer.iteration):06d}"
    if getattr(trainer, "distributed", None) is not None and int(getattr(trainer, "world_size", 1)) > 1:
        destination = (checkpoint_dir / name).resolve()
        if destination.parent != checkpoint_dir:
            raise ValueError(f"refusing to save a recovery outside the run root: {destination}")
        # The index helper requires the absolute destination and stores it
        # relative to the run root; it runs inside the save transaction so an
        # index failure fails every rank instead of only rank 0.
        save_distributed_checkpoint(
            trainer, destination, context=trainer.distributed, index_root=run_dir,
        )
        return destination
    destination = (checkpoint_dir / f"{name}.pt").resolve()
    if destination.parent != checkpoint_dir:
        raise ValueError(f"refusing to save a recovery outside the run root: {destination}")
    trainer.save_checkpoint(destination)
    entry = {
        "iteration": int(trainer.iteration),
        "checkpoint": destination.relative_to(run_dir).as_posix(),
        "sha256": sha(destination),
    }
    index_path = run_dir / "checkpoint_manifest.json"
    history: list[dict[str, object]] = []
    if index_path.is_file():
        try:
            history = list(read_json(index_path).get("history") or [])
        except (OSError, ValueError):
            history = []
    history = [row for row in history if int(row.get("iteration", -1)) != entry["iteration"]] + [entry]
    atomic_json(index_path, {
        "schema": "guandan-serial-compare-single-process-checkpoint-index-v1",
        "latest": entry,
        "history": sorted(history, key=lambda row: int(row["iteration"])),
    })
    return destination


DIAGNOSTIC_METRIC_PREFIXES = (
    "macro/", "health/", "grad_norm/", "clip_frac_by_length/", "runtime/", "sampling/",
)
TELEMETRY_SCHEMA = "guandan-serial-compare-telemetry-v1"


def _metric_row(metrics: Mapping[str, object], *, iteration: int, seconds: float, learner_updates: int) -> dict[str, object]:
    row: dict[str, object] = {
        "iteration": int(iteration),
        "seconds": float(seconds),
        "learner_updates": int(learner_updates),
        "kl_stopped": (
            bool(dict(metrics.get("macro/kl_control", {})).get("stopped"))
            if isinstance(metrics.get("macro/kl_control"), Mapping)
            else bool(metrics.get("macro/kl_control/stopped", False))
        ),
    }
    for key, value in metrics.items():
        if isinstance(key, str) and key.startswith(DIAGNOSTIC_METRIC_PREFIXES):
            if isinstance(value, (int, float, bool)):
                row[key] = value
    return row


def write_arm_diagnostics(
    run_dir: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    arm: str,
    ntp_enabled: bool,
    ranks: int,
    context_provenance: Mapping[str, object] | None = None,
    rank: int = 0,
) -> dict[str, object]:
    """Write the low-frequency section-7 diagnostics for one arm.

    Runs no probe that changes RNG or module mode here (the optional fuse probe
    needs a live rollout and is reported unavailable otherwise).  Files are
    per-rank so two DDP ranks never race on the same atomic temp path.
    """
    from .diagnostics import build_arm_diagnostics, write_diagnostics

    payload = build_arm_diagnostics(
        rows, arm=arm, ntp_enabled=ntp_enabled, ranks=ranks,
        context_provenance=context_provenance,
    )
    iteration = int(rows[-1]["iteration"]) if rows else 0
    directory = Path(run_dir) / "diagnostics"
    write_diagnostics(directory / f"iter{iteration:06d}.rank{int(rank):03d}.json", payload)
    if int(rank) == 0:
        write_diagnostics(directory / "latest.json", payload)
    return payload


def write_arm_telemetry(
    run_dir: str | Path,
    *,
    arm: str,
    rank: int,
    world_size: int,
    device: str,
    rows: Sequence[Mapping[str, object]],
    wall_seconds: float,
    learner_updates: int,
) -> Path:
    """Per-rank GPU peak / timing / throughput record (never fabricated)."""
    payload: dict[str, object] = {
        "schema": TELEMETRY_SCHEMA,
        "arm": str(arm),
        "rank": int(rank),
        "world_size": int(world_size),
        "device": str(device),
        "iterations": int(len(rows)),
        "learner_updates": int(learner_updates),
        "wall_seconds": float(wall_seconds),
        "iteration_p95_seconds": None,
        "gpu_name": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "games_per_second": None,
        "current_macros_per_second": None,
    }
    from .budget import p95_seconds

    payload["iteration_p95_seconds"] = p95_seconds([float(row.get("seconds", 0.0)) for row in rows])
    try:  # pragma: no cover - target-only when CUDA is present
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = int(str(device).split(":")[1]) if ":" in str(device) else 0
            payload["gpu_name"] = torch.cuda.get_device_name(index)
            payload["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(index))
            payload["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(index))
    except Exception:  # noqa: BLE001 - telemetry must never break training
        pass
    if rows:
        last = dict(rows[-1])
        for key in ("runtime/games_per_second", "sampling/games_per_second", "runtime/completed_games_per_second"):
            if isinstance(last.get(key), (int, float)) and not isinstance(last.get(key), bool):
                payload["games_per_second"] = float(last[key])
                break
        for key in ("runtime/current_macros_per_second", "sampling/current_macros_per_second", "macro/current_macros_per_second"):
            if isinstance(last.get(key), (int, float)) and not isinstance(last.get(key), bool):
                payload["current_macros_per_second"] = float(last[key])
                break
    payload["hash"] = canonical({key: value for key, value in payload.items() if key != "hash"})
    directory = Path(run_dir) / "telemetry"
    atomic_json(directory / f"rank{int(rank):03d}.json", payload)
    return directory / f"rank{int(rank):03d}.json"


def _write_shared_json(
    run_dir: str | Path, name: str, payload: object, *, world_size: int, rank: int,
) -> None:
    """Write a per-run JSON without two ranks racing on the same temp path.

    A multi-rank arm writes ``<stem>.rankNNN.json`` per rank; rank 0 also
    publishes the canonical name for tooling.  ``policy_import_manifest`` is
    rank-specific (its ``rng_state_hash`` differs), so the canonical copy is
    explicitly rank 0's.
    """
    directory = Path(run_dir)
    if int(world_size) > 1:
        stem = name[:-5] if name.endswith(".json") else name
        atomic_json(directory / f"{stem}.rank{int(rank):03d}.json", payload)
        if int(rank) == 0:
            atomic_json(directory / name, payload)
    else:
        atomic_json(directory / name, payload)


def _dist_active(world_size: int) -> bool:
    if int(world_size) <= 1:
        return False
    import torch.distributed as dist

    return bool(dist.is_available() and dist.is_initialized())


def _collective_device(device: str):
    import torch
    import torch.distributed as dist

    if str(dist.get_backend()).lower() == "nccl":
        if ":" in str(device):
            return torch.device(str(device))
        index = int(dist.get_rank()) % max(1, int(torch.cuda.device_count()))
        return torch.device("cuda", index)
    return torch.device("cpu")


def _collective_any(flag: bool, *, world_size: int, device: str) -> bool:
    """All-reduce a boolean decision (MAX).  No collective when world_size==1."""
    if not _dist_active(world_size):
        return bool(flag)
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([1 if flag else 0], dtype=torch.int32, device=_collective_device(device))
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(int(tensor.item()))


def _assert_same_iteration(iteration: int, *, world_size: int, device: str) -> None:
    if not _dist_active(world_size):
        return
    import torch.distributed as dist

    gathered: list[object] = [None] * int(world_size)
    dist.all_gather_object(gathered, int(iteration))
    if len(set(gathered)) != 1:
        raise RuntimeError(
            f"DDP ranks disagree on the training iteration before the final save: {gathered}; "
            "refusing to save a misaligned recovery"
        )


def run_training_arm(
    *,
    plan: Mapping[str, object],
    arm: str,
    run_dir: str | Path,
    source_payload: Mapping[str, object],
    source_model_path: str | Path,
    model_paths: Mapping[str, str | Path],
    game,
    device: str,
    ddp_enabled: bool,
    world_size: int,
    clock: Callable[[], float] = time.monotonic,
    max_seconds: float,
    save_interval_seconds: float = 600.0,
    trainer_factory=None,
    import_policy=None,
    initial_state_fingerprint=None,
    resume_from: str | Path | None = None,
    iteration_limit: int | None = None,
    seed_offset: int = 0,
    hard_reserve_seconds: float = 0.0,
    config_override=None,
    pool_entries_override: tuple[Mapping[str, object], ...] | None = None,
    rank: int = 0,
    policy_exporter=None,
) -> dict[str, object]:
    """Train one arm from its own fresh initialization (never from the other arm).

    Returns a machine-readable arm result.  ``attempted``/``applied`` updates and
    the KL stop counters are read back from the trainer's own metrics, so an arm
    that produced zero effective updates is reported as a failure rather than as
    a successful run — unless it was *resumed already at its target iteration*
    (an export/finalize-only recovery) which legitimately applies zero updates.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = dict((plan["arms"] or {})[arm])
    # The worker must prove it trains against the exact 375 artifact named by
    # the plan, not merely a path that happens to exist.
    expected_source_sha = str(plan["models"]["p375"]["sha256"])
    source_path = Path(source_model_path)
    if not source_path.is_file():
        raise PlanError(f"the 375 source policy is missing: {source_path}")
    actual_source_sha = sha(source_path)
    if actual_source_sha != expected_source_sha:
        raise PlanError(
            f"the 375 source policy hash {actual_source_sha} != plan {expected_source_sha}"
        )
    if config_override is not None:
        config = config_override
    else:
        entries = (
            tuple(pool_entries_override) if pool_entries_override is not None
            else build_pool_entries(plan, arm, model_paths)
        )
        config = build_train_config(
            plan, arm, game=game, source_payload=source_payload, device=device,
            ddp_enabled=ddp_enabled, world_size=world_size, opponent_entries=entries,
            seed_offset=seed_offset,
        )
    _write_shared_json(
        run_dir, "resolved_config.json", {"train_config": config.to_dict()},
        world_size=int(world_size), rank=int(rank),
    )
    if trainer_factory is None:
        from guandan_marvel.training.local import IPPOTrainer

        factory = lambda cfg: IPPOTrainer(cfg, game=game, run_dir=run_dir)
    else:
        factory = trainer_factory
    trainer = factory(config)
    import_manifest = None
    if resume_from is not None:
        from guandan_marvel.kaggle.checkpoints import load_distributed_checkpoint

        if int(getattr(trainer, "world_size", 1)) > 1:
            load_distributed_checkpoint(trainer, resume_from, context=trainer.distributed)
        else:
            trainer.load_checkpoint(resume_from)
        resume_mode = "exact_resume"
    else:
        if import_policy is None:
            from guandan_marvel.training.policy_import import import_policy_as_fresh_run as import_policy  # type: ignore[assignment]
        drops = () if arm == "control" else ("ntp_head.",)
        import_manifest = import_policy(
            trainer, source_payload,
            source_role=f"{arm}:p375",
            allowed_drop_prefixes=drops,
            curriculum_offset=int(spec.get("curriculum_offset", 0)),
        )
        _write_shared_json(
            run_dir, "policy_import_manifest.json", import_manifest,
            world_size=int(world_size), rank=int(rank),
        )
        resume_mode = "fresh_import"
    fingerprint_path = run_dir / "initial_state_fingerprint.json"
    if initial_state_fingerprint is None:
        if resume_from is not None and fingerprint_path.is_file():
            # A resumed arm must keep its *original* initial fingerprint: the
            # current weights are progressed, not the initial policy.
            fingerprint_block = read_json(fingerprint_path)
        else:
            from guandan_marvel.training.policy_import import initial_state_fingerprint as fingerprint

            fingerprint_block = fingerprint(trainer, include_ntp=False)
    else:
        fingerprint_block = initial_state_fingerprint(trainer)
    _write_shared_json(
        run_dir, "initial_state_fingerprint.json", fingerprint_block,
        world_size=int(world_size), rank=int(rank),
    )

    target_iterations = int(spec["max_new_iterations"])
    start_iteration = int(trainer.iteration)
    if iteration_limit is not None:
        # "Train N more iterations from the restored point" (preflight/resume).
        target_iterations = min(target_iterations, start_iteration + int(iteration_limit))
    elif resume_from is not None:
        target_iterations = max(target_iterations, start_iteration)
    start_learner_updates = int(trainer.learner_updates)
    ntp_enabled = float(spec["ntp_weight"]) > 0.0
    context_provenance = {
        "training_context_schedule_hash": canonical(config.to_dict().get("training_context_schedule")),
        "training_context_curriculum_hash": canonical(config.to_dict().get("training_context_curriculum")),
        "curriculum_offset": int(spec.get("curriculum_offset", 0)),
        "seed": int(config.seed),
        "seed_offset": int(seed_offset),
        "training_data_generator_version": str(config.training_data_generator_version),
    }
    from .budget import fits_next_iteration

    # All ranks must enter the loop with the same iteration, otherwise the very
    # first collective (stop decision) would be called by a different set of
    # ranks and hang.
    _assert_same_iteration(int(trainer.iteration), world_size=int(world_size), device=str(device))

    started = float(clock())
    last_save = started
    save_seconds = 0.0
    iterations: list[dict[str, object]] = []
    stop_reason = "target_reached"
    world_size = int(world_size)
    while int(trainer.iteration) < target_iterations:
        elapsed_wall = float(clock()) - started
        local_stop = elapsed_wall >= float(max_seconds)
        if not local_stop:
            # p95 + save + hard reserve, *before* each iteration: never discover
            # a reserve breach only after the arm has already blown its quota.
            local_stop = not fits_next_iteration(
                interval_seconds=[float(row["seconds"]) for row in iterations],
                remaining_seconds=float(max_seconds) - elapsed_wall,
                save_seconds=save_seconds,
                reserve_seconds=float(hard_reserve_seconds),
            )
        # A save (barrier + all_gather) and train_iteration (all_reduce) are
        # different collectives: "stop now" must be a group decision, not a
        # per-rank wall clock, or one rank enters the save barrier while the
        # other enters the next iteration's all_reduce.
        if _collective_any(local_stop, world_size=world_size, device=str(device)):
            stop_reason = "budget_stopped"
            break
        iteration_started = float(clock())
        metrics = trainer.train_iteration()
        elapsed = float(clock()) - iteration_started
        iterations.append(_metric_row(
            metrics, iteration=int(trainer.iteration), seconds=elapsed,
            learner_updates=int(metrics.get("statistics/learner_updates", 0)),
        ))
        local_due = float(clock()) - last_save >= float(save_interval_seconds)
        # Any rank being due makes *every* rank save in the same iteration.
        if _collective_any(local_due, world_size=world_size, device=str(device)):
            save_started = float(clock())
            _save_recovery(trainer, run_dir)
            save_seconds = float(clock()) - save_started
            last_save = float(clock())
            write_arm_diagnostics(
                run_dir, iterations, arm=arm, ntp_enabled=ntp_enabled,
                ranks=world_size, context_provenance=context_provenance, rank=int(rank),
            )
            write_arm_telemetry(
                run_dir, arm=arm, rank=int(rank), world_size=world_size, device=str(device),
                rows=iterations, wall_seconds=float(clock()) - started,
                learner_updates=int(trainer.learner_updates) - start_learner_updates,
            )
    # Every rank must have executed the same number of iterations before the
    # final (collective) save.  A mismatch is a bug, never something to paper
    # over by saving a misaligned recovery.
    _assert_same_iteration(int(trainer.iteration), world_size=world_size, device=str(device))
    recovery = _save_recovery(trainer, run_dir)
    final_iteration = int(trainer.iteration)
    applied = int(trainer.learner_updates) - start_learner_updates
    resumed_at_target = (
        resume_from is not None and not iterations and start_iteration >= target_iterations
    )
    if applied <= 0 and not resumed_at_target:
        raise RuntimeError(
            f"arm {arm!r} produced zero applied optimizer updates; refusing to report a successful run"
        )
    diagnostics = write_arm_diagnostics(
        run_dir, iterations, arm=arm, ntp_enabled=ntp_enabled,
        ranks=int(world_size), context_provenance=context_provenance, rank=int(rank),
    )
    telemetry_path = write_arm_telemetry(
        run_dir, arm=arm, rank=int(rank), world_size=int(world_size), device=str(device),
        rows=iterations, wall_seconds=float(clock()) - started,
        learner_updates=applied,
    )
    policy_path = run_dir / f"policy_{arm}_iter{final_iteration:06d}.pt"
    if policy_exporter is not None:
        policy_exporter(trainer, policy_path, final_iteration)
    elif getattr(trainer, "is_primary", True):
        from guandan_marvel.evaluation.checkpoints import freeze_merged_policy

        freeze_merged_policy(trainer, policy_path, iteration=final_iteration)
    return {
        "schema": ARM_RUNTIME_SCHEMA,
        "arm": arm,
        "status": "complete" if stop_reason == "target_reached" else stop_reason,
        "stop_reason": stop_reason,
        "resume_mode": resume_mode,
        "resumed_at_target": bool(resumed_at_target),
        "start_iteration": start_iteration,
        "final_iteration": final_iteration,
        "new_iterations": final_iteration - start_iteration,
        "applied_updates": applied,
        "iterations": iterations,
        "wall_seconds": float(clock()) - started,
        "recovery": str(recovery),
        "policy": str(policy_path),
        "policy_sha256": sha(policy_path) if policy_path.is_file() else None,
        "initial_source_sha256": expected_source_sha,
        "source_model_sha256": actual_source_sha,
        "initial_state_fingerprint": fingerprint_block,
        "opponent_pool_manifest_hash": (
            None if getattr(trainer, "opponent_pool", None) is None
            else str(trainer.opponent_pool.manifest_hash)
        ),
        "target_kl": None if spec.get("target_kl") is None else float(spec["target_kl"]),
        "ntp_enabled": bool(ntp_enabled),
        "import_manifest_hash": None if import_manifest is None else import_manifest.get("hash"),
        "diagnostics_path": str(Path(run_dir) / "diagnostics" / "latest.json"),
        "diagnostics_hash": diagnostics.get("hash"),
        "telemetry_path": str(telemetry_path),
        "record_hash": canonical({
            "arm": arm, "final_iteration": final_iteration, "applied_updates": applied,
            "policy_sha256": sha(policy_path) if policy_path.is_file() else None,
        }),
    }


def verify_arm_pair(control: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, object]:
    """Prove both arms start from the same 375 policy and share no weights."""
    control_fp = control.get("initial_state_fingerprint") or {}
    candidate_fp = candidate.get("initial_state_fingerprint") or {}
    if control_fp.get("hash") != candidate_fp.get("hash"):
        raise PlanError("the two arms did not start from a bitwise-identical shared policy")
    if candidate.get("initial_source_sha256") == control.get("policy_sha256"):
        raise PlanError("the candidate started from the control's final policy")
    if control.get("initial_source_sha256") != candidate.get("initial_source_sha256"):
        raise PlanError("the two arms declare different initial source checkpoints")
    return {
        "shared_initial_fingerprint": control_fp.get("hash"),
        "tensor_count": int(control_fp.get("tensor_count", 0)),
        "shared_bitwise": True,
        "candidate_not_initialised_from_control": True,
    }


__all__ = [
    "ARM_RUNTIME_SCHEMA", "build_pool_entries", "build_train_config",
    "run_training_arm", "verify_arm_pair",
]
