"""Profile-driven A05 training and exact checkpoint continuation.

Shared collection is not algorithm equivalence: IPPO uses GAE/V regression;
VRPO uses a separate full-state action-value critic and Q-boosting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
import os
import uuid

import numpy as np
import torch

from ..cards import Rank
from ..env_batch import GuandanEnvBatch
from ..environment_types import GameConfig
from ..model import PolicyValueNet
from .checkpoint import (CURRENT_PROTOCOL_VERSIONS, IncompatibleCheckpointError,
                         load_checkpoint, save_checkpoint)
from .collector import Deadline, IncompleteTrainingError
from .critic import CentralQCritic
from .runtime import (BudgetExceeded, TRAIN_SEED_BASE, file_sha256, resolved_profile, runtime_metadata, write_evidence)
from .trainer_core import TrainingEngine


@dataclass(frozen=True)
class TrainingResult:
    algorithm: str
    profile: str
    updates: int                         # completed in THIS invocation
    total_token_steps: int               # real samples in THIS invocation
    last_loss: float | None
    checkpoint: str | None
    status: str = "complete"
    lifetime_updates: int = 0
    lifetime_token_steps: int = 0
    start_update: int = 0
    target_updates: int = 0
    rollout_envs: int = 0
    token_steps_per_update: int = 0
    profile_counts_match: bool = False
    elapsed_seconds: float = 0.0
    metrics: list[dict] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)
    resumed_from: str | None = None
    resume_sha256: str | None = None
    runtime: dict = field(default_factory=dict)
    reason: str | None = None
    evidence: str | None = None


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be positive integer")
    return value


def train(*, algorithm="ippo", profile="local_fast", device=None, updates=None,
          checkpoint_dir="runs", resume=None, hidden_dim=32, seed=TRAIN_SEED_BASE,
          rollout_steps=None, rollout_envs=None, deadline=None, max_hours=None,
          epochs=2, gamma=1.0, gae_lambda=0.95):
    """updates is additional updates on resume; overrides are explicitly non-profile tests.

    rollout_steps retains legacy per-environment ticks only for focused tests.
    Acceptance never overrides profile env/tick counts. A shared deadline can
    cover both base execution and resumed execution without resetting the clock.
    """
    started = time.monotonic()
    if algorithm not in {"ippo", "vrpo"}:
        raise ValueError("algorithm must be ippo or vrpo")
    configured = resolved_profile(profile)
    cfg = configured["config"]["training"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _positive(hidden_dim, "hidden_dim")
    count = cfg["rollout_envs"] if rollout_envs is None else _positive(rollout_envs, "rollout_envs")
    aggregate = cfg["token_steps_per_update"] if rollout_steps is None else count * _positive(rollout_steps, "rollout_steps")
    if aggregate % count:
        raise ValueError("aggregate token steps must be divisible by env count")
    target = (cfg["resume_updates"] if resume else cfg["updates_per_algorithm"]) if updates is None else _positive(updates, "updates")
    budget = deadline or Deadline(cfg["max_hours"] if max_hours is None else max_hours)
    requested = device or configured["config"]["device"]
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    chosen = torch.device(requested)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)  # Runtime cost only; does not change profile quantities.
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    metadata = runtime_metadata(str(chosen))
    metadata["scope"] = {"level_rank": int(Rank.FIVE), "previous_result": None, "pure_python": True}
    model_cfg = {
        "architecture": "mean-pooled-token-MLP", "hidden_dim": hidden_dim,
        "critic": "central-Q-team-antisymmetric-688" if algorithm == "vrpo" else "decentralized-V-shared-teams",
        "vocabulary": 256, "channels": 256,
    }
    compatibility = {
        "profile_sha256": configured["sha256"], "envs": count,
        "token_steps_per_update": aggregate, "seed": seed,
        "epochs": epochs, "gamma": gamma, "gae_lambda": gae_lambda,
        "level_rank": int(Rank.FIVE),
    }
    directory = Path(checkpoint_dir).resolve() / profile / algorithm
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    report_path = directory / f"execution_{run_id}.json"
    model = PolicyValueNet(hidden_dim=hidden_dim).to(chosen)
    critic = CentralQCritic(hidden_dim=hidden_dim).to(chosen) if algorithm == "vrpo" else None
    modules = torch.nn.ModuleDict({"actor": model, **({"critic": critic} if critic is not None else {})})
    optimizer = torch.optim.Adam(modules.parameters(), lr=3e-4)
    batch = GuandanEnvBatch([GameConfig(seed=seed+i, level_rank=Rank.FIVE) for i in range(count)], auto_reset=False)
    engine = None
    initial_updates = initial_steps = 0
    metrics, checkpoints = [], []
    latest_checkpoint = None
    reason, status, resume_digest = None, "complete", None
    try:
        budget.check(stage="initialize")
        loaded = None
        if resume:
            resume_digest = file_sha256(resume)
            loaded = load_checkpoint(resume, model=modules, optimizer=optimizer, expected_profile=profile,
                                     expected_algorithm=algorithm, expected_model_config=model_cfg, map_location=chosen)
            if loaded.training_state.get("compatibility") != compatibility:
                raise IncompatibleCheckpointError("training profile/hyperparameters/seed incompatible on resume")
            extra = loaded.training_state.get("engine")
            if not isinstance(extra, dict):
                raise IncompatibleCheckpointError("checkpoint lacks full engine resume state")
            batch.load_serialized(extra["collector"]["envs"])
        else:
            batch.reset(seeds=[seed+i for i in range(count)])
        budget.check(stage="initialize_complete")
        engine = TrainingEngine(algorithm, batch, model, optimizer, critic=critic, device=chosen,
                                seed=seed, epochs=epochs, gamma=gamma, gae_lambda=gae_lambda)
        if loaded is not None:
            engine.load_extra_state(loaded.training_state["engine"])
            if engine.update_count != loaded.metadata.update_count:
                raise IncompatibleCheckpointError("checkpoint update counters disagree")
        initial_updates, initial_steps = engine.update_count, engine.total_token_steps
        for _ in range(target):
            budget.check(stage="before_update", completed_updates=engine.update_count-initial_updates)
            row = engine.collect_and_update(aggregate, deadline=budget)
            metrics.append(row)
            if row["token_steps"] != aggregate or row["rollout_envs"] != count or row["rollout_ticks"] != aggregate // count:
                raise RuntimeError("actual rollout counters do not match requested profile")
            budget.check(stage="after_update")
            if engine.update_count % cfg["checkpoint_interval"] == 0 or len(metrics) == target:
                path = directory / f"{algorithm}_update_{engine.update_count:06d}_{run_id}.pt"
                saved = save_checkpoint(path, model=modules, optimizer=optimizer, profile=profile,
                                        algorithm=algorithm, update=engine.update_count, seed=seed,
                                        model_config=model_cfg, protocol_versions=CURRENT_PROTOCOL_VERSIONS,
                                        training_state={"compatibility": compatibility, "engine": engine.extra_state()})
                latest_checkpoint = str(path)
                checkpoints.append({"path": str(path), "sha256": file_sha256(path), "update": saved.update_count,
                                    "token_steps": engine.total_token_steps})
            budget.check(stage="checkpoint_complete")
    except (IncompleteTrainingError, BudgetExceeded) as exc:
        status, reason = "incomplete", str(exc)
    except Exception as exc:
        write_evidence(report_path, {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                                    "profile": configured, "algorithm": algorithm, "runtime": metadata,
                                    "completed_updates": len(metrics), "metrics": metrics,
                                    "actual_token_steps": 0 if engine is None else engine.total_token_steps-initial_steps})
        raise
    lifetime_updates = initial_updates if engine is None else engine.update_count
    lifetime_steps = initial_steps if engine is None else engine.total_token_steps
    expected_updates = cfg["resume_updates"] if resume else cfg["updates_per_algorithm"]
    result = TrainingResult(
        algorithm, profile, lifetime_updates-initial_updates, lifetime_steps-initial_steps,
        metrics[-1].get("loss") if metrics else None, latest_checkpoint, status, lifetime_updates,
        lifetime_steps, initial_updates, target, count, aggregate,
        count == cfg["rollout_envs"] and aggregate == cfg["token_steps_per_update"] and target == expected_updates,
        time.monotonic()-started, metrics, checkpoints, None if resume is None else str(Path(resume).resolve()),
        resume_digest, metadata, reason, str(report_path),
    )
    write_evidence(report_path, {**asdict(result), "resolved_profile": configured,
                                "actual_ticks_per_env": [] if engine is None else engine.collector.ticks_per_env.tolist(),
                                "actual_resets_per_env": [] if engine is None else engine.reset_counts.tolist(),
                                "terminal_count": 0 if engine is None else engine.collector.terminal_count,
                                "truncation_count": 0 if engine is None else engine.collector.truncation_count})
    return result
