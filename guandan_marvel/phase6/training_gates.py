# SPDX-License-Identifier: MIT
"""Phase-6 formal training-gate tools.

These tools are *implemented but never executed* by phase 6.  The formal gate
targets are absolute iteration counts (50/250/500/1000/1500); a resume computes
``remaining = max(0, target - actual_counter)`` and never adds a new target.
Every runner defaults to dry-run, refuses a formal target without the explicit
confirmation token, and enforces ``max_iterations``/``max_games``/
``max_wall_seconds`` as hard caps.  Reaching a budget writes an
iteration-boundary checkpoint and a recovery record instead of relying on an
external kill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from .design import (
    FORMAL_CONFIRMATION_TOKEN, FORMAL_GATE_GAMES_AT_20_PER_ITER,
    FORMAL_GATE_TARGETS, GATE_EVIDENCE_SCHEMA, GATE_PLAN_SCHEMA, GATE_RUN_SCHEMA,
    TRAINING_GATE_PLAN, canonical_hash,
)

FORMAL_GATES_NOT_RUN = True


class TrainingGateError(RuntimeError):
    """A gate plan or run violates the pre-registered contract."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def absolute_target_iterations(gate: int) -> int:
    try:
        return int(FORMAL_GATE_TARGETS[int(gate)])
    except KeyError as error:
        raise TrainingGateError(f"unknown formal gate {gate!r}") from error


@dataclass(frozen=True)
class GateBudget:
    """Absolute-target budget for one gate run."""

    gate: int
    completed_iterations: int = 0
    completed_games: int = 0
    target_games: int | None = None
    max_iterations: int | None = None
    max_games: int | None = None
    max_wall_seconds: float | None = None
    dry_run: bool = True
    formal_confirmation: str | None = None
    smoke: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.gate, bool) or int(self.gate) not in FORMAL_GATE_TARGETS:
            raise TrainingGateError(f"gate must be one of {sorted(FORMAL_GATE_TARGETS)}")
        for name in ("completed_iterations", "completed_games"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise TrainingGateError(f"{name} must be a non-negative integer")
        if self.target_games is not None and (
            isinstance(self.target_games, bool) or int(self.target_games) < 0
        ):
            raise TrainingGateError("target_games must be a non-negative integer or None")
        for name in ("max_iterations", "max_games"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) <= 0):
                raise TrainingGateError(f"{name} must be positive or None")
        if self.max_wall_seconds is not None:
            value = float(self.max_wall_seconds)
            if not math.isfinite(value) or value <= 0:
                raise TrainingGateError("max_wall_seconds must be finite and positive")
        if self.max_iterations is None and self.max_games is None and self.max_wall_seconds is None:
            raise TrainingGateError(
                "a gate run requires at least one hard limit "
                "(max_iterations/max_games/max_wall_seconds)"
            )
        if not isinstance(self.dry_run, bool):
            raise TypeError("dry_run must be bool")
        if not isinstance(self.smoke, bool):
            raise TypeError("smoke must be bool")
        if self.smoke and int(self.gate) in FORMAL_GATE_TARGETS:
            # Smoke runs may only use tiny explicit limits; they are never a
            # claim that a formal gate was reached.
            if self.max_iterations is None or int(self.max_iterations) > 2:
                raise TrainingGateError(
                    "a smoke gate run is limited to 1-2 iterations and must be labelled smoke"
                )

    @property
    def target_iterations(self) -> int:
        return absolute_target_iterations(self.gate)

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.target_iterations - int(self.completed_iterations))

    @property
    def remaining_games(self) -> int | None:
        if self.target_games is None:
            return None
        return max(0, int(self.target_games) - int(self.completed_games))


def plan_gate_run(budget: GateBudget) -> dict[str, object]:
    """Dry-run plan.  It never constructs a trainer or reads a checkpoint."""
    remaining = budget.remaining_iterations
    if remaining == 0:
        status = "already_satisfied"
    elif budget.dry_run:
        status = "dry_run"
    else:
        status = "ready"
    if not budget.dry_run and budget.formal_confirmation != FORMAL_CONFIRMATION_TOKEN:
        status = "confirmation_required"
    plan: dict[str, object] = {
        "schema": GATE_PLAN_SCHEMA,
        "gate": int(budget.gate),
        "target_iterations": budget.target_iterations,
        "target_games_estimate_at_20_per_iteration": FORMAL_GATE_GAMES_AT_20_PER_ITER[
            int(budget.gate)
        ],
        "target_games_configured": budget.target_games,
        "completed_iterations": int(budget.completed_iterations),
        "completed_games": int(budget.completed_games),
        "remaining_iterations": remaining,
        "remaining_games": budget.remaining_games,
        "max_iterations": budget.max_iterations,
        "max_games": budget.max_games,
        "max_wall_seconds": budget.max_wall_seconds,
        "dry_run": bool(budget.dry_run),
        "smoke": bool(budget.smoke),
        "status": status,
        "will_train": status == "ready",
        "created_at": _timestamp(),
    }
    plan["hash"] = canonical_hash(plan)
    return plan


@dataclass
class GateRunner:
    """Run a gate budget against a trainer factory, with hard caps.

    ``trainer_factory`` is called only when the plan says ``will_train``.  The
    returned trainer must expose ``train_iteration()``, ``iteration``,
    ``completed_games_total`` and, for recovery, ``save_checkpoint(path)``.
    ``counter_reader``/``recovery_loader`` let the caller supply the *actual*
    iteration counter and a same-device reload check instead of trusting the
    CLI's declared counter.
    """

    trainer_factory: Callable[..., Any]
    budget: GateBudget
    checkpoint_path: str | Path
    output_dir: str | Path
    trainer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    counter_reader: Callable[[], int] | None = None
    strength_evidence: Mapping[str, object] | None = None
    recovery_loader: Callable[[Path], Any] | None = None

    def _actual_counter(self, checkpoint: Path) -> tuple[int | None, str]:
        if self.counter_reader is not None:
            return int(self.counter_reader()), "counter_reader"
        if checkpoint.is_file():
            try:
                import torch

                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001 - an unreadable checkpoint is not a counter
                return None, "unreadable_checkpoint"
            if isinstance(payload, Mapping) and "iteration" in payload:
                try:
                    return int(payload["iteration"]), "checkpoint"
                except (TypeError, ValueError):
                    return None, "malformed_checkpoint_iteration"
        return None, "none"

    def run(self) -> dict[str, object]:
        plan = plan_gate_run(self.budget)
        started = time.perf_counter()
        checkpoint = Path(self.checkpoint_path)
        if plan["status"] != "ready":
            status = str(plan["status"])
            actual: int | None = None
            counter_source = "none"
            counter_verified = False
            remaining = int(plan["remaining_iterations"])
            if status == "already_satisfied":
                actual, counter_source = self._actual_counter(checkpoint)
                if actual is None:
                    status = "already_satisfied_unverified"
                else:
                    counter_verified = True
                    remaining = max(0, int(plan["target_iterations"]) - int(actual))
                    if remaining > 0:
                        status = "counter_mismatch"
            record = {
                "schema": GATE_RUN_SCHEMA,
                "status": status,
                "plan_hash": plan["hash"],
                "gate": plan["gate"],
                "dry_run": bool(self.budget.dry_run),
                "trainer_constructed": False,
                "trained_iterations": 0,
                "trained_games": 0,
                "declared_completed_iterations": int(self.budget.completed_iterations),
                "actual_completed_iterations": actual,
                "counter_source": counter_source,
                "counter_verified": counter_verified,
                "remaining_iterations": remaining,
                "checkpoint": None,
                "wall_seconds": time.perf_counter() - started,
                "smoke": bool(self.budget.smoke),
                "budget_exceeded": False,
                "health": {"status": "NOT_RUN", "passed": None},
                "strength": {"recommendation": "NOT_RUN", "auto_expand": False},
                "recovery": {"status": "NOT_RUN", "verified": False},
                "formal_gate_reached": False,
                "formal_gates_not_run": FORMAL_GATES_NOT_RUN,
                "created_at": _timestamp(),
            }
            record["hash"] = canonical_hash(record)
            return record
        kwargs = dict(self.trainer_kwargs)
        if checkpoint.is_file():
            kwargs["resume_from"] = str(checkpoint)
        trainer = self.trainer_factory(**kwargs)
        trained_iterations = 0
        trained_games = 0
        last_iteration_games: int | None = None
        budget_exceeded = False
        stop_reason = "target_reached"
        last_checkpoint: Path | None = None
        while True:
            if int(trainer.iteration) >= plan["target_iterations"]:
                stop_reason = "target_reached"
                break
            remaining_games = plan["remaining_games"]
            if (
                self.budget.max_games is not None
                and last_iteration_games is not None
                and trained_games + int(last_iteration_games) > int(self.budget.max_games)
            ):
                # The previous iteration's game count is known; stop before an
                # iteration that would cross a hard game cap.
                stop_reason = "max_games"
                break
            if self.budget.max_wall_seconds is not None and (
                time.perf_counter() - started >= float(self.budget.max_wall_seconds)
            ):
                stop_reason = "max_wall_seconds"
                break
            if self.budget.max_iterations is not None and trained_iterations >= int(
                self.budget.max_iterations
            ):
                stop_reason = "max_iterations"
                break
            if remaining_games is not None and trained_games >= remaining_games:
                stop_reason = "target_games_reached"
                break
            games_before = int(trainer.completed_games_total)
            trainer.train_iteration()
            played = int(trainer.completed_games_total) - games_before
            trained_iterations += 1
            trained_games += played
            last_iteration_games = played
            save = getattr(trainer, "save_checkpoint", None)
            if callable(save):
                last_checkpoint = Path(save(checkpoint))
            if int(trainer.iteration) >= plan["target_iterations"]:
                stop_reason = "target_reached"
                break
            if self.budget.max_games is not None and trained_games > int(self.budget.max_games):
                budget_exceeded = True
                stop_reason = "max_games_exceeded"
                break
            if self.budget.max_wall_seconds is not None and (
                time.perf_counter() - started >= float(self.budget.max_wall_seconds)
            ):
                budget_exceeded = True
                stop_reason = "max_wall_seconds_exceeded"
                break
        # Save at the final iteration boundary even when a cap stopped the run.
        save = getattr(trainer, "save_checkpoint", None)
        if callable(save) and last_checkpoint is None:
            last_checkpoint = Path(save(checkpoint))
        iteration = int(trainer.iteration)
        remaining = max(0, plan["target_iterations"] - iteration)
        last_metrics = getattr(trainer, "last_metrics", None)
        if isinstance(last_metrics, Mapping):
            health = evaluate_health(last_metrics)
        else:
            health = {"status": "NOT_RUN", "passed": None}
        strength = (
            {**dict(self.strength_evidence), "auto_expand": False}
            if self.strength_evidence is not None
            else {"recommendation": "NOT_RUN", "auto_expand": False}
        )
        recovery: dict[str, object] = {"status": "NOT_RUN", "verified": False}
        if last_checkpoint is not None and self.recovery_loader is not None:
            try:
                self.recovery_loader(last_checkpoint)
                recovery = {"status": "PASS", "verified": True, "path": str(last_checkpoint)}
            except Exception as error:  # noqa: BLE001 - report, do not hide a failed reload
                recovery = {
                    "status": "FAIL",
                    "verified": False,
                    "error": f"{type(error).__name__}: {error}",
                    "path": str(last_checkpoint),
                }
        record: dict[str, object] = {
            "schema": GATE_RUN_SCHEMA,
            "status": "complete" if remaining == 0 else "budget_stopped",
            "stop_reason": stop_reason,
            "plan_hash": plan["hash"],
            "gate": plan["gate"],
            "dry_run": False,
            "trainer_constructed": True,
            "trained_iterations": int(trained_iterations),
            "trained_games": int(trained_games),
            "completed_iterations": iteration,
            "completed_games_total": int(trainer.completed_games_total),
            "remaining_iterations": int(remaining),
            "remaining_games": (
                None if plan["remaining_games"] is None
                else max(0, int(plan["remaining_games"]) - int(trained_games))
            ),
            "budget_exceeded": bool(budget_exceeded),
            "checkpoint": None if last_checkpoint is None else str(last_checkpoint),
            "checkpoint_written_at_iteration_boundary": last_checkpoint is not None,
            "health": health,
            "strength": strength,
            "recovery": recovery,
            "wall_seconds": time.perf_counter() - started,
            "smoke": bool(self.budget.smoke),
            "formal_gate_reached": bool(
                remaining == 0
                and not self.budget.smoke
                and not budget_exceeded
                and health.get("passed") is True
                and recovery.get("verified") is True
            ),
            "formal_gates_not_run": FORMAL_GATES_NOT_RUN,
            "created_at": _timestamp(),
        }
        record["hash"] = canonical_hash(record)
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._write_record(output / f"gate{int(self.budget.gate)}_run.json", record)
        return record

    @staticmethod
    def _write_record(path: Path, payload: Mapping[str, object]) -> None:
        import json

        temporary = path.with_suffix(path.suffix + f".tmp-{id(payload)}")
        temporary.write_text(
            json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

def evaluate_health(
    metrics: Mapping[str, object],
    *,
    learner_updates: int | None = None,
    plan: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Pre-registered health gate over one iteration's metrics."""
    thresholds = dict((plan or TRAINING_GATE_PLAN)["health"])
    problems: list[str] = []
    nonfinite = 0
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            nonfinite += 1
            problems.append(f"non-finite metric {name}")
    if nonfinite > int(thresholds["max_nonfinite_count"]):
        problems.append(f"nonfinite_count={nonfinite}")
    skipped = metrics.get("runtime/amp_skipped_steps")
    updates = learner_updates
    if updates is None and isinstance(metrics.get("statistics/learner_updates"), int):
        updates = int(metrics["statistics/learner_updates"])
    skipped_fraction = None
    if isinstance(skipped, (int, float)) and updates not in (None, 0):
        skipped_fraction = float(skipped) / float(updates)
        if skipped_fraction > float(thresholds["max_amp_skipped_fraction"]):
            problems.append(f"amp_skipped_fraction={skipped_fraction:.6g}")
    if thresholds.get("require_value_ev_defined"):
        ev = metrics.get("macro/value_ev", metrics.get("agent/value_ev"))
        if ev is None:
            problems.append("value EV is undefined")
        elif float(ev) < float(thresholds["min_value_ev"]):
            problems.append(f"value EV {float(ev):.6g} below {thresholds['min_value_ev']}")
    illegal = metrics.get("macro/illegal_active_action_count")
    if thresholds.get("requires_legal_action_check") and illegal not in (None, 0):
        problems.append(f"illegal_active_action_count={illegal}")
    return {
        "schema": "guandan-marvel-phase6-health-gate-v1",
        "passed": not problems,
        "problems": problems,
        "nonfinite_metric_count": nonfinite,
        "amp_skipped_fraction": skipped_fraction,
        "thresholds": thresholds,
    }


def decide_strength(
    *,
    ci_low: float | None,
    ci_high: float | None,
    health_passed: bool,
    recovery_passed: bool,
) -> dict[str, object]:
    """Fixed decision rule; never expands seeds or deals automatically."""
    if ci_low is None or ci_high is None:
        recommendation = "INCONCLUSIVE"
        reason = "paired CI is undefined"
    elif not health_passed or not recovery_passed:
        recommendation = "INCONCLUSIVE"
        reason = "health or recovery gate failed"
    elif float(ci_low) > 0.0:
        recommendation = "CONTINUE"
        reason = "paired CI lower > 0 and health/recovery passed"
    elif float(ci_high) < 0.0:
        recommendation = "STOP"
        reason = "paired CI upper < 0"
    else:
        recommendation = "INCONCLUSIVE"
        reason = "paired CI straddles zero"
    return {
        "schema": "guandan-marvel-phase6-strength-gate-v1",
        "recommendation": recommendation,
        "reason": reason,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "health_passed": bool(health_passed),
        "recovery_passed": bool(recovery_passed),
        "auto_expand": False,
        "fixed_deals_required": True,
    }


def build_gate_evidence(
    *,
    gate: int,
    run_record: Mapping[str, object],
    health: Mapping[str, object],
    strength: Mapping[str, object],
    recovery: Mapping[str, object],
    evaluation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compose the evidence record a formal gate must attach."""
    payload: dict[str, object] = {
        "schema": GATE_EVIDENCE_SCHEMA,
        "gate": int(gate),
        "absolute_target_iterations": absolute_target_iterations(gate),
        "run_hash": run_record.get("hash"),
        "health": dict(health),
        "strength": dict(strength),
        "recovery": dict(recovery),
        "evaluation": None if evaluation is None else dict(evaluation),
        "formal_gate_reached": bool(run_record.get("formal_gate_reached")),
        "formal_gates_not_run": FORMAL_GATES_NOT_RUN,
        "created_at": _timestamp(),
    }
    payload["hash"] = canonical_hash(payload)
    return payload


__all__ = [
    "FORMAL_GATES_NOT_RUN", "GateBudget", "GateRunner", "TrainingGateError",
    "absolute_target_iterations", "build_gate_evidence", "decide_strength",
    "evaluate_health", "plan_gate_run",
]
