# SPDX-License-Identifier: MIT
"""Cumulative-vs-session budget accounting for the serial comparison.

Two clocks are deliberately separated:

* the **cumulative work budget** (``total_seconds`` from bootstrap) survives a
  restart because it is persisted, and is never reset to a fresh four hours;
* the **session deadline** is ``session_started + total_seconds`` and only
  bounds the current process.

The usable allowance is the safe minimum of the two.  A symmetric reduction of
the two training quotas is allowed *only* before either arm has started formal
training, and it freezes an immutable resolved budget plus its subplan hash.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .common import BUDGET_SCHEMA, PlanError, canonical

PHASE_BUDGET_KEYS = {
    "bootstrap/input_check": "dependencies_and_preflight",
    "preflight/control": "dependencies_and_preflight",
    "preflight/candidate": "dependencies_and_preflight",
    "freeze_execution_budget": "dependencies_and_preflight",
    "train/control": "train_control",
    "train/candidate": "train_candidate",
    "freeze_final_checkpoints": "finalize_reserve",
    "evaluate/confirmation_primary": "evaluation_reserve",
    "evaluate/development_diagnostics": "evaluation_reserve",
    "summarize_and_package": "finalize_reserve",
}
HARD_RESERVE_KEYS = ("evaluation_reserve", "finalize_reserve")


class BudgetError(RuntimeError):
    """Raised when a budget invariant would be violated."""


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise BudgetError(f"{name} must be finite and non-negative")
    return result


@dataclass
class BudgetLedger:
    plan_budget: Mapping[str, object]
    clock: Callable[[], float]
    session_started: float
    cumulative_spent: float = 0.0
    phase_spent: dict[str, float] = field(default_factory=dict)
    training_started: bool = False
    training_arm: str | None = None
    iteration_seconds: dict[str, list[float]] = field(default_factory=dict)
    resolved: dict[str, object] | None = None

    # -- construction -----------------------------------------------------
    @classmethod
    def start(cls, plan_budget: Mapping[str, object], *, clock: Callable[[], float]) -> "BudgetLedger":
        ledger = cls(plan_budget=dict(plan_budget), clock=clock, session_started=float(clock()))
        ledger._validate_plan_budget()
        return ledger

    @classmethod
    def resume(
        cls, plan_budget: Mapping[str, object], *, clock: Callable[[], float], state: Mapping[str, object],
    ) -> "BudgetLedger":
        if not isinstance(state, Mapping) or state.get("schema") != BUDGET_SCHEMA:
            raise BudgetError("budget resume state is missing or has an unsupported schema")
        if state.get("hash") != canonical({k: v for k, v in state.items() if k != "hash"}):
            raise BudgetError("budget resume state hash is inconsistent")
        if state.get("plan_budget_hash") != canonical(dict(plan_budget)):
            raise BudgetError("budget resume state belongs to a different plan budget")
        ledger = cls(
            plan_budget=dict(plan_budget),
            clock=clock,
            session_started=float(clock()),
            cumulative_spent=_positive(state.get("cumulative_spent", 0.0), "cumulative_spent"),
            phase_spent={str(k): _positive(v, "phase_spent") for k, v in dict(state.get("phase_spent", {})).items()},
            training_started=bool(state.get("training_started", False)),
            training_arm=None if state.get("training_arm") is None else str(state.get("training_arm")),
            iteration_seconds={
                str(k): [float(x) for x in v] for k, v in dict(state.get("iteration_seconds", {})).items()
            },
            resolved=None if state.get("resolved") is None else dict(state["resolved"]),
        )
        ledger._validate_plan_budget()
        # A resumed session may never hand out more than the cumulative budget
        # allows, even if the new session is freshly started.
        if ledger.cumulative_spent > ledger.total_seconds:
            raise BudgetError("cumulative spent already exceeds the frozen total budget")
        return ledger

    def _validate_plan_budget(self) -> None:
        total = _positive(self.plan_budget.get("total_seconds"), "total_seconds")
        phases = dict(self.plan_budget.get("phases") or {})
        slack = _positive(self.plan_budget.get("shared_slack", 0.0), "shared_slack")
        if total <= 0.0:
            raise BudgetError("total budget must be positive")
        if set(phases) != set(PHASE_BUDGET_KEYS.values()):
            raise BudgetError("budget phase keys do not match the serial phases")
        if sum(_positive(value, "phase allowance") for value in phases.values()) + slack != total:
            raise BudgetError("budget allowances plus slack must equal the total budget")

    # -- clocks -----------------------------------------------------------
    @property
    def total_seconds(self) -> float:
        return float(self.plan_budget["total_seconds"])

    @property
    def phase_allowances(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in dict(self.plan_budget["phases"]).items()}

    def session_deadline(self) -> float:
        return float(self.session_started) + self.total_seconds

    def cumulative_remaining(self) -> float:
        return max(0.0, self.total_seconds - float(self.cumulative_spent))

    def session_remaining(self) -> float:
        return max(0.0, self.session_deadline() - float(self.clock()))

    def remaining_total(self) -> float:
        """Safe minimum of the cumulative and the current-session allowance."""
        return min(self.cumulative_remaining(), self.session_remaining())

    def resolved_arm_quota(self, phase_key: str) -> float:
        """Frozen per-arm quota, honouring a resolved symmetric reduction.

        A reduced budget (for example scale=0.5) records ``arm_quota_seconds``
        in the immutable resolved block.  Every later allowance/remaining
        computation must read *that* value; falling back to the plan's 2400 s
        would silently hand back time the reduction removed.
        """
        allowance = self.phase_allowances[phase_key]
        if phase_key not in ("train_control", "train_candidate"):
            return allowance
        if not self.resolved:
            return allowance
        key = "train_control_seconds" if phase_key == "train_control" else "train_candidate_seconds"
        value = self.resolved.get(key)
        if value is None:
            value = self.resolved.get("arm_quota_seconds", allowance)
        return float(value)

    def phase_remaining(self, phase_key: str) -> float:
        if phase_key not in self.phase_allowances:
            raise BudgetError(f"unknown budget phase key {phase_key!r}")
        spent = float(self.phase_spent.get(phase_key, 0.0))
        allowance = self.resolved_arm_quota(phase_key)
        return max(0.0, min(allowance - spent, self.remaining_total()))

    def remaining_for_phase(self, phase: str) -> float:
        key = PHASE_BUDGET_KEYS.get(phase)
        if key is None:
            raise BudgetError(f"unknown serial phase {phase!r}")
        return self.phase_remaining(key)

    # -- accounting -------------------------------------------------------
    def record(self, phase: str, seconds: float) -> None:
        key = PHASE_BUDGET_KEYS.get(phase)
        if key is None:
            raise BudgetError(f"unknown serial phase {phase!r}")
        value = _positive(seconds, "elapsed seconds")
        self.cumulative_spent += value
        self.phase_spent[key] = float(self.phase_spent.get(key, 0.0)) + value

    def record_iteration(self, arm: str, seconds: float) -> None:
        value = _positive(seconds, "iteration seconds")
        self.iteration_seconds.setdefault(str(arm), []).append(value)

    def p95_iteration_seconds(self, arm: str) -> float | None:
        samples = sorted(self.iteration_seconds.get(str(arm), []))
        if not samples:
            return None
        index = max(0, min(len(samples) - 1, math.ceil(0.95 * len(samples)) - 1))
        return samples[index]

    # -- training gate ----------------------------------------------------
    def mark_training_started(self, arm: str) -> None:
        # Both arms train sequentially in the same serial run; the flag only
        # records that formal training has begun (it freezes the budget).
        if not self.training_started:
            self.training_started = True
            self.training_arm = str(arm)

    def hard_reserve_remaining(self) -> float:
        """Remaining evaluation + finalise reserve that training must not eat."""
        return self._reserve_remaining()

    def _reserve_remaining(self) -> float:
        reserves = 0.0
        for key in HARD_RESERVE_KEYS:
            appendix = self.resolved_arm_quota(key) if self.resolved else self.phase_allowances[key]
            reserves += max(0.0, float(appendix) - float(self.phase_spent.get(key, 0.0)))
        return reserves

    def assert_can_start_formal_training(self, arm: str | None = None) -> None:
        """Gate each formal arm with the budget algebra that arm actually needs.

        * Before the *first* arm (``control``) both arm quotas and the full
          evaluation/finalise reserve must still fit, so a control run can
          never consume the space the candidate arm was promised.  If this
          check passes but the second arm could not later be reserved, the
          first arm is refused *before* it starts.
        * Before the *second* arm only its own remaining quota plus the hard
          reserves must fit.  Re-demanding the already-spent control quota at
          that point is an algebraic error and permanently locks a run root
          whose setup+control exceeded 3000 s.

        Quotas are measured against the *remaining* part of each arm's quota
        (a partially spent arm on resume only needs what is left), and the
        resolved reduction is honoured when one was frozen.
        """
        if arm is None:
            # Backwards-compatible conservative form: both arms at once.
            required = (
                self.remaining_for_phase("train/control")
                + self.remaining_for_phase("train/candidate")
                + self._reserve_remaining()
            )
            available = self.remaining_total()
            if available < required:
                raise BudgetError(
                    "insufficient budget to start formal training: "
                    f"need {required:.1f}s for both arms plus the full evaluation/finalise reserve, "
                    f"only {available:.1f}s remain"
                )
            return
        if arm not in ("control", "candidate"):
            raise BudgetError(f"unknown training arm {arm!r}")
        required = self.remaining_for_phase(f"train/{arm}") + self._reserve_remaining()
        if arm == "control":
            # Reserve the candidate's *remaining* quota too, but never demand it
            # twice: the candidate gate below only asks for it once.
            required += self.remaining_for_phase("train/candidate")
            if self.remaining_total() < required:
                raise BudgetError(
                    "insufficient budget to start the first formal arm: "
                    f"need {required:.1f}s for both arms plus the hard reserves, "
                    f"only {self.remaining_total():.1f}s remain; refusing before control starts"
                )
            return
        available = self.remaining_total()
        if available < required:
            raise BudgetError(
                f"insufficient budget to start {arm!r} training: need {required:.1f}s "
                f"(remaining arm quota plus the evaluation/finalise reserve), "
                f"only {available:.1f}s remain"
            )

    # -- symmetric reduction ---------------------------------------------
    def apply_symmetric_reduction(self, scale: float) -> dict[str, object]:
        if self.resolved is not None:
            raise BudgetError("the resolved budget is immutable once frozen")
        if self.training_started:
            raise BudgetError(
                "a budget reduction is only allowed before either arm has started formal training"
            )
        value = float(scale)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise BudgetError("symmetric reduction scale must lie in (0,1]")
        allowances = self.phase_allowances
        reduced = {
            "train_control": math.floor(allowances["train_control"] * value),
            "train_candidate": math.floor(allowances["train_candidate"] * value),
        }
        if reduced["train_control"] != reduced["train_candidate"]:
            raise BudgetError("symmetric reduction produced unequal per-arm quotas")
        required = sum(reduced.values()) + allowances["evaluation_reserve"] + allowances["finalize_reserve"]
        if required > self.remaining_total():
            raise BudgetError("reduced budget still cannot reserve both arms and the evaluation")
        resolved = {
            "schema": "guandan-serial-compare-resolved-budget-v1",
            "scale": value,
            "arm_quota_seconds": reduced["train_control"],
            "train_control_seconds": reduced["train_control"],
            "train_candidate_seconds": reduced["train_candidate"],
            "evaluation_reserve_seconds": allowances["evaluation_reserve"],
            "finalize_reserve_seconds": allowances["finalize_reserve"],
            "cumulative_spent_at_freeze": float(self.cumulative_spent),
            "symmetric": True,
            "frozen": True,
        }
        resolved["subplan_hash"] = canonical(resolved)
        self.resolved = resolved
        return dict(resolved)

    def freeze(self) -> dict[str, object]:
        """Freeze the *unreduced* plan budget (the default path)."""
        if self.resolved is not None:
            return dict(self.resolved)
        allowances = self.phase_allowances
        resolved = {
            "schema": "guandan-serial-compare-resolved-budget-v1",
            "scale": 1.0,
            "arm_quota_seconds": allowances["train_control"],
            "train_control_seconds": allowances["train_control"],
            "train_candidate_seconds": allowances["train_candidate"],
            "evaluation_reserve_seconds": allowances["evaluation_reserve"],
            "finalize_reserve_seconds": allowances["finalize_reserve"],
            "cumulative_spent_at_freeze": float(self.cumulative_spent),
            "symmetric": True,
            "frozen": True,
        }
        resolved["subplan_hash"] = canonical(resolved)
        self.resolved = resolved
        return dict(resolved)

    # -- persistence ------------------------------------------------------
    def to_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "schema": BUDGET_SCHEMA,
            "plan_budget_hash": canonical(dict(self.plan_budget)),
            "cumulative_spent": float(self.cumulative_spent),
            "phase_spent": {k: float(v) for k, v in sorted(self.phase_spent.items())},
            "training_started": bool(self.training_started),
            "training_arm": self.training_arm,
            "iteration_seconds": {k: list(v) for k, v in sorted(self.iteration_seconds.items())},
            "resolved": self.resolved,
        }
        state["hash"] = canonical(state)
        return state


def p95_seconds(values: Sequence[float]) -> float | None:
    """Nearest-rank 95th percentile of measured iteration wall times."""
    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    index = max(0, min(len(samples) - 1, math.ceil(0.95 * len(samples)) - 1))
    return samples[index]


def fits_next_iteration(
    *,
    interval_seconds: Sequence[float],
    remaining_seconds: float,
    save_seconds: float,
    reserve_seconds: float,
) -> bool:
    """Can one *more* iteration plus its save still leave the hard reserve?

    Checked before every iteration so a long run cannot discover the reserve
    breach two hours later.  With no measured sample the answer is optimistically
    ``True`` (the caller still enforces its own wall-clock limit).
    """
    p95 = p95_seconds(interval_seconds)
    if p95 is None:
        return True
    need = p95 + max(0.0, float(save_seconds))
    return float(remaining_seconds) - need >= max(0.0, float(reserve_seconds))


__all__ = [
    "BudgetError", "BudgetLedger", "HARD_RESERVE_KEYS", "PHASE_BUDGET_KEYS",
    "fits_next_iteration", "p95_seconds",
]
