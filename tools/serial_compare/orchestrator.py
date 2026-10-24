# SPDX-License-Identifier: MIT
"""Serial state machine for the single-upload recipe comparison (spec section 5).

Phase order (frozen)::

    bootstrap/input_check
      -> preflight/control
      -> preflight/candidate
      -> freeze_execution_budget
      -> train/control
      -> train/candidate
      -> freeze_final_checkpoints
      -> evaluate/confirmation_primary
      -> evaluate/development_diagnostics
      -> summarize_and_package

Both preflights must complete before *any* formal training starts.  Every phase
publishes an atomic completion marker bound to the plan/source/model/output
hashes; a marker whose declared output is missing or changed is not evidence of
completion.  A completed control arm is never retrained, and a resume after both
arms finished trains nothing and only re-runs evaluation/packaging.

Phase status vocabulary
-----------------------
``complete``    the phase finished and its outputs validate.
``incomplete``  the phase ran but ran out of its fixed allowance (for example an
                evaluation cut short by the budget).  It is recorded honestly,
                packaging may still run, and a later session retries it.  This
                is *not* a technical failure and never becomes a PASS.
``skipped``     a deliberately not-attempted auxiliary phase (development when
                the primary confirmation is incomplete).
``failed``      a technical failure; the run stops and reports ``failed_phase``.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .budget import BudgetError, BudgetLedger
from .common import (
    EVALUATION_PHASES, PHASE_ORDER, RESULT_SCHEMA, STATE_SCHEMA, TRAINING_PHASES,
    atomic_json, canonical, read_json, validate_plan,
)
from .markers import phase_is_complete, write_marker


class OrchestratorError(RuntimeError):
    """Raised when the serial contract would be violated."""


class SerialCompareActions(Protocol):
    """Production/Kaggle actions; tests inject fakes with the same signatures."""

    def input_check(self, context: "PhaseContext") -> Mapping[str, object]: ...

    def preflight(self, context: "PhaseContext", arm: str) -> Mapping[str, object]: ...

    def freeze_budget(self, context: "PhaseContext", resolved: Mapping[str, object]) -> Mapping[str, object]: ...

    def train(self, context: "PhaseContext", arm: str, *, max_seconds: float) -> Mapping[str, object]: ...

    def freeze_finals(self, context: "PhaseContext") -> Mapping[str, object]: ...

    def evaluate(self, context: "PhaseContext", phase: str, tasks: list[dict[str, object]]) -> Mapping[str, object]: ...

    def summarize(self, context: "PhaseContext") -> Mapping[str, object]: ...


@dataclass
class PhaseContext:
    plan: Mapping[str, object]
    root: Path
    phase: str
    phase_dir: Path
    state: dict[str, object]
    budget: BudgetLedger
    model_paths: dict[str, Path]
    clock: Callable[[], float]
    orchestrator: "SerialCompareOrchestrator | None" = None

    @property
    def plan_hash(self) -> str:
        return str(self.plan["plan_hash"])

    @property
    def source_hash(self) -> str:
        return str(self.plan["source"]["core_source_hash"])

    def arm_result(self, arm: str) -> dict[str, object]:
        return dict((self.state.get("arms", {}) or {}).get(arm, {}))

    def final_paths(self) -> dict[str, object]:
        return dict(self.state.get("finals") or {})


def _as_mapping(value: object, phase: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OrchestratorError(f"action for {phase!r} did not return a mapping")
    return value


class SerialCompareOrchestrator:
    def __init__(
        self,
        plan: Mapping[str, object],
        *,
        root: str | Path,
        actions: SerialCompareActions,
        model_paths: Mapping[str, str | Path],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.plan = plan
        self.root = Path(root).resolve()
        self.actions = actions
        self.model_paths = {str(k): Path(v).resolve() for k, v in model_paths.items()}
        self.clock = clock
        self.plan_hash = str(plan["plan_hash"])
        self.source_hash = str(plan["source"]["core_source_hash"])
        self.state_path = self.root / "state.json"
        self.budget_path = self.root / "budget.json"
        self.state: dict[str, object] = {}
        self.budget: BudgetLedger | None = None

    # -- persistence ------------------------------------------------------
    def _load_or_init_state(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
                raise OrchestratorError("existing state.json has an unsupported schema")
            if state.get("hash") != canonical({k: v for k, v in state.items() if k != "hash"}):
                raise OrchestratorError("existing state.json hash is inconsistent (tampered state)")
            if state.get("plan_hash") != self.plan_hash:
                raise OrchestratorError("existing state belongs to a different plan")
            self.state = dict(state)
            budget_state = read_json(self.budget_path) if self.budget_path.is_file() else None
            if budget_state is None:
                raise OrchestratorError("state exists but the persisted budget is missing")
            self.budget = BudgetLedger.resume(
                self.plan["budget"], clock=self.clock, state=budget_state,
            )
        else:
            self.budget = BudgetLedger.start(self.plan["budget"], clock=self.clock)
            self.state = {
                "schema": STATE_SCHEMA,
                "plan_hash": self.plan_hash,
                "source_hash": self.source_hash,
                "status": "running",
                "current_phase": None,
                "failed_phase": None,
                "error": None,
                "phases": {},
                "arms": {},
                "session_count": 0,
            }
            self._persist()

    def _persist(self) -> None:
        assert self.budget is not None
        self.state["hash"] = canonical({k: v for k, v in self.state.items() if k != "hash"})
        atomic_json(self.state_path, self.state)
        atomic_json(self.budget_path, self.budget.to_state())

    def _phase_dir(self, phase: str) -> Path:
        return self.root / "phases" / phase.replace("/", "_")

    def _phase_status(self, phase: str) -> str:
        return str(dict(self.state.get("phases", {}) or {}).get(phase, {}).get("status", "pending"))

    def _input_hashes(self, phase: str) -> dict[str, str]:
        hashes = {
            "plan_hash": self.plan_hash,
            "source_hash": self.source_hash,
            "p375_sha256": str(self.plan["models"]["p375"]["sha256"]),
            "p250_sha256": str(self.plan["models"]["p250"]["sha256"]),
            "p500_sha256": str(self.plan["models"]["p500"]["sha256"]),
        }
        if phase.startswith("train/") or phase == "freeze_final_checkpoints" or phase.startswith("evaluate/") or phase == "summarize_and_package":
            for arm in ("control", "candidate"):
                preflight = dict(self.state.get("phases", {}) or {}).get(f"preflight/{arm}", {}) or {}
                if preflight.get("status") != "complete":
                    raise OrchestratorError(f"preflight/{arm} must complete before {phase}")
                hashes[f"preflight/{arm}"] = str(preflight.get("marker_hash", ""))
        if phase == "freeze_final_checkpoints" or phase.startswith("evaluate/") or phase == "summarize_and_package":
            for arm in ("control", "candidate"):
                trained = dict(self.state.get("phases", {}) or {}).get(f"train/{arm}", {}) or {}
                if trained.get("status") != "complete":
                    raise OrchestratorError(f"train/{arm} must complete before {phase}")
                hashes[f"train/{arm}"] = str(trained.get("marker_hash", ""))
        return hashes

    def is_complete(self, phase: str) -> tuple[bool, str]:
        try:
            input_hashes = self._input_hashes(phase)
        except OrchestratorError as error:
            return False, str(error)
        return phase_is_complete(
            self._phase_dir(phase), phase=phase, plan_hash=self.plan_hash,
            source_hash=self.source_hash, root=self.root, input_hashes=input_hashes,
        )

    # -- preconditions ----------------------------------------------------
    def _require_marker(self, phase: str) -> None:
        complete, reason = self.is_complete(phase)
        if not complete:
            raise OrchestratorError(f"phase {phase!r} must complete first: {reason}")

    def _require_complete(self, phase: str) -> None:
        self._require_marker(phase)
        if self._phase_status(phase) != "complete":
            raise OrchestratorError(
                f"phase {phase!r} finished with status {self._phase_status(phase)!r}, not 'complete'"
            )

    def _preconditions(self, phase: str) -> None:
        if phase == "bootstrap/input_check":
            return
        if phase == "preflight/control":
            self._require_complete("bootstrap/input_check")
            return
        if phase == "preflight/candidate":
            self._require_complete("preflight/control")
            return
        if phase == "freeze_execution_budget":
            self._require_complete("preflight/control")
            self._require_complete("preflight/candidate")
            return
        if phase in TRAINING_PHASES:
            self._require_complete("freeze_execution_budget")
            self._require_complete("preflight/control")
            self._require_complete("preflight/candidate")
            if phase == "train/candidate":
                self._require_complete("train/control")
            return
        if phase == "freeze_final_checkpoints":
            for arm_phase in TRAINING_PHASES:
                self._require_complete(arm_phase)
            return
        if phase == "evaluate/confirmation_primary":
            self._require_complete("freeze_final_checkpoints")
            return
        if phase == "evaluate/development_diagnostics":
            self._require_marker("evaluate/confirmation_primary")
            return
        if phase == "summarize_and_package":
            self._require_marker("evaluate/confirmation_primary")
            return
        raise OrchestratorError(f"unknown phase {phase!r}")

    # -- execution --------------------------------------------------------
    def _record_phase(
        self, phase: str, *, status: str, metadata: Mapping[str, object], seconds: float,
        marker_hash: str | None = None,
    ) -> None:
        phases = dict(self.state.get("phases", {}) or {})
        phases[phase] = {
            "status": str(status),
            "seconds": float(seconds),
            "metadata": dict(metadata),
            "marker_hash": marker_hash,
        }
        self.state["phases"] = phases
        self.state["current_phase"] = phase

    def _context(self, phase: str) -> PhaseContext:
        return PhaseContext(
            plan=self.plan, root=self.root, phase=phase, phase_dir=self._phase_dir(phase),
            state=self.state, budget=self.budget, model_paths=self.model_paths,
            clock=self.clock, orchestrator=self,
        )  # type: ignore[arg-type]

    def _run_action(self, phase: str) -> Mapping[str, object]:
        context = self._context(phase)
        if phase == "bootstrap/input_check":
            return _as_mapping(self.actions.input_check(context), phase)
        if phase in ("preflight/control", "preflight/candidate"):
            return _as_mapping(self.actions.preflight(context, phase.split("/", 1)[1]), phase)
        if phase == "freeze_execution_budget":
            assert self.budget is not None
            resolved = self.budget.freeze()
            return _as_mapping(self.actions.freeze_budget(context, resolved), phase)
        if phase in TRAINING_PHASES:
            assert self.budget is not None
            arm = phase.split("/", 1)[1]
            # Per-arm gate: the control gate additionally reserves the
            # candidate's quota so it is refused *before* control starts if the
            # second arm could not later run; the candidate gate only asks for
            # its own remaining quota plus the hard reserves.
            self.budget.assert_can_start_formal_training(arm)
            self.budget.mark_training_started(arm)
            max_seconds = self.budget.remaining_for_phase(phase)
            if max_seconds <= 0.0:
                raise BudgetError(f"no remaining budget for {phase}")
            return _as_mapping(self.actions.train(context, arm, max_seconds=max_seconds), phase)
        if phase == "freeze_final_checkpoints":
            return _as_mapping(self.actions.freeze_finals(context), phase)
        if phase in EVALUATION_PHASES:
            tasks = list(dict(self.state.get("tasks", {}) or {}).get(phase, []))
            return _as_mapping(self.actions.evaluate(context, phase, tasks), phase)
        if phase == "summarize_and_package":
            return _as_mapping(self.actions.summarize(context), phase)
        raise OrchestratorError(f"unknown phase {phase!r}")

    def run(self) -> dict[str, object]:
        validate_plan(self.plan)
        self._load_or_init_state()
        assert self.budget is not None
        self.state["session_count"] = int(self.state.get("session_count", 0)) + 1

        for phase in PHASE_ORDER:
            if self._phase_status(phase) == "complete":
                complete, reason = self.is_complete(phase)
                if complete:
                    continue
                error = OrchestratorError(f"completed phase is no longer valid: {reason}")
                self._fail(phase, error)
                raise error
            # The auxiliary development matrix never runs on an incomplete
            # primary confirmation: that would spend the remaining budget on
            # secondary evidence while the formal set is short.
            if phase == "evaluate/development_diagnostics":
                marker_ok, _ = self.is_complete("evaluate/confirmation_primary")
                if marker_ok and self._phase_status("evaluate/confirmation_primary") != "complete":
                    self._record_phase(
                        phase, status="skipped",
                        metadata={"reason": "primary confirmation is incomplete; development deliberately not started"},
                        seconds=0.0,
                    )
                    self._persist()
                    continue
            started = float(self.clock())
            recorded_seconds = False
            try:
                self._preconditions(phase)
                phase_dir = self._phase_dir(phase)
                phase_dir.mkdir(parents=True, exist_ok=True)
                result = self._run_action(phase)
                status = str(result.get("status", "complete"))
                if phase.startswith("preflight/"):
                    if status not in {"pass", "complete"}:
                        raise OrchestratorError(f"{phase} did not pass (status={status!r})")
                    recorded = "complete"
                else:
                    if status not in {"complete", "incomplete"}:
                        raise OrchestratorError(f"{phase} returned an invalid status {status!r}")
                    recorded = status
                declared = result.get("outputs") or []
                paths = []
                for value in declared:
                    candidate = Path(str(value))
                    paths.append(candidate if candidate.is_absolute() else self.root / candidate)
                elapsed = float(self.clock()) - started
                self.budget.record(phase, elapsed)
                recorded_seconds = True
                marker = write_marker(
                    phase_dir, phase=phase, plan_hash=self.plan_hash, source_hash=self.source_hash,
                    input_hashes=self._input_hashes(phase), output_paths=paths, root=self.root,
                    metadata={**dict(result.get("metadata") or {}), "result_status": recorded},
                )
                self._record_phase(
                    phase, status=recorded,
                    metadata={**dict(result.get("metadata") or {}), "budget_seconds": elapsed},
                    seconds=elapsed, marker_hash=str(marker["hash"]),
                )
                if phase in TRAINING_PHASES:
                    self.record_arm(phase.split("/", 1)[1], dict(result.get("metadata") or {}))
                if phase == "freeze_final_checkpoints":
                    self.state["finals"] = dict(result.get("metadata") or {})
                if phase in EVALUATION_PHASES and result.get("summary") is not None:
                    self.record_summary(phase, result["summary"])
                self._persist()
            except BaseException as error:  # noqa: BLE001 - boundary must record every failure
                # A failed/aborted phase still consumed real wall clock.  Record
                # it exactly once so a resume cannot be handed the spent time
                # again (a technical failure never refunds budget).
                failure_seconds = float(self.clock()) - started
                if not recorded_seconds and failure_seconds > 0.0:
                    try:
                        self.budget.record(phase, failure_seconds)
                    except BudgetError:
                        pass
                    recorded_seconds = True
                self._fail(
                    phase, error,
                    seconds=failure_seconds if recorded_seconds else 0.0,
                )
                raise

        statuses = {phase: self._phase_status(phase) for phase in PHASE_ORDER}
        self.state["status"] = (
            "complete" if all(value == "complete" for value in statuses.values())
            else "incomplete"
        )
        self.state["current_phase"] = None
        self._persist()
        return self.result()

    def _fail(self, phase: str, error: BaseException, *, seconds: float = 0.0) -> None:
        self.state["status"] = "failed"
        self.state["failed_phase"] = phase
        self.state["error"] = f"{type(error).__name__}: {error}"
        self.state["traceback"] = traceback.format_exc()
        self._record_phase(
            phase, status="failed", metadata={"error": f"{type(error).__name__}: {error}"},
            seconds=float(seconds),
        )
        self._persist()

    # -- results ----------------------------------------------------------
    def provisional_status(self) -> str:
        """Status implied by every phase except the packaging phase itself."""
        statuses = {
            phase: self._phase_status(phase)
            for phase in PHASE_ORDER if phase != "summarize_and_package"
        }
        if any(value == "failed" for value in statuses.values()):
            return "failed"
        return "complete" if all(value == "complete" for value in statuses.values()) else "incomplete"

    def result(self) -> dict[str, object]:
        assert self.budget is not None
        confirmation = dict(self.state.get("confirmation") or {})
        development = dict(self.state.get("development") or {})
        verdict = dict(confirmation.get("verdict") or {})
        status = str(self.state.get("status"))
        if status == "running":
            status = self.provisional_status()
        payload = {
            "schema": RESULT_SCHEMA,
            "plan_hash": self.plan_hash,
            "status": status,
            "failed_phase": self.state.get("failed_phase"),
            "error": self.state.get("error"),
            "phases": {
                phase: {
                    "status": value.get("status"),
                    "seconds": value.get("seconds"),
                    "marker_hash": value.get("marker_hash"),
                }
                for phase, value in dict(self.state.get("phases", {})).items()
            },
            "budget": {
                "cumulative_spent": float(self.budget.cumulative_spent),
                "resolved": self.budget.resolved,
                "session_remaining": self.budget.session_remaining(),
                "cumulative_remaining": self.budget.cumulative_remaining(),
            },
            "arms": dict(self.state.get("arms") or {}),
            "finals": dict(self.state.get("finals") or {}),
            "confirmation": confirmation,
            "development": development,
            "verdict": verdict,
            "technical_failure": None if status == "complete" else (
                {"phase": self.state.get("failed_phase"), "error": self.state.get("error")}
                if status == "failed" else None
            ),
            "fabledan": "NOT_RUN",
            "final_holdout": "NOT_TOUCHED",
            "formal_production_promotion": False,
            "kaggle_execution": "NOT_RUN",
            "target_dual_t4_validation": "NOT_RUN",
        }
        payload["hash"] = canonical(payload)
        return payload

    # -- hooks used by actions -------------------------------------------
    def record_arm(self, arm: str, payload: Mapping[str, object]) -> None:
        arms = dict(self.state.get("arms", {}) or {})
        arms[str(arm)] = dict(payload)
        self.state["arms"] = arms
        self._persist()

    def record_tasks(self, phase: str, tasks: list[dict[str, object]]) -> None:
        registered = dict(self.state.get("tasks", {}) or {})
        registered[str(phase)] = [dict(task) for task in tasks]
        self.state["tasks"] = registered
        self._persist()

    def record_summary(self, phase: str, summary: Mapping[str, object]) -> None:
        if phase == "evaluate/confirmation_primary":
            self.state["confirmation"] = dict(summary)
        elif phase == "evaluate/development_diagnostics":
            self.state["development"] = dict(summary)
        else:
            raise OrchestratorError(f"{phase!r} does not carry an evaluation summary")
        self._persist()


def resume_summary(root: str | Path) -> dict[str, object]:
    """Inspect a run directory without mutating it (operator helper)."""
    root = Path(root)
    state_path = root / "state.json"
    if not state_path.is_file():
        return {"status": "absent", "root": str(root)}
    state = read_json(state_path)
    return {
        "status": state.get("status"),
        "failed_phase": state.get("failed_phase"),
        "error": state.get("error"),
        "phases": {k: v.get("status") for k, v in dict(state.get("phases", {})).items()},
    }


__all__ = [
    "OrchestratorError", "PhaseContext", "SerialCompareActions",
    "SerialCompareOrchestrator", "resume_summary",
]
