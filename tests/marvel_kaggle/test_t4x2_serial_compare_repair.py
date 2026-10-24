"""Repair-round counterexamples for the serial recipe comparison (round 1).

Every test here was written against the *reviewer's* counterexamples and fails
on the pre-repair implementation.  They run on CPU only: orchestration uses
injected fake actions, training uses a fake trainer, and statistics references
are hand-computed rather than re-using the production helper.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from serial_compare import common, evaluation, markers  # noqa: E402
from serial_compare.budget import BudgetError, BudgetLedger, fits_next_iteration  # noqa: E402
from serial_compare.common import (  # noqa: E402
    PlanError, build_deal_manifest, build_plan, canonical, read_json, sha,
)
from serial_compare import delivery as delivery_module  # noqa: E402
from serial_compare.delivery import (  # noqa: E402
    verify_input_bundle, verify_member_tree, zip_bytes,
)
from serial_compare.orchestrator import (  # noqa: E402
    OrchestratorError, PhaseContext, SerialCompareOrchestrator,
)
from serial_compare.pipeline import ProductionActions, latest_recovery_files  # noqa: E402
from serial_compare.reporting import ensure_minimal_result, write_reports  # noqa: E402

MODEL_HASHES = {"p375": "1" * 64, "p250": "2" * 64, "p500": "3" * 64}


# ---------------------------------------------------------------------------
# plan / fake actions
# ---------------------------------------------------------------------------
def _models() -> dict[str, dict[str, object]]:
    return {
        name: {
            "sha256": digest, "iteration": {"p375": 375, "p250": 250, "p500": 500}[name],
            "training_resume_allowed": False,
        }
        for name, digest in MODEL_HASHES.items()
    }


def _plan() -> dict[str, object]:
    return build_plan(
        source={
            "archive_sha256": "a" * 64, "content_manifest_hash": "b" * 64,
            "git_commit": "f" * 40, "core_source_hash": "c" * 64,
            "runtime_path": "/tmp/source", "python": "3.13.15", "torch": "2.11.0+cu128",
            "backend_hash": "d" * 64, "native_name": "_guandan_cpp.so", "native_sha256": "e" * 64,
        },
        models=_models(),
    )


class TimedFakeActions:
    """Fake production actions whose clock advances by per-phase durations."""

    def __init__(self, now: list[float], *, durations: Mapping[str, float] | None = None,
                 fail_at: str | None = None, confirmation_complete: bool = True):
        self.now = now
        self.durations = dict(durations or {})
        self.fail_at = fail_at
        self.confirmation_complete = confirmation_complete
        self.calls: list[str] = []
        self.shared_fingerprint = {"hash": "shared" + "0" * 58, "tensor_count": 5}

    def _out(self, context, name: str, payload: Mapping[str, object]) -> Path:
        path = context.phase_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _step(self, name: str) -> None:
        self.calls.append(name)
        self.now[0] += float(self.durations.get(name, 0.0))
        if self.fail_at == name:
            raise RuntimeError(f"injected failure at {name}")

    def input_check(self, context):
        self._step("input_check")
        return {"status": "complete", "outputs": [self._out(context, "inputs.json", {"ok": True})]}

    def preflight(self, context, arm):
        self._step(f"preflight:{arm}")
        return {
            "status": "pass",
            "outputs": [self._out(context, "preflight.json", {"arm": arm, "status": "pass"})],
            "metadata": {"initial_state_fingerprint": self.shared_fingerprint,
                         "pool_manifest_hash": None if arm == "control" else "poolhash"},
        }

    def freeze_budget(self, context, resolved):
        self._step("freeze_budget")
        return {"status": "complete", "outputs": [self._out(context, "resolved_budget.json", resolved)]}

    def train(self, context, arm, *, max_seconds):
        self._step(f"train:{arm}")
        policy = context.phase_dir / f"policy_{arm}.pt"
        policy.parent.mkdir(parents=True, exist_ok=True)
        policy.write_text(f"{arm}-final-policy", encoding="utf-8")
        metadata = {
            "arm": arm, "start_iteration": 0, "final_iteration": 4, "applied_updates": 8,
            "initial_source_sha256": MODEL_HASHES["p375"],
            "initial_state_fingerprint": self.shared_fingerprint,
            "policy": str(policy), "policy_sha256": sha(policy),
            "target_kl": None if arm == "control" else 0.015,
            "ntp_enabled": arm == "control",
            "opponent_pool_manifest_hash": None if arm == "control" else "poolhash",
            "max_seconds": max_seconds,
        }
        return {"status": "complete", "outputs": [policy], "metadata": metadata}

    def freeze_finals(self, context):
        self._step("freeze_finals")
        from serial_compare.training import verify_arm_pair

        control = context.arm_result("control")
        candidate = context.arm_result("candidate")
        pair = verify_arm_pair(control, candidate)
        record = {
            "control_final": control["policy"], "control_final_sha256": control["policy_sha256"],
            "candidate_final": candidate["policy"], "candidate_final_sha256": candidate["policy_sha256"],
            "pair": pair,
        }
        return {"status": "complete", "outputs": [self._out(context, "finals.json", record)], "metadata": record}

    def evaluate(self, context, phase, tasks):
        self._step(f"evaluate:{phase}")
        domain = "confirmation" if phase.endswith("confirmation_primary") else "development"
        manifest = build_deal_manifest(domain)
        registered = (
            evaluation.confirmation_tasks(manifest) if domain == "confirmation"
            else evaluation.development_tasks(manifest)
        )
        if domain == "confirmation" and not self.confirmation_complete:
            registered = registered[:10]
        models = dict(_models())
        models["control_final"] = {"sha256": sha(Path(context.final_paths()["control_final"]))}
        models["candidate_final"] = {"sha256": sha(Path(context.final_paths()["candidate_final"]))}
        records = [_record(task, models, win=True) for task in registered]
        summary = (
            evaluation.summarize_confirmation(records, evaluation.confirmation_tasks(manifest))
            if domain == "confirmation"
            else evaluation.summarize_development(records, evaluation.development_tasks(manifest))
        )
        path = self._out(context, f"{domain}_summary.json", summary)
        return {
            "status": "complete" if summary.get("complete") else "incomplete",
            "outputs": [path], "summary": summary,
            "metadata": {"completed_games": summary.get("completed_games")},
        }

    def summarize(self, context):
        self._step("summarize")
        result = context.orchestrator.result() if context.orchestrator else {}
        path = self._out(context, "RESULT.json", result)
        return {"status": "complete", "outputs": [path]}


def _orchestrator(root: Path, actions, clock) -> SerialCompareOrchestrator:
    model_paths = {
        "p375": root / "policy_375.pt", "p250": root / "anchor_250.pt", "p500": root / "policy_500.pt",
    }
    return SerialCompareOrchestrator(
        _plan(), root=root / "artifacts", actions=actions, model_paths=model_paths, clock=clock,
    )


def _record(task: Mapping[str, object], models: Mapping[str, Mapping[str, object]], *,
            win: bool = True, status: str = "complete", trace: str = "a" * 64) -> dict[str, object]:
    candidate_key, opponent_key = evaluation.ARM_SPECS[task["arm"]]
    assignment = int(task["assignment"])
    record = {
        "task_id": task["task_id"], "arm": task["arm"], "deal_id": task["deal_id"],
        "deal_seed": task["deal_seed"], "seat_rotation": assignment,
        "candidate_team": assignment % 2, "temperature": 0.0, "evaluation_horizon": 1024,
        "base_deal_sha256": canonical({"deal": task["deal"]}),
        "candidate_checkpoint_sha256": str(models[candidate_key]["sha256"]),
        "opponent_checkpoint_sha256": None if opponent_key is None else str(models[opponent_key]["sha256"]),
        "status": status, "illegal_count": 0,
        "token_steps": 20, "committed_steps": 8, "outcome_class": "HEAD_LAST",
        "action_trace_sha256": trace,
    }
    if status == "complete":
        record.update({
            "candidate_win": bool(win), "candidate_team_reward": 1.0 if win else -1.0,
            "truncated_count": 0,
        })
    else:
        record.update({"truncated_count": 1, "token_steps": 1024, "committed_steps": 0})
    if status == "complete":
        record["winner_team"] = assignment % 2 if win else 1 - assignment % 2
        record["ranking"] = [1, 2, 4, 3] if record["winner_team"] == 0 else [2, 1, 3, 4]
    record["task_hash"] = canonical(dict(task))
    record["record_hash"] = canonical(record)
    return record


def _final_models() -> dict[str, dict[str, object]]:
    models = dict(_models())
    models["control_final"] = {"sha256": "4" * 64}
    models["candidate_final"] = {"sha256": "5" * 64}
    return models


# ---------------------------------------------------------------------------
# P1-1 double budget gate
# ---------------------------------------------------------------------------
def test_timed_gate_lets_candidate_train_after_1000s_setup_and_2400s_control(tmp_path):
    now = [0.0]
    actions = TimedFakeActions(now, durations={"input_check": 1000.0, "train:control": 2400.0})
    result = _orchestrator(tmp_path, actions, lambda: now[0]).run()
    assert "train:candidate" in actions.calls
    assert result["arms"]["candidate"]["max_seconds"] == pytest.approx(2400.0)
    assert result["status"] == "complete"


def test_timed_gate_starts_candidate_even_with_1200s_setup(tmp_path):
    now = [0.0]
    actions = TimedFakeActions(now, durations={"input_check": 1200.0, "train:control": 2400.0})
    _orchestrator(tmp_path, actions, lambda: now[0]).run()
    assert "train:candidate" in actions.calls


def test_setup_of_6000s_is_refused_before_control_starts(tmp_path):
    now = [0.0]
    actions = TimedFakeActions(now, durations={"input_check": 6000.0})
    with pytest.raises(BudgetError, match="refusing before control"):
        _orchestrator(tmp_path, actions, lambda: now[0]).run()
    assert not any(name == "train:control" for name in actions.calls)


def test_resume_after_interrupted_candidate_does_not_retrain_control(tmp_path):
    now = [0.0]
    actions = TimedFakeActions(
        now, durations={"input_check": 1000.0, "train:control": 2400.0, "train:candidate": 1200.0},
        fail_at="train:candidate",
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        _orchestrator(tmp_path, actions, lambda: now[0]).run()
    # The candidate gate must not refuse the second arm before it can spend its
    # own quota: the failure has to happen *inside* the candidate action.
    assert actions.calls.count("train:candidate") == 1

    resume_now = [now[0]]
    resumed_actions = TimedFakeActions(resume_now, durations={"train:candidate": 1200.0})
    result = _orchestrator(tmp_path, resumed_actions, lambda: resume_now[0]).run()
    assert resumed_actions.calls.count("train:control") == 0
    assert resumed_actions.calls.count("train:candidate") == 1
    assert result["status"] == "complete"


# ---------------------------------------------------------------------------
# P2-2 budget accounting
# ---------------------------------------------------------------------------
def test_failed_phase_wall_clock_is_recorded_and_not_refunded(tmp_path):
    now = [0.0]
    actions = TimedFakeActions(
        now, durations={"train:control": 2900.0, "train:candidate": 1800.0},
        fail_at="train:candidate",
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        _orchestrator(tmp_path, actions, lambda: now[0]).run()
    state = read_json(tmp_path / "artifacts/budget.json")
    assert float(state["cumulative_spent"]) == pytest.approx(4700.0, abs=1.0)
    assert float(state["phase_spent"]["train_candidate"]) == pytest.approx(1800.0, abs=1.0)
    resumed = BudgetLedger.resume(_plan()["budget"], clock=lambda: now[0], state=state)
    assert resumed.cumulative_spent == pytest.approx(4700.0, abs=1.0)
    assert resumed.remaining_for_phase("train/candidate") == pytest.approx(600.0, abs=1.0)


def test_symmetric_reduction_limits_phase_remaining_to_the_resolved_quota():
    ledger = BudgetLedger.start(_plan()["budget"], clock=lambda: 0.0)
    ledger.apply_symmetric_reduction(0.5)
    assert ledger.remaining_for_phase("train/control") == pytest.approx(1200.0)
    assert ledger.remaining_for_phase("train/candidate") == pytest.approx(1200.0)


def test_fits_next_iteration_uses_measured_p95_save_and_hard_reserve():
    assert fits_next_iteration(
        interval_seconds=[100.0, 100.0], remaining_seconds=6200.0,
        save_seconds=10.0, reserve_seconds=6000.0,
    ) is True
    assert fits_next_iteration(
        interval_seconds=[100.0, 100.0], remaining_seconds=6050.0,
        save_seconds=10.0, reserve_seconds=6000.0,
    ) is False
    # The production wiring forwards this remaining reserve to the arm worker.
    ledger = BudgetLedger.start(_plan()["budget"], clock=lambda: 0.0)
    assert ledger.hard_reserve_remaining() == pytest.approx(6600.0)
    ledger.record("evaluate/confirmation_primary", 500.0)
    assert ledger.hard_reserve_remaining() == pytest.approx(6100.0)


class FakeConfig:
    seed = 570_700
    training_data_generator_version = "fake"
    training_context_schedule = {"schedule": "fake"}
    training_context_curriculum = {"curriculum": "fake"}

    def to_dict(self) -> dict[str, object]:
        return {
            "training_context_schedule": self.training_context_schedule,
            "training_context_curriculum": self.training_context_curriculum,
        }


class FakeTrainer:
    world_size = 1
    is_primary = True
    distributed = None
    opponent_pool = None

    def __init__(self, clock: list[float], *, seconds_per_iteration: float = 50.0,
                 start_iteration: int = 0, learner_updates: int = 0):
        self.clock = clock
        self.seconds_per_iteration = seconds_per_iteration
        self.iteration = start_iteration
        self.learner_updates = learner_updates
        self.cfg = FakeConfig()

    def train_iteration(self):
        self.clock[0] += self.seconds_per_iteration
        self.iteration += 1
        self.learner_updates += 1
        return {
            "statistics/learner_updates": 1,
            "macro/value_ev": 0.5,
            "macro/non_forced_normalized_entropy": 0.7,
            "runtime/games_per_second": 3.0,
        }

    def save_checkpoint(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"checkpoint")

    def load_checkpoint(self, path: Path) -> None:
        # A recovery that already sits at the 256-iteration target.
        self.iteration = 256
        self.learner_updates = 99


def _source_file(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "policy_375.pt"
    source.write_bytes(b"the actual 375 artifact")
    plan = copy.deepcopy(_plan())
    plan["models"]["p375"]["sha256"] = sha(source)
    return source, plan


def test_training_loop_stops_on_p95_reserve_before_overspending(tmp_path):
    from serial_compare.training import run_training_arm

    source, plan = _source_file(tmp_path)
    clock = [0.0]
    run_dir = tmp_path / "run"
    result = run_training_arm(
        plan=plan, arm="control", run_dir=run_dir, source_payload={},
        source_model_path=source, model_paths={}, game=object(), device="cpu",
        ddp_enabled=False, world_size=1, clock=lambda: clock[0], max_seconds=100.0,
        trainer_factory=lambda cfg: FakeTrainer(clock),
        import_policy=lambda *args, **kwargs: {"hash": "import"},
        initial_state_fingerprint=lambda trainer: {"hash": "fp"}, config_override=FakeConfig(),
        pool_entries_override=(),
        policy_exporter=lambda trainer, path, iteration: Path(path).write_bytes(b"policy"),
        hard_reserve_seconds=30.0,
    )
    assert result["stop_reason"] == "budget_stopped"
    assert len(result["iterations"]) == 1
    assert result["applied_updates"] == 1
    # Section-7 diagnostics + per-rank telemetry must actually be written.
    assert (run_dir / "diagnostics/latest.json").is_file()
    telemetry = read_json(run_dir / "telemetry/rank000.json")
    assert telemetry["iteration_p95_seconds"] == pytest.approx(50.0)
    assert telemetry["games_per_second"] == pytest.approx(3.0)


def test_candidate_diagnostics_mark_ntp_disabled(tmp_path):
    from serial_compare.training import run_training_arm

    source, plan = _source_file(tmp_path)
    clock = [0.0]
    run_dir = tmp_path / "run"
    run_training_arm(
        plan=plan, arm="candidate", run_dir=run_dir, source_payload={},
        source_model_path=source, model_paths={}, game=object(), device="cpu",
        ddp_enabled=False, world_size=1, clock=lambda: clock[0], max_seconds=40.0,
        trainer_factory=lambda cfg: FakeTrainer(clock),
        import_policy=lambda *args, **kwargs: {"hash": "import"},
        initial_state_fingerprint=lambda trainer: {"hash": "fp"}, config_override=FakeConfig(),
        pool_entries_override=(),
        policy_exporter=lambda trainer, path, iteration: Path(path).write_bytes(b"policy"),
    )
    diagnostics = read_json(run_dir / "diagnostics/latest.json")
    assert diagnostics["ntp"]["status"] == "DISABLED"
    assert diagnostics["ntp"]["loss"] is None


def test_resume_already_at_target_exports_without_zero_update_failure(tmp_path):
    from serial_compare.training import run_training_arm

    source, plan = _source_file(tmp_path)
    clock = [0.0]
    run_dir = tmp_path / "run"
    recovery = tmp_path / "iter000256.pt"
    recovery.write_bytes(b"checkpoint")
    result = run_training_arm(
        plan=plan, arm="control", run_dir=run_dir, source_payload={},
        source_model_path=source, model_paths={}, game=object(), device="cpu",
        ddp_enabled=False, world_size=1, clock=lambda: clock[0], max_seconds=100.0,
        trainer_factory=lambda cfg: FakeTrainer(clock),
        initial_state_fingerprint=lambda trainer: {"hash": "fp"}, config_override=FakeConfig(),
        pool_entries_override=(), resume_from=recovery,
        policy_exporter=lambda trainer, path, iteration: Path(path).write_bytes(b"policy"),
    )
    assert result["resumed_at_target"] is True
    assert result["applied_updates"] == 0
    assert result["new_iterations"] == 0
    assert Path(result["policy"]).is_file()


# ---------------------------------------------------------------------------
# P1-2 partial deals / truncated records
# ---------------------------------------------------------------------------
def test_partial_deal_is_excluded_and_reported_incomplete_not_an_exception():
    manifest = build_deal_manifest("confirmation")
    tasks = evaluation.confirmation_tasks(manifest)
    models = _final_models()
    records = [
        _record(tasks[0], models, win=True), _record(tasks[1], models, win=True),
        _record(tasks[2], models, win=True), _record(tasks[3], models, status="incomplete"),
    ]
    summary = evaluation.summarize_confirmation(records, tasks)
    assert summary["complete"] is False
    assert summary["verdict"]["verdict"] == "EVALUATION_INCOMPLETE"
    assert summary["independent_deals"] == 1
    assert summary["completed_games"] == 3
    assert summary["excluded_incomplete_deals"] == ["confirmation-0001"]


def test_zero_complete_deals_is_incomplete_not_an_exception():
    manifest = build_deal_manifest("confirmation")
    tasks = evaluation.confirmation_tasks(manifest)
    summary = evaluation.summarize_confirmation([], tasks)
    assert summary["complete"] is False
    assert summary["independent_deals"] == 0
    assert summary["win_rate"] is None
    assert summary["verdict"]["verdict"] == "EVALUATION_INCOMPLETE"


def test_all_complete_deals_enter_the_interval():
    manifest = build_deal_manifest("confirmation")
    tasks = evaluation.confirmation_tasks(manifest)
    models = _final_models()
    records = [_record(task, models, win=index % 2 == 0) for index, task in enumerate(tasks[:6])]
    summary = evaluation.summarize_confirmation(records, tasks)
    assert summary["complete"] is False  # only 6 of 256 games here
    assert summary["independent_deals"] == 3
    assert summary["clustered_ci95"]["clusters"] == 3


def test_development_summarize_arm_excludes_partial_deals():
    manifest = build_deal_manifest("development")
    tasks = evaluation.development_tasks(manifest)
    models = _final_models()
    arm_tasks = [task for task in tasks if task["arm"] == "control_vs_375"][:4]
    records = [
        _record(arm_tasks[0], models), _record(arm_tasks[1], models),
        _record(arm_tasks[2], models), _record(arm_tasks[3], models, status="incomplete"),
    ]
    summary = evaluation.summarize_arm(records, "control_vs_375")
    assert summary["independent_deals"] == 1
    assert summary["excluded_incomplete_deals"] == ["development-0001"]


def test_truncated_record_without_action_trace_raises_explicit_error():
    manifest = build_deal_manifest("confirmation")
    tasks = evaluation.confirmation_tasks(manifest)
    models = _final_models()
    truncated = _record(tasks[0], models, status="incomplete", trace="")
    with pytest.raises(evaluation.TruncatedGameError, match="no action-trace identity"):
        evaluation.validate_record(truncated, tasks[0], models=models)
    # The host supplies the trace, so a real truncated record validates.
    complete_trace = _record(tasks[0], models, status="incomplete", trace="f" * 64)
    evaluation.validate_record(complete_trace, tasks[0], models=models)


def test_evaluation_persists_each_game_before_playing_the_next(tmp_path, monkeypatch):
    from serial_compare import evaluation_runner

    manifest = build_deal_manifest("development")
    deals = [dict(row) for row in manifest["deals"][:3]]
    tasks = evaluation.tasks_for(deals, ["control_vs_375"])
    models = {"p375": {"sha256": "3" * 64}, "control_final": {"sha256": "4" * 64}}

    played: list[int] = []
    def fake_play_game(**_kwargs):
        played.append(1)
        if len(played) == 3:
            raise KeyboardInterrupt("simulated platform kill")
        return {
            "status": "complete", "base_deal_sha256": "a" * 64,
            "action_trace_sha256": "b" * 64, "candidate_win": True,
            "candidate_team_reward": 1.0, "token_steps": 10, "committed_steps": 4,
            "illegal_count": 0, "truncated_count": 0, "outcome_class": "HEAD_LAST",
            "winner_team": int(_kwargs["rotation"]) % 2,
            "ranking": [1, 2, 4, 3] if int(_kwargs["rotation"]) % 2 == 0 else [2, 1, 3, 4],
        }

    import guandan_marvel.evaluation.host as host

    class Cache:
        def get(self, name):
            return object()

    out = tmp_path / "games.rank0.jsonl"
    persisted = []
    def persist(record):
        persisted.append(record)
        with out.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()

    monkeypatch.setattr(host, "play_game", fake_play_game)
    with pytest.raises(KeyboardInterrupt):
        evaluation_runner.run_tasks(
            tasks, cache=Cache(), models=models, device="cpu",
            max_steps=1024, on_record=persist,
        )
    # Games 1..N-1 survive the kill; a restart resumes only the missing game.
    records, _ = evaluation.collect_records(tmp_path)
    assert len(records) == 2
    assert len(persisted) == 2
    assert [task["task_id"] for task in evaluation.pending_tasks(tasks, records)] == [
        task["task_id"] for task in tasks[2:]
    ]


def test_conflicting_duplicate_assignment_is_still_rejected():
    manifest = build_deal_manifest("confirmation")
    tasks = evaluation.confirmation_tasks(manifest)
    models = _final_models()
    left = _record(tasks[0], models, win=True)
    right = _record(tasks[0], models, win=False)
    with pytest.raises(evaluation.EvaluationError, match="two unique assignments"):
        evaluation.summarize_arm([left, right], "candidate_vs_control")


# ---------------------------------------------------------------------------
# P1-3 result ZIP recovery
# ---------------------------------------------------------------------------
def _fake_run_dir(artifacts: Path) -> Path:
    run = artifacts / "phases/train_control/run"
    latest = run / "checkpoints/iter000004"
    older = run / "checkpoints/iter000002"
    latest.mkdir(parents=True)
    older.mkdir(parents=True)
    (latest / "rank000.pt").write_bytes(b"latest-recovery")
    (latest / "manifest.json").write_bytes(b"latest-manifest")
    (older / "rank000.pt").write_bytes(b"old-recovery")
    index = {
        "schema": "guandan-serial-compare-single-process-checkpoint-index-v1",
        "latest": {"iteration": 4, "checkpoint": "checkpoints/iter000004"},
        "history": [
            {"iteration": 2, "checkpoint": "checkpoints/iter000002"},
            {"iteration": 4, "checkpoint": "checkpoints/iter000004"},
        ],
    }
    (run / "checkpoint_manifest.json").write_text(json.dumps(index), encoding="utf-8")
    return run


def test_result_zip_carries_latest_recovery_and_resume_inputs(tmp_path):
    artifacts = tmp_path / "artifacts"
    run = _fake_run_dir(artifacts)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "plan.json").write_text("{}", encoding="utf-8")
    (dataset / "deals_confirmation.json").write_text("{}", encoding="utf-8")
    written = write_reports(
        artifacts, {"plan_hash": "p", "status": "complete"}, resume_inputs=dataset,
    )
    archive = Path(written["archive"])
    assert archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert "artifacts/phases/train_control/run/checkpoints/iter000004/rank000.pt" in names
        assert "artifacts/phases/train_control/run/checkpoints/iter000002/rank000.pt" not in names
        assert "input/plan.json" in names
        assert "artifacts/recovery_index.json" in names
        assert "RESULTS_MANIFEST.json" in names
        index = json.loads(bundle.read("artifacts/recovery_index.json"))
        # Portable: the index must not embed a host-absolute run directory.
        assert index["arms"]["control"]["run_dir"] == "artifacts/phases/train_control/run"
        control_files = {row["path"]: row["sha256"] for row in index["arms"]["control"]["files"]}
        recovery_rel = "phases/train_control/run/checkpoints/iter000004/rank000.pt"
        assert control_files[recovery_rel] == sha(run / "checkpoints/iter000004/rank000.pt")
        manifest = json.loads(bundle.read("RESULTS_MANIFEST.json"))
        declared = {row["path"]: row["sha256"] for row in manifest["files"]}
        assert declared["artifacts/phases/train_control/run/checkpoints/iter000004/rank000.pt"] == control_files[recovery_rel]


def test_deleting_the_recovery_invalidates_the_completion_marker(tmp_path):
    artifacts = tmp_path / "artifacts"
    run = _fake_run_dir(artifacts)
    outputs = latest_recovery_files(run)
    assert outputs
    phase_dir = artifacts / "phases/train_control"
    payload = markers.write_marker(
        phase_dir, phase="train/control", plan_hash="p", source_hash="s",
        input_hashes={}, output_paths=outputs, root=artifacts,
    )
    assert payload["outputs"]
    assert markers.phase_is_complete(
        phase_dir, phase="train/control", plan_hash="p", source_hash="s",
        root=artifacts, input_hashes={},
    )[0]
    (run / "checkpoints/iter000004/rank000.pt").unlink()
    complete, reason = markers.phase_is_complete(
        phase_dir, phase="train/control", plan_hash="p", source_hash="s",
        root=artifacts, input_hashes={},
    )
    assert complete is False and "missing" in reason


def test_minimal_result_package_exists_before_training(tmp_path):
    artifacts = tmp_path / "artifacts"
    ensure_minimal_result(artifacts, plan_hash="p")
    archive = artifacts / "guandan_serial_compare_results.zip"
    assert archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert "artifacts/RESULT.json" in names
        assert "RESULTS_MANIFEST.json" in names


def test_summarize_outputs_bind_the_result_archive(tmp_path):
    artifacts = tmp_path / "artifacts"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "plan.json").write_text("{}", encoding="utf-8")

    class FakeOrchestrator:
        def __init__(self):
            self.state: dict[str, object] = {}

        def result(self):
            return {"plan_hash": "p", "status": "complete",
                    "phases": dict(self.state.get("phases") or {})}

    context = PhaseContext(
        plan=_plan(), root=artifacts, phase="summarize_and_package",
        phase_dir=artifacts / "phases/summarize_and_package", state={}, budget=None,
        model_paths={}, clock=lambda: 0.0, orchestrator=FakeOrchestrator(),
    )
    actions = ProductionActions(plan=_plan(), root=artifacts, dataset=dataset, source=tmp_path)
    payload = actions.summarize(context)
    outputs = [str(path) for path in payload["outputs"]]
    assert any(path.endswith(".zip") for path in outputs)
    assert payload["metadata"]["archive_sha256"] == sha(artifacts / "guandan_serial_compare_results.zip")
    result = read_json(artifacts / "RESULT.json")
    assert "summarize_and_package" in result["phases"]


# ---------------------------------------------------------------------------
# P2-3 resume fail-closed / P2-6 resume integrity
# ---------------------------------------------------------------------------
def test_missing_latest_recovery_fails_closed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoint_manifest.json").write_text(json.dumps({
        "schema": "x", "latest": {"iteration": 8, "checkpoint": "checkpoints/iter000008.pt"},
    }), encoding="utf-8")
    actions = ProductionActions(plan=_plan(), root=tmp_path, dataset=tmp_path, source=tmp_path)
    with pytest.raises(PlanError, match="missing"):
        actions._latest_recovery(run)


def test_tampered_single_process_recovery_hash_fails_closed(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/iter000008.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run / "checkpoint_manifest.json").write_text(json.dumps({
        "schema": "x",
        "latest": {"iteration": 8, "checkpoint": "checkpoints/iter000008.pt", "sha256": "0" * 64},
    }), encoding="utf-8")
    actions = ProductionActions(plan=_plan(), root=tmp_path, dataset=tmp_path, source=tmp_path)
    with pytest.raises(PlanError, match="hash changed"):
        actions._latest_recovery(run)


def _bundle(tmp_path: Path) -> tuple[Path, str]:
    directory = tmp_path / "bundle"
    directory.mkdir()
    overlay = {"guandan/x.py": b"x = 1\n"}
    driver = {"serial_compare/x.py": b"y = 2\n"}
    (directory / "plan.json").write_bytes(b"{}")
    (directory / "overlay_source.zip").write_bytes(zip_bytes(overlay))
    (directory / "driver.zip").write_bytes(zip_bytes(driver))
    manifest = {
        "schema": "guandan-serial-compare-delivery-manifest-v1",
        "files": [
            {"path": name, "bytes": (directory / name).stat().st_size, "sha256": sha(directory / name)}
            for name in ("plan.json", "overlay_source.zip", "driver.zip")
        ],
        "overlay_files": [
            {"path": name, "bytes": len(data), "sha256": __import__("hashlib").sha256(data).hexdigest()}
            for name, data in overlay.items()
        ],
        "driver_files": [
            {"path": name, "bytes": len(data), "sha256": __import__("hashlib").sha256(data).hexdigest()}
            for name, data in driver.items()
        ],
    }
    manifest["manifest_hash"] = canonical(manifest)
    (directory / "delivery_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, manifest["manifest_hash"]


def test_resume_bundle_verification_accepts_and_rejects_tampering(tmp_path):
    directory, manifest_hash = _bundle(tmp_path)
    assert verify_input_bundle(directory, manifest_hash, canonical, sha)["manifest_hash"] == manifest_hash
    # Same member name, different bytes: must be rejected before any training.
    (directory / "driver.zip").write_bytes(zip_bytes({"serial_compare/x.py": b"y = 3\n"}))
    with pytest.raises(RuntimeError, match="driver.zip"):
        verify_input_bundle(directory, manifest_hash, canonical, sha)
    # Even if the attacker also rewrites the top-level file hash, the declared
    # per-member hash of driver.zip must still reject the tampering.
    manifest = json.loads((directory / "delivery_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        if row["path"] == "driver.zip":
            row["sha256"] = sha(directory / "driver.zip")
            row["bytes"] = (directory / "driver.zip").stat().st_size
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = canonical(manifest)
    (directory / "delivery_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="driver.zip member changed"):
        verify_input_bundle(directory, manifest["manifest_hash"], canonical, sha)


def test_resume_bundle_from_a_different_manifest_is_rejected(tmp_path):
    directory, _ = _bundle(tmp_path)
    with pytest.raises(RuntimeError, match="different payload/manifest"):
        verify_input_bundle(directory, "9" * 64, canonical, sha)


# ---------------------------------------------------------------------------
# P2-5 portable pool manifest
# ---------------------------------------------------------------------------
def test_pool_manifest_identity_ignores_the_host_absolute_path(tmp_path):
    from guandan_marvel.phase6.opponent_pool import build_manifest

    payload = b"frozen-policy"
    first = tmp_path / "runA" / "anchor.pt"
    second = tmp_path / "runB" / "anchor.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(payload)
    second.write_bytes(payload)
    digest = sha(first)

    def entry(path: Path, digest_value: str = digest) -> dict[str, object]:
        return {
            "id": "p250", "kind": "v2_evaluation", "checkpoint": str(path),
            "checkpoint_sha256": digest_value, "architecture": "v1.4-macro-fuse",
            "protocol_version": "v2", "backend": "linux-cpp",
            "resolved_config_hash": "a" * 64, "source_hash": "b" * 64,
            "policy_version": 250, "weight": 1.0, "active": True, "tags": ["recent"],
        }

    moved = build_manifest(entries=[entry(second)])["manifest_hash"]
    original = build_manifest(entries=[entry(first)])["manifest_hash"]
    assert moved == original
    changed = build_manifest(entries=[entry(first, "c" * 64)])["manifest_hash"]
    assert changed != original


# ---------------------------------------------------------------------------
# P2-7 data isolation
# ---------------------------------------------------------------------------
def test_preflight_and_training_have_distinct_physical_first_deals():
    plan = _plan()
    manifests = {
        "preflight": build_deal_manifest("preflight", count=2),
        "development": build_deal_manifest("development", count=2),
        "confirmation": build_deal_manifest("confirmation", count=2),
    }
    isolation = common.assert_training_seeds_disjoint(plan, manifests, ranks=2, physical=True)
    assert isolation["physical_deal_probe"] == "RUN"
    trading = common.physical_deal_sha256(570_700)
    preflight = common.physical_deal_sha256(1_600_000_000)
    assert trading != preflight
    overlapping = copy.deepcopy(plan)
    overlapping["arms"]["control"]["seed"] = 1_620_000_000
    with pytest.raises(PlanError, match="overlap"):
        common.assert_training_seeds_disjoint(overlapping, manifests, ranks=2, physical=False)


# ---------------------------------------------------------------------------
# P3 inherited non-arm semantics / source identity / verifier
# ---------------------------------------------------------------------------
def test_train_config_inherits_source_schedule_debug_replay_and_totals():
    from guandan_marvel.config import GameSpec, REWARD_SEMANTICS, TRAINING_REWARD_SCALE
    from serial_compare.training import build_train_config

    plan = _plan()
    arm = plan["arms"]["control"]
    game = type("FakeGame", (), {"GAME_SPEC": GameSpec(
        identifier="guandan", num_players=4, num_tokens=64, pad_token=0, num_channels=4,
        max_game_horizon=1024, max_observation_tokens=64, max_action_tokens=64,
        max_available_actions=64,
    )})()
    source = {
        "resolved_config": {
            "model": dict(arm["model"]),
            "ppo": {
                "training_semantics_profile": arm["ppo"]["training_semantics_profile"],
                "policy_regularizer_mode": arm["ppo"]["policy_regularizer_mode"],
            },
            "schedule": {
                "agent_lr_decay_power": 0.0, "critic_lr_decay_power": 0.0,
                "clip_coef_decay_power": 0.0, "reg_coef_decay_power": 0.0,
            },
            "debug": False,
            "replay_buffer_scale": 8,
            "total_iterations": 375,
            "checkpoint_replay_buffer": True,
            "reward_semantics": REWARD_SEMANTICS,
            "reward_scale": TRAINING_REWARD_SCALE,
            "training_data_generator_version": "v",
            "training_context_schedule": None,
            "training_context_curriculum": None,
            "optimizer": {},
            "ddp": {},
        },
    }
    config = build_train_config(
        plan, "control", game=game, source_payload=source, device="cpu",
        ddp_enabled=False, world_size=1,
    )
    assert config.debug is False
    assert config.replay_buffer_scale == 8
    assert config.total_iterations == 375
    assert config.checkpoint_replay_buffer is True
    assert config.schedule.agent_lr_decay_power == 0.0
    assert config.seed == 570_700


def test_arm_rejects_a_source_policy_with_the_wrong_hash(tmp_path):
    from serial_compare.training import run_training_arm

    source, plan = _source_file(tmp_path)
    plan["models"]["p375"]["sha256"] = "0" * 64
    clock = [0.0]
    with pytest.raises(PlanError, match="source policy hash"):
        run_training_arm(
            plan=plan, arm="control", run_dir=tmp_path / "run", source_payload={},
            source_model_path=source, model_paths={}, game=object(), device="cpu",
            ddp_enabled=False, world_size=1, clock=lambda: clock[0], max_seconds=10.0,
            trainer_factory=lambda cfg: FakeTrainer(clock),
            import_policy=lambda *args, **kwargs: {"hash": "import"},
            initial_state_fingerprint=lambda trainer: {"hash": "fp"}, config_override=FakeConfig(),
            pool_entries_override=(),
            policy_exporter=lambda trainer, path, iteration: Path(path).write_bytes(b"policy"),
        )


def test_delivery_verifier_has_no_tautological_check():
    text = (REPO_ROOT / "scripts/verify_t4x2_serial_compare_delivery.py").read_text(encoding="utf-8")
    assert "or True" not in text


# ---------------------------------------------------------------------------
# Round 2: RESUME_RUN_ROOT driver/overlay directory integrity
# ---------------------------------------------------------------------------
def test_forged_driver_and_overlay_directories_are_rejected_before_use(tmp_path):
    directory, manifest_hash = _bundle(tmp_path)
    manifest = verify_input_bundle(directory, manifest_hash, canonical, sha)
    for archive, key, label, member in (
        ("driver.zip", "driver_files", "driver", "serial_compare/x.py"),
        ("overlay_source.zip", "overlay_files", "overlay", "guandan/x.py"),
    ):
        tree = tmp_path / f"{label}-{manifest_hash[:16]}"
        with zipfile.ZipFile(directory / archive) as bundle:
            bundle.extractall(tree)
        assert verify_member_tree(tree, manifest[key], label) is True
        original = (tree / member).read_bytes()
        (tree / member).write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match=f"{label} directory member changed"):
            verify_member_tree(tree, manifest[key], label)
        (tree / member).write_bytes(original)
        (tree / "evil.py").write_bytes(b"x = 1\n")
        with pytest.raises(RuntimeError, match="file set differs"):
            verify_member_tree(tree, manifest[key], label)


def test_bootstrap_uses_versioned_verified_driver_and_overlay_dirs():
    source = delivery_module.BOOTSTRAP
    assert "driver-' + EXPECTED_MANIFEST_HASH[:16]" in source
    assert "overlay-' + EXPECTED_MANIFEST_HASH[:16]" in source
    assert "verify_member_tree(" in source
    # Verification happens before the driver is ever put on sys.path.
    assert source.index("verify_member_tree(") < source.index("sys.path.insert(0, str(driver))")


# ---------------------------------------------------------------------------
# Round 2: hard-reserve CLI wiring / process group / summarize phase
# ---------------------------------------------------------------------------
def test_arm_command_forwards_hard_reserve_and_worker_exposes_the_flag(tmp_path):
    actions = ProductionActions(
        plan=_plan(), root=tmp_path, dataset=tmp_path, source=tmp_path, python="python",
    )
    command = actions._arm_command(
        "control", tmp_path / "run", tmp_path / "result.json", max_seconds=12.0,
        iteration_limit=None, resume_from=None, hard_reserve_seconds=123.5,
    )
    assert "--hard-reserve-seconds" in command
    assert command[command.index("--hard-reserve-seconds") + 1] == "123.5"

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "tools")
    completed = subprocess.run(
        [sys.executable, "-m", "serial_compare.arm_worker", "--help"],
        capture_output=True, text=True, env=environment, timeout=120,
    )
    assert completed.returncode == 0
    assert "--hard-reserve-seconds" in completed.stdout


def test_run_timeout_kills_the_whole_process_group(tmp_path, monkeypatch):
    from serial_compare import pipeline as pipeline_module

    events: list[object] = []

    class FakeProcess:
        pid = 4242
        returncode = -9

        def communicate(self, timeout=None):
            if "raised" not in events:
                events.append("raised")
                raise subprocess.TimeoutExpired("cmd", timeout)
            return ("", "")

        def kill(self):
            events.append("kill")

    def fake_popen(*_args, **kwargs):
        events.append(("popen", kwargs.get("start_new_session")))
        return FakeProcess()

    monkeypatch.setattr(pipeline_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pipeline_module.os, "name", "posix")
    monkeypatch.setattr(pipeline_module.os, "getpgid", lambda pid: 77, raising=False)
    monkeypatch.setattr(pipeline_module.os, "killpg", lambda pg, sig: events.append(("killpg", pg)), raising=False)
    actions = ProductionActions(
        plan=_plan(), root=tmp_path, dataset=tmp_path, source=tmp_path, python="python",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        actions._run(["cmd"], log=tmp_path / "log.txt", timeout=1.0)
    assert ("popen", True) in events
    assert ("killpg", 77) in events


# ---------------------------------------------------------------------------
# Round 2: budget.fit_one_iteration removed (no dead production evidence)
# ---------------------------------------------------------------------------
def test_budget_has_no_dead_fit_one_iteration_helper():
    import inspect

    from serial_compare import budget

    assert not hasattr(budget.BudgetLedger, "fit_one_iteration")
    source = inspect.getsource(budget)
    assert "def fit_one_iteration" not in source
