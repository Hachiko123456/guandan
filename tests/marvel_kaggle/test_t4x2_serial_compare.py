"""Serial recipe comparison: orchestration, budget, recovery, evaluation, delivery.

All orchestration tests inject fake actions, so they run on CPU without the C++
native, two GPUs or any real training.  The statistics references are computed
by hand (never with the same helper the production code uses).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT / "tools"))

from serial_compare import common, evaluation, markers, orchestrator  # noqa: E402
from serial_compare.budget import BudgetError, BudgetLedger  # noqa: E402
from serial_compare.common import (  # noqa: E402
    PlanError, bounded_cluster_ci, build_deal_manifest, build_plan, canonical, formal_verdict,
    read_json, sha, validate_deal_manifest, validate_plan,
)
from serial_compare.delivery import (  # noqa: E402
    MAGIC, collect_driver, collect_overlay, decode_payload, zip_bytes,
)
from serial_compare.orchestrator import OrchestratorError, SerialCompareOrchestrator  # noqa: E402

MODEL_HASHES = {"p375": "1" * 64, "p250": "2" * 64, "p500": "3" * 64}


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


class FakeActions:
    """Deterministic fake actions with injectable failures and a call log."""

    def __init__(self, root: Path, *, fail_at: str | None = None, confirmation_complete: bool = True):
        self.root = root
        self.fail_at = fail_at
        self.confirmation_complete = confirmation_complete
        self.calls: list[str] = []
        self.shared_fingerprint = {"hash": "shared" + "0" * 58, "tensor_count": 5}

    def _out(self, context, name: str, payload: Mapping[str, object]) -> Path:
        path = context.phase_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _maybe_fail(self, name: str) -> None:
        if self.fail_at == name:
            raise RuntimeError(f"injected failure at {name}")

    def input_check(self, context):
        self.calls.append("input_check")
        self._maybe_fail("input_check")
        return {"status": "complete", "outputs": [self._out(context, "inputs.json", {"ok": True})]}

    def preflight(self, context, arm):
        self.calls.append(f"preflight:{arm}")
        self._maybe_fail(f"preflight:{arm}")
        return {
            "status": "pass",
            "outputs": [self._out(context, "preflight.json", {"arm": arm, "status": "pass"})],
            "metadata": {"initial_state_fingerprint": self.shared_fingerprint,
                         "pool_manifest_hash": None if arm == "control" else "poolhash"},
        }

    def freeze_budget(self, context, resolved):
        self.calls.append("freeze_budget")
        self._maybe_fail("freeze_budget")
        return {"status": "complete", "outputs": [self._out(context, "resolved_budget.json", resolved)]}

    def train(self, context, arm, *, max_seconds):
        self.calls.append(f"train:{arm}")
        self._maybe_fail(f"train:{arm}")
        policy = context.phase_dir / f"policy_{arm}.pt"
        policy.parent.mkdir(parents=True, exist_ok=True)
        policy.write_text(f"{arm}-final-policy", encoding="utf-8")
        metadata = {
            "arm": arm,
            "start_iteration": 0,
            "final_iteration": 4,
            "applied_updates": 8,
            "initial_source_sha256": MODEL_HASHES["p375"],
            "initial_state_fingerprint": self.shared_fingerprint,
            "policy": str(policy),
            "policy_sha256": sha(policy),
            "target_kl": None if arm == "control" else 0.015,
            "ntp_enabled": arm == "control",
            "opponent_pool_manifest_hash": None if arm == "control" else "poolhash",
            "max_seconds": max_seconds,
        }
        return {"status": "complete", "outputs": [policy], "metadata": metadata}

    def freeze_finals(self, context):
        self.calls.append("freeze_finals")
        self._maybe_fail("freeze_finals")
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
        self.calls.append(f"evaluate:{phase}")
        self._maybe_fail(f"evaluate:{phase}")
        domain = "confirmation" if phase.endswith("confirmation_primary") else "development"
        manifest = build_deal_manifest(domain)
        tasks = evaluation.confirmation_tasks(manifest) if domain == "confirmation" else evaluation.development_tasks(manifest)
        if domain == "confirmation" and not self.confirmation_complete:
            tasks = tasks[:10]
        models = dict(_models())
        models["control_final"] = {"sha256": sha(Path(context.final_paths()["control_final"]))}
        models["candidate_final"] = {"sha256": sha(Path(context.final_paths()["candidate_final"]))}
        records = [_record(task, models, win=True) for task in tasks]
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
        self.calls.append("summarize")
        self._maybe_fail("summarize")
        result = context.orchestrator.result() if context.orchestrator else {}
        path = self._out(context, "RESULT.json", result)
        return {"status": "complete", "outputs": [path]}


def _record(task: Mapping[str, object], models: Mapping[str, Mapping[str, object]], *, win: bool) -> dict[str, object]:
    candidate_key, opponent_key = evaluation.ARM_SPECS[task["arm"]]
    assignment = int(task["assignment"])
    record = {
        "task_id": task["task_id"],
        "arm": task["arm"],
        "deal_id": task["deal_id"],
        "deal_seed": task["deal_seed"],
        "seat_rotation": assignment,
        "candidate_team": assignment % 2,
        "temperature": 0.0,
        "evaluation_horizon": 1024,
        "base_deal_sha256": canonical({"deal": task["deal"]}),
        "action_trace_sha256": "a" * 64,
        "candidate_checkpoint_sha256": str(models[candidate_key]["sha256"]),
        "opponent_checkpoint_sha256": None if opponent_key is None else str(models[opponent_key]["sha256"]),
        "status": "complete",
        "candidate_win": bool(win),
        "candidate_team_reward": 1.0 if win else -1.0,
        "token_steps": 20,
        "committed_steps": 8,
        "illegal_count": 0,
        "truncated_count": 0,
        "outcome_class": "HEAD_LAST",
        "winner_team": (assignment % 2) if win else 1 - assignment % 2,
    }
    record["ranking"] = [1, 2, 4, 3] if record["winner_team"] == 0 else [2, 1, 3, 4]
    record["task_hash"] = canonical(dict(task))
    record["record_hash"] = canonical(record)
    return record


def _orchestrator(root: Path, actions: FakeActions, clock) -> SerialCompareOrchestrator:
    model_paths = {
        "p375": root / "policy_375.pt", "p250": root / "anchor_250.pt", "p500": root / "policy_500.pt",
    }
    return SerialCompareOrchestrator(
        _plan(), root=root / "artifacts", actions=actions, model_paths=model_paths, clock=clock,
    )


# ---------------------------------------------------------------------------
# plan / deals / tasks
# ---------------------------------------------------------------------------
def test_plan_is_frozen_and_task_counts_are_704():
    plan = _plan()
    validate_plan(plan)
    assert plan["arms"]["control"]["target_kl"] is None
    assert plan["arms"]["candidate"]["target_kl"] == 0.015
    assert plan["arms"]["candidate"]["opponent_pool"]["max_resident_policies"] >= 2
    confirmation = build_deal_manifest("confirmation")
    development = build_deal_manifest("development")
    preflight = build_deal_manifest("preflight")
    assert len(evaluation.confirmation_tasks(confirmation)) == 256
    assert len(evaluation.development_tasks(development)) == 448
    assert 256 + 448 == 704
    disjoint = common.assert_domains_disjoint(
        {"confirmation": confirmation, "development": development, "preflight": preflight}
    )
    assert disjoint["disjoint"] is True
    ids = [task["task_id"] for task in evaluation.confirmation_tasks(confirmation)]
    assert len(ids) == len(set(ids))
    assert all(task["assignment"] in (0, 1) for task in evaluation.confirmation_tasks(confirmation))
    assert {task["deal_id"] for task in evaluation.confirmation_tasks(confirmation)}.isdisjoint(
        {task["deal_id"] for task in evaluation.development_tasks(development)}
    )


def test_plan_rejects_a_changed_recipe():
    plan = json.loads(json.dumps(_plan()))
    plan["arms"]["control"]["ppo"]["base_lr"] = 3e-4
    plan.pop("plan_hash")
    plan["plan_hash"] = canonical(plan)
    with pytest.raises(PlanError, match="learning recipe"):
        validate_plan(plan)


# ---------------------------------------------------------------------------
# statistics (hand-computed)
# ---------------------------------------------------------------------------
def test_bounded_cluster_ci_matches_the_hand_computation():
    values = [0.0, 1.0]
    estimate = 0.5
    variance = ((0.0 - estimate) ** 2 + (1.0 - estimate) ** 2) / (len(values) - 1)
    log_term = math.log(3.0 / 0.05)
    margin = math.sqrt(2.0 * variance * log_term / len(values)) + 3.0 * log_term / len(values)
    ci = bounded_cluster_ci(values)
    assert ci["mean"] == pytest.approx(estimate)
    assert ci["sample_variance"] == pytest.approx(variance)
    assert ci["low"] == pytest.approx(max(0.0, estimate - margin))
    assert ci["high"] == pytest.approx(min(1.0, estimate + margin))
    assert bounded_cluster_ci([0.7])["interval_available"] is False


def test_formal_verdict_rules():
    assert formal_verdict({"low": 0.55, "high": 0.9, "interval_available": True}, complete=True)["verdict"] == "SUPPORTED"
    assert formal_verdict({"low": 0.1, "high": 0.45, "interval_available": True}, complete=True)["verdict"] == "WORSE"
    assert formal_verdict({"low": 0.3, "high": 0.7, "interval_available": True}, complete=True)["verdict"] == "INCONCLUSIVE"
    assert formal_verdict({"low": 0.9, "high": 1.0, "interval_available": True}, complete=False)["verdict"] == "EVALUATION_INCOMPLETE"
    for verdict in ("SUPPORTED", "WORSE", "INCONCLUSIVE"):
        assert formal_verdict({"low": 0.6, "high": 0.9, "interval_available": True}, complete=True)["formal_production_promotion"] is False


def test_records_cluster_by_deal_and_reject_duplicate_assignments():
    manifest = build_deal_manifest("confirmation")
    tasks = evaluation.confirmation_tasks(manifest)
    models = dict(_models())
    models["control_final"] = {"sha256": "4" * 64}
    models["candidate_final"] = {"sha256": "5" * 64}
    records = [_record(task, models, win=True) for task in tasks[:4]]
    summary = evaluation.summarize_arm(records, "candidate_vs_control")
    assert summary["independent_deals"] == 2
    assert summary["win_rate"] == 1.0
    broken = [dict(record) for record in records[:2]]
    broken[1]["seat_rotation"] = 0
    broken[1]["candidate_team"] = 0
    with pytest.raises(evaluation.EvaluationError, match="two unique assignments"):
        evaluation.summarize_arm(broken, "candidate_vs_control")


def test_duplicate_records_are_idempotent_but_conflicts_are_rejected():
    manifest = build_deal_manifest("confirmation")
    tasks = evaluation.confirmation_tasks(manifest)
    models = dict(_models())
    models["control_final"] = {"sha256": "4" * 64}
    models["candidate_final"] = {"sha256": "5" * 64}
    record = _record(tasks[0], models, win=True)
    merged, stats = evaluation.merge_records([record], [dict(record)], tasks=tasks, models=models)
    assert stats["duplicate_identical_records"] == 1 and len(merged) == 1
    # Both alternatives are internally valid and signed, but conflict for the
    # same task. Unsigned mutations are rejected earlier by the hash guard.
    conflict = _record(tasks[0], models, win=False)
    with pytest.raises(evaluation.EvaluationError, match="conflicting duplicate"):
        evaluation.merge_records([record], [conflict], tasks=tasks, models=models)


def test_evaluation_incomplete_is_not_a_pass():
    manifest = build_deal_manifest("confirmation")
    tasks = evaluation.confirmation_tasks(manifest)
    models = dict(_models())
    models["control_final"] = {"sha256": "4" * 64}
    models["candidate_final"] = {"sha256": "5" * 64}
    partial = [_record(task, models, win=True) for task in tasks[:8]]
    summary = evaluation.summarize_confirmation(partial, tasks)
    assert summary["complete"] is False
    assert summary["verdict"]["verdict"] == "EVALUATION_INCOMPLETE"
    assert summary["verdict"]["formal_production_promotion"] is False


def test_schedule_balances_heavy_two_model_arms():
    tasks = evaluation.development_tasks(build_deal_manifest("development"))
    shards = evaluation.schedule(tasks, ranks=2)
    flattened = [task["task_id"] for shard in shards for task in shard]
    assert sorted(flattened) == sorted(task["task_id"] for task in tasks)
    heavy = sum(2 if evaluation.ARM_SPECS[task["arm"]][1] is not None else 1 for task in tasks)
    loads = [sum(2 if evaluation.ARM_SPECS[task["arm"]][1] is not None else 1 for task in shard) for shard in shards]
    assert abs(loads[0] - loads[1]) <= 1
    assert heavy == sum(loads)


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------
def test_budget_never_resets_on_resume_and_uses_the_safe_minimum():
    now = [0.0]
    ledger = BudgetLedger.start(_plan()["budget"], clock=lambda: now[0])
    now[0] = 100.0
    ledger.record("train/control", 100.0)
    state = ledger.to_state()
    now[0] = 50.0  # a new session starts earlier than the persisted spend
    resumed = BudgetLedger.resume(_plan()["budget"], clock=lambda: now[0], state=state)
    assert resumed.cumulative_spent == pytest.approx(100.0)
    assert resumed.cumulative_remaining() == pytest.approx(14400 - 100)
    assert resumed.session_remaining() == pytest.approx(14400)
    assert resumed.remaining_total() == pytest.approx(14400 - 100)


def test_symmetric_reduction_only_before_training_and_once():
    now = [0.0]
    ledger = BudgetLedger.start(_plan()["budget"], clock=lambda: now[0])
    resolved = ledger.apply_symmetric_reduction(0.5)
    assert resolved["symmetric"] is True and resolved["frozen"] is True
    assert resolved["train_control_seconds"] == resolved["train_candidate_seconds"]
    assert resolved["arm_quota_seconds"] == 1200
    with pytest.raises(BudgetError, match="immutable"):
        ledger.apply_symmetric_reduction(0.4)
    other = BudgetLedger.start(_plan()["budget"], clock=lambda: 0.0)
    other.mark_training_started("control")
    with pytest.raises(BudgetError, match="before either arm"):
        other.apply_symmetric_reduction(0.5)


def test_formal_training_requires_both_arm_quota_and_evaluation_reserve():
    now = [0.0]
    ledger = BudgetLedger.start(_plan()["budget"], clock=lambda: now[0])
    ledger.assert_can_start_formal_training()
    now[0] = 9000.0  # session nearly exhausted
    with pytest.raises(BudgetError, match="insufficient budget"):
        ledger.assert_can_start_formal_training()


def test_budget_state_rejects_a_different_plan():
    now = [0.0]
    ledger = BudgetLedger.start(_plan()["budget"], clock=lambda: now[0])
    state = ledger.to_state()
    tampered = dict(state)
    tampered["cumulative_spent"] = 1.0
    with pytest.raises(BudgetError, match="hash is inconsistent"):
        BudgetLedger.resume(_plan()["budget"], clock=lambda: 0.0, state=tampered)


# ---------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------
def test_marker_detects_missing_and_tampered_outputs(tmp_path):
    output = tmp_path / "phase" / "out.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    payload = markers.write_marker(
        output.parent, phase="bootstrap/input_check", plan_hash="p", source_hash="s",
        input_hashes={"x": "y"}, output_paths=[output], root=tmp_path,
    )
    assert markers.phase_is_complete(
        output.parent, phase="bootstrap/input_check", plan_hash="p", source_hash="s",
        root=tmp_path, input_hashes={"x": "y"},
    )[0]
    output.write_text("{\"tampered\": true}", encoding="utf-8")
    complete, reason = markers.phase_is_complete(
        output.parent, phase="bootstrap/input_check", plan_hash="p", source_hash="s",
        root=tmp_path, input_hashes={"x": "y"},
    )
    assert complete is False and "hash changed" in reason
    assert payload["hash"] == canonical({k: v for k, v in payload.items() if k != "hash"})


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def test_happy_path_runs_both_preflights_before_any_training(tmp_path):
    now = [0.0]
    actions = FakeActions(tmp_path)
    result = _orchestrator(tmp_path, actions, lambda: now[0]).run()
    order = actions.calls
    first_train = min(index for index, name in enumerate(order) if name.startswith("train:"))
    assert order.index("preflight:control") < order.index("preflight:candidate") < first_train
    assert result["status"] == "complete"
    assert result["phases"]["evaluate/confirmation_primary"]["status"] == "complete"
    assert result["confirmation"]["verdict"]["verdict"] == "SUPPORTED"
    assert result["formal_production_promotion"] is False
    assert result["fabledan"] == "NOT_RUN" and result["final_holdout"] == "NOT_TOUCHED"


def test_candidate_preflight_failure_prevents_every_training_call(tmp_path):
    now = [0.0]
    actions = FakeActions(tmp_path, fail_at="preflight:candidate")
    with pytest.raises(RuntimeError, match="injected failure"):
        _orchestrator(tmp_path, actions, lambda: now[0]).run()
    assert not any(name.startswith("train:") for name in actions.calls)


def test_resume_after_both_arms_trained_does_zero_training(tmp_path):
    now = [0.0]
    actions = FakeActions(tmp_path, fail_at="evaluate:evaluate/confirmation_primary")
    orchestrator_instance = _orchestrator(tmp_path, actions, lambda: now[0])
    with pytest.raises(RuntimeError, match="injected failure"):
        orchestrator_instance.run()
    assert actions.calls.count("train:control") == 1
    assert actions.calls.count("train:candidate") == 1

    resumed_actions = FakeActions(tmp_path)
    resumed = _orchestrator(tmp_path, resumed_actions, lambda: now[0]).run()
    assert not any(name.startswith("train:") for name in resumed_actions.calls)
    assert not any(name.startswith("preflight:") for name in resumed_actions.calls)
    assert resumed["status"] == "complete"


def test_completed_control_is_never_retrained(tmp_path):
    now = [0.0]
    actions = FakeActions(tmp_path, fail_at="train:candidate")
    with pytest.raises(RuntimeError, match="injected failure"):
        _orchestrator(tmp_path, actions, lambda: now[0]).run()
    assert actions.calls.count("train:control") == 1
    resumed_actions = FakeActions(tmp_path)
    _orchestrator(tmp_path, resumed_actions, lambda: now[0]).run()
    assert not any(name == "train:control" for name in resumed_actions.calls)
    assert resumed_actions.calls.count("train:candidate") == 1


def test_tampered_completed_phase_is_refused_instead_of_rerun(tmp_path):
    now = [0.0]
    actions = FakeActions(tmp_path, fail_at="train:candidate")
    with pytest.raises(RuntimeError):
        _orchestrator(tmp_path, actions, lambda: now[0]).run()
    control_policy = tmp_path / "artifacts/phases/train_control/policy_control.pt"
    assert control_policy.is_file()
    control_policy.write_text("tampered", encoding="utf-8")
    resumed_actions = FakeActions(tmp_path)
    with pytest.raises(OrchestratorError, match="no longer valid"):
        _orchestrator(tmp_path, resumed_actions, lambda: now[0]).run()
    assert not any(name == "train:control" for name in resumed_actions.calls)


def test_candidate_may_not_start_from_the_control_final(tmp_path):
    class BadActions(FakeActions):
        def train(self, context, arm, *, max_seconds):
            result = super().train(context, arm, max_seconds=max_seconds)
            if arm == "candidate":
                result["metadata"]["initial_source_sha256"] = context.arm_result("control")["policy_sha256"]
            return result

    now = [0.0]
    with pytest.raises(PlanError, match="control's final policy"):
        _orchestrator(tmp_path, BadActions(tmp_path), lambda: now[0]).run()


def test_incomplete_confirmation_skips_development_and_is_not_reported_complete(tmp_path):
    now = [0.0]
    actions = FakeActions(tmp_path, confirmation_complete=False)
    result = _orchestrator(tmp_path, actions, lambda: now[0]).run()
    assert not any(name.startswith("evaluate:evaluate/development") for name in actions.calls)
    assert result["status"] == "incomplete"
    assert result["confirmation"]["verdict"]["verdict"] == "EVALUATION_INCOMPLETE"
    assert result["technical_failure"] is None
    assert "summarize" in actions.calls


def test_technical_failure_is_reported_as_failed_not_complete(tmp_path):
    now = [0.0]
    actions = FakeActions(tmp_path, fail_at="summarize")
    with pytest.raises(RuntimeError, match="injected failure"):
        _orchestrator(tmp_path, actions, lambda: now[0]).run()
    state = read_json(tmp_path / "artifacts/state.json")
    assert state["status"] == "failed"
    assert state["failed_phase"] == "summarize_and_package"
    assert "injected failure" in state["error"]


def test_preflight_and_formal_arms_share_one_bitwise_initial_policy(tmp_path):
    now = [0.0]
    actions = FakeActions(tmp_path)
    result = _orchestrator(tmp_path, actions, lambda: now[0]).run()
    assert result["finals"]["pair"]["shared_bitwise"] is True
    assert result["finals"]["pair"]["candidate_not_initialised_from_control"] is True
    for arm in ("control", "candidate"):
        assert result["arms"][arm]["initial_source_sha256"] == MODEL_HASHES["p375"]


# ---------------------------------------------------------------------------
# delivery helpers
# ---------------------------------------------------------------------------
def test_overlay_is_python_only_and_contains_the_new_modules():
    overlay = collect_overlay(REPO_ROOT)
    for name in (
        "guandan_marvel/training/policy_import.py",
        "guandan_marvel/training/local.py",
        "guandan_marvel/training/macro_ppo.py",
        "guandan_marvel/config.py",
        "guandan_marvel/phase6/opponent_pool.py",
        "tools/serial_compare/orchestrator.py",
        "tools/serial_compare/pipeline.py",
    ):
        assert name in overlay, name
    for name, data in overlay.items():
        assert Path(name).suffix == ".py"
        assert (REPO_ROOT / name).read_bytes() == data
    driver = collect_driver(REPO_ROOT)
    assert "serial_compare/arm_worker.py" in driver and "serial_compare/eval_worker.py" in driver


def test_opaque_payload_roundtrip_and_rejection(tmp_path):
    dataset = {"plan.json": b"{}", "delivery_manifest.json": b"{}"}
    payload = MAGIC + __import__("base64").b64encode(zip_bytes(dataset)) + b"\n"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    decoded = decode_payload(path, tmp_path / "decoded")
    assert decoded["sha256"] == sha(path)
    assert (tmp_path / "decoded/plan.json").read_bytes() == b"{}"
    wrong = tmp_path / "wrong.bin"
    wrong.write_bytes(b"NOT-THE-MAGIC\n")
    with pytest.raises(common.PlanError, match="transport version"):
        decode_payload(wrong, tmp_path / "other")
