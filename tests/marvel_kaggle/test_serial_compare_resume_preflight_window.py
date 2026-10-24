from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from serial_compare import evaluation, pipeline  # noqa: E402
from serial_compare.pipeline import ProductionActions  # noqa: E402


class _Budget:
    def remaining_for_phase(self, _phase: str) -> float:
        return 1200.0


def _run_preflight(
    tmp_path: Path, monkeypatch, *, resume_updates: int,
    resume_status: str = "complete", resume_new_iterations: int = 3,
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "deals_preflight.json").write_text("{}", encoding="utf-8")

    plan = {
        "arms": {"candidate": {"seed": 570_700}},
        "models": {},
    }
    phase_dir = tmp_path / "phase"
    context = SimpleNamespace(
        phase="preflight/candidate",
        phase_dir=phase_dir,
        budget=_Budget(),
    )
    actions = ProductionActions(
        plan=plan, root=tmp_path / "artifacts", dataset=dataset, source=tmp_path,
        python="python",
    )
    commands: list[list[str]] = []

    def fake_run(command, *, log, timeout, cwd=None):
        del log, timeout, cwd
        command = list(command)
        commands.append(command)
        result_path = Path(command[command.index("--result") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if "--resume-from" not in command:
            recovery = tmp_path / "recovery" / "iter000002"
            recovery.mkdir(parents=True)
            payload = {
                "status": "complete",
                "start_iteration": 0,
                "final_iteration": 2,
                "new_iterations": 2,
                "applied_updates": 2,
                "recovery": str(recovery),
            }
        else:
            policy = tmp_path / "resume-policy.pt"
            policy.write_bytes(b"policy")
            payload = {
                "status": resume_status,
                "start_iteration": 2,
                "final_iteration": 2 + resume_new_iterations,
                "new_iterations": resume_new_iterations,
                "applied_updates": resume_updates,
                "policy": str(policy),
                "initial_state_fingerprint": {"hash": "fingerprint"},
                "opponent_pool_manifest_hash": "pool-hash",
                "iterations": [
                    {"iteration": 3, "learner_updates": 0, "kl_stopped": True},
                    {"iteration": 4, "learner_updates": resume_updates, "kl_stopped": False},
                    {"iteration": 5, "learner_updates": 0, "kl_stopped": False},
                ],
            }
        result_path.write_text(json.dumps(payload), encoding="utf-8")

    tasks = [{"task_id": "one"}, {"task_id": "two"}]
    records = [{"task_id": "one"}, {"task_id": "two"}]
    monkeypatch.setattr(actions, "_run", fake_run)
    monkeypatch.setattr(actions, "_run_eval", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "validate_deal_manifest", lambda *args, **kwargs: [{}, {}])
    monkeypatch.setattr(pipeline, "collect_records", lambda *args, **kwargs: (records, []))
    monkeypatch.setattr(evaluation, "tasks_for", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(
        evaluation, "validate_preflight_records",
        lambda actual_records, actual_tasks, **kwargs: list(actual_records),
    )

    result = actions.preflight(context, "candidate")
    return result, commands, phase_dir


def _iteration_limit(command: list[str]) -> int:
    return int(command[command.index("--iteration-limit") + 1])


def test_resume_preflight_uses_three_iteration_window_and_records_evidence(
    tmp_path, monkeypatch,
):
    result, commands, phase_dir = _run_preflight(
        tmp_path, monkeypatch, resume_updates=1,
    )

    assert pipeline.PREFLIGHT_RESUME_ITERATION_WINDOW == 3
    assert pipeline.PREFLIGHT_RESUME_ITERATION_WINDOW > 0
    assert [_iteration_limit(command) for command in commands] == [2, 3]
    assert "--resume-from" not in commands[0]
    assert "--resume-from" in commands[1]

    payload = json.loads((phase_dir / "preflight.json").read_text(encoding="utf-8"))
    assert {
        "schema", "arm", "status", "train_updates", "resume_updates",
        "iteration_after_resume", "evaluated_games", "initial_state_fingerprint",
        "opponent_pool_manifest_hash", "kl_stop_path_exercised", "hash",
    } <= payload.keys()
    assert payload["resume_iteration_window"] == 3
    assert payload["resume_new_iterations"] == 3
    assert payload["resume_kl_stops"] == 1
    assert payload["resume_updates"] == 1
    assert payload["kl_stop_path_exercised"] is True
    assert result["status"] == "pass"


def test_resume_preflight_rejects_all_zero_updates(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="produced no effective update"):
        _run_preflight(tmp_path, monkeypatch, resume_updates=0)


def test_resume_preflight_rejects_a_budget_truncated_window(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="bounded iteration window"):
        _run_preflight(
            tmp_path, monkeypatch, resume_updates=1,
            resume_status="budget_stopped", resume_new_iterations=1,
        )

