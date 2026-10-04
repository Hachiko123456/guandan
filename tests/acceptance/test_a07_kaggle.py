"""A07 acceptance: remote_full orchestration is specified, not locally claimed.

This module never launches a Kaggle job, never calls a remote API, and never
runs remote_full training. The only real execution is the bounded A04 smoke
and the separately scoped local_fast recovery tests in test_kaggle_recovery.py.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from guandan.deployment import session
from guandan.deployment.control import CheckpointController
from guandan.training.trainer import TrainingResult

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "project_status" / "STATUS.yaml"


def _synthetic_evaluation_checkpoint(update, algorithm="ippo"):
    """Synthetic counters, NOT a trained model or remote execution evidence."""
    profile = session._profile()
    cfg = profile["config"]["training"]
    steps = update * cfg["token_steps_per_update"]
    return SimpleNamespace(
        metadata=SimpleNamespace(profile="remote_full", algorithm=algorithm,
                                 update_count=update, git_commit="a" * 40),
        training_state={
            "compatibility": {"profile_sha256": profile["sha256"],
                              "envs": cfg["rollout_envs"],
                              "token_steps_per_update": cfg["token_steps_per_update"]},
            "engine": {"algorithm": algorithm, "update_count": update,
                       "collector": {"total_token_steps": steps,
                                     "envs": [b"synthetic-not-an-environment"] * cfg["rollout_envs"],
                                     "ticks_per_env": [steps // cfg["rollout_envs"]] * cfg["rollout_envs"],
                                     "incomplete": False}},
        },
    )


def _synthetic_evaluation_report():
    """Canonical-sized synthetic records only; no real games are played."""
    cfg = session._profile()["config"]["evaluation"]
    per_opponent = cfg["deal_groups_per_pairing"] * len(cfg["seat_rotations"])
    return {"profile": "remote_full", "status": "complete",
            "target_games_per_opponent": per_opponent,
            "completed_games_per_opponent": {name: per_opponent for name in cfg["opponents"]},
            "total_games": per_opponent * len(cfg["opponents"]),
            "records": [{"opponent": opponent, "deal_group": group, "seat_rotation": rotation,
                         "done": True, "terminal": True, "truncated": False}
                        for opponent in cfg["opponents"]
                        for group in range(cfg["deal_groups_per_pairing"])
                        for rotation in cfg["seat_rotations"]],
            "errors": [], "kaggle_verified": False,
            "fixture_scope": "SYNTHETIC COUNTERS AND RECORDS; NOT REAL TRAINING OR EVALUATION"}


def parsed_args(*tokens: str):
    return session.parser().parse_args(tokens)


@pytest.fixture(autouse=True)
def never_run_remote_work(monkeypatch):
    def prohibited(**kwargs):
        raise AssertionError("A07 local test must not run real remote_full training/evaluation")
    monkeypatch.setattr("guandan.training.trainer.train", prohibited)
    monkeypatch.setattr("guandan.evaluation.evaluate_profile", prohibited)


@pytest.fixture
def local_session(tmp_path, monkeypatch):
    """Explicit fake clean-source/runtime seam; not deployment provenance proof."""
    output = tmp_path / "ready_package"
    monkeypatch.setattr(session, "_output_directory", lambda ignored: output)
    monkeypatch.setattr(session, "source_provenance", lambda root: {
        "git_commit": "a" * 40, "git_worktree_clean": True,
        "source_kind": "SYNTHETIC_LOCAL_SESSION_TEST_ONLY",
        "source_manifest_sha256": None,
    })
    return output


def assert_not_kaggle_acceptance(record: dict):
    assert record["accepted"] is False
    assert record["kaggle_verified"] is False


def test_remote_full_profile_and_cli_contract_are_explicit():
    args = parsed_args(
        "--action", "train", "--profile", "remote_full", "--execute",
        "--output-root", "/kaggle/working/guandan",
        "--resume", "/kaggle/input/recovery/manifest.json",
        "--session-hours", "10", "--save-margin-seconds", "300",
        "--checkpoint-seconds", "600",
        "--candidate-checkpoint", "/kaggle/input/candidate.pt",
        "--snapshot-checkpoint", "/kaggle/input/snapshot.pt",
    )
    assert args.action == "train"
    assert args.profile == "remote_full"
    assert args.execute is True
    assert args.output_root == Path("/kaggle/working/guandan")
    assert args.resume == Path("/kaggle/input/recovery/manifest.json")
    assert (args.session_hours, args.save_margin_seconds, args.checkpoint_seconds) == (10, 300, 600)
    assert args.candidate_checkpoint == Path("/kaggle/input/candidate.pt")
    assert args.snapshot_checkpoint == Path("/kaggle/input/snapshot.pt")

    plan = session.run(parsed_args("--action", "plan"))
    assert plan["status"] == "configuration_only"
    assert plan["plan"]["profile"]["name"] == "remote_full"
    assert plan["plan"]["base_target"] == 100
    assert plan["plan"]["resume_target"] == 2
    assert plan["plan"]["goal_update"] == 100
    assert plan["plan"]["automatic_target_reduction"] is False
    assert_not_kaggle_acceptance(plan)


def test_training_plan_has_100_then_102_resume_boundaries():
    fresh = session.training_plan("ippo")
    base_continuation = session.training_plan("ippo", {"durable_updates": 99})
    resume_verification = session.training_plan("ippo", {"durable_updates": 100})
    assert (fresh["start_update"], fresh["goal_update"], fresh["updates_this_session_if_time_allows"]) == (0, 100, 100)
    assert (base_continuation["start_update"], base_continuation["goal_update"], base_continuation["updates_this_session_if_time_allows"]) == (99, 100, 1)
    assert base_continuation["phase"] == "base_continuation"
    assert (resume_verification["start_update"], resume_verification["goal_update"], resume_verification["updates_this_session_if_time_allows"]) == (100, 102, 2)
    assert resume_verification["phase"] == "resume_verification"


@pytest.mark.parametrize("action", ["smoke", "train", "evaluate"])
def test_execute_is_an_explicit_opt_in_for_non_plan_actions(action):
    with pytest.raises(ValueError, match="--execute"):
        session.run(parsed_args("--action", action))


def test_system_action_is_read_only_and_never_starts_training():
    before = STATUS.read_bytes()
    # Local tests must explicitly inject the runtime seam; the production CLI
    # fails closed when CUDA/Kaggle facts are unavailable.
    result = session.run(parsed_args("--action", "system"), runtime_check=lambda _: None)
    after = STATUS.read_bytes()
    assert result["status"] == "environment_inspected"
    assert result["executed_training"] is False
    assert result["accepted"] is False
    assert result["kaggle_verified"] is False
    assert before == after


def test_system_action_without_runtime_seam_fails_closed(monkeypatch):
    monkeypatch.setattr("scripts.kaggle_environment_check.inspect_environment", lambda: {
        "python_version_info": [3, 12, 0], "torch_status": "missing",
        "cuda_available": False, "gpu_count": 0, "gpus": [],
        "kaggle": {"input_exists": False, "working_exists": False, "working_writable": False},
    })
    with pytest.raises(RuntimeError, match="PyTorch|CUDA|Kaggle"):
        session.run(parsed_args("--action", "system"))


def test_output_root_requires_kaggle_working_subdirectory():
    with pytest.raises(ValueError, match="/kaggle/working"):
        session._output_directory(Path("local-output"))
    with pytest.raises(ValueError, match="named subdirectory"):
        session._output_directory(Path("/kaggle/working"))


def test_local_smoke_writes_only_session_evidence_and_not_status(tmp_path, monkeypatch, local_session):
    before = STATUS.read_bytes()
    output = tmp_path / "ready_package"
    monkeypatch.setattr(session, "_output_directory", lambda ignored: output)
    result = session.run(
        parsed_args("--action", "smoke", "--execute", "--output-root", "/kaggle/working/guandan"),
        runtime_check=lambda system: None,
    )
    assert result["status"] == "smoke_complete"
    assert result["smoke"]["status"] == "passed"
    assert result["smoke"]["kind"] == "A04_real_environment_smoke"
    assert result["smoke"]["kaggle_verified"] is False
    assert result["accepted"] is False
    assert result["kaggle_verified"] is False
    assert result["outputs"].startswith(str(output.resolve()))
    session_dir = Path(result["outputs"])
    assert (session_dir / "session_started.json").is_file()
    assert (session_dir / "session_result.json").is_file()
    assert before == STATUS.read_bytes()
    # This is the local ready-package boundary, not a genuine Kaggle result.
    assert result["status"] != "genuine_kaggle"


def test_remote_train_path_rejects_mocked_complete_without_durable_evidence(tmp_path, monkeypatch, local_session):
    before = STATUS.read_bytes()
    output = tmp_path / "ready_package"
    monkeypatch.setattr(session, "_output_directory", lambda ignored: output)
    monkeypatch.setattr(session, "smoke_environment", lambda: {
        "status": "passed", "kind": "mocked_local_smoke", "kaggle_verified": False,
    })
    calls = {}

    def fake_train(**kwargs):
        calls.update(kwargs)
        return TrainingResult(
            algorithm=kwargs["algorithm"], profile=kwargs["profile"], updates=100,
            total_token_steps=102_400, last_loss=None, checkpoint=None, status="complete",
            lifetime_updates=100, lifetime_token_steps=102_400,
            target_updates=100, rollout_envs=8, token_steps_per_update=1024,
        )

    monkeypatch.setattr("guandan.training.trainer.train", fake_train)
    with pytest.raises(ValueError, match="complete training durable_updates"):
        session.run(
            parsed_args("--action", "train", "--execute", "--algorithm", "ippo",
                        "--output-root", "/kaggle/working/guandan"),
            runtime_check=lambda system: None,
        )
    assert calls["profile"] == "remote_full"
    assert calls["device"] == "cuda"
    assert calls["updates"] == 100
    assert "rollout_envs" not in calls
    assert "rollout_steps" not in calls
    assert isinstance(calls["checkpoint_controller"], CheckpointController)
    assert Path(calls["checkpoint_dir"]).is_relative_to(output)
    assert before == STATUS.read_bytes()
    failure = next(Path(local_session).rglob("session_failure.json"))
    assert json.loads(failure.read_text(encoding="utf-8"))["accepted"] is False
    assert not any("remote" in str(path).lower() and path.suffix in {".pt", ".pth"}
                    for path in Path(local_session).rglob("*"))


def test_evaluate_requires_explicit_candidate_and_snapshot_without_remote_run(tmp_path, monkeypatch, local_session):
    monkeypatch.setattr(session, "_output_directory", lambda ignored: tmp_path / "ready_package")
    monkeypatch.setattr(session, "smoke_environment", lambda: {
        "status": "passed", "kind": "mocked_local_smoke", "kaggle_verified": False,
    })
    with pytest.raises(ValueError, match="candidate and frozen snapshot"):
        session.run(
            parsed_args("--action", "evaluate", "--execute",
                        "--output-root", "/kaggle/working/guandan"),
            runtime_check=lambda system: None,
        )

    calls = {}

    class FakeEvaluation:
        def as_dict(self):
            return _synthetic_evaluation_report()

    def fake_evaluate_profile(**kwargs):
        calls.update(kwargs)
        return FakeEvaluation()

    monkeypatch.setattr("guandan.evaluation.evaluate_profile", fake_evaluate_profile)

    def fake_load(path, *, expected_profile, **kwargs):
        assert expected_profile == "remote_full"
        assert kwargs["expected_algorithm"] == "ippo" and kwargs["map_location"] == "cpu"
        return _synthetic_evaluation_checkpoint(102 if Path(path).name == "candidate.pt" else 100)

    monkeypatch.setattr("guandan.training.checkpoint.load_checkpoint", fake_load)
    result = session.run(
        parsed_args("--action", "evaluate", "--execute",
                    "--candidate-checkpoint", "/tmp/candidate.pt",
                    "--snapshot-checkpoint", "/tmp/snapshot.pt",
                    "--output-root", "/kaggle/working/guandan"),
        runtime_check=lambda system: None,
    )
    assert result["status"] == "evaluation_complete"
    assert calls["profile"] == "remote_full"
    assert calls["device"] == "cuda"
    assert calls["candidate_checkpoint"] == Path("/tmp/candidate.pt")
    assert calls["snapshot_checkpoint"] == Path("/tmp/snapshot.pt")
    assert result["accepted"] is False
    assert result["kaggle_verified"] is False


def test_local_acceptance_has_no_status_write_and_no_upload_claim(tmp_path, monkeypatch, local_session):
    """The only local artifact is a ready_package-style evidence directory."""
    before = STATUS.read_bytes()
    output = tmp_path / "ready_package"
    monkeypatch.setattr(session, "_output_directory", lambda ignored: output)
    result = session.run(
        parsed_args("--action", "smoke", "--execute",
                    "--output-root", "/kaggle/working/guandan"),
        runtime_check=lambda system: None,
    )
    report = json.loads((Path(result["outputs"]) / "session_result.json").read_text(encoding="utf-8"))
    assert report["accepted"] is False
    assert report["kaggle_verified"] is False
    assert report["origin"].startswith("executed_session")
    assert Path(result["outputs"]).is_relative_to(output.resolve())
    assert before == STATUS.read_bytes()
    assert not (ROOT / "project_status" / "STATUS.yaml.tmp").exists()


@pytest.mark.parametrize("start", [0, 1, 7, 99, 100, 101])
@pytest.mark.parametrize("algorithm", ["ippo", "vrpo"])
def test_partial_session_plan_never_shrinks_the_canonical_target(start, algorithm):
    plan = session.training_plan(algorithm, {"durable_updates": start})
    goal = 100 if start < 100 else 102
    assert plan["start_update"] == start
    assert plan["goal_update"] == goal
    assert plan["updates_this_session_if_time_allows"] == goal - start
    assert plan["automatic_target_reduction"] is False
    training = plan["profile"]["config"]["training"]
    assert training["rollout_envs"] == 8
    assert training["token_steps_per_update"] == 1024
    assert training["checkpoint_interval"] == 10
    assert training["updates_per_algorithm"] == 100
    assert training["resume_updates"] == 2
    assert plan["profile"]["version"] == "profiles-0.2"


@pytest.mark.parametrize("start", [-1, True, 1.5, "100", 102, 103])
def test_completed_or_invalid_resume_plan_is_refused(start):
    with pytest.raises(ValueError, match="completed target|invalid counter"):
        session.training_plan("ippo", {"durable_updates": start})


@pytest.mark.parametrize("tokens", [
    ["--profile", "local_fast"], ["--runtime-check", "disabled"], ["--local"],
    ["--device", "cpu"], ["--updates", "1"],
])
def test_remote_cli_does_not_expose_local_override_or_runtime_bypass(tokens):
    with pytest.raises(SystemExit) as error:
        parsed_args(*tokens)
    assert error.value.code == 2


def test_runtime_gate_failure_precedes_any_execution_output(tmp_path, monkeypatch):
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(session, "_output_directory", lambda ignored: output)

    def reject_local_runtime(system):
        raise RuntimeError("fixture_has_no_remote_runtime")

    with pytest.raises(RuntimeError, match="fixture_has_no_remote_runtime"):
        session.run(parsed_args("--action", "train", "--execute"), runtime_check=reject_local_runtime)
    assert not output.exists()


def test_dirty_source_cannot_execute_even_with_mock_runtime(local_session, monkeypatch):
    monkeypatch.setattr(session, "source_provenance", lambda root: {
        "git_commit": "a" * 40, "git_worktree_clean": False, "source_kind": "fixture",
    })
    with pytest.raises(ValueError, match="clean committed/exported source"):
        session.run(parsed_args("--action", "train", "--execute"), runtime_check=lambda _: None)
    assert not local_session.exists()


def test_canonical_environment_is_restored_after_exception(monkeypatch):
    monkeypatch.setenv("GUANDAN_PROFILE", "local_fast")
    monkeypatch.setenv("GUANDAN_RESOLVED_PROFILE_JSON", "original-local-marker")
    profile = session.training_plan("ippo")["profile"]
    with pytest.raises(RuntimeError, match="injected"):
        with session.canonical_profile(profile):
            assert os.environ["GUANDAN_PROFILE"] == "remote_full"
            assert json.loads(os.environ["GUANDAN_RESOLVED_PROFILE_JSON"]) == profile
            raise RuntimeError("injected")
    assert os.environ["GUANDAN_PROFILE"] == "local_fast"
    assert os.environ["GUANDAN_RESOLVED_PROFILE_JSON"] == "original-local-marker"


def test_session_incomplete_does_not_turn_partial_training_into_100_updates(local_session, monkeypatch):
    monkeypatch.setattr(session, "smoke_environment", lambda: {"status": "fixture", "kaggle_verified": False})

    def stopped_train(**kwargs):
        assert kwargs["updates"] == 100
        return TrainingResult(
            algorithm="ippo", profile="remote_full", updates=1, total_token_steps=1101,
            last_loss=0.25, checkpoint="fixture-only.pt", status="incomplete",
            lifetime_updates=1, lifetime_token_steps=1024, target_updates=100,
            durable_updates=1, durable_token_steps=1024, discarded_partial_steps=77,
            recovery_manifest="fixture-only.recovery.json", reason="session_save_margin",
        )

    monkeypatch.setattr("guandan.training.trainer.train", stopped_train)
    result = session.run(parsed_args("--action", "train", "--execute"), runtime_check=lambda _: None)
    assert result["status"] == "incomplete"
    assert result["plan"]["goal_update"] == 100
    assert result["plan"]["updates_this_session_if_time_allows"] == 100
    assert result["training"]["updates"] == result["training"]["durable_updates"] == 1
    assert result["training"]["discarded_partial_steps"] == 77
    assert result["recovery_manifest"] == "fixture-only.recovery.json"
    assert_not_kaggle_acceptance(result)
    persisted = json.loads((Path(result["outputs"]) / "session_result.json").read_text(encoding="utf-8"))
    assert persisted == result


@pytest.mark.parametrize("start,additional", [(7, 93), (100, 2), (101, 1)])
def test_mocked_new_session_resume_passes_copied_checkpoint_and_remaining_target(
    local_session, monkeypatch, tmp_path, start, additional,
):
    copied_manifest = tmp_path / "different_input" / "manifest.json"
    copied_checkpoint = copied_manifest.with_name("copied.pt")
    # Explicit simulated input mount, not evidence of an actual new session.
    monkeypatch.setattr(session, "KAGGLE_INPUT", copied_manifest.parent)
    recovery = {
        "durable_updates": start, "checkpoint_path": str(copied_checkpoint),
        "source_git_commit": "a" * 40, "model_config": {"hidden_dim": 32},
        "compatibility": {"seed": 10017, "epochs": 2, "gamma": 1.0, "gae_lambda": 0.95},
    }

    def resolve(path, *, algorithm):
        assert path == copied_manifest
        assert algorithm == "ippo"
        return recovery

    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs)
        assert kwargs["resume"] == str(copied_checkpoint)
        assert kwargs["updates"] == additional
        assert kwargs["seed"] == 10017
        assert kwargs["profile"] == "remote_full" and kwargs["device"] == "cuda"
        assert "rollout_steps" not in kwargs and "rollout_envs" not in kwargs
        return TrainingResult(
            algorithm="ippo", profile="remote_full", updates=additional,
            total_token_steps=additional*1024, last_loss=0.25, checkpoint="mock.pt",
            status="complete", start_update=start, target_updates=additional,
            lifetime_updates=start+additional, lifetime_token_steps=(start+additional)*1024,
        )

    monkeypatch.setattr(session, "resolve_recovery", resolve)
    monkeypatch.setattr(session, "smoke_environment", lambda: {"status": "mocked"})
    monkeypatch.setattr("guandan.training.trainer.train", fake_train)
    with pytest.raises(ValueError, match="complete training durable_updates"):
        session.run(parsed_args("--action", "train", "--execute", "--resume", str(copied_manifest)),
                    runtime_check=lambda _: None)
    assert len(calls) == 1
    failure = next(Path(local_session).rglob("session_failure.json"))
    record = json.loads(failure.read_text(encoding="utf-8"))
    assert record["new_session_resume"] is True
    assert_not_kaggle_acceptance(record)


def test_runtime_failure_is_recorded_without_status_edits(local_session, monkeypatch):
    before = STATUS.read_bytes()
    monkeypatch.setattr(session, "smoke_environment", lambda: {"status": "mocked"})

    def fail_train(**kwargs):
        raise RuntimeError("fixture-failure-not-hidden")

    monkeypatch.setattr("guandan.training.trainer.train", fail_train)
    with pytest.raises(RuntimeError, match="fixture-failure-not-hidden"):
        session.run(parsed_args("--action", "train", "--execute"), runtime_check=lambda _: None)
    failures = list(local_session.rglob("session_failure.json"))
    assert len(failures) == 1
    record = json.loads(failures[0].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"]["message"] == "fixture-failure-not-hidden"
    assert_not_kaggle_acceptance(record)
    assert not list(local_session.rglob("session_result.json"))
    assert STATUS.read_bytes() == before


def test_evaluation_soft_deadline_records_incomplete(local_session, monkeypatch):
    from guandan.deployment.control import SessionStop

    monkeypatch.setattr(session, "smoke_environment", lambda: {"status": "mocked"})
    monkeypatch.setattr("guandan.training.checkpoint.load_checkpoint", lambda path, **kw:
                        _synthetic_evaluation_checkpoint(102 if Path(path).name == "candidate.pt" else 100))

    def expired(**kwargs):
        raise SessionStop("session_save_margin")

    monkeypatch.setattr("guandan.evaluation.evaluate_profile", expired)
    with pytest.raises(SessionStop, match="session_save_margin"):
        session.run(parsed_args("--action", "evaluate", "--execute", "--candidate-checkpoint", "candidate.pt",
                                "--snapshot-checkpoint", "snapshot.pt"), runtime_check=lambda _: None)
    failures = list(local_session.rglob("session_failure.json"))
    assert len(failures) == 1
    record = json.loads(failures[0].read_text(encoding="utf-8"))
    assert record["status"] == "incomplete"
    assert_not_kaggle_acceptance(record)


@pytest.mark.parametrize("candidate_update,snapshot_update", [(6, 5), (100, 100), (102, 99), (101, 100)])
def test_evaluation_refuses_noncanonical_candidate_snapshot_counts(
    candidate_update, snapshot_update, local_session, monkeypatch,
):
    monkeypatch.setattr(session, "smoke_environment", lambda: {"status": "mocked"})
    monkeypatch.setattr("guandan.training.checkpoint.load_checkpoint", lambda path, **kw:
                        _synthetic_evaluation_checkpoint(candidate_update if Path(path).name == "candidate.pt" else snapshot_update))
    with pytest.raises(ValueError, match="update102 candidate and update100 frozen snapshot"):
        session.run(parsed_args("--action", "evaluate", "--execute", "--candidate-checkpoint", "candidate.pt",
                                "--snapshot-checkpoint", "snapshot.pt"), runtime_check=lambda _: None)


def test_a07_runner_reports_only_ready_package_and_never_writes_status(tmp_path, monkeypatch):
    """Real runner/pytest subprocess, trivial fixtures ONLY for report semantics.

    No git init/commit, training, copied historical evidence, or self-recursion.
    Every currently required A07 test file gets one distinctly synthetic test.
    """
    from scripts import run_acceptance as runner

    root = tmp_path / "runner_fixture"
    root.mkdir()
    (root / ".acceptance-fixture").write_text(runner.FIXTURE_MARKER, encoding="utf-8")
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "acceptance_profiles.json").write_bytes(
        (ROOT / "configs" / "acceptance_profiles.json").read_bytes())
    (root / "project_status").mkdir()
    status = root / "project_status" / "STATUS.yaml"
    original = b"stage: A07\nstatus: not_accepted_fixture\naccepted: false\n"
    status.write_bytes(original)
    for index, name in enumerate(runner.STAGE_TESTS["A07"]):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"def test_synthetic_gate_only_{index}():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_ADDOPTS": "",
           "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "run_acceptance.py"),
         "--root", str(root), "--stage", "A07", "--profile", "local_fast"],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    stage = summary["stages"][0]
    assert summary["status"] == "passed"
    assert stage["status"] == "passed"
    assert summary["accepted"] is False and stage["accepted"] is False
    assert summary["counts"]["passed"] == len(runner.STAGE_TESTS["A07"])
    assert summary["counts"]["failed"] == summary["counts"]["skipped"] == 0
    assert status.read_bytes() == original
    # A passed local pytest stage is only a ready-package gate; it is not
    # evidence of a genuine Kaggle execution.
    assert summary["status"] != "genuine_kaggle"



def _synthetic_complete_result(tmp_path, start_update):
    """Synthetic counters only: no trainer or remote_full execution occurs."""
    recovery = None if start_update == 0 else {"durable_updates": start_update}
    plan = session.training_plan("ippo", recovery)
    settings = plan["profile"]["config"]["training"]
    goal = plan["goal_update"]
    additional = goal - start_update
    per_update = settings["token_steps_per_update"]
    artifact_root = tmp_path / f"synthetic-checkpoints-{start_update}"
    artifact_root.mkdir()
    checkpoint = artifact_root / "ippo_synthetic.pt"
    checkpoint.write_bytes(f"synthetic checkpoint {start_update}".encode("ascii"))
    manifest = artifact_root / "ippo_synthetic.recovery.json"
    manifest.write_text("synthetic resolver input", encoding="utf-8")
    result = TrainingResult(
        algorithm="ippo", profile="remote_full", updates=additional,
        total_token_steps=additional * per_update, last_loss=None,
        checkpoint=str(checkpoint), status="complete",
        lifetime_updates=goal, lifetime_token_steps=goal * per_update,
        start_update=start_update, target_updates=additional,
        rollout_envs=settings["rollout_envs"], token_steps_per_update=per_update,
        profile_counts_match=start_update in (0, 100),
        durable_updates=goal, durable_token_steps=goal * per_update,
        discarded_partial_steps=0, discarded_optimizer_steps=0,
        recovery_manifest=str(manifest),
    )
    source = {"git_commit": "synthetic-source-commit"}
    return plan, result, source, artifact_root, checkpoint, manifest


@pytest.mark.parametrize("start_update", [0, 99, 100, 101])
def test_verify_training_completion_accepts_canonical_and_segmented_continuations(
    tmp_path, monkeypatch, start_update,
):
    plan, result, source, artifact_root, checkpoint, manifest = _synthetic_complete_result(
        tmp_path, start_update)
    expected_steps = plan["goal_update"] * plan["profile"]["config"]["training"]["token_steps_per_update"]
    synthetic_recovery = {
        "durable_updates": plan["goal_update"],
        "durable_token_steps": expected_steps,
        "source_git_commit": source["git_commit"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": session.sha256(checkpoint),
    }

    def mocked_resolver(manifest_path, *, algorithm):
        # Synthetic counters are test fixtures, not real training evidence.
        assert Path(manifest_path) == manifest.resolve()
        assert algorithm == "ippo"
        return synthetic_recovery

    monkeypatch.setattr(session, "resolve_recovery", mocked_resolver)
    verified = session._verify_training_completion(
        result, plan, "ippo", source, checkpoint_root=artifact_root)
    assert verified == synthetic_recovery
    assert session.sha256(checkpoint) == verified["checkpoint_sha256"]
    assert Path(verified["checkpoint_path"]).resolve().is_relative_to(artifact_root.resolve())


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("profile_counts_match", 1, "profile_counts_match must be a bool"),
        ("updates", True, "updates counter"),
        ("total_token_steps", 102401, "total_token_steps counter"),
        ("lifetime_updates", 99, "lifetime_updates counter"),
        ("durable_updates", 99, "durable_updates counter"),
        ("durable_token_steps", 102399, "durable_token_steps counter"),
    ],
)
def test_verify_training_completion_rejects_bool_spoof_and_counter_errors(
    tmp_path, monkeypatch, field, value, match,
):
    plan, result, source, artifact_root, checkpoint, manifest = _synthetic_complete_result(tmp_path, 0)
    expected_steps = plan["goal_update"] * plan["profile"]["config"]["training"]["token_steps_per_update"]
    monkeypatch.setattr(session, "resolve_recovery", lambda manifest_path, *, algorithm: {
        "durable_updates": plan["goal_update"],
        "durable_token_steps": expected_steps,
        "source_git_commit": source["git_commit"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": session.sha256(checkpoint),
    })
    if field == "total_token_steps":
        value = result.total_token_steps + 1
    elif field == "lifetime_updates":
        value = result.lifetime_updates - 1
    elif field == "durable_updates":
        value = result.durable_updates - 1
    elif field == "durable_token_steps":
        value = result.durable_token_steps - 1
    forged = replace(result, **{field: value})
    with pytest.raises(ValueError, match=match):
        session._verify_training_completion(
            forged, plan, "ippo", source, checkpoint_root=artifact_root)


def test_verify_training_completion_rejects_checkpoint_source_or_sha_mismatch(
    tmp_path, monkeypatch,
):
    plan, result, source, artifact_root, checkpoint, manifest = _synthetic_complete_result(tmp_path, 0)
    expected_steps = plan["goal_update"] * plan["profile"]["config"]["training"]["token_steps_per_update"]
    monkeypatch.setattr(session, "resolve_recovery", lambda manifest_path, *, algorithm: {
        "durable_updates": plan["goal_update"],
        "durable_token_steps": expected_steps,
        "source_git_commit": "different-source-commit",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": "0" * 64,
    })
    with pytest.raises(ValueError, match="source commit"):
        session._verify_training_completion(
            result, plan, "ippo", source, checkpoint_root=artifact_root)


@pytest.fixture
def workflow_mounts(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    input_root.mkdir()
    working_root.mkdir()
    monkeypatch.setattr(session, "KAGGLE_INPUT", input_root)
    monkeypatch.setattr(session, "KAGGLE_WORKING", working_root)
    return input_root, working_root


def test_workflow_scope_requires_verified_output_and_allows_only_input_or_scope(
    workflow_mounts, monkeypatch,
):
    input_root, working_root = workflow_mounts
    output = working_root / "flow"
    output.mkdir()
    recovery_manifest = input_root / "mounted" / "manifest.json"
    recovery_manifest.parent.mkdir()
    recovery_manifest.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(session, "resolve_recovery", lambda path, *, algorithm: {
        "durable_updates": 99, "source_git_commit": "a" * 40,
        "compatibility": {}, "checkpoint_path": str(recovery_manifest.with_suffix(".pt")),
        "model_config": {},
    })
    result = session.run(
        parsed_args("--action", "plan", "--resume", str(recovery_manifest),
                    "--output-root", str(output)),
        runtime_check=lambda _: None, working_artifacts_root=output,
    )
    assert result["recovery_origin"] == "mounted_input"
    assert result["new_session_resume"] is True
    assert result["genuine_new_kaggle_session_verified"] is False
    with pytest.raises(ValueError, match="working_artifacts_root"):
        session.run(parsed_args("--action", "plan", "--output-root", str(output)),
                    runtime_check=lambda _: None, working_artifacts_root=working_root)
    with pytest.raises(ValueError, match="under /kaggle/input or the explicit working artifact scope"):
        session.run(parsed_args("--action", "plan", "--resume", str(Path("/tmp/manifest.json")),
                               "--output-root", str(output)),
                    runtime_check=lambda _: None, working_artifacts_root=output)


def test_workflow_current_output_resume_is_not_new_session(workflow_mounts, monkeypatch):
    _input_root, working_root = workflow_mounts
    output = working_root / "flow"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(session, "resolve_recovery", lambda path, *, algorithm: {
        "durable_updates": 100, "source_git_commit": "a" * 40,
        "compatibility": {}, "checkpoint_path": str(output / "checkpoint.pt"),
        "model_config": {},
    })
    result = session.run(
        parsed_args("--action", "plan", "--resume", str(manifest),
                    "--output-root", str(output)),
        runtime_check=lambda _: None, working_artifacts_root=output,
    )
    assert result["recovery_origin"] == "current_working_output"
    assert result["new_session_resume"] is False
    assert result["genuine_new_kaggle_session_verified"] is False


def test_workflow_reuses_shared_checkpoint_controller(workflow_mounts, monkeypatch):
    _input_root, working_root = workflow_mounts
    output = working_root / "flow"
    monkeypatch.setattr(session, "source_provenance", lambda root: {
        "git_commit": "a" * 40, "git_worktree_clean": True,
    })
    monkeypatch.setattr(session, "smoke_environment", lambda: {"status": "synthetic"})
    shared = CheckpointController(clock=lambda: 0.0)
    calls = {}

    def synthetic_train(**kwargs):
        calls.update(kwargs)
        return TrainingResult(
            algorithm="ippo", profile="remote_full", updates=0,
            total_token_steps=0, last_loss=None, checkpoint=None,
            status="incomplete", lifetime_updates=0, lifetime_token_steps=0,
            target_updates=100, rollout_envs=8, token_steps_per_update=1024,
            profile_counts_match=False, durable_updates=0, durable_token_steps=0,
            recovery_manifest=None,
        )

    monkeypatch.setattr("guandan.training.trainer.train", synthetic_train)
    result = session.run(
        parsed_args("--action", "train", "--execute", "--output-root", str(output)),
        runtime_check=lambda _: None, checkpoint_controller=shared,
        working_artifacts_root=output,
    )
    assert calls["checkpoint_controller"] is shared
    assert result["checkpoint_controller_reused"] is True
    assert result["working_artifacts_root"] == str(output.resolve())


def test_evaluation_checkpoint_and_report_reject_wrong_source_and_actual_counts(monkeypatch):
    profile = session._profile()
    loaded = _synthetic_evaluation_checkpoint(102)
    with pytest.raises(ValueError, match="source commit"):
        session._verify_evaluation_checkpoint(
            loaded, label="candidate", update=102, algorithm="ippo",
            profile=profile, source={"git_commit": "b" * 40},
        )
    broken = _synthetic_evaluation_checkpoint(102)
    broken.training_state["engine"]["collector"]["total_token_steps"] -= 1
    with pytest.raises(ValueError, match="token counter"):
        session._verify_evaluation_checkpoint(
            broken, label="candidate", update=102, algorithm="ippo",
            profile=profile, source={"git_commit": "a" * 40},
        )
    report = _synthetic_evaluation_report()
    report["total_games"] -= 1
    with pytest.raises(ValueError, match="total_games"):
        session._verify_evaluation_result(report, profile)


@pytest.mark.parametrize("scope_kind", ["whole_working", "outside", "other_scope"])
def test_workflow_scope_cannot_expand_output_boundary(workflow_mounts, scope_kind):
    _input_root, working = workflow_mounts
    output = working / "one-flow"
    scopes = {"whole_working": working, "outside": working.parent / "tmp",
              "other_scope": working / "different-flow"}
    with pytest.raises(ValueError, match="working_artifacts_root"):
        session.run(parsed_args("--action", "plan", "--output-root", str(output)),
                    runtime_check=lambda _: None, working_artifacts_root=scopes[scope_kind])
    assert not output.exists()


@pytest.mark.parametrize("field", ["--resume", "--candidate-checkpoint", "--snapshot-checkpoint"])
def test_workflow_rejects_inputs_outside_scope_including_resolved_parent_traversal(workflow_mounts, field):
    _input_root, working = workflow_mounts
    output = working / "one-flow"
    output.mkdir()
    outside = output / ".." / "another-flow" / "artifact.pt"
    with pytest.raises(ValueError, match="explicit working artifact scope"):
        session.run(parsed_args("--action", "plan", field, str(outside),
                                "--output-root", str(output)),
                    runtime_check=lambda _: None, working_artifacts_root=output)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("field", ["--resume", "--candidate-checkpoint", "--snapshot-checkpoint"])
def test_standalone_still_refuses_working_artifacts_without_explicit_scope(workflow_mounts, field):
    _input_root, working = workflow_mounts
    output = working / "one-flow"
    with pytest.raises(ValueError, match="/kaggle/input"):
        session.run(parsed_args("--action", "plan", field, str(output / "artifact.pt"),
                                "--output-root", str(output)))
    assert not output.exists()


def test_workflow_checks_resolved_link_targets_not_lexical_prefix(workflow_mounts, monkeypatch):
    """Mock only OS link resolution: does not require Windows symlink privileges."""
    _input_root, working = workflow_mounts
    output = working / "one-flow"
    output.mkdir()
    alias = output / "linked-input.pt"
    escaped = working.parent / "outside.pt"
    resolve = Path.resolve

    def simulated_link(path, *args, **kwargs):
        return resolve(escaped, *args, **kwargs) if path == alias else resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", simulated_link)
    with pytest.raises(ValueError, match="explicit working artifact scope"):
        session.run(parsed_args("--action", "plan", "--candidate-checkpoint", str(alias),
                                "--output-root", str(output)),
                    runtime_check=lambda _: None, working_artifacts_root=output)
    assert list(output.iterdir()) == []


def test_workflow_scope_and_shared_controller_are_not_cli_bypass_flags():
    for tokens in (("--working-artifacts-root", "/tmp"), ("--checkpoint-controller", "fake")):
        with pytest.raises(SystemExit) as error:
            parsed_args(*tokens)
        assert error.value.code == 2


def test_expired_shared_budget_stops_next_stage_without_restart(workflow_mounts, monkeypatch):
    from guandan.deployment.control import SessionStop

    _input_root, working = workflow_mounts
    output = working / "budget-flow"
    now = [100.0]
    shared = CheckpointController(session_hours=1, save_margin_seconds=60, clock=lambda: now[0])
    started = shared.started
    monkeypatch.setattr(session, "source_provenance", lambda root: {
        "git_commit": "a" * 40, "git_worktree_clean": True,
    })
    monkeypatch.setattr(session, "smoke_environment", lambda: {"fixture": "synthetic smoke"})
    first = session.run(parsed_args("--action", "smoke", "--execute", "--output-root", str(output)),
                        runtime_check=lambda _: None, checkpoint_controller=shared,
                        working_artifacts_root=output)
    assert first["checkpoint_controller_reused"] is True
    assert first["session_hours"] == 1  # Injected budget, not args' default 10 hours.
    now[0] += shared.soft_seconds
    monkeypatch.setattr(session, "smoke_environment", lambda: pytest.fail("expired budget started new work"))
    with pytest.raises(SessionStop, match="session_save_margin"):
        session.run(parsed_args("--action", "smoke", "--execute", "--output-root", str(output)),
                    runtime_check=lambda _: None, checkpoint_controller=shared,
                    working_artifacts_root=output)
    assert shared.started == started
    failure = json.loads(next(output.rglob("session_failure.json")).read_text(encoding="utf-8"))
    assert failure["status"] == "incomplete"
    assert failure["genuine_new_kaggle_session_verified"] is False


@pytest.mark.parametrize("side", ["candidate", "snapshot"])
@pytest.mark.parametrize("field,value", [
    ("profile", "local_fast"), ("algorithm", "vrpo"), ("git_commit", "b" * 40),
    ("update_count", True), ("profile_sha256", "wrong-profile-hash"),
    ("envs", 4), ("envs", True), ("token_steps_per_update", 256),
    ("engine_algorithm", "vrpo"), ("engine_update", 99), ("engine_update", True),
    ("total_token_steps", 102399), ("total_token_steps", True),
    ("collector_envs", []), ("ticks_per_env", [True] * 8), ("incomplete", True),
])
def test_evaluation_rejects_checkpoint_provenance_and_numeric_forgery(
    local_session, monkeypatch, side, field, value,
):
    checkpoints = {"candidate": _synthetic_evaluation_checkpoint(102),
                   "snapshot": _synthetic_evaluation_checkpoint(100)}
    loaded = checkpoints[side]
    state = loaded.training_state
    if field in ("profile", "algorithm", "git_commit", "update_count"):
        setattr(loaded.metadata, field, value)
    elif field in ("profile_sha256", "envs", "token_steps_per_update"):
        state["compatibility"][field] = value
    elif field in ("engine_algorithm", "engine_update"):
        state["engine"]["algorithm" if field == "engine_algorithm" else "update_count"] = value
    else:
        state["engine"]["collector"]["envs" if field == "collector_envs" else field] = value
    monkeypatch.setattr(session, "smoke_environment", lambda: {"fixture": "synthetic smoke"})
    monkeypatch.setattr("guandan.training.checkpoint.load_checkpoint",
                        lambda path, **kwargs: checkpoints[Path(path).stem])
    with pytest.raises(ValueError):
        session.run(parsed_args("--action", "evaluate", "--execute",
                                "--candidate-checkpoint", "candidate.pt",
                                "--snapshot-checkpoint", "snapshot.pt"), runtime_check=lambda _: None)
    # The autouse guard prohibits the evaluator: rejection must precede it.
    failure = json.loads(next(local_session.rglob("session_failure.json")).read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["genuine_new_kaggle_session_verified"] is False
    assert not list(local_session.rglob("session_result.json"))


@pytest.mark.parametrize("bad_field", [
    "profile", "status", "errors", "total_games", "total_bool", "target_games_per_opponent",
    "completed_games_per_opponent", "missing_opponent", "count_bool", "record_count",
    "nonterminal", "truncated", "duplicate", "group_bool", "rotation_outside",
])
def test_evaluation_cannot_claim_completion_with_inconsistent_game_counts(
    local_session, monkeypatch, bad_field,
):
    report = _synthetic_evaluation_report()
    if bad_field == "profile": report["profile"] = "local_fast"
    elif bad_field == "status": report["status"] = "incomplete"
    elif bad_field == "errors": report["errors"] = ["synthetic evaluation error"]
    elif bad_field == "total_games": report["total_games"] -= 1
    elif bad_field == "total_bool": report["total_games"] = True
    elif bad_field == "target_games_per_opponent": report[bad_field] -= 1
    elif bad_field == "completed_games_per_opponent": report[bad_field]["random"] -= 1
    elif bad_field == "missing_opponent": report["completed_games_per_opponent"].pop("snapshot")
    elif bad_field == "count_bool": report["completed_games_per_opponent"]["random"] = True
    elif bad_field == "record_count": report["records"].pop()
    elif bad_field == "nonterminal": report["records"][0]["terminal"] = False
    elif bad_field == "truncated": report["records"][0]["truncated"] = True
    elif bad_field == "duplicate": report["records"][1] = dict(report["records"][0])
    elif bad_field == "group_bool": report["records"][0]["deal_group"] = True
    elif bad_field == "rotation_outside": report["records"][0]["seat_rotation"] = 4
    monkeypatch.setattr(session, "smoke_environment", lambda: {"fixture": "synthetic smoke"})
    monkeypatch.setattr("guandan.training.checkpoint.load_checkpoint", lambda path, **kwargs:
                        _synthetic_evaluation_checkpoint(102 if Path(path).stem == "candidate" else 100))
    monkeypatch.setattr("guandan.evaluation.evaluate_profile", lambda **kwargs:
                        SimpleNamespace(as_dict=lambda: report))
    with pytest.raises(ValueError, match="evaluation"):
        session.run(parsed_args("--action", "evaluate", "--execute",
                                "--candidate-checkpoint", "candidate.pt",
                                "--snapshot-checkpoint", "snapshot.pt"), runtime_check=lambda _: None)
    failure = json.loads(next(local_session.rglob("session_failure.json")).read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["accepted"] is False
    assert not list(local_session.rglob("session_result.json"))


def test_same_workflow_shares_budget_across_both_algorithms_base_resume_and_evaluation(
    workflow_mounts, monkeypatch,
):
    """Orchestration-only mock: canonical COUNTERS, not real training/games."""
    _input_root, working = workflow_mounts
    output = working / "run-all-flow"
    clock = [50.0]
    shared = CheckpointController(clock=lambda: clock[0])
    started = shared.started
    calls = []
    monkeypatch.setattr(session, "source_provenance", lambda root: {
        "git_commit": "a" * 40, "git_worktree_clean": True,
        "source_kind": "SYNTHETIC WORKFLOW ONLY",
    })
    monkeypatch.setattr(session, "smoke_environment", lambda: {"fixture": "synthetic smoke"})

    def synthetic_resolver(path, *, algorithm):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["fixture"] == "SYNTHETIC COUNTERS NOT REAL TRAINING"
        assert data["algorithm"] == algorithm
        checkpoint = Path(path).with_name(data["checkpoint_filename"])
        assert session.sha256(checkpoint) == data["checkpoint_sha256"]
        return {**data, "checkpoint_path": str(checkpoint)}

    def synthetic_train(**kwargs):
        assert kwargs["checkpoint_controller"] is shared
        assert kwargs["profile"] == "remote_full" and kwargs["device"] == "cuda"
        assert "rollout_steps" not in kwargs and "rollout_envs" not in kwargs
        algorithm = kwargs["algorithm"]
        resumed = None if kwargs["resume"] is None else synthetic_resolver(
            Path(kwargs["resume"]).with_suffix(".recovery.json"), algorithm=algorithm)
        start = 0 if resumed is None else resumed["durable_updates"]
        updates = kwargs["updates"]
        assert (start, updates) in ((0, 100), (100, 2))
        goal = start + updates
        calls.append((algorithm, start, goal))
        root = Path(kwargs["checkpoint_dir"])
        root.mkdir(parents=True)
        checkpoint = root / f"{algorithm}_{goal}.pt"
        checkpoint.write_bytes(b"SYNTHETIC fixture: never trained")
        manifest = checkpoint.with_suffix(".recovery.json")
        data = {"fixture": "SYNTHETIC COUNTERS NOT REAL TRAINING", "algorithm": algorithm,
                "durable_updates": goal, "durable_token_steps": goal * 1024,
                "source_git_commit": "a" * 40, "model_config": {"hidden_dim": 32},
                "compatibility": {"seed": kwargs["seed"], "epochs": 2, "gamma": 1.0, "gae_lambda": 0.95},
                "checkpoint_filename": checkpoint.name, "checkpoint_sha256": session.sha256(checkpoint)}
        manifest.write_text(json.dumps(data), encoding="utf-8")
        clock[0] += 1
        shared.after_update(1.0)
        shared.checkpoint_saved()
        return TrainingResult(
            algorithm=algorithm, profile="remote_full", updates=updates, total_token_steps=updates * 1024,
            last_loss=None, checkpoint=str(checkpoint), status="complete", start_update=start,
            target_updates=updates, lifetime_updates=goal, lifetime_token_steps=goal * 1024,
            rollout_envs=8, token_steps_per_update=1024, profile_counts_match=True,
            durable_updates=goal, durable_token_steps=goal * 1024, recovery_manifest=str(manifest))

    def synthetic_load(path, *, expected_profile, expected_algorithm, map_location):
        assert expected_profile == "remote_full" and map_location == "cpu"
        data = synthetic_resolver(Path(path).with_suffix(".recovery.json"), algorithm=expected_algorithm)
        return _synthetic_evaluation_checkpoint(data["durable_updates"], expected_algorithm)

    def synthetic_evaluate(**kwargs):
        assert kwargs["deadline"] is shared
        assert kwargs["profile"] == "remote_full" and kwargs["device"] == "cuda"
        clock[0] += 1
        return SimpleNamespace(as_dict=_synthetic_evaluation_report)

    monkeypatch.setattr(session, "resolve_recovery", synthetic_resolver)
    monkeypatch.setattr("guandan.training.trainer.train", synthetic_train)
    monkeypatch.setattr("guandan.training.checkpoint.load_checkpoint", synthetic_load)
    monkeypatch.setattr("guandan.evaluation.evaluate_profile", synthetic_evaluate)

    def stage(action, algorithm, *tokens):
        return session.run(parsed_args("--action", action, "--execute", "--algorithm", algorithm,
                                       "--output-root", str(output), *tokens),
                           runtime_check=lambda _: None, checkpoint_controller=shared,
                           working_artifacts_root=output)

    records = []
    for algorithm in ("ippo", "vrpo"):
        base = stage("train", algorithm)
        resumed = stage("train", algorithm, "--resume", base["recovery_manifest"])
        evaluated = stage("evaluate", algorithm,
                          "--candidate-checkpoint", resumed["training"]["checkpoint"],
                          "--snapshot-checkpoint", base["training"]["checkpoint"])
        assert base["status"] == resumed["status"] == "phase_complete"
        assert base["recovery_origin"] == "none"
        assert resumed["recovery_origin"] == "current_working_output"
        assert resumed["new_session_resume"] is False
        assert evaluated["status"] == "evaluation_complete"
        assert evaluated["evaluation"]["total_games"] == 3000
        assert evaluated["evaluation"]["completed_games_per_opponent"] == {
            "random": 1000, "rule": 1000, "snapshot": 1000}
        records.extend((base, resumed, evaluated))
    assert calls == [("ippo", 0, 100), ("ippo", 100, 102), ("vrpo", 0, 100), ("vrpo", 100, 102)]
    assert shared.started == started and clock[0] > started
    assert shared.completed_update_seconds == [1.0] * 4
    assert len(list(output.rglob("session_result.json"))) == 6
    for report in records:
        assert report["checkpoint_controller_reused"] is True
        assert report["genuine_new_kaggle_session_verified"] is False
        assert report["new_session_resume"] is False
        assert report["accepted"] is report["kaggle_verified"] is False
