from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run_acceptance.py"
MARKER = "guandan-acceptance-fixture-v1"
PYTHON = sys.executable


def make_fixture(tmp_path: Path, source: str | None = None) -> Path:
    root = tmp_path / "fixture"
    acceptance = root / "tests" / "acceptance"
    acceptance.mkdir(parents=True)
    (root / ".acceptance-fixture").write_text(MARKER + "\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        "testpaths = [\"tests\"]\n",
        encoding="utf-8",
    )
    (root / "guandan").mkdir()
    (root / "guandan" / "__init__.py").write_text("# fixture package\n", encoding="utf-8")
    if source is not None:
        (acceptance / "test_a00_specification.py").write_text(source, encoding="utf-8")
    git_command = ["git", "-C", str(root)]
    subprocess.run([*git_command, "init", "-q"], check=True, capture_output=True, text=True)
    subprocess.run([*git_command, "add", "."], check=True, capture_output=True, text=True)
    subprocess.run(
        [
            *git_command,
            "-c", "user.name=acceptance-fixture",
            "-c", "user.email=acceptance-fixture@example.invalid",
            "commit", "-qm", "fixture baseline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return root


def run_runner(
    root: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    command = [PYTHON, str(RUNNER), "--root", str(root), *args]
    merged_env = os.environ.copy()
    merged_env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTEST_ADDOPTS": "",
        }
    )
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        command,
        cwd=root,
        env=merged_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout, completed.stderr
    return completed, json.loads(completed.stdout)


def stage_report(root: Path, summary: dict, stage: str = "A00") -> dict:
    report = root / summary["report"]
    assert report.is_file()
    stage_path = report.parent / stage / "report.json"
    assert stage_path.is_file()
    return json.loads(stage_path.read_text(encoding="utf-8"))


def assert_artifact_digests(root: Path, report: dict) -> None:
    for name in ("stdout", "stderr", "junit", "pytest_events"):
        relative = report["artifacts"][name]
        assert relative is not None
        artifact = root / relative
        actual = artifact.read_bytes()
        assert report["artifact_sha256"][name] == hashlib.sha256(actual).hexdigest()


def assert_common_evidence(root: Path, summary: dict, stage: str = "A00") -> dict:
    report = stage_report(root, summary, stage)
    assert report["accepted"] is False
    assert summary["accepted"] is False
    assert report["timestamp_utc"]
    assert report["timestamp_local"].endswith("+08:00")
    assert report["timezone"] == "Asia/Shanghai"
    assert report["elapsed_seconds"] >= 0
    assert set(report["counts"]) == {"passed", "failed", "skipped", "errors", "collected"}
    assert report["provenance_before"]["content_sha256"]
    assert "git_commit" in report["provenance_before"]
    assert "dirty_files" in report["provenance_before"]
    for name in ("stdout", "stderr", "junit", "pytest_events"):
        artifact = report["artifacts"][name]
        if artifact is not None:
            assert (root / artifact).is_file()
    return report


def test_success_records_execution_evidence_but_never_accepts(tmp_path: Path) -> None:
    root = make_fixture(
        tmp_path,
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert 1 + 1 == 2\n",
    )

    completed, summary = run_runner(root, "--stage", "A00")

    assert completed.returncode == 0
    assert summary["status"] == "passed"
    report = assert_common_evidence(root, summary)
    assert report["counts"] == {"passed": 2, "failed": 0, "skipped": 0, "errors": 0, "collected": 2}
    assert report["junit"]["available"] is True
    assert report["returncode"] == 0
    assert report["provenance_after"]["content_sha256"] == report["provenance_before"]["content_sha256"]
    assert_artifact_digests(root, report)


def test_missing_required_tests_is_blocked_and_has_no_pass(tmp_path: Path) -> None:
    root = make_fixture(tmp_path)

    completed, summary = run_runner(root, "--stage", "A00")

    assert completed.returncode == 2
    assert summary["status"] == "blocked"
    report = assert_common_evidence(root, summary)
    assert report["status"] == "blocked"
    assert report["counts"]["collected"] == 0
    assert "missing_required_tests" in report["reasons"]
    assert report["accepted"] is False


def test_failing_required_test_is_nonpassing(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, "def test_failure():\n    assert False\n")

    completed, summary = run_runner(root, "--stage", "A00")

    assert completed.returncode == 1
    assert summary["status"] == "failed"
    report = assert_common_evidence(root, summary)
    assert report["counts"]["failed"] == 1
    assert report["counts"]["collected"] == 1
    assert "pytest_nonzero_exit" in report["reasons"]


def test_skipped_and_xfailed_required_tests_are_nonpassing(tmp_path: Path) -> None:
    root = make_fixture(
        tmp_path,
        "import pytest\n\n"
        "@pytest.mark.skip(reason='required regression')\n"
        "def test_skipped():\n    pass\n\n"
        "@pytest.mark.xfail(reason='required regression')\n"
        "def test_xfailed():\n    assert False\n",
    )

    completed, summary = run_runner(root, "--stage", "A00")

    assert completed.returncode == 1
    assert summary["status"] == "failed"
    report = assert_common_evidence(root, summary)
    assert report["counts"]["collected"] == 2
    assert report["counts"]["skipped"] == 2
    assert report["xfail_reports"] >= 1
    assert "required_tests_skipped_xfailed_or_collection_errors" in report["reasons"]


def test_collect_only_can_never_pass_or_be_accepted(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, "def test_collected():\n    assert True\n")

    completed, summary = run_runner(root, "--stage", "A00", "--collect-only")

    assert completed.returncode == 2
    assert summary["status"] == "collected_only"
    report = assert_common_evidence(root, summary)
    assert report["status"] == "collected_only"
    assert report["collect_only_effective"] is True
    assert report["counts"]["collected"] == 1
    assert report["counts"]["passed"] == 0
    assert "collection_only_is_not_execution" in report["reasons"]


def test_pytest_subprocess_nonzero_exit_is_nonpassing(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, "def test_valid():\n    assert True\n")

    completed, summary = run_runner(
        root,
        "--stage",
        "A00",
        env={"PYTEST_ADDOPTS": "--definitely-not-a-real-pytest-option"},
    )

    assert completed.returncode == 1
    assert summary["status"] == "failed"
    report = assert_common_evidence(root, summary)
    assert report["returncode"] != 0
    assert report["accepted"] is False
    stderr_relative = report["artifacts"]["stderr"]
    assert stderr_relative is not None
    stderr_bytes = (root / stderr_relative).read_bytes()
    assert stderr_bytes
    assert report["artifact_sha256"]["stderr"] == hashlib.sha256(stderr_bytes).hexdigest()


def test_changed_source_fingerprint_and_dirty_file_are_recorded(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, "def test_valid():\n    assert True\n")
    first_completed, first_summary = run_runner(root, "--stage", "A00")
    assert first_completed.returncode == 0
    first_report = assert_common_evidence(root, first_summary)
    first_hash = first_report["provenance_before"]["content_sha256"]

    source = root / "tests" / "acceptance" / "test_a00_specification.py"
    source.write_text("def test_valid():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    second_completed, second_summary = run_runner(root, "--stage", "A00")
    assert second_completed.returncode == 0
    second_report = assert_common_evidence(root, second_summary)

    assert second_report["provenance_before"]["content_sha256"] != first_hash
    dirty_paths = {item["path"] for item in second_report["provenance_before"]["dirty_files"]}
    assert "tests/acceptance/test_a00_specification.py" in dirty_paths


def test_untracked_relevant_file_changes_fingerprint(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, "def test_valid():\n    assert True\n")
    first_completed, first_summary = run_runner(root, "--stage", "A00")
    assert first_completed.returncode == 0
    first_report = assert_common_evidence(root, first_summary)
    first_files = first_report["provenance_before"]["files_sha256"]
    first_hash = first_report["provenance_before"]["content_sha256"]

    untracked = root / "guandan" / "untracked_input.py"
    untracked.write_text("VALUE = 42\n", encoding="utf-8")
    second_completed, second_summary = run_runner(root, "--stage", "A00")
    assert second_completed.returncode == 0
    second_report = assert_common_evidence(root, second_summary)
    second_files = second_report["provenance_before"]["files_sha256"]

    relative = "guandan/untracked_input.py"
    assert relative not in first_files
    assert relative in second_files
    assert second_files[relative] == hashlib.sha256(untracked.read_bytes()).hexdigest()
    assert second_report["provenance_before"]["content_sha256"] != first_hash
    dirty_paths = {item["path"] for item in second_report["provenance_before"]["dirty_files"]}
    assert relative in dirty_paths


def test_run_id_artifacts_are_immutable_and_not_overwritten(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, "def test_valid():\n    assert True\n")
    first_completed, first_summary = run_runner(root, "--stage", "A00")
    assert first_completed.returncode == 0
    first_report = stage_report(root, first_summary)
    assert_artifact_digests(root, first_report)
    first_report_path = root / first_summary["report"]
    first_bytes = first_report_path.read_bytes()
    first_artifact_bytes = {
        name: (root / first_report["artifacts"][name]).read_bytes()
        for name in ("stdout", "stderr", "junit", "pytest_events")
    }

    second_completed, second_summary = run_runner(root, "--stage", "A00")
    assert second_completed.returncode == 0
    assert first_summary["run_id"] != second_summary["run_id"]
    assert first_report_path.read_bytes() == first_bytes
    for name, expected in first_artifact_bytes.items():
        assert (root / first_report["artifacts"][name]).read_bytes() == expected
    assert (root / second_summary["report"]).is_file()

    history = root / "project_status" / "history"
    run_dirs = [path for path in history.iterdir() if path.is_dir()]
    assert len(run_dirs) == 2


def test_all_reports_each_stage_including_absent_stages(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, "def test_a00_only():\n    assert True\n")

    completed, summary = run_runner(root, "--stage", "all")

    assert completed.returncode == 2
    assert summary["status"] == "blocked"
    assert [item["stage"] for item in summary["stages"]] == [f"A{i:02d}" for i in range(9)]
    a00 = next(item for item in summary["stages"] if item["stage"] == "A00")
    assert a00["status"] == "passed"
    for stage in (f"A{i:02d}" for i in range(1, 9)):
        item = next(item for item in summary["stages"] if item["stage"] == stage)
        assert item["status"] == "blocked"
        assert item["accepted"] is False
        assert (root / item["artifacts"]["report"]).is_file()


def test_root_help_documents_fixture_only_guard() -> None:
    completed = subprocess.run(
        [PYTHON, str(RUNNER), "--help"],
        cwd=RUNNER.parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0
    assert "fixture" in completed.stdout.lower()
    assert "acceptance-" in completed.stdout
    assert "fixture" in completed.stdout
    assert "no STATUS changes" in completed.stdout
