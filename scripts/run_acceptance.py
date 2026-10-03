"""Run stage tests and preserve evidence; never grant supervisory acceptance.

Each invocation creates a new history/<run-id>/ directory. Reports/logs are
write-once: there is intentionally no mutable ``latest`` report or STATUS edit.
Fingerprints include on-disk (tracked AND untracked) source, tests, docs, config,
notebooks and root configuration files, rather than treating HEAD as evidence.
The internal pytest worker records actual collection/execution through hooks;
terminal text alone is never used to decide whether a stage passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")
FIXTURE_MARKER = "guandan-acceptance-fixture-v1"
STAGE_TESTS = {
    "A00": ["tests/acceptance/test_a00_specification.py"],
    "A01": ["tests/acceptance/test_a01_cards.py"],
    "A02": ["tests/acceptance/test_a02_rules.py"],
    "A03": ["tests/acceptance/test_a03_actions.py"],
    "A04": ["tests/acceptance/test_a04_environment.py"],
    "A05": ["tests/acceptance/test_a05_training.py"],
    "A06": ["tests/acceptance/test_a06_evaluation.py"],
    "A07": ["tests/acceptance/test_a07_kaggle.py"],
    "A08": ["tests/acceptance/test_a08_belief_search.py"],
}
SOURCE_DIRS = ("guandan", "src", "scripts", "tests", "docs", "configs", "notebooks")
EXCLUDED = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".cache", "cache", "reports", "logs", "history", ".venv", "venv",
    "node_modules", ".ipynb_checkpoints", "build", "dist", ".tox", ".nox",
    "htmlcov",
}
ROOT_SUFFIXES = {".py", ".md", ".rst", ".toml", ".txt", ".yaml", ".yml", ".ini", ".cfg", ".lock"}
COUNT_KEYS = ("passed", "failed", "skipped", "errors", "collected")
# 显式将本仓库包目录加入搜索路径；测试夹具根目录稍后单独处理。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from guandan.execution_profiles import load_execution_profile


def load_profile(root: Path, name: str) -> dict:
    return load_execution_profile(root, name)


def write_once(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def write_json(path: Path, value: object) -> None:
    write_once(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(path: str) -> bool:
    parts = Path(path).parts
    return any(part in EXCLUDED for part in parts) or path.endswith((".pyc", ".pyo"))


def contained(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root)


def snapshot(root: Path) -> dict:
    """Hash actual bytes, including ignored/untracked relevant input files."""
    files: dict[str, str] = {}
    errors: list[str] = []

    def add(path: Path) -> None:
        relative = path.relative_to(root).as_posix()
        if excluded(relative):
            return
        try:
            if not contained(path, root):
                raise ValueError("input path escapes project root")
            files[relative] = sha256(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{relative}: {exc}")

    try:
        for path in root.iterdir():
            if path.is_file() and (path.suffix in ROOT_SUFFIXES or path.name == ".gitignore"):
                add(path)
        for name in SOURCE_DIRS:
            directory = root / name
            if not directory.exists():
                continue
            if not contained(directory, root):
                errors.append(f"{name}: input directory escapes project root")
                continue
            for base, dirs, names in os.walk(directory, onerror=lambda exc: errors.append(str(exc))):
                allowed = []
                for child in dirs:
                    path = Path(base) / child
                    if child in EXCLUDED:
                        continue
                    if path.is_symlink() or path.is_junction():
                        errors.append(f"{path.relative_to(root)}: linked input directory not traversed")
                    else:
                        allowed.append(child)
                dirs[:] = allowed
                for child in names:
                    add(Path(base) / child)
    except OSError as exc:
        errors.append(str(exc))

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True,
                text=True, encoding="utf-8", errors="replace", check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
            return result.stdout if result.returncode == 0 else None
        except OSError:
            return None

    # Ignore an ancestor repository: provenance must describe this root itself.
    top = git("rev-parse", "--show-toplevel")
    git_available = top is not None and Path(top.strip()).resolve() == root
    head = git("rev-parse", "--verify", "HEAD") if git_available else None
    porcelain = git("status", "--porcelain=v1", "-z", "--untracked-files=all") if git_available else None
    dirty_files = []
    if porcelain is not None:
        entries = iter(porcelain.split("\0"))
        for entry in entries:
            if not entry:
                continue
            status, path = entry[:2], entry[3:]
            item = {"status": status, "path": path}
            if "R" in status or "C" in status:
                item["original_path"] = next(entries, "")
            if not excluded(path):
                dirty_files.append(item)
    files = dict(sorted(files.items()))
    return {
        "git_commit": head.strip() if head else None,
        "git_available": git_available,
        "git_worktree_clean": not dirty_files if porcelain is not None else None,
        "dirty_files": dirty_files if porcelain is not None else None,
        "files_sha256": files,
        "content_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "runner_sha256": sha256(Path(__file__)),
        "errors": errors,
        "scope": {"directories": SOURCE_DIRS, "root_suffixes": sorted(ROOT_SUFFIXES),
                  "excluded_parts": sorted(EXCLUDED)},
    }


class ExecutionEvidence:
    """Pytest plugin used only in the isolated worker process."""

    def __init__(self) -> None:
        self.data = {"collected": 0, "nodeids": [], "deselected": 0,
                     "collect_only": False, "reports": [], "collection_issues": [],
                     "session_finished": False}

    def pytest_configure(self, config) -> None:
        # Includes collection-only enabled via config or PYTEST_ADDOPTS.
        self.data["collect_only"] = bool(config.getoption("collectonly"))

    def pytest_collection_finish(self, session) -> None:
        self.data["nodeids"] = [item.nodeid for item in session.items]
        self.data["collected"] = len(session.items)

    def pytest_deselected(self, items) -> None:
        self.data["deselected"] += len(items)

    def pytest_collectreport(self, report) -> None:
        if report.failed or report.skipped:
            self.data["collection_issues"].append({"nodeid": report.nodeid, "outcome": report.outcome})

    def pytest_runtest_logreport(self, report) -> None:
        self.data["reports"].append({"nodeid": report.nodeid, "when": report.when,
                                     "outcome": report.outcome,
                                     "wasxfail": hasattr(report, "wasxfail")})

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        self.data["collected"] = session.testscollected
        self.data["session_finished"] = True
        self.data["exitstatus"] = int(exitstatus)


def pytest_worker(root: str, evidence_path: str, args: list[str]) -> int:
    import pytest

    sys.path.insert(0, root)
    plugin = ExecutionEvidence()
    try:
        return int(pytest.main(args, plugins=[plugin]))
    finally:
        write_json(Path(evidence_path), plugin.data)


def read_junit(path: Path) -> dict:
    result = {"available": False, "counts": dict.fromkeys(COUNT_KEYS[:-1], 0), "cases": []}
    if not path.exists():
        result["error"] = "pytest did not produce a JUnit report"
        return result
    try:
        tree = ET.parse(path)
        if tree.getroot().tag not in ("testsuites", "testsuite"):
            raise ValueError("unexpected JUnit root element")
        for case in tree.iter("testcase"):
            outcome = ("errors" if case.find("error") is not None else
                       "failed" if case.find("failure") is not None else
                       "skipped" if case.find("skipped") is not None else "passed")
            result["counts"][outcome] += 1
            result["cases"].append({"name": case.get("name"), "classname": case.get("classname"),
                                    "outcome": outcome, "time": case.get("time")})
        result["available"] = True
    except (OSError, ET.ParseError, ValueError) as exc:
        result["error"] = str(exc)
    return result


def run_stage(root: Path, run_dir: Path, run_id: str, stage: str, collect_only: bool, profile: dict) -> dict:
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    directory = run_dir / stage
    directory.mkdir()
    before = snapshot(root)
    tests = STAGE_TESTS[stage]
    stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
    junit_path, evidence_path = directory / "junit.xml", directory / "pytest-events.json"
    temp = directory / "tmp"
    temp.mkdir()
    pytest_args = ["-q", "-p", "no:cacheprovider", "--rootdir", str(root),
                   "--basetemp", str(temp / "pytest"), "--junitxml", str(junit_path),
                   "-o", "junit_family=xunit2", "-o", "junit_logging=all", *tests]
    if collect_only:
        pytest_args.append("--collect-only")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--_pytest-worker",
               str(root), str(evidence_path), *pytest_args]
    report = {
        "schema_version": 1, "run_id": run_id, "stage": stage,
        "profile": profile,
        "status": "blocked", "accepted": False, "exit_code": 2,
        "timestamp_utc": now.isoformat(), "timestamp_local": now.astimezone(SHANGHAI).isoformat(),
        "timezone": "Asia/Shanghai", "collect_only_requested": collect_only,
        "tests": tests, "command": command, "pytest_args": pytest_args,
        "counts": dict.fromkeys(COUNT_KEYS, 0), "returncode": None,
        "reasons": [], "provenance_before": before,
    }
    missing = [path for path in tests if not (root / path).is_file()]
    unsafe = [path for path in tests if not contained(root / path, root)]
    stdout, stderr = "", ""
    if missing or not tests:
        report["missing_tests"] = missing
        report["reasons"].append("missing_required_tests")
        stderr = "Required stage test files are missing (or none are configured).\n"
    elif unsafe or before["errors"]:
        report["reasons"].append("unsafe_or_unreadable_inputs")
        stderr = json.dumps({"unsafe_tests": unsafe, "fingerprint_errors": before["errors"]})
    else:
        try:
            completed = subprocess.run(
                command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                     "TMP": str(temp), "TEMP": str(temp), "TMPDIR": str(temp),
                     "GUANDAN_PROFILE": profile["name"],
                     "GUANDAN_RESOLVED_PROFILE_JSON": json.dumps(profile, ensure_ascii=False)},
            )
            stdout, stderr = completed.stdout, completed.stderr
            report["returncode"] = completed.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            stderr = f"Could not execute pytest worker: {exc}\n"
            report["reasons"].append("subprocess_failure")
            report.update(status="failed", exit_code=1)
    write_once(stdout_path, stdout)
    write_once(stderr_path, stderr)
    report["junit"] = junit = read_junit(junit_path)
    evidence = {}
    if evidence_path.exists():
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report["reasons"].append(f"invalid_execution_evidence: {exc}")
    report["counts"].update(junit["counts"])
    report["counts"]["collected"] = evidence.get("collected", 0)
    report["collect_only_effective"] = collect_only or evidence.get("collect_only", False)
    report["deselected"] = evidence.get("deselected", 0)
    report["xfail_reports"] = sum(item["wasxfail"] for item in evidence.get("reports", []))
    report["collection_issues"] = evidence.get("collection_issues", [])

    if report["returncode"] is not None:
        reasons = report["reasons"]
        counts = report["counts"]
        if report["returncode"] != 0:
            reasons.append("pytest_nonzero_exit")
        if not evidence.get("session_finished"):
            reasons.append("missing_complete_execution_evidence")
        if not junit["available"]:
            reasons.append("missing_or_invalid_junit")
        if report["collect_only_effective"]:
            reasons.append("collection_only_is_not_execution")
        if counts["collected"] == 0:
            reasons.append("no_tests_collected")
        if counts["skipped"] or report["xfail_reports"] or report["collection_issues"]:
            reasons.append("required_tests_skipped_xfailed_or_collection_errors")
        if counts["failed"] or counts["errors"]:
            reasons.append("test_failures_or_errors")
        if report["deselected"]:
            reasons.append("required_tests_deselected")
        covered = {nodeid.split("::", 1)[0].replace("\\", "/") for nodeid in evidence.get("nodeids", [])}
        if any(path not in covered for path in tests):
            reasons.append("required_file_has_no_collected_tests")
        if not report["collect_only_effective"] and counts["passed"] != counts["collected"]:
            reasons.append("not_all_collected_tests_passed")
        if not reasons:
            report.update(status="passed", exit_code=0)
        elif report["collect_only_effective"] and report["returncode"] == 0:
            report.update(status="collected_only", exit_code=2)
        elif counts["collected"] == 0 and report["returncode"] == 5:
            report.update(status="blocked", exit_code=2)
        else:
            report.update(status="failed", exit_code=1)

    after = snapshot(root)
    report["provenance_after"] = after
    if (before["content_sha256"] != after["content_sha256"]
            or before["runner_sha256"] != after["runner_sha256"] or after["errors"]):
        report["reasons"].append("inputs_changed_or_became_unreadable_during_run")
        report.update(status="failed", exit_code=1)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["artifacts"] = {"report": (directory / "report.json").relative_to(root).as_posix()}
    report["artifact_sha256"] = {}
    for name, path in (("stdout", stdout_path), ("stderr", stderr_path),
                       ("junit", junit_path), ("pytest_events", evidence_path)):
        report["artifacts"][name] = path.relative_to(root).as_posix() if path.exists() else None
        if path.exists():
            report["artifact_sha256"][name] = sha256(path)
    write_json(directory / "report.json", report)
    return report


def resolve_root(parser: argparse.ArgumentParser, override: Path | None) -> Path:
    root = (override or ROOT).resolve()
    if root == Path(root.anchor) or not root.is_dir():
        parser.error("root must be an existing project directory, not a filesystem root")
    if override is not None:
        marker = root / ".acceptance-fixture"
        try:
            valid = contained(marker, root) and marker.read_text(encoding="utf-8").strip() == FIXTURE_MARKER
        except OSError:
            valid = False
        if not valid or not (root / "pyproject.toml").is_file() or not (root / "tests" / "acceptance").is_dir():
            parser.error("--root is fixture-only: requires pyproject.toml, tests/acceptance, and "
                         f".acceptance-fixture containing {FIXTURE_MARKER!r}")
    history = root / "project_status" / "history"
    if not contained(history, root):
        parser.error("history output path escapes the selected root")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGE_TESTS, "all"], required=True)
    parser.add_argument("--profile", default=os.environ.get("GUANDAN_PROFILE", "local_fast"),
                        help="Execution profile: local_fast for local smoke, remote_full for Kaggle/full runs.")
    parser.add_argument("--show-profile", action="store_true",
                        help="Only print resolved configuration; no tests or training; not acceptance.")
    parser.add_argument("--collect-only", action="store_true",
                        help="Collect tests for diagnostics only; never passes (exit 2 or failure).")
    parser.add_argument("--root", type=Path,
                        help="Test fixtures ONLY: override root; requires pyproject.toml, tests/acceptance "
                        f"and .acceptance-fixture containing {FIXTURE_MARKER!r}. "
                        "Artifacts and pytest temporary files stay under that root; no STATUS changes. "
                        "This is path validation, not a sandbox for untrusted test code.")
    args = parser.parse_args(argv)
    root = resolve_root(parser, args.root)
    try:
        profile = load_profile(root, args.profile)
    except ValueError as exc:
        parser.error(str(exc))
    if args.show_profile:
        print(json.dumps({"status": "configuration_only", "accepted": False, "stage": args.stage, "profile": profile}, ensure_ascii=False, indent=2))
        return 0
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex
    run_dir = root / "project_status" / "history" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    stages = list(STAGE_TESTS) if args.stage == "all" else [args.stage]
    results = [run_stage(root, run_dir, run_id, stage, args.collect_only, profile) for stage in stages]
    exit_code = 1 if any(result["exit_code"] == 1 for result in results) else (
        2 if any(result["exit_code"] == 2 for result in results) else 0)
    status = ("failed" if exit_code == 1 else "passed" if exit_code == 0 else
              "collected_only" if all(item["status"] == "collected_only" for item in results) else "blocked")
    summary = {
        "schema_version": 1, "run_id": run_id, "stage": args.stage,
        "profile": profile,
        "status": status, "accepted": False, "exit_code": exit_code,
        "root": str(root), "timestamp_utc": now.isoformat(),
        "timestamp_local": now.astimezone(SHANGHAI).isoformat(), "timezone": "Asia/Shanghai",
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {"python": sys.executable, "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", "")},
        "counts": {key: sum(result["counts"][key] for result in results) for key in COUNT_KEYS},
        "stages": results, "report": (run_dir / "report.json").relative_to(root).as_posix(),
    }
    write_json(run_dir / "report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_pytest-worker"]:
        raise SystemExit(pytest_worker(sys.argv[2], sys.argv[3], sys.argv[4:]))
    raise SystemExit(main())
