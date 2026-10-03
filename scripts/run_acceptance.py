from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "project_status" / "history"

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


def git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def runtime_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_worktree_clean": git("status", "--porcelain") == "",
    }
    try:
        import numpy
        import torch

        metadata.update(
            {
                "numpy": numpy.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostics must survive import failures
        metadata["runtime_import_error"] = repr(exc)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGE_TESTS, "all"], required=True)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    stages = list(STAGE_TESTS) if args.stage == "all" else [args.stage]
    tests = [path for stage in stages for path in STAGE_TESTS[stage] if (ROOT / path).exists()]
    missing = [path for stage in stages for path in STAGE_TESTS[stage] if not (ROOT / path).exists()]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    if args.collect_only:
        command.append("--collect-only")

    report: dict[str, object] = {
        "status": "blocked" if missing else "not_run",
        "accepted": False,
        "stage": args.stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "runtime": runtime_metadata(),
    }
    if missing:
        report["missing_tests"] = missing
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / f"{args.stage}_latest.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 2

    completed = subprocess.run(command, cwd=ROOT, text=True)
    report["status"] = "passed" if completed.returncode == 0 else "failed"
    report["returncode"] = completed.returncode
    report["tests"] = tests
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{args.stage}_latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
