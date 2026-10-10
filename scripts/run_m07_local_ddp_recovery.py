"""Run a two-process CPU/gloo DDP checkpoint and recovery exercise for M07."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _copy_dataset(delivery: Path, mode: str, destination: Path) -> Path:
    source = delivery / "datasets" / mode
    target = destination / f"guandan-marvel-{mode}"
    shutil.copytree(source, target)
    return target / "guandan_marvel_dataset.json"


def _run(*, manifest: Path, input_root: Path, working_root: Path, source_root: Path) -> None:
    command = [
        sys.executable, "-u", "-m", "guandan_marvel.kaggle.runner",
        "--dataset-manifest", str(manifest),
        "--input-root", str(input_root),
        "--working-root", str(working_root),
        "--source-root", str(source_root),
        "--allow-local-validation", "--local-cpu-test",
    ]
    subprocess.run(command, cwd=source_root, check=True)


def _latest(working: Path) -> tuple[Path, dict[str, object]]:
    output = working / "guandan-marvel-output"
    latest = json.loads((output / "latest_run.json").read_text(encoding="utf-8"))
    run = output / latest["path"]
    return run, latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    delivery = args.delivery.resolve()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    pilot_input = output / "pilot_input"
    pilot_input.mkdir()
    pilot_manifest = _copy_dataset(delivery, "pilot", pilot_input)
    pilot_working = output / "pilot_working"
    pilot_working.mkdir()
    _run(manifest=pilot_manifest, input_root=pilot_input, working_root=pilot_working, source_root=ROOT)
    pilot_run, pilot_latest = _latest(pilot_working)
    pilot_recovery = pilot_working / "guandan-marvel-output" / "guandan-marvel-recovery.zip"
    if not pilot_recovery.is_file():
        raise AssertionError("pilot did not produce a recovery bundle")

    resume_input = output / "resume_input"
    resume_input.mkdir()
    resume_manifest = _copy_dataset(delivery, "resume", resume_input)
    recovery_dataset = resume_input / "recovery"
    recovery_dataset.mkdir()
    shutil.copy2(pilot_recovery, recovery_dataset / "guandan-marvel-recovery.zip")
    resume_working = output / "resume_working"
    resume_working.mkdir()
    _run(manifest=resume_manifest, input_root=resume_input, working_root=resume_working, source_root=ROOT)
    resume_run, resume_latest = _latest(resume_working)

    pilot_progress = json.loads((pilot_run / "progress.json").read_text(encoding="utf-8"))
    resume_progress = json.loads((resume_run / "progress.json").read_text(encoding="utf-8"))
    pilot_checkpoint = json.loads((pilot_run / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    resume_checkpoint = json.loads((resume_run / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    for run in (pilot_run, resume_run):
        for rank in (0, 1):
            path = run / "ranks" / f"rank{rank:03d}" / "diagnostic.jsonl"
            if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                raise AssertionError(f"missing rank diagnostic: {path}")
    if pilot_progress.get("iteration") != 1 or resume_progress.get("iteration") != 2:
        raise AssertionError("local DDP recovery did not advance from iteration 1 to 2")
    summary = {
        "schema": "guandan-marvel-m07-local-ddp-recovery-v1",
        "scope": "local CPU/gloo two-process recovery only; not Kaggle or GPU evidence",
        "pilot": {
            "run": str(pilot_run), "status": pilot_latest["status"], "iteration": pilot_progress["iteration"],
            "checkpoint": pilot_checkpoint["latest"],
        },
        "resume": {
            "run": str(resume_run), "status": resume_latest["status"], "iteration": resume_progress["iteration"],
            "checkpoint": resume_checkpoint["latest"],
        },
        "world_size": 2,
        "launcher": "two_process_env_local_test",
        "backend": "gloo",
        "cuda_used": False,
        "kaggle_verified": False,
        "passes": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
