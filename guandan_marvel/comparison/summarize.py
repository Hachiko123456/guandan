# SPDX-License-Identifier: MIT
"""Merge worker shards, compute statistics, and archive the comparison output."""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

from guandan_marvel.evaluation.common import atomic_write_json, file_sha256, payload_hash

from .loader import load_control, load_deals
from .plan import enumerate_tasks, expected_games
from .stats import build_summary, render_markdown


def _read_shards(output_dir: Path) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for shard in sorted(output_dir.glob("games.worker*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record["task_id"])
            if task_id not in merged:
                merged[task_id] = record
    return list(merged.values())


def _read_worker_status(output_dir: Path) -> tuple[dict[str, int], list[dict[str, object]]]:
    skipped: dict[str, int] = {}
    statuses = []
    for path in sorted(output_dir.glob("worker*_status.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        statuses.append(payload)
        for arm_id, count in (payload.get("skipped_due_to_time") or {}).items():
            skipped[str(arm_id)] = skipped.get(str(arm_id), 0) + int(count)
    return skipped, statuses


def run_summarize(args: argparse.Namespace) -> int:
    control, base = load_control(args.control)
    deals, _rows = load_deals(base, control)
    all_tasks = enumerate_tasks(control, deals)
    output_dir = Path(args.working_root).resolve() / "comparison_out"
    records = _read_shards(output_dir)
    skipped, statuses = _read_worker_status(output_dir)

    summary = build_summary(control, records, skipped, deals)
    recorded_ids = {str(record["task_id"]) for record in records}
    expected_ids = {str(task["task_id"]) for task in all_tasks}
    missing = sorted(expected_ids - recorded_ids)
    complete_games = sum(1 for record in records if record.get("status") == "complete")
    error_games = sum(1 for record in records if record.get("status") == "error")
    incomplete_games = sum(1 for record in records if record.get("status") == "incomplete")
    summary.update({
        "status": "complete" if not missing and error_games == 0 else "partial",
        "completeness": {
            "expected_games": expected_games(control),
            "recorded": len(records),
            "complete": complete_games,
            "errors": error_games,
            "incomplete": incomplete_games,
            "missing": len(missing),
            "workers": statuses,
        },
    })
    summary["hash"] = payload_hash(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    report_md = render_markdown(summary)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8", newline="\n")

    archive_path = Path(args.working_root).resolve() / "m09-fabledan-comparison-output.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, path.relative_to(output_dir).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if bad := archive.testzip():
            raise ValueError(f"comparison output CRC failure: {bad}")
    os.replace(temporary, archive_path)

    for arm_id, arm_stats in summary["arms"].items():
        clusters = arm_stats.get("clusters") or {}
        boot = clusters.get("bootstrap") or {}
        print(json.dumps({
            "event": "comparison_arm_summary",
            "arm_id": arm_id,
            "complete": arm_stats.get("complete"),
            "games": arm_stats.get("games"),
            "errors": arm_stats.get("errors"),
            "win_rate": arm_stats.get("game_level_win_rate"),
            "cluster_mean": boot.get("mean"),
            "bootstrap_ci95_low": boot.get("low"),
            "bootstrap_ci95_high": boot.get("high"),
            "mean_candidate_team_reward": arm_stats.get("mean_candidate_team_reward"),
            "skipped_due_to_time": arm_stats.get("skipped_due_to_time"),
        }, sort_keys=True), flush=True)

    print(json.dumps({
        "event": "comparison_summary",
        "status": summary["status"],
        "primary": summary.get("primary"),
        "expected_games": summary["completeness"]["expected_games"],
        "recorded": summary["completeness"]["recorded"],
        "errors": summary["completeness"]["errors"],
        "missing": summary["completeness"]["missing"],
        "summary_hash": summary["hash"],
        "archive": str(archive_path),
        "archive_sha256": file_sha256(archive_path),
    }, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_summarize(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
