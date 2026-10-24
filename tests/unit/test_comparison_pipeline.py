"""Focused comparison pipeline coverage: summarize end-to-end and validate record checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from guandan_marvel.comparison.plan import (
    BOOTCAMP_FINAL_CANDIDATE_ARM, BOOTCAMP_FINAL_MIRROR_ARM, CONTROL_SCHEMA,
    build_comparison_deals, task_id_for,
)
from guandan_marvel.comparison.summarize import run_summarize
from guandan_marvel.comparison.validate import _check_record
from guandan_marvel.evaluation.common import file_sha256, payload_hash
from guandan_marvel.evaluation.manifests import dataset_identity


def _record(arm: str, deal_id: str, rotation: int, win: bool) -> dict[str, object]:
    return {
        "status": "complete", "arm_id": arm, "task_id": task_id_for(arm, deal_id, rotation),
        "deal_id": deal_id, "seat_rotation": rotation, "candidate_win": win,
        "candidate_team_reward": 1.0 if win else -1.0, "runtime_seconds": 2.0,
        "illegal_count": 0, "truncated_count": 0, "outcome_class": "DOUBLE_DOWN",
    }


def _write_control(tmp_path: Path) -> Path:
    deals = build_comparison_deals(count=1)
    deals_path = tmp_path / "deals.json"
    deals_path.write_text(json.dumps(deals, sort_keys=True), encoding="utf-8")
    body = {
        "schema": CONTROL_SCHEMA,
        "created_at": "2026-10-17T00:00:00+00:00",
        "comparison_id": "focused-pipeline",
        "scope": {"informal": True},
        "arms": [
            {"id": BOOTCAMP_FINAL_CANDIDATE_ARM, "priority": 0, "deals": 1,
             "candidate_kind": "bootcamp_final", "opponent_kind": "m09",
             "candidate_label": "new", "opponent_label": "m09"},
            {"id": BOOTCAMP_FINAL_MIRROR_ARM, "priority": 0, "deals": 1,
             "candidate_kind": "m09", "opponent_kind": "bootcamp_final",
             "candidate_label": "m09", "opponent_label": "new"},
        ],
        "deals_manifest": {
            "path": "deals.json", "bytes": deals_path.stat().st_size,
            "sha256": file_sha256(deals_path), "manifest_hash": deals["hash"],
            "dataset_hash": dataset_identity(deals, "final"), "deal_count": 1,
            "seed_base": 1600000000,
        },
    }
    control = {**body, "hash": payload_hash(body)}
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(control, sort_keys=True), encoding="utf-8")
    return control_path


def test_summarize_end_to_end_reports_complete_paired_endpoint(tmp_path):
    control_path = _write_control(tmp_path)
    deals = json.loads((tmp_path / "deals.json").read_text(encoding="utf-8"))
    deal_id = str(deals["datasets"]["final"][0]["deal_id"])
    output = tmp_path / "comparison_out"
    output.mkdir()
    rows = []
    for rotation in range(4):
        rows.append(_record(BOOTCAMP_FINAL_CANDIDATE_ARM, deal_id, rotation, True))
        rows.append(_record(BOOTCAMP_FINAL_MIRROR_ARM, deal_id, rotation, False))
    with (output / "games.worker0.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "worker0_status.json").write_text(
        json.dumps({"worker_index": 0, "tasks": len(rows), "completed": len(rows),
                    "skipped_due_to_time": {}}), encoding="utf-8",
    )
    exit_code = run_summarize(argparse.Namespace(control=control_path, working_root=tmp_path))
    assert exit_code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    endpoint = summary["paired"][f"{BOOTCAMP_FINAL_CANDIDATE_ARM}_minus_{BOOTCAMP_FINAL_MIRROR_ARM}"]
    assert endpoint["complete"] is True and endpoint["paired_deals"] == 1
    assert endpoint["mean_delta"] == 1.0
    assert summary["primary_endpoint"] == BOOTCAMP_FINAL_CANDIDATE_ARM
    assert summary["primary"]["verdict"] == "bootcamp_final_stronger"
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Primary verdict" in report
    assert "context:" in report
    assert "complete=True" in report
    assert (tmp_path / "m09-fabledan-comparison-output.zip").is_file()


def test_validate_check_record_flags_invalid_games():
    valid = {
        "status": "complete", "illegal_count": 0, "truncated_count": 0,
        "ranking": [1, 2, 3, 4], "fd_diagnostics": None, "deep_verified_commits": 0,
    }
    assert _check_record(valid, "h2h") == []
    illegal = {**valid, "illegal_count": 1}
    assert any("illegal/truncated" in failure for failure in _check_record(illegal, "h2h"))
    truncated = {**valid, "truncated_count": 1}
    assert any("illegal/truncated" in failure for failure in _check_record(truncated, "h2h"))
    bad_ranking = {**valid, "ranking": [1, 2, 3, 3]}
    assert any("ranking is invalid" in failure for failure in _check_record(bad_ranking, "h2h"))
    diagnostics = {**valid, "fd_diagnostics": {"empty_intersection_fallbacks": 1, "classification_fallbacks": 0,
                                               "engine_candidates": 10}, "deep_verified_commits": 1}
    assert any("intersection was empty" in failure for failure in _check_record(diagnostics, "h2h"))
    incomplete = {**valid, "status": "incomplete"}
    assert _check_record(incomplete, "h2h") == ["h2h|None|rotNone: status=incomplete"]
