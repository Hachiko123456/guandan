# SPDX-License-Identifier: MIT
"""Adapter validation: deep-verified games and strict guest invariants."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from guandan_marvel.evaluation.common import atomic_write_json

from .driver import play_comparison_game
from .loader import load_control, load_deals, load_fabledan_model, load_v2_artifact, safe_control_file, validate_control_files
from .plan import BOOTCAMP_FINAL_CONTROL_KEY, BOOTCAMP_FINAL_KIND, ROTATIONS

REPORT_SCHEMA = "guandan-marvel-comparison-validation-v1"


def _check_record(record: dict[str, object], arm_id: str) -> list[str]:
    failures: list[str] = []
    prefix = f"{arm_id}|{record.get('deal_id')}|rot{record.get('seat_rotation')}"
    if record.get("status") != "complete":
        return [f"{prefix}: status={record.get('status')}"]
    if int(record.get("illegal_count", 0)) != 0 or int(record.get("truncated_count", 0)) != 0:
        failures.append(f"{prefix}: illegal/truncated games are not allowed")
    ranking = record.get("ranking")
    if not isinstance(ranking, list) or sorted(ranking) != [1, 2, 3, 4]:
        failures.append(f"{prefix}: ranking is invalid")
    diagnostics = record.get("fd_diagnostics")
    if diagnostics is not None:
        if int(diagnostics.get("empty_intersection_fallbacks", 0)) != 0:
            failures.append(f"{prefix}: guest candidate intersection was empty at least once")
        if int(diagnostics.get("classification_fallbacks", 0)) != 0:
            failures.append(f"{prefix}: guest classification fell back at least once")
        if int(diagnostics.get("engine_candidates", 0)) == 0:
            failures.append(f"{prefix}: guest never saw engine candidates")
        if int(record.get("deep_verified_commits", 0)) <= 0:
            failures.append(f"{prefix}: deep verification did not observe any commit")
    return failures


def run_validation(args: argparse.Namespace) -> int:
    control, base = load_control(args.control)
    validate_control_files(control, base)
    deals, rows = load_deals(base, control)
    validation = control.get("validation") or {}
    deal_count = int(validation.get("deals", 2))
    arm_ids = [str(value) for value in validation.get("arms", ["h2h"])]
    arm_map = {str(arm["id"]): arm for arm in control["arms"]}
    for arm_id in arm_ids:
        if arm_id not in arm_map:
            raise ValueError(f"validation arm {arm_id!r} is not part of the control matrix")

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    loaders: dict[str, str] = {}
    m09_artifacts: dict[str, object] = {}
    arm_kinds = {str(arm_map[arm_id]["candidate_kind"]) for arm_id in arm_ids} | {str(arm_map[arm_id]["opponent_kind"]) for arm_id in arm_ids}
    needs_fd = "fabledan" in arm_kinds
    for key, kind in (
        ("candidate", "m09"),
        ("champion", "m09_champion"),
        (BOOTCAMP_FINAL_CONTROL_KEY, BOOTCAMP_FINAL_KIND),
    ):
        if kind in arm_kinds:
            artifact, mode = load_v2_artifact(safe_control_file(base, control[key], key), device=device, expected_algorithm="ippo")
            m09_artifacts[kind] = artifact
            loaders[kind] = mode
    fabledan_model = None
    if needs_fd:
        fabledan_model = load_fabledan_model(safe_control_file(base, control["fabledan_weights"], "fabledan_weights"))
        loaders["fabledan"] = "numpy-vendor"

    started = time.monotonic()
    failures: list[str] = []
    games: list[dict[str, object]] = []
    diagnostics_totals: dict[str, int] = {}
    plan_total = len(arm_ids) * deal_count * len(ROTATIONS)
    completed = 0
    print(json.dumps({
        "event": "comparison_validation_start",
        "arms": arm_ids,
        "deals": deal_count,
        "planned_games": plan_total,
        "device": str(device),
        "artifact_loaders": loaders,
    }, sort_keys=True), flush=True)
    for arm_id in arm_ids:
        arm = arm_map[arm_id]
        for index in range(deal_count):
            deal = rows[index]
            for rotation in ROTATIONS:
                record = play_comparison_game(
                    arm_id=arm_id,
                    task_id=f"validation|{arm_id}|{deal['deal_id']}|{rotation}",
                    candidate_kind=str(arm["candidate_kind"]),
                    opponent_kind=str(arm["opponent_kind"]),
                    candidate_label=str(arm["candidate_label"]),
                    opponent_label=str(arm["opponent_label"]),
                    deal=deal,
                    rotation=rotation,
                    m09_artifacts=m09_artifacts,
                    fabledan_model=fabledan_model,
                    temperature=float(control.get("temperature", 0.0)),
                    device=device,
                    deep_verify=True,
                )
                for key, value in (record.get("fd_diagnostics") or {}).items():
                    diagnostics_totals[key] = diagnostics_totals.get(key, 0) + int(value)
                game_failures = _check_record(record, arm_id)
                failures.extend(game_failures)
                completed += 1
                print(json.dumps({
                    "event": "comparison_validation_game",
                    "arm_id": arm_id,
                    "deal_id": record.get("deal_id"),
                    "seat_rotation": record.get("seat_rotation"),
                    "status": record.get("status"),
                    "candidate_win": record.get("candidate_win"),
                    "token_steps": record.get("token_steps"),
                    "runtime_seconds": round(float(record.get("runtime_seconds", 0.0)), 2),
                    "deep_verified_commits": record.get("deep_verified_commits"),
                    "failures": len(game_failures),
                    "progress": f"{completed}/{plan_total}",
                }, sort_keys=True), flush=True)
                games.append({
                    "arm_id": arm_id,
                    "deal_id": record.get("deal_id"),
                    "seat_rotation": record.get("seat_rotation"),
                    "status": record.get("status"),
                    "candidate_win": record.get("candidate_win"),
                    "token_steps": record.get("token_steps"),
                    "runtime_seconds": record.get("runtime_seconds"),
                    "deep_verified_commits": record.get("deep_verified_commits"),
                    "fd_diagnostics": record.get("fd_diagnostics"),
                })

    report = {
        "schema": REPORT_SCHEMA,
        "control_hash": control.get("hash"),
        "status": "failed" if failures else "passed",
        "games": len(games),
        "failures": failures,
        "fd_diagnostics_totals": diagnostics_totals,
        "artifact_loaders": loaders,
        "elapsed_seconds": time.monotonic() - started,
        "games_detail": games,
    }
    output_dir = Path(args.working_root).resolve() / "comparison_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "validation_report.json", report)
    print(json.dumps({
        "event": "comparison_adapter_validation",
        "status": report["status"],
        "games": len(games),
        "failures": len(failures),
        "artifact_loaders": loaders,
    }, sort_keys=True), flush=True)
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_validation(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
