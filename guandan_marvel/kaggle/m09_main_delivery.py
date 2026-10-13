# SPDX-License-Identifier: MIT
"""Build M09 segmented main-IPPO exact-resume training delivery."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .bootstrap import validate_recovery_directory
from .budget import build_completed_game_budget_contract
from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .discovery import BOOTSTRAP_NAME, DATASET_MANIFEST_NAME, DATASET_MANIFEST_SCHEMA, validate_dataset_manifest
from .package import DELIVERY_MANIFEST_NAME, SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, validate_notebook, validate_source_archive

MAIN_DELIVERY_SCHEMA = "guandan-marvel-m09-main-ippo-delivery-v1"
MAIN_VALIDATION_SCHEMA = "guandan-marvel-m09-main-ippo-delivery-validation-v1"
MAIN_DELIVERY_ID = "guandan-marvel-m09-main-ippo-segmented-20261013"
MAIN_PLAN_NAME = "marvel_m09_main_training_plan.json"
LOCK_NAME = "marvel_m08_locked_config.json"
NOTEBOOK_NAME = "guandan_marvel_m09_main_training.ipynb"
README_NAME = "README_主训练.md"
SEGMENTS = ("segment0500", "segment1000", "segment1500", "segment2000")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _profile(plan: dict[str, object], segment: dict[str, object]) -> dict[str, object]:
    target = int(segment["target_iteration"])
    games = int(segment["target_completed_games"])
    training = plan["training"]
    return {
        "schema": "guandan-marvel-kaggle-profile-v1",
        "mode": "resume",
        "expected_algorithm": "ippo",
        "required_gpu_count": 2,
        "required_gpu_name_contains": "T4",
        "ddp": {"launcher": "torchrun", "world_size": 2},
        "resume": {
            "target_iterations": target,
            "checkpoint_interval": int(training["checkpoint_interval"]),
            "evaluation_interval": 0,
            "max_wall_seconds": int(training["max_wall_seconds_per_segment"]),
            "checkpoint_keep_last": int(training["checkpoint_keep_last"]),
            "evaluation_keep_last": int(training["evaluation_keep_last"]),
        },
        "evaluation": {"base_deals": 1, "seed_base": 1_000_000_000},
        "completed_game_budget": build_completed_game_budget_contract(
            comparison_id="m09-main-ippo-cumulative-v1", algorithm="ippo",
            target_completed_games=games, global_collect_batch_size=4,
            expected_iterations=target, selection_dataset_hash=plan["plan_hash"],
        ),
        "telemetry_interval_seconds": 5,
        "m09": {
            "role": f"main_ippo_{segment['id']}",
            "main_plan_hash": plan["plan_hash"],
            "selected_algorithm": "ippo",
            "selection_decision_hash": plan["selection_decision_hash"],
            "mandatory_evaluation_after_segment": True,
            "training_distribution": plan["training_distribution"],
            "final_holdout": False,
            "strength_claim": False,
        },
    }


def build_main_delivery(
    repository: str | Path, delivery_root: str | Path, *,
    r2_delivery_root: str | Path, initial_recovery: str | Path,
) -> dict[str, object]:
    repo = Path(repository).resolve()
    output = Path(delivery_root).resolve()
    r2 = Path(r2_delivery_root).resolve()
    recovery = Path(initial_recovery).resolve()
    plan = json.loads((repo / "configs/marvel_m09_main_training_plan.json").read_text(encoding="utf-8"))
    if plan.get("schema") != "guandan-marvel-m09-main-training-plan-v1":
        raise ValueError("unsupported M09 main plan schema")
    if plan.get("plan_hash") != canonical_hash({key: value for key, value in plan.items() if key != "plan_hash"}):
        raise ValueError("M09 main plan hash mismatch")
    r2_dataset = r2 / "datasets/preflight"
    source_validation = validate_source_archive(r2_dataset / SOURCE_ARCHIVE_NAME)
    if source_validation["git_commit"] != plan["source_identity"]["source_commit"]:
        raise ValueError("r2 source commit differs from the main plan")
    if source_validation["archive_sha256"] != plan["source_identity"]["source_archive_sha256"]:
        raise ValueError("r2 source archive differs from the main plan")
    if sha256_file(recovery) != plan["initial_resume"]["recovery_zip_sha256"]:
        raise ValueError("initial IPPO recovery hash differs from the main plan")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copy2(r2 / "guandan_marvel_t4x2.ipynb", output / NOTEBOOK_NAME)
    validate_notebook(output / NOTEBOOK_NAME)
    rows = []
    segment_map = {str(row["id"]): dict(row) for row in plan["segments"]}
    if tuple(segment_map) != SEGMENTS:
        raise ValueError("main plan segment order mismatch")
    for segment_id in SEGMENTS:
        segment = segment_map[segment_id]
        root = output / "datasets" / segment_id
        root.mkdir(parents=True)
        for name in (SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, BOOTSTRAP_NAME):
            shutil.copy2(r2_dataset / name, root / name)
        shutil.copy2(repo / "configs/marvel_m09_main_training_plan.json", root / MAIN_PLAN_NAME)
        shutil.copy2(repo / "configs/marvel_m08_locked_config.json", root / LOCK_NAME)
        if segment_id == "segment0500":
            shutil.copy2(recovery, root / "guandan-marvel-recovery.zip")
        profile = _profile(plan, segment)
        dataset = {
            "schema": DATASET_MANIFEST_SCHEMA,
            "package_id": source_validation["package_id"],
            "created_at": _timestamp(),
            "mode": "resume",
            "source": {
                "archive": SOURCE_ARCHIVE_NAME,
                "archive_sha256": source_validation["archive_sha256"],
                "manifest": SOURCE_MANIFEST_NAME,
                "manifest_sha256": sha256_file(root / SOURCE_MANIFEST_NAME),
                "git_commit": source_validation["git_commit"],
                "content_manifest_hash": json.loads((root / SOURCE_MANIFEST_NAME).read_text(encoding="utf-8"))["manifest_hash"],
            },
            "bootstrap": {"path": BOOTSTRAP_NAME, "sha256": sha256_file(root / BOOTSTRAP_NAME)},
            "profiles": {"resume": profile},
            "recovery": {"required": True, "auto_discovery": True},
            "m09": {
                "role": f"main_ippo_{segment_id}",
                "plan": {"path": MAIN_PLAN_NAME, "sha256": sha256_file(root / MAIN_PLAN_NAME), "plan_hash": plan["plan_hash"]},
                "locked_config": {"path": LOCK_NAME, "sha256": sha256_file(root / LOCK_NAME)},
                "segment": segment,
                "final_holdout_generated": False,
            },
            "claims": {"remote_ready": True, "remote_verified": False, "main_training_complete": False, "m09_accepted": False, "strength_claim": False},
        }
        atomic_json(root / DATASET_MANIFEST_NAME, dataset)
        validate_dataset_manifest(dataset, manifest_path=root / DATASET_MANIFEST_NAME)
        rows.append({"segment": segment_id, "target_iteration": segment["target_iteration"], "target_completed_games": segment["target_completed_games"], "embedded_initial_recovery": segment_id == "segment0500"})
    (output / README_NAME).write_text(_readme(plan), encoding="utf-8", newline="\n")
    validation = validate_main_delivery(output)
    files = [path for path in output.rglob("*") if path.is_file() and path.name != DELIVERY_MANIFEST_NAME]
    body = {
        "schema": MAIN_DELIVERY_SCHEMA,
        "delivery_id": MAIN_DELIVERY_ID,
        "created_at": _timestamp(),
        "source_package_id": source_validation["package_id"],
        "source_commit": source_validation["git_commit"],
        "source_archive_sha256": source_validation["archive_sha256"],
        "main_plan_hash": plan["plan_hash"],
        "selected_algorithm": "ippo",
        "selection_decision_hash": plan["selection_decision_hash"],
        "segments": rows,
        "notebook": NOTEBOOK_NAME,
        "files": relative_file_manifest(output, files),
        "validation": validation,
        "uploaded": False,
        "remote_ready": True,
        "remote_verified": False,
        "main_training_complete": False,
        "final_holdout_run": False,
        "m09_accepted": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    atomic_json(output / DELIVERY_MANIFEST_NAME, manifest)
    return manifest


def _readme(plan: dict[str, object]) -> str:
    return f"""# M09 IPPO 分段主训练交付\n\n主算法：IPPO。Main plan hash：`{plan['plan_hash']}`。\n\n严格顺序：\n\n1. `segment0500`：目录已包含已验证的 r2 IPPO iter100 recovery；单独建 Dataset，Run All。\n2. 下载 output/recovery 并返回审查。必须完成独立 formal selection/champion evaluation 后，才允许下一段。\n3. `segment1000`、`segment1500`、`segment2000`：各自单独建 Dataset，并把上一段 recovery 作为第二个 Kaggle Input。\n4. 每段只训练，不在 DDP 进程内运行长评估；每 500 iteration 的 formal evaluation/champion 是强制独立步骤。\n5. 不得跳段，不得把 segment 完成称为 M09 accepted，不得运行 final holdout。\n\n每个 Dataset 目录中的文件必须全部上传且不得改名。所有段共用 `{NOTEBOOK_NAME}`，Accelerator 为 GPU T4 x2。\n"""


def validate_main_delivery(delivery_root: str | Path) -> dict[str, object]:
    root = Path(delivery_root).resolve()
    plan_hashes = set()
    source_hashes = set()
    rows = []
    for segment_id in SEGMENTS:
        dataset_root = root / "datasets" / segment_id
        payload = json.loads((dataset_root / DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
        validate_dataset_manifest(payload, manifest_path=dataset_root / DATASET_MANIFEST_NAME)
        if payload.get("m09", {}).get("role") != f"main_ippo_{segment_id}":
            raise ValueError("main segment role mismatch")
        profile = payload["profiles"]["resume"]
        if profile.get("expected_algorithm") != "ippo" or int(profile["resume"]["evaluation_interval"]) != 0:
            raise ValueError("main segment profile is not the locked training-only IPPO resume")
        plan_hashes.add(payload["m09"]["plan"]["plan_hash"])
        source_hashes.add(payload["source"]["archive_sha256"])
        rows.append({"segment": segment_id, "target_iteration": profile["resume"]["target_iterations"], "target_completed_games": profile["completed_game_budget"]["target_completed_games"]})
    first_recovery = root / "datasets/segment0500/guandan-marvel-recovery.zip"
    if not first_recovery.is_file():
        raise FileNotFoundError("segment0500 must embed the verified initial recovery")
    return {
        "schema": MAIN_VALIDATION_SCHEMA,
        "valid": True,
        "segments": rows,
        "main_plan_hash": next(iter(plan_hashes)) if len(plan_hashes) == 1 else None,
        "source_archive_sha256": next(iter(source_hashes)) if len(source_hashes) == 1 else None,
        "initial_recovery_sha256": sha256_file(first_recovery),
        "notebook": validate_notebook(root / NOTEBOOK_NAME),
        "remote_verified": False,
    }


__all__ = ["MAIN_DELIVERY_ID", "MAIN_DELIVERY_SCHEMA", "SEGMENTS", "build_main_delivery", "validate_main_delivery"]
