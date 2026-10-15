"""Build the reviewed production-enabled M09 Unified Pipeline Dataset/Notebook."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.kaggle.common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from guandan_marvel.kaggle.m09_unified_delivery import (
    UNIFIED_NOTEBOOK_NAME,
    build_unified_delivery,
    validate_unified_delivery,
)

PRODUCTION_SCHEMA = "guandan-marvel-m09-unified-production-delivery-v1"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _rewrite_json(path: Path, payload: dict[str, object]) -> None:
    atomic_json(path, payload)


def build_production(
    repository: Path,
    output: Path,
    *,
    source_dataset: Path,
    initial_recovery: Path,
    champion_seed: Path,
    legacy_checkpoint: Path,
) -> dict[str, object]:
    build_unified_delivery(
        repository,
        output,
        source_dataset_root=source_dataset,
        initial_recovery=initial_recovery,
        champion_seed=champion_seed,
        legacy_checkpoint=legacy_checkpoint,
    )
    # Baseline validation must pass before enabling production.
    validate_unified_delivery(output)
    dataset = output / "dataset"
    control_path = dataset / "m09_unified_control.json"
    control = _load(control_path)
    control.pop("manifest_hash", None)
    control["remote_validation"]["production_enabled"] = True
    control["evaluation"]["formal_workers_default"] = 2
    control["evaluation"]["dual_gpu_formal_remote_validated"] = True
    control["claims"].update({
        "remote_validated": True,
        "production_segment1000_authorized": True,
        "final_holdout_generated": False,
        "m09_accepted": False,
    })
    control["remote_acceptance"] = {
        "schema": "guandan-marvel-m09-unified-remote-acceptance-v1",
        "remote_a": {
            "status": "accepted_complete",
            "review_hash": "2af3dc6a426b0f2e7a6a006ecf17ebbf91717d64914c20fcbac4fbf12127091b",
            "continuation_sha256": "be6c07d267af388a922ce17da48e151742340b04ae68d9a6bea892b0c6a72016",
        },
        "remote_b": {
            "status": "accepted_complete",
            "review_hash": "ee34f11e12641c7267ab066be2cfaf870d4bee4e0fdb5fc450e01fbc53a764ce",
            "training_commit_hash": "d0233efa3ed20f2df009e67a01978cd14cae7fdf2b00f28ac728213d789a617f",
            "recovery_sha256": "aa7669388c72806a83454c41309ad4fa9cfcd4e7059c25bab9f872afdb3a38c4",
        },
        "remote_c": {
            "status": "accepted_complete",
            "review_hash": "9f4515107e7918d0ab888493743fad609d94cb68118288dd093e5e4501c2e32f",
            "consistency_hash": "59d3cfcc42f8edacc2710fc14b0d0eb96994fe004288818e7356e44f51968092",
        },
    }
    control["production"] = {
        "authorized_segment": "segment1000",
        "start_iteration": 500,
        "target_iteration": 1000,
        "start_completed_games": 2000,
        "target_completed_games": 4000,
        "auto_continuation": False,
        "final_holdout": False,
    }
    control["manifest_hash"] = canonical_hash(control)
    _rewrite_json(control_path, control)

    dataset_manifest_path = dataset / "guandan_marvel_dataset.json"
    dataset_manifest = _load(dataset_manifest_path)
    dataset_manifest["created_at"] = _timestamp()
    dataset_manifest["m09"]["role"] = "unified_pipeline_production_segment1000"
    dataset_manifest["m09"]["unified_control"] = {
        "path": "m09_unified_control.json",
        "sha256": sha256_file(control_path),
        "manifest_hash": control["manifest_hash"],
    }
    dataset_manifest["claims"].update({
        "remote_verified": True,
        "unified_pipeline_remote_validated": True,
        "production_segment1000_authorized": True,
        "m09_accepted": False,
        "strength_claim": False,
    })
    _rewrite_json(dataset_manifest_path, dataset_manifest)

    notebook_path = output / UNIFIED_NOTEBOOK_NAME
    notebook = _load(notebook_path)
    notebook["cells"][0]["source"] = [
        "# Guandan MARVEL M09 Unified Production Pipeline\n",
        "Remote A/B/C 已通过。本版本 production_enabled=true；Run All 会从已验证 iter500 recovery 自动运行且只运行 Segment1000（iter1000/4000 games），随后使用两个独立 T4 evaluation workers 做 deterministic formal selection。auto-continuation 和 final holdout 保持关闭。\n",
    ]
    _rewrite_json(notebook_path, notebook)
    (output / "README_生产Segment1000.md").write_text(
        "# M09 Unified Production Segment1000\n\n"
        "上传 dataset/ 为同一个长期 Kaggle Dataset 的新版本，导入 guandan_marvel_m09_unified_pipeline.ipynb，T4 x2，只添加 production Dataset，Run All。"
        "本版本从 iter500/2000 games 到 iter1000/4000 games，并在 DDP 退出后运行双 GPU formal evaluation。不会自动进入 Segment1500，不运行 final holdout。\n",
        encoding="utf-8",
        newline="\n",
    )

    acceptance = control["remote_acceptance"]
    if not all(acceptance[name]["status"] == "accepted_complete" for name in ("remote_a", "remote_b", "remote_c")):
        raise ValueError("production control requires accepted Remote A/B/C")
    if control["remote_validation"]["production_enabled"] is not True:
        raise ValueError("production control is not enabled")
    if control["evaluation"]["formal_workers_default"] != 2 or control["evaluation"]["dual_gpu_formal_remote_validated"] is not True:
        raise ValueError("production control must use the Remote-C-validated dual formal path")
    if control["auto_continuation"]["enabled"] is not False or control["auto_continuation"]["final_holdout_allowed"] is not False:
        raise ValueError("production must keep auto continuation/final holdout disabled")

    files = [p for p in output.rglob("*") if p.is_file() and p.name != "delivery_manifest.json"]
    validation = {
        "schema": "guandan-marvel-m09-unified-production-validation-v1",
        "valid": True,
        "production_enabled": True,
        "authorized_segment": "segment1000",
        "formal_workers_default": 2,
        "dual_gpu_formal_remote_validated": True,
        "remote_a_accepted": True,
        "remote_b_accepted": True,
        "remote_c_accepted": True,
        "auto_continuation_enabled": False,
        "final_holdout_generated": False,
        "m09_accepted": False,
    }
    body = {
        "schema": PRODUCTION_SCHEMA,
        "delivery_id": "guandan-marvel-m09-unified-production-segment1000-v1",
        "created_at": _timestamp(),
        "dataset": "dataset",
        "notebook": UNIFIED_NOTEBOOK_NAME,
        "source_identity": control["source_identity"],
        "control_manifest_hash": control["manifest_hash"],
        "main_plan_hash": control["main_plan_hash"],
        "validation": validation,
        "files": relative_file_manifest(output, files),
        "final_holdout_generated": False,
        "m09_accepted": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    _rewrite_json(output / "delivery_manifest.json", manifest)
    return manifest


def create_upload_zip(delivery: Path) -> dict[str, object]:
    dataset = delivery / "dataset"
    destination = delivery / "guandan-marvel-m09-unified-production-segment1000-upload.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(dataset.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dataset).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if bad := archive.testzip():
            raise ValueError(f"production upload ZIP CRC failure: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)) or "control_overlay.zip" in names:
            raise ValueError("production upload ZIP inventory is invalid")
        control = json.loads(archive.read("m09_unified_control.json"))
        for row in control["control_files"]:
            data = archive.read(row["path"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"production upload control mismatch: {row['path']}")
    temporary.replace(destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--initial-recovery", type=Path, required=True)
    parser.add_argument("--champion-seed", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_production(
        ROOT,
        args.output.resolve(),
        source_dataset=args.source_dataset.resolve(),
        initial_recovery=args.initial_recovery.resolve(),
        champion_seed=args.champion_seed.resolve(),
        legacy_checkpoint=args.legacy_checkpoint.resolve(),
    )
    upload = create_upload_zip(args.output.resolve())
    print(json.dumps({"manifest": manifest, "upload": upload}, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
