# SPDX-License-Identifier: MIT
"""Build and validate the M09 paired IPPO-vs-VRPO Kaggle selection delivery."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import zipfile

import torch

from .algorithm_compare import ALGORITHM_SELECTION_MANIFEST_SCHEMA, validate_selection_manifest
from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .discovery import BOOTSTRAP_NAME, DATASET_MANIFEST_NAME, DATASET_MANIFEST_SCHEMA, validate_dataset_manifest
from .package import (
    DELIVERY_MANIFEST_NAME, SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME,
    build_source_archive, render_notebook, validate_source_archive,
)

SELECTION_PACKAGE_ID = "guandan-marvel-m09-algorithm-selection-20261013"
SELECTION_DELIVERY_SCHEMA = "guandan-marvel-m09-algorithm-selection-delivery-v1"
SELECTION_VALIDATION_SCHEMA = "guandan-marvel-m09-algorithm-selection-delivery-validation-v1"
SELECTION_NOTEBOOK_NAME = "guandan_marvel_m09_algorithm_selection.ipynb"
SELECTION_README_NAME = "README_算法选择.md"
SELECTION_MANIFEST_NAME = "m09_algorithm_selection.json"
IPPO_CHECKPOINT_NAME = "ippo_iter000100.pt"
VRPO_CHECKPOINT_NAME = "vrpo_iter000100.pt"
IPPO_ANCHOR_NAME = "ippo_evaluation_games.jsonl"
VRPO_ANCHOR_NAME = "vrpo_evaluation_games.jsonl"
PLAN_COPY_NAME = "marvel_m09_full_plan.json"
LOCK_COPY_NAME = "marvel_m08_locked_config.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_metadata(path: Path, expected_algorithm: str) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("evaluation checkpoint payload must be an object")
    expected = {
        "schema": "guandan-marvel-evaluation-checkpoint-v1",
        "algorithm": expected_algorithm,
        "architecture": "v1.3",
        "iteration": 100,
        "training_resume_allowed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{expected_algorithm} checkpoint {key} mismatch")
    provenance = payload.get("training_data_provenance")
    if not isinstance(provenance, dict) or provenance.get("complete") is not True:
        raise ValueError(f"{expected_algorithm} checkpoint training provenance is incomplete")
    return {
        "path": "",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "schema": payload["schema"],
        "algorithm": payload["algorithm"],
        "architecture": payload["architecture"],
        "iteration": int(payload["iteration"]),
        "protocol_version": str(payload["protocol_version"]),
        "source_commit": str(payload["source_parent_commit"]),
        "source_hash": str(payload["source_hash"]),
        "resolved_config_hash": str(payload["resolved_config_hash"]),
        "backend_identity_hash": str((payload.get("backend_identity") or {}).get("hash", "")),
        "training_data_provenance_hash": str(provenance.get("hash", "")),
    }


def _validate_pilot_output(root: Path, *, algorithm: str) -> str:
    profile = json.loads((root / "selected_profile.json").read_text(encoding="utf-8"))
    result = json.loads((root / "run_result.json").read_text(encoding="utf-8"))
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    budget = json.loads((root / "completed_game_budget.json").read_text(encoding="utf-8"))
    evaluation = json.loads((root / "evaluation_summary.json").read_text(encoding="utf-8"))
    if profile.get("algorithm") != algorithm or profile.get("m09", {}).get("role") != f"{algorithm}_pilot":
        raise ValueError(f"{algorithm} pilot profile identity mismatch")
    if result.get("status") != "complete" or int(result.get("exit_code", -1)) != 0:
        raise ValueError(f"{algorithm} pilot run is not complete")
    if progress.get("phase") != "complete" or progress.get("complete") is not True:
        raise ValueError(f"{algorithm} pilot progress is incomplete")
    if budget.get("complete") is not True or budget.get("status") != "complete" or int(budget.get("completed_games", -1)) != 400:
        raise ValueError(f"{algorithm} completed-game budget is incomplete")
    if evaluation.get("status") != "complete" or int(evaluation.get("complete_games", -1)) != 520:
        raise ValueError(f"{algorithm} anchor evaluation is incomplete")
    if evaluation.get("selection_dataset_hash") != profile.get("completed_game_budget", {}).get("selection_dataset_hash"):
        raise ValueError(f"{algorithm} selection dataset identity mismatch")
    if int(evaluation.get("illegal_count", -1)) != 0 or int(evaluation.get("truncated_count", -1)) != 0:
        raise ValueError(f"{algorithm} anchor evaluation has health failures")
    shared = str(profile.get("m09", {}).get("shared_workload_hash", ""))
    if len(shared) != 64:
        raise ValueError(f"{algorithm} shared workload hash is missing")
    return shared


def build_selection_manifest(
    *, ippo_checkpoint: Path, vrpo_checkpoint: Path,
    ippo_anchor: Path, vrpo_anchor: Path,
    plan: dict[str, object],
) -> dict[str, object]:
    ippo = _checkpoint_metadata(ippo_checkpoint, "ippo")
    vrpo = _checkpoint_metadata(vrpo_checkpoint, "vrpo")
    if any(ippo[key] != vrpo[key] for key in ("architecture", "protocol_version", "source_commit", "source_hash", "backend_identity_hash")):
        raise ValueError("IPPO/VRPO checkpoint common identities differ")
    selection = plan["selection_dataset"]
    body = {
        "schema": ALGORITHM_SELECTION_MANIFEST_SCHEMA,
        "package_id": SELECTION_PACKAGE_ID,
        "created_at": _timestamp(),
        "comparison_id": plan["algorithm_pilot"]["comparison_id"],
        "plan_hash": plan["plan_hash"],
        "shared_workload_hash": "6f9bd5c6d9d45fa2adab118be0c043181ab77d68e2d6c21572ec00062a3225ff",
        "algorithms": {
            "ippo": {**ippo, "path": IPPO_CHECKPOINT_NAME},
            "vrpo": {**vrpo, "path": VRPO_CHECKPOINT_NAME},
        },
        "rule_anchors": {
            "ippo": {"path": IPPO_ANCHOR_NAME, "sha256": sha256_file(ippo_anchor), "bytes": ippo_anchor.stat().st_size},
            "vrpo": {"path": VRPO_ANCHOR_NAME, "sha256": sha256_file(vrpo_anchor), "bytes": vrpo_anchor.stat().st_size},
        },
        "selection_dataset": {
            "base_deals": int(selection["base_deals"]),
            "seed_base": int(selection["seed_base"]),
            "dataset_hash": str(selection["dataset_hash"]),
            "level_ranks": list(selection["level_ranks"]),
            "leader_seats": list(selection["leader_seats"]),
            "previous_results": list(selection["previous_results"]),
            "seat_rotations": [0, 1, 2, 3],
        },
        "gate": {
            "direct_candidate": "vrpo",
            "direct_opponent": "ippo",
            "direct_rule": "select VRPO if direct clustered CI low > 0.5; select IPPO if direct clustered CI high < 0.5",
            "paired_rule_fallback": "if direct is inconclusive, compare paired per-deal rule-anchor difference VRPO-IPPO",
            "final_fallback": "retain the locked M08 IPPO baseline when both intervals are inconclusive",
            "final_holdout_forbidden": True,
            "champion_promotion": False,
        },
        "claims": {"formal_champion_promotion": False, "final_holdout": False, "strength_claim": False},
    }
    return {**body, "manifest_hash": canonical_hash(body)}


def _render_selection_notebook() -> dict[str, object]:
    notebook = render_notebook()
    notebook["cells"][0]["source"] = [
        "# Guandan MARVEL M09 IPPO vs VRPO 算法选择\n",
        "只运行冻结 checkpoint 的 paired selection；不训练、不晋升 champion、不运行 final holdout。\n",
    ]
    notebook["cells"][2]["source"] = [line + "\n" for line in r'''selection_spec = DATASET.get('m09', {}).get('algorithm_selection') or {}
selection_relative = Path(str(selection_spec.get('path', '')))
if selection_relative.is_absolute() or '..' in selection_relative.parts:
    raise RuntimeError('algorithm selection manifest path is unsafe')
SELECTION_MANIFEST_PATH = (DATASET_MANIFEST_PATH.parent / selection_relative).resolve()
if not SELECTION_MANIFEST_PATH.is_relative_to(DATASET_MANIFEST_PATH.parent.resolve()) or not SELECTION_MANIFEST_PATH.is_file():
    raise RuntimeError('algorithm selection manifest is missing')
if sha256_file(SELECTION_MANIFEST_PATH) != selection_spec.get('sha256'):
    raise RuntimeError('algorithm selection manifest SHA-256 mismatch')
SELECTION_OUTPUT = WORKING_ROOT / 'guandan-marvel-algorithm-selection'
SELECTION_OUTPUT.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, '-u', '-m', 'guandan_marvel.kaggle.algorithm_compare',
    '--manifest', str(SELECTION_MANIFEST_PATH),
    '--output-root', str(SELECTION_OUTPUT),
    '--device', 'cuda:0',
]
print('RUN:', command, flush=True)
run_with_heartbeat(
    command, cwd=SOURCE_ROOT, env=ENV, label='algorithm_selection',
    timeout_seconds=4200, allowed_returncodes=(0,),
)
SELECTION_ZIP = WORKING_ROOT / 'guandan-marvel-algorithm-selection.zip'
SELECTION_TMP = SELECTION_ZIP.with_suffix('.zip.tmp')
with zipfile.ZipFile(SELECTION_TMP, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
    for path in sorted(SELECTION_OUTPUT.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(SELECTION_OUTPUT).as_posix())
os.replace(SELECTION_TMP, SELECTION_ZIP)
print('Created:', SELECTION_ZIP, 'bytes=', SELECTION_ZIP.stat().st_size, 'sha256=', sha256_file(SELECTION_ZIP), flush=True)'''.splitlines()]
    notebook["cells"][3]["source"] = [line + "\n" for line in r'''from IPython.display import FileLink, display
summary_path = SELECTION_OUTPUT / 'algorithm_selection_summary.json'
decision_path = SELECTION_OUTPUT / 'algorithm_selection_decision.json'
print(summary_path.read_text(encoding='utf-8'))
print(decision_path.read_text(encoding='utf-8'))
display(FileLink(str(SELECTION_ZIP)))'''.splitlines()]
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"selection-notebook#cell{index}")
    return notebook


def _readme() -> str:
    return """# M09 IPPO vs VRPO 算法选择\n\n1. 把 `dataset` 目录内全部文件建立为一个新的私有 Kaggle Dataset。\n2. 导入 `guandan_marvel_m09_algorithm_selection.ipynb`。\n3. 只添加这个 Dataset，Accelerator 选择 GPU T4 x2，Run All。\n4. 下载 `guandan-marvel-algorithm-selection.zip` 和完整 console log。\n\n本交付不训练、不运行 final holdout、不晋升 champion。它在锁定的 65 deal clusters × 4 rotations 上运行 VRPO-vs-IPPO direct head-to-head；若不显著，再使用已冻结的 paired rule anchors；仍不显著则按预案保留 IPPO baseline。\n"""


def build_selection_delivery(
    repository: str | Path, delivery_root: str | Path, *,
    ippo_output_root: str | Path, vrpo_output_root: str | Path,
    commit: str | None = "HEAD", allow_dirty: bool = False,
) -> dict[str, object]:
    repo = Path(repository).resolve()
    output = Path(delivery_root).resolve()
    ippo_root = Path(ippo_output_root).resolve()
    vrpo_root = Path(vrpo_output_root).resolve()
    plan = json.loads((repo / "configs/marvel_m09_full_plan.json").read_text(encoding="utf-8"))
    ippo_workload = _validate_pilot_output(ippo_root, algorithm="ippo")
    vrpo_workload = _validate_pilot_output(vrpo_root, algorithm="vrpo")
    if ippo_workload != vrpo_workload or ippo_workload != "6f9bd5c6d9d45fa2adab118be0c043181ab77d68e2d6c21572ec00062a3225ff":
        raise ValueError("IPPO/VRPO pilot shared workload identities differ")
    if output.exists():
        shutil.rmtree(output)
    dataset_root = output / "dataset"
    dataset_root.mkdir(parents=True)
    source = build_source_archive(
        repo, dataset_root / SOURCE_ARCHIVE_NAME, commit=commit,
        allow_dirty=allow_dirty, package_id=SELECTION_PACKAGE_ID,
    )
    source_manifest = source["source_manifest"]
    atomic_json(dataset_root / SOURCE_MANIFEST_NAME, source_manifest)
    with zipfile.ZipFile(dataset_root / SOURCE_ARCHIVE_NAME) as archive:
        bootstrap_bytes = archive.read("guandan-marvel-source/guandan_marvel/kaggle/bootstrap.py")
    (dataset_root / BOOTSTRAP_NAME).write_bytes(bootstrap_bytes)
    shutil.copy2(repo / "configs/marvel_m09_full_plan.json", dataset_root / PLAN_COPY_NAME)
    shutil.copy2(repo / "configs/marvel_m08_locked_config.json", dataset_root / LOCK_COPY_NAME)
    shutil.copy2(ippo_root / "evaluation_checkpoints/iter000100.pt", dataset_root / IPPO_CHECKPOINT_NAME)
    shutil.copy2(vrpo_root / "evaluation_checkpoints/iter000100.pt", dataset_root / VRPO_CHECKPOINT_NAME)
    shutil.copy2(ippo_root / "evaluation_games.jsonl", dataset_root / IPPO_ANCHOR_NAME)
    shutil.copy2(vrpo_root / "evaluation_games.jsonl", dataset_root / VRPO_ANCHOR_NAME)
    selection_manifest = build_selection_manifest(
        ippo_checkpoint=dataset_root / IPPO_CHECKPOINT_NAME,
        vrpo_checkpoint=dataset_root / VRPO_CHECKPOINT_NAME,
        ippo_anchor=dataset_root / IPPO_ANCHOR_NAME,
        vrpo_anchor=dataset_root / VRPO_ANCHOR_NAME,
        plan=plan,
    )
    atomic_json(dataset_root / SELECTION_MANIFEST_NAME, selection_manifest)
    validate_selection_manifest(selection_manifest, path=dataset_root / SELECTION_MANIFEST_NAME)
    dataset = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "package_id": SELECTION_PACKAGE_ID,
        "created_at": _timestamp(),
        "mode": "preflight",
        "source": {
            "archive": SOURCE_ARCHIVE_NAME, "archive_sha256": source["archive_sha256"],
            "manifest": SOURCE_MANIFEST_NAME, "manifest_sha256": sha256_file(dataset_root / SOURCE_MANIFEST_NAME),
            "git_commit": source_manifest["git_commit"], "content_manifest_hash": source_manifest["manifest_hash"],
        },
        "bootstrap": {"path": BOOTSTRAP_NAME, "sha256": sha256_file(dataset_root / BOOTSTRAP_NAME)},
        "profiles": {"preflight": {
            "schema": "guandan-marvel-kaggle-profile-v1", "mode": "preflight",
            "required_gpu_count": 2, "required_gpu_name_contains": "T4", "telemetry_interval_seconds": 5,
        }},
        "recovery": {"required": False, "auto_discovery": True},
        "m09": {
            "role": "algorithm_selection",
            "algorithm_selection": {"path": SELECTION_MANIFEST_NAME, "sha256": sha256_file(dataset_root / SELECTION_MANIFEST_NAME)},
            "plan_hash": plan["plan_hash"], "final_holdout": False,
        },
        "claims": {"remote_ready": True, "remote_verified": False, "algorithm_selected": False, "m09_accepted": False, "strength_claim": False},
    }
    atomic_json(dataset_root / DATASET_MANIFEST_NAME, dataset)
    validate_dataset_manifest(dataset, manifest_path=dataset_root / DATASET_MANIFEST_NAME)
    (output / SELECTION_NOTEBOOK_NAME).write_text(json.dumps(_render_selection_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / SELECTION_README_NAME).write_text(_readme(), encoding="utf-8", newline="\n")
    validation = validate_selection_delivery(output)
    files = [path for path in output.rglob("*") if path.is_file() and path.name != DELIVERY_MANIFEST_NAME]
    body = {
        "schema": SELECTION_DELIVERY_SCHEMA, "package_id": SELECTION_PACKAGE_ID,
        "created_at": _timestamp(), "source_commit": source_manifest["git_commit"],
        "workspace_snapshot": source_manifest["workspace_snapshot"], "plan_hash": plan["plan_hash"],
        "notebook": SELECTION_NOTEBOOK_NAME, "dataset": "dataset", "files": relative_file_manifest(output, files),
        "validation": validation, "uploaded": False, "remote_ready": True, "remote_verified": False,
        "algorithm_selected": False, "m09_accepted": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    atomic_json(output / DELIVERY_MANIFEST_NAME, manifest)
    return manifest


def validate_selection_delivery(delivery_root: str | Path) -> dict[str, object]:
    root = Path(delivery_root).resolve()
    dataset_root = root / "dataset"
    dataset = json.loads((dataset_root / DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
    validate_dataset_manifest(dataset, manifest_path=dataset_root / DATASET_MANIFEST_NAME)
    if dataset.get("package_id") != SELECTION_PACKAGE_ID or dataset.get("m09", {}).get("role") != "algorithm_selection":
        raise ValueError("algorithm-selection Dataset identity mismatch")
    source = validate_source_archive(dataset_root / SOURCE_ARCHIVE_NAME)
    selection = json.loads((dataset_root / SELECTION_MANIFEST_NAME).read_text(encoding="utf-8"))
    validated = validate_selection_manifest(selection, path=dataset_root / SELECTION_MANIFEST_NAME)
    notebook = json.loads((root / SELECTION_NOTEBOOK_NAME).read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{SELECTION_NOTEBOOK_NAME}#cell{index}")
    return {
        "schema": SELECTION_VALIDATION_SCHEMA, "valid": True,
        "package_id": SELECTION_PACKAGE_ID, "source_commit": source["git_commit"],
        "source_archive_sha256": source["archive_sha256"], "workspace_snapshot": source["workspace_snapshot"],
        "selection_manifest_hash": selection["manifest_hash"],
        "ippo_checkpoint_sha256": selection["algorithms"]["ippo"]["sha256"],
        "vrpo_checkpoint_sha256": selection["algorithms"]["vrpo"]["sha256"],
        "selection_dataset_hash": selection["selection_dataset"]["dataset_hash"],
        "notebook_syntax_valid": True, "remote_verified": False,
    }


__all__ = [
    "SELECTION_DELIVERY_SCHEMA", "SELECTION_MANIFEST_NAME", "SELECTION_NOTEBOOK_NAME",
    "SELECTION_PACKAGE_ID", "build_selection_delivery", "build_selection_manifest",
    "validate_selection_delivery",
]
