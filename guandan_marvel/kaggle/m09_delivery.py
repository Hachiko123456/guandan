# SPDX-License-Identifier: MIT
"""Build and validate the fresh M09 IPPO/VRPO controlled-pilot delivery."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from guandan_marvel.evaluation.manifests import dataset_identity

from .budget import (
    build_completed_game_budget_contract, validate_completed_game_budget_contract,
)
from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .discovery import BOOTSTRAP_NAME, DATASET_MANIFEST_NAME, DATASET_MANIFEST_SCHEMA, validate_dataset_manifest
from .evaluation import build_bounded_selection_manifest
from .package import (
    DELIVERY_MANIFEST_NAME, NOTEBOOK_NAME, README_NAME, SOURCE_ARCHIVE_NAME,
    SOURCE_MANIFEST_NAME, build_source_archive, render_notebook, validate_notebook,
    validate_source_archive,
)

M09_PLAN_SCHEMA = "guandan-marvel-m09-full-plan-v1"
M09_PACKAGE_SCHEMA = "guandan-marvel-m09-kaggle-delivery-v1"
M09_VALIDATION_SCHEMA = "guandan-marvel-m09-delivery-validation-v1"
M09_ROLES = ("preflight", "ippo_pilot", "ippo_resume", "vrpo_pilot", "vrpo_resume")
PLAN_COPY_NAME = "marvel_m09_full_plan.json"
LOCKED_CONFIG_COPY_NAME = "marvel_m08_locked_config.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root).decode("utf-8").strip()


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _status_m08_block(text: str) -> str:
    marker = "  M08_remote_pilot:"
    next_marker = "  M09_full_training:"
    if marker not in text or next_marker not in text:
        raise ValueError("STATUS.yaml is missing M08/M09 stage blocks")
    return text.split(marker, 1)[1].split(next_marker, 1)[0]


def _selection_manifest_from_plan(plan: Mapping[str, object]) -> dict[str, object]:
    selection = plan["selection_dataset"]
    if not isinstance(selection, Mapping):
        raise ValueError("M09 plan selection_dataset must be an object")
    return build_bounded_selection_manifest(
        base_deals=int(selection["base_deals"]),
        seed_base=int(selection["seed_base"]),
        level_ranks=tuple(int(value) for value in selection["level_ranks"]),
        leader_seats=tuple(int(value) for value in selection["leader_seats"]),
        previous_results=tuple(selection["previous_results"]),
    )


def validate_m09_plan(repository: str | Path) -> dict[str, object]:
    root = Path(repository).resolve()
    plan_path = root / "configs/marvel_m09_full_plan.json"
    plan = _load_object(plan_path)
    if plan.get("schema") != M09_PLAN_SCHEMA:
        raise ValueError("unsupported M09 full plan schema")
    declared_plan_hash = str(plan.get("plan_hash", ""))
    actual_plan_hash = canonical_hash({key: value for key, value in plan.items() if key != "plan_hash"})
    if declared_plan_hash != actual_plan_hash:
        raise ValueError("M09 plan hash mismatch")

    acceptance = plan.get("m08_acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("M09 plan m08_acceptance object is required")
    locked_path = root / str(acceptance["locked_config_path"])
    evidence_locked_path = root / str(acceptance["evidence_locked_config_path"])
    locked_sha = sha256_file(locked_path)
    evidence_sha = sha256_file(evidence_locked_path)
    if locked_sha != acceptance.get("locked_config_sha256") or evidence_sha != acceptance.get("evidence_locked_config_sha256"):
        raise ValueError("M08 locked-config SHA-256 does not match the M09 plan")
    if locked_sha != evidence_sha or locked_path.read_bytes() != evidence_locked_path.read_bytes():
        raise ValueError("M08 locked config and accepted evidence copy differ")
    locked = _load_object(locked_path)
    if locked.get("schema") != "guandan-marvel-m08-locked-training-config-v1":
        raise ValueError("unsupported M08 locked config schema")
    claims = locked.get("claims") or {}
    if claims.get("final_holdout_run") is not False or claims.get("model_strength") is not False:
        raise ValueError("M08 locked config contains an invalid strength/final claim")

    comparison = plan.get("algorithm_pilot")
    if not isinstance(comparison, Mapping):
        raise ValueError("M09 algorithm_pilot object is required")
    if tuple(comparison.get("algorithms", ())) != ("ippo", "vrpo"):
        raise ValueError("M09 controlled pilot must compare IPPO then VRPO")
    target_games = int(comparison.get("completed_games_per_algorithm", 0))
    collect = int(comparison.get("global_collect_batch_size", 0))
    iterations = int(comparison.get("expected_iterations", 0))
    if target_games != collect * iterations or target_games <= 0:
        raise ValueError("M09 completed-game budget is not exact")
    if collect != int(locked["training"]["collect_batch_size_global"]):
        raise ValueError("M09 comparison collect batch differs from the M08 locked config")

    selection_manifest = _selection_manifest_from_plan(plan)
    selection_hash = dataset_identity(selection_manifest, "selection")
    if selection_hash != str(plan["selection_dataset"]["dataset_hash"]):
        raise ValueError("M09 selection dataset identity mismatch")
    coverage = selection_manifest["coverage"]
    if len(coverage["level_ranks"]) != 13 or set(coverage["leader_seats"]) != {0, 1, 2, 3}:
        raise ValueError("M09 selection coverage is incomplete")
    if set(coverage["tribute_modes"]) != {"none", "double_down", "head_third", "head_last", "cancelled"}:
        raise ValueError("M09 selection tribute coverage is incomplete")

    package_id = str(plan.get("package_id", ""))
    if not package_id or package_id in set(plan.get("forbidden_reuse_package_ids", ())):
        raise ValueError("M09 package id reuses a forbidden legacy delivery")
    delivery = plan.get("delivery") or {}
    if tuple(delivery.get("dataset_roles", ())) != M09_ROLES:
        raise ValueError("M09 delivery role list is not locked")
    if delivery.get("main_full_generated") is not False or delivery.get("final_holdout_generated") is not False:
        raise ValueError("pre-selection M09 delivery must not contain main full/final holdout jobs")

    return {
        "schema": "guandan-marvel-m09-plan-validation-v1",
        "valid": True,
        "plan": plan,
        "locked_config": locked,
        "plan_hash": declared_plan_hash,
        "locked_config_sha256": locked_sha,
        "selection_dataset_hash": selection_hash,
        "selection_coverage": coverage,
    }


def _evaluation_profile(plan: Mapping[str, object]) -> dict[str, object]:
    selection = plan["selection_dataset"]
    return {
        "base_deals": int(selection["base_deals"]),
        "seed_base": int(selection["seed_base"]),
        "level_ranks": [int(value) for value in selection["level_ranks"]],
        "leader_seats": [int(value) for value in selection["leader_seats"]],
        "previous_results": list(selection["previous_results"]),
        "opponents": list(selection["opponents"]),
        "seat_rotations": [int(value) for value in selection["seat_rotations"]],
        "timeout_seconds": float(selection["timeout_seconds"]),
        "expected_selection_dataset_hash": str(selection["dataset_hash"]),
        "formal_selection": False,
        "final_holdout": False,
    }


def _common_training(plan: Mapping[str, object], locked: Mapping[str, object]) -> dict[str, object]:
    pilot = plan["algorithm_pilot"]
    runtime = locked["runtime"]
    training = locked["training"]
    return {
        "target_iterations": int(pilot["expected_iterations"]),
        "checkpoint_interval": int(runtime["checkpoint_interval"]),
        "evaluation_interval": int(runtime["evaluation_interval"]),
        "rollout_batch_size": int(training["rollout_batch_size_global"]),
        "collect_batch_size": int(training["collect_batch_size_global"]),
        "replay_buffer_scale": int(training["replay_buffer_scale"]),
        "mini_batches": int(training["mini_batches"]),
        "max_sampler_bytes_per_rank": int(pilot["max_sampler_bytes_per_rank"]),
        "seed": int(pilot["training_seed"]),
        "environment_rank_seed_stride": int(pilot["environment_rank_seed_stride"]),
        "training_data_generator_version": str(pilot["training_data_generator_version"]),
        "agent_epoch": int(training["agent_epoch"]),
        "critic_aux_epoch": int(training["critic_aux_epoch"]),
        "agent_merging_beta": float(training["agent_merging_beta"]),
        "amp_enabled": bool(training["amp_enabled"]),
        "compile_enabled": bool(training["compile_enabled"]),
        "use_muon": bool(training["use_muon"]),
        "model": dict(locked["model"]),
        "max_wall_seconds": int(runtime["max_wall_seconds"]),
        "checkpoint_keep_last": int(runtime["checkpoint_keep_last"]),
        "evaluation_keep_last": int(runtime["evaluation_keep_last"]),
    }


def build_m09_profiles(plan: Mapping[str, object], locked: Mapping[str, object]) -> dict[str, dict[str, object]]:
    common_training = _common_training(plan, locked)
    comparison = plan["algorithm_pilot"]
    selection_hash = str(plan["selection_dataset"]["dataset_hash"])
    ddp = {"launcher": "torchrun", "world_size": int(locked["world_size"]), "backend": str(locked["ddp_backend"]), "require_cuda": True}
    evaluation = _evaluation_profile(plan)
    shared_workload = {
        "training": common_training,
        "ddp": ddp,
        "target_completed_games": int(comparison["completed_games_per_algorithm"]),
        "selection_dataset_hash": selection_hash,
        "training_distribution": "m08-rank-two-no-previous-hand-v1",
    }
    shared_workload_hash = canonical_hash(shared_workload)
    profiles: dict[str, dict[str, object]] = {
        "preflight": {
            "schema": "guandan-marvel-kaggle-profile-v1", "mode": "preflight",
            "required_gpu_count": 2, "required_gpu_name_contains": "T4",
            "telemetry_interval_seconds": int(locked["runtime"]["telemetry_interval_seconds"]),
            "m09": {"role": "preflight", "plan_hash": plan["plan_hash"], "final_holdout": False},
        }
    }
    for algorithm in ("ippo", "vrpo"):
        contract = build_completed_game_budget_contract(
            comparison_id=str(comparison["comparison_id"]), algorithm=algorithm,
            target_completed_games=int(comparison["completed_games_per_algorithm"]),
            global_collect_batch_size=int(comparison["global_collect_batch_size"]),
            expected_iterations=int(comparison["expected_iterations"]),
            selection_dataset_hash=selection_hash,
        )
        base = {
            "schema": "guandan-marvel-kaggle-profile-v1",
            "required_gpu_count": 2,
            "required_gpu_name_contains": "T4",
            "ddp": dict(ddp),
            "evaluation": json.loads(json.dumps(evaluation)),
            "completed_game_budget": contract,
            "telemetry_interval_seconds": int(locked["runtime"]["telemetry_interval_seconds"]),
            "m09": {
                "role": f"{algorithm}_pilot", "plan_hash": plan["plan_hash"],
                "locked_config_sha256": plan["m08_acceptance"]["locked_config_sha256"],
                "shared_workload_hash": shared_workload_hash,
                "training_distribution": "m08-rank-two-no-previous-hand-v1",
                "formal_selection": False, "final_holdout": False, "strength_claim": False,
            },
        }
        profiles[f"{algorithm}_pilot"] = {
            **base, "mode": "full", "algorithm": algorithm,
            "user_selected_full_manifest_required": True,
            "training": json.loads(json.dumps(common_training)),
        }
        profiles[f"{algorithm}_resume"] = {
            **base, "mode": "resume", "expected_algorithm": algorithm,
            "m09": {**base["m09"], "role": f"{algorithm}_resume"},
            "resume": {
                "target_iterations": int(comparison["expected_iterations"]),
                "checkpoint_interval": int(locked["runtime"]["checkpoint_interval"]),
                "evaluation_interval": int(locked["runtime"]["evaluation_interval"]),
                "max_wall_seconds": int(locked["runtime"]["max_wall_seconds"]),
                "checkpoint_keep_last": int(locked["runtime"]["checkpoint_keep_last"]),
                "evaluation_keep_last": int(locked["runtime"]["evaluation_keep_last"]),
            },
        }
    return profiles


def validate_m09_profiles(plan: Mapping[str, object], locked: Mapping[str, object], profiles: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if tuple(profiles) != M09_ROLES:
        raise ValueError("M09 profile roles/order mismatch")
    expected_training = _common_training(plan, locked)
    ippo = profiles["ippo_pilot"]
    vrpo = profiles["vrpo_pilot"]
    if ippo["training"] != expected_training or vrpo["training"] != expected_training:
        raise ValueError("M09 pilot profiles do not exactly match the locked common training config")
    if ippo["ddp"] != vrpo["ddp"] or ippo["evaluation"] != vrpo["evaluation"]:
        raise ValueError("IPPO/VRPO pilot DDP or selection profiles differ")
    if ippo["m09"]["shared_workload_hash"] != vrpo["m09"]["shared_workload_hash"]:
        raise ValueError("IPPO/VRPO shared workload identity differs")
    for algorithm in ("ippo", "vrpo"):
        fresh = profiles[f"{algorithm}_pilot"]
        resume = profiles[f"{algorithm}_resume"]
        validate_completed_game_budget_contract(
            fresh["completed_game_budget"], expected_algorithm=algorithm,
            global_collect_batch_size=int(fresh["training"]["collect_batch_size"]),
            target_iterations=int(fresh["training"]["target_iterations"]),
        )
        validate_completed_game_budget_contract(
            resume["completed_game_budget"], expected_algorithm=algorithm,
            target_iterations=int(resume["resume"]["target_iterations"]),
        )
        if resume["expected_algorithm"] != algorithm:
            raise ValueError("resume algorithm binding mismatch")
        if fresh["training"]["checkpoint_keep_last"] != 2 or fresh["training"]["evaluation_keep_last"] != 2:
            raise ValueError("M09 retention must keep exactly the latest two checkpoint/evaluation artifacts")
        if fresh["evaluation"]["expected_selection_dataset_hash"] != plan["selection_dataset"]["dataset_hash"]:
            raise ValueError("M09 profile selection dataset hash mismatch")
    return {
        "schema": "guandan-marvel-m09-profile-validation-v1",
        "valid": True,
        "roles": list(profiles),
        "shared_workload_hash": ippo["m09"]["shared_workload_hash"],
        "target_completed_games_per_algorithm": int(plan["algorithm_pilot"]["completed_games_per_algorithm"]),
        "expected_iterations_per_algorithm": int(plan["algorithm_pilot"]["expected_iterations"]),
        "checkpoint_keep_last": 2,
        "evaluation_keep_last": 2,
        "selection_dataset_hash": plan["selection_dataset"]["dataset_hash"],
    }


def run_m09_safety_check(repository: str | Path, *, require_clean: bool = True) -> dict[str, object]:
    root = Path(repository).resolve()
    validated = validate_m09_plan(root)
    plan = validated["plan"]
    profiles = build_m09_profiles(plan, validated["locked_config"])
    profile_validation = validate_m09_profiles(plan, validated["locked_config"], profiles)
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    if branch != "codex/marvel-guandan-migration":
        raise ValueError(f"M09 must be built from codex/marvel-guandan-migration, got {branch!r}")
    if require_clean and status:
        raise ValueError("M09 delivery safety check requires a clean Git worktree")
    m08_commit = str(plan["m08_acceptance"]["commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", m08_commit, head], cwd=root,
        check=False, capture_output=True, text=True,
    ).returncode == 0
    if not ancestor:
        raise ValueError("M08 acceptance commit is not an ancestor of the delivery source")
    status_block = _status_m08_block((root / "docs/marvel_migration/STATUS.yaml").read_text(encoding="utf-8"))
    if "accepted: true" not in status_block:
        raise ValueError("STATUS.yaml does not mark M08 accepted")
    if f"commit: {m08_commit}" not in status_block and "commit: self" not in status_block:
        raise ValueError("STATUS.yaml M08 commit does not match the accepted commit")
    report = (root / "docs/marvel_migration/reports/M08_remote_pilot.md").read_text(encoding="utf-8")
    if "阶段状态：`accepted`" not in report or "不得复用旧 full Dataset" not in report:
        raise ValueError("M08 report does not carry the accepted/new-delivery boundary")
    return {
        "schema": "guandan-marvel-m09-delivery-safety-check-v1",
        "valid": True,
        "branch": branch,
        "head": head,
        "git_clean": not bool(status),
        "m08_acceptance_commit": m08_commit,
        "m08_commit_is_ancestor": ancestor,
        "m08_locked_config_sha256": validated["locked_config_sha256"],
        "plan_hash": validated["plan_hash"],
        "package_id": plan["package_id"],
        "forbidden_package_ids_absent": plan["package_id"] not in set(plan["forbidden_reuse_package_ids"]),
        "profile_validation": profile_validation,
        "selection_coverage": validated["selection_coverage"],
        "main_full_generated": False,
        "final_holdout_generated": False,
        "remote_verified": False,
        "model_strength": False,
    }


def _readme(plan: Mapping[str, object]) -> str:
    games = int(plan["algorithm_pilot"]["completed_games_per_algorithm"])
    iterations = int(plan["algorithm_pilot"]["expected_iterations"])
    return f"""# 掼蛋 MARVEL M09 受控算法 Pilot 交付\n\n这是新的 M09 Kaggle T4×2 交付，package id 为 `{plan['package_id']}`。它不复用 M07/M08 的旧 `full` Dataset。\n\n## 必须按顺序运行\n\n1. 上传 `datasets/preflight` 中的全部文件并 Run All。\n2. 分别上传 `datasets/ippo_pilot` 和 `datasets/vrpo_pilot`，各自建立独立私有 Dataset，并分别 Run All。\n3. 两个算法都必须精确完成 {games} 个 completed games（{iterations} iterations × global collect 4）；wall time 只允许作为安全停止，不是比较预算。\n4. 若某个 pilot 中断，只使用同算法的 `datasets/<algorithm>_resume` 加该算法最近下载的 recovery Dataset 继续。IPPO/VRPO recovery 不得交叉。\n5. 下载每次运行的 `guandan-marvel-output.zip`、`guandan-marvel-recovery.zip` 和 Kaggle console log，返回本地审查。\n\n## 当前边界\n\n- 当前仅交付相同完成对局预算的 IPPO/VRPO controlled pilot；主算法尚未选择。\n- 主算法 long training Dataset 尚未生成；必须先审查两份真实远端日志和 checkpoint。\n- selection 使用相同的 65 个 deal clusters，覆盖 13 级牌、四 leader seats、none/double-down/head-third/head-last/cancelled 五种贡还上下文。\n- 该 bounded evaluation 不是 formal champion promotion，也不是 final holdout。\n- 未经真实 Kaggle 返回证据，不得称 M09 accepted、正式训练完成或达到任何强度等级。\n\n## 每个 Kaggle Dataset 上传文件\n\n保持目录内文件名不变：`guandan_marvel_dataset.json`、`guandan_marvel_bootstrap.py`、`guandan-marvel-source.zip`、`source_manifest.json`、`{PLAN_COPY_NAME}`、`{LOCKED_CONFIG_COPY_NAME}`。\n\nNotebook 使用本目录的 `{NOTEBOOK_NAME}`，Accelerator 选择 **GPU T4 x2**，点击 Run All，不修改代码。\n"""


def build_m09_delivery(
    repository: str | Path,
    delivery_root: str | Path,
    *,
    commit: str | None = "HEAD",
    allow_dirty: bool = False,
) -> dict[str, object]:
    root = Path(repository).resolve()
    output = Path(delivery_root).resolve()
    safety = run_m09_safety_check(root, require_clean=not allow_dirty)
    validated = validate_m09_plan(root)
    plan = validated["plan"]
    locked = validated["locked_config"]
    profiles = build_m09_profiles(plan, locked)
    profile_validation = validate_m09_profiles(plan, locked, profiles)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    source_build = build_source_archive(
        root, output / SOURCE_ARCHIVE_NAME, commit=commit,
        allow_dirty=allow_dirty, package_id=str(plan["package_id"]),
    )
    source_manifest = source_build["source_manifest"]
    atomic_json(output / SOURCE_MANIFEST_NAME, source_manifest)
    source_manifest_sha = sha256_file(output / SOURCE_MANIFEST_NAME)
    with zipfile.ZipFile(output / SOURCE_ARCHIVE_NAME) as archive:
        prefix = "guandan-marvel-source/"
        bootstrap_bytes = archive.read(prefix + "guandan_marvel/kaggle/bootstrap.py")
        plan_bytes = archive.read(prefix + "configs/marvel_m09_full_plan.json")
        locked_bytes = archive.read(prefix + "configs/marvel_m08_locked_config.json")
    bootstrap_path = output / BOOTSTRAP_NAME
    bootstrap_path.write_bytes(bootstrap_bytes)
    role_rows = []
    for role in M09_ROLES:
        profile = profiles[role]
        mode = str(profile["mode"])
        dataset_root = output / "datasets" / role
        dataset_root.mkdir(parents=True)
        shutil.copy2(output / SOURCE_ARCHIVE_NAME, dataset_root / SOURCE_ARCHIVE_NAME)
        shutil.copy2(output / SOURCE_MANIFEST_NAME, dataset_root / SOURCE_MANIFEST_NAME)
        (dataset_root / BOOTSTRAP_NAME).write_bytes(bootstrap_bytes)
        (dataset_root / PLAN_COPY_NAME).write_bytes(plan_bytes)
        (dataset_root / LOCKED_CONFIG_COPY_NAME).write_bytes(locked_bytes)
        dataset = {
            "schema": DATASET_MANIFEST_SCHEMA,
            "package_id": plan["package_id"],
            "created_at": _timestamp(),
            "mode": mode,
            "source": {
                "archive": SOURCE_ARCHIVE_NAME,
                "archive_sha256": source_build["archive_sha256"],
                "manifest": SOURCE_MANIFEST_NAME,
                "manifest_sha256": source_manifest_sha,
                "git_commit": source_manifest["git_commit"],
                "content_manifest_hash": source_manifest["manifest_hash"],
            },
            "bootstrap": {"path": BOOTSTRAP_NAME, "sha256": sha256_file(dataset_root / BOOTSTRAP_NAME)},
            "profiles": {mode: profile},
            "recovery": {"required": mode == "resume", "auto_discovery": True},
            "m09": {
                "role": role,
                "plan": {"path": PLAN_COPY_NAME, "sha256": sha256_file(dataset_root / PLAN_COPY_NAME), "plan_hash": plan["plan_hash"]},
                "locked_config": {"path": LOCKED_CONFIG_COPY_NAME, "sha256": sha256_file(dataset_root / LOCKED_CONFIG_COPY_NAME)},
                "main_full_generated": False,
                "final_holdout_generated": False,
            },
            "claims": {
                "package_ready": True, "remote_ready": True, "remote_verified": False,
                "formal_training_complete": False, "model_strength": False, "m09_accepted": False,
            },
        }
        atomic_json(dataset_root / DATASET_MANIFEST_NAME, dataset)
        validate_dataset_manifest(dataset, manifest_path=dataset_root / DATASET_MANIFEST_NAME)
        role_rows.append({"role": role, "mode": mode, "dataset": f"datasets/{role}"})
    (output / NOTEBOOK_NAME).write_text(json.dumps(render_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / README_NAME).write_text(_readme(plan), encoding="utf-8", newline="\n")
    (output / SOURCE_ARCHIVE_NAME).unlink()
    (output / SOURCE_MANIFEST_NAME).unlink()
    bootstrap_path.unlink()
    validation = validate_m09_delivery(output)
    files = [path for path in output.rglob("*") if path.is_file() and path.name != DELIVERY_MANIFEST_NAME]
    body = {
        "schema": M09_PACKAGE_SCHEMA,
        "package_id": plan["package_id"],
        "created_at": _timestamp(),
        "source_commit": source_manifest["git_commit"],
        "workspace_snapshot": source_manifest["workspace_snapshot"],
        "plan_hash": plan["plan_hash"],
        "locked_config_sha256": validated["locked_config_sha256"],
        "default_recommended_role": "preflight",
        "datasets": role_rows,
        "notebook": NOTEBOOK_NAME,
        "readme": README_NAME,
        "files": relative_file_manifest(output, files),
        "safety": safety,
        "profile_validation": profile_validation,
        "validation": validation,
        "uploaded": False,
        "remote_ready": True,
        "remote_verified": False,
        "formal_training_complete": False,
        "model_strength": False,
        "m09_accepted": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    atomic_json(output / DELIVERY_MANIFEST_NAME, manifest)
    return manifest


def validate_m09_delivery(delivery_root: str | Path) -> dict[str, object]:
    root = Path(delivery_root).resolve()
    delivery_manifest_path = root / DELIVERY_MANIFEST_NAME
    delivery_manifest_verified = False
    if delivery_manifest_path.is_file():
        payload = _load_object(delivery_manifest_path)
        if payload.get("schema") != M09_PACKAGE_SCHEMA:
            raise ValueError("M09 delivery manifest schema mismatch")
        if payload.get("manifest_hash") != canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"}):
            raise ValueError("M09 delivery manifest hash mismatch")
        declared = {row["path"]: row for row in payload.get("files", [])}
        actual = {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*") if path.is_file() and path.name != DELIVERY_MANIFEST_NAME
        }
        if set(declared) != set(actual):
            raise ValueError("M09 delivery file set mismatch")
        for name, path in actual.items():
            if int(declared[name]["bytes"]) != path.stat().st_size or declared[name]["sha256"] != sha256_file(path):
                raise ValueError(f"M09 delivery file hash mismatch: {name}")
        delivery_manifest_verified = True
    notebook = validate_notebook(root / NOTEBOOK_NAME)
    role_validations = []
    commits = set()
    archive_hashes = set()
    package_ids = set()
    plan_hashes = set()
    locked_hashes = set()
    for role in M09_ROLES:
        dataset_root = root / "datasets" / role
        manifest_path = dataset_root / DATASET_MANIFEST_NAME
        dataset = _load_object(manifest_path)
        validate_dataset_manifest(dataset, manifest_path=manifest_path)
        if dataset.get("m09", {}).get("role") != role:
            raise ValueError(f"M09 dataset role mismatch: {role}")
        if dataset.get("claims", {}).get("remote_verified") is not False or dataset.get("claims", {}).get("m09_accepted") is not False:
            raise ValueError("M09 local delivery contains a false remote/acceptance claim")
        source_validation = validate_source_archive(dataset_root / SOURCE_ARCHIVE_NAME)
        commits.add(source_validation["git_commit"])
        archive_hashes.add(source_validation["archive_sha256"])
        package_ids.add(source_validation["package_id"])
        plan_copy = _load_object(dataset_root / PLAN_COPY_NAME)
        locked_copy_hash = sha256_file(dataset_root / LOCKED_CONFIG_COPY_NAME)
        if sha256_file(dataset_root / PLAN_COPY_NAME) != dataset["m09"]["plan"]["sha256"]:
            raise ValueError(f"M09 plan copy hash mismatch: {role}")
        if plan_copy.get("plan_hash") != dataset["m09"]["plan"]["plan_hash"]:
            raise ValueError(f"M09 plan identity mismatch: {role}")
        if locked_copy_hash != dataset["m09"]["locked_config"]["sha256"]:
            raise ValueError(f"M08 locked config copy hash mismatch: {role}")
        plan_hashes.add(plan_copy["plan_hash"])
        locked_hashes.add(locked_copy_hash)
        role_validations.append({"role": role, "mode": dataset["mode"], "valid": True})
    if any(len(values) != 1 for values in (commits, archive_hashes, package_ids, plan_hashes, locked_hashes)):
        raise ValueError("M09 role datasets do not share one immutable source/plan/config identity")
    package_id = next(iter(package_ids))
    plan = _load_object(root / "datasets/preflight" / PLAN_COPY_NAME)
    if package_id != plan["package_id"] or package_id in set(plan["forbidden_reuse_package_ids"]):
        raise ValueError("M09 delivery package id is invalid or reused")
    return {
        "schema": M09_VALIDATION_SCHEMA,
        "valid": True,
        "delivery_manifest_verified": delivery_manifest_verified,
        "notebook": {**notebook, "path": NOTEBOOK_NAME},
        "datasets": role_validations,
        "source_commit": next(iter(commits)),
        "source_archive_sha256": next(iter(archive_hashes)),
        "package_id": package_id,
        "plan_hash": next(iter(plan_hashes)),
        "locked_config_sha256": next(iter(locked_hashes)),
        "old_full_dataset_reused": False,
        "main_full_generated": False,
        "final_holdout_generated": False,
        "remote_verified": False,
        "m09_accepted": False,
    }


__all__ = [
    "LOCKED_CONFIG_COPY_NAME", "M09_PACKAGE_SCHEMA", "M09_PLAN_SCHEMA", "M09_ROLES",
    "M09_VALIDATION_SCHEMA", "PLAN_COPY_NAME", "build_m09_delivery", "build_m09_profiles",
    "run_m09_safety_check", "validate_m09_delivery", "validate_m09_plan", "validate_m09_profiles",
]
