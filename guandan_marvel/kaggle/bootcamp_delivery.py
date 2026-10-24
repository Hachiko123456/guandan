# SPDX-License-Identifier: MIT
"""Build the M09 bootcamp week-1 segmented training delivery.

Week-1 recipe (see plan JSON for full detail):

* warm start from the M09 iter2000 evaluation checkpoint (policy only;
  optimizer/sampler/counters reset);
* 5x larger sampling batch (global rollout 4 / collect 20 games per
  iteration); the NTP auxiliary loss is disabled (ntp_weight=0.0) because the
  measured next-observation deltas carry no information beyond the acting
  player's own action event;
* bounded anchor evaluations every 250 iterations with a record-only plateau
  gate (alert metadata only; it never stops the run);
* three ~10-hour Kaggle T4x2 sessions chained via recovery bundles.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

import torch

from .budget import build_completed_game_budget_contract
from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .discovery import (
    BOOTSTRAP_NAME, DATASET_MANIFEST_NAME, DATASET_MANIFEST_SCHEMA, validate_dataset_manifest,
)
from .package import (
    SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, build_source_archive, validate_source_archive,
)

BOOTCAMP_DELIVERY_SCHEMA = "guandan-marvel-m09-bootcamp-delivery-v1"
BOOTCAMP_VALIDATION_SCHEMA = "guandan-marvel-m09-bootcamp-delivery-validation-v1"
BOOTCAMP_PLAN_SCHEMA = "guandan-marvel-bootcamp-week1-plan-v1"
BOOTCAMP_DELIVERY_ID = "guandan-marvel-m09-bootcamp-week1-v1"
BOOTCAMP_PLAN_NAME = "marvel_bootcamp_week1_plan.json"
LOCK_NAME = "marvel_m08_locked_config.json"
NOTEBOOK_NAME = "guandan_marvel_bootcamp_week1.ipynb"
README_NAME = "README_BOOTCAMP_WEEK1.md"
WARM_START_NAME = "warm_start_policy.pt"
COMPARISON_ID = "m09-ippo-bootcamp-week1-v1"

SESSIONS: tuple[dict[str, object], ...] = (
    {"id": "session1_start", "mode": "full", "target_iteration": 500, "target_completed_games": 10_000},
    {"id": "session2_continue", "mode": "resume", "target_iteration": 1_000, "target_completed_games": 20_000},
    {"id": "session3_continue", "mode": "resume", "target_iteration": 1_500, "target_completed_games": 30_000},
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _training_block() -> dict[str, object]:
    return {
        "seed": 570700,
        "environment_rank_seed_stride": 1_000_003,
        "rollout_batch_size": 4,
        "collect_batch_size": 20,
        "replay_buffer_scale": 8,
        "mini_batches": 8,
        "agent_epoch": 4,
        "critic_aux_epoch": 4,
        "base_lr": 4.0e-4,
        "base_clip_coef": 0.02,
        "base_reg_coef": 0.1,
        "lr_stable_iterations": 500,
        "reg_stable_iterations": 500,
        "agent_merging_beta": 0.0,
        "use_muon": False,
        "amp_enabled": False,
        "compile_enabled": False,
        "ntp_weight": 0.0,
        "ntp_pos_weight": 10.0,
        "max_sampler_bytes_per_rank": 4_294_967_296,
        "checkpoint_interval": 50,
        "evaluation_interval": 250,
        "checkpoint_keep_last": 2,
        "evaluation_keep_last": 4,
        "max_wall_seconds": 39_600,
        "max_wall_seconds_per_session": 39_600,
        "model": {
            "architecture": "v1.3",
            "history_dim": 128,
            "feature_dim": 256,
            "decoder_layers": 4,
            "decoder_heads": 4,
            "state_layers": 3,
            "action_layers": 2,
            "fuse_layers": 2,
        },
    }


def _evaluation_block() -> dict[str, object]:
    return {
        "base_deals": 8,
        "seed_base": 1_200_000_000,
        "opponents": ["random", "rule"],
        "seat_rotations": [0, 1, 2, 3],
        "timeout_seconds": 1800,
    }


def _plateau_gate_block() -> dict[str, object]:
    return {
        "schema": "guandan-marvel-plateau-gate-v1",
        "enabled": True,
        "metric": "rule_win_rate",
        "min_evaluations": 3,
        "patience": 2,
        "min_delta": 0.01,
    }


def build_bootcamp_plan(*, warm_start_sha256: str, warm_start_iteration: int) -> dict[str, object]:
    body = {
        "schema": BOOTCAMP_PLAN_SCHEMA,
        "plan_id": BOOTCAMP_DELIVERY_ID,
        "created_at": _timestamp(),
        "warm_start": {
            "algorithm": "ippo",
            "iteration": int(warm_start_iteration),
            "policy_sha256": str(warm_start_sha256),
            "note": "M09 unified final candidate ippo_iter002000; policy-only warm start resets optimizer, sampler and counters",
        },
        "training": _training_block(),
        "evaluation": _evaluation_block(),
        "plateau_gate": _plateau_gate_block(),
        "sessions": [dict(row) for row in SESSIONS],
        "training_distribution": {
            "decision": "retain_m08_locked_distribution_for_recipe_comparability",
            "level_rank": "Rank.TWO",
            "previous_result": None,
            "generator_version": "guandan-marvel-env-batch-continuous-prng-v1",
            "note": "M09 main training used the same locked distribution; only the sampling batch, NTP auxiliary loss and seeds change",
        },
        "rationale": {
            "batch": "PPO gradient variance scales with 1/batch; 4 games/iteration was toy-scale",
            "ntp": "removed: 19,039 measured acting-player steps show the next-observation delta is exactly the actor's own action event (0.00% residual information after framing/own-action/channel-spoiled tokens)",
            "plateau_gate": "record-only alert: on 8 fixed deals the deterministic anchor metric can fall below a running max+1pp while learning is slow, so the gate must never stop a session",
        },
        "claims": {
            "final_holdout_generated": False,
            "m09_accepted": False,
            "bootcamp_complete": False,
            "strength_claim": False,
        },
    }
    return {**body, "plan_hash": canonical_hash(body)}


def _budget(*, games: int, iterations: int, plan_hash: str) -> dict[str, object]:
    return build_completed_game_budget_contract(
        comparison_id=COMPARISON_ID, algorithm="ippo",
        target_completed_games=int(games), global_collect_batch_size=20,
        expected_iterations=int(iterations), selection_dataset_hash=str(plan_hash),
    )


def _full_profile(plan: dict[str, object], *, warm_spec: dict[str, object]) -> dict[str, object]:
    session = SESSIONS[0]
    training = dict(plan["training"])
    training["target_iterations"] = int(session["target_iteration"])
    return {
        "schema": "guandan-marvel-kaggle-profile-v1",
        "mode": "full",
        "algorithm": "ippo",
        "required_gpu_count": 2,
        "required_gpu_name_contains": "T4",
        "ddp": {"launcher": "torchrun", "world_size": 2, "backend": "nccl", "require_cuda": True},
        "training": training,
        "warm_start": warm_spec,
        "evaluation": dict(plan["evaluation"]),
        "plateau_gate": dict(plan["plateau_gate"]),
        "completed_game_budget": _budget(
            games=int(session["target_completed_games"]),
            iterations=int(session["target_iteration"]),
            plan_hash=str(plan["plan_hash"]),
        ),
        "telemetry_interval_seconds": 5,
        "m09": {
            "role": str(session["id"]),
            "plan_id": plan["plan_id"],
            "plan_hash": plan["plan_hash"],
            "session": dict(session),
            "final_holdout": False,
            "strength_claim": False,
        },
    }


def _resume_profile(plan: dict[str, object], session: dict[str, object]) -> dict[str, object]:
    training = plan["training"]
    return {
        "schema": "guandan-marvel-kaggle-profile-v1",
        "mode": "resume",
        "expected_algorithm": "ippo",
        "required_gpu_count": 2,
        "required_gpu_name_contains": "T4",
        "ddp": {"launcher": "torchrun", "world_size": 2},
        "resume": {
            "target_iterations": int(session["target_iteration"]),
            "checkpoint_interval": int(training["checkpoint_interval"]),
            "evaluation_interval": int(training["evaluation_interval"]),
            "max_wall_seconds": int(training["max_wall_seconds_per_session"]),
            "checkpoint_keep_last": int(training["checkpoint_keep_last"]),
            "evaluation_keep_last": int(training["evaluation_keep_last"]),
        },
        "evaluation": dict(plan["evaluation"]),
        "plateau_gate": dict(plan["plateau_gate"]),
        "completed_game_budget": _budget(
            games=int(session["target_completed_games"]),
            iterations=int(session["target_iteration"]),
            plan_hash=str(plan["plan_hash"]),
        ),
        "telemetry_interval_seconds": 5,
        "m09": {
            "role": str(session["id"]),
            "plan_id": plan["plan_id"],
            "plan_hash": plan["plan_hash"],
            "session": dict(session),
            "final_holdout": False,
            "strength_claim": False,
        },
    }


def _validate_notebook(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4:
        raise ValueError("notebook must use nbformat 4")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("notebook has no cells")
    sources = [
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]
    joined = "\n".join(sources)
    for marker in ("guandan_marvel_dataset.json", "guandan_marvel.kaggle.runner", "guandan-marvel-recovery.zip"):
        if marker not in joined:
            raise ValueError(f"notebook template lacks expected marker {marker!r}")
    return {"cells": len(cells), "code_cells": len(sources)}


def _session_manifest(
    *, session: dict[str, object], source: dict[str, object], content_manifest_hash: str,
    root: Path, plan: dict[str, object], bootstrap_path: Path, plan_path: Path, lock_path: Path,
) -> dict[str, object]:
    mode = str(session["mode"])
    profile = _full_profile(plan, warm_spec={
        "schema": "guandan-marvel-warm-start-input-v1",
        "path": WARM_START_NAME,
        "sha256": sha256_file(root / WARM_START_NAME),
        "bytes": (root / WARM_START_NAME).stat().st_size,
    }) if mode == "full" else _resume_profile(plan, session)
    manifest = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "package_id": str(source["package_id"]),
        "created_at": _timestamp(),
        "mode": mode,
        "source": {
            "archive": SOURCE_ARCHIVE_NAME,
            "archive_sha256": source["archive_sha256"],
            "manifest": SOURCE_MANIFEST_NAME,
            "manifest_sha256": sha256_file(root / SOURCE_MANIFEST_NAME),
            "git_commit": source["git_commit"],
            "content_manifest_hash": content_manifest_hash,
        },
        "bootstrap": {"path": BOOTSTRAP_NAME, "sha256": sha256_file(bootstrap_path)},
        "profiles": {mode: profile},
        "recovery": {"required": mode == "resume", "auto_discovery": mode == "resume"},
        "m09": {
            "role": str(session["id"]),
            "plan": {"path": BOOTCAMP_PLAN_NAME, "sha256": sha256_file(plan_path), "plan_hash": plan["plan_hash"]},
            "locked_config": {"path": LOCK_NAME, "sha256": sha256_file(lock_path)},
            "session": dict(session),
            "final_holdout_generated": False,
        },
        "claims": {
            "remote_ready": True, "remote_verified": False, "bootcamp_complete": False,
            "m09_accepted": False, "strength_claim": False,
        },
    }
    return manifest


def _write_upload_zip(root: Path, destination: Path) -> Path:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():
            raise ValueError(f"upload ZIP CRC failure: {destination}")
    temporary.replace(destination)
    return destination


def build_bootcamp_delivery(
    repository: str | Path,
    delivery_root: str | Path,
    *,
    warm_start_checkpoint: str | Path,
    notebook_template: str | Path,
    locked_config: str | Path,
    expected_warm_start_sha256: str | None = None,
) -> dict[str, object]:
    repo = Path(repository).resolve()
    output = Path(delivery_root).resolve()
    warm_source = Path(warm_start_checkpoint).resolve()
    if not warm_source.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {warm_source}")
    warm_sha = sha256_file(warm_source)
    if expected_warm_start_sha256 is not None and warm_sha != str(expected_warm_start_sha256):
        raise ValueError("warm-start checkpoint sha256 differs from the expected M09 candidate")
    payload = torch.load(warm_source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "guandan-marvel-evaluation-checkpoint-v1":
        raise ValueError("warm start must be a guandan-marvel-evaluation-checkpoint-v1 artifact")
    if payload.get("algorithm") != "ippo" or payload.get("architecture") != "v1.3":
        raise ValueError("warm start must be an ippo v1.3 policy")
    warm_iteration = int(payload.get("iteration", -1))
    lock_source = Path(locked_config).resolve()
    if not lock_source.is_file():
        raise FileNotFoundError(lock_source)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    sessions_root = output / "datasets"
    sessions_root.mkdir(parents=True)
    staging = output / "_staging"
    staging.mkdir()

    source_build = build_source_archive(
        repo, staging / SOURCE_ARCHIVE_NAME, commit="HEAD", allow_dirty=True,
    )
    atomic_json(staging / SOURCE_MANIFEST_NAME, source_build["source_manifest"])
    source_validation = validate_source_archive(staging / SOURCE_ARCHIVE_NAME)
    content_manifest_hash = json.loads(
        (staging / SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )["manifest_hash"]
    with zipfile.ZipFile(staging / SOURCE_ARCHIVE_NAME) as source_zip:
        (staging / BOOTSTRAP_NAME).write_bytes(
            source_zip.read("guandan-marvel-source/guandan_marvel/kaggle/bootstrap.py")
        )

    plan = build_bootcamp_plan(warm_start_sha256=warm_sha, warm_start_iteration=warm_iteration)
    atomic_json(staging / BOOTCAMP_PLAN_NAME, plan)

    rows = []
    uploads = []
    for session in SESSIONS:
        root = sessions_root / str(session["id"])
        root.mkdir(parents=True)
        for name in (SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, BOOTSTRAP_NAME, BOOTCAMP_PLAN_NAME):
            shutil.copy2(staging / name, root / name)
        shutil.copy2(lock_source, root / LOCK_NAME)
        if session["mode"] == "full":
            shutil.copy2(warm_source, root / WARM_START_NAME)
        manifest = _session_manifest(
            session=session, source=source_validation, content_manifest_hash=content_manifest_hash,
            root=root, plan=plan, bootstrap_path=root / BOOTSTRAP_NAME,
            plan_path=root / BOOTCAMP_PLAN_NAME, lock_path=root / LOCK_NAME,
        )
        atomic_json(root / DATASET_MANIFEST_NAME, manifest)
        validate_dataset_manifest(manifest, manifest_path=root / DATASET_MANIFEST_NAME)
        upload = _write_upload_zip(root, output / f"{session['id']}-upload.zip")
        uploads.append({"session": session["id"], "path": upload.name, "sha256": sha256_file(upload), "bytes": upload.stat().st_size})
        rows.append({
            "session": session["id"],
            "mode": session["mode"],
            "target_iteration": session["target_iteration"],
            "target_completed_games": session["target_completed_games"],
            "manifest_hash": canonical_hash({k: v for k, v in manifest.items() if k != "created_at"}),
        })

    notebook_path = output / NOTEBOOK_NAME
    shutil.copy2(Path(notebook_template).resolve(), notebook_path)
    notebook_validation = _validate_notebook(notebook_path)

    readme = _readme(plan, rows)
    (output / README_NAME).write_text(readme, encoding="utf-8", newline="\n")

    validation = validate_bootcamp_delivery(output)
    files = [
        path for path in output.rglob("*")
        if path.is_file() and path.name != "delivery_manifest.json" and "_staging" not in path.parts
    ]
    body = {
        "schema": BOOTCAMP_DELIVERY_SCHEMA,
        "delivery_id": BOOTCAMP_DELIVERY_ID,
        "created_at": _timestamp(),
        "source_package_id": source_validation["package_id"],
        "source_commit": source_validation["git_commit"],
        "source_archive_sha256": source_validation["archive_sha256"],
        "source_content_manifest_hash": content_manifest_hash,
        "plan_hash": plan["plan_hash"],
        "warm_start": {"sha256": warm_sha, "iteration": warm_iteration},
        "sessions": rows,
        "uploads": uploads,
        "notebook": NOTEBOOK_NAME,
        "notebook_validation": notebook_validation,
        "files": relative_file_manifest(output, files),
        "validation": validation,
        "uploaded": False,
        "remote_ready": True,
        "remote_verified": False,
        "bootcamp_complete": False,
        "final_holdout_run": False,
        "m09_accepted": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    atomic_json(output / "delivery_manifest.json", manifest)
    shutil.rmtree(staging)
    return manifest


def validate_bootcamp_delivery(delivery_root: str | Path) -> dict[str, object]:
    root = Path(delivery_root).resolve()
    session_hashes = []
    rows = []
    for session in SESSIONS:
        session_root = root / "datasets" / str(session["id"])
        manifest_path = session_root / DATASET_MANIFEST_NAME
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_dataset_manifest(payload, manifest_path=manifest_path)
        mode = str(session["mode"])
        if payload.get("mode") != mode:
            raise ValueError(f"{session['id']} dataset mode mismatch")
        if payload["m09"].get("role") != str(session["id"]):
            raise ValueError(f"{session['id']} role mismatch")
        profile = payload["profiles"][mode]
        if mode == "full":
            warm = profile.get("warm_start")
            if not isinstance(warm, dict) or warm.get("schema") != "guandan-marvel-warm-start-input-v1":
                raise ValueError("session1 must declare the warm-start input")
            warm_path = session_root / WARM_START_NAME
            if not warm_path.is_file() or sha256_file(warm_path) != warm["sha256"]:
                raise ValueError("session1 warm-start checkpoint identity mismatch")
            gates = profile.get("plateau_gate")
            if not isinstance(gates, dict) or gates.get("schema") != "guandan-marvel-plateau-gate-v1":
                raise ValueError("session1 plateau gate is missing")
        else:
            controls = profile.get("resume")
            if int(controls["target_iterations"]) != int(session["target_iteration"]):
                raise ValueError(f"{session['id']} resume target mismatch")
            if not bool(payload["recovery"].get("required")):
                raise ValueError(f"{session['id']} must require a recovery input")
        if profile.get("completed_game_budget", {}).get("target_completed_games") != int(session["target_completed_games"]):
            raise ValueError(f"{session['id']} completed-game budget mismatch")
        session_hashes.append(sha256_file(session_root / SOURCE_ARCHIVE_NAME))
        rows.append({
            "session": session["id"], "mode": mode,
            "target_iteration": session["target_iteration"],
            "target_completed_games": session["target_completed_games"],
        })
    if len(set(session_hashes)) != 1:
        raise ValueError("all sessions must share one identical source archive")
    notebook = root / NOTEBOOK_NAME
    if not notebook.is_file():
        raise FileNotFoundError("delivery notebook is missing")
    return {
        "schema": BOOTCAMP_VALIDATION_SCHEMA,
        "valid": True,
        "sessions": rows,
        "source_archive_sha256": session_hashes[0],
        "notebook": _validate_notebook(notebook),
        "remote_verified": False,
    }


def _readme(plan: dict[str, object], rows: list[dict[str, object]]) -> str:
    schedule = "\n".join(
        f"{index + 1}. **{row['session']}**（{row['mode']}）：目标 iteration {row['target_iteration']} / "
        f"{row['target_completed_games']} 局"
        for index, row in enumerate(rows)
    )
    return f"""# M09 Bootcamp Week-1 上传与运行说明

本交付把 bootcamp 第一周拆成 3 个 Kaggle session。所有 session 共用同一个 source archive
（sha256 见 `delivery_manifest.json.source_archive_sha256`），warm start 只出现在 session1。

## 顺序

{schedule}

每个 session 是一个独立 Kaggle Dataset（用对应的 `*-upload.zip` 上传，Kaggle 会自动解压）：

- **session1_start**：单独建 Dataset，导入 `{NOTEBOOK_NAME}`，GPU T4 x2，Run All。
  Output 里下载 `guandan-marvel-output.zip` 和 `guandan-marvel-recovery.zip`。
- **session2_continue**：把 session1 的 recovery.zip 作为**第二个 Kaggle Input**（独立 Dataset），
  再添加 session2 的 Dataset，同一个 Notebook Run All。
- **session3_continue**：用 session2 的 recovery.zip 重复上面的流程。

## 运行中会看到什么

- 每 250 iteration 自动跑一次锚点评估（8 副牌 × 4 座位 × rule/random，共 64 局，约 5-10 分钟）。
- `anchor_history.json` 会随 recovery 一起保存并跨 session 恢复。plateau gate 是永久 record-only：
  只记录告警（`plateau_alert` / `stale_count`），事件与 payload 均无停机语义，session 只会以
  `complete` 或 `budget_stopped` 结束。若看到 `stale_count>=2`，只说明 8 副牌固定牌面的 rule
  读数低于历史最好值 +1pp，需结合 CI（8 副牌时聚类区间退化为 [0,1]）人工判断，不要据此中止计划。
- 每个 session 预计 8-10.5 小时（含评估/启动开销；端到端吞吐 0.334 局/秒，每 session 10,000 局
  ≈ 8.3h 训练 + 约 1-2h 开销），控制在 30h/周的配额内。

## 判定

- 看 `anchor_history.json`：vs rule 的 win_rate 是否持续上升（预期从 ~55% 起步）；
  告警（`stale_count>=2`）不停机、不单独构成结论。
- session3 结束后重建 comparison 交付并运行：
  `python scripts/build_m09_fabledan_comparison_delivery.py --output <dir> --m09-archive <m09.zip>
  --fabledan-weights <npz> --fabledan-src <dir> --bootcamp-final-checkpoint <session3 final 评估 checkpoint>`；
  - 主终点：paired `bootcamp_final_vs_m09_iter2000`（256 副 × 4 轮换 × 2 臂 = 2048 局）的
    cluster-bootstrap 95% CI 下界 > 0；
  - 次要：1024 局 h2h vs FD 的 `by_context` 三读数（全集 / level=15 子集 / OOD 分解）。
- 本交付不生成 final holdout，不宣称 M09 accepted，也不构成强度声明。

计划 hash：`{plan['plan_hash']}`。修改任何训练语义字段都必须重新构建交付（哈希会变）。
"""
