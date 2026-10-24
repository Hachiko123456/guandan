# SPDX-License-Identifier: MIT
"""MARVEL v14 phase-4 segmented Kaggle delivery (start + repeatable resume).

Two datasets share one source package identity:

* ``start``  — ``full`` profile with a policy-only warm start; a fresh trainer
  resets optimizer/sampler/counters.
* ``resume`` — ``resume`` profile that is *repeatable*: every session resumes
  from the latest recovery bundle and runs until the absolute final iteration.
  A session that hits ``max_wall_seconds`` stops at an iteration boundary,
  atomically publishes a recovery bundle and returns ``budget_stopped``; the
  next resume session continues from the same absolute target.

The delivery records the phase-4 reward semantics, the 52-context schedule and
the warm-start curriculum, and carries the warm-start identity, the semantic
contract hash and the native backend hash so a doc/hash mismatch is detectable.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import sys
import time
from pathlib import Path
import shutil
import zipfile

import torch

from ..buffer import BUFFER_STATE_SCHEMA
from ..config import (
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_FUSE_MODEL_ARCHITECTURE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
)
from ..context_schedule import CURRICULUM_SCHEMA, build_rank_leader_schedule, schedule_coverage
from ..reward import REWARD_SEMANTICS, TRAINING_REWARD_SCALE, reward_semantics_block
from ..runtime.sampler import SAMPLER_CONTRACT_SCHEMA, SAMPLER_STATE_SCHEMA
from ..training.checkpoints import build_semantic_configuration, build_semantic_contract
from .budget import build_completed_game_budget_contract
from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .discovery import (
    BOOTSTRAP_NAME, DATASET_MANIFEST_NAME, DATASET_MANIFEST_SCHEMA, validate_dataset_manifest,
)
from .package import (
    SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, build_source_archive, validate_source_archive,
)
from .profiles import build_train_config

PHASE4_DELIVERY_SCHEMA = "guandan-marvel-v14-phase4-delivery-v1"
PHASE4_VALIDATION_SCHEMA = "guandan-marvel-v14-phase4-delivery-validation-v1"
PHASE4_RUNTIME_VALIDATION_SCHEMA = "guandan-marvel-v14-phase4-delivery-runtime-validation-v1"
PHASE4_WARM_START_IDENTITY_SCHEMA = "guandan-marvel-v14-phase4-warm-start-identity-v1"
PHASE4_PLAN_SCHEMA = "guandan-marvel-v14-phase4-plan-v1"
PHASE4_DELIVERY_ID = "guandan-marvel-v14-phase4-reward-context-v1"
PHASE4_PLAN_NAME = "marvel_v14_phase4_plan.json"
NOTEBOOK_NAME = "guandan_marvel_phase4.ipynb"
README_NAME = "README_PHASE4.md"
WARM_START_NAME = "warm_start_policy.pt"
COMPARISON_ID = "v14-phase4-ippo"
ABSOLUTE_FINAL_ITERATION = 1500

# --- phase 4A delivery integration: target model + production warm start ------
# The start profile must deliver the *accepted* phase-3 v1.4 actor-residual-fuse
# architecture under AMP; the packaged warm start must be the production-scale
# M09 v1.3 policy-only checkpoint, never the phase-3 synthetic smoke artifact.
TARGET_MODEL_ARCHITECTURE = MACRO_FUSE_MODEL_ARCHITECTURE  # "v1.4-macro-fuse"
PRODUCTION_WARM_START_ARCHITECTURE = "v1.3"
PRODUCTION_WARM_START_HISTORY_DIM = 128
PRODUCTION_WARM_START_FEATURE_DIM = 256
PRODUCTION_WARM_START_PARAMETER_COUNT = 2_061_314
PRODUCTION_WARM_START_STATE_DICT_KEYS = 78
TARGET_MODEL_PARAMETER_COUNT = 2_783_234
TARGET_MODEL_STATE_DICT_KEYS = 90
# The phase-3 smoke artifact is deliberately *small* (feature_dim=24/history=16).
# It is only ever an acceptable input to the unit tests, never the delivery.
REJECTED_SYNTHETIC_WARM_START_FEATURE_DIM = 24
REJECTED_SYNTHETIC_WARM_START_HISTORY_DIM = 16
CURRICULUM = {
    "schema": "guandan-marvel-context-curriculum-v1",
    "old_context": {"level_rank": 15, "leader_seat": 0, "previous_result": None},
    "warmup_iterations": 20,
    "blend_iterations": 80,
    "resolution": 1000,
}

SEGMENTS: tuple[dict[str, object], ...] = (
    {"id": "start", "mode": "full"},
    {"id": "resume", "mode": "resume"},
)


def _state_dict_parameter_count(state: Mapping[str, object]) -> int:
    return int(sum(int(tensor.numel()) for tensor in state.values()))


def validate_production_warm_start(
    payload: object, *, source: str = "<warm-start>",
) -> dict[str, object]:
    """Fail closed unless *payload* is the production-scale M09 v1.3 policy.

    This delivery gate blocks the phase-3 synthetic smoke artifact
    (``feature_dim=24``/``history_dim=16``) from ever being packaged as the
    formal warm start: the shipped start profile is a production-scale v1.4
    target, so a small synthetic checkpoint could never load into it.
    """
    from ..config import deserialize_train_config

    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: warm start payload must be a mapping")
    if payload.get("schema") != "guandan-marvel-evaluation-checkpoint-v1":
        raise ValueError(
            f"{source}: warm start must be a guandan-marvel-evaluation-checkpoint-v1 artifact"
        )
    if payload.get("algorithm") != "ippo":
        raise ValueError(f"{source}: warm start must be an ippo policy")
    architecture = str(payload.get("architecture"))
    if architecture != PRODUCTION_WARM_START_ARCHITECTURE:
        raise ValueError(
            f"{source}: warm start architecture must be {PRODUCTION_WARM_START_ARCHITECTURE!r} "
            f"(the production M09 v1.3 policy); got {architecture!r}"
        )
    state = payload.get("agent_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{source}: warm start lacks agent_state_dict")
    if any(str(key).startswith("actor_fuse.") for key in state):
        raise ValueError(f"{source}: a production v1.3 warm start must not carry actor_fuse.* keys")
    config = deserialize_train_config(payload.get("resolved_config"))
    if config is None:
        raise ValueError(f"{source}: warm start resolved_config could not be decoded")
    model = config.model
    if (int(model.history_dim), int(model.feature_dim)) != (
        PRODUCTION_WARM_START_HISTORY_DIM, PRODUCTION_WARM_START_FEATURE_DIM,
    ):
        raise ValueError(
            f"{source}: warm start must be the production-scale policy "
            f"(history_dim={PRODUCTION_WARM_START_HISTORY_DIM}, "
            f"feature_dim={PRODUCTION_WARM_START_FEATURE_DIM}); got "
            f"history_dim={model.history_dim}, feature_dim={model.feature_dim}. "
            "The phase-3 synthetic smoke checkpoint is not a valid delivery warm start."
        )
    parameter_count = _state_dict_parameter_count(state)
    if parameter_count != PRODUCTION_WARM_START_PARAMETER_COUNT:
        raise ValueError(
            f"{source}: warm start parameter count must be {PRODUCTION_WARM_START_PARAMETER_COUNT}; "
            f"got {parameter_count}"
        )
    if len(state) != PRODUCTION_WARM_START_STATE_DICT_KEYS:
        raise ValueError(
            f"{source}: warm start state_dict must carry "
            f"{PRODUCTION_WARM_START_STATE_DICT_KEYS} keys; got {len(state)}"
        )
    return {
        "schema": PHASE4_WARM_START_IDENTITY_SCHEMA,
        "architecture": architecture,
        "algorithm": "ippo",
        "history_dim": int(model.history_dim),
        "feature_dim": int(model.feature_dim),
        "parameter_count": parameter_count,
        "state_dict_keys": len(state),
        "iteration": int(payload.get("iteration", -1)),
        "resolved_config_hash": payload.get("resolved_config_hash"),
    }


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
        "base_lr": 4.0e-4,
        "base_clip_coef": 0.02,
        "base_reg_coef": 0.1,
        "lr_stable_iterations": 500,
        "reg_stable_iterations": 500,
        "agent_merging_beta": 0.0,
        "use_muon": False,
        "amp_enabled": True,
        "compile_enabled": False,
        "ntp_weight": 0.0,
        "ntp_pos_weight": 10.0,
        "max_sampler_bytes_per_rank": 4_294_967_296,
        "checkpoint_interval": 50,
        "evaluation_interval": 250,
        "checkpoint_keep_last": 2,
        "evaluation_keep_last": 4,
        "max_wall_seconds": 12_600,
        "max_wall_seconds_per_session": 12_600,
        "target_iterations": ABSOLUTE_FINAL_ITERATION,
        # Phase-4 training context: the 52-context 7A schedule and its curriculum
        # are part of the *profile* so the real TrainConfig (and therefore the
        # semantic contract) can be reconstructed from the delivery alone.
        "training_context_schedule": build_rank_leader_schedule(),
        "training_context_curriculum": dict(CURRICULUM),
        "training_semantics_profile": MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
        "policy_regularizer_mode": MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
        "reward_semantics": REWARD_SEMANTICS,
        "reward_scale": TRAINING_REWARD_SCALE,
        "model": {
            "architecture": TARGET_MODEL_ARCHITECTURE,
            "history_dim": 128,
            "feature_dim": 256,
            "decoder_layers": 4,
            "decoder_heads": 4,
            "state_layers": 3,
            "action_layers": 2,
            "fuse_layers": 2,
            "ffn_ratio": 2,
        },
    }


def _evaluation_block() -> dict[str, object]:
    return {
        "fast_anchor": {
            "role": "health_and_trend_only",
            "base_deals": 8,
            "seed_base": 1_200_000_000,
            "opponents": ["random", "rule"],
            "seat_rotations": [0, 1, 2, 3],
            "timeout_seconds": 1800,
            "strength_claim": False,
        },
        "selection": {
            "role": "selection",
            "base_deals": 52,
            "seed_base": 1_300_000_000,
            "level_ranks": list(range(3, 16)),
            "leader_seats": [0, 1, 2, 3],
            "seat_rotations": [0, 1, 2, 3],
            "opponents": ["random", "rule"],
            "cluster_by": "deal_id",
            "tribute_modes": ["none"],
            "previous_results": [None],
            "note": "7A only: the delivery training distribution has no tribute contexts; 7B runtime support is enabled after the full-observation audit",
        },
        "final_holdout": {
            "role": "final_holdout",
            "seed_base": 1_500_000_000,
            "generated": False,
            "note": "independent seed/hash; one-shot after the candidate is frozen; never used for tuning",
        },
    }


def _budget(*, games: int, iterations: int, plan_hash: str) -> dict[str, object]:
    return build_completed_game_budget_contract(
        comparison_id=COMPARISON_ID, algorithm="ippo",
        target_completed_games=int(games), global_collect_batch_size=20,
        expected_iterations=int(iterations), selection_dataset_hash=str(plan_hash),
    )


def build_phase4_plan(
    *, warm_start_sha256: str, warm_start_iteration: int,
    native_backend_hash: str, semantic_contract_hash: str,
    warm_start_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    schedule = build_rank_leader_schedule()
    training = _training_block()
    games = int(training["collect_batch_size"]) * int(training["target_iterations"])
    identity = dict(warm_start_identity or {})
    body = {
        "schema": PHASE4_PLAN_SCHEMA,
        "plan_id": PHASE4_DELIVERY_ID,
        "created_at": _timestamp(),
        "absolute_final_iteration": ABSOLUTE_FINAL_ITERATION,
        "target_completed_games": games,
        "warm_start": {
            "algorithm": "ippo",
            "architecture": str(identity.get("architecture", PRODUCTION_WARM_START_ARCHITECTURE)),
            "history_dim": int(identity.get("history_dim", PRODUCTION_WARM_START_HISTORY_DIM)),
            "feature_dim": int(identity.get("feature_dim", PRODUCTION_WARM_START_FEATURE_DIM)),
            "parameter_count": int(identity.get("parameter_count", PRODUCTION_WARM_START_PARAMETER_COUNT)),
            "state_dict_keys": int(identity.get("state_dict_keys", PRODUCTION_WARM_START_STATE_DICT_KEYS)),
            "iteration": int(warm_start_iteration),
            "policy_sha256": str(warm_start_sha256),
            "note": "production-scale policy-only warm start; optimizer, sampler, RNG and counters reset",
        },
        "target_model": {
            "architecture": TARGET_MODEL_ARCHITECTURE,
            "parameter_count": TARGET_MODEL_PARAMETER_COUNT,
            "state_dict_keys": TARGET_MODEL_STATE_DICT_KEYS,
            "amp_enabled": bool(training["amp_enabled"]),
        },
        "training": training,
        "reward": reward_semantics_block(),
        "training_distribution": {
            "decision": "phase4_7A_rank_leader_52_contexts",
            "context_schedule": schedule,
            "context_coverage": schedule_coverage(schedule),
            "curriculum": dict(CURRICULUM),
            "generator_version": "guandan-marvel-env-batch-continuous-prng-v1",
            "tribute_7B": "runtime_enabled_no_delivery_coverage",
        },
        "evaluation": _evaluation_block(),
        "semantic_contract_hash": str(semantic_contract_hash),
        "native_backend_hash": str(native_backend_hash),
        "segments": [dict(row) for row in SEGMENTS],
        "budget_stop_semantics": {
            "schema": "guandan-marvel-budget-stop-v1",
            "stop_unit": "iteration_boundary",
            "max_wall_seconds": 12_600,
            "on_stop": "save_checkpoint_then_atomic_recovery_then_return_budget_stopped",
            "platform_hard_kill_required": False,
            "repeatable_until": ABSOLUTE_FINAL_ITERATION,
        },
        "claims": {
            "final_holdout_generated": False,
            "strength_claim": False,
            "remote_verified": False,
        },
    }
    return {**body, "plan_hash": canonical_hash(body)}


def _full_profile(plan: dict[str, object], *, warm_spec: dict[str, object]) -> dict[str, object]:
    training = dict(plan["training"])
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
        "completed_game_budget": _budget(
            games=int(plan["target_completed_games"]),
            iterations=int(training["target_iterations"]),
            plan_hash=str(plan["plan_hash"]),
        ),
        "telemetry_interval_seconds": 5,
        "phase4": {
            "role": "start",
            "plan_id": plan["plan_id"],
            "plan_hash": plan["plan_hash"],
            "absolute_final_iteration": ABSOLUTE_FINAL_ITERATION,
            "budget_stop": "budget_stopped",
            "repeatable_resume": True,
            "final_holdout": False,
            "strength_claim": False,
        },
    }


def _resume_profile(plan: dict[str, object]) -> dict[str, object]:
    training = plan["training"]
    return {
        "schema": "guandan-marvel-kaggle-profile-v1",
        "mode": "resume",
        "expected_algorithm": "ippo",
        "required_gpu_count": 2,
        "required_gpu_name_contains": "T4",
        "ddp": {"launcher": "torchrun", "world_size": 2},
        "resume": {
            "target_iterations": int(training["target_iterations"]),
            "checkpoint_interval": int(training["checkpoint_interval"]),
            "evaluation_interval": int(training["evaluation_interval"]),
            "max_wall_seconds": int(training["max_wall_seconds_per_session"]),
            "checkpoint_keep_last": int(training["checkpoint_keep_last"]),
            "evaluation_keep_last": int(training["evaluation_keep_last"]),
        },
        "evaluation": dict(plan["evaluation"]),
        "completed_game_budget": _budget(
            games=int(plan["target_completed_games"]),
            iterations=int(training["target_iterations"]),
            plan_hash=str(plan["plan_hash"]),
        ),
        "telemetry_interval_seconds": 5,
        "phase4": {
            "role": "resume",
            "plan_id": plan["plan_id"],
            "plan_hash": plan["plan_hash"],
            "absolute_final_iteration": ABSOLUTE_FINAL_ITERATION,
            "budget_stop": "budget_stopped",
            "repeatable_resume": True,
            "final_holdout": False,
            "strength_claim": False,
        },
    }


def _session_manifest(
    *, segment: dict[str, object], source: dict[str, object], content_manifest_hash: str,
    root: Path, plan: dict[str, object], bootstrap_path: Path, plan_path: Path,
) -> dict[str, object]:
    mode = str(segment["mode"])
    warm_spec: dict[str, object] | None = None
    if mode == "full":
        warm_path = root / WARM_START_NAME
        warm_sha = sha256_file(warm_path)
        if warm_sha != str(plan["warm_start"]["policy_sha256"]):
            raise ValueError("packaged warm-start does not match the plan warm-start identity")
        warm_spec = {
            "schema": "guandan-marvel-warm-start-input-v1",
            "path": WARM_START_NAME,
            "sha256": warm_sha,
            "bytes": warm_path.stat().st_size,
            "architecture": plan["warm_start"]["architecture"],
            "history_dim": plan["warm_start"]["history_dim"],
            "feature_dim": plan["warm_start"]["feature_dim"],
            "parameter_count": plan["warm_start"]["parameter_count"],
            "state_dict_keys": plan["warm_start"]["state_dict_keys"],
            "iteration": plan["warm_start"]["iteration"],
        }
    profile = (
        _full_profile(plan, warm_spec=dict(warm_spec))
        if mode == "full" else _resume_profile(plan)
    )
    return {
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
        "phase4": {
            "role": str(segment["id"]),
            "plan": {"path": PHASE4_PLAN_NAME, "sha256": sha256_file(plan_path), "plan_hash": plan["plan_hash"]},
            "semantic_contract_hash": plan["semantic_contract_hash"],
            "native_backend_hash": plan["native_backend_hash"],
            "warm_start_identity": plan["warm_start"]["policy_sha256"],
            "final_holdout_generated": False,
        },
        "claims": {"remote_ready": True, "remote_verified": False, "strength_claim": False},
    }


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


def build_start_semantic_contract(training: dict[str, object], *, run_root: Path) -> dict[str, object]:
    """Build the *real* semantic contract from the start profile's training block."""
    from .. import cpp_game

    preview = {
        "schema": "guandan-marvel-kaggle-profile-v1",
        "mode": "full",
        "algorithm": "ippo",
        "ddp": {"launcher": "torchrun", "world_size": 2, "backend": "nccl", "require_cuda": True},
        "training": dict(training),
    }
    config = build_train_config(preview, game=cpp_game, run_root=run_root, local_cpu_test=False)
    semantic = build_semantic_configuration(config, algorithm="ippo")
    return build_semantic_contract(
        semantic_config=semantic,
        sampler_contract=SAMPLER_CONTRACT_SCHEMA,
        sampler_state_schema=SAMPLER_STATE_SCHEMA,
        buffer_schema=BUFFER_STATE_SCHEMA,
        q_target_contract=None,
    )


def build_phase4_delivery(
    repository: str | Path,
    delivery_root: str | Path,
    *,
    warm_start_checkpoint: str | Path,
    notebook_template: str | Path,
) -> dict[str, object]:
    repo = Path(repository).resolve()
    output = Path(delivery_root).resolve()
    warm_source = Path(warm_start_checkpoint).resolve()
    if not warm_source.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {warm_source}")
    warm_sha = sha256_file(warm_source)
    payload = torch.load(warm_source, map_location="cpu", weights_only=False)
    warm_identity = validate_production_warm_start(payload, source=str(warm_source))
    warm_iteration = int(warm_identity["iteration"])
    from nanopokerlib.guandan import BACKEND_BUILD_IDENTITY

    native_backend_hash = str(BACKEND_BUILD_IDENTITY["embedded_source_sha256"])
    semantic_contract = build_start_semantic_contract(_training_block(), run_root=output)
    semantic_contract_hash = str(semantic_contract["hash"])

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    datasets_root = output / "datasets"
    datasets_root.mkdir(parents=True)
    staging = output / "_staging"
    staging.mkdir()

    source_build = build_source_archive(repo, staging / SOURCE_ARCHIVE_NAME, commit="HEAD", allow_dirty=True)
    atomic_json(staging / SOURCE_MANIFEST_NAME, source_build["source_manifest"])
    source_validation = validate_source_archive(staging / SOURCE_ARCHIVE_NAME)
    content_manifest_hash = json.loads((staging / SOURCE_MANIFEST_NAME).read_text(encoding="utf-8"))["manifest_hash"]
    with zipfile.ZipFile(staging / SOURCE_ARCHIVE_NAME) as source_zip:
        (staging / BOOTSTRAP_NAME).write_bytes(
            source_zip.read("guandan-marvel-source/guandan_marvel/kaggle/bootstrap.py")
        )
    plan = build_phase4_plan(
        warm_start_sha256=warm_sha, warm_start_iteration=warm_iteration,
        native_backend_hash=native_backend_hash, semantic_contract_hash=semantic_contract_hash,
        warm_start_identity=warm_identity,
    )
    atomic_json(staging / PHASE4_PLAN_NAME, plan)

    rows, uploads = [], []
    for segment in SEGMENTS:
        root = datasets_root / str(segment["id"])
        root.mkdir(parents=True)
        for name in (SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, BOOTSTRAP_NAME, PHASE4_PLAN_NAME):
            shutil.copy2(staging / name, root / name)
        if segment["mode"] == "full":
            shutil.copy2(warm_source, root / WARM_START_NAME)
        manifest = _session_manifest(
            segment=segment, source=source_validation, content_manifest_hash=content_manifest_hash,
            root=root, plan=plan, bootstrap_path=root / BOOTSTRAP_NAME, plan_path=root / PHASE4_PLAN_NAME,
        )
        atomic_json(root / DATASET_MANIFEST_NAME, manifest)
        validate_dataset_manifest(manifest, manifest_path=root / DATASET_MANIFEST_NAME)
        upload = _write_upload_zip(root, output / f"{segment['id']}-upload.zip")
        uploads.append({"segment": segment["id"], "path": upload.name, "sha256": sha256_file(upload), "bytes": upload.stat().st_size})
        rows.append({
            "segment": segment["id"], "mode": segment["mode"],
            "absolute_final_iteration": ABSOLUTE_FINAL_ITERATION,
            "manifest_hash": canonical_hash({k: v for k, v in manifest.items() if k != "created_at"}),
        })

    notebook_path = output / NOTEBOOK_NAME
    shutil.copy2(Path(notebook_template).resolve(), notebook_path)
    (output / README_NAME).write_text(_readme(plan, rows), encoding="utf-8", newline="\n")

    validation = validate_phase4_delivery(output)
    files = [
        path for path in output.rglob("*")
        if path.is_file() and path.name != "delivery_manifest.json" and "_staging" not in path.parts
    ]
    body = {
        "schema": PHASE4_DELIVERY_SCHEMA,
        "delivery_id": PHASE4_DELIVERY_ID,
        "created_at": _timestamp(),
        "source_package_id": source_validation["package_id"],
        "source_commit": source_validation["git_commit"],
        "source_archive_sha256": source_validation["archive_sha256"],
        "source_content_manifest_hash": content_manifest_hash,
        "plan_hash": plan["plan_hash"],
        "warm_start": {
            "sha256": warm_sha,
            "iteration": warm_iteration,
            "architecture": warm_identity["architecture"],
            "history_dim": warm_identity["history_dim"],
            "feature_dim": warm_identity["feature_dim"],
            "parameter_count": warm_identity["parameter_count"],
            "state_dict_keys": warm_identity["state_dict_keys"],
        },
        "target_model": dict(plan["target_model"]),
        "semantic_contract_hash": semantic_contract_hash,
        "semantic_contract": semantic_contract,
        "native_backend_hash": native_backend_hash,
        "absolute_final_iteration": ABSOLUTE_FINAL_ITERATION,
        "max_wall_seconds": 12_600,
        "segments": rows,
        "uploads": uploads,
        "notebook": NOTEBOOK_NAME,
        "files": relative_file_manifest(output, files),
        "validation": validation,
        "uploaded": False,
        "remote_ready": True,
        "remote_verified": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    atomic_json(output / "delivery_manifest.json", manifest)
    shutil.rmtree(staging)
    return manifest


def validate_phase4_delivery(delivery_root: str | Path) -> dict[str, object]:
    root = Path(delivery_root).resolve()
    source_hashes = []
    rows = []
    for segment in SEGMENTS:
        session_root = root / "datasets" / str(segment["id"])
        manifest_path = session_root / DATASET_MANIFEST_NAME
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_dataset_manifest(payload, manifest_path=manifest_path)
        mode = str(segment["mode"])
        if payload.get("mode") != mode or payload["phase4"].get("role") != str(segment["id"]):
            raise ValueError(f"{segment['id']} dataset mode/role mismatch")
        profile = payload["profiles"][mode]
        if mode == "full":
            warm = profile.get("warm_start")
            if not isinstance(warm, dict) or warm.get("schema") != "guandan-marvel-warm-start-input-v1":
                raise ValueError("start dataset must declare the warm-start input")
            warm_path = session_root / WARM_START_NAME
            if not warm_path.is_file() or sha256_file(warm_path) != warm["sha256"]:
                raise ValueError("start dataset warm-start identity mismatch")
            # The packaged warm start must be the production-scale v1.3 policy
            # (never the phase-3 synthetic smoke artifact) and it must load.
            warm_payload = torch.load(warm_path, map_location="cpu", weights_only=False)
            warm_identity = validate_production_warm_start(warm_payload, source=str(warm_path))
            if int(warm.get("parameter_count", -1)) != int(warm_identity["parameter_count"]):
                raise ValueError("start dataset warm-start parameter count does not match the packaged file")
            if str(warm.get("architecture", "")) != str(warm_identity["architecture"]):
                raise ValueError("start dataset warm-start architecture does not match the packaged file")
        else:
            controls = profile.get("resume")
            if int(controls["target_iterations"]) != ABSOLUTE_FINAL_ITERATION:
                raise ValueError("resume dataset must target the absolute final iteration")
            if not bool(payload["recovery"].get("required")):
                raise ValueError("resume dataset must require a recovery input")
            # A resume profile must not carry training-semantic overrides: every
            # semantic field is inherited from the validated rank shard.
            forbidden = sorted(
                key for key in ("training", "model", "amp_enabled", "reward_semantics", "reward_scale")
                if key in profile
            )
            if forbidden:
                raise ValueError(
                    "resume profile must not override inherited training semantics: "
                    + ", ".join(forbidden)
                )
        if int(profile["ddp"]["world_size"]) != 2:
            raise ValueError(f"{segment['id']} must declare a two-GPU DDP profile")
        source_hashes.append(sha256_file(session_root / SOURCE_ARCHIVE_NAME))
        rows.append({"segment": segment["id"], "mode": mode, "manifest_hash": payload["phase4"]})
    if len(set(source_hashes)) != 1:
        raise ValueError("all segments must share one identical source archive")
    plan_path = root / "datasets" / "start" / PHASE4_PLAN_NAME
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_hash") != canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("phase4 plan hash mismatch")
    if plan["reward"]["semantics"] != "guandan_rank_points":
        raise ValueError("plan must record the rank-point reward semantics")
    if plan["training_distribution"]["context_coverage"]["context_count"] != 52:
        raise ValueError("plan must declare the 52-context training distribution")
    if plan["budget_stop_semantics"]["max_wall_seconds"] != 12_600:
        raise ValueError("plan must version the per-segment wall clock")
    # --- start profile wiring + real semantic contract -----------------------
    start_training = json.loads(
        (root / "datasets" / "start" / DATASET_MANIFEST_NAME).read_text(encoding="utf-8")
    )["profiles"]["full"]
    training = start_training["training"]
    if training.get("training_context_schedule") is None:
        raise ValueError("start profile must carry the 52-context schedule")
    if training.get("training_context_curriculum") is None:
        raise ValueError("start profile must carry the warm-start curriculum")
    if training.get("training_semantics_profile") != MACRO_DECISION_TRAINING_SEMANTICS_PROFILE:
        raise ValueError("start profile must declare the accepted macro semantics profile")
    if training.get("policy_regularizer_mode") != MACRO_NORMALIZED_POLICY_REGULARIZER_MODE:
        raise ValueError("start profile must declare the accepted policy regularizer mode")
    if training.get("reward_semantics") != REWARD_SEMANTICS or float(training.get("reward_scale")) != TRAINING_REWARD_SCALE:
        raise ValueError("start profile reward semantics/scale must match TrainConfig")
    # Phase 4A: the formal delivery must ship the accepted phase-3 architecture
    # under AMP, never the pre-phase-3 v1.3/AMP-off configuration.
    model_block = training.get("model")
    if not isinstance(model_block, Mapping) or str(model_block.get("architecture")) != TARGET_MODEL_ARCHITECTURE:
        raise ValueError(
            f"start profile model.architecture must be {TARGET_MODEL_ARCHITECTURE!r}; "
            f"got {None if not isinstance(model_block, Mapping) else model_block.get('architecture')!r}"
        )
    if bool(training.get("amp_enabled", False)) is not True:
        raise ValueError("start profile must enable AMP (amp_enabled=true)")
    if int(model_block.get("fuse_layers", 0)) <= 0:
        raise ValueError("start profile must declare a positive fuse_layers for the fused actor")
    expected_contract = build_start_semantic_contract(dict(training), run_root=root)
    if str(expected_contract["hash"]) != str(plan["semantic_contract_hash"]):
        raise ValueError("plan semantic contract hash does not match the real start config")
    plan_target = plan.get("target_model")
    if not isinstance(plan_target, Mapping) or str(plan_target.get("architecture")) != TARGET_MODEL_ARCHITECTURE:
        raise ValueError("plan must record the v1.4 target model architecture")
    if int(plan_target.get("parameter_count", -1)) != TARGET_MODEL_PARAMETER_COUNT:
        raise ValueError("plan target model parameter count must match the v1.4 architecture")
    if bool(plan_target.get("amp_enabled", False)) is not True:
        raise ValueError("plan target model must record amp_enabled=true")
    warm_plan = plan.get("warm_start")
    if not isinstance(warm_plan, Mapping) or str(warm_plan.get("architecture")) != PRODUCTION_WARM_START_ARCHITECTURE:
        raise ValueError("plan must record the production v1.3 warm-start architecture")
    if int(warm_plan.get("parameter_count", -1)) != PRODUCTION_WARM_START_PARAMETER_COUNT:
        raise ValueError("plan warm-start parameter count must be the production M09 size")
    if str(warm_plan.get("policy_sha256")) != str(start_training["warm_start"].get("sha256")):
        raise ValueError("plan warm-start sha256 must match the packaged start warm start")
    for row in rows:
        if str(row["manifest_hash"].get("semantic_contract_hash")) != str(plan["semantic_contract_hash"]):
            raise ValueError("dataset manifest semantic contract hash mismatch")
    # --- evaluation wiring ---------------------------------------------------
    evaluation = start_training["evaluation"]
    selection = evaluation["selection"]
    if int(selection["base_deals"]) != 52:
        raise ValueError("selection evaluation must use exactly 52 deals")
    if [int(v) for v in selection["level_ranks"]] != list(range(3, 16)):
        raise ValueError("selection evaluation must cover levels 3..15")
    if [int(v) for v in selection["leader_seats"]] != [0, 1, 2, 3]:
        raise ValueError("selection evaluation must cover all four leaders")
    if [int(v) for v in selection["seat_rotations"]] != [0, 1, 2, 3]:
        raise ValueError("selection evaluation must rotate all four seats")
    if str(selection.get("cluster_by")) != "deal_id":
        raise ValueError("selection evaluation must cluster by deal")
    if int(evaluation["fast_anchor"]["base_deals"]) != 8:
        raise ValueError("fast anchor must use exactly 8 deals")
    if bool(evaluation["final_holdout"]["generated"]) is not False:
        raise ValueError("final holdout must not be generated by the delivery")
    for name in ("start-upload.zip", "resume-upload.zip"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"delivery upload archive is missing: {name}")
    notebook = root / NOTEBOOK_NAME
    if not notebook.is_file():
        raise FileNotFoundError("delivery notebook is missing")
    return {
        "schema": PHASE4_VALIDATION_SCHEMA,
        "valid": True,
        "segments": rows,
        "source_archive_sha256": source_hashes[0],
        "plan_hash": plan["plan_hash"],
        "semantic_contract_hash": plan["semantic_contract_hash"],
        "native_backend_hash": plan["native_backend_hash"],
        "remote_verified": False,
    }


PHASE4_RUNTIME_PROBE_SCHEMA = "guandan-marvel-v14-phase4-runtime-probe-v1"
DEFAULT_LOCAL_PROBE_HORIZON = 512
# The formal Kaggle run allocates the delivered dense 4096-step trajectory at
# rollout=4/collect=20 across 2xT4.  A local CPU/single-GPU probe uses a
# smaller iteration budget, while keeping the production model architecture, the
# production training semantics (macro PPO/GAE, reward, 52-context curriculum)
# and the production warm start, and shrinks only the iteration *budget*.
RUNTIME_PROBE_ITERATION_BUDGET: Mapping[str, int] = {
    "rollout_batch_size": 1,
    "collect_batch_size": 1,
    "mini_batches": 1,
    "agent_epoch": 1,
    "max_sampler_bytes_per_rank": 4_294_967_296,
}


def _fixed_probe_rollout(game, *, batch: int = 2, steps: int = 3, seed: int = 20241004):
    """A shape-correct forward-only rollout for the *migration equivalence* probe."""
    from ..model import TensorDict

    num_tokens = int(getattr(game, "num_tokens"))
    players = int(getattr(game, "num_players"))
    channels = int(getattr(game, "num_channels"))
    obs_len = int(getattr(game, "max_observation_tokens"))
    action_len = int(getattr(game, "max_action_tokens"))
    actions = int(getattr(game, "max_available_actions"))
    generator = torch.Generator().manual_seed(int(seed))
    obs_tokens = torch.randint(1, num_tokens, (batch, steps, players, obs_len), generator=generator)
    obs_channels = torch.randint(
        0, 2, (batch, steps, players, channels), dtype=torch.uint8, generator=generator,
    )
    avail_tokens = torch.randint(1, num_tokens, (batch, steps, actions, action_len), generator=generator)
    avail_weights = torch.full((batch, steps, actions), 0.5, dtype=torch.float32)
    player_ids = torch.arange(steps).remainder(players).expand(batch, -1).clone()
    num_steps = torch.full((batch,), steps, dtype=torch.long)
    if batch > 1:
        num_steps[-1] = max(1, steps - 1)
        player_ids[-1, num_steps[-1]:] = -1
        avail_weights[-1, num_steps[-1]:, :] = 0.0
    return TensorDict(
        {
            "num_steps": num_steps,
            "player_ids": player_ids,
            "obs_tokens": obs_tokens,
            "obs_channels": obs_channels,
            "avail_tokens": avail_tokens,
            "avail_weights": avail_weights,
        },
        batch_size=[batch],
    )


def _capture_policy_outputs(model, rollout) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        captured["logits"] = output.detach().clone().squeeze(-1)

    handle = model.logit_head.register_forward_hook(_hook)
    try:
        # Migration equality is always same-device FP32, even if the caller
        # entered an ambient autocast context.  It is not a CPU-vs-CUDA test.
        with torch.no_grad(), torch.autocast(
            device_type=next(model.parameters()).device.type, enabled=False,
        ):
            output = model(rollout=rollout)
    finally:
        handle.remove()
    return {
        "logits": captured["logits"],
        "avail_probs": output.avail_probs.detach().clone(),
        "avail_log_probs": output.avail_log_probs.detach().clone(),
        "player_state_value": output.player_state_value.detach().clone(),
    }


def _build_source_v13_policy(payload: Mapping[str, object], *, device: str = "cpu"):
    """Reconstruct the *source* v1.3 policy from the warm-start payload itself."""
    from ..config import deserialize_train_config
    from ..model import Agent

    cfg = deserialize_train_config(payload.get("resolved_config"))
    if cfg is None:
        raise ValueError("warm-start resolved_config could not be decoded")
    cfg.device = str(device)
    model = Agent(
        cfg, enable_value_head=True, enable_ntp_head=Agent.ntp_enabled_for_config(cfg),
    ).to(device)
    model.load_state_dict(payload["agent_state_dict"], strict=True)
    model.eval()
    return model



def _probe_state_equal(actual, expected) -> bool:
    """Exact checkpoint value comparison, independent of tensor storage device."""
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
            and actual.dtype == expected.dtype and actual.shape == expected.shape
            and torch.equal(actual.detach().cpu(), expected.detach().cpu())
        )
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping) and isinstance(expected, Mapping)
            and actual.keys() == expected.keys()
            and all(_probe_state_equal(actual[key], expected[key]) for key in actual)
        )
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        return (
            type(actual) is type(expected) and len(actual) == len(expected)
            and all(_probe_state_equal(a, b) for a, b in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def resolve_phase4_probe_device(device: str = "auto") -> torch.device:
    """Resolve a local probe device; explicit CUDA never silently falls back."""
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid local probe device: {device!r}") from error
    if resolved.type == "cpu":
        if resolved.index is not None:
            raise ValueError("use 'cpu' without a device index")
        return resolved
    if resolved.type != "cuda":
        raise ValueError("local probe device must be auto, cpu, cuda or cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is allowed")
    index = 0 if resolved.index is None else resolved.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is not available")
    return torch.device("cuda", index)


def _build_local_probe_config(profile, *, game, work_dir, device: torch.device):
    # The copied probe profile is explicitly single-rank.  Never convert the
    # packaged two-T4 profile into a partially initialized local DDP job.
    cfg = build_train_config(
        profile, game=game, run_root=work_dir, local_cpu_test=device.type == "cpu",
    )
    cfg.device = str(device)
    if cfg.ddp.enabled:
        raise ValueError("local runtime probe must use exactly one rank")
    return cfg


def run_phase4_runtime_probe(
    profile_training: Mapping[str, object],
    *,
    warm_start_path: str | Path,
    game,
    work_dir: str | Path,
    iterations: int = 1,
    iteration_budget: Mapping[str, int] | None = None,
    ddp_world_size: int = 1,
    require_cuda: bool = False,
    device: str = "auto",
    cuda_init_scale: float = 1024.0,
) -> dict[str, object]:
    """Real start probe: profile -> TrainConfig -> IPPOTrainer -> warm start.

    The delivered 2xT4 NCCL profile is never modified.  This local probe uses
    one explicit CPU/CUDA device and a bounded batch; adaptations are recorded.
    OOM/CUDA errors propagate instead of producing a false CPU-backed GPU PASS.
    """
    from ..training.checkpoints import TrainingCheckpointStore
    from ..training.local import IPPOTrainer
    from .profiles import require_strict_resume_decision

    resolved_device = resolve_phase4_probe_device(device)
    if ddp_world_size != 1:
        raise ValueError("local runtime probe supports one rank only, not Kaggle DDP")
    if require_cuda and resolved_device.type != "cuda":
        raise ValueError("require_cuda conflicts with the selected CPU probe device")
    if int(iterations) < 1:
        raise ValueError("runtime probe iterations must be positive")
    if not math.isfinite(float(cuda_init_scale)) or float(cuda_init_scale) <= 0:
        raise ValueError("local CUDA initial scale must be finite and positive")
    probe_started = time.perf_counter()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(Path(warm_start_path), map_location="cpu", weights_only=False)
    source_identity = validate_production_warm_start(payload, source=str(warm_start_path))

    budget = dict(RUNTIME_PROBE_ITERATION_BUDGET if iteration_budget is None else iteration_budget)
    probe_training = dict(profile_training)
    budget_overrides: dict[str, list[int]] = {}
    for key, value in budget.items():
        delivered = probe_training.get(key)
        if delivered is not None and int(delivered) != int(value):
            budget_overrides[key] = [int(delivered), int(value)]
        probe_training[key] = int(value)
    if str(probe_training.get("model", {}).get("architecture")) != TARGET_MODEL_ARCHITECTURE:
        raise ValueError("runtime probe requires the delivered v1.4 start profile")

    profile = {
        "schema": "guandan-marvel-kaggle-profile-v1",
        "mode": "full",
        "algorithm": "ippo",
        "ddp": {
            "launcher": "torchrun", "world_size": int(ddp_world_size),
            "backend": "nccl" if resolved_device.type == "cuda" else "gloo",
            "require_cuda": resolved_device.type == "cuda",
        },
        "training": probe_training,
    }
    cfg = _build_local_probe_config(
        profile, game=game, work_dir=work_dir, device=resolved_device,
    )
    if str(cfg.model.architecture) != TARGET_MODEL_ARCHITECTURE:
        raise ValueError("runtime probe config did not build the v1.4-macro-fuse architecture")
    if bool(cfg.amp_enabled) is not True:
        raise ValueError("runtime probe config did not enable AMP")

    rollout = _fixed_probe_rollout(game).to(resolved_device)
    source_policy = _build_source_v13_policy(payload, device=str(resolved_device))
    source_outputs = _capture_policy_outputs(source_policy, rollout)
    del source_policy

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    trainer = IPPOTrainer(cfg, game=game, run_dir=work_dir / "probe-run")
    if resolved_device.type == "cuda":
        # This is a one-update *local smoke*, not a fresh formal training run.
        # Record the smaller initial scaler state to avoid spending the bounded
        # single iteration solely backing off 65536.  Packaged defaults and the
        # production trainer/scaler implementation remain untouched.
        trainer.grad_scaler = torch.amp.GradScaler("cuda", init_scale=float(cuda_init_scale))
    try:
        parameter_count = int(sum(p.numel() for p in trainer.agent_resume.parameters()))
        state_dict_keys = len(trainer.agent_resume.state_dict())
        if parameter_count != TARGET_MODEL_PARAMETER_COUNT:
            raise ValueError(
                f"target model parameter count must be {TARGET_MODEL_PARAMETER_COUNT}; got {parameter_count}"
            )
        if state_dict_keys != TARGET_MODEL_STATE_DICT_KEYS:
            raise ValueError(
                f"target model state_dict must carry {TARGET_MODEL_STATE_DICT_KEYS} keys; got {state_dict_keys}"
            )
        fuse = getattr(trainer.agent_resume, "actor_fuse", None)
        if fuse is None:
            raise ValueError("target model did not build the actor_fuse branch")
        if int(torch.count_nonzero(fuse.output_proj.weight)) != 0:
            raise ValueError("target actor_fuse.output_proj must start bitwise zero")

        report = trainer.load_warm_start_policy(payload)
        migration = report["migration"]
        expected_migration = {
            "source_architecture": PRODUCTION_WARM_START_ARCHITECTURE,
            "target_architecture": TARGET_MODEL_ARCHITECTURE,
            "fuse_migrated": True,
            "output_proj_zero_verified": True,
        }
        for key, value in expected_migration.items():
            if migration.get(key) != value:
                raise ValueError(f"warm-start migration {key} must be {value!r}; got {migration.get(key)!r}")
        for key in ("optimizer_reset", "sampler_reset", "counters_reset"):
            if report.get(key) is not True:
                raise ValueError(f"warm-start load must reset {key}")
        for module in (trainer.agent_resume, trainer.agent_merged):
            if int(torch.count_nonzero(module.actor_fuse.output_proj.weight)) != 0:
                raise ValueError("migrated actor_fuse.output_proj is not bitwise zero")

        trainer.agent_merged.eval()
        target_outputs = _capture_policy_outputs(trainer.agent_merged, rollout)
        bitwise = {
            "logits": bool(torch.equal(source_outputs["logits"], target_outputs["logits"])),
            "avail_probs": bool(torch.equal(source_outputs["avail_probs"], target_outputs["avail_probs"])),
            "avail_log_probs": bool(
                torch.equal(source_outputs["avail_log_probs"], target_outputs["avail_log_probs"])
            ),
            "player_state_value": bool(
                torch.equal(source_outputs["player_state_value"], target_outputs["player_state_value"])
            ),
            "argmax": bool(
                torch.equal(
                    torch.argmax(source_outputs["avail_probs"], dim=-1),
                    torch.argmax(target_outputs["avail_probs"], dim=-1),
                )
            ),
        }
        if not all(bitwise.values()):
            raise ValueError(f"v1.3->v1.4 migration changed the pre-step policy: {bitwise}")

        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        before_update = trainer.agent_resume.logit_head.weight.detach().clone()
        training_started = time.perf_counter()
        series = [trainer.train_iteration() for _ in range(int(iterations))]
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        training_seconds = time.perf_counter() - training_started
        final_metrics = series[-1]
        parameter_device = str(next(trainer.agent_resume.parameters()).device)
        if torch.device(parameter_device) != resolved_device:
            raise ValueError("runtime probe parameters are not on the requested device")
        optimizer_updates = int(trainer.learner_updates)
        parameter_changed = not torch.equal(before_update, trainer.agent_resume.logit_head.weight)
        del before_update
        if optimizer_updates < 1 or not parameter_changed:
            raise ValueError("runtime probe made no effective optimizer update (possibly all AMP steps skipped)")
        loss_metrics = {
            key: final_metrics.get(key)
            for key in ("agent/grad_norm", "macro_initial/pg_loss")
        }
        for key, value in loss_metrics.items():
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"runtime probe metric {key} must be finite; got {value!r}")
        amp_metrics = {
            "enabled": bool(final_metrics.get("runtime/amp_enabled")),
            "dtype": final_metrics.get("runtime/amp_autocast_dtype"),
            "scaler_enabled": bool(final_metrics.get("runtime/amp_scaler_enabled")),
            "scaler_scale": float(trainer.grad_scaler.get_scale()),
            "skipped_steps": int(trainer.amp_skipped_steps),
        }
        # CPU BF16 and CUDA FP16 are separate acceptance paths.
        if trainer.amp_device_type == "cpu":
            if trainer.amp_dtype is not torch.bfloat16:
                raise ValueError("CPU AMP probe must use bfloat16 autocast")
            if amp_metrics["dtype"] != "torch.bfloat16":
                raise ValueError("CPU AMP probe metric did not report bfloat16 autocast")
        if resolved_device.type == "cuda":
            if trainer.amp_dtype is not torch.float16 or amp_metrics["dtype"] != "torch.float16":
                raise ValueError("CUDA AMP probe must use float16 autocast")
            if not amp_metrics["scaler_enabled"]:
                raise ValueError("CUDA AMP probe must enable GradScaler")
        memory_metrics = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(resolved_device) if resolved_device.type == "cuda" else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(resolved_device) if resolved_device.type == "cuda" else 0,
        }
        checkpoint_path = trainer.save_checkpoint(work_dir / "probe-checkpoint.pt")
    finally:
        trainer.close()

    reloaded = TrainingCheckpointStore.load(checkpoint_path, device="cpu")
    if str(reloaded.get("architecture")) != TARGET_MODEL_ARCHITECTURE:
        raise ValueError("probe checkpoint did not record the v1.4 architecture")
    if bool(reloaded["amp"]["enabled"]) is not True:
        raise ValueError("probe checkpoint did not record amp_enabled=true")
    if str(reloaded["resolved_config"]["model"]["architecture"]) != TARGET_MODEL_ARCHITECTURE:
        raise ValueError("probe checkpoint resolved_config is not the v1.4 architecture")
    if bool(
        reloaded["semantic_contract"]["semantic_config"]["execution_semantics"]["amp_enabled"]
    ) is not True:
        raise ValueError("probe checkpoint semantic contract did not record amp_enabled=true")

    # Release the first GPU model/buffers before allocating the resume trainer.
    del trainer, rollout
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    resume_cfg = _build_local_probe_config(
        profile, game=game, work_dir=work_dir, device=resolved_device,
    )
    resumed = IPPOTrainer(resume_cfg, game=game, run_dir=work_dir / "probe-resume")
    try:
        decision = require_strict_resume_decision(resumed.load_checkpoint_payload(reloaded))
        if resumed.agent_resume.actor_fuse is None:
            raise ValueError("resumed trainer lost the fused v1.4 actor")
        reload_amp = bool(resumed.amp_enabled)
        reload_device = str(next(resumed.agent_resume.parameters()).device)
        restored_scaler = resumed.grad_scaler.state_dict() == reloaded["grad_scaler_state"]
        restored_models = all(
            _probe_state_equal(model.state_dict(), reloaded[key])
            for model, key in (
                (resumed.agent_resume, "agent_resume_state_dict"),
                (resumed.agent_merged, "agent_state_dict"),
            )
        )
        restored_optimizer = _probe_state_equal(
            resumed.agent_optimizer.state_dict(), reloaded["agent_optimizer_state_dict"],
        )
        restored_counters = (
            resumed.iteration == int(reloaded["iteration"]) == int(iterations)
            and resumed.learner_updates == optimizer_updates == int(reloaded["learner_updates"])
            and resumed.amp_skipped_steps == int(reloaded["amp"]["skipped_steps"])
        )
        if not all((restored_scaler, restored_models, restored_optimizer, restored_counters)):
            raise ValueError("resume did not restore model/optimizer/counters/GradScaler state exactly")
        if torch.device(reload_device) != resolved_device:
            raise ValueError("resume did not restore the selected device")
    finally:
        resumed.close()
    del resumed
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "schema": PHASE4_RUNTIME_PROBE_SCHEMA,
        "valid": True,
        "target_architecture": TARGET_MODEL_ARCHITECTURE,
        "amp_enabled": bool(cfg.amp_enabled),
        "target_parameter_count": parameter_count,
        "target_state_dict_keys": state_dict_keys,
        "warm_start": source_identity,
        "migration": dict(migration),
        "output_proj_zero": True,
        "bitwise_policy_equivalence": bitwise,
        "iterations": int(len(series)),
        "optimizer_updates": optimizer_updates,
        "parameter_changed": parameter_changed,
        "loss_metrics": loss_metrics,
        "amp_metrics": amp_metrics,
        "device": {
            "requested": str(device), "selected": str(resolved_device),
            "parameter_device": parameter_device, "python_executable": sys.executable,
            "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else None,
        },
        "timing": {
            "training_wall_seconds": training_seconds,
            "rollout_seconds": sum(float(row["perf/rollout_seconds"]) for row in series),
            "learner_seconds": sum(float(row["perf/learner_seconds"]) for row in series),
            "total_wall_seconds": time.perf_counter() - probe_started,
        },
        "memory": memory_metrics,
        "checkpoint": {
            "path": str(checkpoint_path),
            "architecture": str(reloaded.get("architecture")),
            "amp_enabled": bool(reloaded["amp"]["enabled"]),
            "resume_kind": decision.kind,
            "reload_amp_enabled": reload_amp,
            "reload_parameter_device": reload_device,
            "grad_scaler_state_restored": restored_scaler,
            "model_states_restored": restored_models,
            "optimizer_state_restored": restored_optimizer,
            "counters_restored": restored_counters,
            "resume_reasons": list(decision.reasons),
        },
        "runtime_adaptation": {
            "delivered_ddp": {"world_size": 2, "backend": "nccl", "require_cuda": True},
            "local_probe_ddp": {"world_size": 1, "enabled": False, "backend": None, "require_cuda": resolved_device.type == "cuda"},
            "iteration_budget_overrides": {key: list(value) for key, value in budget_overrides.items()},
            "probe_max_game_horizon": int(cfg.game.max_game_horizon),
            "fp32_equivalence_device": str(resolved_device),
            "equivalence_dtype": "torch.float32",
            "local_cuda_init_scale": float(cuda_init_scale) if resolved_device.type == "cuda" else None,
            "production_default_init_scale_unchanged": True,
            "kaggle_dual_t4_verified": False,
            "reason": "bounded single-device smoke; not a two-T4/NCCL or full-budget acceptance",
        },
    }


def validate_phase4_delivery_runtime(
    delivery_root: str | Path,
    *,
    work_dir: str | Path | None = None,
    game=None,
    iterations: int = 1,
    iteration_budget: Mapping[str, int] | None = None,
    device: str = "auto",
    max_game_horizon: int = DEFAULT_LOCAL_PROBE_HORIZON,
    cuda_init_scale: float = 1024.0,
) -> dict[str, object]:
    """Fail-closed *runtime* gate: prove the delivery can actually start training."""
    from ..training.checkpoints import TrainingCheckpointStore
    from .profiles import build_resume_train_config

    root = Path(delivery_root).resolve()
    structural = validate_phase4_delivery(root)
    start_root = root / "datasets" / "start"
    manifest = json.loads((start_root / DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
    profile = manifest["profiles"]["full"]
    training = profile["training"]
    ddp = profile["ddp"]
    if int(ddp.get("world_size", 0)) != 2 or str(ddp.get("backend")) != "nccl" or not bool(ddp.get("require_cuda")):
        raise ValueError("start profile must declare a 2-rank NCCL CUDA DDP profile")
    if isinstance(training.get("model"), Mapping) and str(training["model"].get("architecture")) != TARGET_MODEL_ARCHITECTURE:
        raise ValueError("start profile must target the v1.4-macro-fuse architecture")
    if bool(training.get("amp_enabled", False)) is not True:
        raise ValueError("start profile must enable AMP")
    warm_path = start_root / WARM_START_NAME
    if not warm_path.is_file():
        raise FileNotFoundError(f"delivery warm start is missing: {warm_path}")
    # Keep the probe scratch space *outside* the delivery package so the
    # verifier never pollutes the packaged dataset with checkpoints.
    probe_dir = (
        Path(work_dir).resolve() if work_dir is not None
        else root.parent / f"{root.name}_runtime_probe"
    )
    if probe_dir.is_relative_to(root):
        raise ValueError("runtime work directory must be outside the delivery package")
    if game is None:
        from ..training.smoke import CppSmokeGame

        game = CppSmokeGame(max_game_horizon=int(max_game_horizon))
    probe = run_phase4_runtime_probe(
        training, warm_start_path=warm_path, game=game, work_dir=probe_dir,
        iterations=iterations, iteration_budget=iteration_budget, device=device,
        cuda_init_scale=cuda_init_scale,
    )

    # resume must strictly inherit the v1.4/AMP semantics from a real durable
    # checkpoint, and the resume profile must not override any of them.
    resume_manifest = json.loads(
        (root / "datasets" / "resume" / DATASET_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    resume_profile = resume_manifest["profiles"]["resume"]
    inherited = build_resume_train_config(
        resume_profile,
        checkpoint_payload=TrainingCheckpointStore.load(probe["checkpoint"]["path"], device="cpu"),
        game=game, run_root=probe_dir,
    )
    if str(inherited.model.architecture) != TARGET_MODEL_ARCHITECTURE:
        raise ValueError("resume did not inherit the v1.4 architecture from the checkpoint")
    if bool(inherited.amp_enabled) is not True:
        raise ValueError("resume did not inherit amp_enabled=true from the checkpoint")

    return {
        "schema": PHASE4_RUNTIME_VALIDATION_SCHEMA,
        "valid": True,
        "structural": structural,
        "runtime_probe": probe,
        "resume_inheritance": {
            "architecture": str(inherited.model.architecture),
            "amp_enabled": bool(inherited.amp_enabled),
        },
    }


def _readme(plan: dict[str, object], rows: list[dict[str, object]]) -> str:
    schedule = "\n".join(
        f"- **{row['segment']}**（{row['mode']}）：运行到绝对最终 iteration {row['absolute_final_iteration']}"
        for row in rows
    )
    return f"""# MARVEL v14 阶段 4 Kaggle 分段交付

两个 Kaggle Dataset 共用同一份 source package（sha256 见
`delivery_manifest.json.source_archive_sha256`）。`start` 只在首个 session 出现。

## 顺序

{schedule}

- **start**：导入 `{NOTEBOOK_NAME}`，GPU T4 x2，Run All。Output 下载
  `guandan-marvel-output.zip` 与 `guandan-marvel-recovery.zip`。
- **resume**：把上一步的 `guandan-marvel-recovery.zip` 作为第二个 Kaggle Input，
  再加本 segment 的 Dataset，同一 Notebook Run All。**可重复**：只要未到
  iteration {ABSOLUTE_FINAL_ITERATION}，就继续用最新 recovery 重复本步骤。

## 模型与精度

- 目标模型架构：`{plan['target_model']['architecture']}`（阶段 3 已验收的 Actor Residual Fuse），
  参数量 {plan['target_model']['parameter_count']}（state_dict {plan['target_model']['state_dict_keys']} keys）。
- 混合精度 AMP = {plan['target_model']['amp_enabled']}（CUDA 验证/训练用 fp16 autocast + GradScaler；显式 CPU 验证用 bf16）。
- 本地验证支持 `--device auto|cpu|cuda`。明确请求 CUDA 但不可用时直接失败，不回落 CPU。
- 本地 runtime probe 使用单卡 batch=1、1 iteration 和 512-step horizon，CUDA smoke 初始 scale=1024；
  这些适配全部写入报告，不修改正式 T4 x2 / 4096-step profile 或生产 GradScaler 默认值。
- 本机 smoke checkpoint 只验证同设备恢复，不能作为正式双 T4 训练的 recovery 输入，也不证明双卡 NCCL 通过。
- warm start：`{WARM_START_NAME}`，来源 policy `{plan['warm_start']['architecture']}`
  （feature_dim={plan['warm_start']['feature_dim']}, history_dim={plan['warm_start']['history_dim']},
  params={plan['warm_start']['parameter_count']}, iteration={plan['warm_start']['iteration']}），
  仅加载 policy 权重，optimizer/sampler/counters 重置。

## 预算与停止

- 每段 `max_wall_seconds = {plan['budget_stop_semantics']['max_wall_seconds']}`（约 3.5 小时）。
- 到点后在**迭代边界**保存 checkpoint、原子构建 recovery、复制 output，并正常返回
  `budget_stopped`，不依赖 Kaggle 平台强杀自动保存。
- 训练交给完成局数的精确预算：target = collect(20) × {ABSOLUTE_FINAL_ITERATION} = {plan['target_completed_games']} 局。

## 奖励与上下文

- 奖励语义：`{plan['reward']['semantics']}`，raw 范围 {plan['reward']['raw_range']}，
  训练 scale = {plan['reward']['training_scale']}（R_train = R_raw / 3）。
- 训练分布：13 等级 × 4 leader 共 52 个无进贡 context（7A），均匀初始权重；
  课程 warmup={CURRICULUM['warmup_iterations']} / blend={CURRICULUM['blend_iterations']}，
  由 iteration 决定、可精确 resume。
- 7B（贡还 previous_result）运行时支持已启用：完整观测审计（event tokens + channels）未发现任何
  合法碰撞，且无隐藏牌泄漏；本交付的训练分布仍为 7A（不含贡还 context），不声明 7B 覆盖率。

## 评估隔离

- fast anchor：仅健康/趋势，不作强度声明；
- selection：13 等级 × 4 leader × 4 座位轮换、按 deal 聚类；
- final holdout：独立 seed/hash，候选冻结后一次性运行，本交付未生成。

计划 hash：`{plan['plan_hash']}`；语义契约 hash：`{plan['semantic_contract_hash']}`；
native backend hash：`{plan['native_backend_hash']}`。修改任何训练语义字段都必须重建交付。
"""


__all__ = [
    "ABSOLUTE_FINAL_ITERATION", "CURRICULUM", "NOTEBOOK_NAME", "PHASE4_DELIVERY_ID",
    "PHASE4_DELIVERY_SCHEMA", "PHASE4_PLAN_NAME", "PHASE4_PLAN_SCHEMA", "PHASE4_RUNTIME_PROBE_SCHEMA",
    "PHASE4_RUNTIME_VALIDATION_SCHEMA", "PHASE4_VALIDATION_SCHEMA", "PHASE4_WARM_START_IDENTITY_SCHEMA",
    "PRODUCTION_WARM_START_ARCHITECTURE", "PRODUCTION_WARM_START_FEATURE_DIM",
    "PRODUCTION_WARM_START_HISTORY_DIM", "PRODUCTION_WARM_START_PARAMETER_COUNT",
    "PRODUCTION_WARM_START_STATE_DICT_KEYS", "RUNTIME_PROBE_ITERATION_BUDGET", "SEGMENTS",
    "TARGET_MODEL_ARCHITECTURE", "TARGET_MODEL_PARAMETER_COUNT", "TARGET_MODEL_STATE_DICT_KEYS",
    "DEFAULT_LOCAL_PROBE_HORIZON", "resolve_phase4_probe_device",
    "WARM_START_NAME", "build_phase4_delivery", "build_phase4_plan", "run_phase4_runtime_probe",
    "validate_phase4_delivery", "validate_phase4_delivery_runtime", "validate_production_warm_start",
]
