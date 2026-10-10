# SPDX-License-Identifier: MIT
"""Versioned two-rank durable checkpoints and downloadable recovery bundles."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping
import zipfile

import numpy as np
import torch

from guandan_marvel.evaluation.checkpoints import freeze_merged_policy
from guandan_marvel.runtime.distributed import DistributedContext
from guandan_marvel.training.checkpoints import (
    TRAINING_CHECKPOINT_SCHEMA, TrainingCheckpointStore, canonical_json_hash,
    validate_declared_hash, validate_source_fingerprint,
)

from .common import atomic_json, canonical_hash, sha256_file

DDP_CHECKPOINT_SCHEMA = "guandan-marvel-ddp-checkpoint-v2"
DDP_PROVENANCE_SCHEMA = "guandan-marvel-ddp-training-data-provenance-v1"
RECOVERY_MANIFEST_SCHEMA = "guandan-marvel-recovery-manifest-v2"
CHECKPOINT_MANIFEST_SCHEMA = "guandan-marvel-checkpoint-manifest-v2"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_dict_hash(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()



def _update_object_digest(digest: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            _update_object_digest(digest, str(key))
            _update_object_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _update_object_digest(digest, item)
    elif isinstance(value, bytes):
        digest.update(b"bytes")
        digest.update(value)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
    else:
        digest.update(type(value).__qualname__.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))


def continuation_state_hash(payload: Mapping[str, object]) -> str:
    """Hash state that determines the next training update, excluding run paths."""
    keys = (
        "agent_resume_state_dict", "agent_state_dict", "critic_state_dict",
        "agent_optimizer_state_dict", "critic_optimizer_state_dict",
        "replay_buffer_state", "sampler_state", "rng_state", "schedule_state",
        "merge_state", "iteration", "durable_updates", "learner_updates",
        "durable_learner_updates", "counters",
    )
    digest = hashlib.sha256()
    for key in keys:
        if key in payload:
            _update_object_digest(digest, key)
            _update_object_digest(digest, payload[key])
    return digest.hexdigest()

def _rank_summary(payload: Mapping[str, object], *, rank: int, shard: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema": "guandan-marvel-ddp-rank-checkpoint-summary-v2",
        "rank": int(rank),
        "shard": shard.name,
        "shard_sha256": sha256_file(shard),
        "shard_bytes": shard.stat().st_size,
        "checkpoint_schema": payload.get("schema"),
        "algorithm": payload.get("algorithm"),
        "iteration": int(payload.get("iteration", -1)),
        "durable_updates": int(payload.get("durable_updates", -1)),
        "source_hash": payload.get("source_hash"),
        "source_commit": payload.get("source_parent_commit"),
        "semantic_contract_hash": (payload.get("semantic_contract") or {}).get("hash"),
        "resolved_config_hash": payload.get("resolved_config_hash"),
        "backend_identity_hash": (payload.get("backend_identity") or {}).get("hash"),
        "agent_resume_state_hash": _state_dict_hash(payload["agent_resume_state_dict"]),
        "agent_merged_state_hash": _state_dict_hash(payload["agent_state_dict"]),
        "training_data_provenance": deepcopy(payload.get("training_data_provenance")),
        "counters": deepcopy(payload.get("counters")),
        "continuation_state_hash": continuation_state_hash(payload),
    }
    if payload.get("critic_state_dict") is not None:
        summary["critic_state_hash"] = _state_dict_hash(payload["critic_state_dict"])
    return summary


def aggregate_training_provenance(rank_summaries: list[Mapping[str, object]]) -> dict[str, object]:
    provenances = [row.get("training_data_provenance") for row in rank_summaries]
    if not all(isinstance(item, Mapping) for item in provenances):
        raise ValueError("all DDP rank checkpoints require training_data_provenance")
    first = provenances[0]
    for item in provenances[1:]:
        for key in ("domain", "manifest_hash", "generator_version", "seed_semantics"):
            if item.get(key) != first.get(key):
                raise ValueError(f"DDP rank training provenance differs for {key}")
    seeds = sorted({int(item["initial_env_seed"]) for item in provenances if item.get("initial_env_seed") is not None})
    hashes = sorted({str(value) for item in provenances for value in item.get("deal_hashes", [])})
    payload = {
        "schema": DDP_PROVENANCE_SCHEMA,
        "domain": first.get("domain"),
        "manifest_hash": first.get("manifest_hash"),
        "generator_version": first.get("generator_version"),
        "seed_semantics": first.get("seed_semantics"),
        "initial_env_seeds": seeds,
        "deal_hashes": hashes,
        "deal_count": len(hashes),
        "complete": all(bool(item.get("complete", False)) for item in provenances),
        "world_size": len(rank_summaries),
        "rank_provenance_hashes": [item.get("hash") for item in provenances],
    }
    return {**payload, "hash": canonical_hash(payload)}


def _validate_rank_summaries(rows: list[Mapping[str, object]], *, world_size: int) -> None:
    if len(rows) != world_size or sorted(int(row.get("rank", -1)) for row in rows) != list(range(world_size)):
        raise ValueError("DDP checkpoint rank coverage is incomplete")
    shared = (
        "checkpoint_schema", "algorithm", "iteration", "durable_updates", "source_hash",
        "source_commit", "semantic_contract_hash", "backend_identity_hash",
        "agent_resume_state_hash", "agent_merged_state_hash", "critic_state_hash",
    )
    first = rows[0]
    for row in rows[1:]:
        for key in shared:
            if row.get(key) != first.get(key):
                raise ValueError(f"DDP rank checkpoint mismatch for {key}")


def save_distributed_checkpoint(
    trainer: object,
    destination: str | Path,
    *,
    context: DistributedContext,
) -> Path | None:
    """Save one exact-resume shard per rank and atomically publish a manifest directory."""
    destination = Path(destination).resolve()
    temporary = destination.with_name(destination.name + ".tmp")
    if context.is_primary:
        if destination.exists():
            raise FileExistsError(destination)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
    context.barrier()
    payload = trainer.checkpoint_payload()
    shard = temporary / f"rank{context.rank:03d}.pt"
    TrainingCheckpointStore.save(payload, shard)
    summary = _rank_summary(payload, rank=context.rank, shard=shard)
    atomic_json(temporary / f"rank{context.rank:03d}.summary.json", summary)
    rows = [dict(row) for row in context.all_gather_object(summary)]
    _validate_rank_summaries(rows, world_size=context.world_size)
    publish_error = None
    if context.is_primary:
        try:
            provenance = aggregate_training_provenance(rows)
            global_counters = {
                key: sum(int((row.get("counters") or {}).get(key, 0)) for row in rows)
                for key in ("total_env_steps", "total_game_steps", "completed_games_total")
            }
            body = {
                "schema": DDP_CHECKPOINT_SCHEMA,
                "created_at": _timestamp(),
                "world_size": context.world_size,
                "resume_world_size_policy": "exact_same_world_size_only",
                "rank_shards": rows,
                "algorithm": rows[0]["algorithm"],
                "iteration": rows[0]["iteration"],
                "durable_updates": rows[0]["durable_updates"],
                "source_commit": rows[0]["source_commit"],
                "source_hash": rows[0]["source_hash"],
                "semantic_contract_hash": rows[0]["semantic_contract_hash"],
                "rank_resolved_config_hashes": [row["resolved_config_hash"] for row in rows],
                "backend_identity_hash": rows[0]["backend_identity_hash"],
                "global_counters": global_counters,
                "training_data_provenance": provenance,
                "complete": True,
            }
            manifest = {**body, "manifest_hash": canonical_hash(body)}
            atomic_json(temporary / "manifest.json", manifest)
            os.replace(temporary, destination)
        except BaseException as error:
            publish_error = f"{type(error).__name__}: {error}"
    errors = context.all_gather_object(publish_error)
    if errors[0] is not None:
        raise RuntimeError(f"rank 0 failed to publish DDP checkpoint: {errors[0]}")
    return destination if context.is_primary else None


def load_ddp_manifest(path: str | Path, *, verify_shards: bool = True) -> dict[str, object]:
    root = Path(path).resolve()
    manifest_path = root / "manifest.json" if root.is_dir() else root
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != DDP_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported DDP checkpoint manifest")
    declared = payload.get("manifest_hash")
    actual = canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    if declared != actual:
        raise ValueError("DDP checkpoint manifest hash mismatch")
    rows = payload.get("rank_shards")
    if not isinstance(rows, list):
        raise ValueError("DDP checkpoint rank_shards is malformed")
    _validate_rank_summaries(rows, world_size=int(payload.get("world_size", 0)))
    if verify_shards:
        for row in rows:
            shard = manifest_path.parent / str(row["shard"])
            if not shard.is_file() or sha256_file(shard) != row["shard_sha256"]:
                raise ValueError(f"DDP checkpoint shard missing or corrupt: {shard}")
    return payload



def load_rank_checkpoint_for_resume(
    checkpoint: str | Path,
    *,
    rank: int,
    expected_world_size: int,
    expected_source_commit: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the envelope and one rank shard before trainer construction."""
    root = Path(checkpoint).resolve()
    manifest = load_ddp_manifest(root, verify_shards=True)
    if int(manifest["world_size"]) != int(expected_world_size):
        raise ValueError(
            "recovery world size is incompatible: "
            f"checkpoint={manifest['world_size']}, requested={expected_world_size}"
        )
    if expected_source_commit is not None and manifest.get("source_commit") != expected_source_commit:
        raise ValueError(
            "recovery source commit is incompatible with this delivery: "
            f"checkpoint={manifest.get('source_commit')}, delivery={expected_source_commit}; "
            "use the Dataset/source package that created the recovery ZIP"
        )
    rows = {int(row["rank"]): row for row in manifest["rank_shards"]}
    if int(rank) not in rows:
        raise ValueError(f"recovery does not contain rank {rank}")
    row = rows[int(rank)]
    shard = root / str(row["shard"])
    payload = dict(TrainingCheckpointStore.load(shard, device="cpu"))
    if payload.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported rank training checkpoint schema {payload.get('schema')!r}")
    if payload.get("source_parent_commit") != manifest.get("source_commit"):
        raise ValueError("rank shard source commit differs from DDP manifest")
    if payload.get("source_hash") != manifest.get("source_hash"):
        raise ValueError("rank shard source hash differs from DDP manifest")
    if payload.get("resolved_config_hash") != canonical_json_hash(payload.get("resolved_config")):
        raise ValueError("rank shard resolved config hash is inconsistent")
    for name in ("semantic_contract", "backend_identity", "training_data_provenance"):
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"rank shard {name} is missing")
        validate_declared_hash(value, name=f"rank shard {name}")
    source_fingerprint = payload.get("source_fingerprint")
    if not isinstance(source_fingerprint, Mapping):
        raise ValueError("rank shard source_fingerprint is missing")
    validate_source_fingerprint(source_fingerprint, name="rank shard source_fingerprint")
    recomputed = _rank_summary(payload, rank=int(rank), shard=shard)
    for key in (
        "shard_sha256", "shard_bytes", "checkpoint_schema", "algorithm", "iteration",
        "durable_updates", "source_hash", "source_commit", "semantic_contract_hash",
        "resolved_config_hash", "backend_identity_hash", "agent_resume_state_hash",
        "agent_merged_state_hash", "critic_state_hash", "continuation_state_hash",
    ):
        if row.get(key) != recomputed.get(key):
            raise ValueError(f"rank shard identity mismatch for {key}")
    return payload, manifest

def load_distributed_checkpoint(
    trainer: object,
    checkpoint: str | Path,
    *,
    context: DistributedContext,
    expected_source_commit: str | None = None,
):
    payload, manifest = load_rank_checkpoint_for_resume(
        checkpoint, rank=context.rank, expected_world_size=context.world_size,
        expected_source_commit=expected_source_commit,
    )
    loader = getattr(trainer, "load_checkpoint_payload", None)
    if not callable(loader):
        raise TypeError("trainer does not support payload-based resume")
    decision = loader(payload)
    context.barrier()
    return decision, manifest


def freeze_distributed_policy(
    trainer: object,
    path: str | Path,
    *,
    ddp_manifest: Mapping[str, object],
) -> Path:
    destination = freeze_merged_policy(trainer, path)
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    payload["training_data_provenance"] = deepcopy(ddp_manifest["training_data_provenance"])
    payload["distributed_training"] = {
        "schema": DDP_CHECKPOINT_SCHEMA,
        "world_size": ddp_manifest["world_size"],
        "checkpoint_manifest_hash": ddp_manifest["manifest_hash"],
        "global_counters": deepcopy(ddp_manifest["global_counters"]),
    }
    TrainingCheckpointStore.save  # retain import-time validation visibility
    temporary = destination.with_suffix(destination.suffix + ".tmp-ddp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def update_checkpoint_index(
    output_root: str | Path,
    *,
    checkpoint_dir: Path,
    ddp_manifest: Mapping[str, object],
    evaluation_checkpoint: Path | None = None,
    evaluation_status: str = "pending",
    evaluation_error: str | None = None,
) -> Path:
    root = Path(output_root).resolve()
    path = root / "checkpoint_manifest.json"
    previous: dict[str, object] = {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
    history = list(previous.get("history", []))
    entry = {
        "iteration": int(ddp_manifest["iteration"]),
        "checkpoint": checkpoint_dir.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(checkpoint_dir / "manifest.json"),
        "manifest_hash": ddp_manifest["manifest_hash"],
        "evaluation_checkpoint": None if evaluation_checkpoint is None else evaluation_checkpoint.relative_to(root).as_posix(),
        "evaluation_checkpoint_sha256": None if evaluation_checkpoint is None else sha256_file(evaluation_checkpoint),
        "evaluation_status": str(evaluation_status),
        "evaluation_error": evaluation_error,
        "durable": True,
        "created_at": ddp_manifest["created_at"],
    }
    history = [row for row in history if int(row.get("iteration", -1)) != entry["iteration"]] + [entry]
    payload = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "latest": entry,
        "history": sorted(history, key=lambda row: int(row["iteration"])),
        "incomplete_checkpoint_count": 0,
    }
    return atomic_json(path, payload)


def build_recovery_bundle(
    *,
    run_root: str | Path,
    destination: str | Path,
    source_commit: str,
    source_archive_sha256: str,
    package_id: str,
    status: str,
    native_extension: str | Path | None,
) -> Path | None:
    root = Path(run_root).resolve()
    checkpoint_index_path = root / "checkpoint_manifest.json"
    if not checkpoint_index_path.is_file():
        return None
    checkpoint_index = json.loads(checkpoint_index_path.read_text(encoding="utf-8"))
    latest = checkpoint_index.get("latest")
    if not isinstance(latest, Mapping) or bool(latest.get("input_only", False)):
        return None
    checkpoint_root = (root / str(latest["checkpoint"])).resolve()
    if not checkpoint_root.is_dir() or not checkpoint_root.is_relative_to(root):
        raise ValueError("latest checkpoint path is invalid")
    include: list[tuple[Path, str]] = []
    for path in sorted(checkpoint_root.rglob("*")):
        if path.is_file():
            include.append((path, f"checkpoint/{path.relative_to(checkpoint_root).as_posix()}"))
    for name in (
        "checkpoint_manifest.json", "champion_manifest.json", "evaluation_summary.json",
        "resolved_config.json", "runtime_environment.json", "source_manifest.json", "progress.json",
    ):
        path = root / name
        if path.is_file():
            include.append((path, f"artifacts/{name}"))
    runtime_entry = None
    if native_extension is not None:
        extension = Path(native_extension).resolve()
        if extension.is_file():
            runtime_entry = {
                "archive_path": f"runtime/{extension.name}",
                "restore_relative_path": f"nanopokerlib/{extension.name}",
                "sha256": sha256_file(extension),
                "bytes": extension.stat().st_size,
            }
            include.append((extension, runtime_entry["archive_path"]))
    body = {
        "schema": RECOVERY_MANIFEST_SCHEMA,
        "package_id": package_id,
        "created_at": _timestamp(),
        "status": status,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "checkpoint": "checkpoint/manifest.json",
        "checkpoint_manifest_hash": load_ddp_manifest(checkpoint_root, verify_shards=True)["manifest_hash"],
        "iteration": int(latest["iteration"]),
        "runtime_binary": runtime_entry,
        "files": [
            {"path": archive_name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path, archive_name in include
        ],
        "remote_verified": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.writestr("recovery_manifest.json", json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            for path, archive_name in include:
                archive.write(path, archive_name)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA", "DDP_CHECKPOINT_SCHEMA", "DDP_PROVENANCE_SCHEMA",
    "RECOVERY_MANIFEST_SCHEMA", "aggregate_training_provenance", "build_recovery_bundle",
    "continuation_state_hash", "freeze_distributed_policy", "load_ddp_manifest",
    "load_distributed_checkpoint", "load_rank_checkpoint_for_resume",
    "save_distributed_checkpoint", "update_checkpoint_index",
]
