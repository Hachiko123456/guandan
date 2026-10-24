# SPDX-License-Identifier: MIT
"""Versioned two-rank durable checkpoints and downloadable recovery bundles."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
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
RECOVERY_MANIFEST_SCHEMA = "guandan-marvel-recovery-manifest-v3"
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


# The legacy ``continuation_state_hash`` above is kept byte-for-byte: existing
# DDP shard envelopes validate it through ``load_rank_checkpoint_for_resume``.
# The versioned identity below is the *complete* continuation state; it adds the
# fields that ``LocalTrainerBase._checkpoint_common`` persists and
# ``_load_common_state`` restores on resume (AMP scaler, AMP bookkeeping and the
# explicit curriculum origin), which the legacy hash never covered.
CONTINUATION_IDENTITY_SCHEMA = "guandan-marvel-continuation-identity-v2"
CONTINUATION_IDENTITY_KEYS = (
    "agent_resume_state_dict", "agent_state_dict", "critic_state_dict",
    "agent_optimizer_state_dict", "critic_optimizer_state_dict",
    "replay_buffer_state", "sampler_state", "rng_state", "schedule_state",
    "merge_state", "iteration", "durable_updates", "learner_updates",
    "durable_learner_updates", "counters",
    "grad_scaler_state", "amp", "curriculum_offset",
)


def continuation_identity(payload: Mapping[str, object]) -> dict[str, object]:
    """Deterministic, versioned identity of the full continuation state.

    Every field listed in :data:`CONTINUATION_IDENTITY_KEYS` is folded into one
    digest in a fixed order (mappings are ordered by key inside
    ``_update_object_digest``).  A field that the payload does not carry is
    hashed through an explicit absence marker instead of being skipped, so an
    old payload that never recorded a field can never collide with one that
    recorded a value.  Nothing here raises for a legacy payload.
    """
    digest = hashlib.sha256()
    _update_object_digest(digest, "schema")
    _update_object_digest(digest, CONTINUATION_IDENTITY_SCHEMA)
    for key in CONTINUATION_IDENTITY_KEYS:
        _update_object_digest(digest, "field")
        _update_object_digest(digest, key)
        if key in payload:
            _update_object_digest(digest, payload[key])
        else:
            # No typed digest branch starts with this marker, so an explicit
            # absent field is distinguishable from every stored value.
            digest.update(b"missing-field")
    return {"schema": CONTINUATION_IDENTITY_SCHEMA, "hash": digest.hexdigest()}


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
        # Versioned full-continuation identity (legacy recovery manifests do not
        # carry it; see ``_continuation_state_mismatch`` for the fail-closed rule).
        "continuation_identity": continuation_identity(payload),
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


class CheckpointTransactionError(RuntimeError):
    """A checkpoint transaction failed; every rank raises the identical record.

    ``stage`` is the stage that produced the *first* failure in rank order and
    ``failures`` keeps the original exception type/message per rank, so the
    first error is never replaced by a later barrier timeout or a generic
    message.
    """

    def __init__(self, stage: str, failures: "list[tuple[int, str, str]] | tuple[tuple[int, str, str], ...]") -> None:
        self.stage = str(stage)
        self.failures = tuple(
            (int(rank), str(kind), str(message)) for rank, kind, message in failures
        )
        detail = "; ".join(
            f"rank{rank} {kind}: {message}" for rank, kind, message in self.failures
        )
        super().__init__(f"checkpoint transaction failed at stage {self.stage!r}: {detail}")


class CheckpointStateMismatch(ValueError):
    """A complete recovery already occupies the destination for a different state."""


STAGE_PREPARE = "prepare"
STAGE_PREFLIGHT = "preflight"
STAGE_SHARD = "shard"
STAGE_VALIDATE = "validate"
STAGE_PUBLISH = "publish"
STAGE_INDEX = "index"

# Identity of the *continuation state*, not of the shard bytes.  ``torch.save``
# zip timestamps differ between identical saves, so shard hashes cannot be an
# idempotency key; these per-rank summary fields can.
IDEMPOTENCY_KEYS = (
    "checkpoint_schema", "algorithm", "iteration", "durable_updates",
    "source_hash", "source_commit", "semantic_contract_hash",
    "backend_identity_hash", "resolved_config_hash",
    "agent_resume_state_hash", "agent_merged_state_hash", "critic_state_hash",
    "continuation_state_hash",
)


def _error_record(stage: str, error: BaseException) -> dict[str, object]:
    return {
        "error": True,
        "stage": str(stage),
        "type": type(error).__name__,
        "message": str(error),
    }


def _converge(gathered: "list[object]", *, stage: str) -> list[dict[str, object]]:
    """Fail every rank with the same ordered failure table, or return the records."""
    records = [dict(row) for row in gathered]
    failures: list[tuple[int, str, str]] = []
    for index, record in enumerate(records):
        failure = record.get("error") if record else None
        if isinstance(failure, Mapping):
            failures.append((
                index,
                str(failure.get("stage") or stage),
                f"{failure.get('type')}: {failure.get('message')}",
            ))
        elif failure:
            failures.append((index, stage, str(failure)))
    if failures:
        raise CheckpointTransactionError(failures[0][1], failures)
    return records


def _continuation_state_mismatch(
    manifest: Mapping[str, object], rows: "list[Mapping[str, object]]",
) -> str | None:
    """Return why the stored recovery must not be reused, or ``None`` if identical.

    The legacy ``IDEMPOTENCY_KEYS`` are still compared exactly as before.  The
    versioned ``continuation_identity`` is compared only when *both* the stored
    row and the fresh row carry it.  A legacy manifest (or a mixed pair) lacks
    the versioned identity and therefore cannot prove that the AMP scaler, AMP
    bookkeeping and curriculum offset match the fresh state: that is
    **fail-closed** — the save is rejected instead of silently reusing a state
    that may differ, and the old recovery is left untouched.
    """
    if int(manifest.get("world_size", -1)) != len(rows):
        return "the stored recovery world size differs from the running world size"
    stored_rows = manifest.get("rank_shards")
    if not isinstance(stored_rows, list):
        return "the stored recovery has no usable rank summaries"
    stored = {int(row.get("rank", -1)): row for row in stored_rows if isinstance(row, Mapping)}
    if sorted(stored) != list(range(len(rows))):
        return "the stored recovery does not cover every rank"
    for row in rows:
        other = stored[int(row["rank"])]
        for key in IDEMPOTENCY_KEYS:
            if row.get(key) != other.get(key):
                return f"rank {int(row['rank'])} differs in {key}"
        fresh_identity = row.get("continuation_identity")
        stored_identity = other.get("continuation_identity")
        if not isinstance(fresh_identity, Mapping) or not isinstance(stored_identity, Mapping):
            return (
                f"rank {int(row['rank'])} has no {CONTINUATION_IDENTITY_SCHEMA} entry; "
                "a legacy recovery cannot prove that the AMP scaler, AMP state and "
                "curriculum offset match, so it must not be reused"
            )
        if (
            fresh_identity.get("schema") != CONTINUATION_IDENTITY_SCHEMA
            or stored_identity.get("schema") != CONTINUATION_IDENTITY_SCHEMA
        ):
            return (
                f"rank {int(row['rank'])} declares an unsupported continuation identity "
                f"schema (expected {CONTINUATION_IDENTITY_SCHEMA})"
            )
        if fresh_identity.get("hash") != stored_identity.get("hash"):
            return f"rank {int(row['rank'])} differs in continuation_identity"
    return None


def _same_continuation_state(
    manifest: Mapping[str, object], rows: "list[Mapping[str, object]]",
) -> bool:
    """True only when the stored recovery describes exactly the fresh state.

    Legacy manifests without the versioned continuation identity return
    ``False`` (fail closed) and are never silently reused.
    """
    return _continuation_state_mismatch(manifest, rows) is None


def _build_ddp_manifest(rows: "list[Mapping[str, object]]", *, world_size: int) -> dict[str, object]:
    provenance = aggregate_training_provenance(rows)
    global_counters = {
        key: sum(int((row.get("counters") or {}).get(key, 0)) for row in rows)
        for key in ("total_env_steps", "total_game_steps", "completed_games_total")
    }
    body = {
        "schema": DDP_CHECKPOINT_SCHEMA,
        "created_at": _timestamp(),
        "world_size": int(world_size),
        "resume_world_size_policy": "exact_same_world_size_only",
        "rank_shards": [dict(row) for row in rows],
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
    return {**body, "manifest_hash": canonical_hash(body)}


def save_distributed_checkpoint(
    trainer: object,
    destination: str | Path,
    *,
    context: DistributedContext,
    index_root: str | Path | None = None,
) -> Path | None:
    """Save one exact-resume shard per rank as an all-rank transaction.

    Stages ``prepare -> preflight -> shard -> validate -> publish/index`` run in
    a fixed order on every rank and every stage boundary is a collective
    exchange.  A catchable local I/O error is recorded, exchanged, and raised
    on *all* ranks as the same :class:`CheckpointTransactionError` carrying the
    original stage and exception, instead of a rank-0-only raise that leaves
    the other rank in a different collective.

    The save is idempotent with state validation: an existing complete recovery
    for the same iteration is reused only if the per-rank continuation-state
    hashes match; otherwise every rank rejects it (``CheckpointStateMismatch``)
    and the old recovery is left untouched.  Nothing is ever treated as
    complete merely because the directory exists.

    ``index_root`` (optional, old callers unchanged) publishes the run-level
    index inside the same error propagation; an index failure therefore fails
    every rank and leaves the complete directory to be re-indexed next time.
    """
    destination = Path(destination).resolve()
    temporary = destination.with_name(destination.name + ".tmp")
    world_size = int(getattr(context, "world_size", 1))
    rank = int(getattr(context, "rank", 0))
    is_primary = bool(getattr(context, "is_primary", True))
    existing_manifest: dict[str, object] | None = None
    manifest: dict[str, object] | None = None
    try:
        # -- prepare: primary creates the staging directory -----------------
        stage = STAGE_PREPARE
        error: dict[str, object] | None = None
        if is_primary:
            try:
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir(parents=True)
            except BaseException as failure:  # noqa: BLE001 - recorded, converged below
                error = _error_record(stage, failure)
        # Every rank always enters this barrier, even after a local failure.
        context.barrier()

        # -- preflight: validate an existing destination on every rank ------
        stage = STAGE_PREFLIGHT
        if error is None and destination.exists():
            try:
                existing_manifest = load_ddp_manifest(destination, verify_shards=True)
                if int(existing_manifest.get("world_size", -1)) != world_size:
                    raise CheckpointStateMismatch(
                        "recovery world size "
                        f"{existing_manifest.get('world_size')} is incompatible with "
                        f"the running world size {world_size}"
                    )
            except BaseException as failure:  # noqa: BLE001
                error = _error_record(stage, failure)
        _converge(context.all_gather_object({"error": error}), stage=stage)
        error = None

        # -- shard: real payload + real shard serialization on each rank ----
        stage = STAGE_SHARD
        summary: dict[str, object] | None = None
        try:
            payload = trainer.checkpoint_payload()
            shard = temporary / f"rank{rank:03d}.pt"
            TrainingCheckpointStore.save(payload, shard)
            summary = _rank_summary(payload, rank=rank, shard=shard)
            atomic_json(temporary / f"rank{rank:03d}.summary.json", summary)
        except BaseException as failure:  # noqa: BLE001
            summary = None
            error = _error_record(stage, failure)
        records = _converge(
            context.all_gather_object({"error": error, "summary": summary}), stage=stage,
        )
        rows = [dict(record["summary"]) for record in records]
        # Deterministic on every rank: they all validate the same gathered rows.
        _validate_rank_summaries(rows, world_size=world_size)

        # -- validate: idempotency decision must agree on every rank --------
        stage = STAGE_VALIDATE
        errors: dict[str, object] | None = None
        mismatch = (
            None if existing_manifest is None
            else _continuation_state_mismatch(existing_manifest, rows)
        )
        reuse = existing_manifest is not None and mismatch is None
        if existing_manifest is not None and not reuse:
            errors = _error_record(stage, CheckpointStateMismatch(
                f"iteration {rows[0]['iteration']} already holds a complete recovery that "
                f"cannot be reused ({mismatch}); refusing to reuse or overwrite it"
            ))
        decisions = _converge(
            context.all_gather_object({"error": errors, "reuse": bool(reuse)}), stage=stage,
        )
        flags = {bool(record["reuse"]) for record in decisions}
        if len(flags) != 1:
            raise CheckpointTransactionError(
                stage,
                [(rank, "CheckpointStateMismatch", "ranks disagree on the reuse decision")],
            )
        reuse = flags.pop()

        # -- publish + index: both inside the same error propagation --------
        stage = STAGE_PUBLISH
        error = None
        if reuse:
            manifest = existing_manifest
        elif is_primary:
            try:
                manifest = _build_ddp_manifest(rows, world_size=world_size)
                atomic_json(temporary / "manifest.json", manifest)
                os.replace(temporary, destination)
            except BaseException as failure:  # noqa: BLE001
                error = _error_record(stage, failure)
        if error is None and is_primary and index_root is not None:
            try:
                if manifest is None:  # pragma: no cover - reuse implies a manifest
                    raise RuntimeError("no manifest available for the checkpoint index")
                update_checkpoint_index(
                    index_root, checkpoint_dir=destination, ddp_manifest=manifest,
                )
            except BaseException as failure:  # noqa: BLE001
                error = _error_record(STAGE_INDEX, failure)
        _converge(context.all_gather_object({"error": error}), stage=stage)
        return destination if is_primary else None
    finally:
        # After the last exchange: an interrupted attempt never looks complete.
        if is_primary:
            try:
                if temporary.exists():
                    shutil.rmtree(temporary)
            except Exception:  # noqa: BLE001 - cleanup must not mask the outcome
                pass


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
    # New manifests pin the versioned full-continuation identity; verify it
    # against the shard bytes.  Legacy manifests do not carry the field and keep
    # loading exactly as before (the legacy keys above are still mandatory).
    stored_identity = row.get("continuation_identity")
    if stored_identity is not None:
        if (
            not isinstance(stored_identity, Mapping)
            or stored_identity.get("schema") != CONTINUATION_IDENTITY_SCHEMA
        ):
            raise ValueError(
                "rank shard continuation identity schema is unsupported: "
                f"{stored_identity!r}"
            )
        if stored_identity != recomputed.get("continuation_identity"):
            raise ValueError("rank shard identity mismatch for continuation_identity")
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


def prune_run_artifacts(
    output_root: str | Path,
    *,
    checkpoint_keep_last: int,
    evaluation_keep_last: int,
) -> dict[str, object]:
    """Bound disk usage without deleting paths outside versioned artifact namespaces."""
    root = Path(output_root).resolve()
    checkpoint_keep_last = int(checkpoint_keep_last)
    evaluation_keep_last = int(evaluation_keep_last)
    if checkpoint_keep_last <= 0 or evaluation_keep_last < 0:
        raise ValueError("checkpoint_keep_last must be positive and evaluation_keep_last non-negative")
    index_path = root / "checkpoint_manifest.json"
    if not index_path.is_file():
        raise FileNotFoundError("checkpoint manifest is required before retention pruning")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
        raise ValueError("checkpoint manifest schema mismatch before retention pruning")
    history = sorted(
        (dict(row) for row in payload.get("history", [])),
        key=lambda row: int(row["iteration"]),
    )
    if not history:
        raise ValueError("checkpoint retention requires non-empty durable history")

    checkpoint_root = (root / "checkpoints").resolve()
    evaluation_root = (root / "evaluation_checkpoints").resolve()
    for row in history:
        iteration = int(row["iteration"])
        expected = f"checkpoints/iter{iteration:06d}"
        if str(row.get("checkpoint")) != expected:
            raise ValueError(f"unsafe checkpoint history path: {row.get('checkpoint')!r}")
        candidate = (root / expected).resolve()
        if not candidate.is_relative_to(checkpoint_root) or candidate.parent != checkpoint_root:
            raise ValueError(f"checkpoint history path escapes retention root: {expected}")
        if candidate.is_symlink() or not (candidate / "manifest.json").is_file():
            raise ValueError(f"checkpoint history entry is missing or unsafe: {expected}")

    retained_history = history[-checkpoint_keep_last:]
    retained_checkpoint_paths = {str(row["checkpoint"]) for row in retained_history}
    latest = payload.get("latest")
    if not isinstance(latest, Mapping):
        raise ValueError("checkpoint manifest latest entry is required")
    latest_path = str(latest.get("checkpoint", ""))
    if latest_path != str(history[-1]["checkpoint"]):
        raise ValueError("checkpoint manifest latest entry does not match durable history tail")
    retained_checkpoint_paths.add(latest_path)

    checkpoint_candidates: list[tuple[Path, str]] = []
    if checkpoint_root.is_dir():
        for candidate in sorted(checkpoint_root.iterdir()):
            if candidate.is_symlink():
                raise ValueError(f"checkpoint retention refuses symlink: {candidate}")
            if candidate.is_dir() and re.fullmatch(r"iter\d{6}\.tmp(?:-\d+)?", candidate.name):
                # A save interrupted before atomic publication; harmless to drop.
                shutil.rmtree(candidate)
                continue
            if not candidate.is_dir():
                raise ValueError(f"unexpected file in checkpoint retention root: {candidate}")
            if not candidate.name.startswith("iter") or len(candidate.name) != 10 or not candidate.name[4:].isdigit():
                raise ValueError(f"unexpected checkpoint directory name: {candidate.name}")
            resolved = candidate.resolve()
            if resolved.parent != checkpoint_root:
                raise ValueError(f"checkpoint candidate escapes retention root: {candidate}")
            checkpoint_candidates.append((candidate, candidate.relative_to(root).as_posix()))

    evaluation_files: list[Path] = []
    if evaluation_root.is_dir():
        for candidate in sorted(evaluation_root.iterdir(), key=lambda path: path.name):
            if candidate.is_symlink():
                raise ValueError(f"evaluation retention refuses symlink: {candidate}")
            if candidate.is_file() and re.fullmatch(r"iter\d{6}\.pt\.tmp(?:-\d+)?", candidate.name):
                # A freeze interrupted before atomic publication; harmless to drop.
                candidate.unlink()
                continue
            if not candidate.is_file() or not candidate.name.startswith("iter") or not candidate.name.endswith(".pt"):
                raise ValueError(f"unexpected evaluation artifact: {candidate}")
            stem = candidate.stem
            if len(stem) != 10 or not stem[4:].isdigit() or candidate.resolve().parent != evaluation_root:
                raise ValueError(f"unsafe evaluation artifact: {candidate}")
            evaluation_files.append(candidate)
    retained_evaluations = set(evaluation_files[-evaluation_keep_last:]) if evaluation_keep_last else set()

    planned_checkpoint_deletes = [
        (candidate, relative) for candidate, relative in checkpoint_candidates
        if relative not in retained_checkpoint_paths
    ]
    planned_evaluation_deletes = [candidate for candidate in evaluation_files if candidate not in retained_evaluations]
    payload["history"] = retained_history
    payload["retention"] = {
        "schema": "guandan-marvel-artifact-retention-v2",
        "checkpoint_keep_last": checkpoint_keep_last,
        "evaluation_keep_last": evaluation_keep_last,
        "deleted_checkpoints": [],
        "deleted_evaluations": [],
        "retained_checkpoints": sorted(retained_checkpoint_paths),
        "retained_evaluations": sorted(path.relative_to(root).as_posix() for path in retained_evaluations),
        "updated_at": _timestamp(),
    }
    # Publish the bounded index before deletion.  A deletion failure can leave harmless
    # extra files, but can no longer leave the manifest pointing at removed history.
    atomic_json(index_path, payload)

    deleted_checkpoints: list[str] = []
    for candidate, relative in planned_checkpoint_deletes:
        shutil.rmtree(candidate)
        deleted_checkpoints.append(relative)
    deleted_evaluations: list[str] = []
    for candidate in planned_evaluation_deletes:
        candidate.unlink()
        deleted_evaluations.append(candidate.relative_to(root).as_posix())

    payload["retention"]["deleted_checkpoints"] = deleted_checkpoints
    payload["retention"]["deleted_evaluations"] = deleted_evaluations
    payload["retention"]["updated_at"] = _timestamp()
    atomic_json(index_path, payload)
    atomic_json(root / "artifact_retention.json", payload["retention"])
    return dict(payload["retention"])


def _retry_readable(operation, *, attempts: int = 6, delay: float = 0.2):
    """Retry brief sharing violations from concurrent atomic replacements.

    Readers of continuously rewritten JSON artifacts (progress.json, ...) can
    observe a transient lock while another thread performs ``os.replace`` on
    Windows; the same pattern is harmless on Linux but retrying keeps local and
    Kaggle behaviour identical.
    """
    last_error: BaseException | None = None
    for attempt in range(max(int(attempts), 1)):
        try:
            return operation()
        except OSError as error:
            last_error = error
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
    raise last_error  # pragma: no cover - defensive


def _stable_sha256_file(path: Path) -> str:
    return _retry_readable(lambda: sha256_file(path))


def _stable_archive_write(archive: zipfile.ZipFile, path: Path, archive_name: str) -> None:
    _retry_readable(lambda: archive.write(path, archive_name))


def build_recovery_bundle(
    *,
    run_root: str | Path,
    destination: str | Path,
    source_commit: str,
    source_archive_sha256: str,
    source_content_manifest_hash: str,
    package_id: str,
    status: str,
    native_extension: str | Path | None,
    compresslevel: int = 6,
) -> Path | None:
    if isinstance(compresslevel, bool) or not isinstance(compresslevel, int) or not 1 <= compresslevel <= 9:
        raise ValueError("recovery compression level must be an integer in [1,9]")
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
        "anchor_history.json", "pilot_initial_policy.pt", "pilot_initial_identity.json",
        "pilot_run_contract.json",
    ):
        path = root / name
        if path.is_file():
            include.append((path, f"artifacts/{name}"))
    pilot_history = (root / "pilot_run_contract.json").is_file()
    if pilot_history:
        segment_file = root / "pilot_history_segment.json"
        segment = json.loads(segment_file.read_text(encoding="utf-8"))["segment"] if segment_file.is_file() else root.name
        if not segment or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in segment):
            raise ValueError("unsafe pilot history segment")
        current = [root / "metrics.jsonl"]
        for pattern in ("ranks/rank*/pilot_runtime.json", "ranks/rank*/pilot_iterations.jsonl",
                        "ranks/rank*/diagnostic.jsonl", "ranks/rank*/checkpoint_io.jsonl", "ranks/rank*/trainer/ippo_metrics.jsonl"):
            current.extend(root.glob(pattern))
        include.extend((path, f"artifacts/pilot_history/{segment}/{path.relative_to(root).as_posix()}")
                       for path in current if path.is_file())
        prior = root / "pilot_prior_history"
        if prior.is_dir():
            include.extend((path, f"artifacts/pilot_history/{path.relative_to(prior).as_posix()}")
                           for path in sorted(prior.rglob("*")) if path.is_file())
    # The progress heartbeat can replace small artifacts while a large checkpoint
    # is compressed. Pilot manifests hash precisely the immutable bytes written.
    snapshots = {name: _retry_readable(path.read_bytes) for path, name in include
                 if pilot_history and name.startswith("artifacts/")}
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
        "source_content_manifest_hash": source_content_manifest_hash,
        "checkpoint": "checkpoint/manifest.json",
        "checkpoint_manifest_hash": load_ddp_manifest(checkpoint_root, verify_shards=True)["manifest_hash"],
        "iteration": int(latest["iteration"]),
        "runtime_binary": runtime_entry,
        "files": [
            {"path": archive_name,
             "sha256": hashlib.sha256(snapshots[archive_name]).hexdigest() if archive_name in snapshots else _stable_sha256_file(path),
             "bytes": len(snapshots[archive_name]) if archive_name in snapshots else path.stat().st_size}
            for path, archive_name in include
        ],
        "remote_verified": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel, allowZip64=True) as archive:
            archive.writestr("recovery_manifest.json", json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            for path, archive_name in include:
                if archive_name in snapshots:
                    archive.writestr(archive_name, snapshots[archive_name])
                else:
                    _stable_archive_write(archive, path, archive_name)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA", "CONTINUATION_IDENTITY_SCHEMA",
    "CheckpointStateMismatch", "CheckpointTransactionError",
    "DDP_CHECKPOINT_SCHEMA", "DDP_PROVENANCE_SCHEMA",
    "RECOVERY_MANIFEST_SCHEMA", "aggregate_training_provenance", "build_recovery_bundle",
    "continuation_identity", "continuation_state_hash", "freeze_distributed_policy",
    "load_ddp_manifest", "prune_run_artifacts",
    "load_distributed_checkpoint", "load_rank_checkpoint_for_resume",
    "save_distributed_checkpoint", "update_checkpoint_index",
]
