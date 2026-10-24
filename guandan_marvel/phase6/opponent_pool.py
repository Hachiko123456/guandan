# SPDX-License-Identifier: MIT
"""Optional, fail-closed checkpoint opponent pool for phase-6 training.

The pool owns three responsibilities:

1. immutable entry identity (checkpoint SHA256, architecture, protocol,
   backend/build identity, resolved-config hash, source hash) and a strict
   manifest with configured sampling weights;
2. deterministic per-episode seat assignment (one current team, one frozen
   opponent team) with an isolated sampling RNG and persisted state;
3. the runtime router that selects actions from the policy which owns the
   acting seat, while the *current* model continues to run every step so its
   value/GAE clock sees the complete environment timeline.

The default is off: :class:`OpponentPoolConfig` with ``enabled=False`` reads no
file, loads no model, draws no RNG and registers no trajectory field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from guandan_marvel.evaluation.checkpoints import (
    EVALUATION_CHECKPOINT_SCHEMA, V2PolicyArtifact, load_v2_policy,
    verify_policy_identity,
)
from guandan_marvel.evaluation.common import file_sha256
from guandan_marvel.model import KVCache
from guandan_marvel.model.tensordict import TensorDict

from .config_types import OpponentPoolConfig
from .design import (
    CURRENT_POLICY_OWNER_ID, NO_POLICY_OWNER_ID, POOL_ASSIGNMENT_SCHEMA,
    POOL_ENTRY_KINDS, POOL_ENTRY_SCHEMA, POOL_ENTRY_TAGS, POOL_MANIFEST_SCHEMA,
    POOL_ROUTER_STATE_SCHEMA, POOL_SAMPLING_SCHEMA, POOL_SCHEMA,
    POOL_STATE_SCHEMA, canonical_hash,
)


class OpponentPoolError(RuntimeError):
    """An opponent-pool contract was violated; training must not continue."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_weight(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpponentPoolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise OpponentPoolError(f"{name} must be finite and non-negative")
    return result


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
_ENTRY_KEYS = frozenset({
    "id", "kind", "checkpoint", "checkpoint_sha256", "architecture",
    "protocol_version", "backend", "resolved_config_hash", "source_hash",
    "policy_version", "weight", "active", "tags",
})
_MANIFEST_KEYS = frozenset({
    "schema", "manifest_hash", "created_at", "sampling", "assignment", "entries",
})


def _normalize_entry(entry: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(entry, Mapping):
        raise OpponentPoolError("opponent pool entry must be a mapping")
    unknown = sorted(set(entry) - _ENTRY_KEYS)
    if unknown:
        raise OpponentPoolError(f"opponent pool entry has unknown keys: {unknown}")
    entry_id = str(entry.get("id", "")).strip()
    if not entry_id:
        raise OpponentPoolError("opponent pool entry requires a non-empty id")
    kind = str(entry.get("kind", ""))
    if kind not in POOL_ENTRY_KINDS:
        raise OpponentPoolError(
            f"unsupported opponent entry kind {kind!r}; expected {sorted(POOL_ENTRY_KINDS)}"
        )
    checkpoint = entry.get("checkpoint")
    if checkpoint is None or not str(checkpoint).strip():
        raise OpponentPoolError("opponent entry requires a checkpoint path")
    tags = entry.get("tags", ())
    if not isinstance(tags, (list, tuple)):
        raise OpponentPoolError("opponent entry tags must be a list")
    tags = tuple(str(tag) for tag in tags)
    unknown_tags = sorted(set(tags) - set(POOL_ENTRY_TAGS))
    if unknown_tags:
        raise OpponentPoolError(f"unsupported opponent entry tags: {unknown_tags}")
    policy_version = entry.get("policy_version", 0)
    if isinstance(policy_version, bool) or int(policy_version) < 0:
        raise OpponentPoolError("opponent entry policy_version must be a non-negative integer")
    checkpoint_sha = str(entry.get("checkpoint_sha256", "")).lower()
    if len(checkpoint_sha) != 64 or any(ch not in "0123456789abcdef" for ch in checkpoint_sha):
        raise OpponentPoolError("opponent entry checkpoint_sha256 must be a 64-char digest")
    for name in ("architecture", "protocol_version", "backend"):
        if not str(entry.get(name, "")).strip():
            raise OpponentPoolError(f"opponent entry {name} must be non-empty")
    return {
        "id": entry_id,
        "kind": kind,
        "checkpoint": str(Path(str(checkpoint)).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "architecture": str(entry.get("architecture", "")),
        "protocol_version": str(entry.get("protocol_version", "")),
        "backend": str(entry.get("backend", "")),
        "resolved_config_hash": str(entry.get("resolved_config_hash", "")),
        "source_hash": str(entry.get("source_hash", "")),
        "policy_version": int(policy_version),
        "weight": _finite_weight(entry.get("weight", 1.0), name="entry weight"),
        "active": bool(entry.get("active", True)),
        "tags": list(tags),
    }


def _manifest_hash(payload: Mapping[str, object]) -> str:
    # ``created_at`` is audit metadata, not pool identity: the same entries,
    # weights, seed and schemes must produce the same manifest hash across
    # trainer constructions so an exact resume is possible.
    #
    # The host-absolute ``checkpoint`` path is likewise *not* identity: the
    # entry is identified by ``id`` + ``checkpoint_sha256`` + its architecture/
    # protocol/backend/semantics fields.  Hashing the absolute path would make
    # an otherwise byte-identical dataset unrunnable after it is moved (for
    # example a Kaggle output extracted at a new mount point).
    body = {
        key: value for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    entries = body.get("entries")
    if isinstance(entries, list):
        body["entries"] = [
            {key: value for key, value in entry.items() if key != "checkpoint"}
            if isinstance(entry, Mapping) else entry
            for entry in entries
        ]
    return canonical_hash(body)


def build_manifest(
    *,
    entries: Sequence[Mapping[str, object]],
    sampling_seed: int = 0,
    sampling_scheme: str = POOL_SAMPLING_SCHEMA,
    assignment_scheme: str = "team_rotation_v1",
) -> dict[str, object]:
    normalized = [_normalize_entry(entry) for entry in entries]
    ids = [entry["id"] for entry in normalized]
    if len(set(ids)) != len(ids):
        raise OpponentPoolError("opponent pool entry IDs must be unique")
    if not any(entry["active"] and float(entry["weight"]) > 0.0 for entry in normalized):
        raise OpponentPoolError("opponent pool requires an active positive-weight entry")
    payload: dict[str, object] = {
        "schema": POOL_MANIFEST_SCHEMA,
        "created_at": _timestamp(),
        "sampling": {
            "schema": str(sampling_scheme),
            "seed": int(sampling_seed),
            "scheme": "weighted_active_v1",
        },
        "assignment": {
            "schema": POOL_ASSIGNMENT_SCHEMA,
            "scheme": str(assignment_scheme),
        },
        "entries": normalized,
    }
    payload["manifest_hash"] = _manifest_hash(payload)
    return payload


def load_manifest(path: str | Path, *, validate_checkpoints: bool = True) -> dict[str, object]:
    import json

    source = Path(path)
    if not source.is_file():
        raise OpponentPoolError(f"opponent pool manifest is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OpponentPoolError(f"opponent pool manifest is not readable JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise OpponentPoolError("opponent pool manifest must be a JSON object")
    unknown = sorted(set(payload) - _MANIFEST_KEYS)
    if unknown:
        raise OpponentPoolError(f"opponent pool manifest has unknown keys: {unknown}")
    missing = sorted(_MANIFEST_KEYS - set(payload))
    if missing:
        raise OpponentPoolError(f"opponent pool manifest is missing keys: {missing}")
    if payload.get("schema") != POOL_MANIFEST_SCHEMA:
        raise OpponentPoolError(
            f"unsupported opponent manifest schema {payload.get('schema')!r}"
        )
    declared = str(payload.get("manifest_hash", ""))
    actual = _manifest_hash(payload)
    if declared != actual:
        raise OpponentPoolError("opponent manifest hash is inconsistent with its contents")
    sampling = payload.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("schema") != POOL_SAMPLING_SCHEMA:
        raise OpponentPoolError("opponent manifest sampling schema is unsupported")
    assignment = payload.get("assignment")
    if not isinstance(assignment, Mapping) or assignment.get("schema") != POOL_ASSIGNMENT_SCHEMA:
        raise OpponentPoolError("opponent manifest assignment schema is unsupported")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise OpponentPoolError("opponent manifest entries must be a non-empty list")
    entries = [_normalize_entry(entry) for entry in raw_entries]
    entry_ids = [entry["id"] for entry in entries]
    if len(set(entry_ids)) != len(entry_ids):
        raise OpponentPoolError("opponent manifest entry IDs must be unique")
    if not any(entry["active"] and float(entry["weight"]) > 0.0 for entry in entries):
        raise OpponentPoolError("opponent manifest has no active positive-weight entry")
    for entry in entries:
        if validate_checkpoints:
            path_value = Path(str(entry["checkpoint"]))
            if not path_value.is_file():
                raise OpponentPoolError(f"opponent checkpoint is missing: {path_value}")
            if file_sha256(path_value) != entry["checkpoint_sha256"]:
                raise OpponentPoolError(
                    f"opponent checkpoint SHA256 changed: {path_value}"
                )
    return dict(payload)


def write_manifest(payload: Mapping[str, object], path: str | Path) -> Path:
    import json

    normalized = dict(payload)
    normalized["manifest_hash"] = _manifest_hash(normalized)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{id(payload)}")
    temporary.write_text(
        json.dumps(normalized, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


# ---------------------------------------------------------------------------
# Entries, assignment and pool state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpponentPoolEntry:
    owner_id: int
    entry_id: str
    checkpoint: Path
    checkpoint_sha256: str
    architecture: str
    protocol_version: str
    backend: str
    resolved_config_hash: str
    source_hash: str
    policy_version: int
    weight: float
    active: bool
    tags: tuple[str, ...]
    artifact: V2PolicyArtifact

    @property
    def manifest_record(self) -> dict[str, object]:
        return {
            "id": self.entry_id,
            "kind": "v2_evaluation",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "architecture": self.architecture,
            "protocol_version": self.protocol_version,
            "backend": self.backend,
            "resolved_config_hash": self.resolved_config_hash,
            "source_hash": self.source_hash,
            "policy_version": int(self.policy_version),
            "weight": float(self.weight),
            "active": bool(self.active),
            "tags": list(self.tags),
        }

    def identity(self) -> dict[str, object]:
        return {
            "schema": POOL_ENTRY_SCHEMA,
            "owner_id": int(self.owner_id),
            "entry_id": self.entry_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "architecture": self.architecture,
            "protocol_version": self.protocol_version,
            "backend": self.backend,
            "resolved_config_hash": self.resolved_config_hash,
            "source_hash": self.source_hash,
            "policy_version": int(self.policy_version),
        }


@dataclass(frozen=True)
class SeatAssignment:
    assignment_id: int
    assignment_hash: str
    episode_index: int
    slot: int
    current_team: int
    seat_rotation: int
    owners_by_seat: tuple[int, ...]
    opponent_owner_id: int
    opponent_entry_id: str
    opponent_policy_version: int
    sample_seed: int
    draw_index: int
    pool_manifest_hash: str

    def payload(self) -> dict[str, object]:
        return {
            "schema": POOL_ASSIGNMENT_SCHEMA,
            "assignment_id": int(self.assignment_id),
            "assignment_hash": self.assignment_hash,
            "episode_index": int(self.episode_index),
            "slot": int(self.slot),
            "current_team": int(self.current_team),
            "seat_rotation": int(self.seat_rotation),
            "owners_by_seat": [int(value) for value in self.owners_by_seat],
            "opponent_owner_id": int(self.opponent_owner_id),
            "opponent_entry_id": self.opponent_entry_id,
            "opponent_policy_version": int(self.opponent_policy_version),
            "sample_seed": int(self.sample_seed),
            "draw_index": int(self.draw_index),
            "pool_manifest_hash": self.pool_manifest_hash,
        }

    @staticmethod
    def assignment_hash_for(payload: Mapping[str, object]) -> str:
        return canonical_hash({
            key: value for key, value in payload.items() if key != "assignment_hash"
        })

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], *, expected_manifest_hash: str,
    ) -> "SeatAssignment":
        required = {
            "schema", "assignment_id", "assignment_hash", "episode_index", "slot",
            "current_team", "seat_rotation", "owners_by_seat", "opponent_owner_id",
            "opponent_entry_id", "opponent_policy_version", "sample_seed",
            "draw_index", "pool_manifest_hash",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise OpponentPoolError(f"seat assignment is missing keys: {missing}")
        if payload.get("schema") != POOL_ASSIGNMENT_SCHEMA:
            raise OpponentPoolError("seat assignment schema mismatch")
        if str(payload["pool_manifest_hash"]) != str(expected_manifest_hash):
            raise OpponentPoolError("seat assignment belongs to a different pool manifest")
        declared = str(payload["assignment_hash"])
        actual = cls.assignment_hash_for(payload)
        if declared != actual:
            raise OpponentPoolError("seat assignment hash is inconsistent with its contents")
        owners = tuple(int(value) for value in payload["owners_by_seat"])
        if len(owners) != 4:
            raise OpponentPoolError("seat assignment must define exactly four seats")
        current_team = int(payload["current_team"])
        if current_team not in (0, 1):
            raise OpponentPoolError("seat assignment current_team must be 0 or 1")
        current = {seat for seat, owner in enumerate(owners) if owner == CURRENT_POLICY_OWNER_ID}
        if len(current) != 2 or len({seat % 2 for seat in current}) != 1:
            raise OpponentPoolError("seat assignment does not match the team rotation contract")
        if current_team != min(current) % 2:
            raise OpponentPoolError("seat assignment current_team does not match its seats")
        if len({owners[seat] for seat in range(4) if seat not in current}) != 1:
            raise OpponentPoolError("both opponent seats must be controlled by one pool entry")
        return cls(
            assignment_id=int(payload["assignment_id"]),
            assignment_hash=declared,
            episode_index=int(payload["episode_index"]),
            slot=int(payload["slot"]),
            current_team=current_team,
            seat_rotation=int(payload["seat_rotation"]),
            owners_by_seat=owners,
            opponent_owner_id=int(payload["opponent_owner_id"]),
            opponent_entry_id=str(payload["opponent_entry_id"]),
            opponent_policy_version=int(payload["opponent_policy_version"]),
            sample_seed=int(payload["sample_seed"]),
            draw_index=int(payload["draw_index"]),
            pool_manifest_hash=str(payload["pool_manifest_hash"]),
        )


class OpponentPool:
    """Immutable entries + configured sampling + deterministic assignments."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, object],
        entries: Sequence[OpponentPoolEntry],
        sampling_seed: int,
        sampling_scheme: str,
        assignment_scheme: str,
        device: torch.device | str,
        model_feature_identity: Mapping[str, object],
    ) -> None:
        self.manifest = dict(manifest)
        self.manifest_hash = str(manifest["manifest_hash"])
        self.entries = tuple(entries)
        self.sampling_seed = int(sampling_seed)
        self.sampling_scheme = str(sampling_scheme)
        self.assignment_scheme = str(assignment_scheme)
        self.device = torch.device(device)
        self.model_feature_identity = dict(model_feature_identity)
        self._by_owner = {entry.owner_id: entry for entry in self.entries}
        self._draw_index = 0
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.sampling_seed)
        self._episode_counters: dict[int, int] = {}
        self._assignment_counter = 0
        self._open: dict[int, SeatAssignment] = {}
        self._history_chain_hash = canonical_hash({"schema": "phase6-empty-assignment-history"})
        self._history_count = 0

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: OpponentPoolConfig,
        *,
        device: torch.device | str,
        expected_architecture: str,
        expected_protocol_version: str,
        expected_backend: str,
        expected_backend_identity_hash: str | None,
        model_feature_identity: Mapping[str, object],
        batch_slots: int,
    ) -> "OpponentPool":
        if not config.enabled:
            raise OpponentPoolError("cannot construct a pool from a disabled configuration")
        if int(batch_slots) <= 0:
            raise ValueError("batch_slots must be positive")
        if config.manifest_path is not None:
            manifest = load_manifest(config.manifest_path, validate_checkpoints=True)
            sampling = dict(manifest["sampling"])
            if str(sampling.get("schema")) != str(config.sampling_scheme):
                raise OpponentPoolError("manifest sampling scheme differs from the configuration")
            if int(sampling.get("seed", -1)) != int(config.sampling_seed):
                raise OpponentPoolError("manifest sampling seed differs from the configuration")
            if dict(manifest["assignment"]).get("scheme") != config.assignment_scheme:
                raise OpponentPoolError("manifest assignment scheme differs from the configuration")
            raw_entries = [dict(entry) for entry in manifest["entries"]]
        else:
            manifest = build_manifest(
                entries=config.entries,
                sampling_seed=config.sampling_seed,
                sampling_scheme=config.sampling_scheme,
                assignment_scheme=config.assignment_scheme,
            )
            raw_entries = [dict(entry) for entry in manifest["entries"]]
            for entry in raw_entries:
                path_value = Path(str(entry["checkpoint"]))
                if not path_value.is_file():
                    raise OpponentPoolError(f"opponent checkpoint is missing: {path_value}")
                if file_sha256(path_value) != entry["checkpoint_sha256"]:
                    raise OpponentPoolError(
                        f"opponent checkpoint SHA256 changed: {path_value}"
                    )
        entries: list[OpponentPoolEntry] = []
        if len(raw_entries) > int(config.max_resident_policies):
            # This phase keeps every validated entry resident.  If the
            # configured limit cannot hold them all, reject before any model is
            # allocated instead of silently exceeding the GPU budget or evicting
            # an opponent mid-episode.
            raise OpponentPoolError(
                f"opponent pool has {len(raw_entries)} entries but "
                f"max_resident_policies={config.max_resident_policies}; lazy eviction is not "
                "implemented, so the configuration cannot be satisfied without a mid-episode "
                "hot switch"
            )
        for index, raw in enumerate(raw_entries):
            artifact = load_v2_policy(
                str(raw["checkpoint"]), device=device, expected_algorithm="ippo",
            )
            # Freeze at load time, not only when a session is created.
            artifact.model.eval()
            for parameter in artifact.model.parameters():
                parameter.requires_grad_(False)
            if artifact.schema != EVALUATION_CHECKPOINT_SCHEMA:
                raise OpponentPoolError(
                    "opponent pool accepts only evaluation-only checkpoints "
                    f"({EVALUATION_CHECKPOINT_SCHEMA}); got {artifact.schema!r}"
                )
            if _checkpoint_training_resume_allowed(artifact):
                raise OpponentPoolError(
                    "opponent checkpoint advertises training resume; a frozen opponent must be "
                    "evaluation-only"
                )
            if artifact.architecture != expected_architecture:
                raise OpponentPoolError(
                    "opponent architecture differs from the current policy: "
                    f"{artifact.architecture!r} != {expected_architecture!r}"
                )
            # The manifest's own identity fields must agree with the artifact;
            # never trust a declared architecture/protocol/backend on its own.
            if str(raw["architecture"]) != artifact.architecture:
                raise OpponentPoolError(
                    "opponent manifest architecture differs from the checkpoint artifact: "
                    f"{raw['architecture']!r} != {artifact.architecture!r}"
                )
            if str(raw["protocol_version"]) != artifact.protocol_version:
                raise OpponentPoolError(
                    "opponent manifest protocol_version differs from the checkpoint artifact"
                )
            if str(raw["backend"]) != artifact.backend:
                raise OpponentPoolError(
                    "opponent manifest backend differs from the checkpoint artifact"
                )
            if artifact.protocol_version != expected_protocol_version:
                raise OpponentPoolError("opponent protocol version differs from the runtime")
            if artifact.backend != expected_backend:
                raise OpponentPoolError(
                    f"opponent backend {artifact.backend!r} differs from {expected_backend!r}"
                )
            if expected_backend_identity_hash is not None and (
                artifact.backend_identity_hash != expected_backend_identity_hash
            ):
                raise OpponentPoolError("opponent backend/build identity differs from the runtime")
            if artifact.sha256 != str(raw["checkpoint_sha256"]):
                raise OpponentPoolError(
                    f"opponent checkpoint {artifact.path} SHA256 differs from its manifest entry"
                )
            if str(raw["resolved_config_hash"]) not in ("", artifact.resolved_config_hash):
                raise OpponentPoolError("opponent manifest resolved_config_hash mismatch")
            if str(raw["source_hash"]) not in ("", str(artifact.metadata.get("source_hash") or "")):
                raise OpponentPoolError("opponent manifest source_hash mismatch")
            _require_compatible_model_features(artifact, model_feature_identity)
            entries.append(OpponentPoolEntry(
                owner_id=index + 1,
                entry_id=str(raw["id"]),
                checkpoint=artifact.path,
                checkpoint_sha256=artifact.sha256,
                architecture=artifact.architecture,
                protocol_version=artifact.protocol_version,
                backend=artifact.backend,
                resolved_config_hash=artifact.resolved_config_hash,
                source_hash=str(artifact.metadata.get("source_hash") or raw["source_hash"]),
                policy_version=int(raw["policy_version"]),
                weight=float(raw["weight"]),
                active=bool(raw["active"]),
                tags=tuple(str(tag) for tag in raw["tags"]),
                artifact=artifact,
            ))
        return cls(
            manifest=manifest,
            entries=entries,
            sampling_seed=config.sampling_seed,
            sampling_scheme=config.sampling_scheme,
            assignment_scheme=config.assignment_scheme,
            device=device,
            model_feature_identity=model_feature_identity,
        )

    # -- entries ----------------------------------------------------------
    @property
    def active_entries(self) -> tuple[OpponentPoolEntry, ...]:
        return tuple(
            entry for entry in self.entries
            if entry.active and float(entry.weight) > 0.0
        )

    @property
    def loaded_model_count(self) -> int:
        """Number of opponent model instances actually resident in memory."""
        return sum(1 for entry in self.entries if entry.artifact is not None)

    def estimated_parameter_bytes(self) -> int:
        total = 0
        for entry in self.entries:
            if entry.artifact is None:
                continue
            for parameter in entry.artifact.model.parameters():
                total += int(parameter.numel()) * int(parameter.element_size())
        return total

    def entry(self, owner_id: int) -> OpponentPoolEntry:
        try:
            return self._by_owner[int(owner_id)]
        except KeyError as error:
            raise OpponentPoolError(f"unknown opponent owner id {owner_id!r}") from error

    def parameter_hashes(self) -> dict[str, str]:
        return {
            entry.entry_id: _state_dict_hash(entry.artifact.model.state_dict())
            for entry in self.entries
        }

    # -- assignments ------------------------------------------------------
    def assign_episode(self, *, slot: int, episode_index: int) -> SeatAssignment:
        slot = int(slot)
        episode_index = int(episode_index)
        existing = self._open.get(slot)
        if existing is not None:
            if existing.episode_index == episode_index:
                return existing
            raise OpponentPoolError(
                "slot already has an open assignment; an opponent may not be hot-switched "
                "before the episode terminates"
            )
        active = self.active_entries
        if not active:
            raise OpponentPoolError("opponent pool has no active positive-weight entry")
        weights = torch.tensor([entry.weight for entry in active], dtype=torch.float64)
        index = int(torch.multinomial(weights, 1, generator=self._generator).item())
        entry = active[index]
        current_team = (episode_index + slot) % 2
        rotation = slot % 4
        current_seats = {current_team, current_team + 2}
        physical_current_team = current_team
        owners = tuple(
            CURRENT_POLICY_OWNER_ID if seat in current_seats else entry.owner_id
            for seat in range(4)
        )
        sample_seed = int(torch.randint(
            0, 2**31 - 1, (1,), generator=self._generator, dtype=torch.int64,
        ).item())
        draw_index = int(self._draw_index)
        self._draw_index += 2
        self._assignment_counter += 1
        payload = {
            "schema": POOL_ASSIGNMENT_SCHEMA,
            "assignment_id": int(self._assignment_counter),
            "episode_index": episode_index,
            "slot": slot,
            "current_team": physical_current_team,
            "seat_rotation": rotation,
            "owners_by_seat": [int(value) for value in owners],
            "opponent_owner_id": int(entry.owner_id),
            "opponent_entry_id": entry.entry_id,
            "opponent_policy_version": int(entry.policy_version),
            "sample_seed": sample_seed,
            "draw_index": draw_index,
            "pool_manifest_hash": self.manifest_hash,
        }
        payload["assignment_hash"] = SeatAssignment.assignment_hash_for(payload)
        assignment = SeatAssignment.from_payload(
            payload, expected_manifest_hash=self.manifest_hash,
        )
        self._open[slot] = assignment
        self._history_chain_hash = canonical_hash({
            "previous": self._history_chain_hash,
            "assignment": assignment.payload(),
        })
        self._history_count += 1
        return assignment

    def close_episode(self, *, slot: int) -> SeatAssignment:
        slot = int(slot)
        if slot not in self._open:
            raise OpponentPoolError(f"slot {slot} has no open opponent assignment")
        assignment = self._open.pop(slot)
        self._episode_counters[slot] = int(assignment.episode_index) + 1
        return assignment

    def next_episode_index(self, *, slot: int) -> int:
        return int(self._episode_counters.get(int(slot), 0))

    def assignment(self, *, slot: int) -> SeatAssignment:
        try:
            return self._open[int(slot)]
        except KeyError as error:
            raise OpponentPoolError(f"slot {slot} has no open opponent assignment") from error

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> dict[str, object]:
        generator_state = self._generator.get_state().tolist()
        payload: dict[str, object] = {
            "schema": POOL_STATE_SCHEMA,
            "manifest_hash": self.manifest_hash,
            "sampling_seed": int(self.sampling_seed),
            "sampling_scheme": str(self.sampling_scheme),
            "assignment_scheme": str(self.assignment_scheme),
            "generator_state": generator_state,
            "draw_index": int(self._draw_index),
            "assignment_counter": int(self._assignment_counter),
            "episode_counters": {
                str(slot): int(value) for slot, value in sorted(self._episode_counters.items())
            },
            "open_assignments": {
                str(slot): assignment.payload()
                for slot, assignment in sorted(self._open.items())
            },
            "assignment_history_chain_hash": self._history_chain_hash,
            "assignment_count": self._history_count,
        }
        payload["hash"] = canonical_hash(payload)
        return payload

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise OpponentPoolError("opponent pool state must be a mapping")
        if state.get("schema") != POOL_STATE_SCHEMA:
            raise OpponentPoolError("opponent pool state schema mismatch")
        declared = state.get("hash")
        if not isinstance(declared, str) or declared != canonical_hash(
            {key: value for key, value in state.items() if key != "hash"}
        ):
            raise OpponentPoolError("opponent pool state hash is inconsistent")
        if str(state.get("manifest_hash")) != self.manifest_hash:
            raise OpponentPoolError(
                "checkpoint pool manifest differs from the running pool; refusing a forged "
                "exact resume"
            )
        if int(state.get("sampling_seed", -1)) != self.sampling_seed:
            raise OpponentPoolError("pool sampling seed mismatch")
        if str(state.get("sampling_scheme")) != self.sampling_scheme:
            raise OpponentPoolError("pool sampling scheme mismatch")
        if str(state.get("assignment_scheme")) != self.assignment_scheme:
            raise OpponentPoolError("pool assignment scheme mismatch")
        raw_generator = state.get("generator_state")
        if not isinstance(raw_generator, list) or not raw_generator:
            raise OpponentPoolError("pool generator state is missing")
        generator = torch.Generator(device="cpu")
        try:
            generator.set_state(torch.tensor(raw_generator, dtype=torch.uint8))
        except Exception as error:
            raise OpponentPoolError("pool generator state is invalid") from error
        raw_open = state.get("open_assignments")
        if not isinstance(raw_open, Mapping):
            raise OpponentPoolError("pool open assignments must be a mapping")
        restored_open: dict[int, SeatAssignment] = {}
        owner_ids = {entry.owner_id for entry in self.entries}
        for key, raw in raw_open.items():
            slot = int(key)
            if slot < 0:
                raise OpponentPoolError("pool open assignment slot must be non-negative")
            assignment = SeatAssignment.from_payload(
                raw, expected_manifest_hash=self.manifest_hash,
            )
            if assignment.slot != slot:
                raise OpponentPoolError("pool open assignment slot key mismatch")
            if assignment.opponent_owner_id not in owner_ids:
                raise OpponentPoolError("pool assignment references an unknown owner")
            entry = self.entry(assignment.opponent_owner_id)
            if entry.entry_id != assignment.opponent_entry_id:
                raise OpponentPoolError("pool assignment entry id/owner mismatch")
            if entry.policy_version != assignment.opponent_policy_version:
                raise OpponentPoolError("pool assignment policy version mismatch")
            restored_open[slot] = assignment
        raw_counters = state.get("episode_counters")
        if not isinstance(raw_counters, Mapping):
            raise OpponentPoolError("pool episode counters must be a mapping")
        counters = {int(key): int(value) for key, value in raw_counters.items()}
        if any(value < 0 for value in counters.values()):
            raise OpponentPoolError("pool episode counter is negative")
        for slot, assignment in restored_open.items():
            if counters.get(slot, 0) > assignment.episode_index:
                raise OpponentPoolError("pool episode counter is ahead of its open assignment")
        history_count = state.get("assignment_count")
        if isinstance(history_count, bool) or not isinstance(history_count, int) or history_count < 0:
            raise OpponentPoolError("pool assignment count is invalid")
        if history_count < len(restored_open):
            raise OpponentPoolError("pool assignment count is smaller than its open assignments")
        chain_hash = state.get("assignment_history_chain_hash")
        if not isinstance(chain_hash, str) or len(chain_hash) != 64:
            raise OpponentPoolError("pool assignment history chain hash is missing or malformed")
        assignment_counter = int(state.get("assignment_counter", history_count))
        if assignment_counter < history_count:
            raise OpponentPoolError("pool assignment counter is smaller than its history")
        self._generator = generator
        self._draw_index = int(state.get("draw_index", 0))
        self._assignment_counter = assignment_counter
        self._episode_counters = counters
        self._open = restored_open
        self._history_chain_hash = chain_hash
        self._history_count = int(history_count)


def _checkpoint_training_resume_allowed(artifact: V2PolicyArtifact) -> bool:
    """True when the artifact claims to be usable for training resume.

    The evaluation-only schema is necessary but not sufficient: a forged
    evaluation document may also set ``training_resume_allowed=True``.  Both
    signals are checked and either one rejects the pool entry.
    """
    if artifact.schema != EVALUATION_CHECKPOINT_SCHEMA:
        return True
    flag = artifact.metadata.get("training_resume_allowed")
    return flag is True


def _require_compatible_model_features(
    artifact: V2PolicyArtifact, expected: Mapping[str, object],
) -> None:
    model_cfg = getattr(artifact.model, "model_cfg", None)
    if model_cfg is None:
        raise OpponentPoolError("opponent model does not expose its feature configuration")
    for name, expected_value in expected.items():
        actual = getattr(model_cfg, str(name), None)
        if actual is None:
            raise OpponentPoolError(f"opponent model is missing feature field {name!r}")
        if int(actual) != int(expected_value):
            raise OpponentPoolError(
                f"opponent model feature {name!r} differs: {actual!r} != {expected_value!r}"
            )


def _state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Runtime router
# ---------------------------------------------------------------------------
@dataclass
class FrozenPolicySession:
    entry: OpponentPoolEntry
    batch_size: int
    num_players: int
    max_ctx_len: int
    device: torch.device
    cache_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        model = self.entry.artifact.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        decoder = model.decoder.config
        kv_heads = decoder.num_key_value_heads or decoder.num_attention_heads
        head_dim = decoder.hidden_size // decoder.num_attention_heads
        dtype = self.cache_dtype or next(model.parameters()).dtype
        self.kv = KVCache(
            model.decoder_n_layers,
            int(self.batch_size) * int(self.num_players),
            max_ctx_len=int(self.max_ctx_len),
            num_heads=kv_heads,
            head_dim=head_dim,
            value_heads=kv_heads,
            value_head_dim=head_dim,
            dtype=dtype,
            device=self.device,
        )
        self.reset_mask = torch.ones(
            self.kv.batch_size, dtype=torch.bool, device=self.device,
        )

    def reset(self, slots: torch.Tensor | None = None) -> None:
        if slots is None:
            self.kv.reset(self.reset_mask)
            return
        rows = torch.zeros(self.kv.batch_size, dtype=torch.bool, device=self.device)
        for slot in slots.tolist():
            rows[int(slot) * self.num_players:(int(slot) + 1) * self.num_players] = True
        self.kv.reset(rows)

    def sequence_lengths(self) -> torch.Tensor:
        """Per-row cache lengths ``[B*P]`` (all four observers of every slot)."""
        return self.kv.seq_lens.detach().cpu()

    def slot_sequence_lengths(self) -> list[int]:
        lengths = self.sequence_lengths()
        return [
            int(lengths[slot * self.num_players].item())
            for slot in range(int(self.batch_size))
        ]

    def estimated_cache_bytes(self) -> int:
        dtype = self.kv.dtype
        element = torch.empty((), dtype=dtype).element_size()
        return int(
            2 * self.kv.n_layers * self.kv.batch * self.kv.num_heads
            * self.kv.max_ctx_len * self.kv.head_dim * element
        )

    @torch.inference_mode()
    def probabilities(self, step_rollout: TensorDict) -> torch.Tensor:
        output = self.entry.artifact.model(
            rollout=step_rollout, use_cache=True, past_key_values=self.kv,
        )
        probabilities = output.avail_probs.squeeze(1).float()
        if not bool(torch.isfinite(probabilities).all()):
            raise OpponentPoolError("frozen opponent produced a non-finite probability")
        return probabilities

    @torch.inference_mode()
    def populate(self, rollout_buffer: TensorDict) -> None:
        self.reset()
        num_steps = rollout_buffer["num_steps"]
        if not bool(torch.any(num_steps > 0)):
            return
        rollout = TensorDict(
            {
                "num_steps": num_steps,
                "player_ids": rollout_buffer["player_ids"],
                "obs_tokens": rollout_buffer["obs_tokens"],
                "obs_channels": rollout_buffer["obs_channels"],
                "avail_tokens": rollout_buffer["avail_tokens"],
                "avail_weights": rollout_buffer["avail_weights"],
            },
            batch_size=[int(self.batch_size)],
            device=self.device,
        )
        self.entry.artifact.model(
            rollout=rollout, use_cache=True, past_key_values=self.kv, cache_only=True,
        )


class PoolPolicyRouter:
    """Dispatches each acting seat to the policy that owns it.

    The current model is still forwarded on *every* step (by the caller, with
    the sampler's own KV cache) so the training value/GAE clock sees the full
    environment timeline.  A frozen session is forwarded with its own isolated
    KV cache and never receives gradients.
    """

    def __init__(
        self,
        *,
        pool: OpponentPool,
        batch_size: int,
        num_players: int,
        max_ctx_len: int,
        device: torch.device | str,
        max_resident_policies: int = 1,
    ) -> None:
        self.pool = pool
        self.batch_size = int(batch_size)
        self.num_players = int(num_players)
        self.max_ctx_len = int(max_ctx_len)
        self.device = torch.device(device)
        self.max_resident_policies = int(max_resident_policies)
        if self.batch_size <= 0 or self.num_players <= 0 or self.max_ctx_len <= 0:
            raise ValueError("router batch/players/horizon must be positive")
        if self.max_resident_policies <= 0:
            raise ValueError("max_resident_policies must be positive")
        # Independent capacity audit (not just the pool's own entry check): every
        # validated entry is loaded and stays resident because lazy eviction is
        # not implemented.  A router with fewer session slots than pool entries
        # would evict an opponent that is active in the current episodes.
        resident_entries = len(pool.entries)
        if self.max_resident_policies < resident_entries:
            raise OpponentPoolError(
                f"PoolPolicyRouter max_resident_policies={self.max_resident_policies} is smaller "
                f"than the {resident_entries} resident pool entries; lazy eviction is not "
                "implemented, so the session/cache capacity cannot be satisfied"
            )
        self.sessions: dict[str, FrozenPolicySession] = {}
        self._prepared = False
        self.last_step_owner_counts: dict[int, int] = {}
        self.last_step_cache_lengths: dict[str, list[int]] = {}

    # -- lifecycle --------------------------------------------------------
    def prepare(self, *, rollout_buffer: TensorDict | None) -> None:
        for slot in range(self.batch_size):
            if slot not in self.pool._open:
                self.pool.assign_episode(
                    slot=slot, episode_index=self.pool.next_episode_index(slot=slot),
                )
        self._ensure_sessions()
        if rollout_buffer is not None and bool(torch.any(rollout_buffer["num_steps"] > 0)):
            for session in self.sessions.values():
                session.populate(rollout_buffer)
        self._prepared = True

    def _ensure_sessions(self) -> None:
        needed = {
            assignment.opponent_entry_id
            for assignment in self.pool._open.values()
            if assignment.opponent_owner_id != CURRENT_POLICY_OWNER_ID
        }
        for entry_id in needed:
            if entry_id in self.sessions:
                continue
            if len(self.sessions) >= self.max_resident_policies:
                evictable = [
                    key for key in self.sessions
                    if key not in needed
                ]
                if not evictable:
                    raise OpponentPoolError(
                        "max_resident_policies is smaller than the number of opponents active "
                        "in the current episodes; evicting an active opponent would hot-switch "
                        "it mid-episode"
                    )
                victim = evictable[0]
                del self.sessions[victim]
            entry = next(item for item in self.pool.entries if item.entry_id == entry_id)
            self.sessions[entry_id] = FrozenPolicySession(
                entry=entry,
                batch_size=self.batch_size,
                num_players=self.num_players,
                max_ctx_len=self.max_ctx_len,
                device=self.device,
            )

    def on_commit(self, mask: torch.Tensor) -> None:
        mask = mask.to(dtype=torch.bool, device=torch.device("cpu"))
        for slot in mask.nonzero().flatten().tolist():
            slot = int(slot)
            assignment = self.pool.close_episode(slot=slot)
            next_assignment = self.pool.assign_episode(
                slot=slot, episode_index=self.pool.next_episode_index(slot=slot),
            )
            self._ensure_sessions()
            session = self.sessions.get(next_assignment.opponent_entry_id)
            if session is not None:
                session.reset(torch.tensor([slot], dtype=torch.long))

    # -- per-step policy --------------------------------------------------
    @torch.inference_mode()
    def policy_probabilities(
        self,
        *,
        step_rollout: TensorDict,
        parsed_envs_step: Any,
        current_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not self._prepared:
            raise OpponentPoolError("router.prepare() must run before the first rollout step")
        player_ids = parsed_envs_step["player_ids"].to(torch.long)
        if tuple(current_probabilities.shape)[0] != self.batch_size:
            raise OpponentPoolError("current policy probability batch does not match the router")
        owners = torch.full(
            (self.batch_size,), NO_POLICY_OWNER_ID, dtype=torch.long, device=self.device,
        )
        checkpoint_id = torch.full_like(owners, -1)
        policy_version = torch.full_like(owners, -1)
        assignment_id = torch.full_like(owners, -1)
        assigned = torch.full(
            (self.batch_size, self.num_players), NO_POLICY_OWNER_ID,
            dtype=torch.long, device=self.device,
        )
        on_policy = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        probabilities = current_probabilities.clone()
        by_entry: dict[str, list[int]] = {}
        for slot in range(self.batch_size):
            assignment = self.pool.assignment(slot=slot)
            assigned[slot] = torch.tensor(
                assignment.owners_by_seat, dtype=torch.long, device=self.device,
            )
            player = int(player_ids[slot].item())
            if player < 0:
                continue
            owner = int(assignment.owners_by_seat[player])
            owners[slot] = owner
            assignment_id[slot] = int(assignment.assignment_id)
            if owner == CURRENT_POLICY_OWNER_ID:
                continue
            if owner != assignment.opponent_owner_id:
                raise OpponentPoolError("assignment seat owner disagrees with the sampled opponent")
            checkpoint_id[slot] = int(owner) - 1
            policy_version[slot] = int(assignment.opponent_policy_version)
            on_policy[slot] = False
            by_entry.setdefault(assignment.opponent_entry_id, []).append(slot)
        # Every resident opponent cache advances on every environment step,
        # even when the opponent is not acting.  The frozen decision at its
        # next owned step must see the complete visible history, and one slot's
        # action must never advance or stall another slot's cache.
        frozen_probabilities: dict[str, torch.Tensor] = {}
        for entry_id, session in self.sessions.items():
            frozen_probabilities[entry_id] = session.probabilities(step_rollout)
        for entry_id, slots in by_entry.items():
            frozen = frozen_probabilities[entry_id]
            index = torch.tensor(slots, dtype=torch.long, device=self.device)
            probabilities[index] = frozen[index]
        self.last_step_owner_counts = {
            int(owner): int((owners == owner).sum().item())
            for owner in owners.unique().tolist()
        }
        self.last_step_cache_lengths = {
            entry_id: session.slot_sequence_lengths()
            for entry_id, session in self.sessions.items()
        }
        return probabilities, on_policy, {
            "policy_owner": owners,
            "policy_train_mask": on_policy & (owners == CURRENT_POLICY_OWNER_ID),
            "opponent_checkpoint_id": checkpoint_id,
            "opponent_policy_version": policy_version,
            "assignment_id": assignment_id,
            "assigned_seats": assigned,
        }

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": POOL_ROUTER_STATE_SCHEMA,
            "manifest_hash": self.pool.manifest_hash,
            "batch_size": int(self.batch_size),
            "num_players": int(self.num_players),
            "max_ctx_len": int(self.max_ctx_len),
            "current_owner_id": CURRENT_POLICY_OWNER_ID,
            "pool_state": self.pool.state_dict(),
            "resident_entry_ids": sorted(self.sessions),
        }
        payload["state_hash"] = canonical_hash(payload)
        return payload

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != POOL_ROUTER_STATE_SCHEMA:
            raise OpponentPoolError("pool router state schema mismatch")
        declared = state.get("state_hash")
        if declared != canonical_hash(
            {key: value for key, value in state.items() if key != "state_hash"}
        ):
            raise OpponentPoolError("pool router state hash is inconsistent")
        for key, expected in (
            ("batch_size", self.batch_size), ("num_players", self.num_players),
            ("max_ctx_len", self.max_ctx_len), ("current_owner_id", CURRENT_POLICY_OWNER_ID),
        ):
            if int(state.get(key, -1)) != int(expected):
                raise OpponentPoolError(f"pool router state {key} mismatch")
        if str(state.get("manifest_hash")) != self.pool.manifest_hash:
            raise OpponentPoolError("pool router manifest differs from the running pool")
        pool_state = state.get("pool_state")
        if not isinstance(pool_state, Mapping):
            raise OpponentPoolError("pool router state lacks the pool state")
        self.pool.load_state_dict(pool_state)
        self.sessions.clear()
        self._prepared = False

    # -- observability ----------------------------------------------------
    def frozen_parameter_hashes(self) -> dict[str, str]:
        return {
            entry_id: _state_dict_hash(session.entry.artifact.model.state_dict())
            for entry_id, session in self.sessions.items()
        }

    def metrics(self) -> dict[str, object]:
        cache_lengths = {
            entry_id: session.slot_sequence_lengths()
            for entry_id, session in self.sessions.items()
        }
        return {
            "pool/enabled": True,
            "pool/manifest_hash": self.pool.manifest_hash,
            "pool/entry_count": len(self.pool.entries),
            "pool/active_entry_count": len(self.pool.active_entries),
            # Actual resident model instances, not a session count.
            "pool/resident_model_count": int(self.pool.loaded_model_count),
            "pool/resident_session_count": len(self.sessions),
            "pool/resident_entry_ids": sorted(self.sessions),
            "pool/estimated_parameter_bytes": int(self.pool.estimated_parameter_bytes()),
            "pool/estimated_cache_bytes": int(sum(
                session.estimated_cache_bytes() for session in self.sessions.values()
            )),
            "pool/frozen_cache_lengths": cache_lengths,
            "pool/last_step_cache_lengths": {
                key: [int(value) for value in lengths]
                for key, lengths in self.last_step_cache_lengths.items()
            },
            "pool/last_step_owner_counts": {
                str(key): int(value) for key, value in sorted(self.last_step_owner_counts.items())
            },
            "pool/assignment_ids": {
                str(slot): int(assignment.assignment_id)
                for slot, assignment in sorted(self.pool._open.items())
            },
            "pool/opponent_entry_ids": {
                str(slot): assignment.opponent_entry_id
                for slot, assignment in sorted(self.pool._open.items())
            },
        }


__all__ = [
    "FrozenPolicySession", "OpponentPool", "OpponentPoolConfig", "OpponentPoolEntry",
    "OpponentPoolError", "PoolPolicyRouter", "SeatAssignment", "build_manifest",
    "load_manifest", "write_manifest",
]
