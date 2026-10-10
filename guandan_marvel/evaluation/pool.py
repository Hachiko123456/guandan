# SPDX-License-Identifier: MIT
"""Versioned weighted opponent-pool manifest with validated publication."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from .checkpoints import load_legacy_v1_opponent, verify_policy_identity
from .common import atomic_write_json, file_sha256, load_json, payload_hash

OPPONENT_POOL_SCHEMA = "guandan-marvel-opponent-pool-v2"
PRE_IDENTITY_POOL_SCHEMA = "guandan-marvel-opponent-pool-v1"
SAMPLING_SCHEMA = "guandan-marvel-opponent-pool-weighted-sampling-v1"
ENTRY_KINDS = {"champion", "historical_champion", "rule", "legacy_v1"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class OpponentPoolStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self, *, sampling_seed: int = 0) -> dict[str, object]:
        if self.path.exists():
            return self.load(validate_files=True)
        payload: dict[str, object] = {
            "schema": OPPONENT_POOL_SCHEMA,
            "revision": 0,
            "updated_at": _timestamp(),
            "sampling": {"schema": SAMPLING_SCHEMA, "seed": int(sampling_seed)},
            "entries": [],
        }
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload)
        return payload

    def load(self, *, validate_files: bool = True) -> dict[str, object]:
        payload = load_json(self.path)
        if payload.get("schema") not in {OPPONENT_POOL_SCHEMA, PRE_IDENTITY_POOL_SCHEMA} or payload.get("hash") != payload_hash(payload):
            raise ValueError("opponent pool schema/hash mismatch")
        sampling = payload.get("sampling")
        if not isinstance(sampling, Mapping) or sampling.get("schema") != SAMPLING_SCHEMA:
            raise ValueError("opponent pool sampling schema mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("opponent pool entries must be a list")
        ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping) or entry.get("kind") not in ENTRY_KINDS:
                raise ValueError("malformed opponent pool entry")
            entry_id = str(entry.get("id", ""))
            if not entry_id or entry_id in ids:
                raise ValueError("opponent pool entry IDs must be unique")
            ids.add(entry_id)
            weight = entry.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or float(weight) < 0:
                raise ValueError("opponent pool weights must be finite and nonnegative")
            checkpoint = entry.get("checkpoint")
            if validate_files and checkpoint is not None:
                path = Path(str(checkpoint)).resolve()
                if not path.is_file() or file_sha256(path) != entry.get("checkpoint_sha256"):
                    raise ValueError(f"opponent pool checkpoint missing or changed: {path}")
        return payload

    @staticmethod
    def _validate_entry(entry: Mapping[str, object]) -> dict[str, object]:
        normalized = dict(entry)
        kind = normalized.get("kind")
        if kind not in ENTRY_KINDS:
            raise ValueError("unsupported opponent pool entry kind")
        normalized.setdefault("active", True)
        normalized.setdefault("added_at", _timestamp())
        normalized["weight"] = float(normalized.get("weight", 1.0))
        if not np.isfinite(normalized["weight"]) or normalized["weight"] < 0:
            raise ValueError("opponent pool entry weight must be finite and nonnegative")
        checkpoint = normalized.get("checkpoint")
        if kind in {"champion", "historical_champion"}:
            if checkpoint is None:
                raise ValueError("champion pool entry requires checkpoint")
            artifact = verify_policy_identity(normalized.get("policy_identity", {}))
            if Path(str(checkpoint)).resolve() != artifact.path:
                raise ValueError("pool checkpoint path differs from policy identity")
            normalized["checkpoint"] = str(artifact.path)
            normalized["checkpoint_sha256"] = artifact.sha256
            normalized["policy_identity"] = artifact.identity
        elif kind == "legacy_v1":
            if checkpoint is None:
                raise ValueError("legacy pool entry requires checkpoint")
            artifact = load_legacy_v1_opponent(checkpoint)
            expected = normalized.get("legacy_identity")
            if expected is not None and dict(expected) != artifact.identity:
                raise ValueError("legacy pool identity mismatch")
            normalized["checkpoint"] = str(artifact.path)
            normalized["checkpoint_sha256"] = artifact.sha256
            normalized["legacy_identity"] = artifact.identity
        elif checkpoint is not None:
            raise ValueError("rule pool entry must not carry a checkpoint")
        return normalized

    def add_entry(
        self,
        entry: Mapping[str, object],
        *,
        replace_fn: Callable = os.replace,
    ) -> dict[str, object]:
        payload = self.initialize()
        if payload.get("schema") != OPPONENT_POOL_SCHEMA:
            raise ValueError("legacy opponent pool is read-only; create a v2 repair pool")
        normalized = self._validate_entry(entry)
        entries = [dict(item) for item in payload["entries"] if item.get("id") != normalized.get("id")]
        entries.append(normalized)
        payload["entries"] = entries
        payload["revision"] = int(payload["revision"]) + 1
        payload["updated_at"] = _timestamp()
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload, replace_fn=replace_fn)
        return self.load(validate_files=True)

    def recover(self) -> dict[str, object]:
        return self.load(validate_files=True)

    def sample(
        self,
        *,
        sample_seed: int,
        kinds: Iterable[str] = ("champion", "historical_champion"),
    ) -> dict[str, object]:
        payload = self.recover()
        allowed = set(kinds)
        entries = [
            dict(entry) for entry in payload["entries"]
            if entry.get("active", True) and entry.get("kind") in allowed and float(entry.get("weight", 0.0)) > 0
        ]
        if not entries:
            raise ValueError("opponent pool has no active positive-weight entry for the requested kinds")
        weights = np.asarray([float(entry["weight"]) for entry in entries], dtype=np.float64)
        probabilities = weights / weights.sum()
        rng = np.random.default_rng(int(payload["sampling"]["seed"]) + int(sample_seed))
        index = int(rng.choice(len(entries), p=probabilities))
        selected = entries[index]
        selected["sampling_probability"] = float(probabilities[index])
        selected["pool_manifest_hash"] = payload["hash"]
        selected["sample_seed"] = int(sample_seed)
        return selected


__all__ = [
    "ENTRY_KINDS", "OPPONENT_POOL_SCHEMA", "PRE_IDENTITY_POOL_SCHEMA",
    "OpponentPoolStore", "SAMPLING_SCHEMA",
]
