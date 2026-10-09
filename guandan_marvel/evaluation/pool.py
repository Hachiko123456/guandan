# SPDX-License-Identifier: MIT
"""Versioned weighted opponent-pool manifest and deterministic recovery."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .common import atomic_write_json, file_sha256, load_json, payload_hash

OPPONENT_POOL_SCHEMA = "guandan-marvel-opponent-pool-v1"
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
        if payload.get("schema") != OPPONENT_POOL_SCHEMA or payload.get("hash") != payload_hash(payload):
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
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or float(weight) < 0:
                raise ValueError("opponent pool weights must be nonnegative")
            checkpoint = entry.get("checkpoint")
            if validate_files and checkpoint is not None:
                path = Path(str(checkpoint)).resolve()
                if not path.is_file() or file_sha256(path) != entry.get("checkpoint_sha256"):
                    raise ValueError(f"opponent pool checkpoint missing or changed: {path}")
        return payload

    def add_entry(self, entry: Mapping[str, object]) -> dict[str, object]:
        payload = self.initialize()
        normalized = dict(entry)
        if normalized.get("kind") not in ENTRY_KINDS:
            raise ValueError("unsupported opponent pool entry kind")
        normalized.setdefault("active", True)
        normalized.setdefault("added_at", _timestamp())
        normalized["weight"] = float(normalized.get("weight", 1.0))
        checkpoint = normalized.get("checkpoint")
        if checkpoint is not None:
            path = Path(str(checkpoint)).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = file_sha256(path)
            if normalized.get("checkpoint_sha256") not in (None, digest):
                raise ValueError("opponent pool entry checkpoint hash mismatch")
            normalized["checkpoint"] = str(path)
            normalized["checkpoint_sha256"] = digest
        entries = [dict(item) for item in payload["entries"] if item.get("id") != normalized.get("id")]
        entries.append(normalized)
        payload["entries"] = entries
        payload["revision"] = int(payload["revision"]) + 1
        payload["updated_at"] = _timestamp()
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload)
        return self.load(validate_files=True)

    def recover(self) -> dict[str, object]:
        """Reload and validate the complete pool needed to reproduce sampling."""
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


__all__ = ["ENTRY_KINDS", "OPPONENT_POOL_SCHEMA", "OpponentPoolStore", "SAMPLING_SCHEMA"]
