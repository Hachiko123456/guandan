# SPDX-License-Identifier: MIT
"""Durable orchestration primitives for the M09 unified train/evaluate pipeline.

The module deliberately keeps the DDP trainer and formal evaluator in separate
processes.  It owns only the durable state machine, exact segment selection,
training commit, failure isolation, and continuation packaging boundaries.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file

PIPELINE_STATE_SCHEMA = "guandan-marvel-m09-unified-pipeline-state-v1"
TRAINING_COMMIT_SCHEMA = "guandan-marvel-m09-training-commit-v1"
CONTINUATION_SCHEMA = "guandan-marvel-m09-unified-continuation-v1"
CONTINUATION_ARCHIVE_NAME = "guandan-marvel-continuation.zip"
CONTINUATION_MANIFEST_NAME = "unified_continuation.json"
PIPELINE_STATE_NAME = "pipeline_state.json"
TRAINING_COMMIT_NAME = "training_commit.json"
AUTO_CONTINUATION_SCHEMA = "guandan-marvel-m09-auto-continuation-v1"

PIPELINE_PHASES = (
    "DISCOVER_INPUTS",
    "PREFLIGHT",
    "TRAINING",
    "TRAINING_CHECKPOINT_COMMITTED",
    "TRAINING_RECOVERY_COMMITTED",
    "EVALUATION_PREPARING",
    "EVALUATING",
    "EVALUATION_COMPLETE",
    "PROMOTION_COMMITTED",
    "PACKAGING",
    "COMPLETE",
    "TRAINING_FAILED_RECOVERABLE",
    "TRAINING_COMPLETE_EVALUATION_FAILED",
    "EVALUATION_INCOMPLETE_RESUMABLE",
    "PROMOTION_REPAIR_REQUIRED",
    "PACKAGING_FAILED_RECOVERABLE",
)

FAILURE_PHASES = frozenset({
    "TRAINING_FAILED_RECOVERABLE",
    "TRAINING_COMPLETE_EVALUATION_FAILED",
    "EVALUATION_INCOMPLETE_RESUMABLE",
    "PROMOTION_REPAIR_REQUIRED",
    "PACKAGING_FAILED_RECOVERABLE",
})


@dataclass(frozen=True, slots=True)
class SegmentTarget:
    segment_id: str
    target_iteration: int
    target_completed_games: int

    def identity(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "target_iteration": self.target_iteration,
            "target_completed_games": self.target_completed_games,
        }


DEFAULT_SEGMENT_SCHEDULE = (
    SegmentTarget("segment0500", 500, 2_000),
    SegmentTarget("segment1000", 1_000, 4_000),
    SegmentTarget("segment1500", 1_500, 6_000),
    SegmentTarget("segment2000", 2_000, 8_000),
)




@dataclass(frozen=True, slots=True)
class AutoContinuationConfig:
    """Versioned optional multi-segment policy; disabled is the safe default."""

    schema: str = AUTO_CONTINUATION_SCHEMA
    enabled: bool = False
    max_segments_per_run: int = 1
    final_holdout_allowed: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "AutoContinuationConfig":
        if value is None:
            return cls()
        if value.get("schema") != AUTO_CONTINUATION_SCHEMA:
            raise ValueError("unsupported auto-continuation schema")
        result = cls(
            enabled=bool(value.get("enabled", False)),
            max_segments_per_run=int(value.get("max_segments_per_run", 1)),
            final_holdout_allowed=bool(value.get("final_holdout_allowed", False)),
        )
        if result.max_segments_per_run <= 0:
            raise ValueError("auto-continuation max_segments_per_run must be positive")
        if result.final_holdout_allowed:
            raise ValueError("M09 unified auto-continuation may never authorize final holdout")
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "enabled": self.enabled,
            "max_segments_per_run": self.max_segments_per_run,
            "final_holdout_allowed": False,
        }


def continuation_gate(
    *,
    config: AutoContinuationConfig,
    segments_completed_this_run: int,
    evaluation: Mapping[str, object],
    promotion: Mapping[str, object],
    health: Mapping[str, object],
    floors: Mapping[str, object],
) -> dict[str, object]:
    reasons: list[str] = []
    manual_review = False
    if not config.enabled:
        reasons.append("automatic continuation is disabled")
    if segments_completed_this_run >= config.max_segments_per_run:
        reasons.append("per-run automatic segment limit reached")
    if evaluation.get("complete") is not True or not evaluation.get("identity_verified", True):
        reasons.append("formal evaluation is incomplete or identity verification failed")
    health_failures = {key: value for key, value in health.items() if int(value) != 0}
    if health_failures:
        reasons.append(f"evaluation health failures: {health_failures}")
    floor_failures = [name for name, passed in floors.items() if passed is not True]
    if floor_failures:
        manual_review = True
        reasons.append(f"rule/random hard floors require manual review: {floor_failures}")
    status = str(promotion.get("status", ""))
    if status not in {"promote", "inconclusive"}:
        reasons.append(f"promotion status blocks continuation: {status or 'missing'}")
    if promotion.get("promote") is True and status != "promote":
        reasons.append("promotion decision is internally inconsistent")
    allowed = not reasons
    return {
        "schema": "guandan-marvel-m09-continuation-gate-v1",
        "allowed": allowed,
        "manual_review": manual_review,
        "promotion_status": status,
        "retain_existing_champion": status == "inconclusive",
        "reasons": reasons,
        "final_holdout_allowed": False,
    }


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    recovery_zip: Path
    recovery_root: Path


@dataclass(frozen=True, slots=True)
class PipelineHooks:
    train: Callable[[SegmentTarget, Path], TrainingArtifacts]
    evaluate: Callable[[SegmentTarget, Mapping[str, object], Path], Mapping[str, object]]
    promote: Callable[[SegmentTarget, Mapping[str, object], Path], Mapping[str, object]]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _self_hash(payload: Mapping[str, object], field: str = "hash") -> str:
    body = dict(payload)
    body.pop(field, None)
    return canonical_hash(body)


def _validate_self_hash(payload: Mapping[str, object], *, schema: str, label: str) -> None:
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported {label} schema")
    if payload.get("hash") != _self_hash(payload):
        raise ValueError(f"{label} hash mismatch")


def select_next_segment(
    iteration: int,
    completed_games: int,
    schedule: Sequence[SegmentTarget] = DEFAULT_SEGMENT_SCHEDULE,
) -> SegmentTarget | None:
    """Return the first absolute target not yet reached, failing on drift.

    Completed games are exact contractual counters, not an approximate signal.
    A recovery that has crossed one half of a target boundary is rejected rather
    than silently selecting a later segment.
    """
    iteration = int(iteration)
    completed_games = int(completed_games)
    if iteration < 0 or completed_games < 0:
        raise ValueError("recovery counters must be nonnegative")
    ordered = tuple(schedule)
    if not ordered:
        return None
    if any(
        left.target_iteration >= right.target_iteration
        or left.target_completed_games >= right.target_completed_games
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("segment schedule must be strictly increasing")
    for target in ordered:
        iteration_reached = iteration >= target.target_iteration
        games_reached = completed_games >= target.target_completed_games
        if iteration_reached != games_reached:
            raise ValueError(
                "recovery counters cross a segment boundary inconsistently: "
                f"iteration={iteration}, completed_games={completed_games}, target={target.identity()}"
            )
        if not iteration_reached:
            return target
    final = ordered[-1]
    if iteration != final.target_iteration or completed_games != final.target_completed_games:
        raise ValueError(
            "recovery exceeds the frozen M09 schedule; automatic continuation is forbidden"
        )
    return None


def read_recovery_progress(recovery_root: str | Path) -> dict[str, object]:
    root = Path(recovery_root).resolve()
    manifest = _load_object(root / "recovery_manifest.json")
    if manifest.get("schema") not in {
        "guandan-marvel-recovery-manifest-v2",
        "guandan-marvel-recovery-manifest-v3",
    }:
        raise ValueError("unsupported recovery manifest schema")
    declared = manifest.get("manifest_hash")
    if declared != canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"}):
        raise ValueError("recovery manifest hash mismatch")
    progress_path = root / "artifacts" / "progress.json"
    progress = _load_object(progress_path)
    iteration = int(progress.get("iteration", manifest.get("iteration", -1)))
    completed_games = int(progress.get("completed_games", -1))
    if iteration != int(manifest.get("iteration", -1)):
        raise ValueError("recovery/progress iteration mismatch")
    return {
        "iteration": iteration,
        "completed_games": completed_games,
        "status": str(manifest.get("status", "unknown")),
        "source_commit": str(manifest.get("source_commit", "")),
        "source_archive_sha256": str(manifest.get("source_archive_sha256", "")),
        "source_content_manifest_hash": str(manifest.get("source_content_manifest_hash", "")),
        "recovery_manifest_hash": str(declared),
        "progress_sha256": sha256_file(progress_path),
    }


def validate_recovery_archive(path: str | Path) -> dict[str, object]:
    archive_path = Path(path).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"recovery ZIP CRC failure: {bad}")
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("recovery ZIP contains duplicate members")
        for name in names:
            pure = Path(name.replace("/", os.sep))
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe recovery ZIP member: {name}")
        if "recovery_manifest.json" not in names:
            raise ValueError("recovery ZIP is missing recovery_manifest.json")
        manifest = json.loads(archive.read("recovery_manifest.json"))
        declared = manifest.get("manifest_hash")
        actual = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
        if declared != actual:
            raise ValueError("recovery ZIP manifest hash mismatch")
        rows = manifest.get("files")
        if not isinstance(rows, list):
            raise ValueError("recovery ZIP manifest files must be a list")
        declared_paths = {str(row.get("path", "")) for row in rows if isinstance(row, Mapping)}
        if declared_paths != set(names) - {"recovery_manifest.json"}:
            raise ValueError("recovery ZIP file inventory mismatch")
        for row in rows:
            data = archive.read(str(row["path"]))
            if len(data) != int(row["bytes"]) or __import__("hashlib").sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"recovery ZIP member mismatch: {row['path']}")
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "manifest_hash": str(manifest["manifest_hash"]),
        "iteration": int(manifest["iteration"]),
        "status": str(manifest.get("status", "unknown")),
        "crc_valid": True,
    }


@dataclass(frozen=True, slots=True)
class RecoveryCandidate:
    kind: str
    path: Path
    manifest: dict[str, object]
    archive_sha256: str | None


def _recovery_manifest_from_zip(path: Path) -> dict[str, object]:
    validation = validate_recovery_archive(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("recovery_manifest.json"))
    if validation["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("recovery archive validation identity changed")
    return manifest


def discover_latest_recovery(
    input_root: str | Path,
    *,
    source_identity: Mapping[str, object],
) -> RecoveryCandidate:
    """Choose the highest compatible recovery, retaining the static iter500 seed.

    Lower-iteration compatible recoveries are expected because the long-lived
    Unified Dataset always carries the verified iter500 bootstrap recovery.
    Conflicting identities at the highest iteration fail closed.
    """
    root = Path(input_root).resolve()
    candidates: list[RecoveryCandidate] = []
    zip_names = {
        "guandan-marvel-recovery.zip",
        "training_recovery.committed.zip",
        "m09_initial_iter0500_recovery.seed.zip",
    }
    for path in sorted(root.rglob("*.zip")):
        if path.name not in zip_names:
            continue
        try:
            manifest = _recovery_manifest_from_zip(path)
        except Exception as error:
            raise ValueError(f"invalid recovery candidate {path}: {error}") from error
        candidates.append(RecoveryCandidate("zip", path.resolve(), manifest, sha256_file(path)))
    for manifest_path in sorted(root.rglob("recovery_manifest.json")):
        recovery_root = manifest_path.parent.resolve()
        manifest = _load_object(manifest_path)
        declared = manifest.get("manifest_hash")
        if declared != canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"}):
            raise ValueError(f"recovery directory manifest hash mismatch: {manifest_path}")
        candidates.append(RecoveryCandidate("directory", recovery_root, manifest, None))
    matches = [
        item for item in candidates
        if all(str(item.manifest.get(key, "")) == str(value) for key, value in source_identity.items())
    ]
    if not matches:
        raise RuntimeError("no recovery matching the unified training source identity was found")
    highest = max(int(item.manifest.get("iteration", -1)) for item in matches)
    finalists = [item for item in matches if int(item.manifest.get("iteration", -1)) == highest]
    identities = {str(item.manifest.get("manifest_hash", "")) for item in finalists}
    if len(identities) != 1:
        raise RuntimeError(
            f"conflicting highest-iteration recovery inputs found: {[str(item.path) for item in finalists]}"
        )
    directories = [item for item in finalists if item.kind == "directory"]
    return directories[0] if directories else finalists[0]


def materialize_recovery_candidate(candidate: RecoveryCandidate, destination: str | Path) -> Path:
    target = Path(destination).resolve()
    if target.exists():
        shutil.rmtree(target)
    if candidate.kind == "directory":
        shutil.copytree(candidate.path, target, symlinks=False)
    else:
        target.mkdir(parents=True)
        with zipfile.ZipFile(candidate.path) as archive:
            for info in archive.infolist():
                member = Path(info.filename.replace("/", os.sep))
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError(f"unsafe recovery ZIP member: {info.filename}")
            archive.extractall(target)
    progress = read_recovery_progress(target)
    if str(progress["recovery_manifest_hash"]) != str(candidate.manifest["manifest_hash"]):
        raise ValueError("materialized recovery identity mismatch")
    return target


class PipelineStateStore:
    """Hash-protected, fsync-before-rename pipeline state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, object] | None:
        if not self.path.is_file():
            return None
        payload = _load_object(self.path)
        _validate_self_hash(payload, schema=PIPELINE_STATE_SCHEMA, label="pipeline state")
        if payload.get("phase") not in PIPELINE_PHASES:
            raise ValueError("unknown pipeline phase")
        return payload

    def transition(
        self,
        phase: str,
        *,
        segment: SegmentTarget | None,
        details: Mapping[str, object] | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        if phase not in PIPELINE_PHASES:
            raise ValueError(f"unknown pipeline phase: {phase}")
        previous = self.load()
        history = list(previous.get("history", [])) if previous is not None else []
        event = {
            "sequence": len(history) + 1,
            "phase": phase,
            "at": _timestamp(),
            "error": error,
        }
        history.append(event)
        body: dict[str, object] = {
            "schema": PIPELINE_STATE_SCHEMA,
            "version": 1,
            "phase": phase,
            "segment": None if segment is None else segment.identity(),
            "details": dict(details or {}),
            "error": error,
            "updated_at": event["at"],
            "history": history,
        }
        payload = {**body, "hash": canonical_hash(body)}
        atomic_json(self.path, payload)
        return payload


def discover_previous_continuation(
    input_root: str | Path,
    *,
    expected_control_hash: str,
    expected_source_identity: Mapping[str, object],
) -> tuple[Path, dict[str, object]] | None:
    root = Path(input_root).resolve()
    candidates = sorted(root.rglob(CONTINUATION_MANIFEST_NAME))
    validated: list[tuple[Path, dict[str, object]]] = []
    foreign: list[str] = []
    for path in candidates:
        payload = _load_object(path)
        try:
            _validate_self_hash(payload, schema=CONTINUATION_SCHEMA, label="continuation manifest")
        except ValueError:
            foreign.append(str(path))
            continue
        if payload.get("control_hash") != expected_control_hash:
            foreign.append(str(path))
            continue
        source = payload.get("source_identity")
        expected = dict(expected_source_identity)
        if not isinstance(source, Mapping) or any(source.get(key) != value for key, value in expected.items()):
            foreign.append(str(path))
            continue
        validated.append((path, payload))
    if len(validated) > 1:
        identities = {str(item[1].get("hash", "")) for item in validated}
        if len(identities) > 1:
            raise RuntimeError(f"multiple different compatible continuation manifests found: {[str(item[0]) for item in validated]}")
        validated.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    if not validated:
        if candidates and foreign:
            raise RuntimeError(f"continuation inputs were found but none match this unified control: {foreign}")
        return None
    return validated[0]


def commit_training_artifacts(
    *,
    target: SegmentTarget,
    artifacts: TrainingArtifacts,
    committed_root: str | Path,
    source_identity: Mapping[str, object],
    control_hash: str,
) -> dict[str, object]:
    recovery_root = artifacts.recovery_root.resolve()
    progress = read_recovery_progress(recovery_root)
    archive = validate_recovery_archive(artifacts.recovery_zip)
    if progress["iteration"] != target.target_iteration or progress["completed_games"] != target.target_completed_games:
        raise ValueError(
            "training did not reach the exact segment budget: "
            f"actual=({progress['iteration']}, {progress['completed_games']}), target={target.identity()}"
        )
    if archive["iteration"] != target.target_iteration or archive["status"] != "complete":
        raise ValueError("recovery archive is not a complete target-iteration artifact")
    for key in ("source_commit", "source_archive_sha256", "source_content_manifest_hash"):
        if str(progress.get(key)) != str(source_identity.get(key)):
            raise ValueError(f"recovery source identity mismatch: {key}")
    checkpoint_manifest = _load_object(recovery_root / str(_load_object(recovery_root / "recovery_manifest.json")["checkpoint"]))
    ranks = checkpoint_manifest.get("rank_shards")
    if not isinstance(ranks, list) or sorted(int(row.get("rank", -1)) for row in ranks) != [0, 1]:
        raise ValueError("training commit requires complete rank 0/rank 1 checkpoint shards")
    destination_root = Path(committed_root).resolve() / target.segment_id
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "training_recovery.committed.zip"
    temporary = destination.with_suffix(".zip.tmp")
    try:
        with artifacts.recovery_zip.open("rb") as source, temporary.open("wb") as stream:
            shutil.copyfileobj(source, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256_file(temporary) != archive["sha256"]:
            raise ValueError("immutable recovery copy hash mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    body: dict[str, object] = {
        "schema": TRAINING_COMMIT_SCHEMA,
        "segment": target.identity(),
        "source_identity": dict(source_identity),
        "control_hash": str(control_hash),
        "recovery": {
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "manifest_hash": archive["manifest_hash"],
            "crc_valid": True,
        },
        "checkpoint_manifest_hash": checkpoint_manifest.get("manifest_hash"),
        "rank_shards": [
            {
                "rank": int(row["rank"]),
                "shard": str(row["shard"]),
                "sha256": str(row["shard_sha256"]),
            }
            for row in sorted(ranks, key=lambda row: int(row["rank"]))
        ],
        "completed_games": progress["completed_games"],
        "committed_at": _timestamp(),
        "immutable": True,
    }
    payload = {**body, "hash": canonical_hash(body)}
    atomic_json(destination_root / TRAINING_COMMIT_NAME, payload)
    return payload


def load_training_commit(path: str | Path) -> dict[str, object]:
    payload = _load_object(path)
    _validate_self_hash(payload, schema=TRAINING_COMMIT_SCHEMA, label="training commit")
    recovery = payload.get("recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError("training commit recovery identity is missing")
    archive = Path(path).resolve().parent / str(recovery["path"])
    if not archive.is_file() or sha256_file(archive) != recovery.get("sha256"):
        raise ValueError("committed recovery copy is missing or changed")
    return payload


def create_continuation_bundle(
    *,
    output_root: str | Path,
    training_commit_path: str | Path,
    pipeline_state_path: str | Path,
    champion_manifest_path: str | Path,
    opponent_pool_manifest_path: str | Path,
    promotion_transaction_path: str | Path | None,
    evaluation_paths: Sequence[str | Path],
    additional_files: Mapping[str, str | Path] | None = None,
    next_target: SegmentTarget | None,
    source_identity: Mapping[str, object],
    control_hash: str,
    plan_hash: str,
) -> dict[str, object]:
    root = Path(output_root).resolve()
    bundle_root = root / "continuation"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)
    inputs = {
        "training/training_commit.json": Path(training_commit_path),
        "pipeline/pipeline_state.json": Path(pipeline_state_path),
        "champion/champion_manifest.json": Path(champion_manifest_path),
        "champion/opponent_pool_manifest.json": Path(opponent_pool_manifest_path),
    }
    if promotion_transaction_path is not None and Path(promotion_transaction_path).is_file():
        inputs["champion/promotion_transaction.json"] = Path(promotion_transaction_path)
    for index, path in enumerate(evaluation_paths):
        source = Path(path)
        inputs[f"evaluation/{index:03d}-{source.name}"] = source
    for relative, path in dict(additional_files or {}).items():
        normalized = Path(relative)
        if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
            raise ValueError(f"unsafe continuation additional path: {relative}")
        if normalized.as_posix() in inputs:
            raise ValueError(f"duplicate continuation path: {relative}")
        inputs[normalized.as_posix()] = Path(path)
    training_commit = load_training_commit(inputs["training/training_commit.json"])
    committed_recovery = Path(training_commit_path).resolve().parent / str(training_commit["recovery"]["path"])
    inputs["training/training_recovery.committed.zip"] = committed_recovery
    for relative, source in inputs.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    inventory = relative_file_manifest(bundle_root, [path for path in bundle_root.rglob("*") if path.is_file()])
    body: dict[str, object] = {
        "schema": CONTINUATION_SCHEMA,
        "version": 1,
        "source_identity": dict(source_identity),
        "control_hash": str(control_hash),
        "plan_hash": str(plan_hash),
        "training_commit_hash": training_commit["hash"],
        "next_target": None if next_target is None else next_target.identity(),
        "final_holdout_generated": False,
        "files": inventory,
        "created_at": _timestamp(),
    }
    manifest = {**body, "hash": canonical_hash(body)}
    atomic_json(bundle_root / CONTINUATION_MANIFEST_NAME, manifest)
    archive_path = root / CONTINUATION_ARCHIVE_NAME
    temporary = archive_path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(bundle_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle_root).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"continuation ZIP CRC failure: {bad}")
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "manifest": manifest,
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
    }


class UnifiedSegmentPipeline:
    """Execute one segment with durable training-before-evaluation semantics."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        source_identity: Mapping[str, object],
        control_hash: str,
        hooks: PipelineHooks,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.source_identity = dict(source_identity)
        self.control_hash = str(control_hash)
        self.hooks = hooks
        self.state = PipelineStateStore(self.output_root / PIPELINE_STATE_NAME)
        self.committed_root = self.output_root / "committed_training"

    def _existing_commit(self, target: SegmentTarget) -> dict[str, object] | None:
        path = self.committed_root / target.segment_id / TRAINING_COMMIT_NAME
        if not path.is_file():
            return None
        commit = load_training_commit(path)
        if commit.get("segment") != target.identity():
            raise ValueError("existing training commit target mismatch")
        if commit.get("source_identity") != self.source_identity or commit.get("control_hash") != self.control_hash:
            raise ValueError("existing training commit identity mismatch")
        return commit

    def run(
        self,
        *,
        iteration: int,
        completed_games: int,
        pending_target: SegmentTarget | None = None,
        target_override: SegmentTarget | None = None,
    ) -> dict[str, object]:
        self.state.transition("DISCOVER_INPUTS", segment=None, details={"iteration": iteration, "completed_games": completed_games})
        if pending_target is not None and target_override is not None:
            raise ValueError("pending_target and target_override are mutually exclusive")
        target = pending_target or target_override or select_next_segment(iteration, completed_games)
        if pending_target is not None:
            if iteration < pending_target.target_iteration or completed_games < pending_target.target_completed_games:
                raise ValueError("pending evaluation target is newer than the selected recovery")
            if self._existing_commit(pending_target) is None:
                raise ValueError("pending evaluation target has no durable training commit")
        if target_override is not None:
            if iteration >= target_override.target_iteration or completed_games >= target_override.target_completed_games:
                raise ValueError("target_override must advance both recovery counters")
        if target is None:
            return self.state.transition("COMPLETE", segment=None, details={"reason": "schedule_complete"})
        self.state.transition("PREFLIGHT", segment=target, details={"source_identity": self.source_identity})
        commit = self._existing_commit(target)
        if commit is None:
            self.state.transition("TRAINING", segment=target)
            try:
                artifacts = self.hooks.train(target, self.output_root)
                self.state.transition("TRAINING_CHECKPOINT_COMMITTED", segment=target)
                commit = commit_training_artifacts(
                    target=target,
                    artifacts=artifacts,
                    committed_root=self.committed_root,
                    source_identity=self.source_identity,
                    control_hash=self.control_hash,
                )
                self.state.transition("TRAINING_RECOVERY_COMMITTED", segment=target, details={"training_commit_hash": commit["hash"]})
            except BaseException as error:
                self.state.transition(
                    "TRAINING_FAILED_RECOVERABLE",
                    segment=target,
                    error=f"{type(error).__name__}: {error}",
                )
                raise
        else:
            self.state.transition(
                "TRAINING_RECOVERY_COMMITTED",
                segment=target,
                details={"training_commit_hash": commit["hash"], "training_skipped": True},
            )
        self.state.transition("EVALUATION_PREPARING", segment=target, details={"training_commit_hash": commit["hash"]})
        try:
            self.state.transition("EVALUATING", segment=target)
            evaluation = dict(self.hooks.evaluate(target, commit, self.output_root))
            if evaluation.get("complete") is not True:
                self.state.transition(
                    "EVALUATION_INCOMPLETE_RESUMABLE",
                    segment=target,
                    details={"training_commit_hash": commit["hash"], "evaluation": evaluation},
                )
                return self.state.load() or {}
            self.state.transition("EVALUATION_COMPLETE", segment=target, details={"evaluation": evaluation})
        except BaseException as error:
            self.state.transition(
                "TRAINING_COMPLETE_EVALUATION_FAILED",
                segment=target,
                details={"training_commit_hash": commit["hash"]},
                error=f"{type(error).__name__}: {error}",
            )
            raise
        try:
            promotion = dict(self.hooks.promote(target, evaluation, self.output_root))
        except BaseException as error:
            self.state.transition(
                "PROMOTION_REPAIR_REQUIRED",
                segment=target,
                details={"training_commit_hash": commit["hash"], "evaluation": evaluation},
                error=f"{type(error).__name__}: {error}",
            )
            raise
        return self.state.transition(
            "PROMOTION_COMMITTED",
            segment=target,
            details={"training_commit_hash": commit["hash"], "evaluation": evaluation, "promotion": promotion},
        )


__all__ = [
    "AUTO_CONTINUATION_SCHEMA", "AutoContinuationConfig", "CONTINUATION_ARCHIVE_NAME", "CONTINUATION_MANIFEST_NAME", "CONTINUATION_SCHEMA",
    "DEFAULT_SEGMENT_SCHEDULE", "FAILURE_PHASES", "PIPELINE_PHASES", "PIPELINE_STATE_NAME",
    "PIPELINE_STATE_SCHEMA", "PipelineHooks", "PipelineStateStore", "RecoveryCandidate", "SegmentTarget",
    "TRAINING_COMMIT_NAME", "TRAINING_COMMIT_SCHEMA", "TrainingArtifacts", "UnifiedSegmentPipeline",
    "commit_training_artifacts", "continuation_gate", "create_continuation_bundle", "discover_previous_continuation",
    "discover_latest_recovery", "load_training_commit", "materialize_recovery_candidate",
    "read_recovery_progress", "select_next_segment", "validate_recovery_archive",
]
