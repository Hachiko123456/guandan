# SPDX-License-Identifier: MIT
"""Plan-bound, hash-verified, resumable formal selection evaluation.

The durable record is the unit of recovery.  Runtime timing is intentionally
kept out of formal records so one-worker and deterministic multi-worker runs
produce identical records, summaries, and promotion inputs.  Operational
throughput belongs in heartbeat/telemetry artifacts instead.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from .common import atomic_write_json, canonical_json_bytes, payload_hash
from .stats import summarize_records

RESUMABLE_PLAN_SCHEMA = "guandan-marvel-resumable-evaluation-plan-v1"
FORMAL_BINDING_SCHEMA = "guandan-marvel-formal-game-binding-v1"
FORMAL_SUMMARY_SCHEMA = "guandan-marvel-resumable-evaluation-summary-v1"
FORMAL_RESULT_SCHEMA = "guandan-marvel-resumable-evaluation-result-v1"
PROMOTION_TRANSACTION_SCHEMA_V2 = "guandan-marvel-promotion-transaction-v2"


class IncompleteEvaluationError(RuntimeError):
    pass


class RecordConflictError(ValueError):
    pass


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: object) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def task_key(value: Mapping[str, object]) -> tuple[str, str, int]:
    return str(value["opponent_id"]), str(value["deal_id"]), int(value["seat_rotation"])


def _hex64(value: object, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return text.lower()


def build_resumable_evaluation_plan(
    *,
    base_plan: Mapping[str, object],
    materialized_deals: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Upgrade one immutable M06 selection plan into an enumerated task plan."""
    if base_plan.get("mode") != "selection" or base_plan.get("deal_domain") != "selection":
        raise ValueError("resumable formal evaluation only accepts selection plans")
    promotion_allowed = bool(base_plan.get("promotion_allowed", False))
    if base_plan.get("hash") != payload_hash(base_plan):
        raise ValueError("base evaluation plan hash mismatch")
    matrix = base_plan.get("matrix")
    deal_ids = base_plan.get("deal_ids")
    rotations = base_plan.get("seat_rotations")
    if not isinstance(matrix, list) or not isinstance(deal_ids, list) or rotations != [0, 1, 2, 3]:
        raise ValueError("base evaluation plan matrix/deals/rotations are malformed")
    candidate = base_plan.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("base evaluation plan candidate identity is missing")
    candidate_sha = _hex64(candidate.get("checkpoint_sha256"), "candidate checkpoint")
    backend_identity_hash = _hex64(candidate.get("backend_identity_hash"), "backend identity")
    protocol_version = str(candidate.get("protocol_version", ""))
    if not protocol_version:
        raise ValueError("candidate protocol identity is missing")
    deal_hashes: dict[str, str] = {}
    base_deal_hashes: dict[str, str] = {}
    for row in materialized_deals:
        deal_id = str(row.get("deal_id", ""))
        if not deal_id or deal_id in deal_hashes:
            raise ValueError("materialized deal IDs must be unique and nonempty")
        deal_hashes[deal_id] = str(_hex64(row.get("deal_sha256"), f"deal {deal_id}"))
        base_deal_hashes[deal_id] = str(_hex64(row.get("base_deal_sha256", row.get("deal_sha256")), f"base deal {deal_id}"))
    if set(deal_hashes) != set(map(str, deal_ids)):
        raise ValueError("materialized deal identity set differs from the evaluation plan")
    deterministic_policy = {
        "temperature": float(base_plan.get("temperature", 0.0)),
        "mode": "deterministic_greedy" if float(base_plan.get("temperature", 0.0)) == 0.0 else "seeded_sampling",
        "candidate_protocol_version": protocol_version,
    }
    deterministic_policy_hash = _canonical_hash(deterministic_policy)
    tasks: list[dict[str, object]] = []
    seen_opponents: set[str] = set()
    for matrix_row in matrix:
        if not isinstance(matrix_row, Mapping):
            raise ValueError("evaluation matrix row must be an object")
        opponent_id = str(matrix_row.get("id", ""))
        if not opponent_id or opponent_id in seen_opponents:
            raise ValueError("evaluation matrix opponent IDs must be unique")
        seen_opponents.add(opponent_id)
        opponent_sha = _hex64(
            matrix_row.get("checkpoint_sha256"),
            f"opponent {opponent_id} checkpoint",
            allow_none=True,
        )
        for deal_id in map(str, deal_ids):
            for rotation in rotations:
                body = {
                    "task_index": len(tasks),
                    "opponent_id": opponent_id,
                    "opponent_kind": str(matrix_row.get("kind", "")),
                    "opponent_checkpoint_sha256": opponent_sha,
                    "deal_id": deal_id,
                    "deal_sha256": deal_hashes[deal_id],
                    "base_deal_sha256": base_deal_hashes[deal_id],
                    "seat_rotation": int(rotation),
                }
                tasks.append({**body, "task_hash": _canonical_hash(body)})
    stable: dict[str, object] = {
        "schema": RESUMABLE_PLAN_SCHEMA,
        "version": 1,
        "mode": "selection",
        "evaluation_id": str(base_plan.get("evaluation_id", "")),
        "base_plan_hash": str(base_plan["hash"]),
        "candidate": dict(candidate),
        "current_champion": dict(base_plan.get("current_champion") or {}),
        "legacy_anchor": dict(base_plan.get("legacy_anchor") or {}),
        "opponent_pool_sample": dict(base_plan.get("opponent_pool_sample") or {}),
        "matrix": [dict(row) for row in matrix],
        "deal_ids": list(map(str, deal_ids)),
        "deals_manifest_hash": str(base_plan.get("deals_manifest_hash", "")),
        "deal_dataset_hash": str(base_plan.get("deal_dataset_hash", "")),
        "seat_rotations": [0, 1, 2, 3],
        "backend": str(base_plan.get("backend", "")),
        "backend_identity_hash": backend_identity_hash,
        "protocol_version": protocol_version,
        "deterministic_policy": deterministic_policy,
        "deterministic_policy_hash": deterministic_policy_hash,
        "promotion_policy": dict(base_plan.get("promotion_policy") or {}),
        "promotion_allowed": promotion_allowed,
        "data_isolation": dict(base_plan.get("data_isolation") or {}),
        "tasks": tasks,
        "task_count": len(tasks),
        "final_holdout": False,
    }
    plan = {**stable, "created_at": _timestamp()}
    plan["hash"] = payload_hash(plan)
    validate_resumable_plan(plan)
    return plan


def validate_resumable_plan(plan: Mapping[str, object]) -> None:
    if plan.get("schema") != RESUMABLE_PLAN_SCHEMA or plan.get("hash") != payload_hash(plan):
        raise ValueError("resumable evaluation plan schema/hash mismatch")
    if plan.get("mode") != "selection" or type(plan.get("promotion_allowed")) is not bool or plan.get("final_holdout") is not False:
        raise ValueError("resumable plan must be a selection plan and never final holdout")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != int(plan.get("task_count", -1)):
        raise ValueError("resumable plan task count mismatch")
    seen: set[tuple[str, str, int]] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping) or int(task.get("task_index", -1)) != index:
            raise ValueError("resumable plan task index mismatch")
        body = dict(task)
        declared = body.pop("task_hash", None)
        if declared != _canonical_hash(body):
            raise ValueError("resumable plan task hash mismatch")
        key = task_key(task)
        if key in seen:
            raise ValueError(f"duplicate resumable task: {key}")
        seen.add(key)


def bind_formal_record(
    raw_record: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    task: Mapping[str, object],
) -> dict[str, object]:
    """Normalize one gameplay result into a deterministic durable record."""
    validate_resumable_plan(plan)
    expected_key = task_key(task)
    if task_key(raw_record) != expected_key:
        raise ValueError("executor returned a record for a different task")
    record = dict(raw_record)
    expected_raw = {
        "candidate_checkpoint_sha256": plan["candidate"]["checkpoint_sha256"],
        "opponent_checkpoint_sha256": task.get("opponent_checkpoint_sha256"),
        "base_deal_sha256": str(task["base_deal_sha256"]),
        "backend": str(plan["backend"]),
        "protocol_version": str(plan["protocol_version"]),
    }
    for field, expected_value in expected_raw.items():
        if field in record and record[field] != expected_value:
            raise ValueError(f"executor record identity differs from plan: {field}")
        record[field] = expected_value
    record["schema"] = "guandan-marvel-evaluation-game-v1"
    record["status"] = str(record.get("status", "complete"))
    record["seat_rotation"] = int(task["seat_rotation"])
    record["opponent_id"] = str(task["opponent_id"])
    record["opponent_kind"] = str(task["opponent_kind"])
    record["deal_id"] = str(task["deal_id"])
    # Formal identity excludes wall-clock scheduling noise.  Heartbeats retain
    # real timings outside the game record.
    record["runtime_seconds"] = 0.0
    record["actions_per_second"] = None
    binding_body = {
        "schema": FORMAL_BINDING_SCHEMA,
        "evaluation_plan_hash": str(plan["hash"]),
        "base_plan_hash": str(plan["base_plan_hash"]),
        "task_index": int(task["task_index"]),
        "task_hash": str(task["task_hash"]),
        "candidate_checkpoint_sha256": str(plan["candidate"]["checkpoint_sha256"]),
        "opponent_checkpoint_sha256": task.get("opponent_checkpoint_sha256"),
        "deal_id": str(task["deal_id"]),
        "deal_sha256": str(task["deal_sha256"]),
        "base_deal_sha256": str(task["base_deal_sha256"]),
        "seat_rotation": int(task["seat_rotation"]),
        "backend_identity_hash": str(plan["backend_identity_hash"]),
        "protocol_version": str(plan["protocol_version"]),
        "deterministic_policy_hash": str(plan["deterministic_policy_hash"]),
    }
    record["formal_binding"] = {**binding_body, "hash": _canonical_hash(binding_body)}
    record["formal_record_hash"] = _canonical_hash({key: value for key, value in record.items() if key != "formal_record_hash"})
    validate_formal_record(record, plan=plan)
    return record


def validate_formal_record(record: Mapping[str, object], *, plan: Mapping[str, object]) -> dict[str, object]:
    validate_resumable_plan(plan)
    if record.get("schema") != "guandan-marvel-evaluation-game-v1":
        raise ValueError("formal record game schema mismatch")
    binding = record.get("formal_binding")
    if not isinstance(binding, Mapping) or binding.get("schema") != FORMAL_BINDING_SCHEMA:
        raise ValueError("formal record binding is missing")
    body = dict(binding)
    declared_binding_hash = body.pop("hash", None)
    if declared_binding_hash != _canonical_hash(body):
        raise ValueError("formal record binding hash mismatch")
    index = int(binding.get("task_index", -1))
    tasks = plan["tasks"]
    if index < 0 or index >= len(tasks):
        raise ValueError("formal record references an unknown task index")
    task = tasks[index]
    if task_key(task) != task_key(record):
        raise ValueError("formal record task key differs from plan")
    expected = {
        "evaluation_plan_hash": plan["hash"],
        "base_plan_hash": plan["base_plan_hash"],
        "task_hash": task["task_hash"],
        "candidate_checkpoint_sha256": plan["candidate"]["checkpoint_sha256"],
        "opponent_checkpoint_sha256": task.get("opponent_checkpoint_sha256"),
        "deal_id": task["deal_id"],
        "deal_sha256": task["deal_sha256"],
        "base_deal_sha256": task["base_deal_sha256"],
        "seat_rotation": task["seat_rotation"],
        "backend_identity_hash": plan["backend_identity_hash"],
        "protocol_version": plan["protocol_version"],
        "deterministic_policy_hash": plan["deterministic_policy_hash"],
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"formal record binding differs from plan: {key}")
    if record.get("candidate_checkpoint_sha256") != plan["candidate"]["checkpoint_sha256"]:
        raise ValueError("formal record candidate checkpoint differs from plan")
    if record.get("opponent_checkpoint_sha256") != task.get("opponent_checkpoint_sha256"):
        raise ValueError("formal record opponent checkpoint differs from plan")
    if record.get("base_deal_sha256") != task["base_deal_sha256"]:
        raise ValueError("formal record base-deal hash differs from plan")
    if record.get("backend") != plan["backend"] or record.get("protocol_version") != plan["protocol_version"]:
        raise ValueError("formal record backend/protocol identity differs from plan")
    if record.get("status") != "complete":
        raise ValueError("only complete games are resumable formal records")
    if int(record.get("illegal_count", -1)) < 0 or int(record.get("truncated_count", -1)) < 0:
        raise ValueError("formal record health counters are malformed")
    trace = str(record.get("action_trace_sha256", ""))
    _hex64(trace, "action trace")
    declared_record_hash = record.get("formal_record_hash")
    actual_record_hash = _canonical_hash({key: value for key, value in record.items() if key != "formal_record_hash"})
    if declared_record_hash != actual_record_hash:
        raise ValueError("formal record content hash mismatch")
    return dict(record)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class ResumableEvaluationStore:
    def __init__(self, output_dir: str | Path, plan: Mapping[str, object]):
        validate_resumable_plan(plan)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plan = dict(plan)
        self.plan_path = self.output_dir / "evaluation_plan.json"
        if self.plan_path.is_file():
            existing = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if existing.get("hash") != self.plan["hash"] or existing != self.plan:
                raise ValueError("existing evaluation plan differs from the requested plan")
        else:
            atomic_write_json(self.plan_path, self.plan)
        self._tasks_by_key = {task_key(task): dict(task) for task in self.plan["tasks"]}

    def shard_path(self, worker_index: int) -> Path:
        return self.output_dir / f"games.gpu{int(worker_index)}.jsonl"

    def record_paths(self) -> list[Path]:
        return sorted(
            path for path in self.output_dir.glob("games*.jsonl")
            if path.is_file()
        )

    def load_records(self, paths: Iterable[str | Path] | None = None) -> tuple[dict[tuple[str, str, int], dict[str, object]], dict[str, int]]:
        records: dict[tuple[str, str, int], dict[str, object]] = {}
        duplicates = 0
        lines = 0
        for path_value in self.record_paths() if paths is None else [Path(value) for value in paths]:
            path = Path(path_value)
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    lines += 1
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"corrupted JSONL record {path}:{line_number}: {error}") from error
                    if not isinstance(payload, dict):
                        raise ValueError(f"formal JSONL record must be an object: {path}:{line_number}")
                    validated = validate_formal_record(payload, plan=self.plan)
                    key = task_key(validated)
                    if key not in self._tasks_by_key:
                        raise ValueError(f"formal record references an unknown task: {key}")
                    existing = records.get(key)
                    if existing is not None:
                        if existing != validated:
                            raise RecordConflictError(f"conflicting duplicate formal record: {key}")
                        duplicates += 1
                    else:
                        records[key] = validated
        return records, {"lines": lines, "unique": len(records), "identical_duplicates": duplicates}

    def missing_tasks(self) -> list[dict[str, object]]:
        records, _ = self.load_records()
        return [dict(task) for task in self.plan["tasks"] if task_key(task) not in records]

    def run_missing(
        self,
        executor: Callable[[Mapping[str, object]], Mapping[str, object]],
        *,
        worker_index: int = 0,
        world_size: int = 1,
        fail_after: int | None = None,
    ) -> dict[str, object]:
        if world_size <= 0 or worker_index < 0 or worker_index >= world_size:
            raise ValueError("invalid formal evaluation worker assignment")
        # Independent workers read only their own durable shard while running;
        # the parent validates all shards after every worker exits.  This avoids
        # observing another process between write() and fsync().
        existing, _ = self.load_records(None if world_size == 1 else [self.shard_path(worker_index)])
        assigned = [
            dict(task) for task in self.plan["tasks"]
            if int(task["task_index"]) % world_size == worker_index
        ]
        pending = [task for task in assigned if task_key(task) not in existing]
        completed_now = 0
        try:
            for task in pending:
                if fail_after is not None and completed_now >= int(fail_after):
                    raise RuntimeError("controlled formal evaluation failure")
                raw = executor(task)
                record = bind_formal_record(raw, plan=self.plan, task=task)
                _append_jsonl(self.shard_path(worker_index), record)
                existing[task_key(task)] = record
                completed_now += 1
        except BaseException as error:
            failure = {
                "schema": "guandan-marvel-formal-worker-failure-v1",
                "plan_hash": self.plan["hash"],
                "worker_index": worker_index,
                "world_size": world_size,
                "completed_now": completed_now,
                "assigned": len(assigned),
                "error": f"{type(error).__name__}: {error}",
                "failed_at": _timestamp(),
            }
            failure["hash"] = payload_hash(failure)
            atomic_write_json(self.output_dir / f"worker{worker_index}.failure.json", failure)
            raise
        progress = {
            "schema": "guandan-marvel-formal-worker-progress-v1",
            "plan_hash": self.plan["hash"],
            "worker_index": worker_index,
            "world_size": world_size,
            "assigned": len(assigned),
            "completed_before": len(assigned) - len(pending),
            "completed_now": completed_now,
            "complete": all(task_key(task) in existing for task in assigned),
            "finished_at": _timestamp(),
        }
        progress["hash"] = payload_hash(progress)
        atomic_write_json(self.output_dir / f"worker{worker_index}.progress.json", progress)
        return progress

    def finalize(self) -> dict[str, object]:
        records_by_key, load_stats = self.load_records()
        missing = [task_key(task) for task in self.plan["tasks"] if task_key(task) not in records_by_key]
        failures = sorted(self.output_dir.glob("worker*.failure.json"))
        # A stale failure is harmless only when its worker has subsequently
        # completed every assigned task; otherwise publication remains blocked.
        unresolved_failures = []
        for failure_path in failures:
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            worker = int(failure["worker_index"])
            world = int(failure["world_size"])
            assigned = [task for task in self.plan["tasks"] if int(task["task_index"]) % world == worker]
            if any(task_key(task) not in records_by_key for task in assigned):
                unresolved_failures.append(str(failure_path))
        if missing or unresolved_failures:
            raise IncompleteEvaluationError(
                f"formal evaluation incomplete: missing={len(missing)}, unresolved_worker_failures={unresolved_failures}"
            )
        records = [records_by_key[task_key(task)] for task in self.plan["tasks"]]
        merged_path = self.output_dir / "games.jsonl"
        temporary = merged_path.with_suffix(".jsonl.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, merged_path)
        finally:
            temporary.unlink(missing_ok=True)
        cluster_summary = summarize_records(records, plan=self.plan)
        records_hash = _canonical_hash(records)
        stable_summary = {
            "schema": FORMAL_SUMMARY_SCHEMA,
            "plan_hash": self.plan["hash"],
            "base_plan_hash": self.plan["base_plan_hash"],
            "task_count": len(records),
            "records_hash": records_hash,
            "cluster_summary": cluster_summary,
            "load_stats": load_stats,
            "deterministic": True,
            "final_holdout": False,
        }
        # Duplicate physical lines are an operational resume detail and must not
        # change the formal identity.
        stable_summary["load_stats"] = {"unique": len(records)}
        summary = {**stable_summary, "hash": _canonical_hash(stable_summary)}
        atomic_write_json(self.output_dir / "evaluation_summary.json", summary)
        evaluation_hash = _canonical_hash({
            "plan_hash": self.plan["hash"],
            "records_hash": records_hash,
            "summary_hash": summary["hash"],
        })
        result_body = {
            "schema": FORMAL_RESULT_SCHEMA,
            "status": "complete",
            "complete": True,
            "plan_hash": self.plan["hash"],
            "records_hash": records_hash,
            "summary_hash": summary["hash"],
            "evaluation_hash": evaluation_hash,
            "games": len(records),
            "expected_games": int(self.plan["task_count"]),
            "promotion_allowed": bool(self.plan["promotion_allowed"]),
            "final_holdout": False,
        }
        result = {**result_body, "hash": _canonical_hash(result_body)}
        atomic_write_json(self.output_dir / "evaluation_result.json", result)
        return result


class PromotionTransaction:
    """Replay-safe two-publication transaction for champion and pool state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict[str, object] | None:
        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != PROMOTION_TRANSACTION_SCHEMA_V2 or payload.get("hash") != payload_hash(payload):
            raise ValueError("promotion transaction schema/hash mismatch")
        return payload

    def _write(self, payload: Mapping[str, object]) -> dict[str, object]:
        body = dict(payload)
        body.pop("hash", None)
        result = {**body, "hash": payload_hash(body)}
        atomic_write_json(self.path, result)
        return result

    def execute(
        self,
        *,
        decision_hash: str,
        candidate_checkpoint_sha256: str,
        publish_champion: Callable[[], Mapping[str, object]],
        publish_pool: Callable[[Mapping[str, object]], Mapping[str, object]],
    ) -> dict[str, object]:
        identity = {
            "decision_hash": str(decision_hash),
            "candidate_checkpoint_sha256": str(candidate_checkpoint_sha256),
        }
        transaction = self._load()
        if transaction is None:
            transaction = self._write({
                "schema": PROMOTION_TRANSACTION_SCHEMA_V2,
                **identity,
                "state": "prepared",
                "last_completed_state": "prepared",
                "champion_publication": None,
                "pool_publication": None,
                "error": None,
                "created_at": _timestamp(),
            })
        elif any(transaction.get(key) != value for key, value in identity.items()):
            raise ValueError("promotion transaction identity differs from the requested decision")
        if transaction["state"] == "completed":
            return transaction
        resume_from = str(transaction.get("last_completed_state", transaction["state"]))
        try:
            if resume_from == "prepared":
                champion = dict(publish_champion())
                transaction = self._write({
                    **transaction,
                    "state": "champion_published_pool_pending",
                    "last_completed_state": "champion_published_pool_pending",
                    "champion_publication": champion,
                    "error": None,
                })
                resume_from = "champion_published_pool_pending"
            if resume_from == "champion_published_pool_pending":
                pool = dict(publish_pool(dict(transaction["champion_publication"])))
                transaction = self._write({
                    **transaction,
                    "state": "completed",
                    "last_completed_state": "completed",
                    "pool_publication": pool,
                    "error": None,
                    "completed_at": _timestamp(),
                })
            return transaction
        except BaseException as error:
            self._write({
                **transaction,
                "state": "partial_failure",
                "last_completed_state": resume_from,
                "error": f"{type(error).__name__}: {error}",
                "failed_at": _timestamp(),
            })
            raise


__all__ = [
    "FORMAL_BINDING_SCHEMA", "FORMAL_RESULT_SCHEMA", "FORMAL_SUMMARY_SCHEMA",
    "IncompleteEvaluationError", "PROMOTION_TRANSACTION_SCHEMA_V2", "PromotionTransaction",
    "RESUMABLE_PLAN_SCHEMA", "RecordConflictError", "ResumableEvaluationStore",
    "bind_formal_record", "build_resumable_evaluation_plan", "task_key",
    "validate_formal_record", "validate_resumable_plan",
]
