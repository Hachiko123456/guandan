# SPDX-License-Identifier: MIT
"""Plan-bound selection evaluator and atomically reserved final audit runner."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Mapping
from uuid import uuid4

from .checkpoints import load_legacy_v1_opponent, load_v2_policy
from .common import atomic_write_json, file_sha256, payload_hash
from .host import play_game
from .manifests import FinalHoldoutAudit, dataset_identity, deals_for, validate_deals_manifest
from .pool import OpponentPoolStore
from .promotion import ChampionStore, PromotionPolicy, decide_promotion
from .provenance import verify_data_isolation
from .stats import summarize_records

EVALUATION_PLAN_SCHEMA = "guandan-marvel-evaluation-plan-v2"
EVALUATION_RUN_SCHEMA = "guandan-marvel-evaluation-run-v2"
PROMOTION_TRANSACTION_SCHEMA = "guandan-marvel-promotion-transaction-v1"
FIXED_MATRIX = ("champion", "rule", "random", "legacy", "opponent_pool")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_evaluation_plan(
    *,
    mode: str,
    deals_manifest: Mapping[str, object],
    deal_domain: str,
    candidate_identity: Mapping[str, object],
    champion_identity: Mapping[str, object],
    legacy_identity: Mapping[str, object],
    pool_sample: Mapping[str, object],
    temperature: float,
    data_isolation: Mapping[str, object],
    promotion_policy: PromotionPolicy,
    evaluation_id: str,
    identity_hash_override: str | None = None,
) -> dict[str, object]:
    if mode not in {"selection", "final_audit"}:
        raise ValueError("evaluation mode must be selection or final_audit")
    if mode == "selection" and deal_domain != "selection":
        raise ValueError("selection evaluator may only use selection deals")
    if mode == "final_audit" and deal_domain != "final":
        raise ValueError("final audit may only use final deals")
    validate_deals_manifest(deals_manifest)
    matrix = [
        {"id": "champion", "kind": "v2", "checkpoint_sha256": champion_identity["checkpoint_sha256"]},
        {"id": "rule", "kind": "rule", "checkpoint_sha256": None},
        {"id": "random", "kind": "random", "checkpoint_sha256": None},
        {"id": "legacy", "kind": "legacy_v1", "checkpoint_sha256": legacy_identity["checkpoint_sha256"]},
        {"id": "opponent_pool", "kind": "v2", "checkpoint_sha256": pool_sample["checkpoint_sha256"]},
    ]
    stable = {
        "schema": EVALUATION_PLAN_SCHEMA,
        "mode": mode,
        "evaluation_id": str(evaluation_id),
        "deals_manifest_hash": deals_manifest["hash"],
        "deal_dataset_hash": dataset_identity(deals_manifest, deal_domain),
        "deal_domain": deal_domain,
        "deal_ids": [row["deal_id"] for row in deals_for(deals_manifest, deal_domain)],
        "candidate": dict(candidate_identity),
        "current_champion": dict(champion_identity),
        "legacy_anchor": dict(legacy_identity),
        "opponent_pool_sample": dict(pool_sample),
        "matrix": matrix,
        "seat_rotations": [0, 1, 2, 3],
        "cluster_unit": "base_deal_with_four_seat_rotations",
        "temperature": float(temperature),
        "backend": "cpp",
        "data_isolation": dict(data_isolation),
        "promotion_policy": promotion_policy.to_dict(),
        "promotion_allowed": mode == "selection" and bool(data_isolation.get("formal_selection_eligible")) and promotion_policy.mode == "formal",
    }
    plan = {**stable, "created_at": _timestamp(), "identity_hash": identity_hash_override or payload_hash(stable)}
    plan["hash"] = payload_hash(plan)
    return plan


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class EvaluationRunner:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        champion_store: ChampionStore,
        opponent_pool: OpponentPoolStore,
        legacy_checkpoint: str | Path,
        device: str = "cpu",
        temperature: float = 0.0,
        promotion_policy: PromotionPolicy | None = None,
        provenance_policy: str = "formal",
    ):
        self.output_dir = Path(output_dir)
        self.champion_store = champion_store
        self.opponent_pool = opponent_pool
        self.legacy_checkpoint = Path(legacy_checkpoint)
        self.device = device
        self.temperature = float(temperature)
        self.promotion_policy = promotion_policy or PromotionPolicy()
        if provenance_policy not in {"formal", "smoke_only"}:
            raise ValueError("provenance_policy must be formal or smoke_only")
        self.provenance_policy = provenance_policy
        self._v2_cache: dict[str, tuple[str, object]] = {}

    def _load_v2(self, path: str | Path):
        resolved = Path(path).resolve()
        digest = file_sha256(resolved)
        key = str(resolved)
        cached = self._v2_cache.get(key)
        if cached is None or cached[0] != digest:
            self._v2_cache[key] = (digest, load_v2_policy(resolved, device=self.device))
        return self._v2_cache[key][1]

    def _execute(
        self,
        *,
        mode: str,
        deals_manifest: Mapping[str, object],
        deal_domain: str,
        candidate_checkpoint: str | Path,
        final_audit: FinalHoldoutAudit | None = None,
        evaluation_id: str | None = None,
    ) -> dict[str, object]:
        # Freeze the exact caller-provided manifest after validation so the
        # identity checked here is the same data used for all later reads.
        deals_manifest = deepcopy(dict(deals_manifest))
        validate_deals_manifest(deals_manifest)
        actual_dataset_hash = dataset_identity(deals_manifest, deal_domain)
        actual_generator_version = str(deals_manifest["generator_version"])
        actual_deal_count = len(deals_for(deals_manifest, deal_domain))
        if mode == "final_audit":
            if final_audit is None:
                raise ValueError("final audit requires an explicit FinalHoldoutAudit")
            final_audit.validate_dataset_binding(
                actual_dataset_identity=actual_dataset_hash,
                actual_generator_version=actual_generator_version,
                actual_deal_count=actual_deal_count,
            )
        candidate = self._load_v2(candidate_checkpoint)
        champion_manifest = self.champion_store.load()
        champion = self._load_v2(champion_manifest["champion"]["checkpoint_path"])
        legacy = load_legacy_v1_opponent(self.legacy_checkpoint)
        pool_sample = self.opponent_pool.sample(sample_seed=int(candidate.iteration))
        pool_policy = self._load_v2(pool_sample["checkpoint"])
        unique_artifacts = {item.sha256: item for item in (candidate, champion, pool_policy)}
        run_id = str(evaluation_id or uuid4().hex)
        reservation_identity_hash = payload_hash({
            "schema": "guandan-marvel-final-reservation-plan-identity-v1",
            "mode": mode,
            "evaluation_id": run_id,
            "deal_dataset_hash": dataset_identity(deals_manifest, deal_domain),
            "candidate_sha256": candidate.sha256,
            "champion_sha256": champion.sha256,
            "legacy_sha256": legacy.sha256,
            "pool_checkpoint_sha256": pool_policy.sha256,
            "matrix": list(FIXED_MATRIX),
            "seat_rotations": [0, 1, 2, 3],
            "temperature": self.temperature,
            "promotion_policy": self.promotion_policy.to_dict(),
            "provenance_policy": "formal" if mode == "final_audit" else self.provenance_policy,
        })
        final_reserved = False
        if mode == "final_audit":
            final_audit.reserve(
                evaluation_id=run_id,
                candidate_sha256=candidate.sha256,
                plan_identity_hash=reservation_identity_hash,
                config_hash=payload_hash({
                    "temperature": self.temperature,
                    "matrix": list(FIXED_MATRIX),
                    "promotion_policy": self.promotion_policy.to_dict(),
                    "provenance_policy": "formal",
                }),
                actual_dataset_identity=actual_dataset_hash,
                actual_generator_version=actual_generator_version,
                actual_deal_count=actual_deal_count,
            )
            final_reserved = True

        try:
            # Final reservation precedes deal materialization, scoring, or output mutation.
            isolation = verify_data_isolation(
                artifacts=unique_artifacts.values(),
                deals_manifest=deals_manifest,
                domain=deal_domain,
                policy="formal" if mode == "final_audit" else self.provenance_policy,
            )
            plan = build_evaluation_plan(
                mode=mode,
                deals_manifest=deals_manifest,
                deal_domain=deal_domain,
                candidate_identity=candidate.identity,
                champion_identity=champion.identity,
                legacy_identity=legacy.identity,
                pool_sample=pool_sample,
                temperature=self.temperature,
                data_isolation=isolation,
                promotion_policy=self.promotion_policy,
                evaluation_id=run_id,
                identity_hash_override=reservation_identity_hash if mode == "final_audit" else None,
            )
            # Reservation happens before touching any prior output or scoring a final deal.
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("evaluation_plan.json", "games.jsonl", "cluster_summary.json", "result.json"):
                if mode == "final_audit" and (self.output_dir / name).exists():
                    raise FileExistsError(f"refusing to overwrite existing final artifact: {self.output_dir / name}")
            records_path = self.output_dir / "games.jsonl"
            if mode == "selection":
                records_path.unlink(missing_ok=True)
            atomic_write_json(self.output_dir / "evaluation_plan.json", plan)
            if final_reserved:
                final_audit.mark_running(evaluation_id=run_id)

            matrix = {
                "champion": ("v2", champion, None),
                "rule": ("rule", None, None),
                "random": ("random", None, None),
                "legacy": ("legacy_v1", None, legacy),
                "opponent_pool": ("v2", pool_policy, None),
            }
            records: list[dict[str, object]] = []
            run_error: str | None = None
            started = time.perf_counter()
            for opponent_id in FIXED_MATRIX:
                kind, opponent_v2, legacy_artifact = matrix[opponent_id]
                for deal in deals_for(deals_manifest, deal_domain):
                    for rotation in range(4):
                        try:
                            record = play_game(
                                deal=deal,
                                rotation=rotation,
                                opponent_id=opponent_id,
                                opponent_kind=kind,
                                candidate=candidate,
                                opponent_v2=opponent_v2,
                                legacy=legacy_artifact,
                                temperature=self.temperature,
                                device=self.device,
                            )
                        except Exception as error:
                            expected_opponent_sha = next(item["checkpoint_sha256"] for item in plan["matrix"] if item["id"] == opponent_id)
                            record = {
                                "schema": "guandan-marvel-evaluation-game-v1",
                                "status": "incomplete",
                                "opponent_id": opponent_id,
                                "opponent_kind": kind,
                                "deal_id": deal["deal_id"],
                                "deal_seed": deal["seed"],
                                "seat_rotation": rotation,
                                "base_deal_sha256": "0" * 64,
                                "candidate_team": rotation % 2,
                                "candidate_checkpoint_sha256": candidate.sha256,
                                "opponent_checkpoint_sha256": expected_opponent_sha,
                                "illegal_count": 0,
                                "truncated_count": 1,
                                "error": f"{type(error).__name__}: {error}",
                            }
                            run_error = record["error"]
                        records.append(record)
                        _append_jsonl(records_path, record)
                        if run_error is not None:
                            break
                    if run_error is not None:
                        break
                if run_error is not None:
                    break
            runtime_seconds = time.perf_counter() - started
            summary = summarize_records(records, plan=plan)
            summary["runtime_seconds"] = runtime_seconds
            summary_hash = payload_hash(summary)
            summary["hash"] = summary_hash
            atomic_write_json(self.output_dir / "cluster_summary.json", summary)
            evaluation_hash = payload_hash({"plan_hash": plan["hash"], "records": records, "summary_hash": summary_hash})

            decision = None
            champion_before_hash = champion_manifest["hash"]
            champion_after_hash = champion_before_hash
            transaction = None
            transaction_error = None
            if mode == "selection":
                decision = decide_promotion(
                    summary,
                    candidate_identity=candidate.identity,
                    champion_manifest=champion_manifest,
                    evaluation_hash=evaluation_hash,
                    selection_deals_hash=str(deals_manifest["hash"]),
                    plan_hash=str(plan["hash"]),
                    summary_hash=summary_hash,
                    selection_eligible=bool(isolation["formal_selection_eligible"]),
                    policy=self.promotion_policy,
                )
                atomic_write_json(self.output_dir / "promotion_decision.json", decision)
                if decision["promote"]:
                    transaction_id = f"promotion-{decision['hash'][:16]}"
                    transaction = {
                        "schema": PROMOTION_TRANSACTION_SCHEMA,
                        "transaction_id": transaction_id,
                        "state": "prepared",
                        "decision_hash": decision["hash"],
                        "candidate_checkpoint_sha256": candidate.sha256,
                        "created_at": _timestamp(),
                    }
                    transaction["hash"] = payload_hash(transaction)
                    transaction_path = self.output_dir / "promotion_transaction.json"
                    atomic_write_json(transaction_path, transaction)
                    try:
                        promoted = self.champion_store.promote(
                            candidate.identity,
                            decision=decision,
                            expected_plan_hash=str(plan["hash"]),
                            expected_summary_hash=summary_hash,
                            transaction_id=transaction_id,
                        )
                        champion_after_hash = promoted["hash"]
                        transaction["state"] = "champion_published_pool_pending"
                        transaction["champion_manifest_hash"] = promoted["hash"]
                        transaction["hash"] = payload_hash(transaction)
                        atomic_write_json(transaction_path, transaction)
                        updated_pool = self.opponent_pool.add_entry({
                            "id": f"champion-generation-{promoted['generation']}",
                            "kind": "champion",
                            "checkpoint": str(candidate.path),
                            "checkpoint_sha256": candidate.sha256,
                            "policy_identity": candidate.identity,
                            "algorithm": candidate.algorithm,
                            "architecture": candidate.architecture,
                            "weight": 1.0,
                            "promotion_transaction_id": transaction_id,
                            "source_champion_manifest_hash": promoted["hash"],
                        })
                        synced = self.champion_store.mark_pool_synced(
                            transaction_id=transaction_id,
                            pool_manifest_hash=str(updated_pool["hash"]),
                        )
                        champion_after_hash = synced["hash"]
                        transaction["state"] = "completed"
                        transaction["pool_manifest_hash"] = updated_pool["hash"]
                        transaction["champion_manifest_hash"] = synced["hash"]
                        transaction["completed_at"] = _timestamp()
                        transaction["hash"] = payload_hash(transaction)
                        atomic_write_json(transaction_path, transaction)
                    except Exception as error:
                        transaction_error = f"{type(error).__name__}: {error}"
                        transaction["state"] = "partial_failure"
                        transaction["error"] = transaction_error
                        transaction["failed_at"] = _timestamp()
                        transaction["hash"] = payload_hash(transaction)
                        atomic_write_json(transaction_path, transaction)

            status = "complete" if summary["overall"]["complete"] else "incomplete"
            if transaction_error is not None:
                status = "partial_failure"
            result: dict[str, object] = {
                "schema": EVALUATION_RUN_SCHEMA,
                "evaluation_id": run_id,
                "mode": mode,
                "status": status,
                "candidate": candidate.identity,
                "data_isolation": isolation,
                "champion_before_hash": champion_before_hash,
                "champion_after_hash": champion_after_hash,
                "champion_unchanged": champion_before_hash == champion_after_hash,
                "deals_manifest_hash": deals_manifest["hash"],
                "deal_dataset_hash": dataset_identity(deals_manifest, deal_domain),
                "plan_hash": plan["hash"],
                "plan_identity_hash": plan["identity_hash"],
                "summary_hash": summary_hash,
                "evaluation_hash": evaluation_hash,
                "games_recorded": len(records),
                "expected_games": len(FIXED_MATRIX) * len(deals_for(deals_manifest, deal_domain)) * 4,
                "summary_path": str((self.output_dir / "cluster_summary.json").resolve()),
                "games_path": str(records_path.resolve()),
                "promotion_decision": decision,
                "promotion_transaction": transaction,
                "error": transaction_error or run_error,
                "completed_at": _timestamp(),
            }
            result["hash"] = payload_hash(result)
            atomic_write_json(self.output_dir / "result.json", result)
            if final_reserved:
                final_audit.finish(
                    evaluation_id=run_id,
                    status="consumed" if status == "complete" else "failed",
                    evaluation_hash=evaluation_hash,
                    error=None if status == "complete" else (transaction_error or run_error or "final evaluation incomplete"),
                )
            return result
        except BaseException as error:
            if final_reserved:
                try:
                    final_audit.finish(
                        evaluation_id=run_id,
                        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                except Exception:
                    pass
            raise

    def run_selection(self, *, deals_manifest: Mapping[str, object], candidate_checkpoint: str | Path) -> dict[str, object]:
        return self._execute(
            mode="selection",
            deals_manifest=deals_manifest,
            deal_domain="selection",
            candidate_checkpoint=candidate_checkpoint,
        )

    def run_final_audit(
        self,
        *,
        deals_manifest: Mapping[str, object],
        candidate_checkpoint: str | Path,
        final_audit: FinalHoldoutAudit,
        evaluation_id: str | None = None,
    ) -> dict[str, object]:
        return self._execute(
            mode="final_audit",
            deals_manifest=deals_manifest,
            deal_domain="final",
            candidate_checkpoint=candidate_checkpoint,
            final_audit=final_audit,
            evaluation_id=evaluation_id,
        )


__all__ = [
    "EVALUATION_PLAN_SCHEMA", "EVALUATION_RUN_SCHEMA", "FIXED_MATRIX",
    "PROMOTION_TRANSACTION_SCHEMA", "EvaluationRunner", "build_evaluation_plan",
]
