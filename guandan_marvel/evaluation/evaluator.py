# SPDX-License-Identifier: MIT
"""Fixed-matrix selection evaluator and independently audited final runner."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from .checkpoints import load_legacy_v1_opponent, load_v2_policy
from .common import atomic_write_json, canonical_json_bytes, payload_hash
from .host import play_game
from .manifests import FinalHoldoutAudit, deals_for, validate_deals_manifest
from .pool import OpponentPoolStore
from .promotion import ChampionStore, PromotionPolicy, decide_promotion
from .stats import summarize_records

EVALUATION_PLAN_SCHEMA = "guandan-marvel-evaluation-plan-v1"
EVALUATION_RUN_SCHEMA = "guandan-marvel-evaluation-run-v1"
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
) -> dict[str, object]:
    if mode not in {"selection", "final_audit"}:
        raise ValueError("evaluation mode must be selection or final_audit")
    if mode == "selection" and deal_domain != "selection":
        raise ValueError("selection evaluator may only use selection deals")
    if mode == "final_audit" and deal_domain != "final":
        raise ValueError("final audit may only use final deals")
    validate_deals_manifest(deals_manifest)
    plan: dict[str, object] = {
        "schema": EVALUATION_PLAN_SCHEMA,
        "mode": mode,
        "created_at": _timestamp(),
        "deals_manifest_hash": deals_manifest["hash"],
        "deal_domain": deal_domain,
        "deal_ids": [row["deal_id"] for row in deals_for(deals_manifest, deal_domain)],
        "candidate": dict(candidate_identity),
        "current_champion": dict(champion_identity),
        "legacy_anchor": dict(legacy_identity),
        "opponent_pool_sample": dict(pool_sample),
        "matrix": [
            {"id": "champion", "kind": "v2", "checkpoint_sha256": champion_identity["checkpoint_sha256"]},
            {"id": "rule", "kind": "rule"},
            {"id": "random", "kind": "random"},
            {"id": "legacy", "kind": "legacy_v1", "checkpoint_sha256": legacy_identity["checkpoint_sha256"]},
            {"id": "opponent_pool", "kind": "v2", "checkpoint_sha256": pool_sample["checkpoint_sha256"]},
        ],
        "seat_rotations": [0, 1, 2, 3],
        "cluster_unit": "base_deal_with_four_seat_rotations",
        "temperature": float(temperature),
        "backend": "cpp",
        "promotion_allowed": mode == "selection",
    }
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
    ):
        self.output_dir = Path(output_dir)
        self.champion_store = champion_store
        self.opponent_pool = opponent_pool
        self.legacy_checkpoint = Path(legacy_checkpoint)
        self.device = device
        self.temperature = float(temperature)
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self._v2_cache: dict[str, object] = {}

    def _load_v2(self, path: str | Path):
        key = str(Path(path).resolve())
        if key not in self._v2_cache:
            self._v2_cache[key] = load_v2_policy(key, device=self.device)
        return self._v2_cache[key]

    def _execute(
        self,
        *,
        mode: str,
        deals_manifest: Mapping[str, object],
        deal_domain: str,
        candidate_checkpoint: str | Path,
        final_audit: FinalHoldoutAudit | None = None,
    ) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records_path = self.output_dir / "games.jsonl"
        records_path.unlink(missing_ok=True)
        candidate = self._load_v2(candidate_checkpoint)
        champion_manifest = self.champion_store.load()
        champion = self._load_v2(champion_manifest["champion"]["checkpoint_path"])
        legacy = load_legacy_v1_opponent(self.legacy_checkpoint)
        pool_sample = self.opponent_pool.sample(sample_seed=int(candidate.iteration))
        pool_policy = self._load_v2(pool_sample["checkpoint"])
        plan = build_evaluation_plan(
            mode=mode,
            deals_manifest=deals_manifest,
            deal_domain=deal_domain,
            candidate_identity=candidate.identity,
            champion_identity=champion.identity,
            legacy_identity=legacy.identity,
            pool_sample=pool_sample,
            temperature=self.temperature,
        )
        atomic_write_json(self.output_dir / "evaluation_plan.json", plan)
        if mode == "final_audit":
            if final_audit is None:
                raise ValueError("final audit requires an explicit FinalHoldoutAudit")
            audit_payload = final_audit.load()
            if audit_payload.get("locked_candidate_sha256") != candidate.sha256:
                raise ValueError("final audit candidate is not locked")

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
                    except Exception as exc:
                        record = {
                            "schema": "guandan-marvel-evaluation-game-v1",
                            "status": "incomplete",
                            "opponent_id": opponent_id,
                            "opponent_kind": kind,
                            "deal_id": deal["deal_id"],
                            "deal_seed": deal["seed"],
                            "seat_rotation": rotation,
                            "base_deal_sha256": "error",
                            "candidate_team": rotation % 2,
                            "illegal_count": 0,
                            "truncated_count": 1,
                            "error": f"{type(exc).__name__}: {exc}",
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
        summary = summarize_records(records, expected_opponents=FIXED_MATRIX)
        summary["runtime_seconds"] = runtime_seconds
        atomic_write_json(self.output_dir / "cluster_summary.json", summary)
        evaluation_hash = payload_hash({"plan_hash": plan["hash"], "records": records, "summary": summary})

        decision = None
        champion_before_hash = champion_manifest["hash"]
        champion_after_hash = champion_before_hash
        if mode == "selection":
            decision = decide_promotion(
                summary,
                candidate_identity=candidate.identity,
                champion_identity=champion.identity,
                evaluation_hash=evaluation_hash,
                selection_deals_hash=str(deals_manifest["hash"]),
                policy=self.promotion_policy,
            )
            atomic_write_json(self.output_dir / "promotion_decision.json", decision)
            if decision["promote"]:
                promoted = self.champion_store.promote(candidate.identity, decision=decision)
                champion_after_hash = promoted["hash"]
                self.opponent_pool.add_entry({
                    "id": f"champion-generation-{promoted['generation']}",
                    "kind": "champion",
                    "checkpoint": candidate.identity["checkpoint_path"],
                    "checkpoint_sha256": candidate.sha256,
                    "algorithm": candidate.algorithm,
                    "architecture": candidate.architecture,
                    "weight": 1.0,
                    "source_champion_manifest_hash": promoted["hash"],
                })
        else:
            final_audit.record_use(
                candidate_sha256=candidate.sha256,
                evaluation_hash=evaluation_hash,
                purpose="final_audit",
            )

        result: dict[str, object] = {
            "schema": EVALUATION_RUN_SCHEMA,
            "mode": mode,
            "status": "complete" if summary["overall"]["complete"] else "incomplete",
            "candidate": candidate.identity,
            "champion_before_hash": champion_before_hash,
            "champion_after_hash": champion_after_hash,
            "champion_unchanged": champion_before_hash == champion_after_hash,
            "deals_manifest_hash": deals_manifest["hash"],
            "plan_hash": plan["hash"],
            "evaluation_hash": evaluation_hash,
            "games_recorded": len(records),
            "expected_games": len(FIXED_MATRIX) * len(deals_for(deals_manifest, deal_domain)) * 4,
            "summary_path": str((self.output_dir / "cluster_summary.json").resolve()),
            "games_path": str(records_path.resolve()),
            "promotion_decision": decision,
            "error": run_error,
            "completed_at": _timestamp(),
        }
        result["hash"] = payload_hash(result)
        atomic_write_json(self.output_dir / "result.json", result)
        return result

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
    ) -> dict[str, object]:
        return self._execute(
            mode="final_audit",
            deals_manifest=deals_manifest,
            deal_domain="final",
            candidate_checkpoint=candidate_checkpoint,
            final_audit=final_audit,
        )


__all__ = ["EVALUATION_PLAN_SCHEMA", "EVALUATION_RUN_SCHEMA", "EvaluationRunner", "FIXED_MATRIX", "build_evaluation_plan"]
