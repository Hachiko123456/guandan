"""Run the M06 local end-to-end selection/champion/opponent-pool exercise."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan.state import PreviousHandResult
from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
from guandan_marvel.evaluation.common import atomic_write_json
from guandan_marvel.evaluation.evaluator import EvaluationRunner
from guandan_marvel.evaluation.manifests import FinalHoldoutAudit, build_deals_manifest
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore, PromotionPolicy

DEFAULT_CANDIDATE = ROOT / "runs/M05_performance_finish/20261009_finish/cpp_checkpoint/cpp-backend-iter000001-v4.pt"
DEFAULT_LEGACY = ROOT / "runs/a07probe/local_fast/ippo/ippo_update_000001_90453d2696194fc2b755820939dcbed2_periodic.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate = load_v2_policy(args.candidate, device=args.device)
    legacy = load_legacy_v1_opponent(args.legacy)

    deals = build_deals_manifest(
        counts={"training": 8, "selection": 1, "final": 1},
        level_ranks=(5,),
        leader_seats=(0,),
        previous_results=(PreviousHandResult((1, 4, 3, 2), 0, "HEAD_THIRD"),),
    )
    atomic_write_json(output / "deals_manifest.json", deals)
    final_audit = FinalHoldoutAudit(output / "final_holdout_audit.json", deals_manifest=deals)
    final_audit.initialize()

    champion_store = ChampionStore(output / "champion_manifest.json")
    champion = champion_store.bootstrap(candidate.identity, reason="M06 pipeline bootstrap; not a strength promotion")
    pool = OpponentPoolStore(output / "opponent_pool_manifest.json")
    pool.initialize(sampling_seed=20261010)
    pool.add_entry({
        "id": "champion-generation-1", "kind": "champion",
        "checkpoint": str(candidate.path), "checkpoint_sha256": candidate.sha256,
        "algorithm": candidate.algorithm, "architecture": candidate.architecture,
        "policy_identity": candidate.identity, "weight": 1.0, "source_champion_manifest_hash": champion["hash"],
    })
    pool.add_entry({"id": "rule-anchor-v1", "kind": "rule", "weight": 0.25, "version": "v2-public-row-heuristic-v1"})
    pool.add_entry({
        "id": "legacy-v1-anchor", "kind": "legacy_v1",
        "checkpoint": str(legacy.path), "checkpoint_sha256": legacy.sha256,
        "algorithm": legacy.algorithm, "legacy_identity": legacy.identity, "weight": 0.25,
    })

    runner = EvaluationRunner(
        output_dir=output / "selection_run",
        champion_store=champion_store,
        opponent_pool=pool,
        legacy_checkpoint=legacy.path,
        device=args.device,
        temperature=0.0,
        promotion_policy=PromotionPolicy(mode="smoke", min_clusters=1),
        provenance_policy="smoke_only",
    )
    result = runner.run_selection(deals_manifest=deals, candidate_checkpoint=candidate.path)
    recovered_champion = ChampionStore(champion_store.path).load()
    recovered_pool = OpponentPoolStore(pool.path).recover()
    recovery = {
        "schema": "guandan-marvel-m06-recovery-verification-v1",
        "champion_hash": recovered_champion["hash"],
        "champion_generation": recovered_champion["generation"],
        "opponent_pool_hash": recovered_pool["hash"],
        "opponent_pool_revision": recovered_pool["revision"],
        "pool_sample": OpponentPoolStore(pool.path).sample(sample_seed=candidate.iteration),
        "final_holdout_state": FinalHoldoutAudit(final_audit.path, deals_manifest=deals).load()["state"],
        "selection_status": result["status"],
        "promotion_status": result["promotion_decision"]["status"],
        "champion_unchanged": result["champion_unchanged"],
    }
    atomic_write_json(output / "recovery_verification.json", recovery)
    print(json.dumps({"result": result, "recovery": recovery}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" and recovery["champion_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
