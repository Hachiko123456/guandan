"""Independent formal-evaluation worker for the M09 Unified Pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
from guandan_marvel.evaluation.host import play_game
from guandan_marvel.evaluation.manifests import deals_for, validate_deals_manifest
from guandan_marvel.evaluation.resumable import ResumableEvaluationStore, validate_resumable_plan


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    context = _load(args.context.resolve())
    plan = _load(Path(str(context["plan_path"])))
    validate_resumable_plan(plan)
    if plan["hash"] != context.get("plan_hash"):
        raise ValueError("formal worker context/plan identity mismatch")
    deals_manifest = _load(Path(str(context["deals_manifest_path"])))
    validate_deals_manifest(deals_manifest)
    deal_rows = {str(row["deal_id"]): row for row in deals_for(deals_manifest, "selection")}

    candidate = load_v2_policy(Path(str(context["candidate_checkpoint"])), device=args.device)
    champion = load_v2_policy(Path(str(context["champion_checkpoint"])), device=args.device)
    pool_policy = load_v2_policy(Path(str(context["pool_checkpoint"])), device=args.device)
    legacy = load_legacy_v1_opponent(Path(str(context["legacy_checkpoint"])))
    if candidate.sha256 != plan["candidate"]["checkpoint_sha256"]:
        raise ValueError("formal worker candidate checkpoint differs from plan")

    def executor(task):
        opponent_id = str(task["opponent_id"])
        kind = str(task["opponent_kind"])
        opponent_v2 = champion if opponent_id == "champion" else (pool_policy if opponent_id == "opponent_pool" else None)
        legacy_artifact = legacy if opponent_id == "legacy" else None
        return play_game(
            deal=deal_rows[str(task["deal_id"])],
            rotation=int(task["seat_rotation"]),
            opponent_id=opponent_id,
            opponent_kind=kind,
            candidate=candidate,
            opponent_v2=opponent_v2,
            legacy=legacy_artifact,
            temperature=float(plan["deterministic_policy"]["temperature"]),
            device=args.device,
        )

    fail_after = os.environ.get("GUANDAN_M09_EVAL_FAIL_AFTER")
    if fail_after is not None and args.worker_index != int(os.environ.get("GUANDAN_M09_EVAL_FAIL_WORKER", "0")):
        fail_after = None
    store = ResumableEvaluationStore(Path(str(context["output_dir"])), plan)
    progress = store.run_missing(
        executor,
        worker_index=args.worker_index,
        world_size=args.world_size,
        fail_after=None if fail_after is None else int(fail_after),
    )
    print(json.dumps(progress, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
