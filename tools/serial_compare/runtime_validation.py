"""Rules-backed deal validation, before any training or policy allocation."""
from __future__ import annotations

from typing import Mapping

from .common import PlanError, validate_deal_manifest


def validate_runtime_deals(manifests: Mapping[str, Mapping[str, object]], *, reset_native: bool = True) -> dict:
    # Lazy imports preserve the offline driver's dependency-light contract.
    from guandan_marvel.evaluation.manifests import deal_game_config

    rows_checked = 0
    resets = 0
    coverage = {}
    if reset_native:
        from guandan_marvel import cpp_game
    for domain, manifest in manifests.items():
        rows = validate_deal_manifest(manifest, expected_domain=domain)
        for row in rows:
            try:
                config = deal_game_config(row)
                if reset_native:
                    env = cpp_game.EnvBatch(1, configs=[config], auto_reset=False, max_horizon=4)
                    step = env.reset(int(row["seed"]))
                    if int(step.player_ids[0]) not in range(4):
                        raise ValueError("reset produced no active player")
                    resets += 1
            except Exception as error:
                raise PlanError(f"invalid runtime deal {domain}/{row['deal_id']}: {error}") from error
            rows_checked += 1
        coverage[domain] = {
            "rows": len(rows), "level_ranks": sorted({int(r["level_rank"]) for r in rows}),
            "leader_seats": sorted({int(r["leader_seat"]) for r in rows}),
        }
    return {"status": "PASS", "rules_configs_checked": rows_checked,
            "native_resets": resets, "coverage": coverage}
