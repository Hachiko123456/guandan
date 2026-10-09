"""Report dense MARVEL trajectory-buffer memory without allocating the buffers."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guandan_marvel import python_game as game
from guandan_marvel.buffer import trajectory_memory_budget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    default_global = trajectory_memory_budget(
        game.GAME_SPEC, rollout_batch_size=512, collect_batch_size=2048, world_size=1,
    )
    default_two_rank = trajectory_memory_budget(
        game.GAME_SPEC, rollout_batch_size=512, collect_batch_size=2048, world_size=2,
    )
    full_shape_minimal = trajectory_memory_budget(
        game.GAME_SPEC, rollout_batch_size=1, collect_batch_size=1, world_size=1,
    )
    smoke_spec = replace(
        game.GAME_SPEC,
        identifier=f"{game.GAME_IDENTIFIER}-local-smoke-T512",
        max_game_horizon=512,
    )
    local_smoke = trajectory_memory_budget(
        smoke_spec, rollout_batch_size=1, collect_batch_size=1, world_size=1,
    )
    report = {
        "schema": "guandan-marvel-m03-memory-budget-v1",
        "stage": "M02_guandan_adapter_v2_repair",
        "calculation": "pure shape*dtype byte accounting; no dense trajectory allocation",
        "default_protocol": default_global,
        "default_protocol_two_rank_split": default_two_rank,
        "full_shape_minimal_sampler": full_shape_minimal,
        "recommended_local_smoke": {
            "game_spec": {
                "identifier": smoke_spec.identifier,
                "max_game_horizon": smoke_spec.max_game_horizon,
                "note": "test-only reduced horizon; every accepted smoke trajectory must terminate naturally before 512 and this is not default-protocol validation",
            },
            "rollout_batch_size": 1,
            "collect_batch_size": 1,
            "replay_buffer_scale": 1,
            "mini_batches": 1,
            "model": {
                "history_dim": 16,
                "feature_dim": 24,
                "decoder_layers": 1,
                "decoder_heads": 4,
                "state_layers": 1,
                "action_layers": 1,
                "fuse_layers": 1,
            },
            "sampler_budget": local_smoke,
            "required_startup_behavior": "print estimate and reject before allocation when configured per-rank limit is exceeded",
        },
        "later_options_not_implemented_in_m02_repair": [
            "compact token dtypes where vocabulary permits",
            "actual-length trimming after collection",
            "length buckets",
            "chunked/spilled commit storage",
        ],
        "prohibitions": [
            "no legal-action top-N truncation",
            "no reduced-horizon result may be presented as default T=4096 validation",
            "C++ does not automatically solve tensor buffer memory",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "per_trajectory_mib": default_global["per_trajectory_mib"],
        "sampler_gib_global": default_global["sampler_gib_global"],
        "sampler_gib_per_rank_two_way": default_two_rank["sampler_gib_per_rank"],
        "local_smoke_sampler_mib": local_smoke["sampler_bytes_global"] / 2**20,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
