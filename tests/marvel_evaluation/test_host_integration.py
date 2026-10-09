from __future__ import annotations

from guandan.state import PreviousHandResult
from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
from guandan_marvel.evaluation.host import play_game
from guandan_marvel.evaluation.manifests import build_deals_manifest


def test_real_cpp_host_completes_with_dedicated_legacy_adapter(v2_checkpoint, legacy_checkpoint):
    manifest = build_deals_manifest(
        counts={"training": 1, "selection": 1, "final": 1},
        level_ranks=(5,), leader_seats=(0,),
        previous_results=(PreviousHandResult((1, 4, 3, 2), 0, "HEAD_THIRD"),),
    )
    record = play_game(
        deal=manifest["datasets"]["selection"][0], rotation=0,
        opponent_id="legacy", opponent_kind="legacy_v1",
        candidate=load_v2_policy(v2_checkpoint),
        legacy=load_legacy_v1_opponent(legacy_checkpoint),
        temperature=0.0,
    )
    assert record["status"] == "complete"
    assert record["illegal_count"] == 0 and record["truncated_count"] == 0
    assert record["backend"] == "cpp"
    assert sorted(record["ranking"]) == [1, 2, 3, 4]
