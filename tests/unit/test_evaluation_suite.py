from guandan.evaluation_suite import _cluster_summary, build_plan


def test_cluster_summary_counts_rotated_deals_separately_from_groups():
    rows = [
        {"opponent": "random", "deal_group": 0, "base_deal_sha256": "a", "winner_team": 0, "candidate_team": 0, "token_steps": 10, "committed_steps": 4},
        {"opponent": "random", "deal_group": 0, "base_deal_sha256": "a", "winner_team": 1, "candidate_team": 1, "token_steps": 12, "committed_steps": 5},
        {"opponent": "random", "deal_group": 1, "base_deal_sha256": "b", "winner_team": 0, "candidate_team": 1, "token_steps": 14, "committed_steps": 6},
    ]
    result = _cluster_summary(rows, opponent="random", seed=3)
    assert result["games"] == 3
    assert result["independent_deal_groups"] == 2
    assert result["candidate_wins"] == 2
    assert 0 <= result["deal_cluster_bootstrap95"][0] <= result["deal_cluster_bootstrap95"][1] <= 1


def test_plan_is_explicit_about_two_gpu_workers():
    plan = build_plan(profile="local_fast", progress_bundle="progress.zip", batch_size=8)
    assert plan["workers"] == {"ippo": "cuda:0", "vrpo": "cuda:1"}
    assert plan["matchups"] == ["self_old_snapshot", "cross_algorithm_update102"]
