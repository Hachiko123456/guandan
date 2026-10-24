from __future__ import annotations

import math

import pytest

from guandan_marvel.evaluation.stats import clustered_mean_ci95, summarize_records


def _plan(deals=("deal0",), opponents=("champion",)):
    matrix=[]
    for opponent in opponents:
        sha = None if opponent in {"rule","random"} else f"{len(matrix)+2:064x}"
        matrix.append({"id":opponent,"kind":"v2","checkpoint_sha256":sha})
    return {
        "candidate":{"checkpoint_sha256":"1"*64},
        "matrix":matrix,"deal_ids":list(deals),"seat_rotations":[0,1,2,3],
    }


def _record(opponent, deal, rotation, win, *, candidate_sha="1"*64, opponent_sha=None):
    candidate_team=rotation%2
    winner=candidate_team if win else 1-candidate_team
    ranking=[0]*4;ranking[winner]=1;ranking[1-winner]=2;ranking[winner+2]=3;ranking[3-winner]=4
    return {
        "status":"complete","opponent_id":opponent,"deal_id":deal,"seat_rotation":rotation,
        "base_deal_sha256":f"{abs(hash(deal))%(1<<256):064x}","candidate_team":candidate_team,
        "winner_team":winner,"candidate_win":bool(win),"candidate_team_reward":1.0 if win else -1.0,
        "outcome_class":"HEAD_LAST","illegal_count":0,"truncated_count":0,"runtime_seconds":1.0,
        "token_steps":10,"committed_steps":4,"candidate_checkpoint_sha256":candidate_sha,
        "opponent_checkpoint_sha256":opponent_sha,
    }


def _records(plan, wins_by_deal=None):
    rows=[];wins_by_deal=wins_by_deal or {}
    for matrix in plan["matrix"]:
        for deal in plan["deal_ids"]:
            for rotation in plan["seat_rotations"]:
                rows.append(_record(matrix["id"],deal,rotation,wins_by_deal.get(deal,True),opponent_sha=matrix["checkpoint_sha256"]))
    return rows


def test_cluster_ci_is_bounded_and_non_degenerate_for_small_extremes():
    two_wins=clustered_mean_ci95([1.0,1.0])
    two_losses=clustered_mean_ci95([0.0,0.0])
    assert two_wins["low"] < 1.0 and two_wins["high"] == 1.0
    assert two_losses["low"] == 0.0 and two_losses["high"] > 0.0
    assert two_wins["method"] == "bounded-empirical-Bernstein-95"


def test_single_cluster_is_inconclusive():
    ci=clustered_mean_ci95([0.5])
    assert ci["low"] is None and ci["high"] is None and ci["conclusive"] is False


def test_mixed_clusters_and_sufficient_stable_advantage():
    mixed=clustered_mean_ci95([0.0,0.5,0.5,1.0])
    assert mixed["mean"]==0.5 and 0.0<=mixed["low"]<=mixed["high"]<=1.0
    stable=clustered_mean_ci95([0.75]*64)
    assert stable["low"]>0.52 and stable["high"]>stable["mean"]


def test_plan_complete_and_overall_ci_is_not_misreported_across_opponents():
    plan=_plan(deals=("d0","d1"),opponents=("champion","rule"))
    summary=summarize_records(_records(plan),plan=plan)
    assert summary["overall"]["complete"] is True
    assert summary["overall"]["eval/clustered_ci95_low"] is None
    assert summary["overall"]["clustered_ci95"]["method"] == "not-reported-across-correlated-opponents"
    assert summary["per_opponent"]["champion"]["eval/cluster_count"]==2


def test_missing_rotation_and_missing_entire_deal_are_incomplete():
    plan=_plan(deals=("d0","d1"),opponents=("champion",))
    records=_records(plan)
    missing_rotation=records[:-1]
    summary=summarize_records(missing_rotation,plan=plan)
    assert summary["overall"]["complete"] is False
    assert any("missing planned records" in error for error in summary["seat_rotation_coverage"]["errors"])
    missing_deal=[row for row in records if row["deal_id"]=="d0"]
    summary=summarize_records(missing_deal,plan=plan)
    assert summary["overall"]["complete"] is False
    assert any("d1" in error for error in summary["seat_rotation_coverage"]["errors"])


def test_duplicate_extra_and_checkpoint_identity_records_are_rejected():
    plan=_plan(opponents=("champion",))
    records=_records(plan)
    summary=summarize_records(records+[dict(records[0])],plan=plan)
    assert summary["overall"]["complete"] is False
    extra=dict(records[0]);extra["deal_id"]="extra"
    summary=summarize_records(records+[extra],plan=plan)
    assert summary["overall"]["complete"] is False
    wrong=[dict(row) for row in records];wrong[0]["candidate_checkpoint_sha256"]="9"*64
    summary=summarize_records(wrong,plan=plan)
    assert summary["overall"]["complete"] is False


def test_nonfinite_and_out_of_range_records_never_form_valid_summary():
    plan=_plan(opponents=("champion",));records=_records(plan)
    malformed=[dict(row) for row in records];malformed[0]["runtime_seconds"]=float("nan")
    summary=summarize_records(malformed,plan=plan)
    assert summary["overall"]["complete"] is False
    assert any("finite" in error for error in summary["seat_rotation_coverage"]["errors"])
    malformed=[dict(row) for row in records];malformed[0]["candidate_team_reward"]=4.0
    summary=summarize_records(malformed,plan=plan)
    assert summary["overall"]["complete"] is False
