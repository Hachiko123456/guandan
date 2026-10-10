from __future__ import annotations

import copy
from pathlib import Path

import pytest

from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.evaluation.common import payload_hash
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore, PromotionPolicy, decide_promotion


def _summary(*, complete=True, low=0.70, win=0.85, clusters=64, illegal=0, truncated=0):
    entries = {}
    for name in ("champion", "rule", "random", "legacy", "opponent_pool"):
        entries[name] = {
            "complete": complete, "eval/win_rate": win,
            "eval/clustered_ci95_low": low, "eval/clustered_ci95_high": 0.98,
            "eval/cluster_count": clusters,
            "eval/illegal_count": illegal, "eval/truncated_count": truncated,
        }
    return {"per_opponent": entries, "seat_rotation_coverage": {"complete": complete}}


def _decision(store, candidate, *, summary=None, eligible=True, policy=None):
    return decide_promotion(
        summary or _summary(), candidate_identity=candidate.identity,
        champion_manifest=store.load(), evaluation_hash="e"*64,
        selection_deals_hash="d"*64, plan_hash="p"*64, summary_hash="s"*64,
        selection_eligible=eligible, policy=policy or PromotionPolicy(),
    )


def test_incomplete_or_smoke_evaluation_never_promotes(v2_checkpoint, second_v2_checkpoint, tmp_path):
    candidate=load_v2_policy(v2_checkpoint);champion=load_v2_policy(second_v2_checkpoint)
    store=ChampionStore(tmp_path/"champion.json");store.bootstrap(champion.identity)
    assert _decision(store,candidate,summary=_summary(complete=False))["status"]=="inconclusive"
    assert _decision(store,candidate,eligible=False)["status"]=="inconclusive"
    smoke=PromotionPolicy(mode="smoke",min_clusters=1)
    assert _decision(store,candidate,policy=smoke)["status"]=="inconclusive"


def test_point_estimate_501_and_two_all_win_clusters_do_not_promote(v2_checkpoint, second_v2_checkpoint, tmp_path):
    candidate=load_v2_policy(v2_checkpoint);champion=load_v2_policy(second_v2_checkpoint)
    store=ChampionStore(tmp_path/"champion.json");store.bootstrap(champion.identity)
    assert _decision(store,candidate,summary=_summary(low=0.49,win=0.501))["status"]=="inconclusive"
    assert _decision(store,candidate,summary=_summary(low=0.0,win=1.0,clusters=2))["status"]=="inconclusive"
    tiny=PromotionPolicy(mode="smoke",min_clusters=2)
    assert _decision(store,candidate,summary=_summary(low=0.0,win=1.0,clusters=2),policy=tiny)["status"]=="inconclusive"


def test_approved_a_passed_b_is_rejected_and_old_champion_unchanged(v2_checkpoint, second_v2_checkpoint, v2_checkpoint_factory, tmp_path):
    candidate_a=load_v2_policy(v2_checkpoint);current=load_v2_policy(second_v2_checkpoint)
    candidate_b=load_v2_policy(v2_checkpoint_factory("candidate-c.pt",seed=25,mutate=0.02))
    store=ChampionStore(tmp_path/"champion.json");old=store.bootstrap(current.identity)
    decision=_decision(store,candidate_a)
    with pytest.raises(ValueError,match="differs from the approved"):
        store.promote(candidate_b.identity,decision=decision,expected_plan_hash="p"*64,expected_summary_hash="s"*64,transaction_id="tx")
    assert store.load()["hash"]==old["hash"]


def test_tampered_decision_stale_champion_and_replaced_checkpoint_rejected(v2_checkpoint, second_v2_checkpoint, tmp_path):
    candidate=load_v2_policy(v2_checkpoint);current=load_v2_policy(second_v2_checkpoint)
    store=ChampionStore(tmp_path/"champion.json");store.bootstrap(current.identity)
    decision=_decision(store,candidate)
    tampered=copy.deepcopy(decision);tampered["selection_deals_hash"]="x"*64
    with pytest.raises(ValueError,match="schema/hash"):
        store.promote(candidate.identity,decision=tampered,expected_plan_hash="p"*64,expected_summary_hash="s"*64,transaction_id="tx")
    stale=store.load();stale["generation"]+=1;stale["hash"]=payload_hash(stale)
    store.path.write_text(__import__('json').dumps(stale),encoding='utf-8')
    with pytest.raises(ValueError,match="stale"):
        store.promote(candidate.identity,decision=decision,expected_plan_hash="p"*64,expected_summary_hash="s"*64,transaction_id="tx")


def test_checkpoint_content_replacement_rejected(v2_checkpoint, second_v2_checkpoint, tmp_path):
    candidate=load_v2_policy(v2_checkpoint);current=load_v2_policy(second_v2_checkpoint)
    store=ChampionStore(tmp_path/"champion.json");store.bootstrap(current.identity);decision=_decision(store,candidate)
    Path(v2_checkpoint).write_bytes(Path(second_v2_checkpoint).read_bytes())
    with pytest.raises(ValueError,match="content hash changed"):
        store.promote(candidate.identity,decision=decision,expected_plan_hash="p"*64,expected_summary_hash="s"*64,transaction_id="tx")


def test_valid_selection_promotes_and_replace_failure_rolls_back(v2_checkpoint, second_v2_checkpoint, tmp_path):
    candidate=load_v2_policy(v2_checkpoint);current=load_v2_policy(second_v2_checkpoint)
    store=ChampionStore(tmp_path/"champion.json");old=store.bootstrap(current.identity);decision=_decision(store,candidate)
    def fail_replace(_source,_destination): raise OSError("injected replace failure")
    with pytest.raises(OSError,match="injected"):
        store.promote(candidate.identity,decision=decision,expected_plan_hash="p"*64,expected_summary_hash="s"*64,transaction_id="tx",replace_fn=fail_replace)
    assert store.load()["hash"]==old["hash"]
    promoted=store.promote(candidate.identity,decision=decision,expected_plan_hash="p"*64,expected_summary_hash="s"*64,transaction_id="tx")
    assert promoted["champion"]==candidate.identity and promoted["pool_sync"]["status"]=="pending"


def test_opponent_pool_validates_identity_weights_and_recovery(tmp_path, v2_checkpoint, legacy_checkpoint):
    v2=load_v2_policy(v2_checkpoint)
    pool=OpponentPoolStore(tmp_path/"pool.json");pool.initialize(sampling_seed=17)
    pool.add_entry({"id":"c1","kind":"champion","checkpoint":str(v2_checkpoint),"policy_identity":v2.identity,"weight":3})
    from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent
    legacy=load_legacy_v1_opponent(legacy_checkpoint)
    pool.add_entry({"id":"legacy","kind":"legacy_v1","checkpoint":str(legacy_checkpoint),"legacy_identity":legacy.identity,"weight":0.5})
    recovered=OpponentPoolStore(pool.path).recover()
    assert recovered["revision"]==2 and pool.sample(sample_seed=2)["id"]=="c1"
    with pytest.raises(ValueError,match="finite"):
        pool.add_entry({"id":"bad","kind":"rule","weight":float("nan")})


def test_nonfinite_or_malformed_summary_is_rejected(v2_checkpoint,second_v2_checkpoint,tmp_path):
    candidate=load_v2_policy(v2_checkpoint);champion=load_v2_policy(second_v2_checkpoint)
    store=ChampionStore(tmp_path/"champion.json");store.bootstrap(champion.identity)
    bad=_summary();bad["per_opponent"]["champion"]["eval/win_rate"]=float("nan")
    with pytest.raises(ValueError,match="finite"):_decision(store,candidate,summary=bad)
    bad=_summary();del bad["per_opponent"]["legacy"]
    with pytest.raises(ValueError,match="exact fixed matrix"):_decision(store,candidate,summary=bad)


def test_final_mode_decision_cannot_promote_even_with_recomputed_hash(v2_checkpoint,second_v2_checkpoint,tmp_path):
    candidate=load_v2_policy(v2_checkpoint);champion=load_v2_policy(second_v2_checkpoint)
    store=ChampionStore(tmp_path/"champion.json");store.bootstrap(champion.identity);decision=_decision(store,candidate)
    decision["evaluation_mode"]="final_audit";decision["hash"]=payload_hash(decision)
    with pytest.raises(ValueError,match="selection decision"):
        store.promote(candidate.identity,decision=decision,expected_plan_hash="p"*64,expected_summary_hash="s"*64,transaction_id="tx")
