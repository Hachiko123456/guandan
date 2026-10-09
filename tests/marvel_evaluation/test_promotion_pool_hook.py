from __future__ import annotations

import copy
import json
from pathlib import Path
from threading import Event

import pytest
import torch

from guandan_marvel import cpp_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.evaluation.hooks import TrainingEvaluatorHook
from guandan_marvel.evaluation.manifests import FinalHoldoutAudit
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore, PromotionPolicy, decide_promotion
from guandan_marvel.model import Agent
from guandan_marvel.training.checkpoints import build_backend_identity, compute_source_fingerprint, serialize_config, canonical_json_hash


def _identity(name, digest):
    return {
        "checkpoint_path": name, "checkpoint_sha256": digest,
        "algorithm": "ippo", "architecture": "v1.3",
        "protocol_version": cpp_game.protocol_version,
        "backend": "cpp", "backend_identity_hash": "b", "resolved_config_hash": "c",
        "iteration": 1,
    }


def _summary(*, complete=True, low=0.60, win=0.75, clusters=4, illegal=0, truncated=0):
    entries = {}
    for name in ("champion", "rule", "random", "legacy", "opponent_pool"):
        entries[name] = {
            "complete": complete, "eval/win_rate": win,
            "eval/clustered_ci95_low": low, "eval/cluster_count": clusters,
            "eval/illegal_count": illegal, "eval/truncated_count": truncated,
        }
    return {"per_opponent": entries, "seat_rotation_coverage": {"complete": complete}}


def test_incomplete_evaluation_never_promotes():
    decision = decide_promotion(
        _summary(complete=False), candidate_identity=_identity("c", "1" * 64),
        champion_identity=_identity("h", "2" * 64), evaluation_hash="e", selection_deals_hash="d",
    )
    assert decision["status"] == "inconclusive"
    assert decision["promote"] is False


def test_point_estimate_501_does_not_auto_promote():
    decision = decide_promotion(
        _summary(low=0.49, win=0.501), candidate_identity=_identity("c", "1" * 64),
        champion_identity=_identity("h", "2" * 64), evaluation_hash="e", selection_deals_hash="d",
    )
    assert decision["status"] == "inconclusive"
    assert decision["promote"] is False


def test_champion_atomic_failure_keeps_old_manifest(tmp_path):
    store = ChampionStore(tmp_path / "champion.json")
    old = store.bootstrap(_identity("old", "2" * 64))
    decision = decide_promotion(
        _summary(), candidate_identity=_identity("new", "1" * 64),
        champion_identity=old["champion"], evaluation_hash="e", selection_deals_hash="d",
        policy=PromotionPolicy(ci95_low_threshold=0.52, min_clusters=2),
    )
    assert decision["promote"] is True
    before = store.path.read_bytes()
    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")
    with pytest.raises(OSError, match="injected"):
        store.promote(_identity("new", "1" * 64), decision=decision, replace_fn=fail_replace)
    assert store.path.read_bytes() == before
    assert store.load()["hash"] == old["hash"]
    assert not list(tmp_path.glob("*.tmp-*"))


def test_final_holdout_is_candidate_locked_and_one_shot(tmp_path):
    audit = FinalHoldoutAudit(tmp_path / "final-audit.json", deals_manifest_hash="deals")
    audit.initialize()
    audit.lock_candidate("a" * 64)
    with pytest.raises(ValueError, match="candidate identity"):
        audit.record_use(candidate_sha256="b" * 64, evaluation_hash="x", purpose="final_audit")
    with pytest.raises(ValueError, match="only"):
        audit.record_use(candidate_sha256="a" * 64, evaluation_hash="x", purpose="selection")
    used = audit.record_use(candidate_sha256="a" * 64, evaluation_hash="x", purpose="final_audit")
    assert len(used["uses"]) == 1
    with pytest.raises(ValueError, match="already"):
        audit.record_use(candidate_sha256="a" * 64, evaluation_hash="y", purpose="final_audit")


def test_opponent_pool_weights_and_recovery(tmp_path, v2_checkpoint, legacy_checkpoint):
    v2 = load_v2_policy(v2_checkpoint)
    pool = OpponentPoolStore(tmp_path / "pool.json")
    pool.initialize(sampling_seed=17)
    pool.add_entry({"id": "c1", "kind": "champion", "checkpoint": str(v2_checkpoint), "checkpoint_sha256": v2.sha256, "weight": 3})
    pool.add_entry({"id": "c2", "kind": "historical_champion", "checkpoint": str(v2_checkpoint), "checkpoint_sha256": v2.sha256, "weight": 1})
    pool.add_entry({"id": "legacy", "kind": "legacy_v1", "checkpoint": str(legacy_checkpoint), "weight": 0.5})
    recovered = OpponentPoolStore(pool.path).recover()
    assert recovered["revision"] == 3
    assert {entry["id"] for entry in recovered["entries"]} == {"c1", "c2", "legacy"}
    assert pool.sample(sample_seed=2)["id"] in {"c1", "c2"}


def test_training_hook_freezes_merged_policy_without_training_state_pollution(tmp_path):
    cfg = TrainConfig(
        game=cpp_game.GAME_SPEC, device="cpu", max_sampler_bytes_per_rank=1,
        model=ModelConfig(architecture="v1.3", history_dim=8, feature_dim=8, decoder_layers=1,
                          decoder_heads=1, state_layers=1, action_layers=1, fuse_layers=1, ffn_ratio=2),
    )
    model = Agent(cfg, enable_value_head=True)
    class Trainer:
        pass
    trainer=Trainer()
    trainer.algorithm="ippo"; trainer.iteration=4; trainer.agent_merged=model; trainer.backend_name="cpp"
    trainer.backend_identity=build_backend_identity(cpp_game)
    trainer.resolved_config=serialize_config(cfg); trainer.resolved_config_hash=canonical_json_hash(trainer.resolved_config)
    source_base={"schema":"guandan-marvel-source-fingerprint-v1","components":{}}
    trainer.source_fingerprint={**source_base,"hash":canonical_json_hash(source_base)}
    trainer.source_hash=trainer.source_fingerprint["hash"]; trainer.source_parent_commit="f"*40
    trainer.merged_policy_identity="current_policy_copy"; trainer.cfg=cfg
    optimizer_state={"step":7}; sampler_state={"counter":11}; rng_before=torch.get_rng_state().clone()
    weights_before={k:v.detach().clone() for k,v in model.state_dict().items()}
    release=Event(); started=Event()
    def evaluator(path, iteration):
        started.set(); release.wait(5); return {"path":str(path),"iteration":iteration}
    hook=TrainingEvaluatorHook(interval=2,snapshot_dir=tmp_path,evaluator=evaluator)
    result=hook.on_iteration_completed(trainer,iteration=4)
    assert result["status"]=="scheduled" and Path(result["snapshot"]).is_file()
    assert started.wait(2)
    assert optimizer_state=={"step":7} and sampler_state=={"counter":11}
    assert torch.equal(rng_before,torch.get_rng_state())
    for key,value in model.state_dict().items(): torch.testing.assert_close(value,weights_before[key])
    artifact=load_v2_policy(result["snapshot"])
    assert artifact.iteration==4
    release.set(); hook.close(wait=True)


def test_local_trainer_hook_failure_is_recorded_without_poisoning_iteration():
    from guandan_marvel.training.local import LocalTrainerBase
    class Metrics:
        def __init__(self): self.rows=[]
        def log(self, values, *, step): self.rows.append((dict(values), step))
    class BrokenHook:
        def on_iteration_completed(self, trainer, *, iteration):
            raise RuntimeError("evaluation unavailable")
    class FakeTrainer:
        iteration_status="running"; active_iteration=1; durable_updates=0; completed_games_total=0
        learner_updates=2; evaluation_hook=BrokenHook(); metrics=Metrics(); iteration=0
        durable_learner_updates=0; last_metrics={}
    trainer=FakeTrainer()
    result=LocalTrainerBase._finish_iteration(trainer,metrics={"loss":1.0},completed_games=3)
    assert result["evaluation/hook_status"]=="schedule_failed"
    assert "evaluation unavailable" in result["evaluation/hook_error"]
    assert trainer.iteration==trainer.durable_updates==1
    assert trainer.completed_games_total==3
    assert trainer.iteration_status=="completed" and trainer.active_iteration is None
