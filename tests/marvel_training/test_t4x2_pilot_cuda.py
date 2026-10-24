"""Bounded explicit-CUDA integration; never constitutes T4x2/NCCL certification."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.smoke import CppSmokeGame
from guandan_marvel.kaggle.pilot_observability import PilotObserver
from guandan_marvel.kaggle.checkpoints import save_distributed_checkpoint,load_ddp_manifest,freeze_distributed_policy,update_checkpoint_index,build_recovery_bundle
from guandan_marvel.evaluation.checkpoints import load_v2_policy


def test_single_cuda_two_updates_and_exact_restore_with_trimmed_layout(tmp_path: Path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip("explicit CUDA required; no CPU model fallback")
    game=CppSmokeGame(512)
    cfg=TrainConfig(game=game.spec,device="cuda",algo_name="ippo",seed=20261021,
                    rollout_batch_size=1,collect_batch_size=1,mini_batches=1,
                    run_name="t4-pilot-bounded-cuda-compat",logging_dir=str(tmp_path),
                    total_iterations=2,amp_enabled=True,amp_init_scale=1024,
                    trim_completed_padding=True,max_sampler_bytes_per_rank=128*2**20,
                    ppo=PPOConfig(agent_epoch=1,training_semantics_profile="macro_decision_v1",
                                  policy_regularizer_mode="macro_normalized_kl_uniform_v1"),
                    model=ModelConfig(architecture="v1.4-macro-fuse",history_dim=128,feature_dim=256,
                                      decoder_layers=4,decoder_heads=4,state_layers=3,action_layers=2,fuse_layers=2))
    trainer=IPPOTrainer(cfg,game=game,run_dir=tmp_path/"first")
    warm_start_used=False
    production_warm=Path(__file__).resolve().parents[2]/"runs/comparison_delivery/dataset/m09_candidate_iter002000.pt"
    if production_warm.is_file():
        migration=trainer.load_warm_start_policy(torch.load(production_warm,map_location="cpu",weights_only=False))
        assert migration["migration"]["fuse_migrated"] and trainer.iteration==0
        warm_start_used=True
    observer=PilotObserver({"pilot_observability":{"enabled":True}},trainer,tmp_path,tmp_path/"rank0")
    observer.export_initial(mode="full")
    try:
        observer.begin();metrics=trainer.train_iteration();extra=observer.finish(metrics)
        assert metrics["runtime/trim_completed_padding"]
        assert metrics["perf/learner_time_capacity"]<512
        assert extra["pilot/effective_optimizer_updates"]>=1 and extra["pilot/logit_weight_changed"]
        checkpoint=trainer.save_checkpoint(tmp_path/"iter1.pt")
        published=tmp_path/"published"
        distributed=published/"checkpoints/iter000001"
        save_distributed_checkpoint(trainer,distributed,context=trainer.distributed)
        manifest=load_ddp_manifest(distributed,verify_shards=True)
        candidate=freeze_distributed_policy(trainer,tmp_path/"candidate.pt",ddp_manifest=manifest)

        from guandan_marvel.kaggle.common import atomic_json,append_jsonl
        from guandan_marvel.kaggle.pilot import safe_extract
        from guandan_marvel.kaggle.bootstrap import validate_recovery_directory
        import guandan_marvel.kaggle.checkpoints as checkpoint_module
        update_checkpoint_index(published,checkpoint_dir=distributed,ddp_manifest=manifest,evaluation_status="not_due")
        atomic_json(published/"pilot_run_contract.json",{"pilot":"unit test"})
        atomic_json(published/"pilot_history_segment.json",{"segment":"cuda_test"})
        atomic_json(published/"progress.json",{"version":0})
        append_jsonl(published/"metrics.jsonl",{"statistics/iterations":1})
        write=checkpoint_module._stable_archive_write
        changed=[False]
        def mutate_progress_during_checkpoint_write(archive,path,archive_name):
            if not changed[0]:
                atomic_json(published/"progress.json",{"version":1})
                changed[0]=True
            return write(archive,path,archive_name)
        monkeypatch.setattr(checkpoint_module,"_stable_archive_write",mutate_progress_during_checkpoint_write)
        bundle=build_recovery_bundle(run_root=published,destination=published/"recovery.zip",
                    source_commit=trainer.source_parent_commit,source_archive_sha256="a"*64,
                    source_content_manifest_hash="b"*64,package_id="pilot-unit",status="partial",native_extension=None)
        safe_extract(bundle,tmp_path/"recovered")
        validate_recovery_directory(tmp_path/"recovered")
        assert json.loads((tmp_path/"recovered/artifacts/progress.json").read_text())["version"]==0
        assert (tmp_path/"recovered/artifacts/pilot_history/cuda_test/metrics.jsonl").is_file()
        policy=load_v2_policy(candidate,device="cuda",expected_algorithm="ippo")
        assert policy.iteration==1 and policy.model.actor_fuse is not None
        del policy
        first={"updates":trainer.learner_updates,"trimmed_time_capacity":metrics["perf/learner_time_capacity"],
               "peak_allocated_bytes":extra["pilot/peak_allocated_bytes"]}
    finally:trainer.close()
    del trainer,observer
    resumed=IPPOTrainer(cfg,game=game,run_dir=tmp_path/"resumed")
    try:
        decision=resumed.load_checkpoint(checkpoint)
        assert decision.kind=="exact_resume"
        assert resumed.cfg.trim_completed_padding and resumed.sampler.config.trim_completed_padding
        metrics=resumed.train_iteration()
        assert resumed.iteration==2 and resumed.completed_games_total==2
        resumed.save_checkpoint(tmp_path/"iter2.pt")
        print(json.dumps({"status":"PASS","device":torch.cuda.get_device_name(0),"warm_start_used":warm_start_used,"first":first,
                          "restore":decision.kind,"iterations":resumed.iteration,
                          "completed_games":resumed.completed_games_total,
                          "not_proved":["T4x2","NCCL","production horizon4096/full batch","strength"]}))
    finally:resumed.close()
