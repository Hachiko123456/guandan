from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
import torch
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig, deserialize_train_config
from guandan_marvel import python_game
from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.runtime.trajectory_layout import trim_completed_trajectory_padding
from guandan_marvel.training.checkpoints import build_semantic_configuration
from guandan_marvel.kaggle.pilot_delivery import make_profiles, render_notebook
from guandan_marvel.kaggle.pilot_evaluation import tasks_for, summarize, check_records
from guandan_marvel.kaggle.pilot import safe_extract, worker_command, write_reports, phase_health
from guandan_marvel.evaluation.manifests import build_deals_manifest


def batch(lengths=(3,5), horizon=8):
    b=len(lengths)
    terminal=torch.zeros(b,horizon,dtype=torch.bool)
    for i,n in enumerate(lengths):terminal[i,n-1]=True
    return TensorDict({"num_steps":torch.tensor(lengths),"is_terminal":terminal,
                      "obs_tokens":torch.arange(b*horizon*4*2).reshape(b,horizon,4,2),
                      "player_ids":torch.ones(b,horizon,dtype=torch.long),
                      "rewards":torch.randn(b,horizon,4),
                      "policy_train_mask":torch.ones(b,horizon,dtype=torch.bool)},batch_size=[b])


def test_trim_preserves_lengths_valid_values_and_storage_capacity():
    original=batch();cropped=trim_completed_trajectory_padding(original)
    assert cropped["is_terminal"].shape==(2,5)
    assert original["is_terminal"].shape==(2,8)
    assert torch.equal(cropped["num_steps"],original["num_steps"])
    for name in ("obs_tokens","player_ids","rewards","policy_train_mask"):
        assert torch.equal(cropped[name],original[name][:,:5])
    assert cropped["obs_tokens"].data_ptr()==original["obs_tokens"].data_ptr()


@pytest.mark.parametrize("lengths",[(0,5),(9,5)])
def test_trim_refuses_invalid_lengths(lengths):
    x=batch();x["num_steps"]=torch.tensor(lengths)
    with pytest.raises(ValueError):trim_completed_trajectory_padding(x)


def test_trim_refuses_truncation():
    x=batch();x["is_terminal"].zero_()
    with pytest.raises(ValueError,match="incomplete"):trim_completed_trajectory_padding(x)


def test_default_layout_serialization_is_legacy_compatible():
    cfg=TrainConfig(game=python_game.GAME_SPEC,device="cpu")
    assert "trim_completed_padding" not in cfg.to_dict()
    assert "completed_trajectory_layout" not in build_semantic_configuration(cfg,algorithm="vrpo")["execution_semantics"]
    cfg.trim_completed_padding=True
    restored=deserialize_train_config(cfg.to_dict())
    assert restored.trim_completed_padding is True
    assert build_semantic_configuration(cfg,algorithm="vrpo")["execution_semantics"]["completed_trajectory_layout"]=="valid_prefix_v1"


@pytest.mark.parametrize("value",[1,"true",None])
def test_layout_bool_strict(value):
    with pytest.raises(TypeError):TrainConfig(game=python_game.GAME_SPEC,device="cpu",trim_completed_padding=value)


def test_profiles_are_capped_amp_nccl_and_not_unverified_hyperparameter_proposal():
    profiles=make_profiles();start=profiles["precheck"]
    assert start["training"]["target_iterations"]==2
    assert profiles["resume_check"]["resume"]["target_iterations"]==3
    assert profiles["train"]["resume"]["target_iterations"]==50
    assert start["ddp"]=={"world_size":2,"backend":"nccl","require_cuda":True,"find_unused_parameters":False}
    t=start["training"]
    assert t["collect_batch_size"]==20 and t["rollout_batch_size"]==16
    assert t["amp_enabled"] and t["trim_completed_padding"]
    assert t["base_clip_coef"]==0.02 and t["base_reg_coef"]==0.1
    assert t["model"]["architecture"]=="v1.4-macro-fuse" and t["ntp_weight"]==0
    assert t["checkpoint_keep_last"]==1 and t["evaluation_interval"]==0


def manifest():
    return build_deals_manifest(counts={"training":0,"selection":4,"final":0},
                               bases={"training":0,"selection":1800210000,"final":2000000000})


def records_for(tasks):
    rows=[]
    for t in tasks:
        d=t["deal"];arm=t["arm"]
        rows.append({"task_id":t["task_id"],"arm":arm,"status":"complete","deal_id":d["deal_id"],
                     "deal_seed":d["seed"],"level_rank":d["level_rank"],"leader_seat":d["leader_seat"],
                     "seat_rotation":t["rotation"],"candidate_checkpoint_sha256":"c"*64 if arm=="control_vs_rule" else "n"*64,
                     "opponent_checkpoint_sha256":"c"*64 if arm=="candidate_vs_control" else None,
                     "base_deal_sha256":str(d["seed"]).zfill(64),"candidate_win":t["rotation"]%2==0,
                     "candidate_team_reward":1.0 if t["rotation"]%2==0 else -1.0,
                     "illegal_count":0,"truncated_count":0,"temperature":0.0})
    return rows


def test_evaluation_tasks_balanced_unique_two_shards():
    tasks=tasks_for(manifest())
    assert len(tasks)==4*4*3
    assert len({t["task_id"] for t in tasks})==len(tasks)
    assert len(tasks[::2])==len(tasks[1::2])
    assert {t["rotation"] for t in tasks}==set(range(4))


def test_summary_paired_zero_not_fabledan_claim():
    tasks=tasks_for(manifest());rows=records_for(tasks)
    s=summarize(rows,tasks,control_sha="c"*64,candidate_sha="n"*64)
    assert s["paired_rule_improvement"]["mean"]==0
    assert s["verdict"]=="INCONCLUSIVE" and s["fabledan"]=="NOT_RUN"
    assert not s["final_holdout_used"] and not s["formal_promotion"]
    assert s["arms"]["candidate_vs_rule"]["independent_deals"]==4


@pytest.mark.parametrize("mutation",["missing","duplicate","identity","deal","truncated","temperature"])
def test_evaluation_rejects_false_completion(mutation):
    tasks=tasks_for(manifest());rows=records_for(tasks)
    if mutation=="missing":rows.pop()
    elif mutation=="duplicate":rows.append(rows[0])
    elif mutation=="identity":rows[0]["candidate_checkpoint_sha256"]="x"*64
    elif mutation=="deal":rows[0]["base_deal_sha256"]="x"*64
    elif mutation=="truncated":rows[0]["truncated_count"]=1
    elif mutation=="temperature":rows[0]["temperature"]=1.0
    with pytest.raises(ValueError):check_records(rows,tasks,control_sha="c"*64,candidate_sha="n"*64)


@pytest.mark.parametrize("name",["../outside","/outside","dir\\outside","C:/outside"])
def test_archive_extraction_rejects_traversal(tmp_path,name):
    z=tmp_path/"input.zip"
    info=zipfile.ZipInfo("placeholder")
    info.filename=name  # bypass Windows ZipInfo separator normalization to test a hostile input ZIP
    with zipfile.ZipFile(z,"w") as f:f.writestr(info,"bad")
    with pytest.raises(ValueError):safe_extract(z,tmp_path/"unpack")


def test_notebook_contains_offline_checks_and_no_long_training():
    import ast
    notebook=render_notebook()
    code="".join(notebook["cells"][1]["source"])
    ast.parse(code)
    assert "pip install" not in code and "CUDA_VISIBLE_DEVICES" not in code
    assert "RECOVERY_ZIP" in code and "manifest_hash" in code and "T4 x2" in "".join(notebook["cells"][0]["source"])


def test_worker_command_two_ranks_no_cpu_fallback(tmp_path):
    meta={"pilot_id":"test","source":{"git_commit":"a"*40,"archive_sha256":"b"*64,"content_manifest_hash":"c"*64}}
    cmd=worker_command(tmp_path/"profile.json",tmp_path/"run",meta,recovery=None,warm=tmp_path/"warm.pt")
    assert "--nproc_per_node=2" in cmd and "--max-restarts=0" in cmd
    assert "--warm-start-checkpoint" in cmd and "--local-cpu-test" not in cmd


def test_report_survives_empty_failed_run_and_excludes_raw_shards(tmp_path):
    out=tmp_path/"artifacts";out.mkdir()
    (out/"training/checkpoints/iter1").mkdir(parents=True)
    (out/"training/checkpoints/iter1/rank0.pt").write_bytes(b"do not duplicate raw state")
    archive=write_reports(out,{"status":"failed","error":"example"})
    assert (out/"report.html").exists() and (out/"pilot_result.json").exists()
    with zipfile.ZipFile(archive) as z:
        assert "report.html" in z.namelist()
        assert not any("checkpoints/" in p for p in z.namelist())


def test_cuda_valid_prefix_matches_padded_forward_and_gradients():
    if not torch.cuda.is_available():pytest.skip("explicit CUDA model comparison required")
    from guandan_marvel.model import Agent
    torch.manual_seed(32)
    cfg=TrainConfig(game=python_game.GAME_SPEC,device="cuda",algo_name="ippo",
                    model=ModelConfig(architecture="v1.4-macro-fuse",history_dim=16,feature_dim=24,
                                      decoder_layers=1,decoder_heads=4,state_layers=1,action_layers=1,fuse_layers=1))
    first=Agent(cfg,enable_value_head=True).cuda().eval();second=deepcopy(first)
    B,T,P,A,L=2,8,4,3,8
    x=TensorDict({"num_steps":torch.tensor([3,5],device="cuda"),
                  "player_ids":torch.tensor([[0,1,2,-1,-1,-1,-1,-1],[0,1,2,3,0,-1,-1,-1]],device="cuda"),
                  "obs_tokens":torch.randint(1,200,(B,T,P,4),device="cuda"),
                  "obs_channels":torch.zeros(B,T,P,512,dtype=torch.uint8,device="cuda"),
                  "avail_tokens":torch.randint(1,200,(B,T,A,L),device="cuda"),
                  "avail_weights":torch.ones(B,T,A,device="cuda"),
                  "is_terminal":torch.zeros(B,T,dtype=torch.bool,device="cuda")},batch_size=[B])
    x["is_terminal"][0,2]=True;x["is_terminal"][1,4]=True
    cropped=trim_completed_trajectory_padding(x)
    a=first(rollout=x);b=second(rollout=cropped)
    mask=torch.arange(5,device="cuda").unsqueeze(0)<x["num_steps"].unsqueeze(1)
    torch.testing.assert_close(a.avail_log_probs[:,:5][mask],b.avail_log_probs[mask],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(a.player_state_value[:,:5][mask],b.player_state_value[mask],atol=2e-5,rtol=2e-5)
    (a.avail_log_probs[:,:5][mask].sum()+a.player_state_value[:,:5][mask].sum()).backward()
    (b.avail_log_probs[mask].sum()+b.player_state_value[mask].sum()).backward()
    for (n,pa),(_,pb) in zip(first.named_parameters(),second.named_parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad,pb.grad,atol=8e-5,rtol=8e-5,msg=n)

    from guandan_marvel.model.cache import KVCache
    caches=[KVCache(1,B*P,max_ctx_len=T,num_heads=4,head_dim=4,value_heads=4,value_head_dim=4,
                    dtype=torch.float32,device="cuda") for _ in range(2)]
    with torch.inference_mode():
        first(rollout=x,use_cache=True,past_key_values=caches[0],cache_only=True)
        second(rollout=cropped,use_cache=True,past_key_values=caches[1],cache_only=True)
        step=TensorDict({name:(torch.ones_like(value) if name=="num_steps" else value[:,:1])
                         for name,value in x.items()},batch_size=[B])
        after_full=first(rollout=step,use_cache=True,past_key_values=caches[0])
        after_trim=second(rollout=step,use_cache=True,past_key_values=caches[1])
        torch.testing.assert_close(after_full.avail_probs,after_trim.avail_probs,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(after_full.player_state_value,after_trim.player_state_value,atol=2e-5,rtol=2e-5)


def test_phase_health_uses_top_level_iteration_and_requires_both_rank_evidence(tmp_path,monkeypatch):
    from guandan_marvel.kaggle.common import atomic_json,append_jsonl
    import guandan_marvel.kaggle.checkpoints as checkpoints
    root=tmp_path/"run";root.mkdir()
    atomic_json(root/"checkpoint_manifest.json",{"latest":{"checkpoint":"checkpoints/iter000002","iteration":2}})
    monkeypatch.setattr(checkpoints,"load_ddp_manifest",lambda *a,**k:{
        "iteration":2,"global_counters":{"completed_games_total":40},"world_size":2,"manifest_hash":"m"})
    for rank in range(2):
        rr=root/"ranks"/f"rank{rank:03d}"
        atomic_json(rr/"pilot_runtime.json",{"communication_backend":"nccl","local_rank":rank,"device":f"cuda:{rank}","amp":True})
        append_jsonl(rr/"pilot_iterations.jsonl",{"effective_optimizer_updates":1,"peak_allocated_bytes":1000,"logit_weight_changed":True})
        append_jsonl(rr/"diagnostic.jsonl",{"event":"iteration_completed"})
    result=phase_health(root,target=2,resumed=False)
    assert result["status"]=="PASS"
    with pytest.raises(ValueError,match="recovery"):phase_health(root,target=2,resumed=True)
    (root/"ranks/rank001/pilot_iterations.jsonl").write_text("",encoding="utf-8")
    with pytest.raises(ValueError,match="ineffective"):phase_health(root,target=2,resumed=False)


def test_cpu_observer_counts_effective_updates_not_attempts(tmp_path):
    from guandan_marvel.kaggle.pilot_observability import PilotObserver
    parameter=torch.tensor([[1.0]])
    trainer=SimpleNamespace(algorithm="ippo",rank=0,local_rank=0,world_size=2,
        cfg=SimpleNamespace(total_iterations=2,trim_completed_padding=True),device=torch.device("cpu"),
        amp_enabled=True,training_semantics_profile="macro_decision_v1",source_hash="test",
        learner_updates=0,amp_skipped_steps=0,iteration=1,completed_games_total=10,
        agent_resume=SimpleNamespace(logit_head=SimpleNamespace(weight=parameter)),
        grad_scaler=SimpleNamespace(get_scale=lambda:1024.0))
    o=PilotObserver({"pilot_observability":{"enabled":True}},trainer,tmp_path,tmp_path/"rank")
    o.begin();trainer.amp_skipped_steps=1
    with pytest.raises(RuntimeError,match="zero effective"):o.finish({})
    o.begin();trainer.learner_updates=1;parameter.add_(0.1)
    result=o.finish({})
    assert result["pilot/effective_optimizer_updates"]==1 and result["pilot/logit_weight_changed"]


def test_trim_preserves_macro_gae_targets_and_durations():
    from tests.marvel_training._macro_fixtures import make_case
    from guandan_marvel.training.macro_gae import compute_macro_decision_gae
    players=[0,0,1,0,2,3,1,1];commits=[False,True,True,True,True,True,False,True]
    rollout,values,_,_=make_case(players,commits,horizon=13,rewards={(7,0):1.0,(7,1):-1.0},
                               values={(0,0):0.2,(3,0):0.4},dtype=torch.float32)
    trimmed=trim_completed_trajectory_padding(rollout)
    old=compute_macro_decision_gae(rollout,values)
    new=compute_macro_decision_gae(trimmed,values[:,:8])
    torch.testing.assert_close(old.advantage,new.advantage)
    torch.testing.assert_close(old.value_target,new.value_target)
    assert torch.equal(old.duration,new.duration)


@pytest.mark.parametrize("fail_resume,evaluation_only",[(False,False),(True,False),(False,True)])
def test_pipeline_stages_total_fifty_and_keeps_failure_recovery(tmp_path,monkeypatch,fail_resume,evaluation_only):
    import guandan_marvel.kaggle.pilot as pilot
    import guandan_marvel.kaggle.bootstrap as bootstrap
    import guandan_marvel.kaggle.telemetry as telemetry
    from guandan_marvel.kaggle.common import atomic_json,sha256_file
    dataset=tmp_path/"dataset";dataset.mkdir()
    profiles=make_profiles()
    for key,value in profiles.items():atomic_json(dataset/(key+".json"),value)
    deals=manifest();atomic_json(dataset/"development_deals.json",deals)
    meta={"pilot_id":"test", "manifest_hash":"test-manifest", "evaluation_manifest_hash":deals["hash"],
          "evaluation_reserve_seconds":3600,"profiles":{k:k+".json" for k in profiles},
          "source":{"git_commit":"a"*40,"archive_sha256":"b"*64,"content_manifest_hash":"c"*64}}
    (dataset/"warm_start_policy.pt").write_bytes(b"warm")
    monkeypatch.setattr(pilot,"validate_manifest",lambda p:meta)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:2)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda i:"Tesla T4")
    monkeypatch.setattr(torch.distributed,"is_nccl_available",lambda:True)
    monkeypatch.setattr(bootstrap,"validate_recovery_directory",lambda p:json.loads((p/"recovery_manifest.json").read_text()))
    class Recorder:
        def __init__(self,*a,**k):pass
        def start(self):pass
        def close(self):pass
        def set_phase(self,*a):pass
    monkeypatch.setattr(telemetry,"TelemetryRecorder",Recorder)
    monkeypatch.setattr(telemetry,"runtime_environment",lambda:{"unit_test":True})
    calls=[]
    def fake_process(command,**kwargs):
        if "--run-root" in command:
            run=Path(command[command.index("--run-root")+1]);profile=json.loads(Path(command[command.index("--profile")+1]).read_text())
            ctl=profile.get("resume",profile.get("training"));target=ctl["target_iterations"];calls.append(target)
            if target==3 and fail_resume:raise RuntimeError("injected resume failure")
            (run/"checkpoints/last").mkdir(parents=True)
            if profile["mode"]!="resume":(run/"pilot_initial_policy.pt").write_bytes(b"initial-control")
            (run/f"pilot_candidate_iter{target:06d}.pt").write_bytes(f"candidate-{target}".encode())
            atomic_json(run/"progress.json",{"phase":"complete"})
            atomic_json(run/"checkpoint_manifest.json",{"latest":{"iteration":target,"checkpoint":"checkpoints/last"}})
            with zipfile.ZipFile(run/"guandan-marvel-recovery.zip","w") as z:
                z.writestr("recovery_manifest.json",json.dumps({"iteration":target,"checkpoint":"checkpoint/manifest.json"}))
                z.writestr("checkpoint/manifest.json",json.dumps({"global_counters":{"completed_games_total":target*20}}))
        else:
            shard=int(command[command.index("--shard")+1]);out=Path(command[command.index("--output")+1])
            control=Path(command[command.index("--control")+1]);candidate=Path(command[command.index("--candidate")+1])
            tasks=tasks_for(deals);rows=records_for(tasks[shard::2])
            for row in rows:
                row["candidate_checkpoint_sha256"]=sha256_file(control if row["arm"]=="control_vs_rule" else candidate)
                row["opponent_checkpoint_sha256"]=sha256_file(control) if row["arm"]=="candidate_vs_control" else None
            (out/f"games.gpu{shard}.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n",encoding="utf-8")
    monkeypatch.setattr(pilot,"run_process",fake_process)
    monkeypatch.setattr(pilot,"phase_health",lambda root,target,resumed,**kwargs:{"status":"PASS","target_iteration":target,
                        "candidate":str(root/f"pilot_candidate_iter{target:06d}.pt")})
    source=tmp_path/"source";source.mkdir()
    out=tmp_path/"out/artifacts"
    recovery=None
    if evaluation_only:
        recovery=tmp_path/"recovery"
        (recovery/"artifacts").mkdir(parents=True)
        (recovery/"checkpoint").mkdir()
        atomic_json(recovery/"recovery_manifest.json",{"iteration":50,"checkpoint":"checkpoint/manifest.json"})
        atomic_json(recovery/"checkpoint/manifest.json",{"global_counters":{"completed_games_total":1000}})
        atomic_json(recovery/"artifacts/pilot_run_contract.json",{
            "pilot_manifest_hash":meta["manifest_hash"],"cap_iterations":50,"global_collect_batch_size":20,
            "evaluation_manifest_hash":meta["evaluation_manifest_hash"],"source":meta["source"]})
        (recovery/"artifacts/pilot_initial_policy.pt").write_bytes(b"initial-control")
        def export(recovery,destination,**kwargs):
            destination.write_bytes(b"recovered-policy")
            return destination
        monkeypatch.setattr(pilot,"export_recovered_policy",export)
    result=pilot.run_pilot(dataset=dataset,output=out,source=source,recovery=recovery)
    assert (out/"report.html").exists() and (out.parent/"guandan_t4x2_pilot_results.zip").exists()
    assert (out/"latest_recovery.zip").exists()
    if evaluation_only:
        assert calls==[] and result["status"]=="complete"
        assert "ZERO new training updates" in result["training_origin"]
        assert result["training_completed_iterations"]==50 and result["training_completed_games"]==1000
    elif fail_resume:
        assert calls==[2,3] and result["status"]=="failed" and result["training_completed_iterations"]==2
    else:
        assert calls==[2,3,50] and result["status"]=="complete"
        assert result["training_completed_iterations"]==50 and result["training_completed_games"]==1000
        assert result["evaluation"]["fabledan"]=="NOT_RUN"
    assert result["automatic_continuation_to_250"] is False


def test_build_upload_contains_current_source_notebook_and_hash_bound_inputs(tmp_path):
    import ast
    from guandan_marvel.kaggle.pilot_delivery import build_delivery,validate_manifest
    repo=Path(__file__).resolve().parents[2]
    warm=repo/"runs/comparison_delivery/dataset/m09_candidate_iter002000.pt"
    if not warm.is_file():pytest.skip("local production warm-start artifact not supplied in this checkout")
    delivery=tmp_path/"delivery"
    result=build_delivery(repo,delivery,warm)
    manifest=validate_manifest(delivery/"dataset")
    assert manifest["training_cap"]["completed_games"]==1000
    assert manifest["source"]["workspace_snapshot"] is True
    assert result["remote_verified"] is False
    with zipfile.ZipFile(delivery/"dataset/guandan-marvel-source.zip") as z:
        assert "guandan-marvel-source/guandan_marvel/runtime/trajectory_layout.py" in z.namelist()
        assert "guandan-marvel-source/guandan_marvel/kaggle/pilot.py" in z.namelist()
        assert not any(p.endswith((".pyd",".so")) for p in z.namelist())
    notebook=json.loads((delivery/"guandan_t4x2_pilot.ipynb").read_text(encoding="utf-8"))
    ast.parse("".join(notebook["cells"][1]["source"]))
    assert manifest["manifest_hash"] in "".join(notebook["cells"][1]["source"])
    with pytest.raises(FileExistsError):build_delivery(repo,delivery,warm)
    profile=delivery/"dataset/train_profile.json"
    profile.write_text(profile.read_text()+" ",encoding="utf-8")
    with pytest.raises(ValueError,match="content mismatch"):validate_manifest(delivery/"dataset")


def test_report_preserves_corrupt_log_diagnostics_without_claiming_completed_data(tmp_path):
    out=tmp_path/"artifacts";out.mkdir();(out/"training").mkdir()
    (out/"training/metrics.jsonl").write_text('{"statistics/iterations":1}\n{"partial":',encoding="utf-8")
    write_reports(out,{"status":"failed","error":"interrupted"})
    performance=json.loads((out/"performance_summary.json").read_text())
    assert performance["actual_training_iterations_logged"]==1
    assert len(performance["diagnostic_jsonl_errors"])==1
    assert json.loads((out/"pilot_result.json").read_text())["status"]=="failed"


def test_generated_notebook_bootstrap_hash_extract_and_handoff_without_training(tmp_path,monkeypatch):
    import os,sys,types,subprocess
    import guandan_marvel.kaggle.pilot as pilot
    from guandan_marvel.kaggle.pilot_delivery import build_delivery,BOOTSTRAP
    repo=Path(__file__).resolve().parents[2]
    warm=repo/"runs/comparison_delivery/dataset/m09_candidate_iter002000.pt"
    if not warm.is_file():pytest.skip("production artifact not provided")
    delivery=tmp_path/"delivery"
    build_delivery(repo,delivery,warm)
    root=tmp_path/"notebook-run"
    calls=[]
    def fake_run_pilot(**kwargs):
        calls.append(kwargs)
        out=kwargs["output"];out.mkdir()
        (out/"report.html").write_text("<p>mock handoff</p>",encoding="utf-8")
        (out.parent/"guandan_t4x2_pilot_results.zip").write_bytes(b"test")
        return {"status":"complete","simulation":True}
    monkeypatch.setattr(pilot,"run_pilot",fake_run_pilot)
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0))
    fake_torch=types.ModuleType("torch")
    fake_torch.cuda=SimpleNamespace(device_count=lambda:2,get_device_name=lambda i:"Tesla T4")
    monkeypatch.setitem(sys.modules,"torch",fake_torch)
    display=types.ModuleType("IPython.display")
    display.display=lambda *a:None;display.HTML=lambda x:x;display.FileLink=lambda x:x
    monkeypatch.setitem(sys.modules,"IPython.display",display)
    monkeypatch.setattr(sys,"path",list(sys.path))
    for key in ("PYTHONPATH","PYTHONUNBUFFERED","GUANDAN_MARVEL_SOURCE_COMMIT","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","NCCL_DEBUG",
                "TORCH_DISTRIBUTED_DEBUG","TORCH_NCCL_ASYNC_ERROR_HANDLING"):
        monkeypatch.setenv(key,os.environ.get(key,""))
    code=BOOTSTRAP.replace("DATASET_DIR = None",f"DATASET_DIR = Path({str(delivery/'dataset')!r})",1)
    code=code.replace("RUN_ROOT = Path('/kaggle/working/guandan_t4x2_pilot')",f"RUN_ROOT = Path({str(root)!r})",1)
    exec(compile(code,"generated_pilot_notebook","exec"),{})
    assert len(calls)==1 and calls[0]["output"]==root/"artifacts"
    assert calls[0]["source"].joinpath("guandan_marvel/kaggle/pilot.py").is_file()
    assert calls[0]["recovery"] is None and calls[0]["started_at"]>0
    assert (root/"native_build.log").exists()


def test_evaluation_only_recovery_exports_policy_not_training_state(tmp_path,monkeypatch):
    import guandan_marvel.kaggle.bootstrap as bootstrap
    import guandan_marvel.kaggle.checkpoints as checkpoints
    from guandan_marvel.kaggle.pilot import export_recovered_policy
    from guandan_marvel.evaluation.checkpoints import EVALUATION_CHECKPOINT_SCHEMA
    recovery=tmp_path/"recovery";recovery.mkdir()
    (recovery/"checkpoint").mkdir()
    monkeypatch.setattr(bootstrap,"validate_recovery_directory",lambda p:{"checkpoint":"checkpoint/manifest.json"})
    names=("algorithm","architecture","protocol_version","backend","backend_identity","resolved_config",
           "resolved_config_hash","agent_state_dict","iteration","source_fingerprint","source_hash","source_parent_commit")
    payload={k:"unit" for k in names};payload["agent_state_dict"]={"test":torch.ones(1)};payload["iteration"]=50;payload["resolved_config"]={"collect_batch_size":20}
    payload["sampler_state"]={"must_not_export":True}
    manifest={"iteration":50,"global_counters":{"completed_games_total":1000},"training_data_provenance":{"complete":True}}
    def load(path,**kwargs):
        assert path==recovery/"checkpoint" and kwargs["expected_world_size"]==2
        return payload,manifest
    monkeypatch.setattr(checkpoints,"load_rank_checkpoint_for_resume",load)
    result=export_recovered_policy(recovery,tmp_path/"policy.pt",source_commit="a"*40)
    saved=torch.load(result,map_location="cpu",weights_only=False)
    assert saved["schema"]==EVALUATION_CHECKPOINT_SCHEMA and saved["iteration"]==50
    assert "sampler_state" not in saved and saved["training_resume_allowed"] is False
    manifest["iteration"]=51
    with pytest.raises(ValueError,match="50 iterations"):export_recovered_policy(recovery,tmp_path/"bad.pt",source_commit="a"*40)
