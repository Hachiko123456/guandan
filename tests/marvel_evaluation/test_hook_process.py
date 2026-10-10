from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.evaluation.hooks import TrainingEvaluatorHook
from guandan_marvel.training.local import LocalTrainerBase


def _digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous();h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def _trainer(checkpoint):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);artifact=load_v2_policy(checkpoint)
    return SimpleNamespace(
        algorithm=artifact.algorithm,iteration=1,agent_merged=artifact.model,backend_name='cpp',
        backend_identity=payload['backend_identity'],resolved_config=payload['resolved_config'],resolved_config_hash=payload['resolved_config_hash'],
        source_fingerprint=payload.get('source_fingerprint',{'schema':'guandan-marvel-source-fingerprint-v1','components':{},'hash':'x'}),
        source_hash=payload.get('source_hash','fixture'),source_parent_commit=payload.get('source_parent_commit','f'*40),
        merged_policy_identity='current_policy_copy',cfg=artifact.model.cfg,
        sampler=SimpleNamespace(training_data_provenance=lambda: payload['training_data_provenance']),
    )


def _wait(hook,iterations,timeout=20):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        statuses={row['iteration']:row['status'] for row in hook.status_records()}
        if all(statuses.get(i) in {'completed','failed','cancelled','interrupted'} for i in iterations): return statuses
        time.sleep(0.02)
    raise AssertionError(f'timed out waiting for hook tasks: {hook.status_records()}')


def test_real_loader_runs_in_spawned_process_without_changing_training_torch_rng(v2_checkpoint,tmp_path):
    trainer=_trainer(v2_checkpoint)
    hook=TrainingEvaluatorHook(interval=1,snapshot_dir=tmp_path/'hook',evaluator_entrypoint='guandan_marvel.evaluation.workers:load_policy_identity_work')
    torch.manual_seed(12345);before=torch.get_rng_state().clone()
    scheduled=hook.on_iteration_completed(trainer,iteration=1)
    _wait(hook,[1]);events=hook.collect_ready();after=torch.get_rng_state().clone();hook.close()
    assert scheduled['status']=='scheduled' and events[0]['status']=='completed'
    assert torch.equal(before,after)


def test_snapshot_is_frozen_before_live_model_changes(v2_checkpoint,tmp_path):
    trainer=_trainer(v2_checkpoint);expected=_digest(trainer.agent_merged.state_dict())
    hook=TrainingEvaluatorHook(interval=1,snapshot_dir=tmp_path/'hook',evaluator_entrypoint='guandan_marvel.evaluation.workers:load_policy_identity_work',evaluator_config={'delay_seconds':0.2})
    scheduled=hook.on_iteration_completed(trainer,iteration=1)
    with torch.no_grad():
        next(trainer.agent_merged.parameters()).add_(100)
    _wait(hook,[1]);event=hook.collect_ready()[0];hook.close()
    assert event['result']['model_state_sha256']==expected
    assert _digest(trainer.agent_merged.state_dict())!=expected
    assert Path(scheduled['snapshot']).is_file()


def test_worker_failure_survives_next_schedule_and_multiple_results_deliver(v2_checkpoint,tmp_path):
    trainer=_trainer(v2_checkpoint)
    hook=TrainingEvaluatorHook(interval=1,snapshot_dir=tmp_path/'hook',evaluator_entrypoint='guandan_marvel.evaluation.workers:fail_on_iterations_work',evaluator_config={'fail_iterations':[1]},max_pending=2)
    hook.on_iteration_completed(trainer,iteration=1);trainer.iteration=2;hook.on_iteration_completed(trainer,iteration=2)
    _wait(hook,[1,2]);events=hook.collect_ready();hook.close()
    by_iteration={event['iteration']:event for event in events}
    assert by_iteration[1]['status']=='failed' and 'injected worker failure' in by_iteration[1]['error']
    assert by_iteration[2]['status']=='completed'
    journal=(tmp_path/'hook/evaluation_hook_events.jsonl').read_text(encoding='utf-8')
    assert '"status": "failed"' in journal and '"status": "delivered"' in journal


def test_duplicate_iteration_rejected_without_overwriting_snapshot(v2_checkpoint,tmp_path):
    trainer=_trainer(v2_checkpoint)
    hook=TrainingEvaluatorHook(interval=1,snapshot_dir=tmp_path/'hook',evaluator_entrypoint='guandan_marvel.evaluation.workers:load_policy_identity_work')
    first=hook.on_iteration_completed(trainer,iteration=1);second=hook.on_iteration_completed(trainer,iteration=1)
    _wait(hook,[1]);hook.close()
    assert first['status']=='scheduled' and second['status']=='duplicate_rejected'
    assert len(list((tmp_path/'hook').glob('*.pt')))==1


def test_close_records_cancelled_or_shutdown_pending_without_fake_completion(v2_checkpoint,tmp_path):
    trainer=_trainer(v2_checkpoint)
    hook=TrainingEvaluatorHook(interval=1,snapshot_dir=tmp_path/'hook',evaluator_entrypoint='guandan_marvel.evaluation.workers:load_policy_identity_work',evaluator_config={'delay_seconds':1.0},max_pending=2)
    hook.on_iteration_completed(trainer,iteration=1);trainer.iteration=2;hook.on_iteration_completed(trainer,iteration=2)
    delivered=hook.close(wait=False)
    journal=[json.loads(line) for line in (tmp_path/'hook/evaluation_hook_events.jsonl').read_text(encoding='utf-8').splitlines()]
    statuses={row['status'] for row in journal}
    assert 'cancelled' in statuses or 'shutdown_pending' in statuses
    assert not any(row.get('status')=='completed' for row in journal if row.get('iteration')==2)
    assert all(row['status'] in {'cancelled','failed','interrupted','completed'} for row in delivered)


def test_trainer_collects_failed_result_without_poisoning_completed_update(tmp_path):
    class Metrics:
        def __init__(self):self.rows=[]
        def log(self,values,*,step):self.rows.append((dict(values),step))
    class Hook:
        def collect_ready(self):return ({'iteration':0,'status':'failed','error':'worker failed'},)
        def on_iteration_completed(self,trainer,*,iteration):return {'status':'not_due','iteration':iteration}
        def close(self,*,wait=True):return tuple()
    trainer=object.__new__(LocalTrainerBase)
    trainer.iteration_status='running';trainer.active_iteration=1;trainer.durable_updates=0;trainer.completed_games_total=0
    trainer.learner_updates=2;trainer.evaluation_hook=Hook();trainer.metrics=Metrics();trainer.iteration=0
    trainer.durable_learner_updates=0;trainer.last_metrics={};trainer.evaluation_result_history=[];trainer.evaluation_results_path=tmp_path/"evaluation_results.jsonl"
    result=LocalTrainerBase._finish_iteration(trainer,metrics={'loss':1.0},completed_games=3)
    assert result['evaluation/results_collected']==1 and result['evaluation/results_failed']==1
    assert result['evaluation/last_result_status']=='failed'
    assert trainer.iteration==trainer.durable_updates==1 and trainer.completed_games_total==3
    assert trainer.iteration_status=='completed' and trainer.active_iteration is None
    assert 'worker failed' in trainer.evaluation_results_path.read_text(encoding='utf-8')


def test_real_cpp_training_next_update_matches_with_process_evaluation_enabled(tmp_path):
    from guandan_marvel import cpp_game
    from guandan_marvel.config import ModelConfig, TrainConfig
    from guandan_marvel.training.local import IPPOTrainer
    def run(name,enabled):
        cfg=TrainConfig(game=cpp_game.GAME_SPEC,device="cpu",seed=707,rollout_batch_size=1,collect_batch_size=1,mini_batches=1,total_iterations=2,max_sampler_bytes_per_rank=512*2**20,model=ModelConfig(architecture="v1.3",history_dim=8,feature_dim=8,decoder_layers=1,decoder_heads=1,state_layers=1,action_layers=1,fuse_layers=1,ffn_ratio=2))
        cfg.ppo.agent_epoch=1;cfg.compile.enabled=False;cfg.optimizer.use_muon=False
        trainer=IPPOTrainer(cfg,game=cpp_game,run_dir=tmp_path/name)
        hook=None
        if enabled:
            hook=TrainingEvaluatorHook(interval=1,snapshot_dir=tmp_path/name/'eval',evaluator_entrypoint='guandan_marvel.evaluation.workers:load_policy_identity_work',max_pending=2)
            trainer.set_evaluation_hook(hook)
        trainer.train_iteration();trainer.train_iteration()
        state={key:value.detach().clone() for key,value in trainer.agent_resume.state_dict().items()}
        result=(state,trainer.last_trajectory_num_steps,trainer.sampler.total_env_steps,trainer.sampler.total_game_steps,torch.get_rng_state().clone())
        if hook is not None:trainer.close_evaluation_hook(wait=True)
        return result
    control=run('control',False);evaluated=run('evaluated',True)
    assert control[1:4]==evaluated[1:4]
    assert torch.equal(control[4],evaluated[4])
    assert all(torch.equal(control[0][key],evaluated[0][key]) for key in control[0])
