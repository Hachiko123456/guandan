"""Run the M06 repair exercise with a provenance-complete v5 C++ checkpoint and process hook."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from guandan_marvel import cpp_game
from guandan_marvel.config import ModelConfig,TrainConfig
from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent,load_v2_policy
from guandan_marvel.evaluation.common import atomic_write_json
from guandan_marvel.evaluation.hooks import TrainingEvaluatorHook
from guandan_marvel.evaluation.manifests import FinalHoldoutAudit,build_deals_manifest
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore,PromotionPolicy
from guandan_marvel.training.checkpoints import capture_rng_state
from guandan_marvel.training.local import IPPOTrainer

DEFAULT_LEGACY=ROOT/'runs/a07probe/local_fast/ippo/ippo_update_000001_90453d2696194fc2b755820939dcbed2_periodic.pt'

def digest_model(model):
 h=hashlib.sha256()
 for name,tensor in sorted(model.state_dict().items()):
  value=tensor.detach().cpu().contiguous();h.update(name.encode());h.update(value.numpy().tobytes())
 return h.hexdigest()
def digest_object(value):return hashlib.sha256(pickle.dumps(value,protocol=5)).hexdigest()
def digest_rng(state):
 h=hashlib.sha256();h.update(repr(state["python"]).encode());npstate=state["numpy"];h.update(str(npstate[0]).encode());h.update(npstate[1].tobytes());h.update(repr(npstate[2:]).encode());h.update(torch.as_tensor(state["torch_cpu"],dtype=torch.uint8).cpu().numpy().tobytes());
 for item in state.get("torch_cuda") or []:h.update(torch.as_tensor(item,dtype=torch.uint8).cpu().numpy().tobytes())
 return h.hexdigest()
def training_state(trainer):
 state=trainer.sampler.state_dict()
 return {'model':digest_model(trainer.agent_resume),'merged':digest_model(trainer.agent_merged),'optimizer':digest_object(trainer.agent_optimizer.state_dict()),'rng':digest_rng(capture_rng_state()),'sampler_envs':hashlib.sha256(state['envs']).hexdigest(),'sampler_counters':[trainer.sampler.total_env_steps,trainer.sampler.total_game_steps,trainer.completed_games_total],'training_provenance_hash':trainer.sampler.training_data_provenance()['hash']}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--legacy',type=Path,default=DEFAULT_LEGACY);args=parser.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 deals=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':95001,'selection':1_000_000_000,'final':1_500_000_000},level_ranks=(5,),leader_seats=(0,))
 atomic_write_json(out/'deals_manifest.json',deals)
 cfg=TrainConfig(game=cpp_game.GAME_SPEC,device='cpu',seed=95001,rollout_batch_size=1,collect_batch_size=1,mini_batches=1,total_iterations=2,max_sampler_bytes_per_rank=512*2**20,model=ModelConfig(architecture='v1.3',history_dim=8,feature_dim=8,decoder_layers=1,decoder_heads=1,state_layers=1,action_layers=1,fuse_layers=1,ffn_ratio=2))
 cfg.ppo.agent_epoch=1;cfg.compile.enabled=False;cfg.optimizer.use_muon=False
 trainer=IPPOTrainer(cfg,game=cpp_game,run_dir=out/'training')
 trainer.train_iteration();champion_path=trainer.save_checkpoint(out/'checkpoints/champion-iter000001-v5.pt');champion=load_v2_policy(champion_path)
 legacy=load_legacy_v1_opponent(args.legacy)
 champion_store=ChampionStore(out/'champion_manifest.json');champion_manifest=champion_store.bootstrap(champion.identity,reason='M06 repair provenance-complete smoke champion bootstrap')
 pool=OpponentPoolStore(out/'opponent_pool_manifest.json');pool.initialize(sampling_seed=20261010)
 pool.add_entry({'id':'champion-generation-1','kind':'champion','checkpoint':str(champion.path),'policy_identity':champion.identity,'weight':1.0,'source_champion_manifest_hash':champion_manifest['hash']})
 pool.add_entry({'id':'rule-anchor-v1','kind':'rule','weight':0.25,'version':'v2-public-row-heuristic-v1'})
 pool.add_entry({'id':'legacy-v1-anchor','kind':'legacy_v1','checkpoint':str(legacy.path),'legacy_identity':legacy.identity,'weight':0.25})
 final_audit=FinalHoldoutAudit(out/'final_holdout_audit.json',deals_manifest=deals);final_audit.initialize()
 hook=TrainingEvaluatorHook(interval=2,snapshot_dir=out/'async_hook',evaluator_entrypoint='guandan_marvel.evaluation.workers:selection_evaluation_work',evaluator_config={'output_dir':str(out/'async_selection'),'champion_manifest':str(champion_store.path),'opponent_pool_manifest':str(pool.path),'legacy_checkpoint':str(legacy.path),'deals_manifest':str(out/'deals_manifest.json'),'device':'cpu','temperature':0.0,'promotion_policy':PromotionPolicy().to_dict(),'provenance_policy':'formal'},max_pending=1)
 trainer.set_evaluation_hook(hook)
 trainer.train_iteration()
 before=training_state(trainer)
 events=trainer.close_evaluation_hook(wait=True)
 after=training_state(trainer)
 if len(events)!=1 or events[0]['status']!='completed':raise RuntimeError(f'async evaluation failed: {events}')
 worker=events[0]['result'];result=json.loads(Path(worker['result_path']).read_text(encoding='utf-8'))
 games=[json.loads(line) for line in Path(result['games_path']).read_text(encoding='utf-8').splitlines()]
 summary={'schema':'guandan-marvel-m06-repair-exercise-v1','training_state_unchanged_during_async_wait':before==after,'training_state_before':before,'training_state_after':after,'async_event':events[0],'selection_result':result,'games':{'count':len(games),'token_steps':sum(row['token_steps'] for row in games),'committed_steps':sum(row['committed_steps'] for row in games),'all_engine_counts_match':all(row['token_steps']==row['engine_finish_token_step'] and row['committed_steps']==row['engine_finish_committed_step'] for row in games),'all_committed_less_than_tokens':all(row['committed_steps']<row['token_steps'] for row in games),'illegal_count':sum(row['illegal_count'] for row in games),'truncated_count':sum(row['truncated_count'] for row in games)},'training_provenance':trainer.sampler.training_data_provenance(),'final_holdout_state':final_audit.load()['state'],'pool_recovery_hash':OpponentPoolStore(pool.path).recover()['hash']}
 atomic_write_json(out/'repair_exercise_summary.json',summary)
 print(json.dumps(summary,ensure_ascii=False,indent=2))
 if not summary['training_state_unchanged_during_async_wait'] or result['status']!='complete' or result['promotion_decision']['status']!='inconclusive' or summary['final_holdout_state']!='unused':return 1
 return 0
if __name__=='__main__':raise SystemExit(main())
