"""Independent, frozen-snapshot evaluation. Never calls a trainer or optimizer."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from .common import read_json,validate_plan,sha,atomic_json,append
ARMS=('control_vs_rule','candidate_vs_rule','candidate_vs_control')

def expected_tasks(manifest,arms):
    from guandan_marvel.kaggle.pilot_evaluation import tasks_for
    if not arms or set(arms)-set(ARMS):raise ValueError('unknown evaluation arms')
    return [t for t in tasks_for(manifest)if t['arm']in arms]

def assert_outcome(row):
    ranking=row['ranking']
    if sorted(ranking)!=[1,2,3,4]:raise ValueError('invalid terminal ranking')
    first=ranking.index(1);winner=first%2;mate=ranking[(first+2)%4]
    points={2:3,3:2,4:1}[mate];name={2:'DOUBLE_DOWN',3:'HEAD_THIRD',4:'HEAD_LAST'}[mate]
    if row['winner_team']!=winner or row['outcome_class']!=name:raise ValueError('ranking/outcome mismatch')
    reward=points if row['candidate_team']==winner else-points
    if row['candidate_team_reward']!=reward or row['candidate_win']!=(reward>0):raise ValueError('reward/win mismatch')
    if row['rewards']!=[float(points if p%2==winner else-points)for p in range(4)]:raise ValueError('terminal reward vector mismatch')

def evaluate_shard(*,plan,manifest_path,control_path,candidate_path,output,arms,shard,expected_iteration,
                   parent_provenance=None,testing=False,device=None):
    import torch
    from guandan_marvel import cpp_game
    from guandan_marvel.evaluation.checkpoints import load_v2_policy
    from guandan_marvel.evaluation.host import play_game
    validate_plan(plan,testing=testing)
    if not testing:
        if torch.cuda.device_count()!=2 or any('T4'not in torch.cuda.get_device_name(i)for i in (0,1)):raise RuntimeError('two T4 GPUs required; no CPU fallback')
        if device is not None or shard not in (0,1):raise ValueError('production devices are assigned by shard')
        device=f'cuda:{shard}'
    else:device=device or'cpu'
    if not testing:
        from .pipeline import verify_driver_files
        verify_driver_files(plan)
    torch.set_num_threads(1)
    if str(device).startswith('cuda'):torch.cuda.set_device(device)
    if int(cpp_game.GAME_SPEC.max_game_horizon)!=plan['horizon']:raise ValueError('native horizon differs from signed evaluation contract')
    if not plan['start_iteration']<=expected_iteration<=plan['target_iteration']and not testing:raise ValueError('evaluation checkpoint out of bounds')
    control=load_v2_policy(control_path,device=device,expected_algorithm='ippo')
    candidate=load_v2_policy(candidate_path,device=device,expected_algorithm='ippo')
    if candidate.iteration!=expected_iteration:raise ValueError('evaluation loaded the wrong checkpoint iteration')
    expected_control=0 if sha(control_path)==plan.get('origin_control_sha256')else plan['start_iteration']
    if not testing and (control.iteration!=expected_control or sha(control_path)not in (plan['anchor_sha256'],plan['origin_control_sha256'])):raise ValueError('control identity changed')
    if candidate.architecture!=control.architecture or candidate.metadata['source_hash']!=control.metadata['source_hash']:raise ValueError('policy source/architecture mismatch')
    if not testing and candidate.metadata['source_hash']!=plan['base_source']['core_source_hash']:raise ValueError('policy uses a different frozen training core')
    # Validate native and both serialized GameSpecs before even the first game.
    for path in (control_path,candidate_path):
        header=torch.load(path,map_location='cpu',weights_only=True)
        if int(header['resolved_config']['game']['max_game_horizon'])!=plan['horizon']:raise ValueError('policy/game horizon mismatch')
        if not testing and header['resolved_config']['collect_batch_size']!=plan['collect_global']:raise ValueError('policy collection semantics changed')
    manifest=read_json(manifest_path);tasks=expected_tasks(manifest,arms);mine=tasks[shard::2]if not testing else tasks[shard::2]
    training_hashes=set();initial_seeds=set()
    for artifact in (control,candidate):
        p=artifact.training_data_provenance or{}
        if not p.get('complete'):
            # The byte-bound V4 control is a zero-update warm-start snapshot; its
            # inherited 1,000-game provenance lives in the verified parent file.
            known_zero_control=(artifact is control and control.iteration==0
                and (testing or control.sha256==plan['origin_control_sha256'])
                and p.get('deal_count')==0 and parent_provenance is not None)
            if not known_zero_control:raise ValueError('incomplete actual training provenance')
        training_hashes.update(p.get('deal_hashes',[]));initial_seeds.update(p.get('initial_env_seeds',[]))
        if p.get('initial_env_seed')is not None:initial_seeds.add(p['initial_env_seed'])
    if parent_provenance is not None:
        p=read_json(parent_provenance);training_hashes.update(p['training_deal_hashes']);initial_seeds.update(p.get('initial_env_seeds',[]))
    output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/f'games.gpu{shard}.jsonl'
    if path.exists():raise FileExistsError('never append evaluation results to stale identities')
    started=time.monotonic();state={'status':'running','gpu_index':shard,'expected_games':len(mine),'max_steps':plan['horizon'],
        'control_sha256':control.sha256,'candidate_sha256':candidate.sha256,'manifest_hash':manifest['hash'],'plan_hash':plan['plan_hash']}
    atomic_json(output/f'worker{shard}.json',state)
    try:
        for index,t in enumerate(mine,1):
            if t['deal']['seed']in initial_seeds:raise ValueError('evaluation seed overlaps training initialization')
            arm=t['arm'];artifact=control if arm=='control_vs_rule'else candidate
            row=play_game(deal=t['deal'],rotation=t['rotation'],opponent_id=arm,
                opponent_kind='v2'if arm=='candidate_vs_control'else'rule',candidate=artifact,
                opponent_v2=control if arm=='candidate_vs_control'else None,temperature=0.,device=device,
                max_steps=plan['horizon'])
            if row['status']!='complete'or row['illegal_count']or row['truncated_count']:raise RuntimeError('evaluation game incomplete/illegal/truncated')
            if row['base_deal_sha256']in training_hashes:raise ValueError('physical evaluation deal overlaps training')
            assert_outcome(row)
            row.update({'task_id':t['task_id'],'arm':arm,'gpu_index':shard,'evaluation_manifest_hash':manifest['hash'],
                'continuation_plan_hash':plan['plan_hash'],'evaluation_horizon':plan['horizon']})
            append(path,row)
            if index%8==0 or index==len(mine):print(f'[CONTINUE EVAL gpu{shard}] {index}/{len(mine)} games, elapsed={time.monotonic()-started:.1f}s',flush=True)
        state.update({'status':'complete','games':len(mine),'elapsed_seconds':time.monotonic()-started})
    except BaseException as e:
        state.update({'status':'failed','error':repr(e),'elapsed_seconds':time.monotonic()-started});raise
    finally:atomic_json(output/f'worker{shard}.json',state)

def collect_records(directory,tasks,control_path,candidate_path,manifest_hash,plan):
    from guandan_marvel.kaggle.pilot_evaluation import check_records
    records=[]
    for rank in (0,1):
        state=read_json(Path(directory)/f'worker{rank}.json')
        if state['status']!='complete':raise ValueError('incomplete evaluation worker')
        path=Path(directory)/f'games.gpu{rank}.jsonl'
        records.extend(json.loads(s)for s in path.read_text(encoding='utf-8').splitlines()if s.strip())
    check_records(records,tasks,control_sha=sha(control_path),candidate_sha=sha(candidate_path))
    for row in records:
        assert_outcome(row)
        if row.get('evaluation_manifest_hash')!=manifest_hash or row.get('continuation_plan_hash')!=plan['plan_hash']or row.get('evaluation_horizon')!=plan['horizon']:raise ValueError('evaluation contract identity changed')
    return records

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arms',required=True);p.add_argument('--shard',type=int,choices=(0,1),required=True)
    p.add_argument('--expected-iteration',type=int,required=True);p.add_argument('--parent-provenance',type=Path)
    p.add_argument('--local-test',action='store_true');p.add_argument('--device')
    a=p.parse_args(argv);plan=read_json(a.plan)
    if a.local_test!=bool(plan.get('test_only')):raise ValueError('production/test evaluation mismatch')
    evaluate_shard(plan=plan,manifest_path=a.manifest,control_path=a.control,candidate_path=a.candidate,output=a.output,
        arms=a.arms.split(','),shard=a.shard,expected_iteration=a.expected_iteration,parent_provenance=a.parent_provenance,
        testing=a.local_test,device=a.device)
if __name__=='__main__':main()
