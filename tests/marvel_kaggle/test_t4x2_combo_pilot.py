from __future__ import annotations
from copy import deepcopy
import ast,base64,hashlib,importlib.util,io,json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
import torch
from guandan_marvel import python_game
from guandan_marvel.kaggle.combo_delivery import (
    CAP_ITERATIONS, EVALUATION_SEED_BASE, PAYLOAD_NAME, RECIPE, RESULTS_NAME,
    make_combo_profiles, render_combo_notebook, validate_parent_policy,
)
from guandan_marvel.kaggle.common import atomic_json,canonical_hash
from guandan_marvel.kaggle.profiles import build_train_config
from guandan_marvel.kaggle.pilot_delivery import make_profiles
from guandan_marvel.context_schedule import curriculum_alpha,validate_context_curriculum
from guandan_marvel.training.schedules import compute_agent_schedule
from guandan_marvel.buffer import trajectory_memory_budget
from guandan_marvel.training.checkpoints import config_differences,serialize_config
from tests.marvel_kaggle.test_t4x2_learning_pilot import records_for

ROOT=Path(__file__).resolve().parents[2]


def test_combo_is_new_recipe_not_mutation_of_original_pilot():
    profiles=make_combo_profiles();p=profiles['precheck'];t=p['training']
    assert (t['base_clip_coef'],t['base_reg_coef'],t['base_lr'])==(.2,.01,.0001)
    assert (t['rollout_batch_size'],t['collect_batch_size'])==(48,40)
    assert t['checkpoint_interval']==50 and t['evaluation_interval']==0
    assert t['ntp_weight']==.02 and any(r['previous_result']is not None for r in t['training_context_schedule']['contexts'])
    assert len(t['training_context_schedule']['contexts'])==208
    assert p['ddp']=={'world_size':2,'backend':'nccl','require_cuda':True,'find_unused_parameters':False}
    assert p['object_collectives_backend']=='gloo' and p['checkpoint_io']['compression_level']==1
    assert p['pilot_observability']['maximum_iteration']==250
    assert profiles['train']['resume']['target_iterations']==250
    old=make_profiles();assert old['precheck']['training']['base_clip_coef']==.02
    assert old['train']['resume']['target_iterations']==50
    budget=trajectory_memory_budget(python_game.GAME_SPEC,rollout_batch_size=48,collect_batch_size=40,world_size=2)
    assert budget['sampler_bytes_per_rank']==1_553_490_464 < t['max_sampler_bytes_per_rank']


def test_profile_actually_builds_constant_schedules_and_migration_fields(tmp_path):
    cfg=build_train_config(make_combo_profiles()['precheck'],game=python_game,run_root=tmp_path,local_cpu_test=True)
    for i in (1,50,250,501,4000):
        schedule=compute_agent_schedule(cfg,iteration=i)
        assert (schedule.clip_coef,schedule.reg_coef,schedule.agent_lr)==(.2,.01,.0001)
    old=build_train_config(make_profiles()['precheck'],game=python_game,run_root=tmp_path,local_cpu_test=True)
    allowed,forbidden=config_differences(serialize_config(old),serialize_config(cfg))
    assert any('ppo.base_clip_coef:' in x for x in forbidden)
    assert any('training_context_curriculum.iteration_offset:'in x for x in forbidden)
    assert not cfg.belief.build_head and cfg.belief.belief_weight==0 and cfg.ntp_weight==.02


@pytest.mark.parametrize('offset',[-1,True,1.2])
def test_curriculum_offset_rejects_invalid_values(offset):
    c=make_combo_profiles()['precheck']['training']['training_context_curriculum'];c['iteration_offset']=offset
    with pytest.raises(ValueError,match='iteration_offset'):validate_context_curriculum(c)


def test_curriculum_continues_parent_progress_without_resetting_optimizer_counters():
    c=make_combo_profiles()['precheck']['training']['training_context_curriculum']
    assert validate_context_curriculum(c)==c
    assert curriculum_alpha(c,0)==.375
    assert curriculum_alpha(c,50)==1
    original={k:v for k,v in c.items()if k!='iteration_offset'}
    assert curriculum_alpha(original,0)==0
    assert curriculum_alpha(c,5)==curriculum_alpha(original,55)


def test_parent_gate_refuses_wrong_hash_and_old_architecture():
    with pytest.raises(ValueError,match='verified uploaded'):validate_parent_policy({},sha256='x'*64)
    from guandan_marvel.kaggle.combo_delivery import PARENT_SHA256
    with pytest.raises(ValueError,match='algorithm/architecture'):
        validate_parent_policy({'schema':'guandan-marvel-evaluation-checkpoint-v1','algorithm':'ippo','architecture':'v1.3','iteration':50},sha256=PARENT_SHA256)


def test_cpu_metadata_group_does_not_switch_cuda_tensor_backend(monkeypatch):
    import guandan_marvel.runtime.distributed as d
    sentinel=object();calls=[]
    monkeypatch.setattr(d.dist,'is_gloo_available',lambda:True)
    monkeypatch.setattr(d.dist,'new_group',lambda **kw:calls.append(kw)or sentinel)
    monkeypatch.setattr(d.dist,'get_backend',lambda group=None:'gloo'if group is sentinel else'nccl')
    def gather(out,value,group=None):calls.append({'group':group,'value':value});out[:]=[value,'peer']
    monkeypatch.setattr(d.dist,'all_gather_object',gather)
    base=d.DistributedContext(True,True,2,0,0,torch.device('cuda:0'))
    side=base.with_cpu_object_collectives()
    assert side.device==base.device and side.object_collective_backend=='gloo'
    assert d.dist.get_backend()=='nccl' and base.object_group is None
    assert side.all_gather_object({'test':1})==[{'test':1},'peer']
    assert calls[0]=={'backend':'gloo'} and calls[1]['group']is sentinel


def test_combo_notebook_uses_new_opaque_name_and_preserves_manifest_checks(tmp_path):
    h='a'*64;m='b'*64
    n=render_combo_notebook(ROOT,payload_hash=h,manifest_hash=m)
    code=''.join(next(c['source']for c in n['cells']if c['cell_type']=='code'));ast.parse(code)
    assert f'TRANSPORT_FILENAME = {PAYLOAD_NAME!r}'in code
    assert 'guandan_t4x2_combo_v1' in code and RESULTS_NAME in code
    assert f'EXPECTED_PILOT_PAYLOAD_SHA256 = {h!r}'in code
    assert f'EXPECTED_PILOT_MANIFEST_HASH = {m!r}'in code
    assert 'verify_unpacked_pilot' in code and 'restore_runtime_from_recovery' in code
    # Execute only the embedded pure transport support, not the bootstrap/training.
    namespace={};exec(code.split('PILOT_STARTED =',1)[0],namespace)
    dataset=tmp_path/'input';dataset.mkdir();data=b'{"tiny":"transport unit test only"}'
    body={'files':[{'path':'tiny.json','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}
    manifest={**body,'manifest_hash':canonical_hash(body)};stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w')as z:z.writestr('tiny.json',data);z.writestr('pilot_manifest.json',json.dumps(manifest))
    payload=namespace['TRANSPORT_MAGIC']+base64.b64encode(stream.getvalue())+b'\n'
    (dataset/PAYLOAD_NAME).write_bytes(payload)
    root=namespace['materialize_pilot_transport'](dataset,tmp_path/'run',hashlib.sha256(payload).hexdigest(),manifest['manifest_hash'])
    assert (root/'tiny.json').read_bytes()==data


@pytest.mark.parametrize('level',[0,10,True,-1])
def test_recovery_compression_invalid_level_fails_before_io(tmp_path,level):
    from guandan_marvel.kaggle.checkpoints import build_recovery_bundle
    with pytest.raises(ValueError,match='compression level'):
        build_recovery_bundle(run_root=tmp_path,destination=tmp_path/'r.zip',source_commit='a',
            source_archive_sha256='b',source_content_manifest_hash='c',package_id='test',
            status='partial',native_extension=None,compresslevel=level)


def test_fast_recovery_zip_keeps_byte_hashes_and_roundtrips(tmp_path,monkeypatch):
    from guandan_marvel.kaggle import checkpoints as c
    root=tmp_path/'run';cp=root/'checkpoints/iter000002';cp.mkdir(parents=True)
    data=b'not a deserialized checkpoint; archive test\0'*500
    (cp/'rank000.pt').write_bytes(data);atomic_json(cp/'manifest.json',{'manifest_hash':'m'})
    atomic_json(root/'checkpoint_manifest.json',{'latest':{'checkpoint':'checkpoints/iter000002','iteration':2}})
    monkeypatch.setattr(c,'load_ddp_manifest',lambda *a,**k:{'manifest_hash':'m'})
    output=c.build_recovery_bundle(run_root=root,destination=tmp_path/'recovery.zip',source_commit='a',
        source_archive_sha256='b',source_content_manifest_hash='c',package_id='test',
        status='partial',native_extension=None,compresslevel=1)
    with zipfile.ZipFile(output)as z:
        assert z.read('checkpoint/rank000.pt')==data
        manifest=json.loads(z.read('recovery_manifest.json'))
        for row in manifest['files']:
            b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']


@pytest.mark.parametrize('evaluation_only,fail_resume',[(False,False),(True,False),(False,True)])
def test_new_pipeline_caps_250_and_eval_only_never_trains(tmp_path,monkeypatch,evaluation_only,fail_resume):
    import guandan_marvel.kaggle.pilot as p
    from guandan_marvel.kaggle import bootstrap,telemetry
    from guandan_marvel.evaluation.manifests import build_deals_manifest
    dataset=tmp_path/'dataset';dataset.mkdir();profiles=make_combo_profiles()
    for k,v in profiles.items():atomic_json(dataset/(k+'.json'),v)
    deals=build_deals_manifest(counts={'training':0,'selection':128,'final':0},bases={'training':0,'selection':EVALUATION_SEED_BASE,'final':2_000_000_000})
    atomic_json(dataset/'development_deals.json',deals);atomic_json(dataset/'parent_run.json',{'training_deal_hashes':[],'initial_env_seeds':[]})
    (dataset/'warm_start_policy.pt').write_bytes(b'warm')
    meta={'pilot_id':'test-combo','manifest_hash':'manifest','evaluation_manifest_hash':deals['hash'],
        'evaluation_reserve_seconds':5400,'profiles':{k:k+'.json'for k in profiles},'recipe':RECIPE,
        'training_cap':{'absolute_iterations':250,'global_collect_batch_size':40},'results_archive_name':RESULTS_NAME,
        'source':{'git_commit':'a'*40,'archive_sha256':'b'*64,'content_manifest_hash':'c'*64}}
    monkeypatch.setattr(p,'validate_manifest',lambda _:meta)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2);monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'Tesla T4')
    monkeypatch.setattr(torch.distributed,'is_nccl_available',lambda:True)
    monkeypatch.setattr(bootstrap,'validate_recovery_directory',lambda r:json.loads((r/'recovery_manifest.json').read_text()))
    class Recorder:
        def __init__(self,*a,**k):pass
        def start(self):pass
        def close(self):pass
        def set_phase(self,*a):pass
    monkeypatch.setattr(telemetry,'TelemetryRecorder',Recorder);monkeypatch.setattr(telemetry,'runtime_environment',lambda:{})
    calls=[]
    def fake_process(command,**kw):
        if '--run-root'in command:
            run=Path(command[command.index('--run-root')+1]);profile=json.loads(Path(command[command.index('--profile')+1]).read_text())
            target=profile.get('resume',profile.get('training'))['target_iterations'];calls.append(target)
            if target in (3,250):
                prior=run.parent/('precheck'if target==3 else'resume_check')
                assert not (prior/'checkpoints').exists()
                assert (prior/'guandan-marvel-recovery.zip').is_file()
            if fail_resume and target==3:raise RuntimeError('injected failure')
            (run/'checkpoints/last').mkdir(parents=True)
            if profile['mode']!='resume':(run/'pilot_initial_policy.pt').write_bytes(b'control')
            (run/f'pilot_candidate_iter{target:06d}.pt').write_bytes(b'candidate')
            atomic_json(run/'progress.json',{'phase':'complete'});atomic_json(run/'checkpoint_manifest.json',{'latest':{'iteration':target,'checkpoint':'checkpoints/last'}})
            with zipfile.ZipFile(run/'guandan-marvel-recovery.zip','w')as z:
                z.writestr('recovery_manifest.json',json.dumps({'iteration':target,'checkpoint':'checkpoint/manifest.json'}))
                z.writestr('checkpoint/manifest.json',json.dumps({'global_counters':{'completed_games_total':target*40}}))
        else:
            assert command[command.index('--maximum-iteration')+1]=='250' and '--parent-provenance'in command
            shard=int(command[command.index('--shard')+1]);out=Path(command[command.index('--output')+1]);out.mkdir(exist_ok=True)
            from guandan_marvel.kaggle.common import sha256_file
            rows=records_for(pilot_tasks:=__import__('guandan_marvel.kaggle.pilot_evaluation',fromlist=['tasks_for']).tasks_for(deals))[shard::2]
            control=Path(command[command.index('--control')+1]);candidate=Path(command[command.index('--candidate')+1])
            for r in rows:
                r['candidate_checkpoint_sha256']=sha256_file(control if r['arm']=='control_vs_rule'else candidate)
                r['opponent_checkpoint_sha256']=sha256_file(control)if r['arm']=='candidate_vs_control'else None
            (out/f'games.gpu{shard}.jsonl').write_text('\n'.join(json.dumps(r)for r in rows),encoding='utf-8')
    monkeypatch.setattr(p,'run_process',fake_process)
    monkeypatch.setattr(p,'phase_health',lambda root,target,resumed,**kwargs:{'candidate':str(root/f'pilot_candidate_iter{target:06d}.pt'),'global_counters':{'completed_games_total':target*40}})
    source=tmp_path/'source';source.mkdir();out=tmp_path/'run/artifacts';recovery=None
    if evaluation_only:
        recovery=tmp_path/'recovery';(recovery/'artifacts').mkdir(parents=True)
        atomic_json(recovery/'recovery_manifest.json',{'iteration':250})
        atomic_json(recovery/'artifacts/pilot_run_contract.json',{'pilot_manifest_hash':meta['manifest_hash'],'cap_iterations':250,'global_collect_batch_size':40,'evaluation_manifest_hash':deals['hash'],'source':meta['source']})
        (recovery/'artifacts/pilot_initial_policy.pt').write_bytes(b'control')
        def export(r,destination,**kw):
            assert kw['target_iteration']==250;destination.write_bytes(b'candidate');return destination
        monkeypatch.setattr(p,'export_recovered_policy',export)
    result=p.run_pilot(dataset=dataset,output=out,source=source,recovery=recovery)
    assert result['authorized_iteration_cap']==250 and not result['automatic_extension_beyond_cap']
    assert (out.parent/RESULTS_NAME).exists()
    if fail_resume:
        assert calls==[2,3] and result['status']=='failed' and result['training_completed_iterations']==2
    else:
        assert calls==([]if evaluation_only else[2,3,250]) and result['status']=='complete'
        assert result['training_completed_games']==10000 and result['evaluation']['arms']['candidate_vs_rule']['games']==512
        assert not result['evaluation']['final_holdout_used']


@pytest.mark.slow
def test_real_two_rank_cpu_metadata_fast_checkpoint_and_resume(tmp_path,monkeypatch):
    """Explicit tiny CPU/Gloo compatibility only, not T4/CUDA certification."""
    from tests.marvel_training._ddp_launcher import launch_two_ranks
    from guandan_marvel.kaggle.pilot import safe_extract
    monkeypatch.setenv('GUANDAN_MARVEL_SOURCE_COMMIT','a'*40)
    monkeypatch.setenv('PYTHONIOENCODING','utf-8')
    profile=make_combo_profiles()['precheck'];t=profile['training']
    profile.update({'local_test_backend':'cpp','smoke_horizon':512})
    t.update({'rollout_batch_size':4,'collect_batch_size':4,'replay_buffer_scale':1,'mini_batches':1,'agent_epoch':1,
              'max_sampler_bytes_per_rank':256*1024*1024,'model':{'architecture':'v1.4-macro-fuse','history_dim':8,'feature_dim':8,'decoder_heads':1,'decoder_layers':1,'state_layers':1,'action_layers':1,'fuse_layers':1,'ffn_ratio':2}})
    profile_path=tmp_path/'start.json';atomic_json(profile_path,profile)
    worker=ROOT/'tests/marvel_kaggle/_pilot_worker_compat.py';run=tmp_path/'start'
    for r in launch_two_ranks(worker,'--profile',str(profile_path),'--output',str(run),cwd=ROOT,timeout=900):assert r.returncode==0,r.stdout
    recovery=tmp_path/'recovery';safe_extract(run/'guandan-marvel-recovery.zip',recovery)
    resume=make_combo_profiles()['resume_check'];resume['local_test_backend']='cpp';resume['smoke_horizon']=512
    path=tmp_path/'resume.json';atomic_json(path,resume);out=tmp_path/'resume'
    for r in launch_two_ranks(worker,'--profile',str(path),'--output',str(out),'--recovery',str(recovery),cwd=ROOT,timeout=900):assert r.returncode==0,r.stdout
    for rank in (0,1):
        rr=out/'ranks'/f'rank{rank:03d}'
        runtime=json.loads((rr/'pilot_runtime.json').read_text());assert runtime['object_collective_backend']=='gloo'
        rows=[json.loads(s)for s in(rr/'pilot_iterations.jsonl').read_text().splitlines()]
        assert rows[-1]['iteration']==3 and rows[-1]['effective_optimizer_updates']>0
        assert 'cache_prefill_and_allocation' in rows[-1]['rollout_stage_host_seconds']
        io=json.loads((rr/'checkpoint_io.jsonl').read_text().splitlines()[-1]);assert io['compression_level']==1
        events=[json.loads(s)for s in(rr/'diagnostic.jsonl').read_text().splitlines()]
        assert any(e.get('event')=='resumed'and e['decision']in('compatible_resume','exact_resume')for e in events)
