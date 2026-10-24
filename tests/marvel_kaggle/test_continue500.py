"""Continuation safety contracts; no remote training or holdout access."""
import copy,io,json,zipfile
from pathlib import Path
import pytest
from tools.continuation500 import common as c
from tools.continuation500.pipeline import execute_schedule,write_reports

def signed(x,key='plan_hash'):
    x=copy.deepcopy(x);x.pop(key,None);x[key]=c.canonical(x);return x

def production_plan(**changes):
    p=dict(schema=c.SCHEMA,start_iteration=250,target_iteration=500,monitor_iteration=375,
        collect_global=40,horizon=1024,wall_seconds_cap=14400,evaluation_reserve_seconds=5400,
        runtime_source_path=c.RUNTIME_SOURCE_PATH,expected_python_version=c.EXPECTED_PYTHON,expected_backend_hash=c.EXPECTED_BACKEND_HASH,
        base_source=c.BASE_SOURCE,base_recovery_sha256=c.BASE_RECOVERY,anchor_sha256=c.ANCHOR_SHA,
        origin_control_sha256=c.ORIGIN_SHA,v4_payload_sha256=c.V4_PAYLOAD_SHA,
        v4_manifest_hash='d9e7563e131060f7604122a61232071eec26e8f718128ea7164b9004cf758433',
        fabledan='NOT_RUN',final_holdout='NOT_TOUCHED',driver_files=[],driver_archive_sha256='x'*64)
    p.update(changes);return signed(p)

def fake_checkpoint(root,plan,iteration=500):
    root.mkdir();(root/'artifacts').mkdir();(root/'checkpoint').mkdir()
    cfg=dict(collect_batch_size=40,rollout_batch_size=48,ntp_weight=.02,game={'max_game_horizon':1024},
        ppo=dict(base_clip_coef=.2,base_reg_coef=.01,base_lr=.0001))
    c.atomic_json(root/'artifacts/resolved_config.json',{'train_config':cfg})
    c.atomic_json(root/'artifacts/pilot_run_contract.json',c.run_contract(plan))
    ranks=[dict(agent_merged_state_hash='m',agent_resume_state_hash='r',source_hash=c.BASE_SOURCE['core_source_hash'],semantic_contract_hash='s') for _ in range(2)]
    ddp=signed(dict(complete=True,iteration=iteration,world_size=2,global_counters={'completed_games_total':iteration*40},
        source_hash=c.BASE_SOURCE['core_source_hash'],rank_shards=ranks),'manifest_hash')
    c.atomic_json(root/'checkpoint/manifest.json',ddp)
    files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=c.sha(p))for p in root.rglob('*')if p.is_file()]
    rec=signed(dict(iteration=iteration,files=files,checkpoint='checkpoint/manifest.json',checkpoint_manifest_hash=ddp['manifest_hash'],
        source_archive_sha256=c.BASE_SOURCE['archive_sha256'],source_content_manifest_hash=c.BASE_SOURCE['content_manifest_hash'],
        source_commit=c.BASE_SOURCE['git_commit']),'manifest_hash')
    c.atomic_json(root/'recovery_manifest.json',rec);return ddp

@pytest.mark.parametrize('changes',[{'target_iteration':501},{'collect_global':20},{'horizon':4096},
    {'wall_seconds_cap':20000},{'evaluation_reserve_seconds':0},{'anchor_sha256':'wrong'},
    {'base_recovery_sha256':'wrong'},{'base_source':{}},{'runtime_source_path':'/wrong/path'},
    {'expected_python_version':'3.13.1'},{'expected_backend_hash':'wrong'},{'fabledan':'RUN'},{'final_holdout':'USED'}])
def test_signed_but_unauthorized_plan_rejected(changes):
    with pytest.raises(ValueError):c.validate_plan(production_plan(**changes))

def test_only_plan_never_accepted_by_production():
    p=production_plan(test_only=True,target_iteration=4,start_iteration=2)
    with pytest.raises(ValueError,match='test plan'):c.validate_plan(p)
    c.validate_plan(p,testing=True)

def test_real_collect40_counts_500_is_20000_and_contract(tmp_path):
    p=production_plan();root=tmp_path/'state';fake_checkpoint(root,p)
    _,ddp,cfg=c.checkpoint_info(root,p)
    assert ddp['global_counters']['completed_games_total']==20000
    assert cfg['collect_batch_size']==40
    c.atomic_json(root/'artifacts/pilot_run_contract.json',dict(c.run_contract(p),anchor_sha256='bad'))
    with pytest.raises(ValueError):c.checkpoint_info(root,p)

def test_recovery_corruption_and_counter_lies_are_rejected(tmp_path):
    p=production_plan();root=tmp_path/'state';ddp=fake_checkpoint(root,p)
    bundle=tmp_path/'state.zip'
    with zipfile.ZipFile(bundle,'w')as z:
        for f in root.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(root))
    assert c.verify_recovery_zip(bundle,p)[1]['iteration']==500
    (root/'checkpoint/manifest.json').write_text('corrupted')
    with pytest.raises(ValueError,match='member mismatch'):c.checkpoint_info(root,p)

@pytest.mark.parametrize('name',['../escape','/absolute','C:/evil','a\\evil'])
def test_extraction_cannot_escape(tmp_path,name):
    b=io.BytesIO()
    info=zipfile.ZipInfo('placeholder');info.filename=name;info.orig_filename=name
    with zipfile.ZipFile(b,'w')as z:z.writestr(info,b'x')
    b.seek(0)
    with pytest.raises(ValueError):c.extract_safe(b,tmp_path/'out')

def run_schedule(start=250,fail=None):
    calls=[];current=[start]
    def evaluate(name,kind,arms,seconds,**kw):
        calls.append(('eval',name,current[0]))
        if name==fail:raise RuntimeError(name)
        return [name]
    def train(target,name,**kw):
        calls.append(('train',name,target));current[0]=target
        if name==fail:raise TimeoutError('saved budget stop')
    def summarize(name,*args,**kw):calls.append(('summary',name,current[0]))
    try:execute_schedule(production_plan(),get_iteration=lambda:current[0],evaluate=evaluate,train_to=train,summarize_node=summarize)
    except (RuntimeError,TimeoutError):pass
    return calls,current[0]

def test_production_schedule_first_eval_then_real_update_save_resume_and_eval():
    calls,end=run_schedule()
    assert calls[:4]==[('eval','preflight_before_training',250),('train','update_save_gate',252),
        ('train','resume_update_gate',253),('eval','preflight_new_checkpoint',253)]
    assert end==500 and ('eval','monitor_375',375)in calls
    assert calls[-2:]==[('eval','final_500',500),('summary','final_500',500)]
    assert max(t[2]for t in calls if t[0]=='train')==500

def test_preflight_failure_causes_zero_new_training():
    calls,end=run_schedule(fail='preflight_before_training')
    assert end==250 and not any(t[0]=='train'for t in calls)

def test_new_export_failure_prevents_long_training():
    calls,end=run_schedule(fail='preflight_new_checkpoint')
    assert end==253 and not any(t[1]=='train_to_monitor'for t in calls)

def test_midpoint_failure_retains_375_and_never_trains_to500():
    calls,end=run_schedule(fail='monitor_375')
    assert end==375 and not any(t[1]=='train_to_500'for t in calls)

@pytest.mark.parametrize('start',[375,400,498,499,500])
def test_resume_never_goes_backwards_or_beyond500(start):
    calls,end=run_schedule(start)
    assert end==500
    assert all(start<t[2]<=500 for t in calls if t[0]=='train')
    if start==500:assert not any(t[0]=='train'for t in calls)

def test_budget_stop_stops_schedule_at_saved_actual_iteration():
    calls,end=run_schedule(fail='train_to_monitor')
    assert end==375 and calls[-1][1]=='train_to_monitor'

def test_reports_survive_history_conflicts_and_keep_inconclusive_separate(tmp_path):
    root=tmp_path/'artifacts';root.mkdir()
    for segment,value in [('a',1),('b',2)]:
        c.append(root/'phases'/segment/'metrics.jsonl',{'statistics/iterations':251,'loss':value})
    result={'status':'complete','training':{'status':'complete'},'evaluations':{'final_500':{'arms':{'x':{}},'verdict':'INCONCLUSIVE'}}}
    archive=write_reports(root,result)
    assert archive.is_file()and result['status']=='failed'
    assert result['history_conflicts']==[251]
    assert result['training']['status']=='complete'

def test_default_host_horizon_and_explicit_invalid_value_fail_early(monkeypatch):
    from guandan_marvel.evaluation import host
    from types import SimpleNamespace
    monkeypatch.setattr(host.cpp_game,'GAME_SPEC',SimpleNamespace(max_game_horizon=1024))
    for value in (4096,True,0,-1):
        with pytest.raises((ValueError,TypeError)):
            host.play_game(deal={},rotation=0,opponent_id='x',opponent_kind='rule',candidate=None,max_steps=value)


def test_newest_durable_recovers_intermediate_progress_despite_later_error(tmp_path):
    from tools.continuation500.pipeline import newest_durable
    plan=production_plan();root=tmp_path/'artifacts';root.mkdir()
    for iteration,corrupt in [(375,False),(450,True)]:
        phase=root/'phases'/str(iteration);phase.mkdir(parents=True)
        state=tmp_path/str(iteration);fake_checkpoint(state,plan,iteration)
        with zipfile.ZipFile(phase/'guandan-marvel-recovery.zip','w')as z:
            for f in state.rglob('*'):
                if f.is_file():z.writestr(f.relative_to(state).as_posix(),b'corrupt'if corrupt and f.name=='resolved_config.json'else f.read_bytes())
    adopted,errors=newest_durable(root,plan,253)
    assert adopted[0]==375 and errors
    adopted,errors=newest_durable(root,plan,400)
    assert adopted is None and errors # cannot roll back to 375


def test_reports_archive_partial_worker_log_without_masking_original_failure(tmp_path):
    root=tmp_path/'artifacts';root.mkdir()
    f=root/'phases/train/metrics.jsonl';f.parent.mkdir(parents=True)
    f.write_text('{"statistics/iterations":251}\n{"partial":',encoding='utf8')
    result={'status':'failed','error':'worker died','evaluations':{}}
    archive=write_reports(root,result)
    assert archive.is_file() and result['error']=='worker died'
    assert result['diagnostic_jsonl_errors']
    assert len((root/'training_history.jsonl').read_text().splitlines())==1


def test_external_export_preserves_actual_global_counters(monkeypatch,tmp_path):
    import torch
    import guandan_marvel.kaggle.checkpoints as cp
    state={'weight':torch.tensor([1.0,2.0])};h=cp._state_dict_hash(state)
    payload={k:'fixture'for k in ('algorithm','architecture','protocol_version','backend','backend_identity','resolved_config_hash',
        'source_fingerprint','source_hash','source_parent_commit')}
    payload.update(iteration=500,agent_state_dict=state,resolved_config={'collect_batch_size':40})
    ddp={'rank_shards':[{'agent_merged_state_hash':h}],'training_data_provenance':{'complete':True},
        'manifest_hash':'d'*64,'global_counters':{'completed_games_total':20000}}
    monkeypatch.setattr(c,'checkpoint_info',lambda *a,**k:({'checkpoint':'checkpoint/manifest.json'},ddp,{}))
    monkeypatch.setattr(cp,'load_rank_checkpoint_for_resume',lambda *a,**k:(payload,ddp))
    path=c.export_policy(tmp_path,tmp_path/'policy.pt',production_plan())
    saved=torch.load(path,weights_only=True)
    assert saved['iteration']==500
    assert saved['distributed_training']['global_counters']['completed_games_total']==20000
    assert saved['training_resume_allowed']is False


@pytest.mark.parametrize('iteration,fail_first',[(250,True),(500,False),(500,True)])
def test_pipeline_wrapper_preserves_recovery_and500_never_calls_train(tmp_path,monkeypatch,iteration,fail_first):
    """Mocked process/hardware boundaries; real production orchestration/reporting."""
    import torch
    import tools.continuation500.pipeline as p
    import tools.continuation500.evaluate as ev
    import guandan_marvel.kaggle.pilot as legacy
    import guandan_marvel.kaggle.telemetry as telemetry
    from guandan_marvel import cpp_game
    from guandan_marvel.evaluation.manifests import build_deals_manifest
    from guandan_marvel.kaggle.pilot_evaluation import tasks_for
    from tests.marvel_kaggle.test_t4x2_learning_pilot import records_for
    from types import SimpleNamespace
    plan=production_plan();dataset=tmp_path/'input';dataset.mkdir()
    c.atomic_json(dataset/'plan.json',plan)
    (dataset/'anchor_iter250.pt').write_bytes(b'anchor')
    with zipfile.ZipFile(dataset/'base_recovery.zip','w')as z:z.writestr('artifacts/pilot_initial_policy.pt',b'origin')
    manifest=build_deals_manifest(counts={'training':0,'selection':4,'final':0},bases={'training':0,'selection':1822200000,'final':2000000000})
    for name in ('preflight','monitor','final'):c.atomic_json(dataset/(name+'_deals.json'),manifest)
    state=tmp_path/'state';state.mkdir();source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(p,'setup_runtime',lambda *a:(dataset,source,state,dataset/'base_recovery.zip',
        {'iteration':iteration,'global_counters':{'completed_games_total':iteration*40}},{}))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'Tesla T4')
    monkeypatch.setattr(torch.distributed,'is_nccl_available',lambda:True)
    monkeypatch.setattr(torch.distributed,'is_gloo_available',lambda:True)
    monkeypatch.setattr(cpp_game,'GAME_SPEC',SimpleNamespace(max_game_horizon=1024))
    monkeypatch.setattr(p.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*1024**3))
    def fake_sha(path):
        if Path(path).name=='anchor_iter250.pt':return c.ANCHOR_SHA
        if Path(path).name=='origin_control.pt':return c.ORIGIN_SHA
        return c.sha(path)
    monkeypatch.setattr(p,'sha',fake_sha);monkeypatch.setattr(ev,'sha',fake_sha)
    def export(state,destination,plan):destination.write_bytes(b'candidate');return destination
    monkeypatch.setattr(p,'export_policy',export)
    class Recorder:
        def __init__(self,*a,**k):pass
        def start(self):pass
        def close(self):pass
        def set_phase(self,*a):pass
    monkeypatch.setattr(telemetry,'TelemetryRecorder',Recorder)
    monkeypatch.setattr(telemetry,'runtime_environment',lambda:{})
    import guandan_marvel.training.checkpoints as training_cp
    monkeypatch.setattr(training_cp,'build_backend_identity',lambda game:{'hash':c.EXPECTED_BACKEND_HASH})
    training_calls=[]
    def fake_process(command,**kwargs):
        if '--run-root'in command:training_calls.append(command);raise AssertionError('training forbidden in this test')
        if fail_first:raise RuntimeError('injected early evaluation failure')
        get=lambda key:command[command.index(key)+1]
        shard=int(get('--shard'));out=Path(get('--output'));arms=get('--arms').split(',')
        tasks=[t for t in tasks_for(c.read_json(get('--manifest')))if t['arm']in arms][shard::2]
        control,candidate=Path(get('--control')),Path(get('--candidate'))
        records=records_for(tasks)
        for row in records:
            row.update(candidate_checkpoint_sha256=fake_sha(control if row['arm']=='control_vs_rule'else candidate),
                opponent_checkpoint_sha256=fake_sha(control)if row['arm']=='candidate_vs_control'else None,
                ranking=[1,3,2,4],winner_team=0,candidate_team=0,outcome_class='DOUBLE_DOWN',
                candidate_win=True,candidate_team_reward=3,rewards=[3.,-3.,3.,-3.],token_steps=400,
                continuation_plan_hash=plan['plan_hash'],evaluation_horizon=1024,evaluation_manifest_hash=manifest['hash'])
            c.append(out/f'games.gpu{shard}.jsonl',row)
        c.atomic_json(out/f'worker{shard}.json',{'status':'complete'})
    monkeypatch.setattr(legacy,'run_process',fake_process)
    # The setup boundary is mocked; provide the binding receipt it normally writes.
    run_root=tmp_path/'run';run_root.mkdir();c.atomic_json(run_root/'runtime_source_binding.json',{'unit_test':True})
    result=p.run_pipeline(dataset=dataset,run_root=run_root)
    assert not training_calls
    assert result['recovery']['iteration']==iteration
    assert Path(result['results_zip']).is_file()
    if fail_first:
        assert result['status']=='failed' and result['failed_phase']=='preflight_before_training'
        assert result['training']['status']==('complete'if iteration==500 else'restored')
    else:
        assert result['status']=='complete'
        assert result['training']['completed_games_total']==20000
        assert result['training']['new_games_this_session']==0
        assert 'final_500'in result['evaluations']


def test_external_driver_tamper_is_detected_before_worker(monkeypatch,tmp_path):
    import tools.continuation500.pipeline as p
    root=tmp_path/'continuation500';root.mkdir();f=root/'worker.py';f.write_bytes(b'original')
    monkeypatch.setattr(p,'__file__',str(root/'pipeline.py'))
    plan={'driver_files':[{'path':'continuation500/worker.py','bytes':8,'sha256':c.sha(f)}]}
    p.verify_driver_files(plan)
    f.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='driver identity'):p.verify_driver_files(plan)


def test_legacy_recovery_export_also_derives_collect40_not_hardcoded20(monkeypatch,tmp_path):
    import torch
    import guandan_marvel.kaggle.checkpoints as cp
    from guandan_marvel.kaggle.pilot import export_recovered_policy
    from guandan_marvel.kaggle import bootstrap
    monkeypatch.setattr(bootstrap,'validate_recovery_directory',lambda r:c.read_json(r/'recovery_manifest.json'))
    c.atomic_json(tmp_path/'recovery_manifest.json',{'checkpoint':'checkpoint/manifest.json'})
    payload={k:'fixture'for k in ('algorithm','architecture','protocol_version','backend','backend_identity','resolved_config_hash',
        'source_fingerprint','source_hash','source_parent_commit')}
    payload.update(iteration=250,agent_state_dict={'weight':torch.ones(1)},resolved_config={'collect_batch_size':40})
    manifest={'iteration':250,'global_counters':{'completed_games_total':10000},'training_data_provenance':{'complete':True}}
    monkeypatch.setattr(cp,'load_rank_checkpoint_for_resume',lambda *a,**k:(payload,manifest))
    path=export_recovered_policy(tmp_path,tmp_path/'export.pt',source_commit='a'*40,target_iteration=250)
    assert torch.load(path,weights_only=True)['resolved_config']['collect_batch_size']==40
    payload['resolved_config']['collect_batch_size']=40.5
    with pytest.raises(ValueError,match='collect_batch_size'):
        export_recovered_policy(tmp_path,tmp_path/'bad.pt',source_commit='a'*40,target_iteration=250)



def test_binding_installs_at_original_path_and_reuses_without_overwrite(tmp_path):
    from tools.continuation500.pipeline import bind_original_source
    staged=tmp_path/'staged';staged.mkdir();(staged/'core.py').write_bytes(b'frozen')
    target=(tmp_path/'original/source').resolve()
    def check(p):
        if (p/'core.py').read_bytes()!=b'frozen':raise ValueError('core changed')
    assert bind_original_source(staged,target,check)==target
    stamp=(target/'core.py').stat().st_mtime_ns
    other=tmp_path/'new-stage';other.mkdir();(other/'core.py').write_bytes(b'frozen')
    bind_original_source(other,target,check)
    assert other.exists()and (target/'core.py').stat().st_mtime_ns==stamp
    (target/'core.py').write_bytes(b'changed')
    with pytest.raises(ValueError):bind_original_source(other,target,check)
    assert (target/'core.py').read_bytes()==b'changed' # never overwrite user's old source


def test_original_native_already_present_is_not_rewritten(tmp_path):
    source=tmp_path/'source';native=source/'nanopokerlib/native.so';native.parent.mkdir(parents=True);native.write_bytes(b'original-native')
    recovery=tmp_path/'state';recovery.mkdir();(recovery/'native.so').write_bytes(b'original-native')
    record={'runtime_binary':{'archive_path':'native.so','restore_relative_path':'nanopokerlib/native.so','sha256':c.sha(native)}}
    stamp=native.stat().st_mtime_ns
    assert c.restore_native(recovery,source,record)==native
    assert native.stat().st_mtime_ns==stamp


def test_plan_runtime_identity_matches_the_actual_uploaded_anchor_when_available():
    import torch
    anchor=Path(__file__).resolve().parents[2]/'runs/t4x2_continue500_delivery_20261022/base_inputs/anchor_iter250.pt'
    if not anchor.is_file():pytest.skip('real user artifact is validated by offline delivery QA instead')
    header=torch.load(anchor,map_location='cpu',weights_only=True)
    from pathlib import PurePosixPath
    identity=header['backend_identity'];native=identity['implementation']
    assert str(PurePosixPath(native['extension_path']).parent.parent)==c.RUNTIME_SOURCE_PATH
    assert native['python_version']==c.EXPECTED_PYTHON
    assert identity['hash']==c.EXPECTED_BACKEND_HASH
