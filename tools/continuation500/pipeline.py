"""Fail-fast 250 -> 500 continuation, with evaluation before long training."""
from __future__ import annotations
import base64,gc,html,io,json,os,platform,shutil,sys,threading,time,traceback,uuid,zipfile
from pathlib import Path
from .common import (read_json,sha,canonical,atomic_json,append,extract_safe,safe_path,
 validate_plan,checkpoint_info,restore_native,export_policy,run_contract,retire_raw,verify_recovery_zip,BASE_RECOVERY,ORIGIN_SHA)

PACKAGE=__package__

def execute_schedule(plan, *, get_iteration, evaluate, train_to, summarize_node):
    """The single production schedule; injectable actions make fail-stop tests real."""
    all_arms=('control_vs_rule','candidate_vs_rule','candidate_vs_control')
    new_arms=('candidate_vs_rule','candidate_vs_control')
    evaluate('preflight_before_training','preflight',all_arms,900,origin=True)
    if get_iteration()<plan['target_iteration']:
        train_to(min(get_iteration()+2,plan['target_iteration']),'update_save_gate',reserve=6600)
        if get_iteration()<plan['target_iteration']:
            train_to(get_iteration()+1,'resume_update_gate',reserve=6600)
        evaluate('preflight_new_checkpoint','preflight',all_arms,900,origin=True)
    base_final=evaluate('anchor_final_baseline','final',('control_vs_rule',),3600,baseline=True)
    base_monitor=evaluate('anchor_monitor_baseline','monitor',('control_vs_rule',),1200,baseline=True)
    if get_iteration()<plan['monitor_iteration']:
        train_to(plan['monitor_iteration'],'train_to_monitor',reserve=6600)
    if get_iteration()<plan['target_iteration']:
        name='monitor_'+str(get_iteration())
        monitor=evaluate(name,'monitor',new_arms,1800)
        summarize_node(name,'monitor',base_monitor+monitor,len(base_monitor),monitoring=True)
        train_to(plan['target_iteration'],'train_to_500',reserve=5700)
    name='final_'+str(get_iteration())
    final=evaluate(name,'final',new_arms,5400)
    summarize_node(name,'final',base_final+final,len(base_final),monitoring=False)


def bind_original_source(staged,source,check_source):
    """Preserve the original absolute native identity; never overwrite a checkout."""
    source=Path(source)
    if source.resolve()!=source or source.is_symlink():raise ValueError('canonical original runtime path is redirected')
    if source.exists():
        check_source(source)
    else:
        source.parent.mkdir(parents=True,exist_ok=True)
        Path(staged).rename(source)
    check_source(source)
    return source


def setup_runtime(dataset,run_root,plan,recovery_zip=None,*,identity_audit_only=False):
    raw=(dataset/'v4_payload.bin').read_bytes()
    if sha(dataset/'v4_payload.bin')!=plan['v4_payload_sha256']or not raw.startswith(b'GUANDAN-PILOT-TRANSPORT-V3\n'):raise ValueError('wrong frozen V4 input')
    v4=extract_safe(io.BytesIO(base64.b64decode(raw.split(b'\n',1)[1].strip(),validate=True)),run_root/'v4_dataset',max_bytes=64*1024**2)
    meta=read_json(v4/'pilot_manifest.json');verify=lambda d,k: d[k]==canonical({a:b for a,b in d.items()if a!=k})
    if not verify(meta,'manifest_hash')or meta['manifest_hash']!=plan['v4_manifest_hash']:raise ValueError('wrong V4 manifest')
    for r in meta['files']:
        p=safe_path(v4,r['path'])
        if p.stat().st_size!=r['bytes']or sha(p)!=r['sha256']:raise ValueError('corrupt V4 input '+r['path'])
    if sha(v4/'guandan-marvel-source.zip')!=plan['base_source']['archive_sha256']:raise ValueError('wrong original runtime archive')
    parent=extract_safe(v4/'guandan-marvel-source.zip',run_root/'source_staging',max_bytes=128*1024**2)
    staged=parent/'guandan-marvel-source'
    def check_source(source):
        source_manifest=read_json(source/'source_manifest.json')
        if not verify(source_manifest,'manifest_hash')or source_manifest['manifest_hash']!=plan['base_source']['content_manifest_hash']:raise ValueError('runtime source manifest mismatch')
        for row in source_manifest['files']:
            p=safe_path(source,row['path'])
            if p.stat().st_size!=row['bytes']or sha(p)!=row['sha256']:raise ValueError('frozen training core changed')
    check_source(staged)
    if identity_audit_only:
        # Explicit local inspection only. Production pipeline never exposes this.
        source=staged
    else:
        if platform.system()!='Linux'or platform.python_version()!=plan['expected_python_version']:
            raise RuntimeError('Original full-state runtime requires Linux/Python '+plan['expected_python_version']+'; do not rewrite backend hashes or silently migrate')
        source=bind_original_source(staged,plan['runtime_source_path'],check_source)
    atomic_json(run_root/'runtime_source_binding.json',{'original_absolute_path':plan['runtime_source_path'],
        'actual_source':str(source),'identity_audit_only':identity_audit_only,
        'expected_python_version':plan['expected_python_version'],'expected_backend_hash':plan['expected_backend_hash']})
    loaded=sys.modules.get('guandan_marvel')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):raise RuntimeError('restart the kernel; another Guandan source is already imported')
    sys.path.insert(0,str(source))
    os.environ['PYTHONPATH']=os.pathsep.join([str(Path(__file__).resolve().parents[1]),str(source)])
    os.environ['GUANDAN_MARVEL_SOURCE_COMMIT']=plan['base_source']['git_commit']
    os.environ['PYTHONUNBUFFERED']='1'
    os.environ['NCCL_DEBUG']='INFO'
    os.environ['TORCH_DISTRIBUTED_DEBUG']='INFO'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    original=dataset/'base_recovery.zip';selected=original if recovery_zip is None else Path(recovery_zip)
    if recovery_zip is None and sha(original)!=plan['base_recovery_sha256']:raise ValueError('base recovery hash mismatch')
    recovery=extract_safe(selected,run_root/'state_input')
    record,ddp,cfg=checkpoint_info(recovery,plan,verify_files=True)
    saved=read_json(recovery/'artifacts/pilot_run_contract.json')
    if saved.get('schema')!='guandan-continuation500-contract-v1'and sha(selected)!=plan['base_recovery_sha256']:raise ValueError('unrecognized original recovery')
    restore_native(recovery,source,record)
    return v4,source,recovery,selected,ddp,cfg


def verify_driver_files(plan):
    root=Path(__file__).resolve().parents[1]
    for row in plan['driver_files']:
        p=safe_path(root,row['path'])
        if p.stat().st_size!=row['bytes']or sha(p)!=row['sha256']:raise ValueError('external driver identity mismatch: '+row['path'])


def rows(path):return [json.loads(s)for s in Path(path).read_text(encoding='utf-8').splitlines()if s.strip()]

def phase_health(root,plan):
    from guandan_marvel.kaggle.checkpoints import load_ddp_manifest
    index=read_json(root/'checkpoint_manifest.json');cp=safe_path(root,index['latest']['checkpoint'])
    ddp=load_ddp_manifest(cp,verify_shards=True);iteration=int(ddp['iteration'])
    if ddp['world_size']!=2 or ddp['global_counters']['completed_games_total']!=iteration*plan['collect_global']:raise ValueError('wrong real training counter')
    rank_results=[]
    for rank in (0,1):
        rr=root/'ranks'/f'rank{rank:03d}';runtime=read_json(rr/'pilot_runtime.json');its=rows(rr/'pilot_iterations.jsonl')
        if (runtime['device']!=f'cuda:{rank}'or runtime['communication_backend']!='nccl'or runtime['object_collective_backend']!='gloo'or not runtime['amp']):raise ValueError('not actual two-rank CUDA/NCCL/AMP')
        if runtime['source_hash']!=plan['base_source']['core_source_hash']or runtime['plan_hash']!=plan['plan_hash']:raise ValueError('training runtime/driver binding changed')
        if not its or any(r['effective_optimizer_updates']<1 for r in its)or not any(r['logit_weight_changed']for r in its):raise ValueError('ineffective model updates')
        events=rows(rr/'diagnostic.jsonl');resumed=[e for e in events if e.get('event')=='resumed']
        if len(resumed)!=1 or resumed[0]['decision']not in ('exact_resume','compatible_resume'):raise ValueError('training state was not resumed compatibly')
        metrics=rows(rr/'trainer/ippo_metrics.jsonl')
        health=[(r['step'],k,v)for r in metrics for k,v in r.items()if k.startswith('health/')and k!='health/legal_single_action_tokens_last'and isinstance(v,(int,float))and v!=0]
        if health:raise RuntimeError('training health failure: '+str(health[:5]))
        rank_results.append({'rank':rank,'records':len(its),'effective_updates':sum(r['effective_optimizer_updates']for r in its),
            'amp_skipped_updates':sum(r['amp_skipped_updates']for r in its),'resume':resumed,
            'peak_allocated_bytes':max(r['peak_allocated_bytes']for r in its),'peak_reserved_bytes':max(r['peak_reserved_bytes']for r in its)})
    return ddp,rank_results


def write_reports(root,result):
    def report_rows(path):
        values=[]
        for number,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
            if not line.strip():continue
            try:values.append(json.loads(line))
            except (ValueError,TypeError) as e:
                result.setdefault('diagnostic_jsonl_errors',[]).append({'path':str(Path(path).relative_to(root)),'line':number,'error':str(e)})
        return values
    histories=[]
    for pattern in ('inherited_history/**/metrics.jsonl','phases/*/metrics.jsonl'):
        for p in sorted(root.glob(pattern)):
            for row in report_rows(p):histories.append(row)
    unique={}
    for r in histories:
        i=int(r['statistics/iterations'])
        if i in unique and unique[i]!=r:
            result.setdefault('history_conflicts',[]).append(i)
            result['status']='failed';result['history_status']='conflicting_durable_history'
            continue
        unique[i]=r
    hist=[unique[i]for i in sorted(unique)]
    (root/'training_history.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in hist),encoding='utf-8')
    learning=[]
    for name,summary in result.get('evaluations',{}).items():
        if summary.get('arms'):learning.append({'stage':name,**summary})
    atomic_json(root/'learning_curve.json',learning)
    timing={}
    collect=result.get('collect_global',40)
    for name,stage in result.get('training_stages',{}).items():
        start,end=stage.get('start_iteration'),stage.get('actual_iteration');wall=stage.get('wall_seconds',0)
        path=root/'phases'/name/'metrics.jsonl';metrics=report_rows(path)if path.is_file()else[]
        core=sum(float(r.get('perf/rollout_seconds',0))+float(r.get('perf/learner_seconds',0))for r in metrics)
        games=(end-start)*collect if start is not None and end is not None else 0
        timing[name]={'wall_seconds':wall,'core_seconds':core,'start_iteration':start,'actual_iteration':end,
            'games':games,'wall_games_per_second':games/wall if wall else None,'core_games_per_second':games/core if core else None,
            'ranks':stage.get('ranks',[])}
    telemetry=report_rows(root/'telemetry.jsonl')if(root/'telemetry.jsonl').is_file()else[];gpu_groups={}
    for sample in telemetry:
        for gpu in sample.get('gpus',[]):
            if gpu.get('index') is not None:gpu_groups.setdefault((sample['phase'],gpu['index']),[]).append(gpu)
    gpu_summary=[]
    for (phase_name,index),samples in sorted(gpu_groups.items()):
        def avg(key):
            values=[float(s[key])for s in samples if isinstance(s.get(key),(int,float))]
            return sum(values)/len(values)if values else None
        gpu_summary.append({'phase':phase_name,'gpu_index':index,'samples':len(samples),
            'mean_utilization_percent':avg('utilization_percent'),'mean_power_watts':avg('power_watts'),
            'max_memory_used_mib':max((s.get('memory_used_mib')or 0 for s in samples),default=0)})
    atomic_json(root/'performance_summary.json',{'training_stages':timing,'gpu_by_phase':gpu_summary,
        'evaluation_stages':{k:{'wall_seconds':v.get('wall_seconds'),'games':v.get('games_newly_played')}for k,v in result.get('evaluations',{}).items()},
        'note':'Useful throughput includes actual new games. Wall throughput includes checkpoint/I/O; core does not. GPU busy% is not FLOPS.'})
    atomic_json(root/'continuation_result.json',result)
    title='Guandan iter250 → iter500'
    body='<h1>'+title+'</h1><p>训练与评估分别标记；INCONCLUSIVE不是工程故障。FableDan/最终holdout未运行。</p>'
    body+='<h2>本次状态</h2><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>'
    body+='<h2>胜率/得分节点</h2><pre>'+html.escape(json.dumps(learning,ensure_ascii=False,indent=2))+'</pre>'
    body+='<h2>训练曲线</h2>'
    for label,key in [('NTP loss','macro/ntp_loss'),('Value EV','macro/value_ev'),('Clip fraction','macro/clip_frac'),('Approx KL(old || new)','macro/approx_kl_old_new')]:
        values=[float(r[key])for r in hist if isinstance(r.get(key),(int,float))]
        if values:
            lo,hi=min(values),max(values);span=hi-lo or 1.;n=max(len(values)-1,1)
            points=' '.join(f'{20+700*i/n:.2f},{150-120*(v-lo)/span:.2f}'for i,v in enumerate(values))
            body+=f'<h3>{label}</h3><svg viewBox="0 0 760 180"><polyline fill="none" stroke="#1b6ac9" stroke-width="2" points="{points}"/></svg>'
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:18px}svg{width:100%}</style>'+body,encoding='utf-8')
    (root/'REPORT.md').write_text('# 250→500续训结果\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n完整原始日志、逐局评估、模型快照和最近完整恢复包随ZIP保留。\n',encoding='utf-8')
    target=root.parent/'guandan_continue500_results.zip';temp=target.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True)as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(root)
            if any(part in ('checkpoints','checkpoint')for part in rel.parts):continue
            if p.suffix=='.zip'and p.name!='latest_recovery.zip':continue
            z.write(p,rel.as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.zip'else zipfile.ZIP_DEFLATED)
    temp.replace(target);return target


def newest_durable(root,plan,current):
    """Never replace newer durable state with an older or corrupt generation."""
    candidates=[];errors=[]
    for bundle in Path(root).glob('phases/*/guandan-marvel-recovery.zip'):
        try:
            with zipfile.ZipFile(bundle)as z:iteration=int(json.loads(z.read('recovery_manifest.json'))['iteration'])
            if current is None or iteration>current:candidates.append((iteration,bundle))
        except Exception as e:errors.append({'path':str(bundle),'error':repr(e)})
    for iteration,bundle in sorted(candidates,reverse=True):
        try:
            verify_recovery_zip(bundle,plan)
            return (iteration,bundle),errors
        except Exception as e:errors.append({'path':str(bundle),'error':repr(e)})
    return None,errors


def run_pipeline(*,dataset,run_root,recovery_zip=None,started_at=None):
    dataset,run_root=Path(dataset).resolve(),Path(run_root).resolve();started=time.monotonic()if started_at is None else started_at
    plan=validate_plan(read_json(dataset/'plan.json'));verify_driver_files(plan)
    deadline=started+plan['wall_seconds_cap'];root=run_root/'artifacts';root.mkdir(parents=True,exist_ok=False)
    result={'schema':'guandan-continuation500-result-v1','status':'starting','plan_hash':plan['plan_hash'],
        'collect_global':plan['collect_global'],'target_iteration':plan['target_iteration'],'target_total_games':plan['target_iteration']*plan['collect_global'],
        'training':{'status':'not_started','new_games_this_session':0},'training_stages':{},'evaluations':{},
        'fabledan':'NOT_RUN','final_holdout':'NOT_TOUCHED','automatic_extension_beyond_500':False}
    latest_zip=None;latest_policy=None;current=None;current_state=None;recorder=None;source=None;previous_phase=None
    def phase(name):
        result['current_phase']=name;print('[CONTINUE500 PHASE] '+name,flush=True)
        append(root/'events.jsonl',{'event':'phase_start','phase':name,'elapsed_seconds':time.monotonic()-started})
        if recorder is not None:recorder.set_phase(name)
        atomic_json(root/'continuation_result.json',result)
    try:
        phase('verify_inputs_and_original_runtime')
        v4,source,current_state,latest_zip,ddp,cfg=setup_runtime(dataset,run_root,plan,recovery_zip)
        current=int(ddp['iteration']);session_start=current;session_games=ddp['global_counters']['completed_games_total']
        result['training'].update({'status':'complete' if current==plan['target_iteration'] else 'restored','session_start_iteration':current,'actual_iteration':current,
            'completed_games_total':session_games,'source_core_preserved':True,'optimizer_sampler_rng_preserved':True})
        atomic_json(root/'plan.json',plan)
        import torch
        from guandan_marvel import cpp_game
        from guandan_marvel.kaggle.telemetry import TelemetryRecorder,runtime_environment
        from guandan_marvel.kaggle.pilot import run_process
        from guandan_marvel.kaggle.pilot_evaluation import summarize,tasks_for
        from .evaluate import expected_tasks,collect_records
        if torch.cuda.device_count()!=2 or any('T4'not in torch.cuda.get_device_name(i)for i in (0,1)):raise RuntimeError('requires exactly two T4 GPUs, no fallback')
        if not torch.distributed.is_nccl_available()or not torch.distributed.is_gloo_available():raise RuntimeError('required NCCL/Gloo unavailable')
        if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1':raise RuntimeError('clear CUDA_LAUNCH_BLOCKING explicitly')
        if int(cpp_game.GAME_SPEC.max_game_horizon)!=plan['horizon']:raise ValueError('native horizon mismatch before any training')
        if shutil.disk_usage(run_root).free<7*1024**3:raise RuntimeError('insufficient free working disk for safe checkpoint generations')
        from guandan_marvel.training.checkpoints import build_backend_identity
        backend=build_backend_identity(cpp_game)
        atomic_json(root/'backend_identity.json',backend)
        if backend['hash']!=plan['expected_backend_hash']:raise ValueError('original backend identity mismatch before any training: check canonical path/Python/native')
        shutil.copy2(run_root/'runtime_source_binding.json',root/'runtime_source_binding.json')
        atomic_json(root/'runtime_environment.json',runtime_environment())
        recorder=TelemetryRecorder(root/'telemetry.jsonl',interval_seconds=5);recorder.start()
        anchor=root/'anchor_iter250.pt';shutil.copy2(dataset/'anchor_iter250.pt',anchor)
        if sha(anchor)!=plan['anchor_sha256']:raise ValueError('anchor policy mismatch')
        with zipfile.ZipFile(dataset/'base_recovery.zip')as z:origin_bytes=z.read('artifacts/pilot_initial_policy.pt')
        origin=root/'origin_control.pt';origin.write_bytes(origin_bytes)
        if sha(origin)!=plan['origin_control_sha256']:raise ValueError('origin model mismatch')
        inherited=current_state/'artifacts/pilot_history'
        if inherited.is_dir():shutil.copytree(inherited,root/'inherited_history')
        if current==plan['start_iteration']:latest_policy=anchor
        else:latest_policy=export_policy(current_state,root/f'model_iter{current:06d}.pt',plan)
        def evaluate(name,manifest_file,control,candidate,iteration,arms,seconds):
            phase(name);start=time.monotonic();directory=root/'evaluation'/name;directory.mkdir(parents=True)
            manifest=read_json(manifest_file);tasks=expected_tasks(manifest,arms);errors=[]
            limit=min(seconds,deadline-time.monotonic()-180)
            if limit<60:raise TimeoutError('insufficient evaluation/save budget')
            def work(rank):
                try:
                    cmd=[sys.executable,'-m',PACKAGE+'.evaluate','--plan',str(dataset/'plan.json'),'--manifest',str(manifest_file),
                         '--control',str(control),'--candidate',str(candidate),'--output',str(directory),'--arms',','.join(arms),
                         '--shard',str(rank),'--expected-iteration',str(iteration),'--parent-provenance',str(v4/'parent_run.json')]
                    run_process(cmd,log=root/f'{name}.gpu{rank}.console.log',cwd=source,timeout=limit,events=root/f'{name}.gpu{rank}.events.jsonl')
                except BaseException as e:errors.append(repr(e))
            workers=[threading.Thread(target=work,args=(r,))for r in (0,1)]
            for w in workers:w.start()
            for w in workers:w.join()
            if errors:raise RuntimeError('evaluation stopped before continuing training: '+'; '.join(errors))
            records=collect_records(directory,tasks,control,candidate,manifest['hash'],plan)
            for r in records:
                if r['token_steps']>plan['horizon']:raise ValueError('evaluation exceeded declared horizon')
            atomic_json(directory/'manifest.json',manifest)
            result['evaluations'][name]={'status':'complete','games_newly_played':len(records),'iteration':iteration,'wall_seconds':time.monotonic()-start,
                'arms_requested':list(arms),'manifest_hash':manifest['hash']}
            atomic_json(root/'continuation_result.json',result)
            return records
        def train_to(target,name,*,reserve=5700):
            nonlocal current,current_state,latest_zip,latest_policy,previous_phase
            phase(name);remaining=deadline-time.monotonic()-reserve
            if remaining<180:raise TimeoutError('training stopped before sacrificing final evaluation/save reserve')
            run=root/'phases'/name;run.mkdir(parents=True)
            profile=read_json(v4/'train_profile.json');profile['pilot_observability']['maximum_iteration']=plan['target_iteration']
            profile['resume'].update({'target_iterations':target,'max_wall_seconds':min(10800,remaining),'checkpoint_interval':50,
                'evaluation_interval':0,'checkpoint_keep_last':1,'evaluation_keep_last':1})
            atomic_json(run/'profile.json',profile);atomic_json(run/'pilot_run_contract.json',run_contract(plan))
            atomic_json(run/'pilot_history_segment.json',{'segment':name+'_'+uuid.uuid4().hex})
            shutil.copy2(anchor,run/'pilot_initial_policy.pt')
            if(current_state/'artifacts/pilot_history').is_dir():shutil.copytree(current_state/'artifacts/pilot_history',run/'pilot_prior_history')
            if previous_phase is not None:retire_raw(previous_phase,root)
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--max-restarts=0','--monitor-interval=2',
                '-m',PACKAGE+'.worker','--continuation-plan',str(dataset/'plan.json'),'--profile',str(run/'profile.json'),
                '--run-root',str(run),'--package-id',plan['id'],'--source-commit',plan['base_source']['git_commit'],
                '--source-archive-sha256',plan['base_source']['archive_sha256'],
                '--source-content-manifest-hash',plan['base_source']['content_manifest_hash'],'--recovery-root',str(current_state)]
            before=current;start=time.monotonic()
            bundle=run/'guandan-marvel-recovery.zip'
            run_process(command,log=root/f'{name}.console.log',cwd=source,timeout=min(remaining+420,deadline-time.monotonic()-180),events=root/'training.events.jsonl')
            checked,rank_results=phase_health(run,plan)
            actual=int(checked['iteration']);policy=run/f'pilot_candidate_iter{actual:06d}.pt'
            if not policy.is_file():raise RuntimeError('trained candidate export missing')
            payload=torch.load(policy,map_location='cpu',weights_only=True)
            from guandan_marvel.kaggle.checkpoints import _state_dict_hash
            if payload['iteration']!=actual or _state_dict_hash(payload['agent_state_dict'])!=checked['rank_shards'][0]['agent_merged_state_hash']:raise ValueError('new policy export differs from saved training weights')
            if not all(torch.isfinite(t).all()for t in payload['agent_state_dict'].values()):raise ValueError('nonfinite saved model')
            del payload;gc.collect()
            next_state=extract_safe(bundle,run_root/('state_'+name));checkpoint_info(next_state,plan,verify_files=True)
            destination=root/f'model_iter{actual:06d}.pt';shutil.copy2(policy,destination)
            old_state=current_state;current_state=next_state;current=actual;latest_zip=bundle;latest_policy=destination
            # Delete only our previous expanded input after the next bundle is verified.
            old_abs=old_state.resolve()
            if old_abs.is_relative_to(run_root)and old_abs.name.startswith('state_'):shutil.rmtree(old_abs)
            previous_phase=run
            result['training_stages'][name]={'status':'complete'if actual==target else'budget_stopped','start_iteration':before,
                'target_iteration':target,'actual_iteration':actual,'global_counters':checked['global_counters'],
                'ranks':rank_results,'wall_seconds':time.monotonic()-start}
            result['training'].update({'status':'complete'if current==plan['target_iteration']else'in_progress',
                'actual_iteration':current,'completed_games_total':current*plan['collect_global'],
                'new_games_this_session':(current-session_start)*plan['collect_global'],
                'new_games_since_v4':(current-plan['start_iteration'])*plan['collect_global']})
            atomic_json(root/'continuation_result.json',result)
            if actual!=target:raise TimeoutError(f'worker preserved iteration {actual} before requested {target}')
        def evaluate_stage(name,kind,arms,seconds,*,origin=False,baseline=False):
            control_path=origin_control if origin else anchor
            return evaluate(name,dataset/(kind+'_deals.json'),control_path,
                anchor if baseline else latest_policy,plan['start_iteration'] if baseline else current,arms,seconds)
        def summarize_stage(name,kind,records,cached,*,monitoring):
            summary=summarize(records,tasks_for(read_json(dataset/(kind+'_deals.json'))),control_sha=sha(anchor),candidate_sha=sha(latest_policy))
            summary.update({'iteration':current,'cached_control_games':cached,
                'role':'monitoring_only_not_promotion' if monitoring else 'predeclared_development_endpoint_not_final_holdout'})
            atomic_json(root/'evaluation'/name/'summary.json',summary)
            result['evaluations'][name].update(summary)
        origin_control=origin
        execute_schedule(plan,get_iteration=lambda:current,evaluate=evaluate_stage,train_to=train_to,summarize_node=summarize_stage)
        result['training'].update({'status':'complete','actual_iteration':current,'completed_games_total':current*plan['collect_global'],
            'new_games_this_session':(current-session_start)*plan['collect_global'],'new_games_since_v4':(current-plan['start_iteration'])*plan['collect_global']})
        result['status']='complete';result['evaluation_status']='complete'
    except BaseException as e:
        # Also cover failures in post-worker validation/export, not just subprocess errors.
        recovered,adoption_errors=newest_durable(root,plan,current)
        if adoption_errors:result['bundle_validation_errors']=adoption_errors
        if recovered is not None:
            iteration,bundle=recovered;latest_zip=bundle;current=iteration
            saved=bundle.parent/f'pilot_candidate_iter{iteration:06d}.pt'
            if saved.is_file():
                target=root/f'model_iter{iteration:06d}.pt';shutil.copy2(saved,target);latest_policy=target
            else:
                latest_policy=None;result['policy_export_status']='pending_from_saved_recovery'
            result['training'].update({'status':'checkpoint_saved_pending_review','actual_iteration':iteration,
                'completed_games_total':iteration*plan['collect_global'],
                'new_games_this_session':(iteration-session_start)*plan['collect_global'],
                'new_games_since_v4':(iteration-plan['start_iteration'])*plan['collect_global']})
        result['status']='budget_stopped'if isinstance(e,TimeoutError)else'failed'
        result['error']=f'{type(e).__name__}: {e}';result['failed_phase']=result.get('current_phase')
        if current==plan['target_iteration']:
            result['next_action']='evaluation_only_zero_new_updates'if result['training'].get('status')=='complete'else'verify_target_checkpoint_then_evaluation_only'
        else:result['next_action']='review_failure_then_resume_latest_state_not_from_iter50'
        result['evaluation_status']='failed_or_incomplete'
        (root/'pipeline_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        print('[CONTINUE500 STOP] '+result['error'],flush=True)
    finally:
        if recorder is not None:recorder.close()
        if latest_zip is not None and Path(latest_zip).is_file():
            shutil.copy2(latest_zip,root/'latest_recovery.zip')
            result['recovery']={'iteration':current,'path':'latest_recovery.zip','sha256':sha(root/'latest_recovery.zip')}
        if latest_policy is not None:result['latest_policy']=str(Path(latest_policy).relative_to(root))
        result['wall_seconds']=time.monotonic()-started
        archive=write_reports(root,result);result['results_zip']=str(archive)
    return result
