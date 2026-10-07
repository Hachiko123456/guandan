from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from guandan import performance_gate as gate


@pytest.mark.parametrize('value', [0, -1, 601, True, False, float('nan'), float('inf'), -float('inf')])
def test_budget_cannot_be_bypassed(value):
    with pytest.raises(ValueError):
        gate.Config(budget_seconds=value).validate()


@pytest.mark.parametrize('field,value', [('updates_per_shard', 100), ('rollout_steps', 10000),
                                        ('games_per_shard', 1000), ('batch_size', 1024),
                                        ('seed', -1), ('shards_per_algorithm', True)])
def test_shorttest_caps(field, value):
    with pytest.raises(ValueError):
        gate.Config(**{field:value}).validate()


def test_plan_does_not_start_process_or_require_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'spawn_worker', lambda *a: pytest.fail('plan launched worker'))
    result = gate.run(gate.Config(bundle='/not-mounted-yet.zip'), tmp_path / 'plan')
    assert result['status'] == 'plan_only'
    assert result['recommended'] is False
    assert result['global_counts_per_lane'] == {'inference_samples': 1024, 'games':2, 'updates':4, 'training_samples_total':4096, 'training_token_steps_per_algorithm':2048, 'training_token_steps_total':4096}


def test_equal_global_tasks_algorithms_and_shards():
    config = gate.Config(mode='remote', shards_per_algorithm=3)
    for workload in gate.WORKLOADS:
        rows = gate.tasks(config, workload)
        assert len(rows) == 6
        assert [r['algorithm'] for r in rows].count('ippo') == 3
        assert [r['algorithm'] for r in rows].count('vrpo') == 3
        assert len({r['seed'] for r in rows}) == 6
        assert rows == gate.tasks(gate.Config(mode='local', shards_per_algorithm=3), workload)


def measurement(workload='inference', seconds=2):
    return {'valid':True, 'wall_seconds':seconds, 'results':[
        {'valid':True, 'identity':{'algorithm':'ippo', 'shard':0, 'workload':workload,
                                  'model_sha256':'a', 'seed':1, 'workload_config':{}},
         'counts':{'samples':32,'games':1 if workload=='evaluation' else 0,'updates':1 if workload=='train' else 0,'token_steps':32}, 'probe':[0.1,0.2], 'last_loss':0.3, 'game_traces':[{'seed':1,'trace':'abc','winner':0}]}
    ]}


def test_only_complete_matched_finite_work_gets_speedup():
    a,b = measurement(seconds=4), measurement(seconds=2)
    assert gate.compare_lanes(a,b)['speedup'] == 2
    for bad in ({}, {'valid':False}, {**b,'results':[]}, {**b,'wall_seconds':0}, {**b,'wall_seconds':float('nan')}):
        assert gate.compare_lanes(a,bad)['speedup'] is None
    for key,value in [('counts',{}),('identity',{}),('probe',[999])]:
        bad = deepcopy(b)
        bad['results'][0][key] = value
        assert gate.compare_lanes(a,bad)['speedup'] is None


def local_report(seconds):
    return {'mode':'local', 'valid':True, 'source_sha256':'current', 'hardware':{'gpus':['3050ti']},
            'comparisons':{w:{'valid':True,'speedup':2.0} for w in gate.WORKLOADS},
            'measurements':{f'{work}/{lane}':measurement(work, seconds)
                            for work in gate.WORKLOADS for lane in ('cpu','gpu')}}


def test_release_requires_reference_local_and_real_remote():
    local = local_report(1)
    reference = local_report(2)
    assert gate.release_decision(local, local, reference)['status'] == 'pending_remote'
    remote = {'mode':'remote','valid':True,'source_stable':True,'source_sha256':'current',
              'hardware':{'kaggle':True,'gpus':[{'name':'Tesla T4'},{'name':'Tesla T4'}]},
              'comparisons':{w:{'valid':True,'speedup':1.6} for w in gate.WORKLOADS},
              'measurements':{f'{w}/single_t4':measurement(w,1) for w in gate.WORKLOADS}}
    assert gate.release_decision(remote)['recommended'] is False
    assert gate.release_decision(remote,local,reference)['recommended'] is True
    remote['comparisons']['train']['speedup'] = 1.4
    assert gate.release_decision(remote,local,reference)['per_workload']['train']['recommended'] is False
    assert gate.release_decision(remote,local,reference)['per_workload']['evaluation']['recommended'] is True
    remote['hardware']['gpus'][0]['name'] = 'RTX 3050 Ti'
    assert gate.release_decision(remote,local,reference)['recommended'] is False


def test_reference_different_hardware_no_ratio():
    before,after = local_report(2),local_report(1)
    after['hardware'] = {'gpus':['T4']}
    assert not gate.reference_target(before,after)['valid']


def generator():
    spec = importlib.util.spec_from_file_location('prepare_performance', gate.ROOT/'scripts/prepare_kaggle_performance.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_pinned_shorttest_only_plan_default(tmp_path, monkeypatch):
    module = generator()
    notebook = module.render_notebook('a'*64, 'b'*64)
    sources = [c['source'] for c in notebook['cells'] if c['cell_type']=='code']
    for lines in sources:
        ast.parse(''.join(lines))
    code = ''.join(''.join(s) for s in sources)
    assert 'EXECUTE = False' in code
    assert 'a'*64 in code
    assert 'guandan.experiment' not in code
    assert '--budget-seconds' in code
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:pytest.fail('Notebook plan spawned child'))
    config_code = ''.join(sources[0])
    run_code = ''.join(sources[1]).replace("Path('/kaggle/working')", f'Path({str(tmp_path)!r})')
    exec(config_code+'\n'+run_code, {})


@pytest.mark.parametrize('value', [0, -1, 601, float('nan'), float('inf')])
def test_notebook_budget_guard(value, tmp_path):
    module = generator()
    code = module.RUN.replace('__SOURCE_SHA256__','a'*64).replace("Path('/kaggle/working')", f'Path({str(tmp_path)!r})')
    with pytest.raises(ValueError):
        exec(code, {'EXECUTE':False,'BUDGET_SECONDS':value})


def test_timeout_kills_owned_child_and_grandchild(tmp_path, monkeypatch):
    import psutil
    pidfile = tmp_path/'pids.json'
    processes = []
    def spawn(_request, directory):
        directory.mkdir(parents=True)
        code = ('import subprocess,sys,time,json,os; '
                'time.sleep(0.2);c=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); '
                f'open({str(pidfile)!r},"w").write(json.dumps([os.getpid(),c.pid])); '
                'time.sleep(60)')
        options = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
        p = subprocess.Popen([sys.executable,'-c',code], **options)
        if os.name == 'nt':
            p._performance_job = gate.WindowsJob()
            p._performance_job.assign(p)
        processes.append(p)
        return p
    monkeypatch.setattr(gate,'spawn_worker',spawn)
    started = time.monotonic()
    result = gate.run_jobs([{}], tmp_path/'jobs',1,started+1.5)
    assert not result['valid']
    assert time.monotonic()-started < 6
    assert processes[0].poll() is not None
    assert pidfile.exists()
    for pid in json.loads(pidfile.read_text()):
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE


def test_missing_worker_result_cannot_pass(tmp_path, monkeypatch):
    def spawn(_request,directory):
        directory.mkdir(parents=True)
        options = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
        return subprocess.Popen([sys.executable,'-c','pass'], **options)
    monkeypatch.setattr(gate,'spawn_worker',spawn)
    result = gate.run_jobs([{}],tmp_path/'jobs',1,time.monotonic()+5)
    assert not result['valid']
    assert 'missing' in result['results'][0]['error']


def test_real_cpu_inference_and_short_training(tmp_path):
    # Exercise the actual existing trainer rather than a fake zero-work benchmark.
    pytest.importorskip('torch')
    config = gate.Config(inference_batches=2, batch_size=2, rollout_envs=1, rollout_steps=2)
    source = gate.source_digest()
    for workload in ('inference','train'):
        folder = tmp_path/workload
        folder.mkdir()
        request = {'config':gate.asdict(config), 'task':gate.tasks(config,workload)[0], 'device':'cpu',
                   'checkpoints':{}, 'source_sha256':source, 'deadline':time.monotonic()+40}
        result = gate._task(request,folder)
        assert result['valid'],result
        assert result['counts']['samples'] == (4 if workload in ('inference','train') else 0)
        assert result['counts']['updates'] == (2 if workload=='train' else 0)
        assert result['compute_wall_seconds'] > 0
        assert result['end_to_end_wall_seconds'] >= result['steady_wall_seconds']


def test_optional_bundle_sha_validation():
    with pytest.raises(ValueError):
        gate.Config(bundle_sha256='a'*64).validate()
    with pytest.raises(ValueError):
        gate.Config(bundle='x',bundle_sha256='invalid').validate()


def test_zero_completed_work_and_nonfinite_probe_rejected():
    a,b = measurement(),measurement()
    a['results'][0]['counts']['samples'] = 0
    b['results'][0]['counts']['samples'] = 0
    assert gate.compare_lanes(a,b)['speedup'] is None
    a,b = measurement(),measurement()
    a['results'][0]['probe'] = b['results'][0]['probe'] = [float('inf')]
    assert gate.compare_lanes(a,b)['speedup'] is None


def test_generated_package_hash_matches_and_source_verifies(tmp_path):
    import zipfile
    from guandan.deployment.provenance import _read_verified_manifest
    module = generator()
    destination = tmp_path/'package'
    meta = module.build(destination)
    archive = destination/'guandan-performance-source.zip'
    assert gate.digest(archive) == meta['source_sha256']
    notebook = json.loads((destination/'guandan_performance.ipynb').read_text(encoding='utf-8'))
    assert meta['source_sha256'] in ''.join(notebook['cells'][-1]['source'])
    with zipfile.ZipFile(archive) as z:
        assert 'guandan/guandan/performance_gate.py' in z.namelist()
        assert not any('/runs/' in x or x.endswith('.pt') for x in z.namelist())
        z.extractall(tmp_path/'expanded')
    _read_verified_manifest(tmp_path/'expanded/guandan')
    assert meta['status'] == 'pending_remote'


def test_actual_worker_entry_and_owned_process(tmp_path):
    request = {'operation':'task','config':gate.asdict(gate.Config(batch_size=2,inference_batches=1)),
               'task':gate.tasks(gate.Config(),'inference')[0], 'device':'cpu',
               'checkpoints':{},'source_sha256':gate.source_digest()}
    result = gate.run_jobs([request],tmp_path/'actual',1,time.monotonic()+30)
    assert result['valid'],result
    assert result['results'][0]['counts']['samples'] == 2


def test_real_cpu_evaluation_one_terminal_game(tmp_path):
    config = gate.Config(batch_size=2,inference_batches=1)
    request = {'config':gate.asdict(config),'task':gate.tasks(config,'evaluation')[0],
               'device':'cpu','checkpoints':{},'source_sha256':gate.source_digest(),
               'deadline':time.monotonic()+60}
    result = gate._task(request,tmp_path)
    assert result['valid'], result
    assert result['counts']['games']==1
    assert result['counts']['updates']==0
    assert result['counts']['token_steps']>0


def test_steady_wall_is_interval_union_not_sum_of_gpu_latencies():
    assert gate.interval_wall([[1,4],[2,5]]) == 4
    assert gate.interval_wall([[1,4],[5,8]]) == 6
    assert gate.interval_wall([]) is None
    assert gate.interval_wall([[1,1]]) is None


def test_reference_does_not_require_unchanged_inference_2x():
    reference, current = local_report(2), local_report(1)
    for lane in ('cpu','gpu'):
        reference['measurements']['inference/'+lane]['wall_seconds'] = 1
    target = gate.reference_target(reference,current)
    assert target['per_workload']['inference']['passed']
    assert target['per_workload']['train']['passed']
    assert target['per_workload']['evaluation']['passed']


def test_slow_local_gpu_is_not_recommended():
    current = local_report(1)
    current['comparisons']['evaluation']['speedup'] = 0.73
    decision = gate.release_decision(current)
    assert not decision['per_workload']['evaluation']['local_gpu_recommended']
    assert decision['per_workload']['evaluation']['status']=='not_recommended'
    assert decision['status']=='pending_remote'


def test_expanded_source_branch_is_hash_verified():
    code = generator().RUN
    assert 'elif source.is_dir()' in code
    assert 'EXPECTED_MANIFEST_SHA256' in code
    assert "Source file SHA256 mismatch" in code
    assert "actual != set(files)" in code


def test_default_training_global_counts_and_steps_128_supported():
    config = gate.Config().validate()
    assert (config.rollout_envs,config.rollout_steps,config.updates_per_shard)==(8,128,2)
    assert gate.plan(config)['global_counts_per_lane']['training_token_steps_total']==4096
    assert gate.Config(rollout_steps=128).validate().rollout_steps==128


def test_expanded_notebook_rejects_tampered_file_before_launch(tmp_path, monkeypatch):
    import zipfile
    module = generator()
    package = tmp_path/'package'
    module.build(package)
    with zipfile.ZipFile(package/'guandan-performance-source.zip') as archive:
        archive.extractall(tmp_path/'input')
    source = tmp_path/'input/guandan'
    (source/'guandan/performance_gate.py').write_text('tampered')
    notebook = json.loads((package/'guandan_performance.ipynb').read_text(encoding='utf-8'))
    code = ''.join(notebook['cells'][-1]['source'])
    code = code.replace("Path('/kaggle/working')",f'Path({str(tmp_path/"output")!r})')
    code = code.replace("Path('/kaggle/input')",f'Path({str(tmp_path/"input")!r})')
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('tampered source launched'))
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        exec(code, {'EXECUTE':True,'SOURCE_ZIP':str(source),'BUDGET_SECONDS':60,
                    'PROGRESS_BUNDLE':None,'BUNDLE_SHA256':'','LOCAL_REPORT':'','REFERENCE_REPORT':''})


def test_game_trace_mismatch_cannot_pass_reference_or_device_gate():
    a,b = measurement('evaluation'),measurement('evaluation')
    b['results'][0]['game_traces'][0]['winner'] = 1
    assert gate.compare_lanes(a,b)['speedup'] is None


def test_budget_zero_work_shortcut_does_not_import_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'spawn_worker',lambda *a:pytest.fail('expired budget launched work'))
    result = gate.run(gate.Config(budget_seconds=0.001),tmp_path/'expired',execute=True,
                      _started_at=time.monotonic()-1)
    assert not result['valid']
    assert result['status']=='stopped_by_budget'


def test_cli_plan_does_not_read_missing_evidence(tmp_path):
    assert gate.main(['--output',str(tmp_path/'plan'),'--local-report','missing.json'])==0


def test_notebook_expanded_source_success_reaches_only_gate(tmp_path,monkeypatch):
    import zipfile
    module = generator()
    package = tmp_path/'package'
    module.build(package)
    with zipfile.ZipFile(package/'guandan-performance-source.zip') as archive:
        archive.extractall(tmp_path/'input')
    source = tmp_path/'input/guandan'
    notebook = json.loads((package/'guandan_performance.ipynb').read_text(encoding='utf-8'))
    code = ''.join(notebook['cells'][-1]['source'])
    code = code.replace("Path('/kaggle/working')",f'Path({str(tmp_path/"output")!r})')
    code = code.replace("Path('/kaggle/input')",f'Path({str(tmp_path/"input")!r})')
    seen=[]
    class Done:
        returncode=0
        def poll(self): return 0
    def spawn(command,**kwargs):
        seen.append(command)
        return Done()
    monkeypatch.setattr(subprocess,'Popen',spawn)
    exec(code,{'EXECUTE':True,'SOURCE_ZIP':str(source),'BUDGET_SECONDS':60,
               'PROGRESS_BUNDLE':None,'BUNDLE_SHA256':'','LOCAL_REPORT':'','REFERENCE_REPORT':''})
    assert len(seen)==1
    assert 'guandan.performance_gate' in seen[0]
    assert 'guandan.experiment' not in seen[0]
    assert '--execute' in seen[0]
    assert float(seen[0][seen[0].index('--budget-seconds')+1])<60



def _discovery_namespace():
    module = generator()
    namespace = {}
    exec(module.DISCOVERY, namespace)
    return namespace


def _make_expanded_source(root, manifest_bytes=b'valid-manifest'):
    source = root / 'guandan-performance-source' / 'guandan'
    source.mkdir(parents=True)
    manifest = source / 'SOURCE_MANIFEST.json'
    manifest.write_bytes(manifest_bytes)
    return source, hashlib.sha256(manifest_bytes).hexdigest()


def _make_expanded_progress(root, name):
    progress = root / name
    (progress / 'workers' / 'ippo').mkdir(parents=True)
    (progress / 'workers' / 'vrpo').mkdir(parents=True)
    return progress


def test_auto_discovery_matches_screenshot_expanded_inputs(tmp_path):
    namespace = _discovery_namespace()
    input_root = tmp_path / 'input'
    source, manifest_sha = _make_expanded_source(input_root)
    progress = _make_expanded_progress(input_root, 'guandan-training-progress')
    result = namespace['discover_inputs']('', '', input_root, lambda: None, 'not-a-zip-sha', manifest_sha)
    assert Path(result['source']).resolve() == source.resolve()
    assert Path(result['progress']).resolve() == progress.resolve()
    assert result['source_mode'] == 'auto'
    assert result['progress_mode'] == 'auto'


def test_auto_discovery_accepts_renamed_progress_zip(tmp_path):
    import zipfile
    namespace = _discovery_namespace()
    input_root = tmp_path / 'input'
    _, manifest_sha = _make_expanded_source(input_root)
    input_root.mkdir(exist_ok=True)
    archive_path = input_root / 'renamed-training-artifact.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('nested/workers/ippo/checkpoint.pt', b'ippo')
        archive.writestr('nested/workers/vrpo/checkpoint.pt', b'vrpo')
    result = namespace['discover_inputs']('', '', input_root, lambda: None, 'not-a-zip-sha', manifest_sha)
    assert Path(result['progress']).resolve() == archive_path.resolve()


def test_auto_discovery_rejects_multiple_progress_candidates(tmp_path):
    namespace = _discovery_namespace()
    input_root = tmp_path / 'input'
    _, manifest_sha = _make_expanded_source(input_root)
    _make_expanded_progress(input_root, 'progress-one')
    _make_expanded_progress(input_root, 'progress-two')
    with pytest.raises(ValueError, match='multiple progress candidates'):
        namespace['discover_inputs']('', '', input_root, lambda: None, 'not-a-zip-sha', manifest_sha)


def test_auto_discovery_requires_progress_but_none_is_explicit_fresh_seeded(tmp_path):
    namespace = _discovery_namespace()
    input_root = tmp_path / 'input'
    _, manifest_sha = _make_expanded_source(input_root)
    with pytest.raises(ValueError, match='no valid progress'):
        namespace['discover_inputs']('', '', input_root, lambda: None, 'not-a-zip-sha', manifest_sha)
    explicit = namespace['discover_inputs']('', None, input_root, lambda: None, 'not-a-zip-sha', manifest_sha)
    assert explicit['progress'] is None and explicit['progress_mode'] == 'fresh_seeded'


def test_auto_discovery_rejects_input_escape_and_honors_budget(tmp_path):
    namespace = _discovery_namespace()
    input_root = tmp_path / 'input'
    _, manifest_sha = _make_expanded_source(input_root)
    with pytest.raises(ValueError, match='under /kaggle/input'):
        namespace['discover_inputs'](str(tmp_path), None, input_root, lambda: None, 'not-a-zip-sha', manifest_sha)
    calls = []
    def stop_budget():
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError('test budget')
    with pytest.raises(TimeoutError, match='test budget'):
        namespace['discover_inputs']('', None, input_root, stop_budget, 'not-a-zip-sha', manifest_sha)
