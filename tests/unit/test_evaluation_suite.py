from guandan.evaluation_suite import _cluster_summary, build_plan


def test_cluster_summary_counts_rotated_deals_separately_from_groups():
    rows = [
        {"opponent": "random", "deal_group": 0, "base_deal_sha256": "a", "winner_team": 0, "candidate_team": 0, "token_steps": 10, "committed_steps": 4},
        {"opponent": "random", "deal_group": 0, "base_deal_sha256": "a", "winner_team": 1, "candidate_team": 1, "token_steps": 12, "committed_steps": 5},
        {"opponent": "random", "deal_group": 1, "base_deal_sha256": "b", "winner_team": 0, "candidate_team": 1, "token_steps": 14, "committed_steps": 6},
    ]
    result = _cluster_summary(rows, opponent="random", seed=3)
    assert result["games"] == 3
    assert result["independent_deal_groups"] == 2
    assert result["candidate_wins"] == 2
    assert 0 <= result["deal_cluster_bootstrap95"][0] <= result["deal_cluster_bootstrap95"][1] <= 1


def test_plan_is_explicit_about_two_gpu_workers():
    plan = build_plan(profile="local_fast", progress_bundle="progress.zip", batch_size=8)
    assert plan["workers"] == {"ippo": "cuda:0", "vrpo": "cuda:1"}
    assert plan['matchups'] == {'self_old_snapshot': ['random', 'rule', 'snapshot'],
                                 'cross_algorithm_update102': ['snapshot']}
    assert plan['warmup_games_per_opponent'] == 10
    assert plan['target_games_per_opponent_actual'] == 100
    assert plan['expected_formal_games_total'] == 800



def test_progress_callback_writes_live_progress_and_metrics(tmp_path):
    from guandan.evaluation_suite import _progress_callback
    callback = _progress_callback(root=tmp_path, algorithm='ippo', label='old', device='cuda:0')
    callback({'event': 'evaluation_progress', 'phase': 'warmup', 'opponent': 'random',
              'completed_games': 10, 'target_games': 10, 'games_per_second': 1.5,
              'estimated_remaining_seconds': 30})
    progress = (tmp_path / 'ippo' / 'old' / 'live' / 'progress.json').read_text()
    metrics = (tmp_path / 'ippo' / 'old' / 'live' / 'metrics.jsonl').read_text()
    assert 'completed_games' in progress and 'estimated_remaining_seconds' in progress
    assert 'evaluation_progress' in metrics and 'ippo' in progress



def test_warmup_barrier_uses_slower_worker_not_sum_and_budget():
    from guandan.evaluation_suite import _warmup_decision
    def worker(seconds):
        return {'reports': [{'status': 'warmup_complete', 'metadata': {
            'warmup': {'estimated_remaining_seconds': duration}}} for duration in seconds]}
    messages = [worker([30, 10]), worker([40, 20])]
    result = _warmup_decision(messages, auto_continue=False, max_estimated_minutes=2)
    assert result['status'] == 'warmup_complete'
    assert result['continue_formal'] is False
    assert result['estimated_remaining_seconds'] == 60
    assert result['estimated_finish'].endswith('+08:00')
    assert _warmup_decision(messages, auto_continue=True, max_estimated_minutes=.5)['status'] == 'stopped_by_budget'
    assert _warmup_decision(messages, auto_continue=True, max_estimated_minutes=2)['continue_formal'] is True
    assert _warmup_decision(messages[:1], auto_continue=True, max_estimated_minutes=2)['status'] == 'failed'


def test_cli_defaults_are_warmup_only(monkeypatch):
    import guandan.evaluation_suite as suite
    captured = {}
    def fake_run(**kwargs):
        captured.update(kwargs)
        return {'status': 'warmup_complete'}
    monkeypatch.setattr(suite, 'run', fake_run)
    assert suite.main(['--execute', '--progress-bundle', 'fixture.zip']) == 0
    assert captured['warmup_games_per_opponent'] == 10
    assert captured['target_games_per_opponent'] == 100
    assert captured['auto_continue_after_warmup'] is False


def test_suite_import_does_not_initialize_torch():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        "import sys; import guandan.evaluation_suite; assert 'torch' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr



# Regression tests for the worker/barrier failure protocol use bounded fake
# queues, so regressions cannot leave the test suite waiting indefinitely.
import json
from queue import Empty, Queue
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failed_phase', ['warmup', 'formal'])
def test_worker_report_failure_sends_final_immediately(tmp_path, monkeypatch, failed_phase):
    import guandan.evaluation_suite as suite
    spec = suite.EvalWorkerSpec('ippo', 'candidate', 'old', 'cross', 'cpu',
        'local_fast', 2, str(tmp_path), 2_000_000, 10, 100, 60)
    calls = []
    def evaluate(*args, phase, opponents):
        calls.append((phase, tuple(opponents)))
        return {'status': 'failed' if phase == failed_phase else 'warmup_complete',
                'error': 'injected model failure'}
    monkeypatch.setattr(suite, '_run_one', evaluate)
    result_queue = Queue()
    class Controls:
        def get(self, timeout):
            assert failed_phase == 'formal', 'failed warmup must never wait at barrier'
            return {'continue_formal': True}
    suite._worker(spec, result_queue, Controls())
    messages = []
    while not result_queue.empty():
        messages.append(result_queue.get_nowait())
    assert messages[-1]['kind'] == 'final'
    assert messages[-1]['status'] == 'failed'
    assert json.loads((tmp_path / 'ippo' / 'result.json').read_text())['status'] == 'failed'
    if failed_phase == 'warmup':
        assert len(calls) == len(messages) == 1
        assert not any(message['kind'] == 'warmup_complete' for message in messages)
    else:
        assert len(calls) == 3  # two warmup reports, only the first formal report


def _mock_orchestration(monkeypatch, *, messages, exitcodes=(None, None)):
    import guandan.evaluation_suite as suite
    clock = [0.0]
    class Results:
        def __init__(self):
            self.reads = 0
        def get(self, timeout):
            assert timeout > 0
            self.reads += 1
            # Even if production regresses to an unbounded wait, fail this test.
            assert self.reads <= 20, 'parent waited forever for a missing message'
            clock[0] += .2
            if messages:
                return messages.pop(0)
            raise Empty()
    results = Results()
    controls = [Queue(), Queue()]
    queues = iter([results, *controls])
    processes = []
    class Process:
        def __init__(self, *, target, args, name):
            self.exitcode = exitcodes[len(processes)]
            self.terminated = False
            processes.append(self)
        def start(self):
            pass
        def is_alive(self):
            return self.exitcode is None
        def join(self, timeout):
            assert timeout is not None
        def terminate(self):
            self.terminated = True
            self.exitcode = -15
    context = SimpleNamespace(Queue=lambda: next(queues), Process=Process)
    monkeypatch.setattr(suite.mp, 'get_context', lambda method: context)
    monkeypatch.setattr(suite.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(suite, '_WORKER_EXIT_GRACE_SECONDS', 1.0)
    monkeypatch.setattr(suite, 'sha256', lambda p: '0'*64)
    monkeypatch.setattr(suite, '_extract_bundle', lambda bundle, destination: destination)
    monkeypatch.setattr(suite, '_checkpoint_for', lambda root, algorithm, update: root / f'{algorithm}{update}.pt')
    return results, controls, processes


def _warmup_message(algorithm):
    return {'kind': 'warmup_complete', 'algorithm': algorithm, 'reports': [
        {'status': 'warmup_complete', 'metadata': {
            'warmup': {'estimated_remaining_seconds': seconds}}}
        for seconds in [30, 10]]}


def test_parent_early_final_failure_stops_other_worker_without_waiting_for_warmups(tmp_path, monkeypatch):
    import guandan.evaluation_suite as suite
    results, controls, processes = _mock_orchestration(monkeypatch, messages=[
        {'kind': 'final', 'algorithm': 'ippo', 'status': 'failed', 'error': 'OOM'}])
    result = suite.run(execute=True, profile='local_fast', progress_bundle='fixture', output_root=tmp_path, device_ippo='cpu', device_vrpo='cpu',
                       auto_continue_after_warmup=True)
    assert results.reads == 1
    assert result['status'] == 'failed'
    assert result['warmup_barrier']['continue_formal'] is False
    assert all(queue.get_nowait()['status'] == 'failed' for queue in controls)
    assert all(process.terminated for process in processes)
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('exitcode', [0, 137])
@pytest.mark.parametrize('phase', ['warmup', 'final'])
def test_parent_missing_expected_message_fails_after_exit_grace(tmp_path, monkeypatch, exitcode, phase):
    import guandan.evaluation_suite as suite
    messages = [] if phase == 'warmup' else [_warmup_message('ippo'), _warmup_message('vrpo')]
    results, controls, _ = _mock_orchestration(monkeypatch, messages=messages, exitcodes=(exitcode, None))
    result = suite.run(execute=True, profile='local_fast', progress_bundle='fixture', output_root=tmp_path, device_ippo='cpu', device_vrpo='cpu')
    assert result['status'] == 'failed'
    reason = result['warmup_barrier']['reason']
    assert f'exitcode={exitcode}' in reason
    assert 'grace expired' in reason
    assert results.reads >= 3  # allow Queue feeder drain rather than immediate failure
    for queue in controls:
        decisions = []
        while not queue.empty():
            decisions.append(queue.get_nowait())
        assert decisions[-1]['status'] == 'failed'
        assert decisions[-1]['continue_formal'] is False
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'failed'


def test_parent_accepts_queued_messages_during_clean_exit_grace(tmp_path, monkeypatch):
    import guandan.evaluation_suite as suite
    messages = [_warmup_message('ippo'), _warmup_message('vrpo'),
                {'kind': 'final', 'algorithm': 'ippo', 'status': 'warmup_complete'},
                {'kind': 'final', 'algorithm': 'vrpo', 'status': 'warmup_complete'}]
    results, _, _ = _mock_orchestration(monkeypatch, messages=messages, exitcodes=(0, 0))
    result = suite.run(execute=True, profile='local_fast', progress_bundle='fixture', output_root=tmp_path, device_ippo='cpu', device_vrpo='cpu')
    assert results.reads == 4
    assert result['status'] == 'warmup_complete'


def test_parent_rejects_failed_report_hidden_in_warmup_message(tmp_path, monkeypatch):
    import guandan.evaluation_suite as suite
    message = _warmup_message('ippo')
    message['reports'][0]['status'] = 'failed'
    results, controls, _ = _mock_orchestration(monkeypatch, messages=[message])
    result = suite.run(execute=True, profile='local_fast', progress_bundle='fixture', output_root=tmp_path, device_ippo='cpu', device_vrpo='cpu',
                       auto_continue_after_warmup=True)
    assert results.reads == 1
    assert result['status'] == 'failed'
    assert all(queue.get_nowait()['continue_formal'] is False for queue in controls)


def test_plan_default_counts_independent_of_legacy_profile():
    a=build_plan(profile='local_fast')
    b=build_plan(profile='remote_full')
    assert a['expected_formal_games_total']==b['expected_formal_games_total']==800
    assert a['expected_warmup_games_total']==b['expected_warmup_games_total']==80
    assert build_plan(warmup_games_per_opponent=10,target_games_per_opponent=4)['expected_formal_games_total']==32


def test_bad_input_writes_final_failure_without_launching_workers(tmp_path):
    import guandan.evaluation_suite as suite
    r=suite.run(execute=True,profile='local_fast',device_ippo='cpu',device_vrpo='cpu',
                progress_bundle=str(tmp_path/'absent.zip'),output_root=tmp_path/'out')
    assert r['status']=='failed'
    assert json.loads((tmp_path/'out/result.json').read_text())['status']=='failed'


def test_wall_clock_budget_terminates_unresponsive_workers(tmp_path,monkeypatch):
    import guandan.evaluation_suite as suite
    result_queue,_,processes=_mock_orchestration(monkeypatch,messages=[])
    r=suite.run(execute=True,profile='local_fast',progress_bundle='fixture',output_root=tmp_path,
                device_ippo='cpu',device_vrpo='cpu',max_estimated_minutes=.001)
    assert r['status']=='stopped_by_budget'
    assert 'wall_clock' in r['warmup_barrier']['reason']
    assert all(p.terminated for p in processes)
