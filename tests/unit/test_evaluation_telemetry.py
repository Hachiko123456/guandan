import json
import subprocess
import time

from guandan.evaluation_telemetry import EvaluationTelemetry, gpu_snapshot, _number


def test_telemetry_records_unavailable_and_closes(tmp_path):
    sampler = lambda: [{'status': 'unavailable', 'reason': 'CPU test'}]
    t = EvaluationTelemetry(tmp_path, interval=0.05, sampler=sampler).start()
    time.sleep(.12)
    summary = t.close()
    assert summary['samples'] >= 1
    assert summary['status'] == 'unavailable'
    records = [json.loads(x) for x in (tmp_path/'system_metrics.jsonl').read_text().splitlines()]
    assert all(x['gpus'][0]['status'] == 'unavailable' for x in records)
    assert 'cpu' in records[-1]
    assert json.loads((tmp_path/'system.json').read_text())['pid'] > 0


def test_telemetry_summary_does_not_count_missing_as_zero(tmp_path):
    sampler = lambda: [dict(index=0, status='ok', utilization_percent=40, memory_used_bytes=123),
                       dict(index=1, status='ok', utilization_percent=None, memory_used_bytes=None)]
    t = EvaluationTelemetry(tmp_path, sampler=sampler)
    t._record()
    s=t.close()
    assert s['gpus']['0']['average_utilization_percent'] == 40
    assert s['gpus']['1']['average_utilization_percent'] is None
    assert _number('[N/A]') is None
    assert _number('nan') is None


def test_nvidia_smi_fallback_csv(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'pynvml', None)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 0,
        '0, GPU-test, Tesla T4, 12, 3, 100, 15360, 35, 50\n', ''))
    r = gpu_snapshot()[0]
    assert r['utilization_percent'] == 12
    assert r['memory_used_bytes'] == 100*1024**2
    assert r['source'] == 'nvidia-smi'
