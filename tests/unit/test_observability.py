import json
import time

from guandan.observability import RunObservability, atomic_json, collect_system_telemetry


def test_atomic_json_round_trip(tmp_path):
    path = tmp_path / "nested" / "value.json"
    atomic_json(path, {"ok": True, "value": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True, "value": 3}
    assert not list(path.parent.glob("*.tmp"))


def test_run_observability_writes_structured_artifacts_without_sync_log_calls(tmp_path):
    obs = RunObservability(tmp_path, {"run_id": "run-1", "algorithm": "ippo"}, telemetry_interval=0.05)
    obs.event("started", stage="smoke")
    obs.metric(tokens_per_second=12.5, rollout_seconds=0.2)
    obs.progress(status="running", update=1)
    with obs.phase("training", update=1):
        time.sleep(0.01)
    obs.system_metric({"timestamp": "test", "gpus": [{"status": "unavailable", "reason": "test"}]})
    obs.close()

    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["run_id"] == "run-1"
    assert (tmp_path / "events.jsonl").is_file()
    assert (tmp_path / "metrics.jsonl").is_file()
    assert (tmp_path / "system_metrics.jsonl").is_file()
    progress = json.loads((tmp_path / "live" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "running"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event.get("event") == "phase_completed" for event in events)


def test_system_telemetry_has_explicit_gpu_unavailable_or_ok():
    telemetry = collect_system_telemetry()
    assert "gpus" in telemetry
    assert telemetry["gpus"]
    assert all(entry.get("status") in {"ok", "unavailable"} for entry in telemetry["gpus"])
