"""Read-only protocol tests and explicitly mocked process-boundary safety tests."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
from tools.recheck375 import common as c
from tools.recheck375 import protocol as p
from tools.recheck375 import pipeline as pipeline
from tools.recheck375.reporting import write_reports
from tools.recheck375.worker import validate_job

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs/t4x2_recheck375_delivery_20261022/candidate_v1_build/dataset"


def signed(value, key="plan_hash"):
    value = copy.deepcopy(value)
    value.pop(key, None)
    value[key] = c.canonical(value)
    return value


@pytest.fixture(scope="module")
def fixture():
    if not DATA.exists():
        pytest.skip("real cached-data fixture is validated by standalone delivery QA")
    plan = c.read_json(DATA / "plan.json")
    manifest = c.read_json(DATA / "development_deals.json")
    raw, _ = p.read_records(DATA / "cached_prior_games.jsonl")
    old = c.read_json(DATA / "prior_endpoint_summary.json")
    unique, audit = p.audit_cache(raw, manifest, plan, old)
    return SimpleNamespace(plan=plan, manifest=manifest, raw=raw, old=old, cached=unique, audit=audit,
                           tasks=p.tasks_for(manifest, p.NEW_ARMS, plan))


def fake_row(task, fixture):
    """Synthetic 375 outcomes, NEVER confused with real model inference."""
    arm = task["arm"]
    source_arm = "p500_vs_rule" if p.ARMS[arm][1] is None else "p500_vs_p250"
    row = copy.deepcopy(next(r for r in fixture.cached if r["arm"] == source_arm and r["deal_id"] == task["deal"]["deal_id"] and r["seat_rotation"] == task["rotation"]))
    candidate, opponent = p.ARMS[arm]
    row.update(arm=arm, opponent_id=arm, task_id=task["task_id"], task_hash=c.canonical(task),
               candidate_checkpoint_sha256=fixture.plan["models"][candidate]["sha256"],
               opponent_checkpoint_sha256=None if opponent is None else fixture.plan["models"][opponent]["sha256"],
               policies_by_seat=["candidate" if seat % 2 == row["candidate_team"] else arm for seat in range(4)],
               record_origin="new_gpu_execution")
    return row


@pytest.mark.parametrize("changes", [
    {"training_authorized": True}, {"new_training_games_cap": 1}, {"optimizer_updates_cap": 1},
    {"rotations": [0, 1, 2, 3]}, {"horizon": 4096}, {"source": {}}, {"runtime_source_path": "/wrong"},
    {"python": "3.13.1"}, {"torch": "wrong"}, {"backend_hash": "wrong"}, {"deal_count": 256},
    {"development_hash": "wrong"}, {"wall_seconds_cap": 7200}, {"new_unique_strength_games_cap": 1000},
    {"temperature": 1.0}, {"fabledan": "RUN"}, {"final_holdout": "USED"}, {"new_arms": []},
])
def test_signed_unauthorized_plan_is_rejected(fixture, changes):
    plan = copy.deepcopy(fixture.plan)
    plan.update(changes)
    with pytest.raises(ValueError):
        c.validate_plan(signed(plan))


def test_test_only_never_reaches_production(fixture):
    plan = signed(dict(fixture.plan, test_only=True))
    with pytest.raises(ValueError):
        c.validate_plan(plan)
    c.validate_plan(plan, testing=True)


def test_all_real_cached_games_have_equivalent_unique_statistics(fixture):
    assert fixture.audit["prior_actual_records"] == 1536
    assert fixture.audit["identical_rotation_pairs_verified"] == 768
    assert len(fixture.cached) == 768
    summaries = p.summarize_arms(fixture.cached)
    assert summaries["p500_vs_p250"]["win_rate"] == .4453125
    assert summaries["p250_vs_rule"]["win_rate"] == .66015625
    assert summaries["p500_vs_rule"]["win_rate"] == .62890625
    assert all(v["independent_deals"] == 128 and v["records"] == 256 for v in summaries.values())
    assert all(r["record_origin"] == "cached_prior_run" for r in fixture.cached)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "model", "opponent", "trace", "rank", "horizon", "seed", "plan"])
def test_cache_corruption_is_rejected_before_gpu(fixture, mutation):
    rows = copy.deepcopy(fixture.raw)
    if mutation == "missing": rows.pop()
    if mutation == "duplicate": rows.append(rows[0])
    if mutation == "model": rows[0]["candidate_checkpoint_sha256"] = "f" * 64
    if mutation == "opponent": rows[-1]["opponent_checkpoint_sha256"] = "f" * 64
    if mutation == "trace": rows[0]["action_trace_sha256"] = "f" * 64
    if mutation == "rank": rows[0]["ranking"] = [1, 1, 1, 1]
    if mutation == "horizon": rows[0]["evaluation_horizon"] = 4096
    if mutation == "seed": rows[0]["deal_seed"] += 1
    if mutation == "plan": rows[0]["continuation_plan_hash"] = "wrong"
    with pytest.raises((ValueError, KeyError)):
        p.audit_cache(rows, fixture.manifest, fixture.plan, fixture.old)


def test_both_gpus_receive_every_arm_and_both_team_assignments(fixture):
    jobs = p.schedule(fixture.tasks)
    assert sum(map(len, jobs)) == 768
    assert len({t["task_id"] for job in jobs for t in job}) == 768
    for subset in jobs:
        assert len(subset) == 384
        for arm in p.NEW_ARMS:
            rows = [t for t in subset if t["arm"] == arm]
            assert len(rows) == 128
            assert sum(t["rotation"] == 0 for t in rows) == 64
    assert jobs == p.schedule(fixture.tasks)


def test_pending_resume_work_is_rebalanced_not_pinned_to_old_worker(fixture):
    pending = p.schedule(fixture.tasks)[1][13:]
    jobs = p.schedule(pending)
    assert all(jobs)
    cost = [sum(1 if p.ARMS[t["arm"]][1] is None else 2 for t in job) for job in jobs]
    assert abs(cost[0] - cost[1]) <= 2
    assert {t["task_id"] for job in jobs for t in job} == {t["task_id"] for t in pending}


@pytest.mark.parametrize("count", [0, 1, 24, 99, 767, 768])
def test_schedule_never_repeats_verified_games(fixture, count):
    inherited = [{"task_id": t["task_id"]} for t in fixture.tasks[:count]]
    calls = []
    def replay(): calls.append(("replay", []))
    def evaluate(name, tasks):
        calls.append((name, [t["task_id"] for t in tasks]))
        return [{"task_id": t["task_id"]} for t in tasks]
    pipeline.execute_schedule(fixture.tasks, inherited, replay, evaluate)
    new_ids = [key for name, ids in calls if name != "replay" for key in ids]
    assert len(new_ids) == 768 - count and len(set(new_ids)) == len(new_ids)
    assert not set(new_ids) & {r["task_id"] for r in inherited}
    if count == 768: assert not calls
    else: assert calls[0][0] == "replay"


@pytest.mark.parametrize("failure", ["replay", "p375_smoke"])
def test_early_gate_failure_prevents_long_evaluation(fixture, failure):
    calls = []
    def replay():
        calls.append("replay")
        if failure == "replay": raise RuntimeError("bad cached replay")
    def evaluate(name, tasks):
        calls.append(name)
        raise RuntimeError("bad 375 smoke")
    with pytest.raises(RuntimeError):
        pipeline.execute_schedule(fixture.tasks, [], replay, evaluate)
    assert "p375_main" not in calls


def test_replay_bad_trace_is_not_accepted(fixture):
    row = copy.deepcopy(fixture.cached[0])
    row["action_trace_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="replay mismatch"):
        p.compare_replay(row, fixture.cached[0])


def test_partial_tail_is_ignored_but_interior_corruption_is_not(tmp_path):
    path = tmp_path / "games.jsonl"
    path.write_text('{"a":1}\n{"partial":', encoding="utf8")
    assert p.read_records(path, allow_partial_tail=True) == ([{"a": 1}], [2])
    path.write_text('{"partial":\n{"a":1}\n', encoding="utf8")
    with pytest.raises(ValueError): p.read_records(path, allow_partial_tail=True)


@pytest.mark.parametrize("name", ["../outside", "/outside", "C:/outside", "a\\outside"])
def test_zip_cannot_escape_owned_root(tmp_path, name):
    stream = io.BytesIO()
    member = zipfile.ZipInfo("dummy"); member.filename = name; member.orig_filename = name
    with zipfile.ZipFile(stream, "w") as bundle: bundle.writestr(member, b"x")
    stream.seek(0)
    with pytest.raises(ValueError): c.extract_safe(stream, tmp_path / "out")


def test_job_cannot_expand_scope_or_model_matrix(fixture):
    tasks = p.tasks_for(fixture.manifest, p.NEW_ARMS, fixture.plan, first_deals=4)
    job = {"phase": "p375_smoke", "plan_hash": fixture.plan["plan_hash"], "gpu_index": 0, "tasks": tasks, "replay_records": {}}
    validate_job(signed(job, "job_hash"), fixture.plan, fixture.manifest)
    for change in ({"phase": "train"}, {"gpu_index": 2}, {"tasks": fixture.tasks}):
        with pytest.raises(ValueError): validate_job(signed(dict(job, **change), "job_hash"), fixture.plan, fixture.manifest)


def test_result_resume_preserves_valid_games_and_rejects_other_plan(fixture, tmp_path):
    root = tmp_path / "run/artifacts"; root.mkdir(parents=True)
    c.atomic_json(root / "plan.json", fixture.plan)
    new = [fake_row(t, fixture) for t in fixture.tasks[:7]]
    archive = write_reports(root, fixture.plan, {"status": "failed", "stages": {}}, fixture.cached, new)
    out = tmp_path / "resume"; out.mkdir()
    records, identity = pipeline.load_resume(archive, out, fixture.plan, fixture.tasks)
    assert len(records) == 7 and identity["verified_games"] == 7
    other = signed(dict(fixture.plan, id="another-package"))
    out2 = tmp_path / "wrong"; out2.mkdir()
    with pytest.raises(ValueError, match="another package"):
        pipeline.load_resume(archive, out2, other, fixture.tasks)


def test_two_unique_assignments_are_averaged_not_divided_by_four(fixture):
    rows = [fake_row(t, fixture) for t in fixture.tasks]
    for row in rows:
        row["candidate_win"] = True
    summary = p.summarize_arms(rows)
    assert all(v["win_rate"] == 1 for v in summary.values())
    assert all(v["records"] == 256 for v in summary.values())


@pytest.mark.parametrize("failure", [None, "cache_replay", "bad_replay_trace", "p375_smoke", "p375_main", "budget_main"])
def test_real_orchestration_with_mocked_gpu_boundaries(fixture, tmp_path, monkeypatch, failure):
    """Actual schedule/report/resume code, NOT target hardware certification."""
    import torch
    from guandan_marvel import cpp_game
    from guandan_marvel.kaggle import telemetry
    import guandan_marvel.training.checkpoints as cp
    monkeypatch.setattr(pipeline, "verify_driver", lambda plan: None)
    monkeypatch.setattr(pipeline, "verify_input_files", lambda dataset: {})
    monkeypatch.setattr(pipeline, "validate_models", lambda *args: {})
    source = tmp_path / "source"; source.mkdir()
    monkeypatch.setattr(pipeline, "install_runtime", lambda *args: source)
    monkeypatch.setattr(torch, "__version__", fixture.plan["torch"])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda rank: "Tesla T4")
    monkeypatch.setattr(cpp_game, "GAME_SPEC", SimpleNamespace(max_game_horizon=1024))
    monkeypatch.setattr(cp, "build_backend_identity", lambda game: {"hash": fixture.plan["backend_hash"]})
    monkeypatch.setattr(telemetry, "runtime_environment", lambda: {})
    class Recorder:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def close(self): pass
        def set_phase(self, name): pass
    monkeypatch.setattr(telemetry, "TelemetryRecorder", Recorder)
    calls = []
    fail = [failure]
    def fake_workers(jobs, *, output, **kwargs):
        phase = jobs[0]["phase"]
        calls.append((phase, [t["task_id"] for j in jobs for t in j["tasks"]]))
        output.mkdir(parents=True, exist_ok=True)
        for rank, job in enumerate(jobs):
            values = []
            for task in job["tasks"]:
                row = copy.deepcopy(job["replay_records"][task["task_id"]]) if phase == "cache_replay" else fake_row(task, fixture)
                row["gpu_index"] = rank
                if fail[0] == "bad_replay_trace" and phase == "cache_replay": row["action_trace_sha256"] = "f" * 64
                c.append(output / f"games.gpu{rank}.jsonl", row)
                values.append(row)
                if (fail[0] == phase or fail[0] == "budget_main" and phase == "p375_main") and len(values) == 3: break
            c.atomic_json(output / f"worker{rank}.json", {"status": "complete", "job_hash": job["job_hash"], "plan_hash": fixture.plan["plan_hash"],
                "phase": phase, "gpu_index": rank, "games": len(values), "games_attempted": len(values),
                "training_updates": 0, "frozen_model_hashes_unchanged": True})
            if fail[0] == phase or fail[0] == "budget_main" and phase == "p375_main":
                was_budget = fail[0] == "budget_main"
                fail[0] = None
                if was_budget: raise TimeoutError("injected wall budget stop")
                raise RuntimeError("injected worker failure")
    monkeypatch.setattr(pipeline, "run_workers", fake_workers)
    result = pipeline.run_pipeline(dataset=DATA, run_root=tmp_path / "first")
    assert result["new_training_games"] == result["optimizer_updates"] == 0
    assert Path(result["results_zip"]).is_file()
    if failure is None:
        assert result["status"] == "complete"
        assert result["new_verified_unique_games_total"] == 768
        assert result["actual_game_attempts_this_session"] == 792
        assert result["comparison"]["formal_promotion"] is False
        assert [name for name, _ in calls] == ["cache_replay", "p375_smoke", "p375_main"]
        calls.clear()
        complete_resume = pipeline.run_pipeline(dataset=DATA, run_root=tmp_path / "already_done", resume_results=result["results_zip"])
        assert complete_resume["status"] == "complete" and not calls
        assert complete_resume["new_verified_unique_games_this_session"] == complete_resume["actual_game_attempts_this_session"] == 0
    else:
        assert result["status"] == ("budget_stopped" if failure == "budget_main" else "failed")
        if failure in ("cache_replay", "bad_replay_trace", "p375_smoke"):
            assert not any(name == "p375_main" for name, _ in calls)
        if failure in ("p375_main", "budget_main"):
            assert result["new_verified_unique_games_total"] == 27 # 24 smoke + 3 verified partial main
        if failure == "p375_smoke":
            assert result["new_verified_unique_games_total"] == 3
        if failure != "bad_replay_trace":
            verified = result["new_verified_unique_games_total"]
            calls.clear()
            resumed = pipeline.run_pipeline(dataset=DATA, run_root=tmp_path / "resumed", resume_results=result["results_zip"])
            assert resumed["status"] == "complete"
            assert resumed["new_verified_unique_games_this_session"] == 768 - verified


def test_corrupt_input_causes_no_worker_or_runtime_install(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "verify_driver", lambda plan: None)
    def bad_input(dataset): raise ValueError("input hash mismatch")
    monkeypatch.setattr(pipeline, "verify_input_files", bad_input)
    def forbidden(*args, **kwargs): raise AssertionError("must fail before runtime/worker")
    monkeypatch.setattr(pipeline, "install_runtime", forbidden)
    monkeypatch.setattr(pipeline, "run_workers", forbidden)
    result = pipeline.run_pipeline(dataset=DATA, run_root=tmp_path / "bad_input")
    assert result["status"] == "failed" and result["new_verified_unique_games_total"] == 0
    assert result["actual_game_attempts_this_session"] == 0


def test_worker_failure_terminates_peer_before_waiting_full_budget(fixture, tmp_path, monkeypatch):
    import os
    made = []
    class Process:
        def __init__(self, command, **kwargs):
            self.rank = len(made); made.append(self)
            self.returncode = 1 if self.rank == 0 else None
            self.stdout = iter([]); self.pid = 70000 + self.rank; self.terminated = False
            assert "torch.distributed.run" not in command and "--train" not in command
        def poll(self): return self.returncode
        def terminate(self): self.terminated = True; self.returncode = -1
        def kill(self): self.terminate()
        def wait(self, timeout=None): return self.returncode
    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    if os.name != "nt":
        monkeypatch.setattr(pipeline.os, "killpg", lambda pid, sig: made[pid - 70000].terminate())
    tasks = p.tasks_for(fixture.manifest, p.NEW_ARMS, fixture.plan, first_deals=4)
    jobs = []
    for rank, assigned in enumerate(p.schedule(tasks)):
        job = {"phase": "p375_smoke", "gpu_index": rank, "plan_hash": fixture.plan["plan_hash"], "tasks": assigned}
        jobs.append(signed(job, "job_hash"))
    with pytest.raises(RuntimeError, match="stopping its peer"):
        pipeline.run_workers(jobs, dataset=DATA, source=tmp_path, output=tmp_path / "workers", timeout=10000)
    assert made[1].terminated



def test_explicit_weight_mutation_is_never_reused(fixture, tmp_path):
    root = tmp_path / "artifacts"
    phase = root / "phases/p375_main"; phase.mkdir(parents=True)
    c.append(phase / "games.gpu0.jsonl", fake_row(fixture.tasks[0], fixture))
    c.atomic_json(phase / "worker0.json", {"frozen_model_hashes_unchanged": False})
    records, diagnostics = pipeline.gather_new(root, [], fixture.tasks, fixture.plan)
    assert not records and diagnostics[0]["excluded_from_resume"]


def test_resume_member_tampering_is_detected(fixture, tmp_path):
    root = tmp_path / "run/artifacts"; root.mkdir(parents=True)
    c.atomic_json(root / "plan.json", fixture.plan)
    archive = write_reports(root, fixture.plan, {"status": "failed", "stages": {}}, fixture.cached,
                            [fake_row(fixture.tasks[0], fixture)])
    bad = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(bad, "w") as changed:
        for info in original.infolist():
            data = original.read(info.filename)
            if info.filename == "new_games.jsonl": data += b" "
            changed.writestr(info.filename, data)
    out = tmp_path / "resume"; out.mkdir()
    with pytest.raises(ValueError, match="corrupt evaluation resume"):
        pipeline.load_resume(bad, out, fixture.plan, fixture.tasks)


def test_same_seat_assignment_metadata_must_be_exact(fixture):
    row = fake_row(fixture.tasks[0], fixture)
    row["logical_to_physical"] = [3, 2, 1, 0]
    with pytest.raises(ValueError, match="logical_to_physical"):
        p.validate_row(row, fixture.tasks[0], fixture.plan)



def test_resume_supports_kaggle_autoexpanded_result_directory(fixture, tmp_path):
    root = tmp_path / "run/artifacts"; root.mkdir(parents=True)
    c.atomic_json(root / "plan.json", fixture.plan)
    archive = write_reports(root, fixture.plan, {"status": "failed", "stages": {}}, fixture.cached,
                            [fake_row(t, fixture) for t in fixture.tasks[:7]])
    expanded = tmp_path / "kaggle_input/expanded_dataset"
    expanded.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle: bundle.extractall(expanded)
    resume_root = tmp_path / "new_run"; resume_root.mkdir()
    values, identity = pipeline.load_resume(expanded, resume_root, fixture.plan, fixture.tasks)
    assert len(values) == 7 and identity["source_kind"] == "kaggle_expanded_directory"
    assert identity["sha256"] is None and identity["manifest_hash"]
    (expanded / "undeclared.txt").write_text("must reject", encoding="utf8")
    other = tmp_path / "other"; other.mkdir()
    with pytest.raises(ValueError, match="undeclared"):
        pipeline.load_resume(expanded, other, fixture.plan, fixture.tasks)
