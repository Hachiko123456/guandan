"""Fail-fast eval-only orchestration with resumable, independently checked games."""
from __future__ import annotations
import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from .common import (append, atomic_json, canonical, extract_safe, install_runtime, read_json,
                     safe_path, sha, state_hash, validate_plan, verify_driver, verify_hash, verify_input_files)
from .protocol import (ARMS, CACHED_ARMS, NEW_ARMS, audit_cache, read_records, schedule,
                       tasks_for, validate_records, validate_row, compare_replay)


def execute_schedule(tasks, inherited, replay, evaluate):
    """One production schedule; tests inject actions at process boundaries only."""
    completed = {r["task_id"] for r in inherited}
    expected = {t["task_id"] for t in tasks}
    if not completed <= expected:
        raise ValueError("inherited tasks outside this plan")
    if completed == expected:
        return
    replay()
    smoke = [t for t in tasks if t["deal_index"] < 4 and t["task_id"] not in completed]
    if smoke:
        records = evaluate("p375_smoke", smoke)
        if {r["task_id"] for r in records} != {t["task_id"] for t in smoke}:
            raise ValueError("incomplete 375 smoke, long evaluation forbidden")
        completed.update(r["task_id"] for r in records)
    remaining = [t for t in tasks if t["task_id"] not in completed]
    if remaining:
        records = evaluate("p375_main", remaining)
        if {r["task_id"] for r in records} != {t["task_id"] for t in remaining}:
            raise ValueError("incomplete main evaluation")


def validate_models(dataset, plan, physical_hashes):
    """CPU header/tensor checks; no optimizer or trainer is constructed."""
    import torch
    result = {}
    forbidden = set(physical_hashes)
    for name, spec in plan["models"].items():
        path = Path(dataset) / spec["path"]
        if sha(path) != spec["sha256"]:
            raise ValueError("wrong packaged checkpoint " + name)
        header = torch.load(path, map_location="cpu", weights_only=True)
        if header["schema"] != "guandan-marvel-evaluation-checkpoint-v1" or header.get("training_resume_allowed") is not False:
            raise ValueError("only evaluation-only checkpoints are allowed")
        if header["iteration"] != spec["iteration"] or header["source_hash"] != plan["source"]["core_source_hash"]:
            raise ValueError("wrong model iteration/source")
        if header["backend_identity"]["hash"] != plan["backend_hash"]:
            raise ValueError("model/backend identity changed")
        if header["resolved_config"]["game"]["max_game_horizon"] != 1024:
            raise ValueError("checkpoint horizon mismatch")
        if state_hash(header["agent_state_dict"]) != spec["state_hash"] or not all(torch.isfinite(t).all() for t in header["agent_state_dict"].values()):
            raise ValueError("checkpoint nonfinite or differs from declared weights")
        provenance = header["training_data_provenance"]
        if not provenance.get("complete") or set(provenance["deal_hashes"]) & forbidden:
            raise ValueError("incomplete provenance or train/evaluation overlap")
        result[name] = {"sha256": spec["sha256"], "iteration": spec["iteration"], "state_hash": spec["state_hash"],
                        "training_deal_count": provenance["deal_count"], "all_finite": True, "evaluation_only": True}
    parent = read_json(Path(dataset) / "parent_provenance.json")
    if set(parent["training_deal_hashes"]) & forbidden:
        raise ValueError("evaluation overlaps warm-start parent training")
    return result


def load_resume(bundle, run_root, plan, tasks):
    if bundle is None:
        return [], None
    supplied = Path(bundle).resolve()
    if supplied.is_dir():
        root = supplied
        members = list(root.rglob("*"))
        if any(p.is_symlink() for p in members) or sum(p.stat().st_size for p in members if p.is_file()) > 64 * 1024**2:
            raise ValueError("unsafe/oversize expanded resume directory")
        source_kind = "kaggle_expanded_directory"
    elif supplied.is_file():
        root = extract_safe(supplied, Path(run_root) / "resume_input", max_bytes=64 * 1024**2)
        source_kind = "zip"
    else:
        raise FileNotFoundError("resume path must be this package's result ZIP or its extracted directory")
    manifest = read_json(root / "resume_manifest.json")
    verify_hash(manifest, "manifest_hash")
    if manifest["schema"] != "guandan-recheck375-resume-v1" or manifest["plan_hash"] != plan["plan_hash"]:
        raise ValueError("resume result belongs to another package/plan")
    declared = {row["path"] for row in manifest["files"]}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != declared | {"resume_manifest.json"} or not {"new_games.jsonl", "plan.json", "result.json"} <= declared:
        raise ValueError("undeclared/missing resume members")
    for row in manifest["files"]:
        path = safe_path(root, row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("corrupt evaluation resume member")
    records, ignored = read_records(root / "new_games.jsonl")
    if ignored:
        raise ValueError("canonical resume records may not have a partial tail")
    validate_records(records, tasks, plan, allow_subset=True)
    if len(records) != manifest["verified_new_unique_games"]:
        raise ValueError("resume counter disagrees with actual records")
    return records, {"path": str(supplied), "source_kind": source_kind, "sha256": sha(supplied) if source_kind == "zip" else None,
                     "manifest_hash": manifest["manifest_hash"], "verified_games": len(records)}


def run_workers(jobs, *, dataset, source, output, timeout):
    """Kill both independent process groups promptly if either worker fails."""
    if timeout < 1:
        raise TimeoutError("no evaluation budget remains")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    processes, readers = [], []
    started = time.monotonic()
    streams = []
    try:
        for rank, job in enumerate(jobs):
            if not job["tasks"]:
                continue
            job_path = output / f"job{rank}.json"
            atomic_json(job_path, job)
            command = [sys.executable, "-m", __package__ + ".worker", "--dataset", str(dataset),
                       "--job", str(job_path), "--output", str(output)]
            append(output / "process_events.jsonl", {"event": "start", "gpu_index": rank, "command": command})
            process = subprocess.Popen(command, cwd=source, env=dict(os.environ), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace",
                                       bufsize=1, start_new_session=os.name != "nt")
            stream = (output / f"gpu{rank}.console.log").open("w", encoding="utf8", buffering=1)
            streams.append(stream)
            def copy_output(proc=process, target=stream):
                for line in proc.stdout:
                    target.write(line)
                    print(line, end="", flush=True)
            reader = threading.Thread(target=copy_output, daemon=True)
            reader.start()
            processes.append((rank, process))
            readers.append(reader)
        while True:
            failures = [(rank, proc.returncode) for rank, proc in processes if proc.poll() not in (None, 0)]
            if failures:
                raise RuntimeError("evaluation worker failed; stopping its peer: " + str(failures))
            if all(proc.poll() == 0 for _, proc in processes):
                break
            if time.monotonic() - started > timeout:
                raise TimeoutError("evaluation safety wall budget reached; preserve completed games")
            time.sleep(.25)
    finally:
        for rank, process in processes:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            append(output / "process_events.jsonl", {"event": "end", "gpu_index": rank, "returncode": process.returncode})
        for reader in readers:
            reader.join(timeout=10)
        for stream in streams:
            stream.close()


def gather_new(root, inherited, tasks, plan):
    """Keep verifiable completed games even after a later worker crashes."""
    records = list(inherited)
    diagnostics = []
    for path in sorted(Path(root).glob("phases/p375_*/games.gpu*.jsonl")):
        try:
            rank = int(path.stem[-1])
            status_path = path.parent / f"worker{rank}.json"
            if status_path.is_file() and read_json(status_path).get("frozen_model_hashes_unchanged") is False:
                raise ValueError("worker explicitly detected modified model weights")
            values, ignored = read_records(path, allow_partial_tail=True)
            validate_records(values, tasks, plan, allow_subset=True)
            records.extend(values)
            if ignored:
                diagnostics.append({"file": str(path), "ignored_incomplete_tail_lines": ignored})
        except Exception as error:
            diagnostics.append({"file": str(path), "error": repr(error), "excluded_from_resume": True})
    validate_records(records, tasks, plan, allow_subset=True)
    return records, diagnostics


def run_pipeline(*, dataset, run_root, resume_results=None, started_at=None):
    dataset, run_root = Path(dataset).resolve(), Path(run_root).resolve()
    started = time.monotonic() if started_at is None else started_at
    plan = validate_plan(read_json(dataset / "plan.json"))
    verify_driver(plan)
    deadline = started + plan["wall_seconds_cap"]
    root = run_root / "artifacts"
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "plan.json", plan)
    result = {"schema": "guandan-recheck375-result-v1", "status": "starting", "plan_hash": plan["plan_hash"],
              "new_training_games": 0, "optimizer_updates": 0, "training_authorized": False,
              "fabledan": "NOT_RUN", "final_holdout": "NOT_TOUCHED", "stages": {}, "formal_promotion": False}
    cached, inherited, tasks, recorder = [], [], [], None
    manifest = None

    def phase(name):
        result["current_phase"] = name
        append(root / "events.jsonl", {"event": "phase_start", "phase": name, "elapsed_seconds": time.monotonic() - started})
        print("[RECHECK375 PHASE] " + name + " (training updates=0)", flush=True)
        if recorder is not None:
            recorder.set_phase(name)
        atomic_json(root / "result.json", result)

    try:
        phase("validate_inputs_and_cache_equivalence")
        verify_input_files(dataset)
        manifest = read_json(dataset / "development_deals.json")
        raw, _ = read_records(dataset / "cached_prior_games.jsonl")
        cached, audit = audit_cache(raw, manifest, plan, read_json(dataset / "prior_endpoint_summary.json"))
        atomic_json(root / "cache_equivalence.json", audit)
        atomic_json(root / "development_deals.json", manifest)
        (root / "cached_unique_games.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cached), encoding="utf8")
        physical = {r["deal_id"]: r["base_deal_sha256"] for r in cached}
        result["models"] = validate_models(dataset, plan, physical.values())
        atomic_json(root / "model_validation.json", result["models"])
        tasks = tasks_for(manifest, NEW_ARMS, plan)
        inherited, resume_identity = load_resume(resume_results, run_root, plan, tasks)
        result["resume"] = resume_identity
        result["inherited_verified_new_games"] = len(inherited)
        phase("bind_original_runtime_and_hardware")
        source = install_runtime(dataset, run_root, plan)
        from guandan_marvel import cpp_game
        from guandan_marvel.training.checkpoints import build_backend_identity
        from guandan_marvel.kaggle.telemetry import TelemetryRecorder, runtime_environment
        import torch
        if str(torch.__version__) != plan["torch"] or torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in (0, 1)):
            raise RuntimeError("original PyTorch and exactly two T4 GPUs required; no fallback")
        if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
            raise RuntimeError("clear CUDA_LAUNCH_BLOCKING explicitly before performance evaluation")
        backend = build_backend_identity(cpp_game)
        atomic_json(root / "backend_identity.json", backend)
        if backend["hash"] != plan["backend_hash"] or cpp_game.GAME_SPEC.max_game_horizon != 1024:
            raise ValueError("original native/Python/path/horizon identity differs")
        atomic_json(root / "runtime_environment.json", runtime_environment())
        recorder = TelemetryRecorder(root / "telemetry.jsonl", interval_seconds=5)
        recorder.start()

        def evaluate(name, pending, *, replay_records=None, phase_cap=None):
            phase(name)
            limit = deadline - time.monotonic() - 300
            if phase_cap is not None:
                limit = min(limit, phase_cap)
            if limit < 60:
                raise TimeoutError("not enough evaluation/save budget")
            assigned = schedule(pending)
            jobs = []
            for rank, subset in enumerate(assigned):
                job = {"schema": "guandan-recheck375-job-v1", "phase": name, "gpu_index": rank,
                       "plan_hash": plan["plan_hash"], "tasks": subset, "physical_deal_hashes": physical,
                       "replay_records": {} if replay_records is None else {t["task_id"]: replay_records[t["task_id"]] for t in subset}}
                job["job_hash"] = canonical(job)
                jobs.append(job)
            directory = root / "phases" / name
            start = time.monotonic()
            result["stages"][name] = {"status": "running", "expected_games": len(pending),
                                      "assignment": [{"gpu_index": rank, "games": len(job["tasks"]),
                                                      "arms": dict(collections.Counter(t["arm"] for t in job["tasks"]))} for rank, job in enumerate(jobs)]}
            run_workers(jobs, dataset=dataset, source=source, output=directory, timeout=limit)
            records = []
            for rank, job in enumerate(jobs):
                if not job["tasks"]:
                    continue
                worker = read_json(directory / f"worker{rank}.json")
                if worker["status"] != "complete" or worker["job_hash"] != job["job_hash"] or worker["training_updates"] != 0 or not worker["frozen_model_hashes_unchanged"]:
                    raise ValueError("worker did not complete frozen evaluation")
                values, ignored = read_records(directory / f"games.gpu{rank}.jsonl")
                if ignored:
                    raise ValueError("worker declared complete with a partial output line")
                validate_records(values, job["tasks"], plan)
                if any(r["gpu_index"] != rank for r in values):
                    raise ValueError("worker used an unassigned device")
                for row in values:
                    if row["base_deal_sha256"] != physical[row["deal_id"]]:
                        raise ValueError("new game differs from cached physical deal")
                    if replay_records is not None:
                        compare_replay(row, replay_records[row["task_id"]])
                records.extend(values)
            validate_records(records, pending, plan)
            result["stages"][name].update(status="complete", actual_games=len(records), wall_seconds=time.monotonic() - start)
            atomic_json(root / "result.json", result)
            return records

        def replay():
            expected = {r["task_id"]: r for r in cached}
            return evaluate("cache_replay", tasks_for(manifest, CACHED_ARMS, plan, first_deals=4), replay_records=expected, phase_cap=900)

        execute_schedule(tasks, inherited, replay,
                         lambda name, pending: evaluate(name, pending, phase_cap=900 if name == "p375_smoke" else None))
        new, errors = gather_new(root, inherited, tasks, plan)
        if errors:
            raise ValueError("raw game persistence diagnostics require review")
        validate_records(new, tasks, plan)
        # Cross-bind physical deals across both cached and newly computed arms.
        validate_records(cached + new, tasks_for(manifest, tuple(ARMS), plan), plan)
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "budget_stopped" if isinstance(error, TimeoutError) else "failed"
        result["failed_phase"] = result.get("current_phase")
        result["error"] = f"{type(error).__name__}: {error}"
        (root / "traceback.log").write_text(traceback.format_exc(), encoding="utf8")
        print("[RECHECK375 STOP] " + result["error"], flush=True)
    finally:
        if recorder is not None:
            recorder.close()
        try:
            new, diagnostics = gather_new(root, inherited, tasks, plan) if tasks else ([], [])
        except Exception as error:
            new, diagnostics = list(inherited), [{"error": repr(error), "needs_review": True}]
            result["status"] = "failed"
        result["record_diagnostics"] = diagnostics
        result["wall_seconds"] = time.monotonic() - started
        from .reporting import write_reports
        result["results_zip"] = str(write_reports(root, plan, result, cached, new, inherited_count=len(inherited)))
    return result
