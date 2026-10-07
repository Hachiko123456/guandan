"""Bounded, opt-in hardware comparison. No changes to learning algorithms.

The parent imports only the standard library. Torch import, CUDA initialization,
bundle validation and every workload live in owned, timeout-killable subprocesses.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
ALGORITHMS = ("ippo", "vrpo")
WORKLOADS = ("inference", "train", "evaluation")


def positive_seconds(value, maximum=600):
    if isinstance(value, bool):
        raise ValueError("budget must be finite, positive and <=600 seconds")
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError("budget must be finite, positive and <=600 seconds")
    return value


@dataclass(frozen=True)
class Config:
    mode: str = "local"
    budget_seconds: float = 300
    seed: int = 7319
    shards_per_algorithm: int = 1
    batch_size: int = 32
    inference_batches: int = 16
    games_per_shard: int = 1
    updates_per_shard: int = 2
    rollout_envs: int = 8
    rollout_steps: int = 128
    bundle: str | None = None
    bundle_sha256: str | None = None

    def validate(self):
        if self.mode not in ("local", "remote"):
            raise ValueError("mode must be local or remote")
        if type(self.budget_seconds) not in (int,float):
            raise ValueError("budget_seconds must be numeric")
        positive_seconds(self.budget_seconds)
        for key, cap in (("shards_per_algorithm", 4), ("batch_size", 256),
                         ("inference_batches", 128), ("games_per_shard", 4),
                         ("updates_per_shard", 2), ("rollout_envs", 8),
                         ("rollout_steps", 256)):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"{key} must be an integer in [1,{cap}]; shorttest only")
        if type(self.seed) is not int or not 0 <= self.seed < 100_000:
            raise ValueError("seed must be an integer in [0,100000)")
        if self.bundle_sha256 is not None:
            if not self.bundle or len(self.bundle_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.bundle_sha256):
                raise ValueError("SHA256 requires a bundle ZIP and 64 lowercase hex digits")
        return self


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source_digest():
    h = hashlib.sha256()
    for scope, pattern in (("guandan", "*.py"), ("configs", "*.json")):
        for path in sorted((ROOT / scope).rglob(pattern)):
            h.update(path.relative_to(ROOT).as_posix().encode())
            h.update(path.read_bytes())
    h.update((ROOT / "pyproject.toml").read_bytes())
    return h.hexdigest()


def tasks(config, workload):
    return [{"algorithm": alg, "shard": shard, "workload": workload,
             "seed": config.seed + shard * 100 + index * 10_000}
            for shard in range(config.shards_per_algorithm)
            for index, alg in enumerate(ALGORITHMS)]


def plan(config):
    config.validate()
    count = 2 * config.shards_per_algorithm
    return {"schema": SCHEMA, "status": "plan_only", "shorttest_only": True,
            "config": asdict(config), "comparison_lanes": ["cpu", "gpu"] if config.mode == "local" else ["single_t4", "dual_t4"],
            "workloads": list(WORKLOADS), "tasks_per_workload_per_lane": count,
            "global_counts_per_lane": {"inference_samples": count * config.batch_size * config.inference_batches,
                                       "games": count * config.games_per_shard,
                                       "updates": count * config.updates_per_shard,
                                       "training_samples_total": count * config.updates_per_shard * config.rollout_envs * config.rollout_steps,
                                       "training_token_steps_per_algorithm": config.updates_per_shard * config.rollout_envs * config.rollout_steps,
                                       "training_token_steps_total": count * config.updates_per_shard * config.rollout_envs * config.rollout_steps},
            "fairness": "same global tasks, seed, model, shard size; single serial vs dual concurrent, not twice the work; compare lane totals",
            "training_weights": "fresh seeded short training; bundle only used for inference/evaluation, never resumed",
            "release_status": "pending_remote", "recommended": False}


class WindowsJob:
    """Handle-scoped kill-on-close job. Never kills by PID/name or shell command."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        class Basic(c.Structure):
            _fields_ = [("PerProcessUserTimeLimit", c.c_int64), ("PerJobUserTimeLimit", c.c_int64),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                        ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                        "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
        class Extended(c.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                        ("ProcessMemoryLimit", c.c_size_t), ("JobMemoryLimit", c.c_size_t),
                        ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t)]
        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes, kernel.CreateJobObjectW.restype = [c.c_void_p, w.LPCWSTR], w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        kernel.SetInformationJobObject.restype = w.BOOL
        kernel.AssignProcessToJobObject.argtypes, kernel.AssignProcessToJobObject.restype = [w.HANDLE, w.HANDLE], w.BOOL
        kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [w.HANDLE], w.BOOL
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        import ctypes
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def kill_owned_tree(process):
    """Only owned process groups/jobs; Windows handle avoids PID-reuse races."""
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        job = getattr(process, "_performance_job", None)
        if job is not None:
            job.close()  # also kills grandchildren if the leader has already exited
        elif process.poll() is None:
            process.kill()  # only for a child that failed before job attachment
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()


def spawn_worker(request, directory):
    directory.mkdir(parents=True, exist_ok=False)
    request = {**request, "wait_for_owner": os.name == "nt"}
    write_json(directory / "request.json", request)
    log = (directory / "worker.log").open("w", encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    # Do not inherit profile overrides that would change one lane's workload.
    env.pop("GUANDAN_PROFILE", None)
    env.pop("GUANDAN_RESOLVED_PROFILE_JSON", None)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    job = WindowsJob() if os.name == "nt" else None
    process = None
    try:
        process = subprocess.Popen([sys.executable, "-B", "-u", "-m", "guandan.performance_gate",
                                    "--worker-request", str(directory / "request.json")],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, **options)
        if job is not None:
            # Worker is held at a stdlib-only ready-file barrier until protected.
            process._performance_job = job
            job.assign(process)
            (directory / "owner-ready").touch()
    except BaseException:
        if job is not None:
            job.close()
        if process is not None:
            process.kill()
            process.wait(timeout=1)
        raise
    finally:
        log.close()
    return process


def interval_wall(intervals):
    """Union of completed compute intervals on this host's common perf_counter clock.

    Serial workers add; overlapping GPU workers count once. This excludes periods
    when every worker was initializing. It is not a CUDA kernel-time measurement.
    """
    if not intervals or any(len(x) != 2 or not all(math.isfinite(t) for t in x) or x[1] <= x[0] for x in intervals):
        return None
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start,end])
    return sum(end-start for start,end in merged)


def run_jobs(requests, directory, concurrency, deadline):
    """All initialization and CUDA work is charged to the shared monotonic budget."""
    started = time.monotonic()
    pending = list(enumerate(requests))
    active, results = {}, []
    next_log = started
    try:
        while pending or active:
            if time.monotonic() >= deadline:
                break
            while pending and len(active) < concurrency and time.monotonic() < deadline:
                index, req = pending.pop(0)
                folder = directory / f"job-{index:03d}"
                req = {**req, "deadline": deadline}
                active[index] = (spawn_worker(req, folder), folder)
            for index, (process, folder) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                path = folder / "result.json"
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    row = {"valid": False, "error": "missing or malformed worker result"}
                row.update(returncode=code, job=index, log=str(folder / "worker.log"))
                if code != 0:
                    row["valid"] = False
                results.append(row)
                kill_owned_tree(process)
                del active[index]
            if time.monotonic() >= next_log:
                progress = {"event": "progress", "stage": directory.parent.name + "/" + directory.name,
                            "completed_tasks": len(results), "target_tasks": len(requests),
                            "active_tasks": len(active), "elapsed_seconds": time.monotonic() - started,
                            "remaining_budget_seconds": max(0, deadline-time.monotonic()), "eta_seconds": None}
                # Read actual per-task counts, not invented completion from time elapsed.
                progress["active_counts"] = {}
                for index, (_, folder) in active.items():
                    try:
                        progress["active_counts"][str(index)] = json.loads((folder / "progress.json").read_text())
                    except (OSError, ValueError):
                        progress["active_counts"][str(index)] = {"phase": "initializing", "samples": 0, "games": 0, "updates": 0}
                write_json(directory / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                next_log = time.monotonic() + 5
            if active:
                time.sleep(min(0.05, max(0, deadline-time.monotonic())))
    finally:
        for index, (process, folder) in active.items():
            kill_owned_tree(process)
            try:
                partial = json.loads((folder / "progress.json").read_text())
            except (OSError, ValueError):
                partial = {"samples": 0, "games": 0, "updates": 0}
            results.append({"job": index, "valid": False, "error": "timeout_or_interruption",
                            "counts": partial, "log": str(folder / "worker.log")})
    results.sort(key=lambda row: row["job"])
    wall = time.monotonic()-started
    valid = len(results) == len(requests) and not pending and all(x.get("valid") is True for x in results)
    aggregate = {"samples": sum(int(x.get("counts", {}).get("samples", 0)) for x in results),
                 "games": sum(int(x.get("counts", {}).get("games", 0)) for x in results),
                 "updates": sum(int(x.get("counts", {}).get("updates", 0)) for x in results),
                 "token_steps": sum(int(x.get("counts", {}).get("token_steps", 0)) for x in results)}
    intervals = [x.get("steady_interval") for x in results]
    steady = interval_wall(intervals) if valid and all(intervals) else None
    steady_counts = {key:sum(int(x.get("steady_counts", {}).get(key, 0)) for x in results) for key in aggregate}
    per_algorithm = {}
    for alg in ALGORITHMS:
        rows = [x for x in results if x.get("identity", {}).get("algorithm") == alg]
        per_algorithm[alg] = {"completed_tasks": sum(x.get("valid") is True for x in rows),
                              "counts": {key:sum(int(x.get("counts", {}).get(key,0)) for x in rows) for key in aggregate}}
    return {"valid": valid, "wall_seconds": wall, "end_to_end_wall_seconds": wall,
            "per_algorithm": per_algorithm,
            "steady_wall_seconds": steady, "steady_counts": steady_counts, "aggregate_counts": aggregate,
            "target_tasks": len(requests), "completed_tasks": sum(x.get("valid") is True for x in results), "results": results}


def compare_lanes(reference, candidate):
    """No valid matched work = no ratio. Includes import/startup in wall time."""
    answer = {"valid": False, "speedup": None, "reason": "incomplete or unmatched work"}
    if not reference.get("valid") or not candidate.get("valid"):
        return answer
    if reference.get("aggregate_counts") != candidate.get("aggregate_counts"):
        return {**answer, "reason": "lane aggregate counts differ"}
    left, right = reference.get("results", []), candidate.get("results", [])
    if not left or len(left) != len(right):
        return answer
    for a, b in zip(left, right):
        if (a.get("identity") != b.get("identity") or a.get("counts") != b.get("counts")
                or not a.get("valid") or not b.get("valid") or not a.get("identity")):
            return answer
        kind = a["identity"]["workload"]
        key = {"inference": "samples", "train": "updates", "evaluation": "games"}.get(kind)
        if key is None or any(type(x.get("counts", {}).get(key)) is not int or x["counts"][key] <= 0 for x in (a,b)):
            return answer
        if kind == "evaluation" and (not a.get("game_traces") or a.get("game_traces") != b.get("game_traces")):
            return {**answer, "reason": "terminal game trace/winner mismatch"}
        if kind == "train" and (any(type(x.get("last_loss")) not in (int,float) or not math.isfinite(x["last_loss"]) for x in (a,b))
                                or not math.isclose(a["last_loss"], b["last_loss"], rel_tol=1e-3, abs_tol=1e-5)):
            return {**answer, "reason": "training finite/numerical check failed"}
        # Inference probes use equal CPU-generated inputs and tolerate FP32 rounding.
        if a["identity"]["workload"] == "inference":
            p, q = a.get("probe", []), b.get("probe", [])
            if not p or len(p) != len(q) or any(not math.isfinite(x) or not math.isfinite(y) or not math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-5) for x, y in zip(p, q)):
                return {**answer, "reason": "numerical probe mismatch"}
    seconds = [reference.get("wall_seconds"), candidate.get("wall_seconds")]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in seconds):
        return answer
    steady_values = [reference.get("steady_wall_seconds"), candidate.get("steady_wall_seconds")]
    steady_speedup = None
    if (reference.get("steady_counts") == candidate.get("steady_counts") and reference.get("steady_counts")
            and any(reference["steady_counts"].values())
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in steady_values)):
        steady_speedup = steady_values[0] / steady_values[1]
    return {"valid": True, "speedup": seconds[0]/seconds[1], "steady_speedup": steady_speedup,
            "reason": "matched complete lane work; end-to-end wall includes startup"}


def reference_target(reference, current):
    """End-to-end improvement is per workload; unchanged inference need not be 2x."""
    no = {"valid": False, "passed": False, "per_workload": {}}
    if not reference or reference.get("mode") != "local" or current.get("mode") != "local":
        return no
    if reference.get("hardware") != current.get("hardware") or not current.get("hardware"):
        return no
    per = {}
    for work in WORKLOADS:
        ratios = {lane:compare_lanes(reference.get("measurements", {}).get(f"{work}/{lane}", {}),
                                    current.get("measurements", {}).get(f"{work}/{lane}", {})) for lane in ("cpu", "gpu")}
        valid = all(row["valid"] for row in ratios.values())
        factor = min(row["speedup"] for row in ratios.values()) if valid else None
        threshold = 1/1.2 if work == "inference" else 2.0
        per[work] = {"valid": valid, "speedup": factor, "threshold": threshold,
                     "passed": valid and factor >= threshold, "lanes": ratios}
    return {"valid": all(x["valid"] for x in per.values()), "per_workload": per,
            "passed": all(x["passed"] for x in per.values()),
            "basis": "train/evaluation >=2x end-to-end; inference numeric check + <=20% slowdown guard"}


def release_decision(current, local_evidence=None, reference=None):
    """Per-workload recommendation. Local laptop timing NEVER replaces T4 evidence."""
    hardware = current.get("hardware", {})
    cards = hardware.get("gpus", [])
    remote_host = (current.get("mode") == "remote" and hardware.get("kaggle") and len(cards) >= 2
                   and all("T4" in x.get("name", "") for x in cards[:2]))
    local = current if current.get("mode") == "local" else local_evidence
    improvement = reference_target(reference, local or {})
    per = {}
    for work in ("train", "evaluation"):
        local_pair = (local or {}).get("comparisons", {}).get(work, {})
        local_valid = bool(local_pair.get("valid"))
        local_gpu_wins = local_valid and local_pair.get("speedup", 0) > 1.0
        local_inference_ok = bool((local or {}).get("comparisons", {}).get("inference", {}).get("valid"))
        reference_ok = improvement.get("per_workload", {}).get(work, {}).get("passed", False)
        inference_not_regressed = improvement.get("per_workload", {}).get("inference", {}).get("passed", False)
        remote_pair = current.get("comparisons", {}).get(work, {}) if remote_host else {}
        remote_inference_ok = bool(current.get("comparisons", {}).get("inference", {}).get("valid")) if remote_host else False
        source_match = (local or {}).get("source_sha256") == current.get("source_sha256") and bool(current.get("source_sha256"))
        matched_work = remote_host and compare_lanes((local or {}).get("measurements", {}).get(f"{work}/gpu", {}),
                                                     current.get("measurements", {}).get(f"{work}/single_t4", {}))["valid"]
        # Local GPU need not beat CPU to permit future T4 measurement; however we
        # never label this laptop GPU configuration efficient when it actually lost.
        allowed = bool(remote_host and current.get("source_stable") is True and source_match and matched_work and local_valid and local_inference_ok
                       and reference_ok and inference_not_regressed and remote_inference_ok
                       and remote_pair.get("valid") and remote_pair.get("speedup", 0) >= 1.5
                       and not current.get("budget_exceeded"))
        status = "recommended" if allowed else ("pending_remote" if current.get("mode") == "local" and local_gpu_wins else "not_recommended")
        per[work] = {"recommended": allowed, "status": status,
                     "local_gpu_vs_cpu_speedup": local_pair.get("speedup"),
                     "local_gpu_recommended": local_gpu_wins,
                     "local_reference_target_met": reference_ok,
                     "remote_dual_vs_single_speedup": remote_pair.get("speedup"),
                     "reason": "all measured gates passed" if allowed else "missing/failed correctness, local >=2x, inference guard, matched workload or real T4 >=1.5x"}
    recommended = any(x["recommended"] for x in per.values())
    return {"recommended": recommended,
            "status": "partial_recommendation" if recommended and not all(x["recommended"] for x in per.values()) else
                      "shorttest_passed" if recommended else "pending_remote" if current.get("mode") == "local" else "not_recommended",
            "per_workload": per, "local_reference_target": improvement,
            "scope": "only complete measured workload; not model strength or 100% GPU utilization"}


def run(config, output, *, execute=False, reference=None, local_evidence=None, _started_at=None):
    started = time.monotonic() if _started_at is None else min(time.monotonic(), _started_at)  # before validation, plan writing, imports or hardware discovery
    config.validate()
    if type(execute) is not bool:
        raise ValueError("execute must be bool")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {**plan(config), "mode": config.mode, "valid": False, "measurements": {}, "comparisons": {}}
    write_json(output / "plan.json", report)
    if not execute:
        write_json(output / "result.json", report)
        return report
    # Reserve cleanup/report time inside the total budget; never extend it per stage.
    deadline = started + config.budget_seconds - min(8.0, config.budget_seconds * 0.1)
    try:
        preflight = run_jobs([{"operation": "preflight", "config": asdict(config)}], output / "preflight", 1, deadline)
        if not preflight["valid"]:
            report["preflight"] = preflight
            hardware_path = output / "preflight/job-000/hardware.json"
            if hardware_path.is_file():
                report["hardware"] = json.loads(hardware_path.read_text())
            raise RuntimeError("preflight failed or timed out; inspect preflight/job-000/worker.log")
        info = preflight["results"][0]
        report.update(hardware=info["hardware"], source_sha256=info["source_sha256"], bundle_identity=info["bundle_identity"])
        lanes = ("cpu", "gpu") if config.mode == "local" else ("single_t4", "dual_t4")
        for workload in WORKLOADS:
            # Equal slices prevent the first lane consuming the second lane's entire budget.
            for lane in lanes:
                remaining_lanes = 6 - len(report["measurements"])
                slice_deadline = min(deadline, time.monotonic() + max(0, deadline-time.monotonic()) / remaining_lanes)
                jobs = []
                for index, task in enumerate(tasks(config, workload)):
                    device = "cpu" if lane == "cpu" else f"cuda:{index % 2 if lane == 'dual_t4' else 0}"
                    jobs.append({"operation": "task", "config": asdict(config), "task": task,
                                 "device": device, "checkpoints": info["checkpoints"], "source_sha256": info["source_sha256"]})
                report["measurements"][f"{workload}/{lane}"] = run_jobs(jobs, output / workload / lane,
                                                                 2 if lane == "dual_t4" else 1, slice_deadline)
                write_json(output / "result.json", {**report, "status": "running"})
            report["comparisons"][workload] = compare_lanes(*(report["measurements"][f"{workload}/{lane}"] for lane in lanes))
        report["valid"] = all(x["valid"] for x in report["comparisons"].values())
        report["status"] = "complete" if report["valid"] else "incomplete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    report["source_stable"] = bool(report.get("source_sha256")) and source_digest() == report["source_sha256"]
    if report.get("source_sha256") and not report["source_stable"]:
        report.update(valid=False, status="invalid", error="source changed during comparison")
    report["elapsed_seconds"] = time.monotonic()-started
    report["budget_exceeded"] = report["elapsed_seconds"] > config.budget_seconds
    if report["budget_exceeded"]:
        report.update(valid=False, status="stopped_by_budget")
    report["release"] = release_decision(report, local_evidence, reference)
    report["recommended"] = report["release"]["recommended"]
    report["release_status"] = report["release"]["status"]
    if config.mode == "local":
        report["local_reference_target"] = reference_target(reference, report)
    write_json(output / "result.json", report)
    return report


def _preflight(request, folder):
    import torch
    import psutil
    config = Config(**request["config"]).validate()
    hardware = {"system": platform.platform(), "hostname": platform.node(), "python": platform.python_version(),
                "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                "cpu_count": os.cpu_count(), "affinity": psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None,
                "ram_bytes": psutil.virtual_memory().total,
                "kaggle": platform.system() == "Linux" and Path("/kaggle/working").is_dir(), "gpus": []}
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        hardware["gpus"].append({"index": index, "name": props.name, "memory_bytes": props.total_memory,
                                 "capability": list(torch.cuda.get_device_capability(index)), "uuid": str(getattr(props, "uuid", "unavailable"))})
    write_json(folder / "hardware.json", hardware)
    if len(hardware["gpus"]) < (2 if config.mode == "remote" else 1):
        raise RuntimeError("explicit CPU/GPU comparison requires CUDA; no silent fallback")
    if config.mode == "remote" and (not hardware["kaggle"] or any("T4" not in x["name"] for x in hardware["gpus"][:2])):
        raise RuntimeError("pending_remote: remote gate requires actual Kaggle T4 x2")
    checkpoints, bundle_identity = {}, None
    if config.bundle:
        from .evaluation_inputs import prepare_checkpoints
        bundle = Path(config.bundle).resolve()
        if config.bundle_sha256:
            if not bundle.is_file() or digest(bundle) != config.bundle_sha256:
                raise ValueError("bundle SHA256 mismatch (directory cannot have ZIP SHA)")
        checkpoints = {k: str(v) for k, v in prepare_checkpoints(bundle, folder / "bundle").items()}
        bundle_identity = {k: digest(v) for k, v in checkpoints.items()}
    return {"valid": True, "hardware": hardware, "source_sha256": source_digest(),
            "checkpoints": checkpoints, "bundle_identity": bundle_identity}


def _task(request, folder):
    task_start = time.perf_counter()
    import torch
    from .model import PolicyValueNet
    config = Config(**request["config"]).validate()
    task = request["task"]
    device = torch.device(request["device"])
    torch.set_num_threads(1)
    torch.manual_seed(task["seed"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(task["seed"])
    torch.use_deterministic_algorithms(True)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    training_clock = {}
    def check(**diagnostic):
        if time.monotonic() >= request["deadline"]:
            raise TimeoutError("shared benchmark deadline")
        if diagnostic.get("stage") == "initialize_complete":
            sync()
            training_clock["initialized"] = time.perf_counter()
        if diagnostic.get("stage") == "before_update" and "steady_start" not in training_clock:
            sync()
            training_clock["steady_start"] = time.perf_counter()

    class Limit:
        def check(self, **diagnostic):
            check(**diagnostic)

    counts = {"samples": 0, "games": 0, "updates": 0, "token_steps": 0}
    model = PolicyValueNet(hidden_dim=32)
    checkpoint = request["checkpoints"].get(task["algorithm"] + "102")
    if checkpoint and task["workload"] != "train":
        from .training.snapshot import load_policy_snapshot
        snapshot = load_policy_snapshot(checkpoint)
        model = PolicyValueNet(hidden_dim=snapshot.model_config["hidden_dim"])
        model.load_state_dict(snapshot.model_state)
    model_hash = hashlib.sha256(b"".join(k.encode() + v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest()
    identity = {**task, "model_sha256": model_hash,
                "workload_config": {k:v for k,v in asdict(config).items() if k not in ("mode", "budget_seconds", "bundle", "bundle_sha256")}}
    result = {"valid": False, "identity": identity, "counts": counts, "device": str(device),
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "scope": "short benchmark; no full training or ability claim", "source_sha256": request["source_sha256"]}
    phase_start = task_start

    def progress():
        write_json(folder / "progress.json", {**counts, "phase": task["workload"], "elapsed_seconds": time.perf_counter()-phase_start})

    progress()
    try:
        check()
        if source_digest() != request["source_sha256"]:
            raise ValueError("source changed during benchmark; rerun after edits finish")
        steady_start = time.perf_counter()
        result["initialization_wall_seconds"] = steady_start - task_start
        if task["workload"] == "inference":
            # Explicit synthetic microbenchmark, never reported as games/training samples.
            generator = torch.Generator().manual_seed(task["seed"])
            b = config.batch_size
            inputs = (torch.randint(1, 256, (b, 128), generator=generator),
                      torch.randn(b, 256, generator=generator),
                      torch.arange(32).expand(b, 32).clone(), torch.ones(b, 32, dtype=torch.bool))
            model.to(device).eval()
            inputs = tuple(x.to(device) for x in inputs)
            with torch.inference_mode():
                model(*inputs)  # one explicit uncounted kernel warmup, still charged to wall budget
                sync()
                start = time.perf_counter()
                for i in range(config.inference_batches):
                    check()
                    logits, value = model(*inputs)
                    sync()  # count only completed GPU work
                    if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                        raise ValueError("nonfinite inference")
                    if i == 0:
                        result["probe"] = logits[0].cpu().tolist() + value[:min(b, 8)].cpu().tolist()
                    counts["samples"] += b
                    progress()
                sync()
                result["compute_wall_seconds"] = time.perf_counter()-start
                result["steady_interval"] = [start, time.perf_counter()]
                result["steady_wall_seconds"] = result["steady_interval"][1]-start
                result["steady_counts"] = dict(counts)
                result["initialization_wall_seconds"] = start-task_start
        elif task["workload"] == "train":
            from .training.trainer import train
            update_marks = []
            def on_update(event):
                sync()
                update_marks.append((time.perf_counter(), event["lifetime_token_steps"], event["update"]))
                counts.update(samples=event["lifetime_token_steps"], token_steps=event["lifetime_token_steps"], updates=event["update"])
                progress()
            sync()
            start = time.perf_counter()
            trained = train(algorithm=task["algorithm"], profile="local_fast", device=str(device),
                            updates=config.updates_per_shard, rollout_envs=config.rollout_envs,
                            rollout_steps=config.rollout_steps, seed=task["seed"], hidden_dim=32,
                            checkpoint_dir=folder / "short-training", deadline=Limit(), progress_callback=on_update)
            sync()
            result["compute_wall_seconds"] = time.perf_counter()-start
            if update_marks and "steady_start" in training_clock:
                first = training_clock["steady_start"]
                last = update_marks[-1]
                result["steady_interval"] = [first, last[0]]
                result["steady_wall_seconds"] = last[0]-first
                result["steady_counts"] = {"samples": last[1], "token_steps": last[1], "games": 0, "updates": last[2]}
                result["initialization_wall_seconds"] = first-task_start
                result["steady_scope"] = "before first rollout to final update callback; all updates, excludes trainer initialization, includes checkpoint IO"
            else:
                result["steady_wall_seconds"] = None
                result["steady_counts"] = {"samples": 0, "token_steps": 0, "games": 0, "updates": 0}
                result["steady_scope"] = "unavailable: no complete training interval"
            counts.update(samples=trained.total_token_steps, token_steps=trained.total_token_steps, updates=trained.updates)
            if trained.status != "complete" or trained.last_loss is None or not math.isfinite(trained.last_loss):
                raise ValueError("short training incomplete or nonfinite")
            result["last_loss"] = trained.last_loss
            if trained.evidence:
                training_evidence = json.loads(Path(trained.evidence).read_text(encoding="utf-8"))
                counts["games"] = int(training_evidence.get("terminal_count", 0))
                if result.get("steady_counts") is not None:
                    result["steady_counts"]["games"] = counts["games"]
        else:
            from dataclasses import asdict as record_dict
            from .evaluation import play_game
            from .evaluation_agents import EvaluationSession
            if not checkpoint:
                checkpoint = str(folder / "seeded-policy.pt")
                torch.save({"policy_snapshot_version": 1, "model_state": model.state_dict(),
                            "model_config": {"architecture": "mean-pooled-token-MLP", "hidden_dim": 32, "vocabulary": 256, "channels": 256},
                            "metadata": {"profile": "local_fast", "algorithm": task["algorithm"], "update": 0},
                            "training_deals": []}, checkpoint)
            session = EvaluationSession(checkpoint, checkpoint, device=str(device))
            sync()
            start = time.perf_counter()
            def game_progress(event):
                write_json(folder / "progress.json", {**counts, "phase": "evaluation",
                           "inflight_token_steps": event.get("token_steps", 0), "inflight_games": event.get("inflight_games", 1),
                           "elapsed_seconds": time.perf_counter()-phase_start})
            with (folder / "games.jsonl").open("w", encoding="utf-8") as stream:
                for i in range(config.games_per_shard):
                    check()
                    row = play_game(opponent="random", deal_group=i//4, seat_rotation=i%4,
                                    seed=2_000_000 + task["seed"]*10 + i//4,
                                    candidate_checkpoint=checkpoint, snapshot_checkpoint=checkpoint,
                                    device=str(device), deadline=Limit(), agent_factory=session.make_agent,
                                    progress_callback=game_progress)
                    sync()
                    if not row.done or row.truncated or not row.terminal:
                        raise ValueError("nonterminal evaluation")
                    counts["games"] += 1
                    counts["token_steps"] += row.token_steps
                    result.setdefault("game_traces", []).append({"seed":row.seed,"rotation":row.seat_rotation,
                        "trace":row.action_trace_sha256,"winner":row.winner_team,"tokens":row.token_steps})
                    stream.write(json.dumps(record_dict(row)) + "\n")
                    stream.flush()
                    progress()
            result["compute_wall_seconds"] = time.perf_counter()-start
            result["steady_interval"] = [start, time.perf_counter()]
            result["steady_wall_seconds"] = result["steady_interval"][1]-start
            result["steady_counts"] = dict(counts)
            result["initialization_wall_seconds"] = start-task_start
        check()
        if source_digest() != request["source_sha256"]:
            raise ValueError("source changed during benchmark")
        if task["workload"] == "inference":
            expected = counts["samples"] == config.batch_size * config.inference_batches
        elif task["workload"] == "train":
            expected = (counts["updates"] == config.updates_per_shard and counts["samples"] ==
                        config.updates_per_shard * config.rollout_envs * config.rollout_steps)
        else:
            expected = counts["games"] == config.games_per_shard and counts["token_steps"] > 0
        if not expected:
            raise ValueError("actual completed counts differ from target")
        result["valid"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["task_wall_seconds"] = time.perf_counter()-phase_start
    elapsed = result.get("compute_wall_seconds", 0)
    result["throughput_per_second"] = {k: v / elapsed for k,v in counts.items()} if result["valid"] and elapsed > 0 else None
    steady = result.get("steady_wall_seconds")
    result["steady_throughput_per_second"] = {k:v/steady for k,v in result.get("steady_counts", {}).items()} if result["valid"] and steady and steady>0 else None
    result["end_to_end_wall_seconds"] = result["task_wall_seconds"]
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    progress()
    return result


def worker(path):
    path = Path(path)
    started = time.monotonic()
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
        cfg = Config(**request["config"]).validate()
        deadline = request.get("deadline")
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("worker requires finite parent deadline")
        request["deadline"] = min(deadline, started + cfg.budget_seconds)
        while request.get("wait_for_owner") and not (path.parent / "owner-ready").exists():
            if time.monotonic() >= request["deadline"]:
                raise TimeoutError("owner did not attach process job")
            time.sleep(0.01)
        if time.monotonic() >= request["deadline"]:
            raise TimeoutError("budget exhausted before imports")
        result = _preflight(request, path.parent) if request["operation"] == "preflight" else _task(request, path.parent)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    if "request" in locals() and type(request.get("deadline")) in (int,float) and time.monotonic() >= request["deadline"]:
        result.update(valid=False, error="worker deadline exceeded")
    result["worker_wall_seconds"] = time.monotonic()-started
    write_json(path.parent / "result.json", result)
    return 0 if result.get("valid") else 2


def main(argv=None):
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description="Opt-in short performance gate; never long training")
    parser.add_argument("--mode", choices=("local", "remote"), default="local")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget-seconds", type=positive_seconds, default=300)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bundle")
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--reference-report")
    parser.add_argument("--local-report")
    parser.add_argument("--worker-request", help=argparse.SUPPRESS)
    for name in ("seed", "shards_per_algorithm", "batch_size", "inference_batches", "games_per_shard",
                 "updates_per_shard", "rollout_envs", "rollout_steps"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(Config(), name))
    args = vars(parser.parse_args(argv))
    if hasattr(signal, "SIGTERM"):
        def interrupted(_signal, _frame):
            raise KeyboardInterrupt("termination requested; cleaning owned children")
        signal.signal(signal.SIGTERM, interrupted)
    if args["worker_request"]:
        return worker(args["worker_request"])
    args.pop("worker_request")
    def evidence(path):
        if not path or not args["execute"]:
            return None
        source = Path(path)
        if source.stat().st_size > 32*1024**2:
            raise ValueError("evidence report too large")
        value = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("evidence report must be object")
        return value
    reference = evidence(args.pop("reference_report"))
    local = evidence(args.pop("local_report"))
    output = args.pop("output") or str(ROOT / "runs" / ("performance-" + uuid.uuid4().hex[:12]))
    execute = args.pop("execute")
    result = run(Config(**args), output, execute=execute, reference=reference, local_evidence=local,
                 _started_at=invocation_started)
    print(json.dumps({"status": result["status"], "valid": result["valid"], "recommended": result["recommended"],
                      "output": str(Path(output).resolve()), "comparisons": result["comparisons"]}, indent=2))
    return 0 if not execute or result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
