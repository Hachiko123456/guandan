# SPDX-License-Identifier: MIT
"""Kaggle T4x2 pilot orchestration: 2 + 1 + 47 = at most 50 absolute iterations."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import zipfile
import uuid
from .common import append_jsonl, atomic_json, canonical_hash, sha256_file
from .pilot_delivery import CAP_ITERATIONS, GLOBAL_WALL_SECONDS, validate_manifest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_extract(archive: Path, destination: Path):
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing to extract over existing state")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if len(set(z.namelist())) != len(z.namelist()):
            raise ValueError("duplicate ZIP paths")
        for info in z.infolist():
            path = (destination / info.filename).resolve()
            if (not path.is_relative_to(destination) or "\\" in info.orig_filename or ":" in info.orig_filename
                    or ((info.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("unsafe ZIP member")
        z.extractall(destination)


def restore_runtime_from_recovery(archive: Path, source: Path, manifest, destination: Path):
    from .bootstrap import validate_recovery_directory
    safe_extract(archive, destination)
    recovery = validate_recovery_directory(destination)
    for name, expected in (("source_archive_sha256", manifest["source"]["archive_sha256"]),
                           ("source_content_manifest_hash", manifest["source"]["content_manifest_hash"]),
                           ("source_commit", manifest["source"]["git_commit"])):
        if recovery.get(name) != expected:
            raise ValueError(f"recovery is bound to different source: {name}")
    contract = read_json(destination / "artifacts/pilot_run_contract.json")
    if contract.get("pilot_manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("recovery belongs to another pilot configuration")
    if not (destination / "artifacts/pilot_initial_policy.pt").is_file():
        raise ValueError("recovery lacks the fixed before-training control policy")
    binary = recovery.get("runtime_binary") or {}
    src = (destination / str(binary.get("archive_path", ""))).resolve()
    target = (source / str(binary.get("restore_relative_path", ""))).resolve()
    if (not src.is_relative_to(destination.resolve()) or not target.is_relative_to(source.resolve())
            or target.parent != source.resolve()/"nanopokerlib" or not src.is_file()
            or sha256_file(src) != binary.get("sha256")):
        raise ValueError("invalid source-bound native runtime in recovery")
    shutil.copy2(src, target)
    return recovery


def run_process(command, *, log: Path, cwd: Path, timeout: float, events: Path):
    if timeout <= 0:
        raise TimeoutError("no phase wall budget remaining")
    append_jsonl(events, {"event": "command_start", "command": command, "timeout_seconds": timeout,
                         "timestamp": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               bufsize=1, start_new_session=os.name != "nt")
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        def copy_output():
            for line in process.stdout:
                stream.write(line)
                print(line, end="", flush=True)
        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"phase timeout; see {log}")
                time.sleep(0.5)
        except BaseException:
            if process.poll() is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=15)
            raise
        finally:
            reader.join(timeout=20)
    append_jsonl(events, {"event": "command_end", "returncode": process.returncode,
                         "seconds": time.monotonic()-started, "log": str(log),
                         "timestamp": datetime.now(timezone.utc).isoformat()})
    if process.returncode:
        raise RuntimeError(f"phase subprocess failed (exit={process.returncode}); see {log}")


def phase_health(root: Path, *, target: int, resumed: bool, expected_world_size: int = 2, expected_games_per_iter: int = 20):
    from .checkpoints import load_ddp_manifest
    index = read_json(root/"checkpoint_manifest.json")
    latest = index.get("latest") or {}
    checkpoint = (root/str(latest.get("checkpoint", ""))).resolve()
    if not checkpoint.is_relative_to(root.resolve()):
        raise ValueError("unsafe checkpoint index")
    manifest = load_ddp_manifest(checkpoint, verify_shards=True)
    counters = manifest["global_counters"]
    actual = int(manifest["iteration"])
    if actual != target or int(counters["completed_games_total"]) != target*expected_games_per_iter:
        raise RuntimeError(f"target not completed: actual={counters}, expected={target} iter / {target*expected_games_per_iter} games")
    if manifest["world_size"] != expected_world_size:
        raise ValueError("checkpoint world size mismatch")
    ranks = []
    for rank in range(expected_world_size):
        rr = root/"ranks"/f"rank{rank:03d}"
        runtime = read_json(rr/"pilot_runtime.json")
        iterations = [json.loads(line) for line in (rr/"pilot_iterations.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if (runtime.get("communication_backend") != "nccl" or runtime.get("local_rank") != rank
                or runtime.get("device") != f"cuda:{rank}" or not runtime.get("amp")):
            raise ValueError(f"rank {rank} did not run explicit CUDA/NCCL/AMP")
        if not iterations or any(row["effective_optimizer_updates"] < 1 or row["peak_allocated_bytes"] <= 0 for row in iterations):
            raise ValueError(f"rank {rank} has missing/ineffective updates or memory evidence")
        if not any(row["logit_weight_changed"] for row in iterations):
            raise ValueError(f"rank {rank} logit-head sentinel never changed")
        events = [json.loads(line) for line in (rr/"diagnostic.jsonl").read_text(encoding="utf-8").splitlines() if line]
        resume_rows = [e for e in events if e.get("event") == "resumed"]
        if resumed and (len(resume_rows) != 1 or resume_rows[0].get("decision") not in ("exact_resume", "compatible_resume")):
            raise ValueError(f"rank {rank} recovery was not verified")
        ranks.append({"runtime": runtime, "iteration_records": len(iterations),
                      "effective_updates": sum(row["effective_optimizer_updates"] for row in iterations),
                      "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in iterations),
                      "resume": resume_rows})
    coverage = {str(row["rank"]): (row.get("training_data_provenance") or {}).get("observed_context_coverage")
                for row in manifest.get("rank_shards", [])}
    return {"status": "PASS", "target_iteration": target, "global_counters": counters,
            "observed_context_coverage_by_rank": coverage,
            "checkpoint_manifest_hash": manifest["manifest_hash"], "ranks": ranks,
            "candidate": str(root/f"pilot_candidate_iter{target:06d}.pt")}


def export_recovered_policy(recovery: Path, destination: Path, *, source_commit: str, expected_world_size: int = 2, target_iteration: int = 50):
    from .bootstrap import validate_recovery_directory
    from .checkpoints import load_rank_checkpoint_for_resume
    from guandan_marvel.evaluation.checkpoints import EVALUATION_CHECKPOINT_SCHEMA, _atomic_torch_save
    record=validate_recovery_directory(recovery)
    manifest_path=(recovery/str(record["checkpoint"])).resolve()
    if not manifest_path.is_relative_to(recovery.resolve()):
        raise ValueError("unsafe recovered checkpoint path")
    payload, manifest=load_rank_checkpoint_for_resume(manifest_path.parent,rank=0,
                       expected_world_size=expected_world_size,expected_source_commit=source_commit)
    if target_iteration not in (50, 250):
        raise ValueError("unsupported versioned evaluation-only iteration cap")
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, dict) or isinstance(resolved.get("collect_batch_size"), bool):
        raise ValueError("recovered policy lacks a valid collect_batch_size contract")
    collect = resolved.get("collect_batch_size")
    if not isinstance(collect, int) or collect <= 0:
        raise ValueError("recovered policy has invalid collect_batch_size")
    if int(manifest["iteration"])!=target_iteration or int(manifest["global_counters"]["completed_games_total"])!=target_iteration*collect:
        raise ValueError(f"evaluation-only restore requires completed {target_iteration} iterations / {target_iteration*collect} games")
    names=("algorithm","architecture","protocol_version","backend","backend_identity","resolved_config",
           "resolved_config_hash","agent_state_dict","iteration","source_fingerprint","source_hash","source_parent_commit")
    policy={name:payload[name] for name in names}
    policy.update({"schema":EVALUATION_CHECKPOINT_SCHEMA,"training_data_provenance":manifest["training_data_provenance"],
                   "merged_policy":{"identity":"current_policy_copy","beta":0.0,"frozen_for_evaluation":True},
                   "resume_classification":"evaluation_only","training_resume_allowed":False})
    _atomic_torch_save(policy,destination)
    return destination


def worker_command(profile: Path, run_root: Path, meta, *, recovery: Path | None, warm: Path):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               "--max-restarts=0", "--monitor-interval=2", "-m", "guandan_marvel.kaggle.worker",
               "--profile", str(profile), "--run-root", str(run_root), "--package-id", meta["pilot_id"],
               "--source-commit", meta["source"]["git_commit"],
               "--source-archive-sha256", meta["source"]["archive_sha256"],
               "--source-content-manifest-hash", meta["source"]["content_manifest_hash"]]
    command += ["--recovery-root", str(recovery)] if recovery else ["--warm-start-checkpoint", str(warm)]
    return command


def remove_owned_raw_checkpoints(root: Path, owner: Path, *, reason: str = "next phase restored and validated; compressed recovery retained"):
    target = (root/"checkpoints").resolve()
    if not target.is_relative_to(owner.resolve()) or target == owner.resolve():
        raise ValueError("refusing out-of-run checkpoint deletion")
    # Every retired raw shard remains in a previously verified compressed recovery bundle.
    if not (root/"guandan-marvel-recovery.zip").is_file():
        raise ValueError("cannot retire raw checkpoints without a recovery bundle")
    if target.exists():
        shutil.rmtree(target)
    atomic_json(root/"raw_checkpoint_retention.json", {"raw_shards_retired": True,
                "reason": reason})


def _line_svg(values, *, color="#2563eb"):
    values = [(i, float(v)) for i,v in enumerate(values) if isinstance(v, (int,float))]
    if not values:
        return "<p>暂无该指标</p>"
    hi, lo = max(v for _,v in values), min(v for _,v in values)
    span = hi-lo or 1.0
    last = max(i for i,_ in values) or 1
    points = " ".join(f"{20+540*i/last:.1f},{130-100*(v-lo)/span:.1f}" for i,v in values)
    return f'<svg viewBox="0 0 580 155"><polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/><text x="20" y="151">min={lo:.5g} max={hi:.5g}</text></svg>'


def write_reports(output: Path, result):
    diagnostic_errors=[]
    def diagnostic_rows(path):
        rows=[]
        for line_no,line in enumerate(path.read_text(encoding="utf-8",errors="replace").splitlines(),1):
            if line:
                try:rows.append(json.loads(line))
                except (ValueError,TypeError) as error:
                    diagnostic_errors.append({"path":str(path),"line":line_no,"error":str(error)})
        return rows
    metrics=[]
    history_paths = sorted(output.glob("recovered_history/**/metrics.jsonl")) + sorted(output.glob("*/metrics.jsonl"))
    for path in history_paths:
        for row in diagnostic_rows(path):
            row["pilot_stage"]=path.parent.name;metrics.append(row)
    unique={}
    for row in metrics:
        iteration=row.get("statistics/iterations")
        if iteration not in unique:
            unique[iteration]=row
        else:
            old={k:v for k,v in unique[iteration].items() if k!="pilot_stage"}
            new={k:v for k,v in row.items() if k!="pilot_stage"}
            if old!=new:
                diagnostic_errors.append({"iteration":iteration,"error":"conflicting recovered iteration metrics"})
    metrics=sorted(unique.values(),key=lambda row: row.get("statistics/iterations",0))
    with (output/"training_history.jsonl").open("w",encoding="utf-8") as f:
        for row in metrics:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    telemetry=[]
    if (output/"telemetry.jsonl").exists():
        telemetry=diagnostic_rows(output/"telemetry.jsonl")
    utilization={}
    for gpu in (0,1):
        samples=[g.get("utilization_percent") for row in telemetry if row.get("phase") in ("precheck","resume_check","training") for g in row.get("gpus",[]) if g.get("index")==gpu and isinstance(g.get("utilization_percent"),(int,float))]
        utilization[str(gpu)]={"samples":len(samples),"mean_percent":sum(samples)/len(samples) if samples else None,
                               "max_percent":max(samples) if samples else None}
    stage_timing={}
    for name, health in result.get("stages", {}).items():
        core=sum(float(row.get("pilot/iteration_seconds",0)) for row in metrics if row.get("pilot_stage")==name)
        wall=float(health.get("wall_seconds",0))
        stage_timing[name]={"phase_wall_seconds":wall,"effective_iteration_seconds":core,
                            "startup_checkpoint_recovery_and_other_overhead_seconds":max(0.0,wall-core)}
    performance={"stage_timing":stage_timing,"diagnostic_jsonl_errors":diagnostic_errors,
                 "gpu_utilization_including_checkpoint_time":utilization,
                 "actual_training_iterations_logged":len(metrics),
                 "note":"No promised utilization/speedup: CPU engine, rollout, small model and checkpoint IO can limit utilization.",
                 "profile_choices":{"DDP":"one NCCL rank per T4","AMP":"FP16 learner, FP32 sensitive losses",
                                    "rollout_global":16,"collect_global":20,"trim_completed_padding":True,
                                    "horizon_capacity":4096,"compile":False,"cpu_threads_per_rank":1}}
    iteration_rows=[]
    for pattern in ("*/ranks/rank*/pilot_iterations.jsonl", "recovered_history/**/pilot_iterations.jsonl"):
        for path in sorted(output.glob(pattern)):
            iteration_rows.extend(diagnostic_rows(path))
    windows=[]
    by_rank={}
    for row in iteration_rows:
        rank=str(row.get("rank")); by_rank.setdefault(rank, []).append(row)
        try:
            end=datetime.fromisoformat(row["timestamp"]).timestamp()
            windows.append((end-float(row["iteration_seconds"]),end))
        except (KeyError, ValueError, TypeError):
            continue
    split={label:{str(g):[] for g in (0,1)} for label in ("inside_iteration_windows","outside_iteration_windows")}
    for row in telemetry:
        if row.get("phase") not in ("precheck","resume_check","training"):
            continue
        try:
            timestamp=datetime.fromisoformat(row["timestamp"]).timestamp()
        except (KeyError,ValueError,TypeError):
            continue
        label="inside_iteration_windows" if any(a<=timestamp<=b for a,b in windows) else "outside_iteration_windows"
        for gpu in row.get("gpus",[]):
            value=gpu.get("utilization_percent")
            if str(gpu.get("index")) in split[label] and isinstance(value,(int,float)):
                split[label][str(gpu["index"])].append(float(value))
    performance["sampled_gpu_busy_split"]={k:{g:{"samples":len(v),"mean_percent":sum(v)/len(v) if v else None}for g,v in rr.items()}for k,rr in split.items()}
    performance["sampled_gpu_busy_split_note"]="Approximate log-time windows, sampled every 5s; GPU busy is not useful FLOPS or Tensor Core throughput."
    performance["per_rank_iteration_summary"]={rank:{
        "iterations_logged":len(rows),
        "mean_iteration_seconds":sum(float(r["iteration_seconds"])for r in rows)/len(rows),
        "mean_rollout_seconds":sum(float(r.get("rollout_seconds") or 0)for r in rows)/len(rows),
        "mean_learner_seconds":sum(float(r.get("learner_seconds") or 0)for r in rows)/len(rows),
        "peak_allocated_bytes":max(int(r.get("peak_allocated_bytes",0))for r in rows),
        "peak_reserved_bytes":max(int(r.get("peak_reserved_bytes",0))for r in rows),
    }for rank,rows in by_rank.items()}
    checkpoint_io=[]
    for pattern in ("*/ranks/rank*/checkpoint_io.jsonl", "recovered_history/**/checkpoint_io.jsonl"):
        for path in sorted(output.glob(pattern)):
            checkpoint_io.extend(diagnostic_rows(path))
    performance["checkpoint_io_records"]=checkpoint_io
    performance["rollout_stage_timing_note"]="Per-rank host intervals include existing CUDA sync/dispatch; not isolated GPU kernel timing."
    recipe=result.get("experiment_recipe")
    if isinstance(recipe,dict):
        performance["profile_choices"].update({"rollout_global":recipe["rollout_global"],
            "metadata_backend":"Gloo side group; gradients remain NCCL", "checkpoint_interval":50,
            "recovery_zip_compression_level":1})
        performance["comparison_note"]="No promised speedup. Compare measured end-to-end throughput and per-rank timings, not busy percentage alone."
    atomic_json(output/"performance_summary.json",performance)
    atomic_json(output/"pilot_result.json",result)
    charts=""
    for title,key in (("宏 PPO loss","macro/pg_loss"),("Value loss","macro/value_loss"),
                      ("Clip fraction","macro/clip_frac"),("Normalized entropy","macro/normalized_entropy"),
                      ("KL(pi || uniform), not old/new KL","macro/macro_kl"),
                      ("Approx KL(old || new)","macro/approx_kl_old_new"),("Value EV","macro/value_ev"),
                      ("Grad norm","agent/grad_norm"),("Learning rate","agent/learning_rate"),
                      ("AMP skipped updates","pilot/amp_skipped_updates"),("Iteration seconds","pilot/iteration_seconds"),
                      ("Raw team reward (not strength)","guandan/raw_team0_reward_mean")):
        charts+=f"<section><h2>{html.escape(title)}</h2>{_line_svg([r.get(key) for r in metrics])}</section>"
    for gpu in (0,1):
        samples=[g.get("utilization_percent") for row in telemetry for g in row.get("gpus",[]) if g.get("index")==gpu]
        charts+=f"<section><h2>GPU {gpu} utilization %</h2>{_line_svg(samples)}</section>"
    doc=("<!doctype html><meta charset='utf-8'><title>掼蛋 T4×2 Pilot</title>"
         "<style>body{font:16px system-ui;max-width:1120px;margin:40px auto;padding:0 20px;background:#f8fafc;color:#172033}"
         "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;border-radius:12px}"
         ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}section{background:white;padding:18px;border-radius:12px}"
         "svg{width:100%}h2{font-size:17px}.warning{background:#fff7d6;padding:16px;border-radius:10px}</style>"
         "<h1>掼蛋 v1.4 · T4×2 短训实验</h1>"
         "<p class='warning'>开发评估，不是 final holdout。FableDan 未运行，不能据此声称超过 FableDan。"
         f"至多 {result.get('authorized_iteration_cap',50)} iteration / {result.get('authorized_new_training_games',1000)} 新训练局；预算不足会明确标为未完成。</p>"
         f"<h2>结果与完整性</h2><pre>{html.escape(json.dumps(result,ensure_ascii=False,indent=2))}</pre>"
         f"<div class='grid'>{charts}</div><h2>双卡性能与限制</h2><pre>{html.escape(json.dumps(performance,ensure_ascii=False,indent=2))}</pre>")
    (output/"report.html").write_text(doc,encoding="utf-8")
    (output/"REPORT.md").write_text("# T4×2 pilot 结果\n\n```json\n"+json.dumps(result,ensure_ascii=False,indent=2)+"\n```\n\n完整逐局评估、rank 日志、指标、性能遥测和可恢复压缩包随结果保留。\n",encoding="utf-8")
    archive_name = result.get("results_archive_name", "guandan_t4x2_pilot_results.zip")
    if not isinstance(archive_name, str) or Path(archive_name).name != archive_name or any(c in archive_name for c in ("/", "\\", ":")):
        raise ValueError("unsafe result archive name")
    archive=output.parent/archive_name
    with zipfile.ZipFile(archive.with_suffix(".tmp"),"w",compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1 if result.get("authorized_iteration_cap")==250 else 6,allowZip64=True) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not any(part in {"checkpoints", "checkpoint"} or part.endswith("_recovery") for part in path.relative_to(output).parts):
                z.write(path,path.relative_to(output).as_posix())
    archive.with_suffix(".tmp").replace(archive)
    return archive


def run_pilot(*, dataset: Path, output: Path, source: Path, recovery: Path | None = None, started_at: float | None = None):
    import torch
    from .bootstrap import validate_recovery_directory
    from .telemetry import TelemetryRecorder, runtime_environment
    from .pilot_evaluation import summarize, tasks_for
    started = time.monotonic() if started_at is None else started_at
    deadline=started+GLOBAL_WALL_SECONDS
    output=Path(output).resolve();source=Path(source).resolve();dataset=Path(dataset).resolve()
    output.mkdir(parents=True,exist_ok=False)
    result={"schema":"guandan-marvel-t4x2-pilot-result-v1","status":"starting","stages":{},
            "fabledan":"NOT_RUN","final_holdout":"NOT_TOUCHED","remote_verified":False}
    recorder=None
    previous_phase=None
    recovery_validated=False
    try:
        meta=validate_manifest(dataset)
        cap = int((meta.get("training_cap") or {}).get("absolute_iterations", 50))
        if cap not in (50, 250):
            raise ValueError("unsupported versioned training cap")
        collect_games = int((meta.get("training_cap") or {}).get("global_collect_batch_size", 20))
        if collect_games <= 0:
            raise ValueError("invalid frozen global_collect_batch_size")
        result.update({"authorized_iteration_cap": cap, "authorized_new_training_games": cap*collect_games,
                       "results_archive_name": meta.get("results_archive_name", "guandan_t4x2_pilot_results.zip"),
                       "experiment_recipe": meta.get("recipe", "frozen original pilot"),
                       "warm_start_parent": meta.get("parent_run")})
        if torch.cuda.device_count()!=2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError("This frozen delivery requires exactly two NVIDIA T4 GPUs. Select T4 x2 in Kaggle.")
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("NCCL unavailable; no Gloo/CPU fallback")
        if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
            raise RuntimeError("CUDA_LAUNCH_BLOCKING=1 would invalidate the performance pilot; clear it explicitly")
        atomic_json(output/"input_manifest.json",meta)
        atomic_json(output/"runtime_environment.json",runtime_environment())
        if (output.parent/"native_build.log").exists():shutil.copy2(output.parent/"native_build.log",output/"native_build.log")
        recorder=TelemetryRecorder(output/"telemetry.jsonl",interval_seconds=5);recorder.start()
        contract={"pilot_manifest_hash":meta["manifest_hash"],"cap_iterations":cap,"global_collect_batch_size":meta.get("training_cap",{}).get("global_collect_batch_size",20),
                  "evaluation_manifest_hash":meta["evaluation_manifest_hash"],"source":meta["source"]}
        atomic_json(output/"pilot_run_contract.json",contract)
        recovered_candidate=None
        if recovery is not None:
            restored=validate_recovery_directory(recovery)
            saved_contract=read_json(recovery/"artifacts/pilot_run_contract.json")
            if saved_contract!=contract:raise ValueError("recovery/config contract mismatch")
            recovery_validated=True
            initial=recovery/"artifacts/pilot_initial_policy.pt"
            if not initial.is_file():raise ValueError("recovery missing untrained control")
            shutil.copy2(initial,output/"pilot_initial_policy.pt")
            if (recovery/"artifacts/pilot_history").is_dir():
                shutil.copytree(recovery/"artifacts/pilot_history",output/"recovered_history")
            recovered_iteration=int(restored["iteration"])
            if recovered_iteration>cap:raise ValueError(f"recovery exceeds authorized iteration {cap}")
            if recovered_iteration==cap:
                export_options = {"target_iteration": cap} if cap != 50 else {}
                recovered_candidate=export_recovered_policy(recovery,output/f"recovered_candidate_iter{cap:06d}.pt",
                                                           source_commit=meta["source"]["git_commit"], **export_options)
                result["training_origin"]="verified completed recovery; ZERO new training updates this session"
                phases=[]
            else:
                phases=[("training",cap,"train",recovery)]
        else:
            phases=[("precheck",2,"precheck",None),("resume_check",3,"resume_check",None),("training",cap,"train",None)]
        for phase,target,profile_name,explicit_recovery in phases:
            remaining=deadline-time.monotonic()-int(meta["evaluation_reserve_seconds"])-600
            if remaining<120:raise TimeoutError("insufficient budget for next training phase plus evaluation/save reserve")
            recorder.set_phase(phase)
            phase_root=output/phase;phase_root.mkdir()
            atomic_json(phase_root/"pilot_run_contract.json",contract)
            atomic_json(phase_root/"pilot_history_segment.json",{"segment":phase+"_"+uuid.uuid4().hex})
            control=output/"pilot_initial_policy.pt"
            if control.exists():shutil.copy2(control,phase_root/"pilot_initial_policy.pt")
            profile=deepcopy(read_json(dataset/meta["profiles"][profile_name]))
            controls=profile["resume"] if profile["mode"]=="resume" else profile["training"]
            controls["max_wall_seconds"]=min(float(controls["max_wall_seconds"]),remaining)
            if "max_wall_seconds_per_session" in controls:controls["max_wall_seconds_per_session"]=controls["max_wall_seconds"]
            atomic_json(phase_root/"profile.json",profile)
            phase_recovery=explicit_recovery
            if profile["mode"]=="resume" and phase_recovery is None:
                phase_recovery=output/(phase+"_recovery")
                safe_extract(previous_phase/"guandan-marvel-recovery.zip",phase_recovery)
                validate_recovery_directory(phase_recovery)
            if cap == 250 and previous_phase is not None and phase_recovery is not None:
                # The verified staged copy + retained atomic ZIP are sufficient
                # for retry even if the new worker fails before loading. Keeping
                # another raw generation can exhaust Kaggle's working disk when
                # a later checkpoint temporarily holds old and new shards.
                remove_owned_raw_checkpoints(previous_phase, output,
                    reason="verified staged recovery and compressed ZIP retained; retire redundant prior raw generation before next phase")
                append_jsonl(output/"events.jsonl", {"event":"retired_redundant_raw_before_next_phase",
                    "phase":phase,"retired_root":str(previous_phase),"verified_recovery":str(phase_recovery),
                    "timestamp":datetime.now(timezone.utc).isoformat()})
            if phase_recovery is not None and (phase_recovery/"artifacts/pilot_history").is_dir():
                shutil.copytree(phase_recovery/"artifacts/pilot_history",phase_root/"pilot_prior_history")
            command=worker_command(phase_root/"profile.json",phase_root,meta,recovery=phase_recovery,
                                   warm=dataset/"warm_start_policy.pt")
            start=time.monotonic()
            run_process(command,log=output/(phase+".console.log"),cwd=source,
                        timeout=min(remaining+540,deadline-time.monotonic()-120),events=output/"events.jsonl")
            progress_path=phase_root/"progress.json"
            if progress_path.exists() and read_json(progress_path).get("phase")=="budget_stopped":
                raise TimeoutError(f"worker stopped at an iteration boundary before target {target}")
            games_per_iter=int(meta.get("training_cap",{}).get("global_collect_batch_size",20))
            health=phase_health(phase_root,target=target,resumed=profile["mode"]=="resume",
                                expected_games_per_iter=games_per_iter)
            health["wall_seconds"]=time.monotonic()-start
            result["stages"][phase]=health
            atomic_json(output/"pilot_result.json",result)
            if phase=="precheck":shutil.copy2(phase_root/"pilot_initial_policy.pt",control)
            if previous_phase is not None:
                remove_owned_raw_checkpoints(previous_phase,output)
            # Extracted input shards are already represented in their verified ZIP; remove own temporary copies.
            if phase_recovery is not None and phase_recovery.is_relative_to(output):
                shutil.rmtree(phase_recovery)
            previous_phase=phase_root
        result["training_completed_iterations"]=cap
        result["training_completed_games"]=cap*collect_games
        result["training_verified"]=True
        recorder.set_phase("evaluation")
        evaluation_root=output/"evaluation";evaluation_root.mkdir()
        candidate=recovered_candidate if recovered_candidate is not None else Path(result["stages"]["training"]["candidate"])
        control=output/"pilot_initial_policy.pt"
        shutil.copy2(candidate,output/f"candidate_iter{cap:06d}.pt")
        eval_deadline=min(deadline-120,time.monotonic()+int(meta["evaluation_reserve_seconds"]))
        errors=[]
        def evaluate_gpu(shard):
            try:
                cmd=[sys.executable,"-m","guandan_marvel.kaggle.pilot_evaluation","--manifest",str(dataset/"development_deals.json"),
                     "--control",str(control),"--candidate",str(candidate),"--output",str(evaluation_root),"--shard",str(shard)]
                if cap != 50:
                    cmd += ["--maximum-iteration", str(cap), "--parent-provenance", str(dataset/"parent_run.json")]
                run_process(cmd,log=output/f"evaluation.gpu{shard}.console.log",cwd=source,
                            timeout=eval_deadline-time.monotonic(),events=output/f"evaluation.events.gpu{shard}.jsonl")
            except BaseException as error:errors.append(repr(error))
        workers=[threading.Thread(target=evaluate_gpu,args=(i,)) for i in range(2)]
        for w in workers:w.start()
        for w in workers:w.join()
        if errors:raise RuntimeError("evaluation failed/incomplete: "+"; ".join(errors))
        records=[]
        for i in range(2):records.extend(json.loads(line) for line in (evaluation_root/f"games.gpu{i}.jsonl").read_text(encoding="utf-8").splitlines() if line)
        evaluation=summarize(records,tasks_for(read_json(dataset/"development_deals.json")),
                             control_sha=sha256_file(control),candidate_sha=sha256_file(candidate))
        atomic_json(evaluation_root/"summary.json",evaluation)
        shutil.copy2(dataset/"development_deals.json",evaluation_root/"development_deals.json")
        result["evaluation"]=evaluation
        result["status"]="complete"
        result["remote_verified"]=True
    except BaseException as error:
        result["status"]="budget_stopped" if isinstance(error,TimeoutError) else "failed"
        result["error"]=f"{type(error).__name__}: {error}"
        (output/"pipeline_traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        result["traceback_file"]="pipeline_traceback.log"
        print("[PILOT STOP] "+result["error"],flush=True)
        append_jsonl(output/"events.jsonl",{"event":"pipeline_stopped","error":result["error"],
                                             "timestamp":datetime.now(timezone.utc).isoformat()})
    finally:
        if recorder is not None:recorder.close()
        recoveries=[]
        for bundle in output.glob("*/guandan-marvel-recovery.zip"):
            try:
                with zipfile.ZipFile(bundle) as z:
                    entry=json.loads(z.read("recovery_manifest.json"))
                    checkpoint_manifest=json.loads(z.read(str(entry["checkpoint"])))
                recoveries.append((int(entry["iteration"]),bundle,int(checkpoint_manifest["global_counters"]["completed_games_total"])))
            except Exception:
                pass
        if not recoveries and recovery is not None and recovery_validated:
            original=output.parent/"restored_recovery_input.zip"
            with zipfile.ZipFile(original,"w",compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
                for path in sorted(recovery.rglob("*")):
                    if path.is_file():z.write(path,path.relative_to(recovery).as_posix())
            try:
                with zipfile.ZipFile(original) as z:
                    entry=json.loads(z.read("recovery_manifest.json"))
                    checkpoint_manifest=json.loads(z.read(str(entry["checkpoint"])))
                recoveries.append((int(entry["iteration"]),original,int(checkpoint_manifest["global_counters"]["completed_games_total"])))
            except Exception:
                pass
        if recoveries:
            iteration,bundle,completed_games=max(recoveries,key=lambda x:x[0])
            shutil.copy2(bundle,output/"latest_recovery.zip")
            result["recovery"]={"iteration":iteration,"path":"latest_recovery.zip",
                                "sha256":sha256_file(output/"latest_recovery.zip")}
            result.setdefault("training_completed_iterations",iteration)
            result.setdefault("training_completed_games",completed_games)
        result["wall_seconds"]=time.monotonic()-started
        result["automatic_extension_beyond_cap"]=False
        if result.get("authorized_iteration_cap", 50) == 50:
            result["automatic_continuation_to_250"]=False
        archive=write_reports(output,result)
        result["results_zip"]=str(archive)
    return result
