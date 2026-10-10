# SPDX-License-Identifier: MIT
"""Kaggle-facing M07 orchestrator.  It never uploads and never fakes /kaggle evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import socket
import threading
import time
import uuid
import zipfile

import torch

from .checkpoints import build_recovery_bundle, load_ddp_manifest
from .common import append_jsonl, atomic_json, canonical_hash, sha256_file
from .discovery import discover_recovery, validate_dataset_manifest
from .evaluation import run_smoke_evaluation, write_empty_champion, write_empty_evaluation
from .telemetry import TelemetryRecorder, runtime_environment


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Console:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        text = str(message)
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        printable = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(printable, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(text.rstrip("\n") + "\n")
            stream.flush()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _is_kaggle(input_root: Path, working_root: Path) -> bool:
    return input_root.as_posix().startswith("/kaggle/input") and working_root.as_posix().startswith("/kaggle/working")


def _preflight(profile: dict[str, object], *, require_remote: bool, source_root: Path) -> dict[str, object]:
    environment = runtime_environment()
    checks: dict[str, object] = {
        "source_root_exists": source_root.is_dir(),
        "torchrun_available": shutil.which("torchrun") is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "distributed_available": torch.distributed.is_available(),
        "nccl_available": torch.distributed.is_nccl_available(),
    }
    required_count = int(profile.get("required_gpu_count", 2))
    required_name = str(profile.get("required_gpu_name_contains", "T4"))
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    checks["gpu_names"] = gpu_names
    checks["required_gpu_count"] = required_count
    checks["required_gpu_name_contains"] = required_name
    checks["gpu_count_ok"] = torch.cuda.device_count() == required_count
    checks["gpu_names_ok"] = len(gpu_names) == required_count and all(required_name.lower() in name.lower() for name in gpu_names)
    try:
        from guandan_marvel import cpp_game
        identity = cpp_game.get_backend_build_identity()
        checks["cpp_backend"] = cpp_game.BACKEND
        checks["cpp_source_matches_binary"] = identity.get("source_matches_loaded_binary") is not False
        checks["cpp_extension_sha256"] = identity.get("extension_sha256")
    except Exception as error:
        checks["cpp_import_error"] = f"{type(error).__name__}: {error}"
        checks["cpp_source_matches_binary"] = False
    remote_ok = all(bool(checks.get(key)) for key in (
        "source_root_exists", "torchrun_available", "cuda_available", "distributed_available",
        "nccl_available", "gpu_count_ok", "gpu_names_ok", "cpp_source_matches_binary",
    ))
    if require_remote and not remote_ok:
        raise RuntimeError(f"Kaggle T4x2 preflight failed: {checks}")
    return {
        "schema": "guandan-marvel-kaggle-preflight-v1",
        "status": "passed" if remote_ok else "local_only",
        "checks": checks,
        "environment": environment,
        "remote_preflight_passed": remote_ok,
        "remote_verified": False,
        "timestamp": _timestamp(),
    }


def _safe_extract_recovery(bundle: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe recovery archive member: {info.filename}")
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"recovery archive member escapes destination: {info.filename}")
        archive.extractall(destination)
    return destination


def _validate_recovery_root(recovery_root: Path) -> dict[str, object]:
    recovery = _load_json(recovery_root / "recovery_manifest.json")
    if recovery.get("schema") != "guandan-marvel-recovery-manifest-v1":
        raise ValueError("unsupported recovery manifest schema")
    declared = recovery.get("manifest_hash")
    actual = canonical_hash({key: value for key, value in recovery.items() if key != "manifest_hash"})
    if declared != actual:
        raise ValueError("recovery manifest hash mismatch")
    for row in recovery.get("files", []):
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe recovery file path: {relative}")
        path = (recovery_root / relative).resolve()
        if not path.is_relative_to(recovery_root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"recovery file is missing: {relative}")
        if path.stat().st_size != int(row.get("bytes", -1)) or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"recovery file hash mismatch: {relative}")
    return recovery


def _rank0_checkpoint_from_recovery(recovery_root: Path) -> Path:
    recovery = _validate_recovery_root(recovery_root)
    checkpoint_manifest = (recovery_root / str(recovery["checkpoint"])).resolve()
    payload = load_ddp_manifest(checkpoint_manifest, verify_shards=True)
    row = next(item for item in payload["rank_shards"] if int(item["rank"]) == 0)
    return checkpoint_manifest.parent / str(row["shard"])


def _build_output_zip(run_root: Path, destination: Path) -> Path:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(run_root.rglob("*")):
                if path.is_file() and path.name not in {"guandan-marvel-output.zip"}:
                    archive.write(path, path.relative_to(run_root).as_posix())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _run_torchrun(
    *,
    profile_path: Path,
    run_root: Path,
    manifest: dict[str, object],
    source: dict[str, object],
    recovery_root: Path | None,
    console: Console,
    local_cpu_test: bool,
) -> int:
    worker_args = [
        "-m", "guandan_marvel.kaggle.worker",
        "--profile", str(profile_path), "--run-root", str(run_root),
        "--package-id", str(manifest["package_id"]),
        "--source-commit", str(source["git_commit"]),
        "--source-archive-sha256", str(source["archive_sha256"]),
    ]
    if recovery_root is not None:
        worker_args.extend(["--recovery-root", str(recovery_root)])
    if local_cpu_test:
        worker_args.append("--local-cpu-test")
    env = os.environ.copy()
    source_root = Path(env["GUANDAN_MARVEL_SOURCE_ROOT"]).resolve()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(source_root) + (os.pathsep + current_pythonpath if current_pythonpath else "")
    if local_cpu_test and os.name == "nt":
        # This local CPU-only PyTorch build has no libuv support.  The Kaggle
        # delivery still uses torchrun; only the local gloo recovery exercise
        # starts equivalent rank processes directly to bypass torchrun's own
        # elastic rendezvous store.
        env["USE_LIBUV"] = "0"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        processes = []
        for rank in range(2):
            rank_env = env.copy()
            rank_env.update({
                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                "WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank),
            })
            command = [sys.executable, "-u", *worker_args]
            console.write(f"LOCAL_DDP_COMMAND rank={rank}: " + json.dumps(command, ensure_ascii=False))
            processes.append((rank, subprocess.Popen(
                command, cwd=source_root, env=rank_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )))
        threads = []
        def pump(rank, process):
            assert process.stdout is not None
            for line in process.stdout:
                console.write(f"[rank{rank}] {line.rstrip()}")
        for rank, process in processes:
            thread = threading.Thread(target=pump, args=(rank, process), daemon=True)
            thread.start(); threads.append(thread)
        codes = [process.wait() for _, process in processes]
        for thread in threads:
            thread.join(timeout=5)
        return max(codes)

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun executable is unavailable")
    command = [torchrun, "--standalone", "--nproc_per_node=2", *worker_args]
    console.write("COMMAND: " + json.dumps(command, ensure_ascii=False))
    with subprocess.Popen(
        command,
        cwd=source_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            console.write(line.rstrip("\n"))
        return process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--allow-local-validation", action="store_true")
    parser.add_argument("--local-cpu-test", action="store_true")
    args = parser.parse_args(argv)

    dataset_path = args.dataset_manifest.resolve()
    manifest = _load_json(dataset_path)
    validate_dataset_manifest(manifest, manifest_path=dataset_path)
    source = dict(manifest["source"])
    input_root = args.input_root.resolve()
    working_root = args.working_root.resolve()
    source_root = args.source_root.resolve()
    kaggle = _is_kaggle(input_root, working_root)
    if not kaggle and not args.allow_local_validation:
        raise RuntimeError("local execution requires --allow-local-validation; no /kaggle path is fabricated")
    if args.local_cpu_test and not args.allow_local_validation:
        raise RuntimeError("--local-cpu-test is allowed only for explicit local validation")
    mode = str(manifest["mode"])
    profile = json.loads(json.dumps(manifest["profiles"][mode]))
    if args.local_cpu_test and mode in {"pilot", "resume", "full"}:
        training = profile["training"]
        training["target_iterations"] = 1
        training["additional_iterations"] = 1
        training["checkpoint_interval"] = 1
        training["evaluation_interval"] = 0
        training["rollout_batch_size"] = 2
        training["collect_batch_size"] = 2
        training["replay_buffer_scale"] = 1
        training["mini_batches"] = 1
        training["agent_epoch"] = 1
        training["critic_aux_epoch"] = 1
        training["max_sampler_bytes_per_rank"] = 128 * 2**20
        training["model"] = {
            "architecture": "v1.3", "history_dim": 16, "feature_dim": 24,
            "decoder_layers": 1, "decoder_heads": 4, "state_layers": 1,
            "action_layers": 1, "fuse_layers": 1,
        }
        profile["smoke_horizon"] = 512
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{mode}-{uuid.uuid4().hex[:8]}"
    output_root = working_root / "guandan-marvel-output"
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    console = Console(run_root / "console.log")
    console.write(f"M07 mode={mode} run_id={run_id} kaggle={kaggle} local_cpu_test={args.local_cpu_test}")
    os.environ["GUANDAN_MARVEL_SOURCE_ROOT"] = str(source_root)
    os.environ["GUANDAN_MARVEL_SOURCE_COMMIT"] = str(source["git_commit"])
    os.environ["GUANDAN_MARVEL_SOURCE_ARCHIVE_SHA256"] = str(source["archive_sha256"])
    shutil.copy2(dataset_path.parent / str(source["manifest"]), run_root / "source_manifest.json")
    atomic_json(run_root / "runtime_environment.json", runtime_environment())
    profile_path = run_root / "selected_profile.json"
    atomic_json(profile_path, profile)
    write_empty_champion(run_root, reason="mode has not run evaluation")
    write_empty_evaluation(run_root, reason="mode has not run evaluation")
    (run_root / "metrics.jsonl").touch()
    (run_root / "telemetry.jsonl").touch()
    telemetry = TelemetryRecorder(
        run_root / "telemetry.jsonl", interval_seconds=float(profile.get("telemetry_interval_seconds", 5)),
    )
    telemetry.start()
    returncode = 1
    status = "failed"
    recovery_root = args.recovery_root.resolve() if args.recovery_root is not None else None
    try:
        require_remote = kaggle and mode in {"preflight", "pilot", "resume", "full", "eval_only"}
        preflight = _preflight(profile, require_remote=require_remote and not args.local_cpu_test, source_root=source_root)
        atomic_json(run_root / "preflight.json", preflight)
        if mode == "preflight":
            atomic_json(run_root / "resolved_config.json", {
                "schema": "guandan-marvel-kaggle-resolved-config-v1", "mode": mode,
                "profile": profile, "source": source, "remote_verified": False,
            })
            atomic_json(run_root / "checkpoint_manifest.json", {
                "schema": "guandan-marvel-checkpoint-manifest-v1", "latest": None,
                "history": [], "reason": "preflight performs no training",
            })
            atomic_json(run_root / "progress.json", {
                "schema": "guandan-marvel-progress-v1", "phase": "preflight_complete",
                "iteration": 0, "completed_games": 0, "env_steps": 0, "game_steps": 0,
                "latest_checkpoint": None, "latest_evaluation": None,
                "heartbeat_at": _timestamp(), "complete": True,
                "remote_preflight_passed": preflight["remote_preflight_passed"], "remote_verified": False,
            })
            append_jsonl(run_root / "metrics.jsonl", {
                "schema": "guandan-marvel-kaggle-metrics-v1", "mode": mode,
                "remote_preflight_passed": preflight["remote_preflight_passed"], "timestamp": _timestamp(),
            })
            returncode, status = 0, "complete"
        else:
            recovery_required = bool((manifest.get("recovery") or {}).get("required", False))
            if recovery_root is None:
                discovered = discover_recovery(
                    input_root,
                    source_commit=str(source["git_commit"]),
                    source_archive_sha256=str(source["archive_sha256"]),
                    required=recovery_required,
                )
                if discovered is not None:
                    recovery_root = _safe_extract_recovery(discovered.bundle_path, working_root / "guandan-marvel-recovery")
            if mode == "eval_only":
                if recovery_root is None:
                    raise RuntimeError("eval_only requires a unique matching recovery bundle")
                checkpoint = _rank0_checkpoint_from_recovery(recovery_root)
                evaluation_cfg = profile.get("evaluation") or {}
                run_smoke_evaluation(
                    checkpoint,
                    output_root=run_root,
                    base_deals=int(evaluation_cfg.get("base_deals", 1)),
                    seed_base=int(evaluation_cfg.get("seed_base", 870700)),
                    device="cuda:0" if torch.cuda.is_available() else "cpu",
                )
                atomic_json(run_root / "checkpoint_manifest.json", {
                    "schema": "guandan-marvel-checkpoint-manifest-v1",
                    "latest": {"checkpoint": str(checkpoint), "input_only": True, "sha256": sha256_file(checkpoint)},
                    "history": [],
                })
                atomic_json(run_root / "resolved_config.json", {
                    "schema": "guandan-marvel-kaggle-resolved-config-v1", "mode": mode,
                    "profile": profile, "source": source, "remote_verified": False,
                })
                atomic_json(run_root / "progress.json", {
                    "schema": "guandan-marvel-progress-v1", "phase": "eval_only_complete",
                    "iteration": 0, "heartbeat_at": _timestamp(), "complete": True,
                    "latest_checkpoint": str(checkpoint), "latest_evaluation": "evaluation_summary.json",
                    "remote_verified": False,
                })
                returncode, status = 0, "complete"
            else:
                telemetry.set_phase("training")
                returncode = _run_torchrun(
                    profile_path=profile_path,
                    run_root=run_root,
                    manifest=manifest,
                    source=source,
                    recovery_root=recovery_root,
                    console=console,
                    local_cpu_test=args.local_cpu_test,
                )
                if returncode == 0 and (run_root / "progress.json").is_file():
                    final_progress = _load_json(run_root / "progress.json")
                    if final_progress.get("phase") == "budget_stopped":
                        returncode = 2
                status = "complete" if returncode == 0 else ("budget_stopped" if returncode == 2 else "failed")
                if returncode not in {0, 2}:
                    raise RuntimeError(f"torchrun failed with exit code {returncode}")
    except BaseException as error:
        console.write(f"ERROR: {type(error).__name__}: {error}")
        atomic_json(run_root / "runner_failure.json", {
            "schema": "guandan-marvel-kaggle-runner-failure-v1",
            "error_type": type(error).__name__, "error": str(error), "timestamp": _timestamp(),
            "remote_verified": False,
        })
        status = "failed"
        returncode = 1
    finally:
        telemetry.set_phase(status)
        telemetry.close()
        native_extension = None
        if not args.local_cpu_test:
            try:
                from guandan_marvel import cpp_game
                identity = cpp_game.get_backend_build_identity()
                candidate = Path(str(identity.get("extension_path", "")))
                native_extension = candidate if candidate.is_file() else None
            except Exception:
                pass
        try:
            build_recovery_bundle(
                run_root=run_root,
                destination=run_root / "guandan-marvel-recovery.zip",
                source_commit=str(source["git_commit"]),
                source_archive_sha256=str(source["archive_sha256"]),
                package_id=str(manifest["package_id"]),
                status=status,
                native_extension=native_extension,
            )
        except Exception as error:
            console.write(f"RECOVERY_BUNDLE_ERROR: {type(error).__name__}: {error}")
        atomic_json(run_root / "run_result.json", {
            "schema": "guandan-marvel-kaggle-run-result-v1",
            "run_id": run_id, "mode": mode, "status": status, "exit_code": returncode,
            "kaggle_runtime": kaggle, "local_cpu_test": args.local_cpu_test,
            "package_ready": True, "remote_verified": False,
            "finished_at": _timestamp(),
        })
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(output_root / "latest_run.json", {
            "schema": "guandan-marvel-latest-run-v1", "run_id": run_id,
            "path": run_root.relative_to(output_root).as_posix(), "status": status,
        })
        try:
            _build_output_zip(run_root, output_root / "guandan-marvel-output.zip")
            recovery = run_root / "guandan-marvel-recovery.zip"
            if recovery.is_file():
                shutil.copy2(recovery, output_root / "guandan-marvel-recovery.zip")
        except Exception as error:
            console.write(f"OUTPUT_BUNDLE_ERROR: {type(error).__name__}: {error}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
