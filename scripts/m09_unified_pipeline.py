"""Run one automatic M09 Unified Segment -> commit -> formal evaluation cycle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Mapping

from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
from guandan_marvel.evaluation.common import atomic_write_json
from guandan_marvel.evaluation.dual_gpu import DeterministicWorkerGroup, WorkerCommand, compare_formal_outputs
from guandan_marvel.evaluation.evaluator import build_evaluation_plan
from guandan_marvel.evaluation.manifests import build_deals_manifest
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore, PromotionPolicy, decide_promotion
from guandan_marvel.evaluation.provenance import verify_data_isolation
from guandan_marvel.evaluation.resumable import PromotionTransaction, ResumableEvaluationStore, build_resumable_evaluation_plan, validate_resumable_plan
from guandan_marvel.kaggle.budget import build_completed_game_budget_contract
from guandan_marvel.kaggle.common import atomic_json, canonical_hash, sha256_file
from guandan_marvel.kaggle.unified_pipeline import (
    DEFAULT_SEGMENT_SCHEDULE, PIPELINE_STATE_NAME, AutoContinuationConfig,
    PipelineHooks, RecoveryCandidate, SegmentTarget, TrainingArtifacts,
    UnifiedSegmentPipeline, create_continuation_bundle, discover_previous_continuation,
    materialize_recovery_candidate, read_recovery_progress, select_next_segment,
    validate_recovery_archive,
)
from guandan.state import PreviousHandResult


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_from_zip(path: Path) -> dict[str, object]:
    import zipfile
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read("recovery_manifest.json"))
    if not isinstance(payload, dict):
        raise ValueError("recovery manifest must be an object")
    return payload


def _control_hash(control: Mapping[str, object]) -> str:
    return canonical_hash({key: value for key, value in control.items() if key != "manifest_hash"})


def _safe_control_file(base: Path, spec: Mapping[str, object], label: str) -> Path:
    relative = Path(str(spec.get("path", "")))
    path = (base / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"unsafe or missing control file: {label}")
    if sha256_file(path) != spec.get("sha256"):
        raise ValueError(f"control file hash mismatch: {label}")
    return path


def _gpu_snapshot() -> list[dict[str, object]]:
    command = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if completed.returncode != 0:
            return [{"status": "unavailable", "error": completed.stderr.strip()}]
        rows = []
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == 6:
                rows.append({"index": int(values[0]), "name": values[1], "utilization_percent": float(values[2]), "memory_used_mib": float(values[3]), "memory_total_mib": float(values[4]), "power_watts": float(values[5])})
        return rows
    except Exception as error:
        return [{"status": "unavailable", "error": f"{type(error).__name__}: {error}"}]


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "at": _timestamp(), **fields}, sort_keys=True, ensure_ascii=False), flush=True)


def _stream(command: list[str], *, cwd: Path, env: Mapping[str, str]) -> int:
    _emit("subprocess_started", command=command, cwd=str(cwd))
    process = subprocess.Popen(command, cwd=cwd, env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip("\n"), flush=True)
    code = int(process.wait())
    _emit("subprocess_complete", exit_code=code, command=command)
    return code


def _export_candidate(training_path: Path, destination: Path) -> Path:
    import torch
    payload = torch.load(training_path, map_location="cpu", weights_only=False)
    required = ("algorithm", "architecture", "protocol_version", "backend", "backend_identity", "resolved_config", "resolved_config_hash", "agent_state_dict", "iteration", "merged_policy", "source_fingerprint", "source_hash", "source_parent_commit", "training_data_provenance")
    if any(key not in payload for key in required):
        raise ValueError("training checkpoint lacks evaluation export fields")
    body = {key: payload[key] for key in required}
    body.update(schema="guandan-marvel-evaluation-checkpoint-v1", resume_classification="evaluation_only", training_resume_allowed=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(body, temporary)
    os.replace(temporary, destination)
    return destination


def _copy_verified_continuation(manifest_path: Path, *, output_root: Path) -> dict[str, object]:
    manifest = _load(manifest_path)
    source_root = manifest_path.parent.resolve()
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("continuation file inventory is missing")
    for row in files:
        relative = Path(str(row["path"]))
        source = (source_root / relative).resolve()
        if not source.is_relative_to(source_root) or not source.is_file():
            raise FileNotFoundError(f"continuation file missing: {relative}")
        if source.stat().st_size != int(row["bytes"]) or sha256_file(source) != row["sha256"]:
            raise ValueError(f"continuation file identity mismatch: {relative}")
    training_commit_source = source_root / "training/training_commit.json"
    training_commit = _load(training_commit_source)
    segment_id = str(training_commit["segment"]["segment_id"])
    mapping = {
        Path("training/training_commit.json"): output_root / "committed_training" / segment_id / "training_commit.json",
        Path("training/training_recovery.committed.zip"): output_root / "committed_training" / segment_id / "training_recovery.committed.zip",
        Path("pipeline/pipeline_state.json"): output_root / PIPELINE_STATE_NAME,
        Path("champion/champion_manifest.json"): output_root / "champion_state/champion_manifest.json",
        Path("champion/opponent_pool_manifest.json"): output_root / "champion_state/opponent_pool_manifest.json",
        Path("champion/promotion_transaction.json"): output_root / "champion_state/promotion_transaction.json",
    }
    for row in files:
        relative = Path(str(row["path"]))
        destination = mapping.get(relative)
        if destination is None and relative.parts and relative.parts[0] in {"evaluation", "candidates", "champion_state"}:
            destination = output_root / relative
        if destination is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256_file(destination) != row["sha256"]:
            shutil.copy2(source_root / relative, destination)
    return training_commit


def _runtime_dataset(*, dataset_manifest_path: Path, target: SegmentTarget, runtime_root: Path) -> Path:
    from guandan_marvel.kaggle.bootstrap import resolve_dataset_input

    resolved_dataset = resolve_dataset_input(dataset_manifest_path)
    dataset = _load(dataset_manifest_path)
    profile = json.loads(json.dumps(dataset["profiles"][dataset["mode"]]))
    profile["mode"] = "resume"
    profile["expected_algorithm"] = "ippo"
    profile["ddp"] = {"launcher": "torchrun", "world_size": 2}
    profile["resume"] = {"target_iterations": target.target_iteration, "checkpoint_interval": 25, "evaluation_interval": 0, "max_wall_seconds": 15_000, "checkpoint_keep_last": 2, "evaluation_keep_last": 2}
    profile["completed_game_budget"] = build_completed_game_budget_contract(
        comparison_id="m09-main-ippo-cumulative-v1", algorithm="ippo",
        target_completed_games=target.target_completed_games, global_collect_batch_size=4,
        expected_iterations=target.target_iteration,
        selection_dataset_hash=str(dataset["m09"]["plan"]["plan_hash"]),
    )
    profile.setdefault("m09", {})
    profile["m09"].update({"role": f"unified_{target.segment_id}", "mandatory_evaluation_after_segment": True, "final_holdout": False, "strength_claim": False})
    runtime = json.loads(json.dumps(dataset))
    runtime["mode"] = "resume"
    runtime["profiles"] = {"resume": profile}
    runtime["recovery"] = {"required": True, "auto_discovery": False}
    runtime["m09"]["role"] = f"unified_{target.segment_id}"
    runtime["m09"]["segment"] = target.identity()
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    base = dataset_manifest_path.parent
    shutil.copy2(base / str(dataset["source"]["manifest"]), runtime_root / str(dataset["source"]["manifest"]))
    if resolved_dataset.source.kind == "zip":
        shutil.copy2(resolved_dataset.source.path, runtime_root / str(dataset["source"]["archive"]))
    else:
        source_root_name = str(resolved_dataset.source.content_manifest["source_root"])
        shutil.copytree(resolved_dataset.source.path, runtime_root / source_root_name, symlinks=False)
    for key in ("plan", "locked_config"):
        item = dataset["m09"][key]
        shutil.copy2(base / str(item["path"]), runtime_root / str(item["path"]))
    path = runtime_root / "guandan_marvel_dataset.json"
    atomic_json(path, runtime)
    return path


def _initialize_or_restore_champion(*, output_root: Path, champion_seed: Path, legacy_checkpoint: Path) -> tuple[ChampionStore, OpponentPoolStore]:
    champion_store = ChampionStore(output_root / "champion_state/champion_manifest.json")
    pool_store = OpponentPoolStore(output_root / "champion_state/opponent_pool_manifest.json")
    if champion_store.path.is_file():
        champion_store.load()
        pool_store.recover()
        return champion_store, pool_store
    destination = output_root / "champion_state/champions/generation000001_iter000100.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(champion_seed, destination)
    champion_artifact = load_v2_policy(destination, device="cpu", expected_algorithm="ippo")
    champion = champion_store.bootstrap(champion_artifact.identity, reason="M09 Unified iter100 initial champion seed")
    pool_store.initialize(sampling_seed=20261014)
    pool_store.add_entry({"id": "champion-generation-1", "kind": "champion", "checkpoint": str(destination), "checkpoint_sha256": champion_artifact.sha256, "policy_identity": champion_artifact.identity, "weight": 1.0, "source_champion_manifest_hash": champion["hash"]})
    legacy = load_legacy_v1_opponent(legacy_checkpoint)
    pool_store.add_entry({"id": "legacy-v1-anchor", "kind": "legacy_v1", "checkpoint": str(legacy_checkpoint), "checkpoint_sha256": legacy.sha256, "legacy_identity": legacy.identity, "weight": 0.25})
    pool_store.add_entry({"id": "rule-anchor-v1", "kind": "rule", "version": "v2-public-row-heuristic-v1", "weight": 0.25})
    return champion_store, pool_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    args = parser.parse_args(argv)

    dataset_manifest_path = args.dataset_manifest.resolve()
    control_path = args.control_manifest.resolve()
    input_root = args.input_root.resolve()
    working_root = args.working_root.resolve()
    source_root = args.source_root.resolve()
    recovery_root = args.recovery_root.resolve()
    output_root = working_root / "guandan-marvel-unified-output"
    output_root.mkdir(parents=True, exist_ok=True)

    control = _load(control_path)
    if control.get("schema") != "guandan-marvel-m09-unified-control-v1":
        raise ValueError("unsupported unified control schema")
    control_hash = _control_hash(control)
    if control.get("manifest_hash") != control_hash:
        raise ValueError("unified control manifest hash mismatch")
    control_base = control_path.parent.resolve()
    source_identity = dict(control["source_identity"])
    champion_seed = _safe_control_file(control_base, control["champion_seed"], "champion seed")
    legacy_checkpoint = _safe_control_file(control_base, control["legacy_checkpoint"], "legacy checkpoint")
    acceptance_profiles = _safe_control_file(control_base, control["acceptance_profiles"], "acceptance profiles")
    profile_target = source_root / "configs/acceptance_profiles.json"
    profile_target.parent.mkdir(parents=True, exist_ok=True)
    if not profile_target.is_file() or sha256_file(profile_target) != sha256_file(acceptance_profiles):
        shutil.copy2(acceptance_profiles, profile_target)
    if sha256_file(profile_target) != control["acceptance_profiles"]["sha256"]:
        raise ValueError("materialized acceptance profile identity mismatch")
    _emit("acceptance_profiles_materialized", path=str(profile_target), sha256=sha256_file(profile_target))
    worker_script = _safe_control_file(control_base, control["formal_worker"], "formal worker")
    plan_path = _safe_control_file(control_base, control["main_training_plan"], "main training plan")
    main_plan = _load(plan_path)
    if main_plan.get("plan_hash") != control.get("main_plan_hash"):
        raise ValueError("unified control/main plan hash mismatch")
    AutoContinuationConfig.from_mapping(control.get("auto_continuation"))
    validation_mode = os.environ.get("GUANDAN_M09_REMOTE_VALIDATION", "").strip().upper()
    validation_control = dict(control.get("remote_validation") or {})
    if not validation_mode and not bool(validation_control.get("production_enabled", False)):
        validation_mode = str(validation_control.get("default_mode", "A")).upper()
        _emit("remote_validation_defaulted", mode=validation_mode, reason="production Segment1000 is gated")
    if validation_mode and validation_mode not in set(validation_control.get("modes", [])):
        raise ValueError(f"unsupported GUANDAN_M09_REMOTE_VALIDATION={validation_mode!r}")

    previous = discover_previous_continuation(input_root, expected_control_hash=control_hash, expected_source_identity=source_identity)
    restored_state = None
    if previous is not None:
        _emit("continuation_restore_started", manifest=str(previous[0]))
        _copy_verified_continuation(previous[0], output_root=output_root)
        restored_state = _load(output_root / PIPELINE_STATE_NAME)
        _emit("continuation_restore_complete", phase=restored_state.get("phase"))

    champion_store, pool_store = _initialize_or_restore_champion(output_root=output_root, champion_seed=champion_seed, legacy_checkpoint=legacy_checkpoint)
    recovery_progress = read_recovery_progress(recovery_root)
    _emit("recovery_selected", **recovery_progress, gpu=_gpu_snapshot())

    if validation_mode == "C":
        if restored_state is None or restored_state.get("phase") != "COMPLETE":
            raise RuntimeError("Remote C requires a completed Remote A Notebook Output")
        segment = restored_state.get("segment") or {}
        segment_id = str(segment.get("segment_id", ""))
        single_root = output_root / "evaluation" / segment_id
        plan = _load(single_root / "evaluation_plan.json")
        if not (single_root / "evaluation_summary.json").is_file() or plan.get("promotion_allowed") is not False:
            raise RuntimeError("Remote C requires the completed non-promoting smoke evaluation from Remote A")
        dual_root = output_root / "evaluation" / f"{segment_id}-dual-remote-c"
        dual_root.mkdir(parents=True, exist_ok=True)
        candidate_path = Path(str(plan["candidate"]["checkpoint_path"]))
        champion_path = Path(str(champion_store.load()["champion"]["checkpoint_path"]))
        pool_path = Path(str(plan["opponent_pool_sample"]["checkpoint"]))
        context = {
            "schema": "guandan-marvel-m09-formal-worker-context-v1",
            "plan_hash": plan["hash"], "plan_path": str(single_root / "evaluation_plan.json"),
            "deals_manifest_path": str(single_root / "deals_manifest.json"),
            "candidate_checkpoint": str(candidate_path), "champion_checkpoint": str(champion_path),
            "pool_checkpoint": str(pool_path), "legacy_checkpoint": str(legacy_checkpoint),
            "output_dir": str(dual_root),
        }
        context_path = dual_root / "worker_context.json"
        atomic_json(context_path, context)
        commands = [WorkerCommand(index, index, (sys.executable, "-u", str(worker_script), "--context", str(context_path), "--worker-index", str(index), "--world-size", "2", "--device", "cuda:0")) for index in range(2)]
        DeterministicWorkerGroup(output_dir=dual_root, plan=plan, commands=commands, base_env=os.environ, heartbeat_seconds=float(control.get("monitoring", {}).get("heartbeat_seconds", 20))).run()
        ResumableEvaluationStore(dual_root, plan).finalize()
        consistency = compare_formal_outputs(single_root, dual_root)
        result = {"schema": "guandan-marvel-m09-remote-c-result-v1", "status": "passed", "complete": True, "segment_id": segment_id, "consistency": consistency, "final_holdout_generated": False, "finished_at": _timestamp()}
        atomic_json(output_root / "remote_c_result.json", result)
        atomic_json(output_root / "latest_run.json", {"schema": "guandan-marvel-m09-unified-latest-run-v1", "status": "remote_c_complete", "exit_code": 0, "remote_validation": "C", "remote_c_result": str(output_root / "remote_c_result.json"), "final_holdout_generated": False, "finished_at": _timestamp()})
        _emit("remote_c_complete", consistency=consistency, gpu=_gpu_snapshot())
        return 0

    heartbeat_stop = threading.Event()
    heartbeat_started = time.monotonic()

    def heartbeat() -> None:
        interval = float(control.get("monitoring", {}).get("heartbeat_seconds", 20))
        while not heartbeat_stop.wait(interval):
            state = _load(output_root / PIPELINE_STATE_NAME) if (output_root / PIPELINE_STATE_NAME).is_file() else {}
            segment = state.get("segment") or {}
            evaluation_root = output_root / "evaluation" / str(segment.get("segment_id", "unknown"))
            completed = sum(_count_jsonl(path) for path in evaluation_root.glob("games.gpu*.jsonl"))
            total = int(_load(evaluation_root / "evaluation_plan.json").get("task_count", 0)) if (evaluation_root / "evaluation_plan.json").is_file() else 0
            _emit("pipeline_heartbeat", phase=state.get("phase", "DISCOVER_INPUTS"), training_iteration=recovery_progress["iteration"], training_completed_games=recovery_progress["completed_games"], evaluation_completed=completed, evaluation_total=total, elapsed_seconds=round(time.monotonic() - heartbeat_started, 3), gpu=_gpu_snapshot())

    heartbeat_thread = threading.Thread(target=heartbeat, name="unified-pipeline-heartbeat", daemon=True)
    heartbeat_thread.start()
    current_recovery = {"root": recovery_root}

    def train(target: SegmentTarget, pipeline_output: Path) -> TrainingArtifacts:
        runtime_root = output_root / "runtime" / target.segment_id / "dataset"
        runtime_manifest = _runtime_dataset(dataset_manifest_path=dataset_manifest_path, target=target, runtime_root=runtime_root)
        training_work = output_root / "runtime" / target.segment_id / "training_work"
        command = [sys.executable, "-u", "-m", "guandan_marvel.kaggle.runner", "--dataset-manifest", str(runtime_manifest), "--input-root", str(input_root), "--working-root", str(training_work), "--source-root", str(source_root), "--recovery-root", str(current_recovery["root"])]
        code = _stream(command, cwd=source_root, env=os.environ)
        if code != 0:
            raise RuntimeError(f"DDP training subprocess failed or stopped before exact target: exit={code}")
        output = training_work / "guandan-marvel-output"
        recovery_zip = output / "guandan-marvel-recovery.zip"
        validation = validate_recovery_archive(recovery_zip)
        extracted = output_root / "runtime" / target.segment_id / "completed_recovery"
        materialize_recovery_candidate(RecoveryCandidate("zip", recovery_zip, _load_from_zip(recovery_zip), validation["sha256"]), extracted)
        current_recovery["root"] = extracted
        return TrainingArtifacts(recovery_zip=recovery_zip, recovery_root=extracted)

    def _prepare_evaluation(target: SegmentTarget, commit: Mapping[str, object]) -> tuple[Path, dict[str, object], Path]:
        evaluation_root = output_root / "evaluation" / target.segment_id
        evaluation_root.mkdir(parents=True, exist_ok=True)
        committed_dir = output_root / "committed_training" / target.segment_id
        committed_zip = committed_dir / str(commit["recovery"]["path"])
        extracted = output_root / "runtime" / target.segment_id / "committed_recovery"
        candidate_manifest = _load_from_zip(committed_zip)
        materialize_recovery_candidate(RecoveryCandidate("zip", committed_zip, candidate_manifest, sha256_file(committed_zip)), extracted)
        checkpoint_path = extracted / str(candidate_manifest["checkpoint"])
        checkpoint_manifest = _load(checkpoint_path)
        rank0 = next(row for row in checkpoint_manifest["rank_shards"] if int(row["rank"]) == 0)
        training_checkpoint = checkpoint_path.parent / str(rank0["shard"])
        candidate_path = output_root / "candidates" / f"ippo_iter{target.target_iteration:06d}.pt"
        if not candidate_path.is_file():
            _export_candidate(training_checkpoint, candidate_path)
        candidate = load_v2_policy(candidate_path, device="cpu", expected_algorithm="ippo")
        champion_manifest = champion_store.load()
        champion_path = Path(str(champion_manifest["champion"]["checkpoint_path"]))
        champion = load_v2_policy(champion_path, device="cpu", expected_algorithm="ippo")
        pool_sample = pool_store.sample(sample_seed=int(candidate.iteration))
        pool_path = Path(str(pool_sample["checkpoint"]))
        pool_policy = load_v2_policy(pool_path, device="cpu")
        legacy = load_legacy_v1_opponent(legacy_checkpoint)
        deals_path = evaluation_root / "deals_manifest.json"
        if deals_path.is_file():
            deals = _load(deals_path)
        else:
            selection_clusters = int(validation_control.get("selection_clusters", 2)) if validation_mode in {"A", "B"} else 32
            deals = build_deals_manifest(
                counts={"training": 0, "selection": selection_clusters, "final": 0},
                bases={"training": 999_990_000, "selection": 1_000_000_000, "final": 1_500_000_000},
                level_ranks=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                leader_seats=(0, 1, 2, 3),
                previous_results=(
                    None,
                    PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN"),
                    PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD"),
                    PreviousHandResult((1, 2, 4, 3), 0, "HEAD_LAST"),
                    PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD", True),
                ),
            )
            atomic_write_json(deals_path, deals)
        isolation = verify_data_isolation(artifacts={artifact.sha256: artifact for artifact in (candidate, champion, pool_policy)}.values(), deals_manifest=deals, domain="selection", policy="formal")
        resumable_plan_path = evaluation_root / "evaluation_plan.json"
        if resumable_plan_path.is_file():
            plan = _load(resumable_plan_path)
            validate_resumable_plan(plan)
            if plan["candidate"]["checkpoint_sha256"] != candidate.sha256:
                raise ValueError("resumed evaluation candidate differs from the durable plan")
            if plan["current_champion"].get("checkpoint_sha256") != champion.sha256:
                raise ValueError("resumed evaluation champion differs from the durable plan")
        else:
            policy = control["promotion_policy"]
            smoke_validation = validation_mode in {"A", "B"}
            promotion_policy = PromotionPolicy(
                mode="smoke" if smoke_validation else "formal",
                ci95_low_threshold=float(policy["ci95_low_threshold"]),
                min_clusters=1 if smoke_validation else int(policy["min_clusters"]),
                rule_min_win_rate=float(policy["rule_min_win_rate"]),
                random_min_win_rate=float(policy["random_min_win_rate"]),
            )
            base_plan = build_evaluation_plan(
                mode="selection", deals_manifest=deals, deal_domain="selection",
                candidate_identity=candidate.identity, champion_identity=champion.identity,
                legacy_identity=legacy.identity, pool_sample=pool_sample, temperature=0.0,
                data_isolation=isolation, promotion_policy=promotion_policy,
                evaluation_id=f"m09-{target.segment_id}-{candidate.sha256[:16]}",
            )
            plan = build_resumable_evaluation_plan(base_plan=base_plan, materialized_deals=isolation["materialized_deals"])
            atomic_write_json(resumable_plan_path, plan)
        context = {
            "schema": "guandan-marvel-m09-formal-worker-context-v1",
            "plan_hash": plan["hash"], "plan_path": str(resumable_plan_path),
            "deals_manifest_path": str(deals_path), "candidate_checkpoint": str(candidate_path),
            "champion_checkpoint": str(champion_path), "pool_checkpoint": str(pool_path),
            "legacy_checkpoint": str(legacy_checkpoint), "output_dir": str(evaluation_root),
        }
        context_path = evaluation_root / "worker_context.json"
        atomic_json(context_path, context)
        return evaluation_root, plan, context_path

    def evaluate(target: SegmentTarget, commit: Mapping[str, object], pipeline_output: Path) -> Mapping[str, object]:
        evaluation_root, plan, context_path = _prepare_evaluation(target, commit)
        workers = int(control["evaluation"]["formal_workers_default"])
        requested = os.environ.get("GUANDAN_M09_FORMAL_WORKERS")
        if requested is not None:
            workers = int(requested)
        dual_remote_validated = bool(control["evaluation"].get("dual_gpu_formal_remote_validated", False))
        if workers == 2 and not dual_remote_validated and os.environ.get("GUANDAN_M09_ALLOW_EXPERIMENTAL_DUAL") != "1":
            raise RuntimeError("dual-GPU formal evaluation is experimental until Remote C; explicit opt-in is required")
        if workers not in {1, 2}:
            raise ValueError("formal_workers must be 1 or 2")
        commands = [
            WorkerCommand(index, index, (sys.executable, "-u", str(worker_script), "--context", str(context_path), "--worker-index", str(index), "--world-size", str(workers), "--device", "cuda:0"))
            for index in range(workers)
        ]
        group = DeterministicWorkerGroup(output_dir=evaluation_root, plan=plan, commands=commands, base_env=os.environ, heartbeat_seconds=float(control.get("monitoring", {}).get("heartbeat_seconds", 20)))
        group.run()
        store = ResumableEvaluationStore(evaluation_root, plan)
        result = store.finalize()
        summary = _load(evaluation_root / "evaluation_summary.json")
        cluster = summary["cluster_summary"]
        health = {
            "illegal": sum(int(row["eval/illegal_count"]) for row in cluster["per_opponent"].values()),
            "truncated": sum(int(row["eval/truncated_count"]) for row in cluster["per_opponent"].values()),
            "nonfinite": 0,
        }
        floors = {
            "rule": float(cluster["per_opponent"]["rule"]["eval/win_rate"]) >= float(control["promotion_policy"]["rule_min_win_rate"]),
            "random": float(cluster["per_opponent"]["random"]["eval/win_rate"]) >= float(control["promotion_policy"]["random_min_win_rate"]),
        }
        return {**result, "identity_verified": True, "summary_path": str(evaluation_root / "evaluation_summary.json"), "health": health, "floors": floors, "worker_count": workers}

    def promote(target: SegmentTarget, evaluation: Mapping[str, object], pipeline_output: Path) -> Mapping[str, object]:
        evaluation_root = output_root / "evaluation" / target.segment_id
        plan = _load(evaluation_root / "evaluation_plan.json")
        summary = _load(evaluation_root / "evaluation_summary.json")
        champion_before = champion_store.load()
        values = control["promotion_policy"]
        plan_policy = dict(plan.get("promotion_policy") or {})
        smoke_validation = plan_policy.get("mode") == "smoke"
        policy = PromotionPolicy(mode="smoke" if smoke_validation else "formal", ci95_low_threshold=float(values["ci95_low_threshold"]), min_clusters=1 if smoke_validation else int(values["min_clusters"]), rule_min_win_rate=float(values["rule_min_win_rate"]), random_min_win_rate=float(values["random_min_win_rate"]))
        decision = decide_promotion(
            summary["cluster_summary"], candidate_identity=plan["candidate"], champion_manifest=champion_before,
            evaluation_hash=str(evaluation["evaluation_hash"]), selection_deals_hash=str(plan["deals_manifest_hash"]),
            plan_hash=str(plan["hash"]), summary_hash=str(summary["hash"]),
            selection_eligible=bool(plan["data_isolation"]["formal_selection_eligible"]), policy=policy,
        )
        atomic_write_json(evaluation_root / "promotion_decision.json", decision)
        if not decision["promote"]:
            return decision
        candidate_path = Path(str(plan["candidate"]["checkpoint_path"]))
        generation = int(champion_before["generation"]) + 1
        champion_path = output_root / "champion_state/champions" / f"generation{generation:06d}_iter{target.target_iteration:06d}.pt"
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        if not champion_path.is_file():
            shutil.copy2(candidate_path, champion_path)
        promoted_candidate = dict(plan["candidate"])
        promoted_candidate["checkpoint_path"] = str(champion_path.resolve())
        transaction = PromotionTransaction(evaluation_root / "promotion_transaction.json")
        transaction_id = f"promotion-{decision['hash'][:16]}"

        def publish_champion():
            current = champion_store.load()
            if current.get("promotion", {}).get("decision_hash") == decision["hash"]:
                return {"manifest_hash": current["hash"], "generation": current["generation"]}
            updated = champion_store.promote(promoted_candidate, decision=decision, expected_plan_hash=str(plan["hash"]), expected_summary_hash=str(summary["hash"]), transaction_id=transaction_id)
            return {"manifest_hash": updated["hash"], "generation": updated["generation"]}

        def publish_pool(champion_publication):
            current_pool = pool_store.recover()
            entry_id = f"champion-generation-{generation}"
            if not any(entry.get("id") == entry_id for entry in current_pool["entries"]):
                current_pool = pool_store.add_entry({"id": entry_id, "kind": "champion", "checkpoint": str(champion_path), "checkpoint_sha256": sha256_file(champion_path), "policy_identity": promoted_candidate, "weight": 1.0, "promotion_transaction_id": transaction_id, "source_champion_manifest_hash": champion_publication["manifest_hash"]})
            current_champion = champion_store.load()
            sync = current_champion.get("pool_sync") or {}
            if sync.get("status") != "synced":
                current_champion = champion_store.mark_pool_synced(transaction_id=transaction_id, pool_manifest_hash=str(current_pool["hash"]))
            return {"manifest_hash": current_pool["hash"], "revision": current_pool["revision"], "champion_manifest_hash": current_champion["hash"]}

        transaction_result = transaction.execute(decision_hash=str(decision["hash"]), candidate_checkpoint_sha256=str(plan["candidate"]["checkpoint_sha256"]), publish_champion=publish_champion, publish_pool=publish_pool)
        return {**decision, "transaction": transaction_result}

    pending_target = None
    if restored_state and restored_state.get("phase") in {"TRAINING_COMPLETE_EVALUATION_FAILED", "EVALUATION_INCOMPLETE_RESUMABLE", "PROMOTION_REPAIR_REQUIRED", "PACKAGING_FAILED_RECOVERABLE"}:
        segment = restored_state.get("segment") or {}
        pending_target = SegmentTarget(str(segment["segment_id"]), int(segment["target_iteration"]), int(segment["target_completed_games"]))

    target_override = None
    if pending_target is None and validation_mode in {"A", "B"}:
        target_override = SegmentTarget(
            f"remote_{validation_mode.lower()}_smoke0502",
            int(validation_control["smoke_target_iteration"]),
            int(validation_control["smoke_target_completed_games"]),
        )
    pipeline = UnifiedSegmentPipeline(output_root=output_root, source_identity=source_identity, control_hash=control_hash, hooks=PipelineHooks(train=train, evaluate=evaluate, promote=promote))
    exit_code = 0
    caught: BaseException | None = None
    try:
        state = pipeline.run(iteration=int(recovery_progress["iteration"]), completed_games=int(recovery_progress["completed_games"]), pending_target=pending_target, target_override=target_override)
    except BaseException as error:
        caught = error
        state = pipeline.state.load() or {}
        exit_code = 1
        _emit("pipeline_error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5.0)

    segment = state.get("segment") or {}
    segment_id = str(segment.get("segment_id", ""))
    commit_path = output_root / "committed_training" / segment_id / "training_commit.json"
    continuation = None
    if segment_id and commit_path.is_file():
        try:
            target_obj = SegmentTarget(segment_id, int(segment["target_iteration"]), int(segment["target_completed_games"]))
            pipeline.state.transition("PACKAGING", segment=target_obj, details={"pre_packaging_phase": state.get("phase")})
            evaluation_root = output_root / "evaluation" / segment_id
            additional: dict[str, Path] = {}
            evaluation_history = output_root / "evaluation"
            if evaluation_history.is_dir():
                for path in evaluation_history.rglob("*"):
                    if path.is_file():
                        additional[f"evaluation/{path.relative_to(evaluation_history).as_posix()}"] = path
            for root_name in ("candidates", "champion_state/champions"):
                source_dir = output_root / root_name
                if source_dir.is_dir():
                    for path in source_dir.rglob("*"):
                        if path.is_file():
                            additional[f"{root_name}/{path.relative_to(source_dir).as_posix()}"] = path
            next_target = select_next_segment(target_obj.target_iteration, target_obj.target_completed_games)
            plan_hash = str(control["main_plan_hash"])
            if (evaluation_root / "evaluation_plan.json").is_file():
                plan_hash = str(_load(evaluation_root / "evaluation_plan.json")["hash"])
            bundle_kwargs = {
                "output_root": output_root,
                "training_commit_path": commit_path,
                "pipeline_state_path": output_root / PIPELINE_STATE_NAME,
                "champion_manifest_path": champion_store.path,
                "opponent_pool_manifest_path": pool_store.path,
                "promotion_transaction_path": None,
                "evaluation_paths": [],
                "additional_files": additional,
                "next_target": next_target,
                "source_identity": source_identity,
                "control_hash": control_hash,
                "plan_hash": plan_hash,
            }
            continuation = create_continuation_bundle(**bundle_kwargs)
            final_phase = "COMPLETE" if exit_code == 0 and state.get("phase") == "PROMOTION_COMMITTED" else str(state.get("phase"))
            pipeline.state.transition(final_phase, segment=target_obj, details={"continuation_archive": continuation["archive"], "pre_packaging_phase": state.get("phase")}, error=state.get("error") if final_phase != "COMPLETE" else None)
            # Rebuild once so the portable bundle contains the terminal state,
            # not only the transient PACKAGING phase.
            continuation = create_continuation_bundle(**bundle_kwargs)
        except BaseException as packaging_error:
            exit_code = 1
            caught = caught or packaging_error
            pipeline.state.transition("PACKAGING_FAILED_RECOVERABLE", segment=SegmentTarget(segment_id, int(segment["target_iteration"]), int(segment["target_completed_games"])), error=f"{type(packaging_error).__name__}: {packaging_error}")
            _emit("packaging_error", error=f"{type(packaging_error).__name__}: {packaging_error}")

    latest = {
        "schema": "guandan-marvel-m09-unified-latest-run-v1",
        "status": "complete" if exit_code == 0 else "failed_recoverable",
        "exit_code": exit_code,
        "pipeline_state": str(output_root / PIPELINE_STATE_NAME),
        "continuation_archive": None if continuation is None else continuation["archive"],
        "continuation_sha256": None if continuation is None else continuation["archive_sha256"],
        "final_holdout_generated": False,
        "finished_at": _timestamp(),
    }
    atomic_json(output_root / "latest_run.json", latest)
    if continuation is not None and os.environ.get("GUANDAN_M09_KEEP_RUNTIME", "0") != "1":
        runtime_root = output_root / "runtime"
        if runtime_root.is_dir():
            shutil.rmtree(runtime_root)
            _emit("runtime_cleanup_complete", path=str(runtime_root))
    _emit("pipeline_finished", **latest, gpu=_gpu_snapshot())
    if caught is not None:
        print(f"RECOVERABLE_ERROR: {type(caught).__name__}: {caught}", file=sys.stderr, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
