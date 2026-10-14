"""Run one M09 formal selection/champion evaluation from prior Kaggle notebook outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile

PROCESS_STARTED = time.monotonic()


def emit(event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "elapsed_seconds": round(time.monotonic() - PROCESS_STARTED, 3),
        **fields,
    }
    print("FORMAL_EVAL " + json.dumps(payload, sort_keys=True, ensure_ascii=False), flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def safe_file(base: Path, spec: dict[str, object], label: str) -> Path:
    path = (base / str(spec["path"])).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise RuntimeError(f"{label} missing")
    observed = sha256_file(path)
    if observed != spec["sha256"]:
        raise RuntimeError(f"{label} hash mismatch")
    return path


def gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    if result.returncode != 0:
        return {"status": "error", "returncode": result.returncode, "stderr": result.stderr.strip()[:500]}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, utilization, memory_used, memory_total, power = parts
        try:
            gpus.append({
                "index": int(index),
                "name": name,
                "utilization_percent": float(utilization),
                "memory_used_mib": float(memory_used),
                "memory_total_mib": float(memory_total),
                "power_watts": float(power),
            })
        except ValueError:
            gpus.append({"raw": line})
    return {"status": "ok", "gpus": gpus}


def count_jsonl(path: str | Path) -> int:
    source = Path(path)
    if not source.is_file():
        return 0
    with source.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def evaluation_position(
    completed: int, *, total: int, lanes: tuple[str, ...], games_per_lane: int,
) -> dict[str, object]:
    bounded = max(0, min(int(completed), int(total)))
    if bounded >= total:
        return {"lane": "complete", "lane_index": len(lanes), "lane_completed": games_per_lane}
    lane_index = min(bounded // games_per_lane, len(lanes) - 1)
    return {
        "lane": lanes[lane_index],
        "lane_index": lane_index,
        "lane_completed": bounded - lane_index * games_per_lane,
    }


def monitor_evaluation(
    records_path: Path,
    *,
    total_games: int,
    lanes: tuple[str, ...],
    games_per_lane: int,
    heartbeat_seconds: float,
    stop: threading.Event,
) -> None:
    started = time.monotonic()
    last_completed = 0
    last_progress = started
    while True:
        now = time.monotonic()
        completed = min(count_jsonl(records_path), total_games)
        if completed > last_completed:
            last_completed = completed
            last_progress = now
        elapsed = max(now - started, 1e-9)
        rate = completed / elapsed
        eta = None if rate <= 0 else max(0.0, (total_games - completed) / rate)
        emit(
            "evaluation_heartbeat",
            games_completed=completed,
            games_total=total_games,
            progress_percent=round(100.0 * completed / total_games, 2),
            games_per_second=round(rate, 4),
            eta_seconds=None if eta is None else round(eta, 1),
            stalled_seconds=round(now - last_progress, 1),
            records_bytes=records_path.stat().st_size if records_path.is_file() else 0,
            gpu=gpu_snapshot(),
            **evaluation_position(completed, total=total_games, lanes=lanes, games_per_lane=games_per_lane),
        )
        if stop.wait(heartbeat_seconds):
            break


def find_source(input_root: Path, expected: dict[str, object]) -> Path:
    found: list[tuple[Path, Path]] = []
    for manifest_path in input_root.rglob("source_manifest.json"):
        try:
            manifest = load(manifest_path)
            root = manifest_path.parent
            if (
                manifest.get("git_commit") == expected["source_commit"]
                and manifest.get("manifest_hash") == expected["source_manifest_hash"]
                and (root / "guandan_marvel").is_dir()
            ):
                binaries = sorted((root / "nanopokerlib").glob("_guandan_cpp*.so"))
                if len(binaries) == 1 and sha256_file(binaries[0]) == expected["runtime_binary_sha256"]:
                    found.append((root, binaries[0]))
        except Exception:
            pass
    if not found:
        raise RuntimeError("matching built r2 source output not found; add the main-training notebook output as input")
    identities = {(load(root / "source_manifest.json")["manifest_hash"], sha256_file(binary)) for root, binary in found}
    if len(identities) != 1:
        raise RuntimeError("conflicting source outputs found")
    return sorted(found, key=lambda item: (len(str(item[0])), str(item[0])))[0][0]


def materialize_source(
    source: Path,
    working: Path,
    expected: dict[str, object],
    acceptance_profiles: Path,
) -> Path:
    target = (working / "guandan-marvel-source").resolve()
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=False)
    manifest = load(target / "source_manifest.json")
    binaries = sorted((target / "nanopokerlib").glob("_guandan_cpp*.so"))
    if manifest.get("git_commit") != expected["source_commit"] or manifest.get("manifest_hash") != expected["source_manifest_hash"]:
        raise RuntimeError("materialized source manifest mismatch")
    if len(binaries) != 1 or sha256_file(binaries[0]) != expected["runtime_binary_sha256"]:
        raise RuntimeError("materialized runtime binary mismatch")
    profile_target = target / "configs/acceptance_profiles.json"
    profile_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(acceptance_profiles, profile_target)
    return target


def find_recovery(input_root: Path, expected: dict[str, object], working: Path) -> Path:
    directories = []
    for manifest_path in input_root.rglob("recovery_manifest.json"):
        try:
            manifest = load(manifest_path)
            if (
                manifest.get("source_commit") == expected["source_commit"]
                and manifest.get("iteration") == expected["candidate_iteration"]
                and manifest.get("checkpoint_manifest_hash") == expected["candidate_checkpoint_manifest_hash"]
            ):
                directories.append(manifest_path.parent)
        except Exception:
            pass
    if directories:
        return sorted(directories, key=str)[0]
    for archive in input_root.rglob("guandan-marvel-recovery*.zip"):
        try:
            with zipfile.ZipFile(archive) as bundle:
                manifest = json.loads(bundle.read("recovery_manifest.json"))
                if (
                    manifest.get("source_commit") == expected["source_commit"]
                    and manifest.get("iteration") == expected["candidate_iteration"]
                    and manifest.get("checkpoint_manifest_hash") == expected["candidate_checkpoint_manifest_hash"]
                ):
                    target = working / "materialized-recovery"
                    if target.exists():
                        shutil.rmtree(target)
                    target.mkdir(parents=True)
                    bundle.extractall(target)
                    return target
        except Exception:
            pass
    raise RuntimeError("matching iter500 recovery not found; add the segment0500 notebook output as input")


def find_champion(input_root: Path, expected: dict[str, object]) -> Path:
    found = [
        path for path in input_root.rglob("iter000100.pt")
        if path.is_file() and sha256_file(path) == expected["initial_champion_sha256"]
    ]
    if not found:
        raise RuntimeError("initial iter100 champion not found; add the r2 IPPO pilot notebook output as input")
    return sorted(found, key=str)[0]


def export_candidate(training_path: Path, destination: Path) -> Path:
    import torch

    payload = torch.load(training_path, map_location="cpu", weights_only=False)
    required = (
        "algorithm", "architecture", "protocol_version", "backend", "backend_identity",
        "resolved_config", "resolved_config_hash", "agent_state_dict", "iteration",
        "merged_policy", "source_fingerprint", "source_hash", "source_parent_commit",
        "training_data_provenance",
    )
    if any(key not in payload for key in required):
        raise ValueError("training checkpoint lacks evaluation export fields")
    body = {key: payload[key] for key in required}
    body.update(
        schema="guandan-marvel-evaluation-checkpoint-v1",
        resume_classification="evaluation_only",
        training_resume_allowed=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(body, temporary)
    os.replace(temporary, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--working-root", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args(argv)
    input_root = args.input_root.resolve()
    working_root = args.working_root.resolve()
    emit("startup", control_manifest=str(args.control_manifest), input_root=str(input_root), working_root=str(working_root), gpu=gpu_snapshot())

    control = load(args.control_manifest)
    base = args.control_manifest.parent.resolve()
    expected = dict(control["expected"])
    heartbeat_seconds = float(dict(control.get("monitoring", {})).get("heartbeat_seconds", 20))
    if not 5 <= heartbeat_seconds <= 300:
        raise ValueError("monitoring heartbeat_seconds must be between 5 and 300")
    legacy = safe_file(base, dict(control["legacy_checkpoint"]), "legacy checkpoint")
    acceptance_profiles = safe_file(base, dict(control["acceptance_profiles"]), "acceptance profiles")
    emit(
        "control_validated",
        manifest_hash=control.get("manifest_hash"),
        runner_sha256=dict(control["script"])["sha256"],
        heartbeat_seconds=heartbeat_seconds,
    )

    emit("source_discovery_started")
    source_input = find_source(input_root, expected)
    emit("source_discovered", source_input=str(source_input))
    source = materialize_source(source_input, working_root, expected, acceptance_profiles)
    emit("source_materialized", source_root=str(source), runtime_binary_sha256=expected["runtime_binary_sha256"])

    sys.path.insert(0, str(source))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["GUANDAN_MARVEL_SOURCE_ROOT"] = str(source)
    os.environ["GUANDAN_MARVEL_SOURCE_COMMIT"] = str(expected["source_commit"])
    emit("runtime_import_started")
    from guandan.state import PreviousHandResult
    from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
    from guandan_marvel.evaluation.common import atomic_write_json
    from guandan_marvel.evaluation.evaluator import EvaluationRunner, FIXED_MATRIX
    from guandan_marvel.evaluation.manifests import build_deals_manifest, deals_for
    from guandan_marvel.evaluation.pool import OpponentPoolStore
    from guandan_marvel.evaluation.promotion import ChampionStore, PromotionPolicy
    from guandan_marvel.kaggle.bootstrap import validate_recovery_directory
    emit("runtime_import_complete")

    emit("recovery_discovery_started", candidate_iteration=expected["candidate_iteration"])
    recovery = find_recovery(input_root, expected, working_root)
    recovery_manifest = validate_recovery_directory(recovery)
    candidate_manifest = recovery / str(recovery_manifest["checkpoint"])
    checkpoint_manifest = load(candidate_manifest)
    rank0 = next(row for row in checkpoint_manifest["rank_shards"] if int(row["rank"]) == 0)
    training = candidate_manifest.parent / str(rank0["shard"])
    emit("recovery_validated", recovery_root=str(recovery), training_checkpoint=str(training))

    champion_input = find_champion(input_root, expected)
    emit("champion_discovered", champion_input=str(champion_input), champion_sha256=expected["initial_champion_sha256"])

    output = working_root / "guandan-marvel-formal-eval-iter0500"
    output.mkdir(parents=True, exist_ok=True)
    emit("candidate_export_started")
    candidate = export_candidate(training, output / "candidates/ippo_iter000500.pt")
    champion_path = output / "champion_state/champions/generation000001_iter000100.pt"
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(champion_input, champion_path)
    emit("candidate_export_complete", candidate_sha256=sha256_file(candidate), candidate_bytes=candidate.stat().st_size)

    emit("policy_validation_started", gpu=gpu_snapshot())
    candidate_artifact = load_v2_policy(candidate, device="cuda:0", expected_algorithm="ippo")
    champion_artifact = load_v2_policy(champion_path, device="cuda:0", expected_algorithm="ippo")
    legacy_artifact = load_legacy_v1_opponent(legacy)
    emit(
        "policy_validation_complete",
        candidate_iteration=candidate_artifact.iteration,
        champion_iteration=champion_artifact.iteration,
        legacy_algorithm=legacy_artifact.algorithm,
        gpu=gpu_snapshot(),
    )

    deals = build_deals_manifest(
        counts={"training": 0, "selection": 32, "final": 0},
        bases={"training": 999990000, "selection": 1000000000, "final": 1500000000},
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
    atomic_write_json(output / "deals_manifest.json", deals)
    champion_store = ChampionStore(output / "champion_state/champion_manifest.json")
    champion = champion_store.bootstrap(champion_artifact.identity, reason="M09 iter100 selected IPPO initial champion")
    pool = OpponentPoolStore(output / "champion_state/opponent_pool_manifest.json")
    pool.initialize(sampling_seed=20261014)
    pool.add_entry({
        "id": "champion-generation-1", "kind": "champion", "checkpoint": str(champion_artifact.path),
        "checkpoint_sha256": champion_artifact.sha256, "policy_identity": champion_artifact.identity,
        "weight": 1.0, "source_champion_manifest_hash": champion["hash"],
    })
    pool.add_entry({"id": "rule-anchor-v1", "kind": "rule", "weight": 0.25, "version": "v2-public-row-heuristic-v1"})
    pool.add_entry({
        "id": "legacy-v1-anchor", "kind": "legacy_v1", "checkpoint": str(legacy_artifact.path),
        "checkpoint_sha256": legacy_artifact.sha256, "legacy_identity": legacy_artifact.identity,
        "weight": 0.25,
    })

    lanes = tuple(FIXED_MATRIX)
    selection_deals = len(deals_for(deals, "selection"))
    games_per_lane = selection_deals * 4
    total_games = games_per_lane * len(lanes)
    selection_output = output / "selection_run"
    runner = EvaluationRunner(
        output_dir=selection_output,
        champion_store=champion_store,
        opponent_pool=pool,
        legacy_checkpoint=legacy,
        device="cuda:0",
        temperature=0.0,
        promotion_policy=PromotionPolicy(
            mode="formal", ci95_low_threshold=0.52, min_clusters=32,
            rule_min_win_rate=0.40, random_min_win_rate=0.70,
        ),
        provenance_policy="formal",
    )
    emit(
        "evaluation_started",
        selection_deals=selection_deals,
        seat_rotations=4,
        lanes=list(lanes),
        games_per_lane=games_per_lane,
        games_total=total_games,
        heartbeat_seconds=heartbeat_seconds,
        gpu=gpu_snapshot(),
    )
    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_evaluation,
        kwargs={
            "records_path": selection_output / "games.jsonl",
            "total_games": total_games,
            "lanes": lanes,
            "games_per_lane": games_per_lane,
            "heartbeat_seconds": heartbeat_seconds,
            "stop": stop,
        },
        name="formal-eval-heartbeat",
        daemon=True,
    )
    monitor.start()
    evaluation_started = time.perf_counter()
    try:
        result = runner.run_selection(deals_manifest=deals, candidate_checkpoint=candidate)
    finally:
        stop.set()
        monitor.join(timeout=heartbeat_seconds + 6)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    completed_games = count_jsonl(selection_output / "games.jsonl")
    emit(
        "evaluation_complete",
        games_completed=completed_games,
        games_total=total_games,
        evaluation_elapsed_seconds=round(evaluation_elapsed, 3),
        promotion_status=(result.get("promotion_decision") or {}).get("status"),
        gpu=gpu_snapshot(),
    )

    portable = {
        "schema": "guandan-marvel-m09-portable-champion-state-v1",
        "candidate_iteration": 500,
        "evaluation_result": result,
        "champion_manifest": champion_store.load(),
        "opponent_pool_manifest": pool.recover(),
        "candidate_checkpoint": {"path": "candidates/ippo_iter000500.pt", "sha256": sha256_file(candidate)},
        "initial_champion_checkpoint": {
            "path": "champion_state/champions/generation000001_iter000100.pt",
            "sha256": sha256_file(champion_path),
        },
        "elapsed_seconds": evaluation_elapsed,
        "final_holdout_used": False,
    }
    atomic_write_json(output / "formal_evaluation_summary.json", portable)

    zip_path = working_root / "guandan-marvel-formal-eval-iter0500.zip"
    temporary_zip = zip_path.with_suffix(".zip.tmp")
    emit("archive_started", archive=str(zip_path))
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output).as_posix())
    os.replace(temporary_zip, zip_path)
    final = {
        "status": "complete",
        "promotion": result.get("promotion_decision"),
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256_file(zip_path),
    }
    emit("complete", **final)
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
