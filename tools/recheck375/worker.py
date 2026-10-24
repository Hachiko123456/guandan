"""Independent GPU evaluation worker. No trainer, optimizer or training CLI."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from .common import append, atomic_json, canonical, read_json, sha, state_hash, validate_plan, verify_driver, verify_hash
from .protocol import ARMS, CACHED_ARMS, NEW_ARMS, compare_replay, tasks_for, validate_row


def validate_job(job, plan, manifest):
    verify_hash(job, "job_hash")
    if job["plan_hash"] != plan["plan_hash"] or job["gpu_index"] not in (0, 1):
        raise ValueError("wrong worker job identity")
    phase = job["phase"]
    if phase == "cache_replay":
        allowed = tasks_for(manifest, CACHED_ARMS, plan, first_deals=4)
    elif phase == "p375_smoke":
        allowed = tasks_for(manifest, NEW_ARMS, plan, first_deals=4)
    elif phase == "p375_main":
        smoke = {t["task_id"] for t in tasks_for(manifest, NEW_ARMS, plan, first_deals=4)}
        allowed = [t for t in tasks_for(manifest, NEW_ARMS, plan) if t["task_id"] not in smoke]
    elif phase == "local_replay" and plan.get("test_only"):
        allowed = tasks_for(manifest, NEW_ARMS, plan)
    else:
        raise ValueError("unrecognized phase; no training operation exists")
    by_id = {t["task_id"]: t for t in allowed}
    ids = [t["task_id"] for t in job["tasks"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate worker tasks")
    for task in job["tasks"]:
        if by_id.get(task["task_id"]) != task:
            raise ValueError("worker task is outside the authorized evaluation plan")
    expected = job.get("replay_records", {})
    if phase in ("cache_replay", "local_replay"):
        if set(expected) != set(ids):
            raise ValueError("replay is missing exact expected records")
        for task in job["tasks"]:
            validate_row(expected[task["task_id"]], task, plan)
    elif expected:
        raise ValueError("unexpected replay records")


def run_job(dataset, job_path, output, *, testing=False, device=None):
    import torch
    from guandan_marvel import cpp_game
    from guandan_marvel.evaluation.checkpoints import load_v2_policy
    from guandan_marvel.evaluation.host import play_game
    from guandan_marvel.training.checkpoints import build_backend_identity

    dataset, output = Path(dataset), Path(output)
    plan = validate_plan(read_json(dataset / "plan.json"), testing=testing)
    if testing != bool(plan.get("test_only")):
        raise ValueError("production/test evaluation mismatch")
    if not testing:
        verify_driver(plan)
        if str(torch.__version__) != plan["torch"]:
            raise RuntimeError("PyTorch changed since cached evaluation; refusing an unverified replay environment")
        if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in (0, 1)):
            raise RuntimeError("requires exactly two T4 GPUs; no fallback")
        if device is not None:
            raise ValueError("production device is selected by the bound job")
    elif device is None:
        raise ValueError("local test requires an EXPLICIT device; no silent fallback")
    manifest = read_json(dataset / "development_deals.json")
    job = read_json(job_path)
    validate_job(job, plan, manifest)
    rank = job["gpu_index"]
    device = device if testing else f"cuda:{rank}"
    torch.set_num_threads(1)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    if cpp_game.GAME_SPEC.max_game_horizon != 1024:
        raise ValueError("native horizon mismatch before any game")
    backend = build_backend_identity(cpp_game)
    if not testing and backend["hash"] != plan["backend_hash"]:
        raise ValueError("native absolute path/Python/binary identity changed")
    needed = sorted({model for t in job["tasks"] for model in ARMS[t["arm"]] if model is not None})
    models, hashes, training_hashes = {}, {}, set()
    initial_seeds = set()
    for name in needed:
        row = plan["models"][name]
        path = dataset / row["path"]
        if sha(path) != row["sha256"]:
            raise ValueError("model SHA256 changed")
        artifact = load_v2_policy(path, device=device, expected_algorithm="ippo")
        if artifact.iteration != row["iteration"] or artifact.metadata["source_hash"] != plan["source"]["core_source_hash"]:
            raise ValueError("wrong checkpoint iteration/source")
        provenance = artifact.training_data_provenance or {}
        if not provenance.get("complete"):
            raise ValueError("incomplete training provenance")
        training_hashes.update(provenance.get("deal_hashes", []))
        initial_seeds.update(provenance.get("initial_env_seeds", []))
        if provenance.get("initial_env_seed") is not None:
            initial_seeds.add(provenance["initial_env_seed"])
        artifact.model.eval()
        for parameter in artifact.model.parameters():
            parameter.requires_grad_(False)
        hashes[name] = state_hash(artifact.model.state_dict())
        if hashes[name] != row["state_hash"]:
            raise ValueError("loaded model state differs from the packaged checkpoint")
        models[name] = artifact
    parent = read_json(dataset / "parent_provenance.json")
    training_hashes.update(parent["training_deal_hashes"])
    initial_seeds.update(parent["initial_env_seeds"])
    path = output / f"games.gpu{rank}.jsonl"
    if path.exists():
        raise FileExistsError("never append a new worker identity to stale output")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {"schema": "guandan-recheck375-worker-v1", "status": "running", "plan_hash": plan["plan_hash"],
              "job_hash": job["job_hash"], "gpu_index": rank, "device": str(device), "phase": job["phase"],
              "expected_games": len(job["tasks"]), "games": 0, "games_attempted": 0, "training_updates": 0,
              "source_hash": plan["source"]["core_source_hash"], "backend_hash": backend["hash"]}
    atomic_json(output / f"worker{rank}.json", status)
    try:
        with torch.inference_mode():
            for index, task in enumerate(job["tasks"], 1):
                if task["deal"]["seed"] in initial_seeds:
                    raise ValueError("evaluation seed overlaps training initialization")
                candidate, opponent = ARMS[task["arm"]]
                status["games_attempted"] = index
                row = play_game(deal=task["deal"], rotation=task["rotation"], opponent_id=task["arm"],
                                opponent_kind="rule" if opponent is None else "v2", candidate=models[candidate],
                                opponent_v2=None if opponent is None else models[opponent],
                                temperature=0.0, device=device, max_steps=1024)
                row.update({"task_id": task["task_id"], "task_hash": canonical(task), "arm": task["arm"], "gpu_index": rank,
                            "evaluation_protocol": plan["protocol"], "evaluation_plan_hash": plan["plan_hash"],
                            "evaluation_manifest_hash": manifest["hash"], "evaluation_horizon": 1024,
                            "record_origin": "validation_replay" if job["phase"] in ("cache_replay", "local_replay") else "new_gpu_execution"})
                validate_row(row, task, plan)
                if row["base_deal_sha256"] in training_hashes:
                    raise ValueError("physical evaluation deal overlaps actual training")
                if row["base_deal_sha256"] != job["physical_deal_hashes"][task["deal"]["deal_id"]]:
                    raise ValueError("same evaluation deal differs from prior run")
                if job["phase"] in ("cache_replay", "local_replay"):
                    compare_replay(row, job["replay_records"][task["task_id"]])
                append(path, row)
                status["games"] = index
                if index % 4 == 0 or index == len(job["tasks"]):
                    print(f"[RECHECK375 {job['phase']} gpu{rank}] {index}/{len(job['tasks'])} games, {time.monotonic()-started:.1f}s, training_updates=0", flush=True)
        for name, model in models.items():
            if state_hash(model.model.state_dict()) != hashes[name]:
                raise ValueError("evaluation modified model weights")
        status.update({"status": "complete", "frozen_model_hashes_unchanged": True})
    except BaseException as error:
        status.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        try:
            status["frozen_model_hashes_unchanged"] = all(state_hash(model.model.state_dict()) == hashes[name] for name, model in models.items())
        except Exception as hash_error:
            status["freeze_attestation_error"] = repr(hash_error)
        status["elapsed_seconds"] = time.monotonic() - started
        status["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else 0
        status["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device)) if str(device).startswith("cuda") else 0
        atomic_json(output / f"worker{rank}.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    run_job(args.dataset, args.job, args.output, testing=args.local_test, device=args.device)


if __name__ == "__main__":
    main()
