"""Run the one-shot M09 final holdout for the frozen iter2000 candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import zipfile

from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
from guandan_marvel.evaluation.common import atomic_write_json, payload_hash
from guandan_marvel.evaluation.evaluator import EvaluationRunner, FIXED_MATRIX
from guandan_marvel.evaluation.manifests import FinalHoldoutAudit, deals_for, validate_deals_manifest
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore, PromotionPolicy


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: str | Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(base: Path, spec: dict[str, object], label: str) -> Path:
    relative = Path(str(spec.get("path", "")))
    path = (base / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"unsafe/missing final control file: {label}")
    if path.stat().st_size != int(spec.get("bytes", -1)) or _sha256(path) != spec.get("sha256"):
        raise ValueError(f"final control identity mismatch: {label}")
    return path


def _gpu_snapshot() -> list[dict[str, object]]:
    command = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        rows = []
        for line in result.stdout.splitlines():
            values = [x.strip() for x in line.split(",")]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args(argv)
    control_path = args.control.resolve(); base = control_path.parent.resolve(); working = args.working_root.resolve()
    control = _load(control_path)
    if control.get("schema") != "guandan-marvel-m09-final-holdout-control-v1" or control.get("hash") != payload_hash(control):
        raise ValueError("final holdout control schema/hash mismatch")
    candidate_path = _safe_file(base, control["candidate_checkpoint"], "candidate")
    champion_seed = _safe_file(base, control["champion_checkpoint"], "champion")
    legacy_path = _safe_file(base, control["legacy_checkpoint"], "legacy")
    deals_path = _safe_file(base, control["deals_manifest"], "deals")
    deals = _load(deals_path); validate_deals_manifest(deals)
    if len(deals_for(deals, "final")) != int(control["final_clusters"]):
        raise ValueError("final holdout deal count differs from control")

    output = working / "guandan-marvel-m09-final-holdout"
    if output.exists():
        raise FileExistsError(f"refusing to reuse existing final holdout output: {output}")
    output.mkdir(parents=True)
    candidate_target = output / "candidate/ippo_iter002000.pt"; candidate_target.parent.mkdir(parents=True); shutil.copy2(candidate_path, candidate_target)
    champion_target = output / "champion/champions/generation000001_iter000100.pt"; champion_target.parent.mkdir(parents=True); shutil.copy2(champion_seed, champion_target)
    legacy_target = output / "legacy/legacy_v1_anchor.pt"; legacy_target.parent.mkdir(parents=True); shutil.copy2(legacy_path, legacy_target)
    shutil.copy2(deals_path, output / "final_deals_manifest.json")

    candidate = load_v2_policy(candidate_target, device="cuda:0", expected_algorithm="ippo")
    champion_artifact = load_v2_policy(champion_target, device="cuda:0", expected_algorithm="ippo")
    legacy = load_legacy_v1_opponent(legacy_target)
    if candidate.sha256 != control["candidate_checkpoint"]["sha256"] or candidate.iteration != 2000:
        raise ValueError("frozen final candidate identity mismatch")

    champion_store = ChampionStore(output / "champion/champion_manifest.json")
    champion = champion_store.bootstrap(champion_artifact.identity, reason="frozen pre-final iter100 champion baseline")
    pool = OpponentPoolStore(output / "champion/opponent_pool_manifest.json")
    pool.initialize(sampling_seed=20261015)
    pool.add_entry({"id": "champion-generation-1", "kind": "champion", "checkpoint": str(champion_target), "checkpoint_sha256": champion_artifact.sha256, "policy_identity": champion_artifact.identity, "weight": 1.0, "source_champion_manifest_hash": champion["hash"]})
    pool.add_entry({"id": "legacy-v1-anchor", "kind": "legacy_v1", "checkpoint": str(legacy_target), "checkpoint_sha256": legacy.sha256, "legacy_identity": legacy.identity, "weight": 0.25})
    pool.add_entry({"id": "rule-anchor-v1", "kind": "rule", "version": "v2-public-row-heuristic-v1", "weight": 0.25})

    audit = FinalHoldoutAudit(output / "final_holdout_audit.json", deals_manifest=deals)
    audit.initialize(); audit.lock_candidate(candidate.sha256)
    runner = EvaluationRunner(output_dir=output / "final_run", champion_store=champion_store, opponent_pool=pool, legacy_checkpoint=legacy_target, device="cuda:0", temperature=0.0, promotion_policy=PromotionPolicy(mode="formal"), provenance_policy="formal")
    total = len(FIXED_MATRIX) * len(deals_for(deals, "final")) * 4
    stop = threading.Event(); started = time.monotonic()
    def heartbeat():
        while not stop.wait(float(control["heartbeat_seconds"])):
            completed = _count_jsonl(output / "final_run/games.jsonl")
            elapsed = time.monotonic() - started
            print(json.dumps({"event": "final_holdout_heartbeat", "completed": completed, "total": total, "percent": 100.0 * completed / total, "elapsed_seconds": elapsed, "eta_seconds": None if completed == 0 else elapsed * (total - completed) / completed, "gpu": _gpu_snapshot()}, sort_keys=True), flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    evaluation_id = str(control["evaluation_id"])
    try:
        result = runner.run_final_audit(deals_manifest=deals, candidate_checkpoint=candidate_target, final_audit=audit, evaluation_id=evaluation_id)
    finally:
        stop.set(); thread.join(timeout=5)
    if result.get("status") != "complete" or result.get("games_recorded") != total:
        raise RuntimeError(f"final holdout did not complete exactly: {result}")
    summary = _load(output / "final_run/cluster_summary.json")
    final_report = {"schema": "guandan-marvel-m09-final-holdout-report-v1", "status": "complete", "evaluation_id": evaluation_id, "candidate": candidate.identity, "champion": champion_artifact.identity, "deal_manifest_hash": deals["hash"], "final_dataset_hash": audit.dataset_hash, "evaluation_result": result, "summary": summary, "audit": audit.load(), "final_holdout_consumed": True, "m09_accepted": False, "finished_at": _timestamp()}
    final_report["hash"] = payload_hash(final_report)
    atomic_write_json(output / "final_holdout_report.json", final_report)
    archive_path = working / "guandan-marvel-m09-final-holdout.zip"; temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file(): archive.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if bad := archive.testzip(): raise ValueError(f"final output CRC failure: {bad}")
    os.replace(temporary, archive_path)
    print(json.dumps({"status": "complete", "archive": str(archive_path), "archive_sha256": _sha256(archive_path), "games": total, "evaluation_hash": result["evaluation_hash"], "report_hash": final_report["hash"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
