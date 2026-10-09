"""Verify inherited M05 corpus identity and replay a 13-level repair sample."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guandan_marvel.testing.differential import replay_corpus_record
from scripts.verify_marvel_m05_differential import run_directed_coverage, run_privacy_reorders

ORIGINAL_COMMIT = "4ba65bfa20df2843fbc3cefb5299e885c14e2479"
SEMANTIC_FILES = (
    "guandan/cards.py", "guandan/combos.py", "guandan/round.py",
    "guandan/state.py", "docs/rules.md", "guandan_marvel/game.py",
    "nanopokerlib/cpp/guandan/guandan.cpp",
    "nanopokerlib/cpp/guandan/guandan.hpp",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def git_blob_sha256(commit: str, path: str) -> str:
    data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT)
    return normalized_text_sha256(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    original_evidence = REPO_ROOT / "docs/marvel_migration/evidence/M05_differential_performance"
    original_summary = json.loads((original_evidence / "differential_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((original_evidence / "corpus_manifest.json").read_text(encoding="utf-8"))

    shard_results = []
    records = []
    total_records = 0
    total_actions = 0
    for shard in manifest["shards"]:
        path = REPO_ROOT / shard["path"]
        actual_hash = sha256(path)
        actual_bytes = path.stat().st_size
        valid = actual_hash == shard["sha256"] and actual_bytes == shard["bytes"]
        shard_results.append({
            "path": shard["path"], "expected_sha256": shard["sha256"],
            "actual_sha256": actual_hash, "expected_bytes": shard["bytes"],
            "actual_bytes": actual_bytes, "valid": valid,
        })
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                total_records += 1
                total_actions += len(record["steps"])
                if len({item["level_rank"] for item in records}) < 13:
                    known = {item["level_rank"] for item in records}
                    if record["level_rank"] not in known:
                        records.append(record)
    records.sort(key=lambda item: item["level_rank"])
    replayed = [replay_corpus_record(record) | {"level_rank": record["level_rank"]} for record in records]
    directed = run_directed_coverage()
    privacy = run_privacy_reorders()

    semantic_files = []
    for relative in SEMANTIC_FILES:
        current = normalized_text_sha256((REPO_ROOT / relative).read_bytes())
        original = git_blob_sha256(ORIGINAL_COMMIT, relative)
        semantic_files.append({
            "path": relative, "original_sha256": original,
            "current_sha256": current, "unchanged": original == current,
        })
    counts_valid = (
        total_records == manifest["games"] == original_summary["counts"]["completed_games"]
        and total_actions == manifest["actions"] == original_summary["counts"]["actions_executed"]
    )
    result = {
        "schema": "guandan-marvel-m05-repair-corpus-verification-v1",
        "stage": "M05_differential_performance_repair",
        "original_implementation_commit": ORIGINAL_COMMIT,
        "inherited_evidence": {
            "summary": "docs/marvel_migration/evidence/M05_differential_performance/differential_summary.json",
            "manifest": "docs/marvel_migration/evidence/M05_differential_performance/corpus_manifest.json",
            "complete_games": original_summary["counts"]["completed_games"],
            "actions": original_summary["counts"]["actions_executed"],
            "states_compared": original_summary["counts"]["states_compared"],
            "intermediate_state_subset": original_summary["counts"]["intermediate_states_sampled"],
            "serialization_roundtrips": original_summary["counts"]["serialization_roundtrips"],
            "mismatches": original_summary["counts"]["mismatches"],
        },
        "shards": shard_results,
        "records_counted": total_records,
        "actions_counted": total_actions,
        "counts_valid": counts_valid,
        "replayed_13_levels": replayed,
        "all_13_levels_replayed": len(replayed) == 13,
        "directed_complex_actions": directed,
        "privacy_reorders": privacy,
        "semantic_files": semantic_files,
        "semantic_execution_files_unchanged": all(item["unchanged"] for item in semantic_files),
    }
    result["valid"] = bool(
        all(item["valid"] for item in shard_results)
        and counts_valid and len(replayed) == 13
        and directed["valid"] and privacy["valid"]
        and result["semantic_execution_files_unchanged"]
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": result["valid"], "records": total_records,
        "actions": total_actions, "levels_replayed": len(replayed),
        "output": str(output),
    }, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
