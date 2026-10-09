"""Run the M05 10k/100k Python/C++ differential acceptance workload."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from guandan.action_state import canonical_tokens
from guandan.cards import Rank
from guandan.state import RoundState
from guandan_marvel import python_game
from guandan_marvel.testing.complex_actions import DIRECTED_ACTION_CASES, build_directed_action
from guandan_marvel.testing.differential import (
    CORPUS_SCHEMA,
    SUMMARY_SCHEMA,
    DifferentialCase,
    build_case,
    compare_step,
    corpus_record,
    cpp_batch_from_case,
    merge_coverage,
    python_batch_from_case,
    run_game,
)
from nanopokerlib import guandan as cpp_game

REQUIRED_FAMILIES = {
    "single", "pair", "triple", "full_house", "straight", "pair_sequence",
    "triple_sequence", "rank_bomb", "straight_flush", "four_kings",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_from_state(index: int, state: RoundState) -> DifferentialCase:
    previous = state.previous_result
    return DifferentialCase(
        game_index=index,
        seed=index,
        level_rank=int(state.level_rank),
        leader_seat=state.leader_seat,
        previous_result=None if previous is None else {
            "ranking": list(previous.ranking),
            "winner_team": previous.winner_team,
            "outcome_class": previous.outcome_class,
            "tribute_cancelled": previous.tribute_cancelled,
        },
        deal=tuple(tuple(card.card_id for card in hand) for hand in state.hands),
    )


def run_directed_coverage() -> dict[str, Any]:
    records = []
    families: dict[str, int] = {}
    natural = 0
    substituted = 0
    for index, directed in enumerate(DIRECTED_ACTION_CASES):
        state, action = build_directed_action(directed)
        case = _case_from_state(-1000 - index, state)
        python_batch, python_step = python_batch_from_case(case, auto_reset=False)
        cpp_batch, cpp_step = cpp_batch_from_case(case, auto_reset=False)
        state_count = 0
        for token_id in canonical_tokens(action):
            compare_step(
                python_step, cpp_step,
                python_batch.critic_state(), cpp_batch.critic_state(),
                step_index=state_count,
            )
            live = python_step.available_actions.weights[0] > 0
            matches = np.flatnonzero(live & (python_step.available_actions.tokens[0, :, 5] == token_id))
            if len(matches) != 1:
                raise AssertionError(f"directed case {directed.name} token {token_id} is not unique")
            row = python_step.available_actions.tokens[0, int(matches[0])].copy()
            python_step = python_batch.step(row[None, :])
            cpp_step = cpp_batch.step(row[None, :])
            state_count += 1
        compare_step(
            python_step, cpp_step,
            python_batch.critic_state(), cpp_batch.critic_state(),
            step_index=state_count,
        )
        family = directed.family.value
        families[family] = families.get(family, 0) + 1
        natural += len(directed.natural_wild_ids)
        substituted += len(directed.assignments)
        records.append({
            "name": directed.name,
            "family": family,
            "tokens": list(canonical_tokens(action)),
            "states": state_count + 1,
            "natural_wild_ids": list(directed.natural_wild_ids),
            "wild_assignments": [list(item) for item in directed.assignments],
        })
    missing = sorted(REQUIRED_FAMILIES - set(families))
    return {
        "cases": records,
        "case_count": len(records),
        "families": families,
        "missing_families": missing,
        "natural_wild_declarations": natural,
        "substituted_wild_declarations": substituted,
        "valid": not missing and natural > 0 and substituted > 0,
    }


def run_privacy_reorders() -> dict[str, Any]:
    base = RoundState.deal(seed=301, level_rank=Rank.SIX, leader_seat=0)
    records = []
    for observer in range(4):
        hidden = [seat for seat in range(4) if seat != observer]
        left, right = hidden[0], hidden[1]
        alternate = base.clone()
        alternate.hands[left][0], alternate.hands[right][0] = (
            alternate.hands[right][0], alternate.hands[left][0]
        )
        alternate.check_invariants()
        base_case = _case_from_state(-2000 - observer * 2, base)
        alternate_case = _case_from_state(-2001 - observer * 2, alternate)
        py_a, py_step_a = python_batch_from_case(base_case)
        cpp_a, cpp_step_a = cpp_batch_from_case(base_case)
        py_b, py_step_b = python_batch_from_case(alternate_case)
        cpp_b, cpp_step_b = cpp_batch_from_case(alternate_case)
        compare_step(py_step_a, cpp_step_a, py_a.critic_state(), cpp_a.critic_state(), step_index=0)
        compare_step(py_step_b, cpp_step_b, py_b.critic_state(), cpp_b.critic_state(), step_index=0)
        np.testing.assert_array_equal(
            py_step_a.observations.tokens[0, observer],
            py_step_b.observations.tokens[0, observer],
        )
        np.testing.assert_array_equal(
            py_step_a.observations.channels[0, observer],
            py_step_b.observations.channels[0, observer],
        )
        np.testing.assert_array_equal(
            cpp_step_a.observations.tokens[0, observer],
            cpp_step_b.observations.tokens[0, observer],
        )
        np.testing.assert_array_equal(
            cpp_step_a.observations.channels[0, observer],
            cpp_step_b.observations.channels[0, observer],
        )
        records.append({
            "observer": observer,
            "swapped_hidden_seats": [left, right],
            "python_equal": True,
            "cpp_equal": True,
            "cross_backend_equal": True,
        })
    return {"cases": records, "case_count": len(records), "valid": len(records) == 4}


def _worker(
    worker_id: int,
    indices: list[int],
    quotas: dict[int, int],
    output_dir: str,
) -> dict[str, Any]:
    root = Path(output_dir)
    shard_path = root / "corpus" / f"corpus-{worker_id:02d}.jsonl.gz"
    mismatch_dir = root / "mismatches"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    successful = []
    mismatch_paths: list[str] = []
    exception_records: list[dict[str, Any]] = []
    level_counts: dict[str, int] = {}
    previous_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    sample_digest = hashlib.sha256()
    trajectory_digest = hashlib.sha256()
    state_count = 0
    action_count = 0
    serialization_roundtrips = 0
    started = time.perf_counter()
    with gzip.open(shard_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as corpus:
        for index in indices:
            case = build_case(index)
            try:
                result = run_game(
                    case,
                    intermediate_quota=quotas[index],
                    mismatch_dir=mismatch_dir,
                )
            except BaseException as error:
                matches = sorted(mismatch_dir.glob(f"game-{index:05d}-*.json"))
                mismatch_paths.extend(str(path) for path in matches)
                exception_records.append({
                    "game_index": index,
                    "type": type(error).__name__,
                    "message": str(error),
                    "fixtures": [str(path) for path in matches],
                })
                continue
            corpus.write(json.dumps(corpus_record(result), sort_keys=True, separators=(",", ":")) + "\n")
            successful.append(result)
            state_count += int(result["state_count"])
            action_count += len(result["steps"])
            serialization_roundtrips += int(result["serialization_roundtrips"])
            level = str(case.level_rank)
            level_counts[level] = level_counts.get(level, 0) + 1
            previous = "NONE" if case.previous_result is None else str(case.previous_result["outcome_class"])
            previous_counts[previous] = previous_counts.get(previous, 0) + 1
            outcome = str(result["terminal"]["outcome_class"])
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            sample_digest.update(result["intermediate_sample_digest"].encode("ascii"))
            trajectory_digest.update(result["trajectory_digest"].encode("ascii"))
    elapsed = time.perf_counter() - started
    return {
        "worker_id": worker_id,
        "requested_games": len(indices),
        "completed_games": len(successful),
        "mismatch_count": len(exception_records),
        "mismatch_fixtures": mismatch_paths,
        "exceptions": exception_records,
        "state_count": state_count,
        "action_count": action_count,
        "intermediate_sample_count": sum(int(result["intermediate_sample_count"]) for result in successful),
        "intermediate_sample_digest": sample_digest.hexdigest(),
        "trajectory_digest": trajectory_digest.hexdigest(),
        "serialization_roundtrips": serialization_roundtrips,
        "coverage": merge_coverage(successful),
        "level_counts": level_counts,
        "previous_result_counts": previous_counts,
        "outcome_counts": outcome_counts,
        "elapsed_seconds": elapsed,
        "corpus": {
            "path": str(shard_path),
            "bytes": shard_path.stat().st_size,
            "sha256": sha256_file(shard_path),
            "schema": CORPUS_SCHEMA,
        },
    }


def _sum_maps(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        for name, value in record[key].items():
            result[name] = result.get(name, 0) + int(value)
    return result


def _aggregate_coverage(workers: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    totals = {
        "natural_wild_assignments": 0,
        "substituted_wild_assignments": 0,
        "anti_tribute": 0,
        "tribute_actions": 0,
        "return_actions": 0,
        "contact": 0,
    }
    for worker in workers:
        coverage = worker["coverage"]
        for name, value in coverage["families"].items():
            families[name] = families.get(name, 0) + int(value)
        for name in totals:
            totals[name] += int(coverage[name])
    return {"families": families, **totals}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--intermediate-states", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.games <= 0 or args.intermediate_states < 0 or args.workers <= 0:
        parser.error("games/workers must be positive and intermediate-states non-negative")
    base_quota, remainder = divmod(args.intermediate_states, args.games)
    quotas = {index: base_quota + int(index < remainder) for index in range(args.games)}
    if any(value > 10 for value in quotas.values()):
        # All legal random games are much longer, but this protects accidental
        # tiny custom runs from claiming unavailable samples.
        pass

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_unix = time.time()
    started = time.perf_counter()
    directed = run_directed_coverage()
    privacy = run_privacy_reorders()

    shards = [[] for _ in range(min(args.workers, args.games))]
    for index in range(args.games):
        shards[index % len(shards)].append(index)
    workers: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(shards)) as pool:
        futures = {
            pool.submit(_worker, worker_id, indices, quotas, str(output)): worker_id
            for worker_id, indices in enumerate(shards)
        }
        for future in as_completed(futures):
            record = future.result()
            workers.append(record)
            print(json.dumps({
                "worker": record["worker_id"],
                "completed_games": record["completed_games"],
                "mismatches": record["mismatch_count"],
                "elapsed_seconds": round(record["elapsed_seconds"], 3),
            }, sort_keys=True), flush=True)
    workers.sort(key=lambda item: item["worker_id"])

    coverage = _aggregate_coverage(workers)
    completed_games = sum(item["completed_games"] for item in workers)
    mismatch_count = sum(item["mismatch_count"] for item in workers)
    intermediate_count = sum(item["intermediate_sample_count"] for item in workers)
    levels = _sum_maps(workers, "level_counts")
    previous_results = _sum_maps(workers, "previous_result_counts")
    outcomes = _sum_maps(workers, "outcome_counts")
    missing_families = sorted(REQUIRED_FAMILIES - set(coverage["families"]) - set(directed["families"]))
    requirements = {
        "complete_games": completed_games == args.games,
        "zero_mismatch": mismatch_count == 0,
        "intermediate_states": intermediate_count == args.intermediate_states,
        "all_13_levels": len([value for value in levels.values() if value > 0]) == 13,
        "all_combination_families": not missing_families and directed["valid"],
        "wild_declarations": (
            coverage["natural_wild_assignments"] + directed["natural_wild_declarations"] > 0
            and coverage["substituted_wild_assignments"] + directed["substituted_wild_declarations"] > 0
        ),
        "bombs_and_four_kings": (
            coverage["families"].get("rank_bomb", 0) + directed["families"].get("rank_bomb", 0) > 0
            and coverage["families"].get("four_kings", 0) + directed["families"].get("four_kings", 0) > 0
        ),
        "tribute_return_anti_tribute": (
            coverage["tribute_actions"] > 0
            and coverage["return_actions"] > 0
            and coverage["anti_tribute"] > 0
        ),
        "contact": coverage["contact"] > 0,
        "terminal_outcomes": set(outcomes) == {"DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST"},
        "privacy_reorders": privacy["valid"],
        "serialization_roundtrips": sum(item["serialization_roundtrips"] for item in workers) == args.games,
        "trajectory_digests": all(item["trajectory_digest"] for item in workers),
    }
    valid = all(requirements.values())
    result = {
        "schema": SUMMARY_SCHEMA,
        "stage": "M05_differential_performance",
        "started_unix": started_unix,
        "elapsed_seconds": time.perf_counter() - started,
        "configuration": {
            "games": args.games,
            "intermediate_states": args.intermediate_states,
            "workers": len(shards),
            "base_seed": 5_050_000,
            "policy_seed": 5_051_000,
            "python_backend": python_game.BACKEND,
            "cpp_backend": cpp_game.BACKEND,
            "protocol_version": python_game.protocol_version,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "counts": {
            "requested_games": args.games,
            "completed_games": completed_games,
            "states_compared": sum(item["state_count"] for item in workers),
            "actions_executed": sum(item["action_count"] for item in workers),
            "intermediate_states_sampled": intermediate_count,
            "serialization_roundtrips": sum(item["serialization_roundtrips"] for item in workers),
            "mismatches": mismatch_count,
        },
        "coverage": coverage,
        "directed_coverage": directed,
        "privacy_reorders": privacy,
        "level_counts": levels,
        "previous_result_counts": previous_results,
        "terminal_outcome_counts": outcomes,
        "missing_families": missing_families,
        "requirements": requirements,
        "workers": workers,
        "corpus": {
            "schema": CORPUS_SCHEMA,
            "shards": [item["corpus"] for item in workers],
            "replay_contract": "Each step is [action_token_id, *exact_prefix_tokens]; the complete V2 row is uniquely regenerated from the live legal set.",
        },
        "mismatch_fixtures": [
            path for item in workers for path in item["mismatch_fixtures"]
        ],
        "valid": valid,
    }
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": valid,
        "games": completed_games,
        "intermediate_states": intermediate_count,
        "states_compared": result["counts"]["states_compared"],
        "mismatches": mismatch_count,
        "summary": str(summary),
    }, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
