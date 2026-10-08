"""Generate M02 repair evidence for actual complex action execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guandan.combos import CombinationKind
from guandan_marvel.testing.complex_actions import DIRECTED_ACTION_CASES, execute_directed_case


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = [execute_directed_case(case) for case in DIRECTED_ACTION_CASES]
    report = {
        "schema": "guandan-marvel-m02-complex-action-coverage-v1",
        "stage": "M02_guandan_adapter_v2_repair",
        "source_head_before_repair_commit": git("rev-parse", "HEAD"),
        "inventory_from_enum": [kind.value for kind in CombinationKind],
        "candidate_enumeration_coverage": sorted({case["family"] for case in cases}),
        "actual_complete_action_execution_coverage": sorted({case["family"] for case in cases if case["commit_executed"]}),
        "wild_modes": sorted({case["wild_mode"] for case in cases}),
        "max_prefix_tokens": max(case["prefix_tokens"] for case in cases),
        "all_commit_executed": all(case["commit_executed"] for case in cases),
        "all_serialization_resume": all(case["serialization_resume"] for case in cases),
        "all_v1_transition_equal": all(case["v1_transition_equal"] for case in cases),
        "all_card_conservation": all(case["card_conservation"] for case in cases),
        "real_terminal_cases": sum(case["real_terminal"] for case in cases),
        "note": "Directed cases retain one extra actor card, so they verify full complex action commit/public event semantics but intentionally do not claim terminal games. Real terminals remain covered by the separate 15-game rerun.",
        "cases": cases,
    }
    if set(report["actual_complete_action_execution_coverage"]) != {kind.value for kind in CombinationKind}:
        raise AssertionError("not every CombinationKind was actually committed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "actual_complete_action_execution_coverage", "wild_modes", "max_prefix_tokens",
        "all_serialization_resume", "all_v1_transition_equal",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
