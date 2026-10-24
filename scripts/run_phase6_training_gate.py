# SPDX-License-Identifier: MIT
"""Phase-6 training-gate tool entry point (dry-run by default).

    D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase6_training_gate.py --gate 1 --completed-iterations 0 --max-iterations 2

This phase implements the tool but does not execute a formal gate.  ``--execute``
requires the explicit confirmation token *and* an externally supplied trainer
factory; without both it refuses.  Crossing the budget writes an
iteration-boundary checkpoint and a recovery record.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guandan_marvel.phase6.design import (  # noqa: E402
    FORMAL_CONFIRMATION_TOKEN, FORMAL_GATE_TARGETS,
)
from guandan_marvel.phase6.training_gates import (  # noqa: E402
    GateBudget, GateRunner, plan_gate_run,
)


def _load_factory(spec: str):
    if ":" not in spec:
        raise SystemExit("--factory must be module:callable")
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise SystemExit(f"trainer factory {spec!r} is not callable")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=int, required=True, choices=sorted(FORMAL_GATE_TARGETS))
    parser.add_argument("--completed-iterations", type=int, default=0)
    parser.add_argument("--completed-games", type=int, default=0)
    parser.add_argument("--target-games", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-formal", default=None)
    parser.add_argument("--factory", default=None)
    parser.add_argument("--checkpoint", default="runs/phase6/training_gate/checkpoint.pt")
    parser.add_argument("--output-dir", default="runs/phase6/training_gate")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    dry_run = not args.execute
    budget = GateBudget(
        gate=args.gate,
        completed_iterations=args.completed_iterations,
        completed_games=args.completed_games,
        target_games=args.target_games,
        max_iterations=args.max_iterations,
        max_games=args.max_games,
        max_wall_seconds=args.max_wall_seconds,
        dry_run=dry_run,
        formal_confirmation=args.confirm_formal,
        smoke=args.smoke,
    )
    if dry_run:
        plan = plan_gate_run(budget)
        plan["formal_gates_not_run"] = True
        plan["confirmation_token_required"] = FORMAL_CONFIRMATION_TOKEN
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0
    if args.confirm_formal != FORMAL_CONFIRMATION_TOKEN:
        raise SystemExit(
            "refusing a formal gate run without --confirm-formal "
            f"{FORMAL_CONFIRMATION_TOKEN}"
        )
    if not args.factory:
        raise SystemExit(
            "refusing to guess a trainer: pass --factory module:callable; this phase implements "
            "the tool but does not execute a formal gate automatically"
        )
    runner = GateRunner(
        trainer_factory=_load_factory(args.factory),
        budget=budget,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    record = runner.run()
    print(json.dumps(record, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
