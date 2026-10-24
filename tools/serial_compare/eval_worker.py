# SPDX-License-Identifier: MIT
"""Per-shard evaluation worker.

Each worker loads the frozen artifacts once and plays only the task ids it was
given.  Records are appended per game (resume-friendly) and validated before
they are written, so a corrupt record can never enter the statistics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--finals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    import sys
    sys.path.insert(0, str(Path(args.source).resolve()))
    from serial_compare.evaluation_runner import ArtifactCache, model_paths_for_run, run_tasks

    plan = _read(args.plan)
    tasks = _read(args.tasks)
    finals = _read(args.finals) if Path(args.finals).is_file() else None
    paths = model_paths_for_run(plan, args.dataset, finals)
    # The formal evaluation compares the *frozen final* policies, so their
    # hashes extend the plan's p250/p375/p500 identity map.
    models = dict(plan["models"])
    if finals:
        import hashlib

        for name in ("control_final", "candidate_final"):
            if finals.get(name):
                digest = hashlib.sha256(Path(str(finals[name])).read_bytes()).hexdigest()
                models[name] = {"sha256": digest}
    cache = ArtifactCache(paths, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as stream:
        def persist(record) -> None:
            # Flush *and* fsync per game: a platform kill keeps games 1..N-1.
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        run_tasks(
            tasks, cache=cache, models=models, device=args.device,
            max_steps=int(plan["evaluation"]["horizon"]), on_record=persist,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - target-only entry point
    raise SystemExit(main())
