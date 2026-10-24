"""One entry point for the next validated Kaggle delivery. No default old package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.kaggle_pipeline import core


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=core.DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor", help="Offline dependency/account check; no API calls")
    sub.add_parser("check", help="Authenticate and read account Dataset/Notebook lists; no uploads or runs")
    publish = sub.add_parser("publish", help="Validate and publish the current delivery pointer; local only")
    publish.add_argument("--delivery", type=Path, required=True)
    publish.add_argument("--qa", type=Path, required=True)
    sub.add_parser("current", help="Resolve and verify the current delivery pointer; local only")
    prune = sub.add_parser("prune", help="Plan local old-delivery cleanup; deletion requires --execute")
    prune.add_argument("--keep-previous", type=int)
    prune.add_argument("--execute", action="store_true")
    for action in ("prepare", "run"):
        command = sub.add_parser(action)
        command.add_argument("--delivery", type=Path, help="Explicit new serial-compare delivery directory")
        command.add_argument("--build", action="store_true", help="Build from config.build, then validate")
        command.add_argument("--current", action="store_true", help="Use the QA-published current delivery")
        command.add_argument("--dry-run", action="store_true", help="Offline; never uploads or starts a job")
        if action == "run":
            command.add_argument("--run-id")
            command.add_argument("--no-wait", action="store_true")
    for action in ("status", "fetch", "watch"):
        sub.add_parser(action).add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        config = core.load_config(args.config)
        if args.action == "doctor":
            result = core.doctor(config)
            emit(result)
            return 0 if result["status"] == "READY" else 2
        if args.action == "check":
            emit(core.check_connection(config))
            return 0
        if args.action == "publish":
            emit(core.publish_delivery(config, args.delivery, args.qa))
            return 0
        if args.action == "current":
            delivery, record = core.current_delivery(config)
            emit({**record, "resolved_delivery": str(delivery), "remote_calls": False})
            return 0
        if args.action == "prune":
            emit(core.prune_deliveries(config, keep_previous=args.keep_previous, execute=args.execute))
            return 0
        if args.action in ("prepare", "run"):
            if sum(bool(value) for value in (args.build, args.delivery, args.current)) > 1:
                raise core.PipelineError("choose exactly one of --build, --delivery, or --current")
            if args.action == "run" and args.run_id and (args.build or args.delivery or args.current):
                raise core.PipelineError("--run-id cannot be combined with new delivery/build/current")
            if args.dry_run:
                delivery = core.current_delivery(config)[0] if args.current else args.delivery
                emit(core.prepare(config, delivery, dry_run=True))
                return 0
            delivery = core.current_delivery(config)[0] if args.current else (
                core.build_delivery(config) if args.build else args.delivery
            )
            prepared = core.prepare(config, delivery) if delivery is not None or args.action == "prepare" else None
            if args.action == "prepare":
                emit(prepared)
                return 0
            run_id = prepared["run_id"] if prepared else args.run_id
            result = core.submit(config, run_id, emit=emit)
            emit(result)
            if not args.no_wait:
                result = core.watch(config, result["run_id"], emit=emit)
                emit(result)
            return 3 if result["status"] == "FETCHED_INCOMPLETE" else 0
        if args.action == "status":
            result = core.local_status(config, args.run_id)
        else:
            result = getattr(core, args.action)(config, args.run_id, **({"emit": emit} if args.action == "watch" else {}))
        emit(result)
        return 3 if result["status"] == "FETCHED_INCOMPLETE" else 0
    except KeyboardInterrupt:
        emit({"status": "CLIENT_STOPPED", "message": "Remote jobs continue. Resume with status/watch; do not create another run."})
        return 130
    except Exception as error:
        # SDK raw errors and malformed credential files must never be echoed.
        if isinstance(error, core.PipelineError):
            message = str(error)
        else:
            from tools.kaggle_pipeline.backend import BackendError
            from tools.kaggle_pipeline.delivery_registry import RegistryError
            message = str(error) if isinstance(error, (BackendError, RegistryError)) else "Local operation failed (" + type(error).__name__ + "); no raw response was printed."
        emit({"status": "ERROR", "message": message})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
