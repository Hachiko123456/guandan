# SPDX-License-Identifier: MIT
"""Offline QA of the *actual delivered* DMC Gate L payload (stage 5, 前置任务 A1).

Decodes the real ``.bin`` into an isolated directory and verifies the payload /
manifest / per-file hashes, the plan (schema, hash, **budget fields**, the
**pre-declared gate table** and boundaries), the pinned Gate L training config,
the bound Notebook constants, and byte-identity of the shipped overlay / driver
/ config against the current workspace.

The gate thresholds and budgets below are an independent restatement of
``docs/dmc_prompts/05_gate_l_local.md`` (they are deliberately NOT imported
from the builder, so a wrong constant cannot pass both).  The budget check
implements the platform convention of ``tools/kaggle_pipeline/core.py``
(``prepare`` rejects a plan whose ``total_seconds + 300`` exceeds the platform
timeout): ``budgets.total_seconds <= timeout - 300``.

This verifier never imports the workspace ``guandan_marvel`` package, never
touches the Kaggle SDK, and performs no network or subprocess calls.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PAYLOAD = "guandan_serial_compare_payload_v1.bin"
NOTEBOOK = "guandan_serial_compare_v1.ipynb"
MAGIC = b"GUANDAN-DMC-GATEL-V1\n"
MAX_PAYLOAD_BYTES = 256 * 1024**2
PLAN_SCHEMA = "guandan-dmc-gate-l-plan-v1"
MANIFEST_SCHEMA = "guandan-dmc-gate-l-delivery-manifest-v1"
CONFIG_PATH = "configs/dmc_gate_l.json"
PIPELINE_CONFIG_PATH = "configs/kaggle_pipeline.json"
#: core.py: platform timeout must cover the plan budget plus 300 seconds.
BUDGET_MARGIN_SECONDS = 300

EXPECTED_BUDGETS = {
    "total_seconds": 14400,
    "train_seconds": 11400,
    "eval_reserve_seconds": 2400,
    "platform_timeout_seconds": 15000,
}
EXPECTED_SCHEDULE = {
    "checkpoint_interval_games": 512,
    "mini_eval": {"fraction_of_train_budget": 0.5, "vs_rule_games": 96},
    "final_eval": {"vs_rule_games": 256, "vs_random_games": 256},
    "eval_seed_base": {
        "mini_vs_rule": 91_000_000,
        "final_vs_rule": 92_000_000,
        "final_vs_random": 93_000_000,
    },
}
EXPECTED_GATES = {
    "health": {"illegal": 0, "nonfinite": 0, "truncated": 0},
    "q_calibration": {"final_qz_r2_min": 0.10, "must_rise": True},
    "vs_rule": {"final_min": 0.65, "mini_min_for_continue": 0.60, "must_rise_from_mini": True},
    "vs_random": {"final_min": 0.85},
    "throughput": {"min_games_in_budget": 12000, "stretch_games": 20000, "budget_seconds": 14400},
}
EXPECTED_BOUNDARIES = {
    "fabledan": "NOT_RUN",
    "final_holdout": "NOT_TOUCHED",
    "belief": "DISABLED",
    "opponent_pool": "DISABLED",
    "formal_production_promotion": False,
}
EXPECTED_CONFIG_VALUES = {
    "schema": "guandan-marvel-dmc-train-config-v1",
    "device": "cuda",
    "num_envs": 32,
    "total_games": 20000,
    "games_per_cycle": 256,
    "gradient_replay_ratio": 1.0,
    "batch_size": 256,
    "buffer_capacity": 500000,
    "max_obs_tokens": 1024,
    "epsilon_start": 0.10,
    "epsilon_final": 0.02,
    "epsilon_decay_samples": 5000000,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "checkpoint_interval_cycles": 2,
}
REQUIRED_OVERLAY = (
    "guandan/action_state.py",
    "guandan/cards.py",
    "guandan/environment.py",
    "guandan/environment_types.py",
    "guandan/state.py",
    "guandan_marvel/backends.py",
    "guandan_marvel/config.py",
    "guandan_marvel/context_schedule.py",
    "guandan_marvel/game.py",
    "guandan_marvel/python_game.py",
    "guandan_marvel/reward.py",
    "guandan_marvel/dmc/__init__.py",
    "guandan_marvel/dmc/candidates.py",
    "guandan_marvel/dmc/learner.py",
    "guandan_marvel/dmc/q_agent.py",
    "guandan_marvel/dmc/replay.py",
    "guandan_marvel/dmc/sampler.py",
    "guandan_marvel/dmc/train.py",
    "guandan_marvel/evaluation/agents.py",
)
REQUIRED_DRIVER = (
    "dmc_gate_l/__init__.py",
    "dmc_gate_l/runner.py",
    "dmc_gate_l/evaluate.py",
)
FORBIDDEN_OVERLAY_SUFFIXES = {
    ".so", ".pyd", ".dll", ".dylib", ".o", ".obj", ".a", ".lib", ".cpp", ".cc", ".c",
    ".hpp", ".h", ".pyx", ".pxd", ".cu",
}


class QAFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QAFailure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> str:
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _platform_timeout(config_path: Path | None) -> int:
    path = Path(config_path) if config_path is not None else REPO / PIPELINE_CONFIG_PATH
    try:
        value = read_json(path).get("timeout_seconds")
    except (OSError, ValueError) as error:
        raise QAFailure(f"cannot read the pipeline timeout from {path}: {error}") from None
    if type(value) is not int or value <= 0:
        raise QAFailure(f"pipeline timeout_seconds is invalid in {path}")
    return value


def verify(
    delivery: Path,
    *,
    repo: Path = REPO,
    output: Path | None = None,
    timeout_seconds: int | None = None,
    pipeline_config: Path | None = None,
) -> dict:
    delivery = Path(delivery).resolve()
    repo = Path(repo).resolve()
    _require((delivery / PAYLOAD).is_file(), "payload file is missing")
    _require((delivery / NOTEBOOK).is_file(), "notebook file is missing")
    _require((delivery / "plan.json").is_file(), "delivery plan.json is missing")
    record: dict = {"schema": "guandan-dmc-delivery-qa-v1", "delivery": str(delivery)}
    timeout = int(timeout_seconds) if timeout_seconds is not None else _platform_timeout(pipeline_config)
    record["platform_timeout_seconds"] = timeout

    with tempfile.TemporaryDirectory(prefix="dmc-gate-l-qa-") as temporary:
        decoded = Path(temporary) / "decoded"
        raw = (delivery / PAYLOAD).read_bytes()
        _require(len(raw) <= MAX_PAYLOAD_BYTES, "payload exceeds the platform size cap")
        _require(raw.startswith(MAGIC), "wrong opaque transport version")
        _require(not zipfile.is_zipfile(io.BytesIO(raw)), "payload is not an opaque transport")
        payload_sha = sha_bytes(raw)
        try:
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw[len(MAGIC):].strip(), validate=True))) as bundle:
                members = bundle.infolist()
                _require(
                    len({item.filename.casefold() for item in members}) == len(members),
                    "duplicate payload members",
                )
                for item in members:
                    _require(
                        item.orig_filename == item.filename
                        and "/" not in item.orig_filename
                        and "\\" not in item.orig_filename,
                        "non-flat payload member",
                    )
                bundle.extractall(decoded)
        except (ValueError, OSError, zipfile.BadZipFile) as error:
            raise QAFailure(f"payload decode failed: {error}") from None
        record["payload"] = {"sha256": payload_sha, "bytes": len(raw)}

        manifest = read_json(decoded / "delivery_manifest.json")
        _require(
            manifest.get("schema") == MANIFEST_SCHEMA,
            f"unsupported delivery manifest schema {manifest.get('schema')!r}",
        )
        _require(
            manifest["manifest_hash"] == canonical({k: v for k, v in manifest.items() if k != "manifest_hash"}),
            "inner delivery manifest hash is inconsistent",
        )
        expected_members = {"delivery_manifest.json", *(row["path"] for row in manifest["files"])}
        _require(
            {path.name for path in decoded.iterdir()} == expected_members,
            "undeclared payload files",
        )
        for row in manifest["files"]:
            path = decoded / row["path"]
            _require(path.is_file(), f"declared payload file missing: {row['path']}")
            _require(
                path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                f"payload file hash mismatch: {row['path']}",
            )
        record["manifest_hash"] = manifest["manifest_hash"]

        _require(
            sha(decoded / "plan.json") == sha(delivery / "plan.json"),
            "top-level plan.json differs from the payload's plan.json",
        )

        # ---- plan: schema, hash, budgets, gates, boundaries, bindings ----
        plan = read_json(decoded / "plan.json")
        _require(plan.get("schema") == PLAN_SCHEMA, f"unsupported plan schema {plan.get('schema')!r}")
        plan_hash = plan.get("plan_hash")
        _require(
            plan_hash == canonical({k: v for k, v in plan.items() if k != "plan_hash"}),
            "plan hash is internally inconsistent",
        )
        _require(plan_hash == manifest["plan_hash"], "plan hash disagrees with the manifest")
        record["plan_hash"] = plan_hash

        budgets = plan.get("budgets")
        _require(isinstance(budgets, dict), "plan has no budgets mapping")
        for field, expected in EXPECTED_BUDGETS.items():
            value = budgets.get(field)
            _require(type(value) is int, f"budgets.{field} must be an integer")
            _require(value == expected, f"budgets.{field} drifted from the pre-declared value: {value} != {expected}")
        _require(
            budgets["total_seconds"] + BUDGET_MARGIN_SECONDS <= timeout,
            f"plan budget {budgets['total_seconds']}s exceeds the platform timeout {timeout}s "
            f"minus the {BUDGET_MARGIN_SECONDS}s margin (core.py convention)",
        )
        _require(
            budgets["train_seconds"] + budgets["eval_reserve_seconds"] <= budgets["total_seconds"],
            "train + eval reserve exceed the total plan budget",
        )
        record["budget_total_seconds"] = budgets["total_seconds"]

        schedule = plan.get("schedule")
        _require(schedule == EXPECTED_SCHEDULE, "plan schedule drifted from the pre-declared values")
        _require(plan.get("gates") == EXPECTED_GATES, "plan gates drifted from the pre-declared table (not allowed)")
        _require(plan.get("boundaries") == EXPECTED_BOUNDARIES, "plan boundaries drifted from the pre-declared values")
        _require(isinstance(plan.get("deviations"), list) and plan["deviations"], "plan carries no deviation notes")

        config_bytes = (decoded / "gate_l_config.json").read_bytes()
        config = json.loads(config_bytes)
        for field, expected in EXPECTED_CONFIG_VALUES.items():
            _require(
                config.get(field) == expected,
                f"training config {field} drifted from the pre-declared value: {config.get(field)!r} != {expected!r}",
            )
        _require(
            sha_bytes(config_bytes) == manifest["config_sha256"],
            "training config hash disagrees with the manifest",
        )
        _require(
            sha_bytes(config_bytes) == plan["config"]["sha256"],
            "training config hash disagrees with the plan binding",
        )
        _require(
            config_bytes == (repo / CONFIG_PATH).read_bytes(),
            "delivered training config differs from the workspace configs/dmc_gate_l.json",
        )
        record["config_sha256"] = sha_bytes(config_bytes)

        # ---- notebook bindings ----
        notebook = read_json(delivery / NOTEBOOK)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        _require(len(code_cells) == 1, "the notebook must contain exactly one run-all code cell")
        code = "".join(code_cells[0]["source"])
        ast.parse(code)
        _require(repr(payload_sha) in code, "the Notebook is not bound to this payload hash")
        _require(repr(manifest["manifest_hash"]) in code, "the Notebook is not bound to this manifest hash")
        _require(repr(plan_hash) in code, "the Notebook is not bound to this plan hash")
        _require("GUANDAN-DMC-GATEL-V1" in code, "the Notebook does not check the DMC transport magic")
        _require("dmc_gate_l" in code, "the Notebook does not import the packaged driver")
        _require("run_notebook" in code, "the Notebook does not call the packaged driver")
        record["notebook"] = {"sha256": sha(delivery / NOTEBOOK), "code_cell_parsed": True}

        # ---- overlay / driver workspace identity ----
        def _archive_rows(archive_name: str, key: str, required, workspace_prefix: str = "") -> int:
            rows = {row["path"]: row for row in manifest[key]}
            _require(bool(rows), f"{archive_name} is empty")
            contents: dict[str, bytes] = {}
            with zipfile.ZipFile(decoded / archive_name) as bundle:
                _require(sorted(bundle.namelist()) == sorted(rows), f"{archive_name} member list changed")
                for name in rows:
                    data = bundle.read(name)
                    contents[name] = data
                    _require(
                        len(data) == rows[name]["bytes"] and sha_bytes(data) == rows[name]["sha256"],
                        f"{archive_name} member changed: {name}",
                    )
            for required_name in required:
                _require(required_name in rows, f"the payload is missing {required_name}")
            for name, data in contents.items():
                if not str(name).endswith(".py"):
                    raise QAFailure(f"{archive_name} contains a non-Python member: {name}")
                if Path(name).suffix in FORBIDDEN_OVERLAY_SUFFIXES:
                    raise QAFailure(f"{archive_name} contains a native source: {name}")
                ast.parse(data.decode("utf-8-sig"))
                workspace = repo.joinpath(workspace_prefix, name)
                _require(workspace.is_file(), f"{archive_name} file no longer exists in the workspace: {name}")
                _require(
                    workspace.read_bytes() == data,
                    f"{archive_name} file differs from the workspace: {name}",
                )
            return len(rows)

        record["overlay_files"] = _archive_rows("overlay_source.zip", "overlay_files", REQUIRED_OVERLAY)
        # Driver members are keyed by import path (``dmc_gate_l/...``); on disk
        # they live under ``tools/``.
        record["driver_files"] = _archive_rows("driver.zip", "driver_files", REQUIRED_DRIVER, workspace_prefix="tools")

        # ---- SHA256SUMS ----
        sums_path = delivery / "SHA256SUMS"
        if sums_path.is_file():
            sums = read_json(sums_path)
            _require(
                sums["hash"] == canonical({k: v for k, v in sums.items() if k != "hash"}),
                "SHA256SUMS hash is inconsistent",
            )
            for row in sums["files"]:
                _require(
                    (delivery / row["path"]).is_file() and sha(delivery / row["path"]) == row["sha256"],
                    f"SHA256SUMS entry mismatch: {row['path']}",
                )

    record["status"] = "PASS"
    record["hash"] = canonical({k: v for k, v in record.items() if k != "hash"})
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="write the QA record JSON here")
    parser.add_argument("--repo", type=Path, default=REPO, help="workspace root for identity checks")
    parser.add_argument("--timeout-seconds", type=int, default=None,
                        help="override the platform timeout (default: configs/kaggle_pipeline.json)")
    parser.add_argument("--pipeline-config", type=Path, default=None,
                        help="pipeline config supplying timeout_seconds")
    args = parser.parse_args(argv)
    try:
        record = verify(
            args.delivery,
            repo=args.repo,
            output=args.output,
            timeout_seconds=args.timeout_seconds,
            pipeline_config=args.pipeline_config,
        )
    except QAFailure as error:
        failure = {"schema": "guandan-dmc-delivery-qa-v1", "status": "FAIL", "error": str(error)}
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
