# SPDX-License-Identifier: MIT
"""Offline QA of the *actual delivered* DMC Gate A payload (stage 6).

Decodes the real ``.bin`` into an isolated directory and verifies:

* payload / manifest / per-file hashes, and that the top-level ``plan.json`` is
  byte-identical to the payload's;
* the plan schema, internal hash and manifest binding;
* the **budget arithmetic** of a 9 h session (``total + 300 <= platform
  timeout``, ``train + eval reserve <= total``, the config's own wall budget
  equal to the plan's train budget, the checkpoint cadence equal to
  ``games_per_cycle x checkpoint_interval_cycles``);
* the **stage-07A parallel topology** declaration (learner/collector devices,
  two T4s, one cycle of behavioural weight lag);
* the **four-lane evaluation matrix** (rule / m09 / random / legacy, 512 games
  each, declared drivers, declared rotations, declared seed bases);
* the **pre-declared gate table** (vs rule >= 75% *and* above Gate L's 65.2%,
  h2h m09 >= 60%, random >= 90%, zero health violations, throughput
  record-only);
* the **frozen m09 identity**: the checkpoint's own recorded
  ``backend_identity`` is re-read here with ``torch`` and compared against the
  plan *and* against the shipped native runtime (binary SHA256, the four cpp
  source files, their manifest hash, the recorded absolute extension path);
* the pinned Gate A training config;
* the bound Notebook constants (payload/manifest/plan hashes, the transport
  magic, the driver import, the pinned source root);
* byte-identity of the shipped overlay / driver against the current workspace;
* ``SHA256SUMS``.

The thresholds and budgets below are an independent restatement of
``docs/dmc_prompts/06_gate_a_kaggle.md`` (2026-10-26 revision); they are
deliberately NOT imported from the builder, so a wrong constant cannot pass
both.  The verifier never imports the workspace ``guandan_marvel`` package,
never touches the Kaggle SDK, and performs no network or subprocess calls.
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
MAGIC = b"GUANDAN-DMC-GATEA-V1\n"
MAX_PAYLOAD_BYTES = 256 * 1024**2
PLAN_SCHEMA = "guandan-dmc-gate-a-plan-v1"
MANIFEST_SCHEMA = "guandan-dmc-gate-a-delivery-manifest-v1"
CONFIG_PATH = "configs/dmc_gate_a.json"
PIPELINE_CONFIG_PATH = "configs/kaggle_pipeline.json"
SOURCE_ROOT = "/kaggle/working/guandan-marvel-source"
#: core.py: platform timeout must cover the plan budget plus 300 seconds.
BUDGET_MARGIN_SECONDS = 300

EXPECTED_BUDGETS = {
    "total_seconds": 32_100,
    "train_seconds": 28_700,
    "eval_reserve_seconds": 3_400,
    "platform_timeout_seconds": 32_400,
}
EXPECTED_TOPOLOGY = {
    "mode": "stage07a_parallel",
    "learner_device": "cuda:0",
    "collector_device": "cuda:1",
    "required_gpu_count": 2,
    "gpu_name_contains": "T4",
    "behavioral_weight_lag_cycles": 1,
}
EXPECTED_LANES = [
    {"kind": "rule", "games": 512, "seed_base": 101_000_000, "rotations": 2,
     "driver": "dmc_env_fast"},
    {"kind": "m09", "games": 512, "seed_base": 102_000_000, "rotations": 4,
     "driver": "env_batch_reference"},
    {"kind": "random", "games": 512, "seed_base": 103_000_000, "rotations": 2,
     "driver": "dmc_env_fast"},
    {"kind": "legacy", "games": 512, "seed_base": 104_000_000, "rotations": 2,
     "driver": "dmc_env_fast"},
]
EXPECTED_EVAL_ORDER = ["rule", "m09", "random", "legacy"]
EXPECTED_CHECKPOINT_INTERVAL_GAMES = 2_048
EXPECTED_GATES = {
    "health": {"illegal": 0, "nonfinite": 0, "truncated": 0},
    "vs_rule": {
        "pass_min": 0.75,
        "abort_below": 0.68,
        "reference_gate_l_win_rate": 0.652,
        "must_exceed_reference": True,
    },
    "h2h_m09": {"pass_min": 0.60, "abort_below": 0.55},
    "vs_random": {"pass_min": 0.90, "abort_below": 0.85},
    "throughput": {"mode": "record_only"},
}
EXPECTED_BOUNDARIES = {
    "fabledan": "NOT_RUN",
    "final_holdout": "NOT_TOUCHED",
    "belief": "DISABLED",
    "opponent_pool": "DISABLED",
    "formal_production_promotion": False,
}
EXPECTED_RESUME = {
    "mode": "optional_input",
    "input_root": "/kaggle/input",
    "filename": "dmc_gate_a_resume.pt",
    "requirement": "optional",
    "session_target_games": 40_000,
    "total_games": 100_000,
    "sessions_expected": "2-3",
    "replay_reset_expected": True,
}
EXPECTED_CONFIG_VALUES = {
    "schema": "guandan-marvel-dmc-train-config-v1",
    "run_name": "dmc_gate_a",
    "device": "cuda",
    "num_envs": 64,
    "env_driver": "fast",
    "total_games": 100_000,
    "games_per_cycle": 512,
    "gradient_replay_ratio": 1.0,
    "batch_size": 1024,
    "accumulation_steps": 4,
    "obs_head_tokens": 128,
    "buffer_capacity": 500000,
    "max_obs_tokens": 1024,
    "epsilon_start": 0.10,
    "epsilon_final": 0.02,
    "epsilon_decay_samples": 5_000_000,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "checkpoint_interval_cycles": 4,
    "max_wall_seconds": 28_700,
}
EXPECTED_ENV_DRIVER = {
    "name": "dmc_env_fast",
    "version": "dmc-env-fast-v1",
    "status": "ENABLED",
    "differential_suite": "dmc-env-fast-differential-v1",
    "differential_games": 200,
}
#: The frozen m09 candidate, pinned in the prompt as ``af0df1ea…4ca9c8``.
FROZEN_M09_SHA256 = "af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8"
FROZEN_M09_ITERATION = 2000
FROZEN_M09_ALGORITHM = "ippo"
NATIVE_SOURCE_PATHS = (
    "nanopokerlib/cpp/bindings.cpp",
    "nanopokerlib/cpp/core/types.hpp",
    "nanopokerlib/cpp/guandan/guandan.cpp",
    "nanopokerlib/cpp/guandan/guandan.hpp",
)
REQUIRED_OVERLAY = (
    "guandan/action_state.py",
    "guandan/cards.py",
    "guandan/environment.py",
    "guandan/environment_types.py",
    "guandan/state.py",
    "guandan/training/snapshot.py",
    "guandan_marvel/backends.py",
    "guandan_marvel/config.py",
    "guandan_marvel/game.py",
    "guandan_marvel/python_game.py",
    "guandan_marvel/dmc/__init__.py",
    "guandan_marvel/dmc/candidates.py",
    "guandan_marvel/dmc/env_fast.py",
    "guandan_marvel/dmc/learner.py",
    "guandan_marvel/dmc/parallel.py",
    "guandan_marvel/dmc/q_agent.py",
    "guandan_marvel/dmc/replay.py",
    "guandan_marvel/dmc/sampler.py",
    "guandan_marvel/dmc/train.py",
    "guandan_marvel/evaluation/agents.py",
    "guandan_marvel/evaluation/checkpoints.py",
    "guandan_marvel/model/agent.py",
    "nanopokerlib/__init__.py",
    "nanopokerlib/guandan.py",
    "nanopokerlib/build_identity.py",
)
REQUIRED_DRIVER = (
    "dmc_gate_a/__init__.py",
    "dmc_gate_a/lanes.py",
    "dmc_gate_a/m09.py",
    "dmc_gate_a/runner.py",
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


def _native_manifest(contents: dict[str, bytes]) -> dict:
    """Independent restatement of ``nanopokerlib.build_identity``'s digest."""
    files = [
        {"path": name, "bytes": len(contents[name]), "sha256": sha_bytes(contents[name])}
        for name in NATIVE_SOURCE_PATHS
    ]
    payload = {"schema": "guandan-native-build-identity-v1", "files": files}
    return {**payload, "hash": canonical(payload)}


def _load_checkpoint_identity(path: Path) -> dict:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - QA runs where torch exists
        raise QAFailure(
            "torch is required to re-read the frozen m09 checkpoint's recorded "
            f"backend identity ({error})"
        ) from None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("backend_identity")
    _require(isinstance(identity, dict), "the frozen m09 checkpoint has no backend identity")
    implementation = identity.get("implementation")
    _require(isinstance(implementation, dict), "the m09 backend identity has no implementation")
    return {
        "identity": identity,
        "implementation": implementation,
        "algorithm": payload.get("algorithm"),
        "architecture": payload.get("architecture"),
        "iteration": int(payload.get("iteration", 0)),
    }


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
    record: dict = {"schema": "guandan-dmc-gate-a-delivery-qa-v1", "delivery": str(delivery)}
    timeout = int(timeout_seconds) if timeout_seconds is not None else _platform_timeout(pipeline_config)
    record["platform_timeout_seconds"] = timeout

    with tempfile.TemporaryDirectory(prefix="dmc-gate-a-qa-") as temporary:
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
        _require(plan.get("stage") == "gate_a", "plan stage is not gate_a")
        _require(
            plan.get("prompt") == "docs/dmc_prompts/06_gate_a_kaggle.md",
            "plan does not name the Gate A prompt",
        )

        budgets = plan.get("budgets")
        _require(isinstance(budgets, dict), "plan has no budgets mapping")
        for field, expected in EXPECTED_BUDGETS.items():
            value = budgets.get(field)
            _require(type(value) is int, f"budgets.{field} must be an integer")
            _require(value == expected, f"budgets.{field} drifted: {value} != {expected}")
        _require(
            budgets["total_seconds"] + BUDGET_MARGIN_SECONDS <= timeout,
            f"plan budget {budgets['total_seconds']}s exceeds the platform timeout {timeout}s "
            f"minus the {BUDGET_MARGIN_SECONDS}s margin (core.py convention)",
        )
        _require(
            budgets["platform_timeout_seconds"] == timeout,
            "plan platform_timeout_seconds disagrees with the pipeline config",
        )
        _require(
            budgets["train_seconds"] + budgets["eval_reserve_seconds"] <= budgets["total_seconds"],
            "train + eval reserve exceed the total plan budget",
        )
        _require(
            budgets["total_seconds"] <= budgets["platform_timeout_seconds"] - BUDGET_MARGIN_SECONDS,
            "the plan budget leaves no 300 s margin inside the platform timeout",
        )
        record["budget_total_seconds"] = budgets["total_seconds"]

        _require(plan.get("topology") == EXPECTED_TOPOLOGY, "plan topology drifted from stage 07A")
        _require(plan.get("gates") == EXPECTED_GATES, "plan gates drifted from the pre-declared table")
        _require(plan.get("boundaries") == EXPECTED_BOUNDARIES, "plan boundaries drifted")
        _require(plan.get("resume") == EXPECTED_RESUME, "plan resume contract drifted")
        _require(isinstance(plan.get("deviations"), list) and plan["deviations"], "plan carries no deviation notes")

        driver = plan.get("env_driver")
        _require(isinstance(driver, dict), "plan declares no env_driver block")
        for field, expected in EXPECTED_ENV_DRIVER.items():
            _require(
                driver.get(field) == expected,
                f"plan env_driver.{field} drifted: {driver.get(field)!r} != {expected!r}",
            )
        exception = driver.get("lane_exception") or {}
        _require(exception.get("lane") == "m09", "the m09 driver exception is not declared")
        _require(
            exception.get("driver") == "env_batch_reference",
            "the m09 lane must declare the reference adapter",
        )
        record["env_driver"] = driver["version"]

        schedule = plan.get("schedule")
        _require(isinstance(schedule, dict), "plan has no schedule mapping")
        _require(
            schedule.get("checkpoint_interval_games") == EXPECTED_CHECKPOINT_INTERVAL_GAMES,
            "checkpoint cadence drifted from 2,048 games",
        )
        _require(schedule.get("eval_order") == EXPECTED_EVAL_ORDER, "eval order drifted")
        lanes = schedule.get("lanes")
        _require(isinstance(lanes, list) and len(lanes) == 4, "the evaluation matrix is not four lanes")
        for row, expected in zip(lanes, EXPECTED_LANES):
            for field, value in expected.items():
                _require(
                    row.get(field) == value,
                    f"lane {expected['kind']} {field} drifted: {row.get(field)!r} != {value!r}",
                )
            _require(
                type(row.get("expected_seconds")) in (int, float) and row["expected_seconds"] > 0,
                f"lane {expected['kind']} declares no time estimate",
            )
        _require(
            sum(float(row["expected_seconds"]) for row in lanes) <= budgets["eval_reserve_seconds"],
            "the declared lane estimates exceed the eval reserve",
        )
        record["lanes"] = [row["kind"] for row in lanes]

        # ---- config ----
        config_bytes = (decoded / "gate_a_config.json").read_bytes()
        config = json.loads(config_bytes)
        for field, expected in EXPECTED_CONFIG_VALUES.items():
            _require(
                config.get(field) == expected,
                f"training config {field} drifted: {config.get(field)!r} != {expected!r}",
            )
        _require(
            int(config["games_per_cycle"]) * int(config["checkpoint_interval_cycles"])
            == int(schedule["checkpoint_interval_games"]),
            "the config's checkpoint cadence disagrees with the plan",
        )
        _require(
            float(config["max_wall_seconds"]) == float(budgets["train_seconds"]),
            "the config's wall budget disagrees with the plan's train budget",
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
            "delivered training config differs from the workspace configs/dmc_gate_a.json",
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
        _require("GUANDAN-DMC-GATEA-V1" in code, "the Notebook does not check the DMC transport magic")
        _require("dmc_gate_a" in code, "the Notebook does not import the packaged driver")
        _require("run_notebook" in code, "the Notebook does not call the packaged driver")
        _require(SOURCE_ROOT in code, "the Notebook does not restore the pinned m09 source root")
        _require(
            "dmc_gate_a_resume.pt" in json.dumps(plan["resume"]),
            "the resume filename is not pinned in the plan",
        )
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
        # Driver members are keyed by import path (``dmc_gate_a/...``); on disk
        # they live under ``tools/``.
        record["driver_files"] = _archive_rows("driver.zip", "driver_files", REQUIRED_DRIVER, workspace_prefix="tools")

        # ---- frozen m09 identity ----
        block = plan.get("m09_lane")
        _require(isinstance(block, dict), "plan declares no m09_lane block")
        _require(block.get("loader") == "official", "the m09 lane must use the official loader")
        candidate = block.get("frozen_candidate") or {}
        _require(candidate.get("sha256") == FROZEN_M09_SHA256, "the frozen m09 candidate hash drifted")
        _require(int(candidate.get("iteration", -1)) == FROZEN_M09_ITERATION, "the m09 iteration drifted")
        _require(candidate.get("algorithm") == FROZEN_M09_ALGORITHM, "the m09 algorithm drifted")

        m09_rows = {row["path"]: row for row in manifest["m09_files"]}
        _require(bool(m09_rows), "the m09 bundle is empty")
        with zipfile.ZipFile(decoded / "m09_bundle.zip") as bundle:
            _require(sorted(bundle.namelist()) == sorted(m09_rows), "m09 bundle member list changed")
            m09_contents: dict[str, bytes] = {}
            for name in m09_rows:
                data = bundle.read(name)
                m09_contents[name] = data
                _require(
                    len(data) == m09_rows[name]["bytes"] and sha_bytes(data) == m09_rows[name]["sha256"],
                    f"m09 bundle member changed: {name}",
                )
        _require(
            candidate["path"] in m09_contents,
            "the m09 bundle does not carry the frozen candidate",
        )
        _require(
            sha_bytes(m09_contents[candidate["path"]]) == FROZEN_M09_SHA256,
            "the bundled m09 candidate is not the frozen one",
        )
        anchor = block.get("legacy_anchor") or {}
        _require(anchor.get("path") in m09_contents, "the m09 bundle does not carry the legacy anchor")
        _require(
            sha_bytes(m09_contents[anchor["path"]]) == anchor.get("sha256"),
            "the bundled legacy anchor hash disagrees with the plan",
        )

        runtime = block.get("runtime") or {}
        _require(runtime.get("source_root") == SOURCE_ROOT, "the m09 source root drifted")
        binary_name = runtime.get("binary")
        _require(binary_name in m09_contents, "the m09 bundle does not carry the native runtime")
        native: dict[str, bytes] = {}
        for row in runtime.get("cpp_sources") or []:
            manifest_path = str(row.get("manifest_path"))
            payload_path = str(row.get("payload_path"))
            _require(
                manifest_path in NATIVE_SOURCE_PATHS,
                f"undeclared native source in the m09 bundle: {manifest_path}",
            )
            _require(payload_path in m09_contents, f"missing native source payload: {payload_path}")
            native[manifest_path] = m09_contents[payload_path]
            _require(
                sha_bytes(m09_contents[payload_path]) == row.get("sha256"),
                f"native source hash disagrees with the plan: {payload_path}",
            )
        _require(
            set(native) == set(NATIVE_SOURCE_PATHS),
            "the m09 bundle does not carry exactly the four native sources",
        )
        native_digest = _native_manifest(native)
        _require(
            native_digest["hash"] == candidate.get("embedded_source_sha256"),
            "the shipped native sources do not reproduce the checkpoint's embedded source hash",
        )
        _require(
            sha_bytes(m09_contents[binary_name]) == candidate.get("extension_sha256"),
            "the shipped runtime binary does not match the plan's extension hash",
        )

        # The checkpoint's *own* recorded identity is the authority.
        identity_path = Path(temporary) / "candidate.pt"
        identity_path.write_bytes(m09_contents[candidate["path"]])
        recorded = _load_checkpoint_identity(identity_path)
        implementation = recorded["implementation"]
        for field in (
            "extension_sha256", "extension_path", "embedded_source_sha256", "compiler",
            "python_implementation", "python_version", "python_cache_tag",
        ):
            _require(
                str(implementation.get(field)) == str(candidate.get(field)),
                f"the plan's m09 {field} disagrees with the checkpoint's recorded identity",
            )
        _require(
            recorded["identity"].get("hash") == candidate.get("backend_identity_hash"),
            "the plan's m09 backend identity hash disagrees with the checkpoint",
        )
        _require(
            str(implementation["extension_path"])
            == f"{SOURCE_ROOT}/nanopokerlib/{implementation['extension_filename']}",
            "the recorded extension path does not sit at the pinned source root",
        )
        _require(
            str(implementation["extension_filename"]) == Path(str(binary_name)).name,
            "the shipped binary filename differs from the recorded extension filename",
        )
        _require(
            recorded["algorithm"] == FROZEN_M09_ALGORITHM
            and recorded["iteration"] == FROZEN_M09_ITERATION,
            "the bundled checkpoint is not the frozen iter2000 candidate",
        )
        record["m09"] = {
            "checkpoint_sha256": candidate["sha256"],
            "backend_identity_hash": candidate["backend_identity_hash"],
            "native_source_manifest_hash": native_digest["hash"],
            "python_version": candidate["python_version"],
            "extension_filename": candidate["extension_filename"],
        }

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
        failure = {"schema": "guandan-dmc-gate-a-delivery-qa-v1", "status": "FAIL", "error": str(error)}
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
