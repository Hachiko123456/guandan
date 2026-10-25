# SPDX-License-Identifier: MIT
"""Build the immutable DMC Gate L delivery (stage 5).

The delivery directory contains exactly what the unchanged Kaggle pipeline
consumes (``tools/kaggle_pipeline/core.py``):

* ``guandan_serial_compare_payload_v1.bin`` / ``guandan_serial_compare_v1.ipynb``
  — the pipeline's transport filenames are serial-compare module constants
  (``core.verify_staging`` copies exactly these names); the *content* schemas
  below are DMC's own;
* ``plan.json`` — the hashed run plan (budgets, schedule, pre-declared gates);
* plus delivery records (manifest / SHA256SUMS / verification / README).

The payload is an opaque transport (DMC magic + base64 zip) holding the
DMC-only overlay of this workspace (``guandan``, ``guandan_marvel`` Python
sources, byte-identical), the ``dmc_gate_l`` driver, the Gate L training
config and the plan.  Gate L runs the explicit ``python_reference`` backend,
so no native binary is shipped and none is needed on the Kaggle runtime.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Mapping

REPO = Path(__file__).resolve().parents[1]

# Pipeline transport constants (tools/kaggle_pipeline/core.py + registry).
PAYLOAD = "guandan_serial_compare_payload_v1.bin"
NOTEBOOK = "guandan_serial_compare_v1.ipynb"
MAGIC = b"GUANDAN-DMC-GATEL-V1\n"
MAX_PAYLOAD_BYTES = 256 * 1024**2

PLAN_SCHEMA = "guandan-dmc-gate-l-plan-v1"
MANIFEST_SCHEMA = "guandan-dmc-gate-l-delivery-manifest-v1"
VERIFICATION_SCHEMA = "guandan-dmc-gate-l-delivery-verification-v1"

OVERLAY_ROOTS = ("guandan", "guandan_marvel")
DRIVER_PACKAGE = "tools/dmc_gate_l"
CONFIG_PATH = "configs/dmc_gate_l.json"
FORBIDDEN_SUFFIXES = {
    ".so", ".pyd", ".dll", ".dylib", ".o", ".obj", ".a", ".lib", ".cpp", ".cc", ".c",
    ".hpp", ".h", ".pyx", ".pxd", ".cu",
}

#: Pre-declared gate table — verbatim from docs/dmc_prompts/05_gate_l_local.md.
#: The verifier refuses any delivery whose thresholds differ from these.
GATES = {
    "health": {"illegal": 0, "nonfinite": 0, "truncated": 0},
    "q_calibration": {"final_qz_r2_min": 0.10, "must_rise": True},
    "vs_rule": {"final_min": 0.65, "mini_min_for_continue": 0.60, "must_rise_from_mini": True},
    "vs_random": {"final_min": 0.85},
    "throughput": {"min_games_in_budget": 12000, "stretch_games": 20000, "budget_seconds": 14400},
}
BUDGETS = {
    "total_seconds": 14400,
    "train_seconds": 11400,
    "eval_reserve_seconds": 2400,
    "platform_timeout_seconds": 15000,
}
SCHEDULE = {
    "checkpoint_interval_games": 512,
    "mini_eval": {"fraction_of_train_budget": 0.5, "vs_rule_games": 96},
    "final_eval": {"vs_rule_games": 256, "vs_random_games": 256},
    "eval_seed_base": {
        "mini_vs_rule": 91_000_000,
        "final_vs_rule": 92_000_000,
        "final_vs_random": 93_000_000,
    },
}
BOUNDARIES = {
    "fabledan": "NOT_RUN",
    "final_holdout": "NOT_TOUCHED",
    "belief": "DISABLED",
    "opponent_pool": "DISABLED",
    "formal_production_promotion": False,
}
#: Stage 07C: the delivery declares *which* environment driver the run uses and
#: the differential suite that gates it.  Both are hashed into ``plan_hash``, so
#: a run can never quietly fall back to the slow adapter (or claim an
#: equivalence it did not execute).
ENV_DRIVER = {
    "name": "dmc_env_fast",
    "module": "guandan_marvel/dmc/env_fast.py",
    "version": "dmc-env-fast-v1",
    "status": "ENABLED",
    "scope": "collection and embedded evaluation (tools/dmc_gate_l/evaluate.py)",
    "reference_driver": "guandan_marvel.game.EnvBatch (python_reference)",
    "differential_suite": "dmc-env-fast-differential-v1",
    "differential_games": 200,
    "differential_equivalence": "byte-exact, no tolerance, no state sampling",
}
#: Stage 07A's corrected early-stop rule (``07A_gate_l_repair_throughput.md``
#: tail, 2026-10-25): the obsolete "< 1.2 games/s -> stop" line is void.  The
#: rule extrapolates the *sustained* rate after ``after_games`` and stops only
#: when the projection cannot reach the throughput gate's own game floor within
#: the throughput gate's own budget — i.e. exactly when the run is already
#: incapable of passing ``gates.throughput``.
EARLY_STOP = {
    "rule": "progress_extrapolation",
    "source": "docs/dmc_prompts/07A_gate_l_repair_throughput.md (Gate L 重跑附加规则, 2026-10-25 修正)",
    "after_games": 1000,
    "extrapolation_budget_seconds": 14400,
    "min_extrapolated_games": 12000,
    "supersedes": "the earlier '< 1.2 games/s -> early stop' line is void",
    "on_trigger": (
        "stop training at the next cycle boundary, record stop_reason="
        "early_stop_extrapolation, still run the pre-declared evaluations"
    ),
    "note": (
        "extrapolated_games = games_completed / elapsed_seconds * "
        "extrapolation_budget_seconds; the horizon equals gates.throughput."
        "budget_seconds and the floor equals gates.throughput.min_games_in_budget, "
        "so the stop line is the throughput gate itself (12000/14400 = 0.833 "
        "games/s). 07A's parenthetical '~0.95 games/s' assumed a 12632 s horizon; "
        "this delivery pins the horizon to the hashed gate budget instead."
    ),
}
DEVIATIONS = [
    "batch_size=1024 with accumulation_steps=4 (micro-batch 256, stage 07B): the lock "
    "table's batch is the optimizer-step batch; the 256-row micro-batch is the T4 memory "
    "ceiling measured in stage 4, and the gradient/data ratio stays 1.0 (optimizer steps "
    "divide by the 1024-row batch, never by a micro-batch)",
    "observation window 1024 = head 128 + tail 896 (stage 07B): the opening history "
    "(BOS/level card, first play, tribute/return records) is pinned instead of being "
    "tail-truncated; tokens between the two segments are dropped and counted as "
    "obs_tail_drops (obs_head_drops stays 0 by construction)",
    "environment driving uses the stage-07C DMC-only thin driver "
    "(guandan_marvel/dmc/env_fast.py) instead of EnvBatch: it encodes only the acting "
    "seat's 512 channels, submits tokens without rebuilding V2 action rows, and validates "
    "at sequence boundaries instead of per token. guandan/ and EnvBatch are unmodified; "
    "the driver is gated by the byte-exact differential suite "
    "dmc-env-fast-differential-v1 (>=200 random games, no tolerance), and the embedded "
    "evaluation runs the same driver so train and eval share one distribution",
    "checkpoint cadence is 512 games = 2 cycles of the locked 256-game cycle "
    "(prompt wording 'every 500 games' rounded to a whole cycle multiple)",
    "the payload/notebook filenames are the unchanged pipeline's serial-compare transport "
    "constants; all content schemas in this delivery are DMC's own",
    "kaggle.ps1 fetch will report FETCHED_INCOMPLETE for DMC runs because the pipeline's "
    "result verifier is serial-compare bound; artifacts/RESULT.json + RESULTS_MANIFEST.json "
    "are the authoritative Gate L verification",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Mapping) -> str:
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def zip_bytes(files: Mapping[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    return stream.getvalue()


def collect_overlay(repo: Path) -> dict[str, bytes]:
    repo = Path(repo).resolve()
    files: dict[str, bytes] = {}
    for root in OVERLAY_ROOTS:
        base = repo / root
        if not base.is_dir():
            raise FileNotFoundError(f"overlay root is missing: {base}")
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(repo).as_posix()
            data = path.read_bytes()
            ast.parse(data.decode("utf-8-sig"))
            files[relative] = data
    if not files:
        raise ValueError("the overlay is empty")
    for path in repo.joinpath("guandan").rglob("*"):
        if path.is_file() and path.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError(f"unexpected native file in the overlay source tree: {path}")
    return files


def collect_driver(repo: Path) -> dict[str, bytes]:
    repo = Path(repo).resolve()
    files: dict[str, bytes] = {}
    for path in sorted((repo / DRIVER_PACKAGE).glob("*.py")):
        data = path.read_bytes()
        ast.parse(data.decode("utf-8-sig"))
        files[f"dmc_gate_l/{path.name}"] = data
    if not files:
        raise ValueError("the driver is empty")
    return files


def build_plan(repo: Path, git_commit: str) -> dict:
    plan = {
        "schema": PLAN_SCHEMA,
        "stage": "gate_l",
        "prompt": "docs/dmc_prompts/05_gate_l_local.md",
        "source": {"git_commit": git_commit, "branch": "dmc-v1"},
        "config": {"path": CONFIG_PATH, "sha256": sha_bytes((repo / CONFIG_PATH).read_bytes())},
        "budgets": dict(BUDGETS),
        "schedule": {**{key: value for key, value in SCHEDULE.items()}},
        "gates": json.loads(json.dumps(GATES)),
        "boundaries": dict(BOUNDARIES),
        "env_driver": dict(ENV_DRIVER),
        "early_stop": json.loads(json.dumps(EARLY_STOP)),
        "deviations": list(DEVIATIONS),
    }
    plan["plan_hash"] = canonical({key: value for key, value in plan.items() if key != "plan_hash"})
    return plan


BOOTSTRAP = r'''from pathlib import Path
import base64, hashlib, io, json, os, shutil, sys, time, traceback, zipfile
STARTED = time.monotonic()
PAYLOAD_SHA256 = __PAYLOAD_SHA__
MANIFEST_HASH = __MANIFEST_SHA__
PLAN_HASH = __PLAN_SHA__
RUN_ROOT = Path('/kaggle/working/dmc_gate_l_v1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def unpack(bundle, root, *, flat=False, limit=1024 * 1024**2):
    members = bundle.infolist()
    if len({item.filename.casefold() for item in members}) != len(members) or sum(item.file_size for item in members) > limit:
        raise RuntimeError('duplicate/oversize archive')
    for item in members:
        name = item.orig_filename
        target = (root / name).resolve()
        if name != item.filename or any(character in name for character in ('\\', ':', '\0')):
            raise RuntimeError('unsafe archive member')
        if target == root.resolve() or not target.is_relative_to(root.resolve()):
            raise RuntimeError('archive member escapes its root')
        if ((item.external_attr >> 16) & 0o170000) == 0o120000:
            raise RuntimeError('archive symlink')
        if flat and ('/' in name or item.is_dir()):
            raise RuntimeError('flat archive required')
    root.mkdir(parents=True, exist_ok=True)
    bundle.extractall(root)


print('[DMC-GATE-L] one payload, budgeted DMC training + embedded greedy evaluation; no upload from code', flush=True)
try:
    if RUN_ROOT.exists():
        raise RuntimeError('RUN_ROOT already exists; download the old output and use a fresh session')
    input_dir = RUN_ROOT / 'input'
    candidates = list(Path('/kaggle/input').rglob('guandan_serial_compare_payload_v1.bin'))
    matches = [p for p in candidates if p.is_file() and not p.is_symlink() and p.stat().st_size <= 256 * 1024**2 and sha(p) == PAYLOAD_SHA256]
    if len(matches) != 1:
        raise RuntimeError('expected exactly one matching DMC Gate L payload; do not mix old packages: ' + str(candidates))
    raw = matches[0].read_bytes()
    magic = b'GUANDAN-DMC-GATEL-V1\n'
    if not raw.startswith(magic):
        raise RuntimeError('wrong opaque transport version')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw[len(magic):].strip(), validate=True))) as bundle:
        meta = json.loads(bundle.read('delivery_manifest.json'))
        if meta['manifest_hash'] != MANIFEST_HASH or canonical({k: v for k, v in meta.items() if k != 'manifest_hash'}) != meta['manifest_hash']:
            raise RuntimeError('Notebook/payload manifest mismatch')
        if {item.filename for item in bundle.infolist()} != {'delivery_manifest.json', *(row['path'] for row in meta['files'])}:
            raise RuntimeError('undeclared payload files')
        for row in meta['files']:
            data = bundle.read(row['path'])
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise RuntimeError('payload hash mismatch: ' + row['path'])
        unpack(bundle, input_dir, flat=True)
    plan = json.loads((input_dir / 'plan.json').read_text(encoding='utf-8'))
    if canonical({k: v for k, v in plan.items() if k != 'plan_hash'}) != plan['plan_hash'] or plan['plan_hash'] != PLAN_HASH:
        raise RuntimeError('plan hash is inconsistent')
    for archive_name, root_name, key in (('driver.zip', 'driver', 'driver_files'), ('overlay_source.zip', 'src', 'overlay_files')):
        rows = {row['path']: row for row in meta[key]}
        root = RUN_ROOT / root_name
        with zipfile.ZipFile(input_dir / archive_name) as bundle:
            members = bundle.infolist()
            if sorted(item.orig_filename for item in members) != sorted(rows):
                raise RuntimeError(archive_name + ' member list changed')
            for item in members:
                data = bundle.read(item.orig_filename)
                row = rows[item.orig_filename]
                if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                    raise RuntimeError(archive_name + ' member changed: ' + item.orig_filename)
            unpack(bundle, root, limit=128 * 1024**2)
        for name, row in rows.items():
            extracted = root / name
            if not extracted.is_file() or extracted.stat().st_size != row['bytes'] or sha(extracted) != row['sha256']:
                raise RuntimeError(archive_name + ' extracted tree changed: ' + name)
    driver_root, source_root = RUN_ROOT / 'driver', RUN_ROOT / 'src'
    sys.path.insert(0, str(driver_root))
    sys.path.insert(0, str(source_root))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(driver_root), str(source_root)])
    os.environ['PYTHONUNBUFFERED'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(index) for index in range(2)):
        raise RuntimeError('Gate L requires exactly two T4 GPUs, no fallback')
    from dmc_gate_l.runner import run_notebook
    result = run_notebook(dataset=input_dir, run_root=RUN_ROOT, clock=lambda: time.monotonic() - STARTED)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    from IPython.display import FileLink, display
    display(FileLink(str(RUN_ROOT / 'guandan_dmc_gate_l_results.zip')))
    if result.get('gates', {}).get('overall_verdict') != 'PASS':
        print('[DMC-GATE-L] gates are not all PASS; see artifacts/RESULT.json — archive this session, do not blindly rerun', flush=True)
except BaseException as error:
    diagnostics = RUN_ROOT.parent / (RUN_ROOT.name + '_bootstrap') if RUN_ROOT.parent.exists() else Path('/kaggle/working/dmc_gate_l_bootstrap')
    try:
        diagnostics.mkdir(parents=True, exist_ok=True)
        (diagnostics / 'bootstrap_error.json').write_text(json.dumps({'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc(), 'plan_hash': PLAN_HASH}, ensure_ascii=False, indent=2), encoding='utf8')
    except Exception:
        pass
    raise
'''


def build_bootstrap_code(payload_sha256: str, manifest_hash: str, plan_hash: str) -> str:
    code = (
        BOOTSTRAP
        .replace("__PAYLOAD_SHA__", repr(payload_sha256))
        .replace("__MANIFEST_SHA__", repr(manifest_hash))
        .replace("__PLAN_SHA__", repr(plan_hash))
    )
    ast.parse(code)
    return code


def build_delivery(repo: Path, output: Path, *, git_commit: str) -> dict:
    repo = Path(repo).resolve()
    output = Path(output).resolve()
    staging = output.with_name(output.name + "_build")
    if output.exists() or staging.exists():
        raise FileExistsError("never overwrite an old delivery")
    staging.mkdir(parents=True)

    config_bytes = (repo / CONFIG_PATH).read_bytes()
    overlay = collect_overlay(repo)
    driver = collect_driver(repo)
    plan = build_plan(repo, git_commit)

    payload_members = {
        "plan.json": json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "gate_l_config.json": config_bytes,
        "overlay_source.zip": zip_bytes(overlay),
        "driver.zip": zip_bytes(driver),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "plan_hash": plan["plan_hash"],
        "config_sha256": sha_bytes(config_bytes),
        "overlay_files": [
            {"path": name, "bytes": len(data), "sha256": sha_bytes(data)}
            for name, data in sorted(overlay.items())
        ],
        "driver_files": [
            {"path": name, "bytes": len(data), "sha256": sha_bytes(data)}
            for name, data in sorted(driver.items())
        ],
        "files": [
            {"path": name, "bytes": len(data), "sha256": sha_bytes(data)}
            for name, data in sorted(payload_members.items())
        ],
        "source": dict(plan["source"]),
        "boundaries": dict(BOUNDARIES),
    }
    manifest["manifest_hash"] = canonical({key: value for key, value in manifest.items() if key != "manifest_hash"})
    payload_members["delivery_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )

    payload = MAGIC + base64.b64encode(zip_bytes(payload_members)) + b"\n"
    if len(payload) > MAX_PAYLOAD_BYTES or zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValueError("invalid/oversize opaque upload")

    code = build_bootstrap_code(sha_bytes(payload), manifest["manifest_hash"], plan["plan_hash"])
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": False},
        },
        "cells": [
            {"id": "scope", "cell_type": "markdown", "metadata": {}, "source": [
                "# DMC Gate L：预算内训练 + 内嵌 ε=0 评估（一次 Run All）\n",
                "Dataset 只上传 `" + PAYLOAD + "`（管线传输常量名，内容为 DMC 交付），GPU 选 T4×2，关闭 Internet。\n",
                "预算 14400s：训练 11400s（cycle 边界安全停止、每 512 局 checkpoint），~50% 时点 mini 评估 vs rule 96 局，终点 vs rule 256 + vs random 256。\n",
                "门槛在 plan.json 中预声明并随包哈希；无 FableDan、无 final holdout、无 belief/对手池。\n",
            ]},
            {"id": "run-all", "cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": code.splitlines(keepends=True)},
        ],
    }

    final_dir = staging / "delivery"
    final_dir.mkdir()
    (final_dir / PAYLOAD).write_bytes(payload)
    (final_dir / NOTEBOOK).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (final_dir / "plan.json").write_bytes(payload_members["plan.json"])
    (final_dir / "DELIVERY_MANIFEST.json").write_bytes(payload_members["delivery_manifest.json"])
    verification = {
        "schema": VERIFICATION_SCHEMA,
        "status": "BUILT_VERIFY_PENDING",
        "payload": PAYLOAD,
        "payload_bytes": len(payload),
        "payload_sha256": sha_bytes(payload),
        "notebook": NOTEBOOK,
        "notebook_sha256": sha(final_dir / NOTEBOOK),
        "plan_hash": plan["plan_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "config_sha256": sha_bytes(config_bytes),
        "overlay_files": len(overlay),
        "driver_files": len(driver),
        "budgets": dict(BUDGETS),
        "boundaries": dict(BOUNDARIES),
        "kaggle_execution": "NOT_RUN",
        "delivery_verify": "NOT_RUN",
    }
    (final_dir / "DELIVERY_VERIFICATION.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    readme = repo / "docs" / "dmc_prompts" / "05_gate_l_local.md"
    if readme.is_file():
        (final_dir / "STAGE_PROMPT.md").write_bytes(readme.read_bytes())
    sums = {
        "schema": "guandan-dmc-gate-l-sha256sums-v1",
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(final_dir.iterdir()) if path.is_file()
        ],
    }
    sums["hash"] = canonical(sums)
    (final_dir / "SHA256SUMS").write_text(
        json.dumps(sums, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    final_dir.rename(output)
    staging.rename(output.with_name(output.name + "_build"))
    return verification


def current_git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True, check=True,
    )
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new delivery directory (never overwritten)")
    parser.add_argument("--git-commit", default=None, help="source commit to pin (default: git rev-parse HEAD)")
    args = parser.parse_args(argv)
    commit = args.git_commit if args.git_commit is not None else current_git_commit(REPO)
    verification = build_delivery(REPO, args.output, git_commit=commit)
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
