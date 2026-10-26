# SPDX-License-Identifier: MIT
"""Build the immutable DMC Gate A delivery (stage 6).

The delivery directory contains exactly what the unchanged Kaggle pipeline
consumes (``tools/kaggle_pipeline/core.py``):

* ``guandan_serial_compare_payload_v1.bin`` / ``guandan_serial_compare_v1.ipynb``
  — the pipeline's transport filenames are serial-compare module constants
  (``core.verify_staging`` copies exactly these names); the *content* schemas
  below are DMC's own;
* ``plan.json`` — the hashed run plan (topology, budgets, lane matrix,
  pre-declared gates, boundaries, deviations);
* plus delivery records (manifest / SHA256SUMS / verification / README).

Payload contents (all opaque, base64 zip behind ``GUANDAN-DMC-GATEA-V1``):

``overlay_source.zip``
    the ``guandan`` / ``guandan_marvel`` / ``nanopokerlib`` Python sources of
    this workspace, byte-identical (the ``nanopokerlib`` package is included
    because the frozen m09 runtime imports it; it carries no native file —
    those travel in the m09 bundle, hashed separately);
``driver.zip``
    the ``dmc_gate_a`` driver package;
``gate_a_config.json``
    the pinned Gate A training config;
``m09_bundle.zip``
    the frozen m09 evidence: the iter2000 candidate checkpoint, the legacy V1
    anchor, and the **exact** native runtime (binary + the four cpp sources)
    that the checkpoint's backend identity was written with.

The m09 identity is not hand-typed: this builder reads it out of the frozen
checkpoint and refuses to build unless the shipped binary and sources
reproduce it byte for byte.  See ``tools/dmc_gate_a/m09.py`` for why that is
the only way to keep the official loader honest.

Usage::

    python scripts/build_dmc_gate_a_delivery.py \\
        --output runs/dmc_gate_a_delivery_20261026/delivery \\
        --m09-source D:/project/kaggle_train/guandan-marvel-m09-final-holdout-iter2000-v1/dataset
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
MAGIC = b"GUANDAN-DMC-GATEA-V1\n"
MAX_PAYLOAD_BYTES = 256 * 1024**2

PLAN_SCHEMA = "guandan-dmc-gate-a-plan-v1"
MANIFEST_SCHEMA = "guandan-dmc-gate-a-delivery-manifest-v1"
VERIFICATION_SCHEMA = "guandan-dmc-gate-a-delivery-verification-v1"
SOURCE_MANIFEST_SCHEMA = "guandan-dmc-gate-a-m09-bundle-manifest-v1"

OVERLAY_ROOTS = ("guandan", "guandan_marvel", "nanopokerlib")
#: Roots whose tree must be free of native files.  ``nanopokerlib`` is exempt:
#: the local build artifact (``*.pyd``/``*.so``) is never shipped, and the
#: *frozen* binary travels in the m09 bundle where it is hash-pinned against the
#: checkpoint's own recorded identity.
NATIVE_FREE_ROOTS = ("guandan", "guandan_marvel")
DRIVER_PACKAGE = "tools/dmc_gate_a"
CONFIG_PATH = "configs/dmc_gate_a.json"
PROMPT_PATH = "docs/dmc_prompts/06_gate_a_kaggle.md"
FORBIDDEN_OVERLAY_SUFFIXES = {
    ".so", ".pyd", ".dll", ".dylib", ".o", ".obj", ".a", ".lib", ".cpp", ".cc", ".c",
    ".hpp", ".h", ".pyx", ".pxd", ".cu",
}

#: The four files ``nanopokerlib.build_identity`` hashes into the backend
#: identity.  They must be the bytes the m09 build used, not this repo's.
NATIVE_SOURCE_PATHS = (
    "nanopokerlib/cpp/bindings.cpp",
    "nanopokerlib/cpp/core/types.hpp",
    "nanopokerlib/cpp/guandan/guandan.cpp",
    "nanopokerlib/cpp/guandan/guandan.hpp",
)

#: Runtime layout: the notebook restores the source tree here, because the
#: frozen checkpoint's ``extension_path`` names this exact absolute path.
SOURCE_ROOT = "/kaggle/working/guandan-marvel-source"
RUN_ROOT = "/kaggle/working/dmc_gate_a_v1"

#: 9 h session arithmetic (06_gate_a_kaggle.md, 2026-10-26 revision).
#: ``total + 300 <= platform timeout`` is the pipeline's own convention
#: (``tools/kaggle_pipeline/core.py``); the 600 s between the plan budget and
#: the platform timeout covers kernel start-up, payload unpack, the m09
#: precheck and results packaging.
BUDGETS = {
    "total_seconds": 32_100,
    "train_seconds": 28_700,
    "eval_reserve_seconds": 3_400,
    "platform_timeout_seconds": 32_400,
}
SCHEDULE = {
    "checkpoint_interval_games": 2_048,
    "eval_order": ["rule", "m09", "random", "legacy"],
    "lanes": [
        {
            "kind": "rule",
            "games": 512,
            "seed_base": 101_000_000,
            "rotations": 2,
            "driver": "dmc_env_fast",
            "expected_seconds": 480,
            "note": "continuation of Gate L's 65.2% measurement (same pairing)",
        },
        {
            "kind": "m09",
            "games": 512,
            "seed_base": 102_000_000,
            "rotations": 4,
            "driver": "env_batch_reference",
            "expected_seconds": 1_600,
            "note": (
                "128 deals x 4 rotations against the frozen iter2000 candidate "
                "through the official loader/session"
            ),
        },
        {
            "kind": "random",
            "games": 512,
            "seed_base": 103_000_000,
            "rotations": 2,
            "driver": "dmc_env_fast",
            "expected_seconds": 620,
            "note": "health floor",
        },
        {
            "kind": "legacy",
            "games": 512,
            "seed_base": 104_000_000,
            "rotations": 2,
            "driver": "dmc_env_fast",
            "expected_seconds": 700,
            "note": "comparability with the historical m09 anchor",
        },
    ],
}
#: Pre-declared gate table — verbatim from docs/dmc_prompts/06_gate_a_kaggle.md.
#: The verifier refuses any delivery whose thresholds differ from these.
GATES = {
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
BOUNDARIES = {
    "fabledan": "NOT_RUN",
    "final_holdout": "NOT_TOUCHED",
    "belief": "DISABLED",
    "opponent_pool": "DISABLED",
    "formal_production_promotion": False,
}
TOPOLOGY = {
    "mode": "stage07a_parallel",
    "learner_device": "cuda:0",
    "collector_device": "cuda:1",
    "required_gpu_count": 2,
    "gpu_name_contains": "T4",
    "behavioral_weight_lag_cycles": 1,
}
ENV_DRIVER = {
    "name": "dmc_env_fast",
    "module": "guandan_marvel/dmc/env_fast.py",
    "version": "dmc-env-fast-v1",
    "status": "ENABLED",
    "scope": "collection and the rule/random/legacy evaluation lanes",
    "reference_driver": "guandan_marvel.game.EnvBatch (python_reference)",
    "differential_suite": "dmc-env-fast-differential-v1",
    "differential_games": 200,
    "differential_equivalence": "byte-exact, no tolerance, no state sampling",
    "lane_exception": {
        "lane": "m09",
        "driver": "env_batch_reference",
        "reason": (
            "the thin driver encodes only the acting seat's channels (stage 07C "
            "lever 1); a V2 token-level policy consumes all four seats, so the h2h "
            "lane runs on the reference adapter the official host uses"
        ),
    },
}
RESUME = {
    "mode": "optional_input",
    "input_root": "/kaggle/input",
    "filename": "dmc_gate_a_resume.pt",
    "requirement": "optional",
    "session_target_games": 40_000,
    "total_games": 100_000,
    "sessions_expected": "2-3",
    "replay_reset_expected": True,
}
DEVIATIONS = [
    "the 9 h session is split as train 28,800 s + eval reserve 3,000 s + 600 s of "
    "start-up/packaging slack inside a 32,400 s platform timeout: "
    "06_gate_a_kaggle.md's '~30,000 s train budget' is the target and 28,800 s is "
    "the value that still satisfies the pipeline's 'plan budget + 300 s <= "
    "platform timeout' rule with room for the four evaluation lanes",
    "batch_size=1024 with accumulation_steps=4 (micro-batch 256, stage 07B): the "
    "lock table's batch is the optimizer-step batch; the gradient/data ratio stays 1.0",
    "observation window 1024 = head 128 + tail 896 (stage 07B), unchanged from Gate L",
    "checkpoint cadence is 2,048 games = 4 cycles of the locked 512-game cycle "
    "(checkpoint_interval_cycles=4)",
    "the m09 lane runs on the reference EnvBatch adapter, not the stage-07C thin "
    "driver: the thin driver zeroes the three non-acting seats' channels and a V2 "
    "token-level policy reads all four (see plan.env_driver.lane_exception)",
    "the m09 lane pairs 128 deals x 4 rotations (host-style seat_assignment 0..3) "
    "while the rule/random/legacy lanes keep Gate L's 2-rotation pairing, so the "
    "rule lane stays directly comparable with the 65.2% Gate L measurement",
    "the payload carries the frozen native runtime (1.0 MB binary + the four exact "
    "cpp sources) because the official m09 loader validates the checkpoint's "
    "recorded cpp build identity, which includes the extension hash, its absolute "
    "path and the interpreter patch version; nothing else can reproduce that lane",
    "the m09 lane is proven runnable before training starts (precheck); an "
    "unreproducible runtime aborts the session in seconds instead of after 8 h",
    "the payload/notebook filenames are the unchanged pipeline's serial-compare "
    "transport constants; all content schemas in this delivery are DMC's own",
    "kaggle.ps1 fetch will report FETCHED_INCOMPLETE for DMC runs because the "
    "pipeline's result verifier is serial-compare bound; artifacts/RESULT.json + "
    "RESULTS_MANIFEST.json are the authoritative Gate A verification",
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
            info = zipfile.ZipInfo(name, (2026, 10, 26, 0, 0, 0))
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
    for root in NATIVE_FREE_ROOTS:
        for path in repo.joinpath(root).rglob("*"):
            if path.is_file() and path.suffix in FORBIDDEN_OVERLAY_SUFFIXES:
                raise ValueError(
                    f"unexpected native file in the overlay source tree: {path} "
                    "(native build inputs travel in the m09 bundle only)"
                )
    return files


def collect_driver(repo: Path) -> dict[str, bytes]:
    repo = Path(repo).resolve()
    files: dict[str, bytes] = {}
    for path in sorted((repo / DRIVER_PACKAGE).glob("*.py")):
        data = path.read_bytes()
        ast.parse(data.decode("utf-8-sig"))
        files[f"dmc_gate_a/{path.name}"] = data
    if not files:
        raise ValueError("the driver is empty")
    return files


def _native_manifest(rows: Mapping[str, bytes]) -> dict:
    files = [
        {"path": name, "bytes": len(rows[name]), "sha256": sha_bytes(rows[name])}
        for name in NATIVE_SOURCE_PATHS
    ]
    payload = {"schema": "guandan-native-build-identity-v1", "files": files}
    return {**payload, "hash": canonical(payload)}


def _extract_native_sources(source: Path) -> dict[str, bytes]:
    """Pull the four cpp files the m09 build used out of its source archive."""
    archive = Path(source) / "guandan-marvel-source.zip"
    if not archive.is_file():
        raise FileNotFoundError(
            f"the m09 source archive is missing: {archive} (it carries the exact cpp "
            "sources the frozen checkpoint's backend identity was written with)"
        )
    found: dict[str, bytes] = {}
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for relative in NATIVE_SOURCE_PATHS:
            matches = [name for name in names if name.endswith(relative)]
            if len(matches) != 1:
                raise ValueError(
                    f"expected exactly one {relative} inside {archive}, found {matches}"
                )
            found[relative] = bundle.read(matches[0])
    return found


def collect_m09_bundle(source: Path) -> tuple[dict[str, bytes], dict]:
    """Assemble the frozen m09 evidence and derive its identity from the checkpoint.

    Every pinned value in the plan comes from the checkpoint itself; the build
    fails unless the shipped binary and native sources reproduce it.
    """
    source = Path(source).resolve()
    candidate_path = source / "ippo_iter002000_final_candidate.pt"
    anchor_path = source / "legacy_v1_anchor.pt"
    runtime_dir = source / "runtime"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"frozen m09 candidate is missing: {candidate_path}")
    if not anchor_path.is_file():
        raise FileNotFoundError(f"frozen legacy anchor is missing: {anchor_path}")

    import torch

    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    identity = payload.get("backend_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("the frozen m09 candidate carries no backend identity")
    implementation = identity.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("the frozen m09 candidate's backend identity has no implementation")
    if payload.get("algorithm") != "ippo":
        raise ValueError(f"unexpected m09 algorithm {payload.get('algorithm')!r}")

    extension_filename = str(implementation["extension_filename"])
    binary_path = runtime_dir / extension_filename
    if not binary_path.is_file():
        raise FileNotFoundError(f"frozen m09 runtime binary is missing: {binary_path}")
    binary_sha = sha(binary_path)
    if binary_sha != implementation["extension_sha256"]:
        raise ValueError(
            "the shipped runtime binary does not match the checkpoint's recorded "
            f"extension sha256: {binary_sha} != {implementation['extension_sha256']}"
        )

    sources = _extract_native_sources(source)
    manifest = _native_manifest(sources)
    if manifest["hash"] != implementation["embedded_source_sha256"]:
        raise ValueError(
            "the archived cpp sources do not reproduce the checkpoint's embedded "
            f"source hash: {manifest['hash']} != {implementation['embedded_source_sha256']}"
        )
    if implementation["current_source_manifest"]["hash"] != manifest["hash"]:
        raise ValueError("the checkpoint's recorded source manifest is self-inconsistent")

    recorded_path = str(implementation["extension_path"])
    expected_path = f"{SOURCE_ROOT}/nanopokerlib/{extension_filename}"
    if recorded_path != expected_path:
        raise ValueError(
            "the frozen checkpoint was written with a different runtime root; the "
            f"notebook restores to {SOURCE_ROOT}, but the identity names {recorded_path}"
        )

    bundle_files: dict[str, bytes] = {}
    for relative, data in sources.items():
        bundle_files[relative] = data
    bundle_files[f"nanopokerlib/{extension_filename}"] = binary_path.read_bytes()
    bundle_files["m09/ippo_iter002000_final_candidate.pt"] = candidate_path.read_bytes()
    bundle_files["m09/legacy_v1_anchor.pt"] = anchor_path.read_bytes()

    block = {
        "loader": "official",
        "frozen_candidate": {
            "path": "m09/ippo_iter002000_final_candidate.pt",
            "sha256": sha(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "algorithm": str(payload["algorithm"]),
            "architecture": str(payload.get("architecture")),
            "iteration": int(payload.get("iteration", 0)),
            "backend_identity_hash": str(identity["hash"]),
            "extension_sha256": str(implementation["extension_sha256"]),
            "extension_filename": extension_filename,
            "extension_path": recorded_path,
            "embedded_source_sha256": str(implementation["embedded_source_sha256"]),
            "python_implementation": str(implementation["python_implementation"]),
            "python_version": str(implementation["python_version"]),
            "python_cache_tag": str(implementation["python_cache_tag"]),
            "compiler": str(implementation["compiler"]),
            "source": (
                "frozen M09 final-holdout candidate (docs/marvel_migration/evidence/"
                "M09_final_holdout, evaluation_id m09-final-iter2000-v1)"
            ),
        },
        "runtime": {
            "source_root": SOURCE_ROOT,
            "binary": f"nanopokerlib/{extension_filename}",
            "cpp_sources": [
                {
                    "manifest_path": relative,
                    "payload_path": relative,
                    "sha256": sha_bytes(sources[relative]),
                    "bytes": len(sources[relative]),
                }
                for relative in NATIVE_SOURCE_PATHS
            ],
        },
        "legacy_anchor": {
            "path": "m09/legacy_v1_anchor.pt",
            "sha256": sha(anchor_path),
            "bytes": anchor_path.stat().st_size,
            "protocol": "GD-ACTION-0.1|GD-ENV-0.1",
        },
    }
    return bundle_files, block


def build_plan(repo: Path, git_commit: str, m09_block: Mapping) -> dict:
    plan = {
        "schema": PLAN_SCHEMA,
        "stage": "gate_a",
        "prompt": PROMPT_PATH,
        "source": {"git_commit": git_commit, "branch": "dmc-v1"},
        "config": {"path": CONFIG_PATH, "sha256": sha_bytes((repo / CONFIG_PATH).read_bytes())},
        "topology": dict(TOPOLOGY),
        "budgets": dict(BUDGETS),
        "schedule": json.loads(json.dumps(SCHEDULE)),
        "gates": json.loads(json.dumps(GATES)),
        "boundaries": dict(BOUNDARIES),
        "env_driver": json.loads(json.dumps(ENV_DRIVER)),
        "resume": dict(RESUME),
        "m09_lane": json.loads(json.dumps(m09_block)),
        "runtime": {"source_root": SOURCE_ROOT, "run_root": RUN_ROOT},
        "deviations": list(DEVIATIONS),
    }
    plan["plan_hash"] = canonical({key: value for key, value in plan.items() if key != "plan_hash"})
    return plan


BOOTSTRAP = r'''from pathlib import Path
import base64, hashlib, io, json, os, platform, shutil, sys, time, traceback, zipfile
STARTED = time.monotonic()
PAYLOAD_SHA256 = __PAYLOAD_SHA__
MANIFEST_HASH = __MANIFEST_SHA__
PLAN_HASH = __PLAN_SHA__
RUN_ROOT = Path('/kaggle/working/dmc_gate_a_v1')
SOURCE_ROOT = Path('/kaggle/working/guandan-marvel-source')


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


print('[DMC-GATE-A] one payload: parallel DMC training + four-lane greedy evaluation; no upload from code', flush=True)
try:
    if RUN_ROOT.exists():
        raise RuntimeError('RUN_ROOT already exists; download the old output and use a fresh session')
    if SOURCE_ROOT.exists():
        raise RuntimeError('SOURCE_ROOT already exists; the m09 runtime identity needs a pristine path')
    input_dir = RUN_ROOT / 'input'
    candidates = list(Path('/kaggle/input').rglob('guandan_serial_compare_payload_v1.bin'))
    matches = [p for p in candidates if p.is_file() and not p.is_symlink() and p.stat().st_size <= 256 * 1024**2 and sha(p) == PAYLOAD_SHA256]
    if len(matches) != 1:
        raise RuntimeError('expected exactly one matching DMC Gate A payload; do not mix old packages: ' + str(candidates))
    raw = matches[0].read_bytes()
    magic = b'GUANDAN-DMC-GATEA-V1\n'
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
    # Interpreter precheck *before* importing anything native: the frozen m09
    # backend identity pins the interpreter patch version.
    recorded = plan['m09_lane']['frozen_candidate']
    if platform.python_version() != recorded['python_version'] or sys.implementation.cache_tag != recorded['python_cache_tag']:
        raise RuntimeError(
            'interpreter mismatch for the frozen m09 runtime: this kernel is CPython '
            + platform.python_version() + ' (' + sys.implementation.cache_tag + '), the checkpoint '
            + 'records ' + recorded['python_version'] + ' (' + recorded['python_cache_tag'] + '); '
            + 'the official loader cannot reproduce the backend identity here'
        )
    driver_root, source_root = RUN_ROOT / 'driver', SOURCE_ROOT
    for archive_name, root, key in (('driver.zip', driver_root, 'driver_files'), ('overlay_source.zip', source_root, 'overlay_files'), ('m09_bundle.zip', source_root, 'm09_files')):
        rows = {row['path']: row for row in meta[key]}
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
    sys.path.insert(0, str(driver_root))
    sys.path.insert(0, str(source_root))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(driver_root), str(source_root)])
    os.environ['PYTHONUNBUFFERED'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(index) for index in range(2)):
        raise RuntimeError('Gate A requires exactly two T4 GPUs, no fallback')
    from dmc_gate_a.runner import run_notebook
    result = run_notebook(dataset=input_dir, run_root=RUN_ROOT, clock=lambda: time.monotonic() - STARTED)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    from IPython.display import FileLink, display
    display(FileLink(str(RUN_ROOT / 'guandan_dmc_gate_a_results.zip')))
    if result.get('gates', {}).get('overall_verdict') != 'PASS':
        print('[DMC-GATE-A] verdict is not PASS; see artifacts/RESULT.json — archive this session, do not blindly rerun', flush=True)
except BaseException as error:
    diagnostics = RUN_ROOT.parent / (RUN_ROOT.name + '_bootstrap') if RUN_ROOT.parent.exists() else Path('/kaggle/working/dmc_gate_a_bootstrap')
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


def build_delivery(repo: Path, output: Path, *, git_commit: str, m09_source: Path) -> dict:
    repo = Path(repo).resolve()
    output = Path(output).resolve()
    staging = output.with_name(output.name + "_build")
    if output.exists() or staging.exists():
        raise FileExistsError("never overwrite an old delivery")
    staging.mkdir(parents=True)

    config_bytes = (repo / CONFIG_PATH).read_bytes()
    overlay = collect_overlay(repo)
    driver = collect_driver(repo)
    m09_files, m09_block = collect_m09_bundle(m09_source)
    plan = build_plan(repo, git_commit, m09_block)

    payload_members = {
        "plan.json": json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "gate_a_config.json": config_bytes,
        "overlay_source.zip": zip_bytes(overlay),
        "driver.zip": zip_bytes(driver),
        "m09_bundle.zip": zip_bytes(m09_files),
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
        "m09_files": [
            {"path": name, "bytes": len(data), "sha256": sha_bytes(data)}
            for name, data in sorted(m09_files.items())
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
                "# DMC Gate A：并行训练 + 四臂 ε=0 评估（一次 Run All）\n",
                "Dataset 只上传 `" + PAYLOAD + "`（管线传输常量名，内容为 DMC 交付），GPU 选 T4×2，关闭 Internet。\n",
                "预算 31,800s：训练 28,800s（collector cuda:1 / learner cuda:0，cycle 边界安全停止，每 2,048 局 checkpoint），"
                "其余覆盖 rule / m09-iter2000 h2h / random / legacy 四臂各 512 局。\n",
                "m09 车道在训练前先做身份预检（官方加载器 + 冻结 cpp runtime）；门槛在 plan.json 中预声明并随包哈希；"
                "无 FableDan、无 final holdout、无 belief/对手池。\n",
                "跨 session 续训：把上一次输出 zip 里的 `dmc_gate_a_resume.pt` 作为新 Dataset 上传，本 notebook 自动发现并 resume。\n",
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
        "m09_files": len(m09_files),
        "m09_candidate_sha256": plan["m09_lane"]["frozen_candidate"]["sha256"],
        "budgets": dict(BUDGETS),
        "topology": dict(TOPOLOGY),
        "boundaries": dict(BOUNDARIES),
        "kaggle_execution": "NOT_RUN",
        "delivery_verify": "NOT_RUN",
    }
    (final_dir / "DELIVERY_VERIFICATION.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    readme = repo / PROMPT_PATH
    if readme.is_file():
        (final_dir / "STAGE_PROMPT.md").write_bytes(readme.read_bytes())
    sums = {
        "schema": "guandan-dmc-gate-a-sha256sums-v1",
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
    parser.add_argument("--m09-source", type=Path, required=True,
                        help="frozen m09 dataset directory (checkpoint + legacy anchor + runtime/ + source zip)")
    parser.add_argument("--git-commit", default=None, help="source commit to pin (default: git rev-parse HEAD)")
    args = parser.parse_args(argv)
    commit = args.git_commit if args.git_commit is not None else current_git_commit(REPO)
    verification = build_delivery(REPO, args.output, git_commit=commit, m09_source=args.m09_source)
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
