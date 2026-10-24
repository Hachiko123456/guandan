# SPDX-License-Identifier: MIT
"""Build the single opaque Dataset payload + Notebook for the serial comparison.

The payload intentionally contains:

* the byte-identical original source archive and Linux native (the frozen model
  artifacts bind this exact backend identity);
* a Python-only overlay of *this workspace's* ``guandan``/``guandan_marvel``
  modules plus the new ``tools/serial_compare`` driver, byte-identical to the
  files at build time;
* the three immutable policy artifacts (375 start, 250 pool/anchor, 500 probe);
* the deterministic deal manifests for the preflight/development/confirmation
  namespaces and the frozen plan.

The overlay never contains ``.cpp``/``.hpp``/native binaries, so the original
native's embedded source digest still matches the frozen C++ sources.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Mapping

from .common import (
    PlanError, atomic_json, build_deal_manifest, build_plan, canonical, read_json, sha,
    validate_deal_manifest, validate_plan,
)

PAYLOAD = "guandan_serial_compare_payload_v1.bin"
NOTEBOOK = "guandan_serial_compare_v1.ipynb"
MAGIC = b"GUANDAN-SERIALCOMPARE-V1\n"
MAX_PAYLOAD_BYTES = 256 * 1024**2

SOURCE_IDENTITY = {
    "archive_sha256": "5afaf81a5e76e70e56c6f871898313df4962fabe3f9cef0818931a2c53b15a2a",
    "content_manifest_hash": "6574388f92df8457731215f1e08707e5b5b9368bb1d46c165bb3aa2be2997c2a",
    "git_commit": "fa4a7af3227dd8d76663bd9cf2b0e3a659d70373",
    "core_source_hash": "0492d679f3ccea7c6388676463f6b266e178201498285ecc84e84de20d1657ef",
    "runtime_path": "/kaggle/working/guandan_t4x2_combo_v1/source/guandan-marvel-source",
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "backend_hash": "6e6875fdd2f754a6fbd5eef0b6cb540d29fa5456727d9eaab34cedc8852167ba",
    "native_name": "_guandan_cpp.cpython-313-x86_64-linux-gnu.so",
    "native_sha256": "240bc25c523f33f85144af03cb6e9ec2da6f5cbb5a6b05a0c37ea1e8a7262a72",
}
MODEL_FILES = {
    "p375": ("model_iter000375.pt", "d523e557975ec10b4e6e8347d49a0a9226c188a30bb22cf9088bb78ac16374ee"),
    "p250": ("anchor_iter250.pt", "965c60603b6aa8b5fe107d00c1a998e1389b64d104c4610386786b9487849f5f"),
    "p500": ("model_iter000500.pt", "2139e12728e64c6dba63de34e9a2673b9fc15b8f4251c3e019edb8c2cb80ba42"),
}
PAYLOAD_MODEL_NAMES = {"p375": "policy_375.pt", "p250": "anchor_250.pt", "p500": "policy_500.pt"}
OVERLAY_ROOTS = ("guandan", "guandan_marvel", "tools/serial_compare")
FORBIDDEN_OVERLAY_SUFFIXES = {
    ".so", ".pyd", ".dll", ".dylib", ".o", ".obj", ".a", ".lib", ".cpp", ".cc", ".c",
    ".hpp", ".h", ".pyx", ".pxd", ".cu",
}


BUNDLE_VERIFY_SOURCE = r'''
def verify_input_bundle(input_dir, expected_manifest_hash, canonical_fn, sha_fn):
    """Fail closed on a tampered/mismatched input (payload *or* resume bundle)."""
    manifest_path = input_dir / 'delivery_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('input bundle has no delivery_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    body = {key: value for key, value in manifest.items() if key != 'manifest_hash'}
    if manifest.get('manifest_hash') != canonical_fn(body):
        raise RuntimeError('input delivery manifest hash is inconsistent')
    if expected_manifest_hash is not None and manifest.get('manifest_hash') != expected_manifest_hash:
        raise RuntimeError('input bundle belongs to a different payload/manifest version')
    declared = {row['path']: row for row in manifest.get('files', [])}
    present = {path.name for path in input_dir.iterdir() if path.is_file()} - {'delivery_manifest.json'}
    if present != set(declared):
        raise RuntimeError('input bundle file list differs from its manifest')
    for name, row in declared.items():
        path = input_dir / name
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha_fn(path) != row['sha256']:
            raise RuntimeError('input bundle file changed: ' + name)
    for archive_name, key in (('overlay_source.zip', 'overlay_files'), ('driver.zip', 'driver_files')):
        rows = {row['path']: row for row in manifest.get(key, [])}
        archive = input_dir / archive_name
        if not archive.is_file():
            raise RuntimeError('input bundle is missing ' + archive_name)
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.namelist()
            if sorted(members) != sorted(rows):
                raise RuntimeError('input ' + archive_name + ' member list changed')
            for name in members:
                data = bundle.read(name)
                row = rows[name]
                if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                    raise RuntimeError('input ' + archive_name + ' member changed: ' + name)
    return manifest


def verify_member_tree(root, rows, label):
    """Compare an already-extracted directory with the manifest's member list.

    Used for ``RUN_ROOT/driver-*`` and ``RUN_ROOT/overlay-*`` so a resume root
    that already contains these directories cannot bypass the archive-hash
    check by simply existing.
    """
    declared = {row['path']: row for row in rows}
    present = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    if present != set(declared):
        missing = sorted(set(declared) - present)
        extra = sorted(present - set(declared))
        raise RuntimeError(label + ' directory file set differs from the manifest: missing='
                           + repr(missing[:5]) + ' extra=' + repr(extra[:5]))
    for name, row in declared.items():
        data = (root / name).read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise RuntimeError(label + ' directory member changed: ' + name)
    return True
'''

_BUNDLE_GLOBALS: dict[str, object] = {"json": json, "zipfile": zipfile, "hashlib": hashlib}
exec(BUNDLE_VERIFY_SOURCE, _BUNDLE_GLOBALS)  # noqa: S102 - our own static source
verify_input_bundle = _BUNDLE_GLOBALS["verify_input_bundle"]  # type: ignore[assignment]
verify_member_tree = _BUNDLE_GLOBALS["verify_member_tree"]  # type: ignore[assignment]


def zip_bytes(files: Mapping[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    return stream.getvalue()


def collect_overlay(repo: str | Path) -> dict[str, bytes]:
    repo = Path(repo).resolve()
    files: dict[str, bytes] = {}
    for root in OVERLAY_ROOTS:
        base = repo / root
        if not base.is_dir():
            raise PlanError(f"overlay root is missing: {base}")
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(repo).as_posix()
            data = path.read_bytes()
            ast.parse(data.decode("utf-8-sig"))
            files[relative] = data
    if not files:
        raise PlanError("the overlay is empty")
    for path in repo.joinpath("guandan").rglob("*"):
        if path.is_file() and path.suffix in FORBIDDEN_OVERLAY_SUFFIXES:
            raise PlanError(f"unexpected binary/native file in the overlay source tree: {path}")
    for name in files:
        if Path(name).suffix in FORBIDDEN_OVERLAY_SUFFIXES:
            raise PlanError(f"overlay may not contain native sources: {name}")
    return files


def collect_driver(repo: str | Path) -> dict[str, bytes]:
    repo = Path(repo).resolve()
    files: dict[str, bytes] = {}
    for path in sorted((repo / "tools/serial_compare").glob("*.py")):
        data = path.read_bytes()
        ast.parse(data.decode("utf-8-sig"))
        files[f"serial_compare/{path.name}"] = data
    if not files:
        raise PlanError("the driver is empty")
    return files


def _model_identity(path: Path, expected_sha: str, *, name: str) -> dict[str, object]:
    import torch

    if not path.is_file():
        raise PlanError(f"missing model {name!r}: {path}")
    actual = sha(path)
    if actual != expected_sha:
        raise PlanError(f"model {name!r} hash mismatch: {actual} != {expected_sha}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "guandan-marvel-evaluation-checkpoint-v1":
        raise PlanError(f"model {name!r} is not an evaluation-only checkpoint")
    if payload.get("training_resume_allowed") is not False:
        raise PlanError(f"model {name!r} claims training resume; it is not a frozen policy")
    if payload.get("backend_identity", {}).get("hash") != SOURCE_IDENTITY["backend_hash"]:
        raise PlanError(f"model {name!r} was produced by a different backend identity")
    if str(payload.get("source_hash")) != SOURCE_IDENTITY["core_source_hash"]:
        raise PlanError(f"model {name!r} was produced by a different core source")
    state = payload.get("agent_state_dict") or {}
    if not state or any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise PlanError(f"model {name!r} has a non-finite state dict")
    resolved = payload.get("resolved_config") or {}
    if canonical(resolved) != payload.get("resolved_config_hash"):
        raise PlanError(f"model {name!r} resolved_config hash is inconsistent")
    ppo = dict(resolved.get("ppo") or {})
    if ppo.get("training_semantics_profile") != "macro_decision_v1":
        raise PlanError(f"model {name!r} was not trained with macro_decision_v1")
    if ppo.get("policy_regularizer_mode") != "macro_normalized_kl_uniform_v1":
        raise PlanError(f"model {name!r} was not trained with macro_normalized_kl_uniform_v1")
    return {
        "sha256": actual,
        "iteration": int(payload.get("iteration", -1)),
        "bytes": path.stat().st_size,
        "architecture": str(payload.get("architecture")),
        "protocol_version": str(payload.get("protocol_version")),
        "backend": str(payload.get("backend")),
        "backend_identity_hash": str(payload["backend_identity"]["hash"]),
        "resolved_config_hash": str(payload.get("resolved_config_hash")),
        "source_hash": str(payload.get("source_hash")),
        "training_resume_allowed": False,
        "model": dict(resolved.get("model") or {}),
        "ppo": {
            "training_semantics_profile": ppo.get("training_semantics_profile"),
            "policy_regularizer_mode": ppo.get("policy_regularizer_mode"),
            "base_lr": ppo.get("base_lr"),
            "base_clip_coef": ppo.get("base_clip_coef"),
            "base_reg_coef": ppo.get("base_reg_coef"),
        },
        "ntp_weight": float(resolved.get("ntp_weight", 0.0) or 0.0),
        "state_hash": canonical({
            key: hashlib.sha256(
                tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest() for key, tensor in sorted(state.items())
        }),
    }


def extract_frozen_source(v4_payload: Path) -> bytes:
    raw = Path(v4_payload).read_bytes()
    if sha(v4_payload) != "0b018993dfa09528ed6c5dd225a825d9d7867a2f1df42963bdb1344ed5dd7347":
        raise PlanError("wrong original v4 payload input")
    if not raw.startswith(b"GUANDAN-PILOT-TRANSPORT-V3\n"):
        raise PlanError("wrong original transport version")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw.split(b"\n", 1)[1].strip(), validate=True))) as bundle:
        data = bundle.read("guandan-marvel-source.zip")
    if hashlib.sha256(data).hexdigest() != SOURCE_IDENTITY["archive_sha256"]:
        raise PlanError("the frozen source archive hash changed")
    return data


def extract_native(base_recovery: Path) -> bytes:
    with zipfile.ZipFile(base_recovery) as bundle:
        record = json.loads(bundle.read("recovery_manifest.json"))
        binary = record.get("runtime_binary") or {}
        data = bundle.read(binary["archive_path"])
    if hashlib.sha256(data).hexdigest() != SOURCE_IDENTITY["native_sha256"]:
        raise PlanError("the frozen Linux native hash changed")
    if binary.get("restore_relative_path") != f"nanopokerlib/{SOURCE_IDENTITY['native_name']}":
        raise PlanError("unexpected native restore path in the original recovery")
    return data


BOOTSTRAP = r'''from pathlib import Path
import base64, hashlib, io, json, os, platform, shutil, sys, time, traceback, zipfile
STARTED = time.monotonic()
EXPECTED_PAYLOAD_SHA256 = __PAYLOAD_SHA__
EXPECTED_MANIFEST_HASH = __MANIFEST_SHA__
PAYLOAD_PATH = None  # optional exact mounted path; normally leave None
RESUME_RUN_ROOT = None  # optional advanced: an existing RUN_ROOT to resume
RUN_ROOT = Path('/kaggle/working/guandan_serial_compare_v1')
RUNTIME_SOURCE_PATH = __RUNTIME_PATH__
EXPECTED_PYTHON = __EXPECTED_PYTHON__
EXPECTED_BACKEND_HASH = __EXPECTED_BACKEND_HASH__


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


__BUNDLE_VERIFY__
print('[SERIAL-COMPARE] one payload, two isolated arms, serial; no Kaggle upload from code', flush=True)
try:
    if RESUME_RUN_ROOT is not None:
        RUN_ROOT = Path(RESUME_RUN_ROOT)
        if not (RUN_ROOT / 'input' / 'plan.json').is_file():
            raise RuntimeError('RESUME_RUN_ROOT does not contain a serial-compare run')
    elif RUN_ROOT.exists():
        raise RuntimeError('RUN_ROOT already exists; download the old output and use a new RUN_ROOT/fresh session')
    input_dir = RUN_ROOT / 'input'
    if not (input_dir / 'plan.json').is_file():
        candidates = [Path(PAYLOAD_PATH)] if PAYLOAD_PATH is not None else list(Path('/kaggle/input').rglob('guandan_serial_compare_payload_v1.bin'))
        matches = [p for p in candidates if p.is_file() and not p.is_symlink() and p.stat().st_size <= 256 * 1024**2 and sha(p) == EXPECTED_PAYLOAD_SHA256]
        if len(matches) != 1:
            raise RuntimeError('expected exactly one matching serial-compare payload; do not mix old packages: ' + str(candidates))
        raw = matches[0].read_bytes()
        magic = b'GUANDAN-SERIALCOMPARE-V1\n'
        if not raw.startswith(magic):
            raise RuntimeError('wrong opaque transport version')
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw[len(magic):].strip(), validate=True))) as bundle:
            meta = json.loads(bundle.read('delivery_manifest.json'))
            if meta['manifest_hash'] != EXPECTED_MANIFEST_HASH or canonical({k: v for k, v in meta.items() if k != 'manifest_hash'}) != meta['manifest_hash']:
                raise RuntimeError('Notebook/payload manifest mismatch')
            if {item.filename for item in bundle.infolist()} != {'delivery_manifest.json', *(row['path'] for row in meta['files'])}:
                raise RuntimeError('undeclared payload files')
            for row in meta['files']:
                data = bundle.read(row['path'])
                if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                    raise RuntimeError('payload hash mismatch: ' + row['path'])
            unpack(bundle, input_dir, flat=True)
    # Applies to both a freshly unpacked payload and a RESUME_RUN_ROOT bundle:
    # the input tree, the inner manifest and every overlay/driver member must
    # still hash to what the delivery declared.
    bundle_manifest = verify_input_bundle(input_dir, EXPECTED_MANIFEST_HASH, canonical, sha)
    plan = json.loads((input_dir / 'plan.json').read_text(encoding='utf-8'))
    if sha(input_dir / 'plan.json') and canonical({k: v for k, v in plan.items() if k != 'plan_hash'}) != plan['plan_hash']:
        raise RuntimeError('plan hash is inconsistent')

    # Versioned by the delivery manifest hash: a stale/forged ``driver`` or
    # ``overlay`` directory in a reused RUN_ROOT is never trusted.  An existing
    # versioned directory is still compared member-by-member before use.
    driver = RUN_ROOT / ('driver-' + EXPECTED_MANIFEST_HASH[:16])
    if driver.is_dir():
        verify_member_tree(driver, bundle_manifest['driver_files'], 'driver')
    else:
        with zipfile.ZipFile(input_dir / 'driver.zip') as bundle:
            unpack(bundle, driver, limit=8 * 1024**2)
        verify_member_tree(driver, bundle_manifest['driver_files'], 'driver')

    staged = RUN_ROOT / 'source_staging'
    if not (staged / 'guandan-marvel-source').is_dir():
        with zipfile.ZipFile(input_dir / 'frozen_source.zip') as bundle:
            unpack(bundle, staged, limit=256 * 1024**2)
    source = staged / 'guandan-marvel-source'
    manifest = json.loads((source / 'source_manifest.json').read_text(encoding='utf-8-sig'))
    if canonical({k: v for k, v in manifest.items() if k != 'manifest_hash'}) != manifest['manifest_hash']:
        raise RuntimeError('frozen source manifest hash is inconsistent')
    if manifest['manifest_hash'] != plan['source']['content_manifest_hash']:
        raise RuntimeError('frozen source content differs from the plan')
    for row in manifest['files']:
        path = (source / row['path']).resolve()
        if not path.is_relative_to(source.resolve()) or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError('frozen source member changed: ' + row['path'])

    overlay_root = RUN_ROOT / ('overlay-' + EXPECTED_MANIFEST_HASH[:16])
    if overlay_root.is_dir():
        verify_member_tree(overlay_root, bundle_manifest['overlay_files'], 'overlay')
    else:
        with zipfile.ZipFile(input_dir / 'overlay_source.zip') as bundle:
            unpack(bundle, overlay_root, limit=64 * 1024**2)
        verify_member_tree(overlay_root, bundle_manifest['overlay_files'], 'overlay')
    for path in overlay_root.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix in {'.cpp', '.hpp', '.h', '.so', '.pyd', '.pyx'}:
            raise RuntimeError('overlay must stay Python-only: ' + str(path))

    canonical_source = Path(RUNTIME_SOURCE_PATH)
    if canonical_source.resolve() != canonical_source:
        raise RuntimeError('the original absolute source path is redirected')
    if canonical_source.exists():
        canonical_manifest = canonical_source / 'source_manifest.json'
        if not canonical_manifest.is_file() or sha(canonical_manifest) != sha(source / 'source_manifest.json'):
            raise RuntimeError('conflicting original source checkout')
    else:
        canonical_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, canonical_source)
    # The Python-only overlay is applied to the *final* source every session so a
    # reused checkout can never silently run the frozen (pre-overlay) modules.
    for path in overlay_root.rglob('*'):
        if not path.is_file():
            continue
        target = canonical_source / path.relative_to(overlay_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    source = canonical_source
    final_manifest = json.loads((source / 'source_manifest.json').read_text(encoding='utf-8-sig'))
    if final_manifest['manifest_hash'] != plan['source']['content_manifest_hash']:
        raise RuntimeError('overlay changed the frozen source manifest')
    native_input = input_dir / 'frozen_native.so'
    if sha(native_input) != plan['source']['native_sha256']:
        raise RuntimeError('frozen native hash changed')
    if platform.system() != 'Linux' or platform.python_version() != EXPECTED_PYTHON:
        raise RuntimeError('original full-state runtime requires Linux/Python ' + EXPECTED_PYTHON + '; got ' + platform.system() + '/' + platform.python_version())
    native_target = source / 'nanopokerlib' / path_placeholder
    native_target.parent.mkdir(parents=True, exist_ok=True)
    if native_target.exists() and sha(native_target) != plan['source']['native_sha256']:
        raise RuntimeError('conflicting original native; never overwrite a loaded shared library')
    if not native_target.exists():
        shutil.copyfile(native_input, native_target)

    sys.path.insert(0, str(driver))
    sys.path.insert(0, str(source))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(driver), str(source)])
    os.environ['GUANDAN_MARVEL_SOURCE_COMMIT'] = plan['source']['git_commit']
    os.environ['PYTHONUNBUFFERED'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from guandan_marvel import cpp_game
    from guandan_marvel.training.checkpoints import build_backend_identity
    backend = build_backend_identity(cpp_game)
    if backend['hash'] != EXPECTED_BACKEND_HASH:
        raise RuntimeError('original backend identity mismatch before any training; refusing to rewrite hashes')
    import torch
    if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(index) for index in (0, 1)):
        raise RuntimeError('this package requires exactly two T4 GPUs, no fallback')
    if not torch.distributed.is_nccl_available() or not torch.distributed.is_gloo_available():
        raise RuntimeError('required NCCL/Gloo backends are unavailable')

    from serial_compare.pipeline import run_notebook
    result = run_notebook(dataset=input_dir, run_root=RUN_ROOT, source=source, clock=lambda: time.monotonic() - STARTED)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    from IPython.display import FileLink, HTML, display
    display(HTML((RUN_ROOT / 'artifacts' / 'report.html').read_text(encoding='utf8')))
    archive = RUN_ROOT / 'guandan_serial_compare_results.zip'
    display(FileLink(str(archive)))
    if result['status'] == 'complete':
        pass
    elif result['status'] == 'incomplete':
        print('[SERIAL-COMPARE] INCOMPLETE: verified progress preserved; resume with the same payload and RESUME_RUN_ROOT', flush=True)
    else:
        raise RuntimeError('Stopped safely at ' + str(result.get('failed_phase')) + ': ' + str(result.get('error')))
except BaseException as error:
    diagnostics = (RUN_ROOT.parent / (RUN_ROOT.name + '_bootstrap')) if RUN_ROOT is not None else Path('/kaggle/working/serial_compare_bootstrap')
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / 'bootstrap_error.json').write_text(json.dumps({'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc(), 'training_updates_this_session': None}, ensure_ascii=False, indent=2), encoding='utf8')
    raise
'''


def build(
    repo: str | Path,
    output: str | Path,
    *,
    model_dir: str | Path,
    base_inputs: str | Path,
    validation_record: str | Path | None = None,
) -> dict[str, object]:
    repo = Path(repo).resolve()
    output = Path(output).resolve()
    staging = output.with_name(output.name + "_build")
    if output.exists() or staging.exists():
        raise FileExistsError("never overwrite an old delivery")
    staging.mkdir(parents=True)
    dataset = staging / "dataset"
    dataset.mkdir()
    model_dir = Path(model_dir).resolve()
    base_inputs = Path(base_inputs).resolve()

    models: dict[str, dict[str, object]] = {}
    for name, (filename, expected) in MODEL_FILES.items():
        models[name] = _model_identity(model_dir / filename, expected, name=name)
        (dataset / PAYLOAD_MODEL_NAMES[name]).write_bytes((model_dir / filename).read_bytes())

    source_archive = extract_frozen_source(base_inputs / "v4_payload.bin")
    (dataset / "frozen_source.zip").write_bytes(source_archive)
    native = extract_native(base_inputs / "base_recovery.zip")
    (dataset / "frozen_native.so").write_bytes(native)

    overlay = collect_overlay(repo)
    (dataset / "overlay_source.zip").write_bytes(zip_bytes(overlay))
    driver = collect_driver(repo)
    (dataset / "driver.zip").write_bytes(zip_bytes(driver))

    manifests = {
        domain: build_deal_manifest(domain)
        for domain in ("preflight", "development", "confirmation")
    }
    for domain, manifest in manifests.items():
        validate_deal_manifest(manifest, expected_domain=domain)
        atomic_json(dataset / f"deals_{domain}.json", manifest)

    plan = build_plan(source=SOURCE_IDENTITY, models=models)
    validate_plan(plan)
    atomic_json(dataset / "plan.json", plan)

    readme = repo / "docs/t4x2_serial_compare.md"
    if readme.is_file():
        shutil.copy2(readme, dataset / "README_UPLOAD.md")
    if validation_record is not None:
        validation = read_json(validation_record)
        for row in validation.get("source_files", []):
            if sha(repo / row["path"]) != row["sha256"]:
                raise PlanError("stale local validation: " + row["path"])
        atomic_json(dataset / "LOCAL_VALIDATION.json", validation)

    manifest = {
        "schema": "guandan-serial-compare-delivery-manifest-v1",
        "plan_hash": plan["plan_hash"],
        "overlay_files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(overlay.items())
        ],
        "driver_files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(driver.items())
        ],
        "deal_manifests": {domain: manifest_item["hash"] for domain, manifest_item in manifests.items()},
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(dataset.iterdir())
        ],
        "fabledan": "NOT_RUN",
        "final_holdout": "NOT_TOUCHED",
        "formal_production_promotion": False,
    }
    manifest["manifest_hash"] = canonical(manifest)
    atomic_json(dataset / "delivery_manifest.json", manifest)

    payload = MAGIC + base64.b64encode(zip_bytes({path.name: path.read_bytes() for path in dataset.iterdir()})) + b"\n"
    if len(payload) > MAX_PAYLOAD_BYTES or zipfile.is_zipfile(io.BytesIO(payload)):
        raise PlanError("invalid/oversize opaque upload")
    final_dir = staging / "delivery"
    final_dir.mkdir()
    (final_dir / PAYLOAD).write_bytes(payload)

    code = (
        BOOTSTRAP
        .replace("__BUNDLE_VERIFY__", BUNDLE_VERIFY_SOURCE)
        .replace("__PAYLOAD_SHA__", repr(sha(final_dir / PAYLOAD)))
        .replace("__MANIFEST_SHA__", repr(manifest["manifest_hash"]))
        .replace("__RUNTIME_PATH__", repr(SOURCE_IDENTITY["runtime_path"]))
        .replace("__EXPECTED_PYTHON__", repr(SOURCE_IDENTITY["python"]))
        .replace("__EXPECTED_BACKEND_HASH__", repr(SOURCE_IDENTITY["backend_hash"]))
        .replace("path_placeholder", repr(SOURCE_IDENTITY["native_name"]))
    )
    ast.parse(code)
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
                "# 一次上传：双配方串行训练与统一评估\n",
                "Dataset 只上传 `" + PAYLOAD + "`，GPU 选 T4×2，Internet 可关闭，Run All。\n",
                "两条隔离训练臂从同一 375 策略独立开始：control 保留 NTP/self-play，candidate 关闭 NTP、启用冻结 250/375 池并带 target-KL。\n",
                "两臂预检都通过后才开始正式训练；随后统一 256 场主确认 + 448 场开发辅助，共 704 场。\n",
                "总预算 14400 秒；技术失败与统计 INCONCLUSIVE 分开报告。无 FableDan、无 final holdout、formal promotion=false。\n",
            ]},
            {"id": "run-all", "cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": code.splitlines(keepends=True)},
        ],
    }
    atomic_json(final_dir / NOTEBOOK, notebook)

    shutil.copy2(dataset / "plan.json", final_dir / "plan.json")
    shutil.copy2(dataset / "delivery_manifest.json", final_dir / "DELIVERY_MANIFEST.json")
    if (dataset / "README_UPLOAD.md").is_file():
        shutil.copy2(dataset / "README_UPLOAD.md", final_dir / "README_UPLOAD.md")
    if (dataset / "LOCAL_VALIDATION.json").is_file():
        shutil.copy2(dataset / "LOCAL_VALIDATION.json", final_dir / "LOCAL_VALIDATION.json")

    sums = {
        "schema": "guandan-serial-compare-sha256sums-v1",
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(final_dir.iterdir()) if path.is_file()
        ],
    }
    sums["hash"] = canonical(sums)
    atomic_json(final_dir / "SHA256SUMS", sums)

    verification = {
        "schema": "guandan-serial-compare-delivery-verification-v1",
        "status": "BUILT_TARGET_PREFLIGHT_PENDING",
        "payload": PAYLOAD,
        "payload_bytes": len(payload),
        "payload_sha256": sha(final_dir / PAYLOAD),
        "notebook": NOTEBOOK,
        "notebook_sha256": sha(final_dir / NOTEBOOK),
        "manifest_hash": manifest["manifest_hash"],
        "plan_hash": plan["plan_hash"],
        "overlay_files": len(overlay),
        "driver_files": len(driver),
        "source_archive_sha256": SOURCE_IDENTITY["archive_sha256"],
        "native_sha256": SOURCE_IDENTITY["native_sha256"],
        "fabledan": "NOT_RUN",
        "final_holdout": "NOT_TOUCHED",
        "formal_production_promotion": False,
        "kaggle_execution": "NOT_RUN",
        "target_dual_t4_validation": "NOT_RUN",
    }
    atomic_json(final_dir / "DELIVERY_VERIFICATION.json", verification)
    final_dir.rename(output)
    staging.rename(output.with_name(output.name + "_build"))
    return verification


def decode_payload(payload_path: str | Path, destination: str | Path) -> dict[str, object]:
    """Decode the *actual delivered* payload into an isolated directory."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    raw = Path(payload_path).read_bytes()
    if not raw.startswith(MAGIC):
        raise PlanError("wrong opaque transport version")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw[len(MAGIC):].strip(), validate=True))) as bundle:
        members = bundle.infolist()
        if len({item.filename.casefold() for item in members}) != len(members):
            raise PlanError("duplicate payload members")
        for item in members:
            if item.orig_filename != item.filename or "/" in item.filename or "\\" in item.filename:
                raise PlanError("non-flat payload member")
        bundle.extractall(destination)
    return {"sha256": sha(payload_path), "bytes": len(raw)}


__all__ = [
    "BUNDLE_VERIFY_SOURCE", "MAGIC", "MODEL_FILES", "NOTEBOOK", "OVERLAY_ROOTS", "PAYLOAD",
    "SOURCE_IDENTITY", "build", "collect_driver", "collect_overlay", "decode_payload",
    "extract_frozen_source", "extract_native", "verify_input_bundle", "verify_member_tree",
    "zip_bytes",
]
