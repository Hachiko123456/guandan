# SPDX-License-Identifier: MIT
"""The frozen m09-iter2000 lane: identity pinning + the official loader.

``docs/dmc_prompts/06_gate_a_kaggle.md`` requires the head-to-head lane to use
the **frozen m09 candidate** (SHA256 ``af0df1ea…4ca9c8``) through the
**official loader**, not a local reimplementation of inference.

That is only possible if the runtime can reproduce the backend identity the
checkpoint recorded when it was written.  ``build_backend_identity`` hashes the
extension's SHA256, its build identity, *its absolute path*, the interpreter
version/cache tag and the four native source files — so the lane is
reproducible only when

* the exact ``_guandan_cpp`` binary is loaded (``extension_sha256``),
* it sits at the path recorded in the checkpoint (``extension_path``),
* the four ``nanopokerlib/cpp`` sources are byte-identical
  (``embedded_source_sha256``),
* and the interpreter is CPython 3.13.15 (``python_version`` /
  ``python_cache_tag``).

The delivery therefore ships the frozen binary + the exact native sources
(``m09/runtime`` and ``m09/cpp`` inside the payload), the notebook restores them
to the recorded source root, and this module *checks* every one of those facts
against the checkpoint's own recorded identity **before** training starts.  A
mismatch is fatal and loud: an h2h lane measured on a different runtime would
be a different experiment, and the Gate A gate table has no room for that.

Nothing here weakens a check: the actual load still goes through
:func:`guandan_marvel.evaluation.checkpoints.load_v2_policy` with all of its
validations intact.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping


class M09LaneUnavailable(RuntimeError):
    """The frozen m09 lane cannot be reproduced on this runtime."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_source_manifest(root: Path, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebuild ``nanopokerlib.build_identity``'s manifest from the payload.

    ``rows`` maps the *manifest* path (``nanopokerlib/cpp/...``) to the
    payload-relative file that carries those exact bytes, so the digest is
    computed over the real file contents while keeping the labels the
    checkpoint recorded.
    """
    files = []
    for row in rows:
        path = Path(root) / str(row["payload_path"])
        if not path.is_file():
            raise M09LaneUnavailable(f"native source is missing: {path}")
        data = path.read_bytes()
        files.append({
            "path": str(row["manifest_path"]),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    payload = {"schema": "guandan-native-build-identity-v1", "files": files}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def verify_bundle_files(dataset: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Check every shipped m09 file against the plan's pinned hashes."""
    dataset = Path(dataset)
    block = plan["m09_lane"]
    candidate = block["frozen_candidate"]
    runtime = block["runtime"]
    anchor = block["legacy_anchor"]
    checked: dict[str, str] = {}

    def _check(relative: str, expected: str, label: str) -> Path:
        path = dataset / relative
        if not path.is_file():
            raise M09LaneUnavailable(f"{label} is missing from the payload: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise M09LaneUnavailable(
                f"{label} hash mismatch: {relative} sha256={actual} != pinned {expected}"
            )
        checked[relative] = actual
        return path

    _check(candidate["path"], candidate["sha256"], "frozen m09 candidate")
    _check(anchor["path"], anchor["sha256"], "frozen legacy anchor")
    binary = _check(runtime["binary"], candidate["extension_sha256"], "m09 native runtime")
    for row in runtime["cpp_sources"]:
        _check(str(row["payload_path"]), str(row["sha256"]), "m09 native source")
    manifest = native_source_manifest(dataset, runtime["cpp_sources"])
    if manifest["hash"] != candidate["embedded_source_sha256"]:
        raise M09LaneUnavailable(
            "shipped native sources do not reproduce the checkpoint's embedded "
            f"source hash: {manifest['hash']} != {candidate['embedded_source_sha256']}"
        )
    return {
        "checked_files": len(checked),
        "native_source_manifest_hash": manifest["hash"],
        "binary": str(binary.relative_to(dataset)),
        "binary_bytes": binary.stat().st_size,
    }


def _identity_diff(actual: Mapping[str, Any], recorded: Mapping[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key in (
        "extension_sha256", "extension_path", "embedded_source_sha256", "compiler",
        "cxx_standard", "python_implementation", "python_version", "python_cache_tag",
    ):
        if actual.get(key) != recorded.get(key):
            diff[key] = {"actual": actual.get(key), "recorded": recorded.get(key)}
    return diff


def check_runtime_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the live cpp build identity with the checkpoint's recorded one."""
    recorded = plan["m09_lane"]["frozen_candidate"]
    try:
        from guandan_marvel import cpp_game
        from guandan_marvel.training.checkpoints import build_backend_identity
    except Exception as error:  # native extension missing / wrong interpreter
        raise M09LaneUnavailable(
            "the native guandan runtime is not importable on this interpreter "
            f"(python {platform.python_version()}, cache tag "
            f"{getattr(sys.implementation, 'cache_tag', '?')}); the frozen m09 "
            f"checkpoint records python {recorded['python_version']} / "
            f"{recorded['python_cache_tag']} and extension "
            f"{recorded['extension_filename']}. Underlying error: "
            f"{type(error).__name__}: {error}"
        ) from None
    identity = build_backend_identity(cpp_game)
    implementation = dict(identity["implementation"])
    diff = _identity_diff(implementation, recorded)
    if identity["hash"] != recorded["backend_identity_hash"] or diff:
        raise M09LaneUnavailable(
            "the live cpp build identity does not match the frozen m09 checkpoint; "
            f"identity hash {identity['hash']} != {recorded['backend_identity_hash']}, "
            f"differing fields: {sorted(diff)} — details: "
            + json.dumps(diff, sort_keys=True)
        )
    return {
        "identity_hash": identity["hash"],
        "extension_path": implementation["extension_path"],
        "python_version": implementation["python_version"],
        "python_cache_tag": implementation["python_cache_tag"],
        "compiler": implementation["compiler"],
    }


def load_m09_artifact(dataset: Path, plan: Mapping[str, Any], *, device: str) -> Any:
    """Load the frozen candidate through the official evaluation loader."""
    from guandan_marvel.evaluation.checkpoints import load_v2_policy

    candidate = plan["m09_lane"]["frozen_candidate"]
    path = Path(dataset) / candidate["path"]
    try:
        artifact = load_v2_policy(
            path, device=device, expected_algorithm=str(candidate["algorithm"]),
        )
    except M09LaneUnavailable:
        raise
    except Exception as error:
        raise M09LaneUnavailable(
            "the official loader rejected the frozen m09 candidate: "
            f"{type(error).__name__}: {error}"
        ) from None
    if artifact.sha256 != candidate["sha256"]:
        raise M09LaneUnavailable(
            f"loaded artifact sha256 {artifact.sha256} != pinned {candidate['sha256']}"
        )
    if int(artifact.iteration) != int(candidate["iteration"]):
        raise M09LaneUnavailable(
            f"loaded artifact iteration {artifact.iteration} != pinned {candidate['iteration']}"
        )
    return artifact


def load_legacy_artifact(dataset: Path, plan: Mapping[str, Any]) -> Any:
    """Load the frozen V1 anchor through the official opponent loader."""
    from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent

    anchor = plan["m09_lane"]["legacy_anchor"]
    path = Path(dataset) / anchor["path"]
    artifact = load_legacy_v1_opponent(path)
    if artifact.sha256 != anchor["sha256"]:
        raise M09LaneUnavailable(
            f"loaded legacy anchor sha256 {artifact.sha256} != pinned {anchor['sha256']}"
        )
    return artifact


def precheck(
    dataset: Path,
    plan: Mapping[str, Any],
    *,
    device: str,
    loader=None,
    load_legacy: bool = True,
    warmup: bool = True,
) -> dict[str, Any]:
    """Fail-fast proof that the m09 lane can run on this runtime.

    Runs before training so an unreproducible runtime costs seconds, not the
    eight-hour session.  ``loader`` is an explicit test seam: production passes
    ``None`` and the official loader is used, and the record says which one ran.
    """
    files = verify_bundle_files(dataset, plan)
    identity = check_runtime_identity(plan)
    artifact = (loader or load_m09_artifact)(dataset, plan, device=device)
    record: dict[str, Any] = {
        "status": "READY",
        "loader": "official" if loader is None else "injected",
        "checkpoint_sha256": artifact.sha256,
        "checkpoint_iteration": int(artifact.iteration),
        "architecture": getattr(artifact, "architecture", None),
        "algorithm": getattr(artifact, "algorithm", None),
        "device": str(device),
        **files,
        **identity,
    }
    if warmup:
        record.update(_warmup(artifact, device=device))
    if load_legacy:
        legacy = load_legacy_artifact(dataset, plan)
        record["legacy_anchor"] = {
            "status": "READY",
            "sha256": legacy.sha256,
            "protocol_versions": dict(legacy.protocol_versions),
        }
    return record


def _warmup(artifact: Any, *, device: str) -> dict[str, Any]:
    import numpy as np
    import torch

    from guandan_marvel.evaluation.agents import V2PolicySession
    from guandan_marvel.game import EnvBatch

    session = V2PolicySession(artifact, seed=20261026, temperature=0.0, device=device)
    env = EnvBatch(1, auto_reset=False)
    step = env.reset(20261026)
    probabilities = session.observe_and_score(step)
    if not bool(torch.isfinite(probabilities).all()):
        raise M09LaneUnavailable("the m09 warmup produced non-finite probabilities")
    index = session.select_index(probabilities, step)
    row = np.asarray(step.available_actions.tokens[0, index], dtype=np.int64)
    step = env.step(row.reshape(1, -1))
    return {
        "warmup_steps": 1,
        "warmup_probability_sum": float(probabilities.sum().item()),
        "warmup_selected_token": int(row[5]),
        "warmup_terminal": bool(step.is_terminal[0]),
    }


__all__ = [
    "M09LaneUnavailable",
    "check_runtime_identity",
    "load_legacy_artifact",
    "load_m09_artifact",
    "native_source_manifest",
    "precheck",
    "verify_bundle_files",
]
