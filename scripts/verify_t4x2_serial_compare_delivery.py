"""Offline QA of the *actual delivered* serial-compare payload.

Decodes the real ``.bin`` into an isolated directory and verifies the payload /
manifest / per-file hashes, the plan and deal manifests, byte-identical overlay
and driver sources against the current workspace, the bound Notebook constants,
and that the driver imports from the decoded directory alone.

Never imports the workspace ``guandan_marvel`` package for these checks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from serial_compare.common import canonical, read_json, sha, validate_deal_manifest, validate_plan  # noqa: E402
from serial_compare.delivery import (  # noqa: E402
    MAGIC, MODEL_FILES, NOTEBOOK, PAYLOAD, SOURCE_IDENTITY, decode_payload,
)
from serial_compare.common import PlanError  # noqa: E402

REQUIRED_NEW_MODULES = (
    "guandan_marvel/training/policy_import.py",
    "guandan_marvel/training/local.py",
    "guandan_marvel/training/macro_ppo.py",
    "guandan_marvel/config.py",
    "guandan_marvel/phase6/opponent_pool.py",
    "tools/serial_compare/orchestrator.py",
    "tools/serial_compare/evaluation.py",
    "tools/serial_compare/budget.py",
    "tools/serial_compare/pipeline.py",
    "tools/serial_compare/runtime_validation.py",
)


class QAFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QAFailure(message)


def _zip_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as bundle:
        return {name: bundle.read(name) for name in bundle.namelist()}


def _isolated_import_check(driver: Path) -> dict[str, object]:
    """Import the decoded driver with the decoded driver as the *only* path."""
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(driver)!r})\n"
        "import serial_compare.common as common\n"
        "import serial_compare.evaluation as evaluation\n"
        "print(json.dumps({'common': common.__file__, 'tasks': len(evaluation.confirmation_tasks(common.build_deal_manifest('confirmation')))}))\n"
    )
    with tempfile.TemporaryDirectory(prefix="serial-compare-import-") as temporary:
        script_path = Path(temporary) / "check.py"
        script_path.write_text(script, encoding="utf-8")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            capture_output=True, text=True, env=environment, timeout=120,
        )
    if completed.returncode != 0:
        raise QAFailure(f"isolated driver import failed: {completed.stderr.strip()[:400]}")
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    common_path = Path(payload["common"]).resolve()
    _require(common_path.is_relative_to(driver.resolve()), "the driver did not import from the decoded directory")
    _require(int(payload["tasks"]) == 256, "confirmation task count changed in the decoded driver")
    return {"common": str(common_path), "confirmation_tasks": int(payload["tasks"])}


def verify(delivery: Path, *, repo: Path = REPO, output: Path | None = None) -> dict[str, object]:
    delivery = delivery.resolve()
    payload_path = delivery / PAYLOAD
    _require(payload_path.is_file(), "payload file is missing")
    record: dict[str, object] = {"schema": "guandan-serial-compare-artifact-qa-v1", "delivery": str(delivery)}
    with tempfile.TemporaryDirectory(prefix="serial-compare-qa-") as temporary:
        decoded = Path(temporary) / "decoded"
        decode = decode_payload(payload_path, decoded)
        record["payload"] = decode
        manifest = read_json(decoded / "delivery_manifest.json")
        _require(
            manifest["manifest_hash"] == canonical({k: v for k, v in manifest.items() if k != "manifest_hash"}),
            "inner delivery manifest hash is inconsistent",
        )
        _require({path.name for path in decoded.iterdir()} == {"delivery_manifest.json", *(row["path"] for row in manifest["files"])}, "undeclared payload files")
        for row in manifest["files"]:
            path = decoded / row["path"]
            _require(path.is_file(), f"declared payload file missing: {row['path']}")
            _require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], f"payload file hash mismatch: {row['path']}")
        top_manifest = read_json(delivery / "DELIVERY_MANIFEST.json")
        _require(top_manifest["manifest_hash"] == manifest["manifest_hash"], "top-level DELIVERY_MANIFEST differs from the payload")
        plan = read_json(decoded / "plan.json")
        validate_plan(plan)
        _require(plan["plan_hash"] == manifest["plan_hash"], "plan hash disagrees with the manifest")
        record["plan_hash"] = plan["plan_hash"]

        deals = {}
        for domain in ("preflight", "development", "confirmation"):
            manifest_item = read_json(decoded / f"deals_{domain}.json")
            validate_deal_manifest(manifest_item, expected_domain=domain)
            _require(manifest["deal_manifests"][domain] == manifest_item["hash"], f"{domain} deal manifest hash mismatch")
            deals[domain] = {"hash": manifest_item["hash"], "deals": manifest_item["deal_count"]}
        record["deals"] = deals
        _require(int(deals["confirmation"]["deals"]) == 128, "confirmation must have 128 deals")
        _require(int(deals["development"]["deals"]) == 32, "development must have 32 deals")

        models = {}
        for name, (_filename, expected) in MODEL_FILES.items():
            path = decoded / {"p375": "policy_375.pt", "p250": "anchor_250.pt", "p500": "policy_500.pt"}[name]
            actual = sha(path)
            _require(actual == expected, f"payload model {name!r} is not the expected artifact")
            _require(plan["models"][name]["sha256"] == expected, f"plan model {name!r} hash mismatch")
            models[name] = actual
        record["models"] = models

        source_bytes = (decoded / "frozen_source.zip").read_bytes()
        _require(hashlib.sha256(source_bytes).hexdigest() == SOURCE_IDENTITY["archive_sha256"], "frozen source archive changed")
        with zipfile.ZipFile(decoded / "frozen_source.zip") as bundle:
            source_manifest = json.loads(bundle.read("guandan-marvel-source/source_manifest.json"))
            _require(source_manifest["manifest_hash"] == SOURCE_IDENTITY["content_manifest_hash"], "frozen source content manifest changed")
            for row in source_manifest["files"]:
                data = bundle.read(f"guandan-marvel-source/{row['path']}")
                _require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], f"frozen source member changed: {row['path']}")
        _require(sha(decoded / "frozen_native.so") == SOURCE_IDENTITY["native_sha256"], "frozen native hash changed")

        overlay = {name: data for name, data in _zip_members(decoded / "overlay_source.zip").items()}
        _require(bool(overlay), "overlay is empty")
        for name, data in overlay.items():
            _require(Path(name).suffix not in {".so", ".cpp", ".hpp", ".h", ".pyd", ".pyx"}, f"overlay contains a native source: {name}")
            workspace = repo / name
            _require(workspace.is_file(), f"overlay file no longer exists in the workspace: {name}")
            _require(workspace.read_bytes() == data, f"overlay file differs from the workspace: {name}")
        for required in REQUIRED_NEW_MODULES:
            _require(required in overlay, f"the payload is missing the new module {required}")
        record["overlay_files"] = len(overlay)

        driver_members = _zip_members(decoded / "driver.zip")
        _require(bool(driver_members), "driver archive is empty")
        for name, data in driver_members.items():
            workspace = repo / "tools" / name
            _require(workspace.is_file() and workspace.read_bytes() == data, f"driver file differs from the workspace: {name}")
        driver_dir = Path(temporary) / "driver"
        with zipfile.ZipFile(decoded / "driver.zip") as bundle:
            bundle.extractall(driver_dir)
        record["driver_files"] = len(driver_members)
        record["isolated_import"] = _isolated_import_check(driver_dir)

        notebook = read_json(delivery / NOTEBOOK)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        _require(len(code_cells) == 1, "the notebook must contain exactly one run-all code cell")
        code = "".join(code_cells[0]["source"])
        ast.parse(code)
        _require(repr(sha(payload_path)) in code, "the Notebook is not bound to this payload hash")
        _require(repr(manifest["manifest_hash"]) in code, "the Notebook is not bound to this manifest hash")
        _require(SOURCE_IDENTITY["runtime_path"] in code, "the Notebook does not restore the original runtime path")
        _require("run_notebook" in code, "the Notebook does not call the packaged driver")
        _require("serial_compare" in code, "the Notebook does not import the serial_compare driver")
        record["notebook"] = {"sha256": sha(delivery / NOTEBOOK), "code_cell_parsed": True}

        sums = read_json(delivery / "SHA256SUMS")
        _require(sums["hash"] == canonical({k: v for k, v in sums.items() if k != "hash"}), "SHA256SUMS hash is inconsistent")
        listed = {row["path"]: row["sha256"] for row in sums["files"]}
        for name, digest in listed.items():
            _require((delivery / name).is_file() and sha(delivery / name) == digest, f"SHA256SUMS entry mismatch: {name}")

        _require(plan["boundaries"]["fabledan"] == "NOT_RUN", "FableDan boundary changed")
        _require(plan["boundaries"]["final_holdout"] == "NOT_TOUCHED", "final holdout boundary changed")
        _require(plan["boundaries"]["formal_production_promotion"] is False, "formal promotion must stay false")

    record["status"] = "PASS"
    record["hash"] = canonical({k: v for k, v in record.items() if k != "hash"})
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        record = verify(args.delivery, output=args.output)
    except (QAFailure, PlanError) as error:
        failure = {"schema": "guandan-serial-compare-artifact-qa-v1", "status": "FAIL", "error": str(error)}
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
