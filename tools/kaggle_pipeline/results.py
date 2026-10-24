"""Validate result bundles without extracting or unpickling model files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from tools.serial_compare.common import PHASE_ORDER, canonical


class ResultError(ValueError):
    pass


def _record(value: dict, field: str = "hash") -> None:
    if value.get(field) != canonical({k: v for k, v in value.items() if k != field}):
        raise ResultError("result record hash mismatch")


def verify_results(folder: Path, plan_hash: str) -> dict:
    archives = list(folder.rglob("guandan_serial_compare_results.zip"))
    if not archives:
        raise ResultError("missing serial-compare result archive; diagnostics retained")
    # The driver intentionally writes byte-identical copies at root/artifacts.
    if len(archives) > 1:
        from tools.serial_compare.common import sha
        if len({sha(path) for path in archives}) != 1:
            raise ResultError("conflicting result archives; do not guess the latest")
    archive = archives[0]
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [m.filename for m in members]
        if len({n.casefold() for n in names}) != len(names):
            raise ResultError("duplicate result archive members")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ResultError("unsafe result archive member")
        if "RESULTS_MANIFEST.json" not in names:
            raise ResultError("missing result manifest")
        manifest = json.loads(bundle.read("RESULTS_MANIFEST.json"))
        _record(manifest)
        if manifest.get("schema") != "guandan-serial-compare-results-manifest-v1":
            raise ResultError("unsupported result manifest")
        rows = manifest.get("files", [])
        declared = {r["path"]: r for r in rows}
        if len(declared) != len(rows) or set(declared) != set(names) - {"RESULTS_MANIFEST.json"}:
            raise ResultError("result file set differs from manifest")
        for name, row in declared.items():
            if bundle.getinfo(name).file_size != row["bytes"]:
                raise ResultError("result file size mismatch")
            digest = hashlib.sha256()
            with bundle.open(name) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ResultError("result file digest mismatch")
        if "artifacts/RESULT.json" not in names:
            raise ResultError("missing business result")
        result = json.loads(bundle.read("artifacts/RESULT.json"))
        _record(result)
        if result.get("schema") != "guandan-serial-compare-result-v1" or result.get("plan_hash") != plan_hash:
            raise ResultError("result belongs to another experiment")
        complete = (
            result.get("status") == "complete"
            and not result.get("technical_failure")
            and all(result.get("phases", {}).get(p, {}).get("status") == "complete" for p in PHASE_ORDER)
            and manifest.get("mode") == "full" and manifest.get("evidence_complete") is True
            and not manifest.get("omissions")
            and result.get("confirmation", {}).get("complete") is True
            and result.get("development", {}).get("complete") is True
        )
        recovery = json.loads(bundle.read("artifacts/recovery_index.json")) if "artifacts/recovery_index.json" in names else {}
        if recovery:
            _record(recovery)
        for arm in ("control", "candidate"):
            item = recovery.get("arms", {}).get(arm, {})
            files = item.get("files", [])
            if not item.get("latest") or not files:
                complete = False
            for row in files:
                record = declared.get("artifacts/" + row["path"])
                if record is None or record["sha256"] != row["sha256"] or record["bytes"] != row["bytes"]:
                    raise ResultError("recovery index is inconsistent with retained files")
    return {
        "status": "VERIFIED" if complete else "FETCHED_INCOMPLETE",
        "integrity": "PASS", "business_status": result.get("status"),
        "verdict": result.get("verdict", {}).get("verdict"),
        "archive": str(archive), "plan_hash": plan_hash,
    }
