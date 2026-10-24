"""Transport-only hotfix: preserve v2 training source/config/weights byte-for-byte.

Upload is header + BASE64(ZIP), not a renamed ZIP. No training or network.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import inspect
import io
import json
from pathlib import Path
import zipfile

TRANSPORT_MAGIC = b"GUANDAN-PILOT-TRANSPORT-V3\n"
TRANSPORT_FILENAME = "pilot_payload_v3.bin"
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 64 * 1024 * 1024


def _transport_hash(data):
    return hashlib.sha256(data).hexdigest()


def _transport_file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transport_canonical_hash(value):
    return _transport_hash(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _transport_diagnostic(directory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "bootstrap_diagnostics.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "bootstrap_diagnostics.json")
    with (directory / "bootstrap_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")


def _transport_flat_name(info):
    name = info.orig_filename
    if (not name or name in (".", "..") or name != info.filename or info.is_dir()
            or any(value in name for value in ("/", "\\", ":", "\0"))
            or ((info.external_attr >> 16) & 0o170000) == 0o120000):
        raise ValueError("unsafe transport member: " + repr(name))
    return name


def _verify_transport_container(container, expected_manifest_hash):
    infos = container.infolist()
    names = [_transport_flat_name(info) for info in infos]
    if len(names) != len(set(name.casefold() for name in names)):
        raise ValueError("duplicate/case-colliding transport member")
    if sum(info.file_size for info in infos) > MAX_EXTRACTED_BYTES:
        raise ValueError("transport extracted size exceeds safety budget")
    if "pilot_manifest.json" not in names:
        raise ValueError("transport does not contain pilot_manifest.json")
    manifest = json.loads(container.read("pilot_manifest.json"))
    actual = _transport_canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if manifest.get("manifest_hash") != actual or actual != expected_manifest_hash:
        raise ValueError("transport inner pilot manifest identity mismatch")
    expected = [row["path"] for row in manifest["files"]]
    if len(set(expected)) != len(expected) or set(names) != {"pilot_manifest.json", *expected}:
        raise ValueError("transport contents differ from frozen pilot manifest")
    for row in manifest["files"]:
        data = container.read(row["path"])
        if len(data) != row["bytes"] or _transport_hash(data) != row["sha256"]:
            raise ValueError("transport inner file identity mismatch: " + row["path"])
    return manifest


def verify_unpacked_pilot(dataset, expected_manifest_hash, diagnostics):
    dataset = Path(dataset).resolve()
    report = {"schema": "guandan-pilot-bootstrap-diagnostics-v3", "stage": "inner_input_validation",
              "status": "checking", "dataset_dir": str(dataset), "files": []}
    try:
        manifest = json.loads((dataset / "pilot_manifest.json").read_text(encoding="utf-8"))
        actual = _transport_canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
        if manifest.get("manifest_hash") != actual or actual != expected_manifest_hash:
            raise ValueError("Notebook/Dataset inner manifest mismatch")
        errors = []
        for row in manifest["files"]:
            relative = str(row["path"])
            raw = dataset / relative
            if (not relative or relative in (".", "..") or any(x in relative for x in ("/", "\\", ":", "\0"))
                    or raw.is_symlink()):
                raise ValueError("unsafe input name: " + repr(relative))
            path = raw.resolve()
            if not path.is_relative_to(dataset):
                raise ValueError("input escapes Dataset")
            detail = {"path": str(path), "expected_bytes": row["bytes"], "expected_sha256": row["sha256"],
                      "exists": path.exists(), "is_file": path.is_file(), "is_directory": path.is_dir()}
            if not path.is_file():
                detail["error"] = "missing_file" if not path.exists() else "not_a_file"
                if relative == "guandan-marvel-source.zip":
                    detail["expanded_source_directory_present"] = (dataset / "guandan-marvel-source").is_dir()
            else:
                detail["actual_bytes"] = path.stat().st_size
                detail["actual_sha256"] = _transport_file_hash(path)
                if detail["actual_bytes"] != row["bytes"]:
                    detail["error"] = "size_mismatch"
                elif detail["actual_sha256"] != row["sha256"]:
                    detail["error"] = "sha256_mismatch"
            if "error" in detail:
                errors.append(detail)
            report["files"].append(detail)
        report["inventory"] = [{"name": p.name, "is_directory": p.is_dir(),
                                 "bytes": p.stat().st_size if p.is_file() else None}
                                for p in list(dataset.iterdir())[:50]]
        if errors:
            report["status"] = "failed"
            report["errors"] = errors
            _transport_diagnostic(diagnostics, report)
            raise RuntimeError("Pilot input check failed: " + json.dumps(errors, ensure_ascii=False)
                               + "; diagnostics=" + str(Path(diagnostics) / "bootstrap_diagnostics.json"))
        report["status"] = "complete"
        report["manifest_hash"] = actual
        _transport_diagnostic(diagnostics, report)
        print("[PILOT INPUT] all frozen files verified; manifest=" + actual, flush=True)
        return manifest
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        _transport_diagnostic(diagnostics, report)
        raise


def materialize_pilot_transport(input_root, run_root, expected_payload_hash, expected_manifest_hash,
                                explicit_payload=None):
    input_root, run_root = Path(input_root), Path(run_root)
    diagnostics = run_root.parent / (run_root.name + "_bootstrap")
    report = {"schema": "guandan-pilot-bootstrap-diagnostics-v3", "stage": "transport_decode",
              "status": "checking", "input_root": str(input_root), "expected_payload_sha256": expected_payload_hash}
    try:
        paths = [Path(explicit_payload)] if explicit_payload is not None else list(input_root.rglob(TRANSPORT_FILENAME))
        report["payload_candidates"] = [str(p) for p in paths]
        matching = [p for p in paths if p.is_file() and not p.is_symlink()
                    and p.stat().st_size <= MAX_PAYLOAD_BYTES and _transport_file_hash(p) == expected_payload_hash]
        if len(matching) != 1:
            report["candidate_details"] = [{"path": str(p), "is_file": p.is_file(),
                                             "bytes": p.stat().st_size if p.is_file() else None,
                                             "sha256": _transport_file_hash(p) if p.is_file() and p.stat().st_size <= MAX_PAYLOAD_BYTES else None}
                                            for p in paths]
            raise RuntimeError("Expected one matching " + TRANSPORT_FILENAME
                               + ". Upload the .bin directly as a Dataset file, not a ZIP or the whole delivery folder. "
                               + "Set PILOT_PAYLOAD_PATH only if multiple matching copies are mounted.")
        payload = matching[0]
        report["selected_payload"] = str(payload)
        data = payload.read_bytes()
        if not data.startswith(TRANSPORT_MAGIC):
            raise ValueError("not a v3 base64 transport payload (renaming a ZIP is not sufficient)")
        decoded = base64.b64decode(data[len(TRANSPORT_MAGIC):].strip(), validate=True)
        if len(decoded) > MAX_PAYLOAD_BYTES:
            raise ValueError("decoded transport exceeds safety budget")
        destination = run_root.parent / (run_root.name + "_dataset")
        if destination.exists():
            raise FileExistsError(str(destination) + " already exists; choose a new RUN_ROOT or a fresh session")
        with zipfile.ZipFile(io.BytesIO(decoded)) as archive:
            manifest = _verify_transport_container(archive, expected_manifest_hash)
            destination.mkdir(parents=True)
            archive.extractall(destination)
        verify_unpacked_pilot(destination, expected_manifest_hash, diagnostics)
        report.update({"status": "complete", "actual_payload_sha256": _transport_hash(data),
                       "decoded_zip_bytes": len(decoded), "dataset_dir": str(destination),
                       "manifest_hash": manifest["manifest_hash"]})
        _transport_diagnostic(diagnostics, report)
        print("[PILOT TRANSPORT] verified and decoded " + str(payload), flush=True)
        return destination
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        _transport_diagnostic(diagnostics, report)
        print("[PILOT TRANSPORT ERROR] " + str(error) + "; diagnostics=" + str(diagnostics), flush=True)
        raise


def render_transport_notebook(original, payload_hash, manifest_hash):
    result = json.loads(json.dumps(original))
    code_cells = [cell for cell in result["cells"] if cell.get("cell_type") == "code"]
    if len(code_cells) != 1:
        raise ValueError("expected one v2 bootstrap code cell")
    code = "".join(code_cells[0]["source"])
    if f"EXPECTED_PILOT_MANIFEST_HASH = {manifest_hash!r}" not in code:
        raise ValueError("original Notebook not bound to expected v2 manifest")
    start = code.index("if DATASET_DIR is None:")
    end = code.index("if RUN_ROOT.exists():", start)
    replacement = """DIAGNOSTIC_DIR = RUN_ROOT.parent/(RUN_ROOT.name+'_bootstrap')
if DATASET_DIR is None:
    DATASET_DIR = materialize_pilot_transport(Path('/kaggle/input'), RUN_ROOT,
        EXPECTED_PILOT_PAYLOAD_SHA256, EXPECTED_PILOT_MANIFEST_HASH, PILOT_PAYLOAD_PATH)
DATASET_DIR = Path(DATASET_DIR)
meta = verify_unpacked_pilot(DATASET_DIR, EXPECTED_PILOT_MANIFEST_HASH, DIAGNOSTIC_DIR)
"""
    code = code[:start] + replacement + code[end:]
    code = code.replace("RUN_ROOT = Path('/kaggle/working/guandan_t4x2_pilot')",
                        "RUN_ROOT = Path('/kaggle/working/guandan_t4x2_pilot_v3')", 1)
    code = code.replace("DATASET_DIR = None  # optionally set Path('/kaggle/input/your-dataset')",
                        "DATASET_DIR = None  # populated by verified opaque transport", 1)
    definitions = [_transport_hash, _transport_file_hash, _transport_canonical_hash, _transport_diagnostic,
                   _transport_flat_name, _verify_transport_container, verify_unpacked_pilot, materialize_pilot_transport]
    support = ("from __future__ import annotations\nfrom pathlib import Path\n"
               "import base64, hashlib, io, json, zipfile\n"
               f"TRANSPORT_MAGIC = {TRANSPORT_MAGIC!r}\nTRANSPORT_FILENAME = {TRANSPORT_FILENAME!r}\n"
               f"MAX_PAYLOAD_BYTES = {MAX_PAYLOAD_BYTES}\nMAX_EXTRACTED_BYTES = {MAX_EXTRACTED_BYTES}\n"
               f"EXPECTED_PILOT_PAYLOAD_SHA256 = {payload_hash!r}\n"
               "PILOT_PAYLOAD_PATH = None  # optional full path to matching .bin\n\n")
    support += "\n\n".join(inspect.getsource(function) for function in definitions) + "\n\n"
    final_code = support + code
    ast.parse(final_code)
    code_cells[0]["source"] = final_code.splitlines(keepends=True)
    code_cells[0]["execution_count"] = None
    code_cells[0]["outputs"] = []
    result["cells"][0]["source"] = [
        "# 掼蛋 T4×2 Pilot v3 — 上传传输修复版\n",
        "**只将 `pilot_payload_v3.bin` 作为 Dataset 文件上传，不要上传 ZIP 或整个文件夹。**\n",
        "导入本 Notebook，Add Input 选择该 Dataset，设置 GPU T4×2，Internet 可关闭。\n",
        "严格校验并本地还原原始 v2 源码、配置和权重，未取消任何哈希检查。\n",
        "2 iter → 恢复到3 → 绝对50 iter，共1000训练局；开发评估384局；3.5小时总安全预算。\n",
        "输入诊断在 `/kaggle/working/guandan_t4x2_pilot_v3_bootstrap/`；如报错请下载诊断 JSON 和日志。\n"]
    return result


def repackage(source_delivery: Path, output: Path):
    source_delivery, output = source_delivery.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("choose a new output directory; never overwrite v2")
    source_zip = source_delivery / "pilot-upload.zip"
    original = json.loads((source_delivery / "guandan_t4x2_pilot.ipynb").read_text(encoding="utf-8"))
    data = source_zip.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("pilot_manifest.json"))
        _verify_transport_container(archive, manifest["manifest_hash"])
    payload = TRANSPORT_MAGIC + base64.b64encode(data) + b"\n"
    if zipfile.is_zipfile(io.BytesIO(payload)):
        raise AssertionError("opaque transport must not itself be an archive")
    output.mkdir(parents=True)
    target = output / TRANSPORT_FILENAME
    target.write_bytes(payload)
    payload_hash = _transport_hash(payload)
    notebook = render_transport_notebook(original, payload_hash, manifest["manifest_hash"])
    notebook_path = output / "guandan_t4x2_pilot_v3.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verification = {"schema": "guandan-pilot-transport-verification-v3", "status": "BUILT_NOT_RUN_ON_KAGGLE",
                    "transport": "header + base64(original ZIP); not an archive; not just renamed",
                    "payload": target.name, "payload_bytes": len(payload), "payload_sha256": payload_hash,
                    "notebook": notebook_path.name, "notebook_sha256": _transport_file_hash(notebook_path),
                    "inner_zip_sha256": _transport_hash(data), "inner_pilot_manifest_hash": manifest["manifest_hash"],
                    "training_source_archive_sha256": manifest["source"]["archive_sha256"],
                    "training_source_and_config_changed": False, "training_launched": False,
                    "original_v2_inputs": manifest["files"]}
    (output / "TRANSPORT_VERIFICATION.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repackage(args.source_delivery, args.output), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
