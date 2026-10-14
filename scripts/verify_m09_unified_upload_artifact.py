"""Verify the exact M09 Unified upload ZIP under simulated Kaggle auto-unpack."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.evaluation.host import _base_deal_hash
from guandan_marvel.evaluation.manifests import build_deals_manifest, deal_game_config
from guandan_marvel.evaluation.provenance import materialize_deal_identity
from guandan_marvel import cpp_game
from guandan_marvel.kaggle.common import atomic_json, sha256_file


def _extract(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        if bad := archive.testzip():
            raise ValueError(f"ZIP CRC failure: {bad}")
        for info in archive.infolist():
            pure = Path(info.filename.replace("/", "\\"))
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe ZIP member: {info.filename}")
        archive.extractall(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    upload = args.upload_zip.resolve()
    with tempfile.TemporaryDirectory(prefix="m09-upload-preflight-") as temp_name:
        temp = Path(temp_name)
        dataset = temp / "input/guandan-marvel-m09"
        dataset.mkdir(parents=True)
        _extract(upload, dataset)
        control = json.loads((dataset / "m09_unified_control.json").read_text(encoding="utf-8"))
        for row in control["control_files"]:
            path = dataset / row["path"]
            if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                raise ValueError(f"control file mismatch after outer extraction: {row['path']}")

        source_zip = dataset / "guandan-marvel-source.zip"
        source_unpack_parent = dataset / "guandan-marvel-source"
        source_unpack_parent.mkdir()
        _extract(source_zip, source_unpack_parent)
        source_zip.unlink()
        recovery_zip = dataset / "m09_initial_iter0500_recovery.seed.zip"
        recovery_unpack = dataset / "m09_initial_iter0500_recovery.seed"
        recovery_unpack.mkdir()
        _extract(recovery_zip, recovery_unpack)
        recovery_zip.unlink()

        overlay = dataset / "control_overlay"
        source_root = source_unpack_parent / "guandan-marvel-source"
        sys.path.insert(0, str(overlay))
        sys.path.insert(1, str(source_root))
        from guandan_marvel.kaggle.bootstrap import prepare_inputs, resolve_dataset_input
        from guandan_marvel.kaggle.unified_pipeline import discover_latest_recovery, materialize_recovery_candidate, SegmentTarget

        resolved = resolve_dataset_input(dataset / "guandan_marvel_dataset.json")
        if resolved.source.kind != "directory":
            raise AssertionError(f"expected auto-unpacked source directory, got {resolved.source.kind}")
        empty = temp / "empty-input"; empty.mkdir()
        prepared = prepare_inputs(dataset / "guandan_marvel_dataset.json", input_root=empty, working_root=temp / "working")
        selected = discover_latest_recovery(dataset, source_identity=control["source_identity"])
        recovery_root = materialize_recovery_candidate(selected, temp / "selected-recovery", copy_directory=False)
        recovery_manifest = json.loads((recovery_root / "recovery_manifest.json").read_text(encoding="utf-8"))
        if int(recovery_manifest["iteration"]) != 500:
            raise AssertionError("wrong recovery iteration")

        pipeline_script = overlay / "scripts/m09_unified_pipeline.py"
        spec = importlib.util.spec_from_file_location("packaged_m09_unified_pipeline", pipeline_script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        acceptance = dataset / control["acceptance_profiles"]["path"]
        profile_target = Path(prepared["source_root"]) / "configs/acceptance_profiles.json"
        profile_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(acceptance, profile_target)
        if sha256_file(profile_target) != control["acceptance_profiles"]["sha256"]:
            raise AssertionError("acceptance profile materialization mismatch")
        runtime_manifest = module._runtime_dataset(
            dataset_manifest_path=dataset / "guandan_marvel_dataset.json",
            target=SegmentTarget("remote_a_smoke0502", 502, 2008),
            runtime_root=temp / "runtime-dataset",
        )
        runtime_resolved = resolve_dataset_input(runtime_manifest)
        if runtime_resolved.source.kind != "directory":
            raise AssertionError("temporary runtime Dataset did not preserve auto-unpacked source")

        packaged_provenance = overlay / "guandan_marvel/evaluation/provenance.py"
        if packaged_provenance.read_bytes() != (ROOT / "guandan_marvel/evaluation/provenance.py").read_bytes():
            raise AssertionError("packaged provenance implementation is stale")
        deals = build_deals_manifest(counts={"training": 0, "selection": 2, "final": 0})
        hashes = []
        for deal in deals["datasets"]["selection"]:
            identity = materialize_deal_identity(deal)
            env = cpp_game.EnvBatch(1, configs=[deal_game_config(deal)], auto_reset=False)
            env.reset(int(deal["seed"]))
            host_hash = _base_deal_hash(env)
            if identity["base_deal_sha256"] != host_hash:
                raise AssertionError({"deal_id": deal["deal_id"], "planned": identity["base_deal_sha256"], "host": host_hash})
            hashes.append({"deal_id": deal["deal_id"], "deal_sha256": identity["deal_sha256"], "base_deal_sha256": host_hash})

        result = {
            "schema": "guandan-marvel-m09-unified-upload-artifact-preflight-v1",
            "status": "passed",
            "upload_zip": str(upload),
            "upload_zip_sha256": sha256_file(upload),
            "control_manifest_hash": control["manifest_hash"],
            "outer_control_files_verified": len(control["control_files"]),
            "simulated_kaggle_source_kind": resolved.source.kind,
            "selected_recovery_iteration": 500,
            "acceptance_profile_materialized": True,
            "runtime_dataset_source_kind": runtime_resolved.source.kind,
            "packaged_provenance_current": True,
            "real_cpp_deal_hash_checks": hashes,
            "final_holdout_generated": False,
        }
        result["hash"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
