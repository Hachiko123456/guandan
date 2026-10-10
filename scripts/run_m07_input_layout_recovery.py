"""Focused local CPU check for auto-unpacked M07 source and recovery inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha(path) for path in root.rglob("*") if path.is_file()}


def _profile(manifest_path: Path, mode: str) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = payload["profiles"][mode]
    profile["local_test_backend"] = "cpp"
    profile["smoke_horizon"] = 512
    profile["heartbeat_interval_seconds"] = 1
    profile["telemetry_interval_seconds"] = 1
    profile["evaluation"] = {"base_deals": 1, "seed_base": 970700, "opponents": ["random"], "seat_rotations": [0]}
    if mode == "pilot":
        profile["algorithm"] = "ippo"
        profile["training"] = {
            "target_iterations": 1, "checkpoint_interval": 1, "evaluation_interval": 0,
            "rollout_batch_size": 2, "collect_batch_size": 2, "replay_buffer_scale": 1,
            "mini_batches": 1, "max_sampler_bytes_per_rank": 128 * 2**20,
            "seed": 570700, "agent_epoch": 1, "critic_aux_epoch": 1,
            "amp_enabled": False, "compile_enabled": False, "max_wall_seconds": 300,
            "model": {"architecture": "v1.3", "history_dim": 8, "feature_dim": 8, "decoder_layers": 1, "decoder_heads": 1, "state_layers": 1, "action_layers": 1, "fuse_layers": 1},
        }
    else:
        profile["resume"] = {"additional_iterations": 1, "checkpoint_interval": 1, "evaluation_interval": 0, "max_wall_seconds": 300}
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run(manifest: Path, input_root: Path, working: Path) -> Path:
    command = [
        sys.executable, "-u", "-m", "guandan_marvel.kaggle.runner",
        "--dataset-manifest", str(manifest), "--input-root", str(input_root),
        "--working-root", str(working), "--source-root", str(ROOT),
        "--allow-local-validation", "--local-cpu-test", "--local-process-timeout-seconds", "180",
    ]
    subprocess.run(command, cwd=ROOT, check=True, timeout=200)
    output = working / "guandan-marvel-output"
    latest = json.loads((output / "latest_run.json").read_text(encoding="utf-8"))
    return output / latest["path"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    delivery, output = args.delivery.resolve(), args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    pilot_input = output / "pilot-input"
    pilot_dataset = pilot_input / "wrapper/pilot"
    shutil.copytree(delivery / "datasets/pilot", pilot_dataset)
    pilot_manifest = pilot_dataset / "guandan_marvel_dataset.json"
    _profile(pilot_manifest, "pilot")
    pilot_working = output / "pilot-working"; pilot_working.mkdir()
    pilot_run = _run(pilot_manifest, pilot_input, pilot_working)
    recovery_zip = pilot_working / "guandan-marvel-output/guandan-marvel-recovery.zip"

    resume_input = output / "resume-input"
    resume_dataset = resume_input / "outer/dataset/resume"
    shutil.copytree(delivery / "datasets/resume", resume_dataset)
    resume_manifest = resume_dataset / "guandan_marvel_dataset.json"
    _profile(resume_manifest, "resume")
    source_zip = resume_dataset / "guandan-marvel-source.zip"
    source_wrapper = resume_dataset / "kaggle-auto/source"
    source_wrapper.mkdir(parents=True)
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(source_wrapper)
    source_zip.unlink()
    recovery_root = resume_input / "other-dataset/wrapped/recovery"
    recovery_root.mkdir(parents=True)
    with zipfile.ZipFile(recovery_zip) as archive:
        archive.extractall(recovery_root)
    source_tree = source_wrapper / "guandan-marvel-source"
    source_before, recovery_before = _tree_hashes(source_tree), _tree_hashes(recovery_root)

    resume_working = output / "resume-working"; resume_working.mkdir()
    resume_run = _run(resume_manifest, resume_input, resume_working)
    progress = json.loads((resume_run / "progress.json").read_text(encoding="utf-8"))
    if progress.get("iteration") != 2 or progress.get("phase") != "complete":
        raise AssertionError(f"unpacked recovery did not continue to iteration 2: {progress}")
    if source_before != _tree_hashes(source_tree) or recovery_before != _tree_hashes(recovery_root):
        raise AssertionError("read-only unpacked input content was modified")
    resolved = json.loads((resume_run / "resolved_config.json").read_text(encoding="utf-8"))
    result = {
        "schema": "guandan-marvel-m07-input-layout-cpu-resume-v1",
        "scope": "local CPU/gloo; no Kaggle/T4/NCCL claim",
        "source_input": "unpacked_directory_without_zip",
        "recovery_input": "unpacked_directory_without_zip",
        "source_wrapper": str(source_tree),
        "recovery_wrapper": str(recovery_root),
        "pilot_run": str(pilot_run),
        "resume_run": str(resume_run),
        "resume_iteration": progress["iteration"],
        "resume_inherited": resolved["resume_inherited"],
        "inputs_unchanged": True,
        "passes": True,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
