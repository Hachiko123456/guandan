"""Exercise M08 single/dual control launchers locally with a reduced CPU profile."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _prepare(delivery: Path, mode: str, destination: Path) -> Path:
    source = delivery / "datasets" / mode
    target = destination / mode
    shutil.copytree(source, target)
    manifest_path = target / "guandan_marvel_dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest["profiles"][mode]
    training = profile["training"]
    training.update({
        "target_iterations": 1,
        "checkpoint_interval": 1,
        "evaluation_interval": 0,
        "rollout_batch_size": 2,
        "collect_batch_size": 2,
        "replay_buffer_scale": 1,
        "mini_batches": 1,
        "max_sampler_bytes_per_rank": 128 * 2**20,
        "environment_rank_seed_stride": 1,
        "agent_epoch": 1,
        "critic_aux_epoch": 1,
        "model": {
            "architecture": "v1.3", "history_dim": 16, "feature_dim": 24,
            "decoder_layers": 1, "decoder_heads": 4, "state_layers": 1,
            "action_layers": 1, "fuse_layers": 1,
        },
        "max_wall_seconds": 600,
    })
    profile["smoke_horizon"] = 512
    profile["local_test_backend"] = "python_reference"
    profile["telemetry_interval_seconds"] = 1
    profile["heartbeat_interval_seconds"] = 1
    profile["benchmark"]["warmup_iterations"] = 0
    profile["benchmark"]["measured_iterations"] = 1
    profile["benchmark"]["workload_hash"] = _canonical({
        "algorithm": profile["algorithm"], "training": training,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _run(manifest: Path, *, input_root: Path, working_root: Path) -> Path:
    command = [
        sys.executable, "-u", "-m", "guandan_marvel.kaggle.runner",
        "--dataset-manifest", str(manifest),
        "--input-root", str(input_root),
        "--working-root", str(working_root),
        "--source-root", str(ROOT),
        "--allow-local-validation", "--local-cpu-test",
        "--local-process-timeout-seconds", "600",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    latest = json.loads((working_root / "guandan-marvel-output/latest_run.json").read_text(encoding="utf-8"))
    return working_root / "guandan-marvel-output" / latest["path"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    input_root = output / "input"
    input_root.mkdir(parents=True)
    rows = []
    for mode in ("single_gpu_control", "dual_gpu_control"):
        mode_input = input_root / mode
        mode_input.mkdir()
        manifest = _prepare(args.delivery.resolve(), mode, mode_input)
        working = output / f"working-{mode}"
        working.mkdir()
        run_root = _run(manifest, input_root=mode_input, working_root=working)
        benchmark = json.loads((run_root / "benchmark_summary.json").read_text(encoding="utf-8"))
        progress = json.loads((run_root / "progress.json").read_text(encoding="utf-8"))
        rows.append({"mode": mode, "run_root": str(run_root), "benchmark": benchmark, "progress": progress})
    if rows[0]["benchmark"]["workload_hash"] != rows[1]["benchmark"]["workload_hash"]:
        raise AssertionError("single/dual local controls used different workload hashes")
    summary = {
        "schema": "guandan-marvel-m08-local-control-smoke-v1",
        "scope": "local CPU reduced-profile launcher/checkpoint smoke; not GPU performance evidence",
        "rows": rows,
        "passes": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
