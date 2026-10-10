"""Run M07 repair CPU/gloo DDP, evaluation, failure, budget, and recovery exercises."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def _small_training(*, target: int, evaluation_interval: int = 1, max_wall_seconds: float = 600) -> dict[str, object]:
    return {
        "target_iterations": int(target),
        "checkpoint_interval": 1,
        "evaluation_interval": int(evaluation_interval),
        "rollout_batch_size": 2,
        "collect_batch_size": 2,
        "replay_buffer_scale": 1,
        "mini_batches": 1,
        "max_sampler_bytes_per_rank": 128 * 2**20,
        "seed": 470700,
        "agent_epoch": 1,
        "critic_aux_epoch": 1,
        "amp_enabled": False,
        "compile_enabled": False,
        "max_wall_seconds": float(max_wall_seconds),
        "model": {
            "architecture": "v1.3", "history_dim": 8, "feature_dim": 8,
            "decoder_layers": 1, "decoder_heads": 1, "state_layers": 1,
            "action_layers": 1, "fuse_layers": 1,
        },
    }


def _copy_dataset(delivery: Path, mode: str, destination: Path, *, mutate=None) -> Path:
    source = delivery / "datasets" / mode
    target = destination / f"guandan-marvel-{mode}"
    shutil.copytree(source, target)
    manifest_path = target / "guandan_marvel_dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest["profiles"][mode]
    if mutate is not None:
        mutate(profile)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _common_profile(profile: dict[str, object]) -> None:
    profile["local_test_backend"] = "cpp"
    profile["smoke_horizon"] = 512
    profile["heartbeat_interval_seconds"] = 1
    profile["telemetry_interval_seconds"] = 1
    profile["evaluation"] = {
        "base_deals": 1, "seed_base": 870700,
        "opponents": ["random"], "seat_rotations": [0], "timeout_seconds": 120,
    }


def _pilot_mutator(*, target: int, evaluation_interval: int = 1, faults=None, max_wall_seconds=600):
    def mutate(profile):
        _common_profile(profile)
        profile["algorithm"] = "ippo"
        profile["training"] = _small_training(
            target=target, evaluation_interval=evaluation_interval,
            max_wall_seconds=max_wall_seconds,
        )
        if faults:
            profile["fault_injection"] = faults
    return mutate



def _full_mutator(profile):
    _common_profile(profile)
    training = profile["training"]
    training["target_iterations"] = 1
    training["checkpoint_interval"] = 1
    training["evaluation_interval"] = 0
    training["max_wall_seconds"] = 900

def _resume_mutator(*, additional: int = 1, evaluation_interval: int = 1):
    def mutate(profile):
        _common_profile(profile)
        profile["resume"] = {
            "additional_iterations": int(additional),
            "checkpoint_interval": 1,
            "evaluation_interval": int(evaluation_interval),
            "max_wall_seconds": 600,
        }
    return mutate


def _eval_mutator(profile):
    _common_profile(profile)


def _run(*, manifest: Path, input_root: Path, working_root: Path, expected_codes=(0,), timeout=240) -> tuple[Path, int, float]:
    command = [
        sys.executable, "-u", "-m", "guandan_marvel.kaggle.runner",
        "--dataset-manifest", str(manifest),
        "--input-root", str(input_root),
        "--working-root", str(working_root),
        "--source-root", str(ROOT),
        "--allow-local-validation", "--local-cpu-test",
        "--local-process-timeout-seconds", str(timeout - 10),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, timeout=timeout)
    elapsed = time.perf_counter() - started
    if completed.returncode not in expected_codes:
        raise RuntimeError(f"runner exit {completed.returncode}, expected {expected_codes}: {manifest}")
    output = working_root / "guandan-marvel-output"
    latest = json.loads((output / "latest_run.json").read_text(encoding="utf-8"))
    return output / latest["path"], completed.returncode, elapsed


def _checkpoint(run: Path) -> tuple[dict[str, object], dict[str, object]]:
    index = json.loads((run / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    latest = index["latest"]
    manifest = json.loads((run / latest["checkpoint"] / "manifest.json").read_text(encoding="utf-8"))
    return index, manifest


def _recovery_iteration(bundle: Path) -> int:
    with zipfile.ZipFile(bundle) as archive:
        payload = json.loads(archive.read("recovery_manifest.json").decode("utf-8"))
    return int(payload["iteration"])


def _scenario_root(output: Path, name: str) -> tuple[Path, Path]:
    root = output / name
    inputs, working = root / "input", root / "working"
    inputs.mkdir(parents=True)
    working.mkdir(parents=True)
    return inputs, working


def _resume_from(delivery: Path, output: Path, name: str, recovery: Path, *, evaluation_interval=1, timeout=240) -> Path:
    inputs, working = _scenario_root(output, name)
    manifest = _copy_dataset(delivery, "resume", inputs, mutate=_resume_mutator(evaluation_interval=evaluation_interval))
    recovery_dir = inputs / "recovery"
    recovery_dir.mkdir()
    shutil.copy2(recovery, recovery_dir / "guandan-marvel-recovery.zip")
    run, code, _ = _run(
        manifest=manifest, input_root=inputs, working_root=working, expected_codes=(0,), timeout=timeout,
    )
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    delivery, output = args.delivery.resolve(), args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    scenarios: dict[str, object] = {}

    # Two uninterrupted updates, both with real spawned C++ evaluation.
    inputs, working = _scenario_root(output, "continuous")
    manifest = _copy_dataset(delivery, "pilot", inputs, mutate=_pilot_mutator(target=2))
    continuous, code, elapsed = _run(manifest=manifest, input_root=inputs, working_root=working)
    continuous_index, continuous_manifest = _checkpoint(continuous)
    scenarios["continuous"] = {"run": str(continuous), "exit_code": code, "elapsed_seconds": elapsed}

    # One update + recovery + one resumed update; compare full continuation state.
    inputs, working = _scenario_root(output, "split")
    manifest = _copy_dataset(delivery, "pilot", inputs, mutate=_pilot_mutator(target=1))
    split, _, _ = _run(manifest=manifest, input_root=inputs, working_root=working)
    split_recovery = working / "guandan-marvel-output/guandan-marvel-recovery.zip"
    resumed = _resume_from(delivery, output, "split_resume", split_recovery)
    resumed_index, resumed_manifest = _checkpoint(resumed)
    continuous_hashes = [row["continuation_state_hash"] for row in continuous_manifest["rank_shards"]]
    resumed_hashes = [row["continuation_state_hash"] for row in resumed_manifest["rank_shards"]]
    if continuous_hashes != resumed_hashes:
        raise AssertionError("continuous and evaluation-enabled resume continuation state diverged")
    scenarios["split_resume"] = {
        "split_run": str(split), "resumed_run": str(resumed),
        "continuous_hashes": continuous_hashes, "resumed_hashes": resumed_hashes,
        "equal": True,
    }

    # Original full model/batch/epoch semantics, but only one smoke-horizon update, then resume.
    inputs, working = _scenario_root(output, "full_config")
    manifest = _copy_dataset(delivery, "full", inputs, mutate=_full_mutator)
    full_run, _, full_elapsed = _run(
        manifest=manifest, input_root=inputs, working_root=working, timeout=600,
    )
    full_recovery = working / "guandan-marvel-output/guandan-marvel-recovery.zip"
    full_resumed = _resume_from(
        delivery, output, "full_config_resume", full_recovery, evaluation_interval=0, timeout=600,
    )
    full_resolved = json.loads((full_resumed / "resolved_config.json").read_text(encoding="utf-8"))["train_config"]
    full_expected = {
        "history_dim": 128, "feature_dim": 256, "decoder_layers": 4,
        "rollout_batch_size": 4, "collect_batch_size": 4, "agent_epoch": 4,
    }
    full_actual = {
        "history_dim": full_resolved["model"]["history_dim"],
        "feature_dim": full_resolved["model"]["feature_dim"],
        "decoder_layers": full_resolved["model"]["decoder_layers"],
        "rollout_batch_size": full_resolved["rollout_batch_size"],
        "collect_batch_size": full_resolved["collect_batch_size"],
        "agent_epoch": full_resolved["ppo"]["agent_epoch"],
    }
    if full_actual != full_expected:
        raise AssertionError(f"full recovery semantics changed: {full_actual}")
    scenarios["full_config_resume"] = {
        "full_run": str(full_run), "resume_run": str(full_resumed),
        "full_update_elapsed_seconds": full_elapsed, "expected": full_expected,
        "actual": full_actual,
    }

    # eval_only uses the downloaded recovery directly and performs real C++ evaluation.
    inputs, working = _scenario_root(output, "eval_only")
    manifest = _copy_dataset(delivery, "eval_only", inputs, mutate=_eval_mutator)
    recovery_dir = inputs / "recovery"; recovery_dir.mkdir()
    shutil.copy2(split_recovery, recovery_dir / "guandan-marvel-recovery.zip")
    eval_run, _, _ = _run(manifest=manifest, input_root=inputs, working_root=working)
    eval_summary = json.loads((eval_run / "evaluation_summary.json").read_text(encoding="utf-8"))
    if eval_summary.get("status") != "complete" or eval_summary.get("complete_games") != 1:
        raise AssertionError("eval_only real C++ evaluation did not complete")
    scenarios["eval_only"] = {"run": str(eval_run), "summary": eval_summary}

    # First evaluation failure: durable bundle already exists and can directly resume.
    inputs, working = _scenario_root(output, "first_eval_failure")
    manifest = _copy_dataset(delivery, "pilot", inputs, mutate=_pilot_mutator(target=1, faults={"evaluation_fail_iterations": [1]}))
    failed_first, code, _ = _run(manifest=manifest, input_root=inputs, working_root=working, expected_codes=(1,))
    first_bundle = working / "guandan-marvel-output/guandan-marvel-recovery.zip"
    if _recovery_iteration(first_bundle) != 1:
        raise AssertionError("first evaluation failure did not preserve iteration-1 recovery")
    first_resumed = _resume_from(delivery, output, "first_eval_failure_resume", first_bundle)
    scenarios["first_eval_failure"] = {"run": str(failed_first), "exit_code": code, "recovery_iteration": 1, "resume_run": str(first_resumed)}

    # Later evaluation failure: iteration 2 is durable before evaluation starts.
    inputs, working = _scenario_root(output, "later_eval_failure")
    manifest = _copy_dataset(delivery, "pilot", inputs, mutate=_pilot_mutator(target=2, faults={"evaluation_fail_iterations": [2]}))
    failed_later, code, _ = _run(manifest=manifest, input_root=inputs, working_root=working, expected_codes=(1,))
    later_bundle = working / "guandan-marvel-output/guandan-marvel-recovery.zip"
    if _recovery_iteration(later_bundle) != 2:
        raise AssertionError("later evaluation failure did not preserve latest durable checkpoint")
    scenarios["later_eval_failure"] = {"run": str(failed_later), "exit_code": code, "recovery_iteration": 2}

    # Bundle failure at iteration 2 leaves the previous iteration-1 bundle intact.
    inputs, working = _scenario_root(output, "bundle_failure")
    manifest = _copy_dataset(delivery, "pilot", inputs, mutate=_pilot_mutator(target=2, evaluation_interval=0, faults={"bundle_fail_iterations": [2]}))
    bundle_failed, code, _ = _run(manifest=manifest, input_root=inputs, working_root=working, expected_codes=(1,))
    bundle = working / "guandan-marvel-output/guandan-marvel-recovery.zip"
    preserved_iteration = _recovery_iteration(bundle)
    if preserved_iteration not in {1, 2}:
        raise AssertionError("bundle failure left no valid recovery")
    scenarios["bundle_failure"] = {"run": str(bundle_failed), "exit_code": code, "recovery_iteration": preserved_iteration}

    # Rank-local budget skew becomes one shared decision at the completed-update boundary.
    inputs, working = _scenario_root(output, "budget_skew")
    faults = {"budget_elapsed_offset_seconds": {"0": 0, "1": 1000}, "rank_boundary_delay_seconds": {"1": 0.25}}
    manifest = _copy_dataset(delivery, "pilot", inputs, mutate=_pilot_mutator(target=3, evaluation_interval=0, faults=faults, max_wall_seconds=500))
    budget_run, code, _ = _run(manifest=manifest, input_root=inputs, working_root=working, expected_codes=(2,))
    _, budget_manifest = _checkpoint(budget_run)
    diagnostics = []
    for rank in (0, 1):
        rows = [json.loads(line) for line in (budget_run / f"ranks/rank{rank:03d}/diagnostic.jsonl").read_text(encoding="utf-8").splitlines()]
        decision = next(row for row in rows if row.get("event") == "budget_decision")
        final = rows[-1]
        diagnostics.append({"rank": rank, "decision": decision["decision"], "votes": decision["votes"], "final_event": final["event"]})
    if not all(row["decision"] is True and row["final_event"] == "budget_stopped" for row in diagnostics):
        raise AssertionError("ranks did not share the budget-stop decision")
    if int(budget_manifest["iteration"]) != 1:
        raise AssertionError("budget stop was not published at the same durable iteration")
    if len({row["agent_resume_state_hash"] for row in budget_manifest["rank_shards"]}) != 1:
        raise AssertionError("budget stop checkpoint model hashes differ across ranks")
    scenarios["budget_skew"] = {"run": str(budget_run), "exit_code": code, "iteration": 1, "diagnostics": diagnostics}

    # Actual heartbeat lines must have reached Notebook-visible stdout/console.
    console = (continuous / "console.log").read_text(encoding="utf-8")
    heartbeat_lines = [line for line in console.splitlines() if "[M07 heartbeat]" in line]
    required = ("mode=pilot", "phase=", "iteration=", "games=", "env_steps=", "eta=", "gpu0=", "r0:", "r1:")
    if not heartbeat_lines or not all(any(token in line for line in heartbeat_lines) for token in required):
        raise AssertionError("console heartbeat is incomplete")
    scenarios["heartbeat"] = {"line_count": len(heartbeat_lines), "sample": heartbeat_lines[:5]}

    summary = {
        "schema": "guandan-marvel-m07-repair-local-ddp-v1",
        "scope": "local CPU/gloo two-process repair verification; not Kaggle/T4/NCCL evidence",
        "world_size": 2,
        "backend": "cpp environment + CPU/gloo DDP",
        "kaggle_verified": False,
        "scenarios": scenarios,
        "passes": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
