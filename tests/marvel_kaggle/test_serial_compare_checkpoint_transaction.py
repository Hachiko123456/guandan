"""07A checkpoint-save-transaction counterexamples (repair round 3).

Every test executes the real production chain:
``training._save_recovery`` -> ``save_distributed_checkpoint`` ->
``TrainingCheckpointStore.save`` -> shard serialization -> manifest -> index ->
real ``load_ddp_manifest`` recovery read.  Nothing replaces the save with a fake
file write; the trainer only supplies a static, schema-complete payload (no CPU
model training).  Two-rank cases run as real Gloo processes.

Scenario coverage is listed in
``runs/t4x2_serial_compare_implementation_20261023/repair_round3/REPAIR_SPEC_AND_ACCEPTANCE.md``.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from serial_compare.common import read_json  # noqa: E402
from serial_compare.common import sha as file_sha  # noqa: E402

WORKER = REPO / "tests/marvel_training/_serial_compare_checkpoint_transaction_worker.py"
CASE_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
def _canonical(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hashed(body: dict) -> dict:
    return {**body, "hash": _canonical(body)}


def write_fixture(workdir: Path) -> Path:
    """A minimal, schema-complete identity fixture shared by every process."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "policy_375.pt").write_bytes(b"the actual 375 artifact")
    resolved = {"fixture": "serial-compare-checkpoint-transaction", "total_iterations": 3}
    semantic = _hashed({
        "schema": "guandan-marvel-training-contract-v4", "algorithm": "ippo",
        "semantic_config": {"fixture": "07A"},
    })
    backend = _hashed({
        "schema": "guandan-marvel-backend-identity-v1", "backend": "python_reference",
        "identifier": "fixture", "protocol_version": "fixture-v2",
    })
    files = [{"path": "fixture.py", "bytes": 1, "sha256": "0" * 64}]
    fingerprint = _hashed({
        "schema": "guandan-marvel-source-fingerprint-v1",
        "components": {"fixture": {"files": files, "hash": _canonical(files)}},
    })
    provenance = _hashed({
        "schema": "guandan-marvel-ddp-training-data-provenance-v1", "domain": "training",
        "manifest_hash": "a" * 64, "generator_version": "fixture",
        "seed_semantics": "fixture", "initial_env_seed": 570_700,
        "deal_hashes": ["b" * 64], "complete": True,
    })
    identity = {
        "resolved_config": resolved,
        "resolved_config_hash": _canonical(resolved),
        "semantic_contract": semantic,
        "backend_identity": backend,
        "source_fingerprint": fingerprint,
        "source_hash": fingerprint["hash"],
        "training_data_provenance": provenance,
    }
    (workdir / "fixture.json").write_text(json.dumps(identity), encoding="utf-8")
    return workdir


def launch_two_ranks(workdir: Path, *args: str, timeout: float = CASE_TIMEOUT):
    """Run the worker twice; always reap both children, even on timeout."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    base = dict(os.environ)
    base.update({
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
        "WORLD_SIZE": "2", "PYTHONPATH": str(REPO / "tools"),
    })
    processes = []
    for rank in range(2):
        env = dict(base)
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        command = [
            sys.executable, str(WORKER),
            "--workdir", str(workdir),
            "--events", str(workdir / "events.jsonl"),
            "--result", str(workdir / "result.json"),
            *args, "--rank", str(rank),
        ]
        processes.append(subprocess.Popen(
            command, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        ))
    outputs: list[str] = []
    deadline = time.time() + timeout
    try:
        for process in processes:
            remaining = max(deadline - time.time(), 1.0)
            try:
                output, _ = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                output, _ = process.communicate(timeout=None) if process.poll() is not None else ("", "")
                outputs.append(output)
                raise AssertionError(
                    f"rank process exceeded {timeout:.0f}s; capture so far:\n{output}"
                )
            outputs.append(output)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            try:
                process.communicate(timeout=5)
            except Exception:  # noqa: BLE001 - reaping must never mask the failure
                process.kill()
    return [int(process.returncode) for process in processes], outputs


def result_of(workdir: Path, rank: int) -> dict:
    path = workdir / f"result.rank{rank}.json"
    assert path.is_file(), f"rank {rank} produced no result file"
    return json.loads(path.read_text(encoding="utf-8"))


def failures(workdir: Path) -> tuple[dict, dict]:
    zero, one = result_of(workdir, 0), result_of(workdir, 1)
    assert zero["ok"] is False and one["ok"] is False, (zero, one)
    return zero, one


def assert_joint_failure(workdir: Path, *, stage: str, contains: str | None = None) -> tuple[dict, dict]:
    zero, one = failures(workdir)
    assert zero["error_stage"] == stage, zero
    assert one["error_stage"] == stage, one
    # Identical transaction record built from the gathered failure table.
    assert zero["message_sha256"] == one["message_sha256"], (zero["error"], one["error"])
    if contains is not None:
        assert contains in zero["error"], zero["error"]
    return zero, one


def _independent_manifest_hash(manifest: dict) -> str:
    return _canonical({key: value for key, value in manifest.items() if key != "manifest_hash"})


def _read_manifest(checkpoint: Path) -> dict:
    from guandan_marvel.kaggle.checkpoints import load_ddp_manifest

    manifest = load_ddp_manifest(checkpoint, verify_shards=True)
    assert manifest["manifest_hash"] == _independent_manifest_hash(manifest)
    return manifest


def _main_events(workdir: Path, rank: int) -> list[dict]:
    path = workdir / f"events.rank{rank}.jsonl"
    assert path.is_file(), f"rank {rank} produced no events"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(workdir: Path, *args: str):
    return launch_two_ranks(workdir, *args)


# ---------------------------------------------------------------------------
# Scenario 1: first two-rank save
# ---------------------------------------------------------------------------
def test_two_rank_first_save_writes_valid_shards_manifest_and_portable_index(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "1", "--state-version", "1",
    )
    assert codes == [0, 0], output
    results = [result_of(workdir, 0), result_of(workdir, 1)]
    assert all(item["ok"] for item in results)
    run = workdir / "run"
    index = read_json(run / "checkpoint_manifest.json")
    relative = index["latest"]["checkpoint"]
    assert relative == "checkpoints/iter000001"
    assert not Path(relative).is_absolute()
    checkpoint = run / relative
    assert checkpoint.is_dir()
    manifest = _read_manifest(checkpoint)
    assert manifest["iteration"] == 1 and manifest["world_size"] == 2
    assert sorted(path.name for path in checkpoint.glob("rank0*.pt")) == [
        "rank000.pt", "rank001.pt",
    ]
    assert (checkpoint / "rank000.summary.json").is_file()
    assert (checkpoint / "rank001.summary.json").is_file()
    assert index["latest"]["manifest_sha256"] == file_sha(checkpoint / "manifest.json")
    assert checkpoint.resolve().is_relative_to(run.resolve())


# ---------------------------------------------------------------------------
# Scenario 2: periodic save exactly on the last iteration, then the final save
# ---------------------------------------------------------------------------
def test_periodic_save_on_the_last_iteration_then_final_save_is_idempotent(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "run-loop", "--target-iterations", "2",
        "--save-interval", "0", "--seconds-per-iteration", "1",
    )
    assert codes == [0, 0], output
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        assert payload["ok"] is True, payload.get("error")
        assert payload["result"]["final_iteration"] == 2
        saves = [event for event in _main_events(workdir, rank) if event["kind"] == "save"]
        assert [event["iteration"] for event in saves] == [1, 2, 2], saves
        # The second save of iteration 2 is a reuse: same manifest bytes *and* mtime.
        assert saves[-1]["sha256"] == saves[-2]["sha256"]
        assert saves[-1]["mtime_ns"] == saves[-2]["mtime_ns"]
    index = read_json(workdir / "run" / "checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 2
    assert [row["iteration"] for row in index["history"]] == [1, 2]
    _read_manifest(workdir / "run" / index["latest"]["checkpoint"])


# ---------------------------------------------------------------------------
# Scenario 3: at-target resume in a new process
# ---------------------------------------------------------------------------
def test_at_target_resume_reads_recovery_with_zero_updates_and_does_not_rewrite_it(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
    )
    assert codes == [0, 0], output
    index_path = workdir / "run" / "checkpoint_manifest.json"
    index_before = read_json(index_path)
    manifest_before = (workdir / "run" / index_before["latest"]["checkpoint"] / "manifest.json")
    manifest_sha_before = file_sha(manifest_before)

    codes, output = _run(workdir, "--mode", "at-target", "--iteration", "2")
    assert codes == [0, 0], output
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        assert payload["ok"] is True, payload.get("error")
        assert payload["result"]["resumed_at_target"] is True
        assert payload["result"]["applied_updates"] == 0
        assert payload["result"]["new_iterations"] == 0
        assert payload["watched_before"] == payload["watched_after"]
    index_after = read_json(index_path)
    assert index_after["latest"]["iteration"] == 2
    assert index_after["latest"]["checkpoint"] == index_before["latest"]["checkpoint"]
    _read_manifest(workdir / "run" / index_after["latest"]["checkpoint"])
    assert file_sha(manifest_before) == manifest_sha_before


# ---------------------------------------------------------------------------
# Scenario 4: new updates after a periodic save -> the final save must persist them
# ---------------------------------------------------------------------------
def test_final_save_after_a_previous_periodic_save_persists_the_new_state(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "run-loop", "--target-iterations", "3",
        "--save-interval", "1.5", "--seconds-per-iteration", "1",
    )
    assert codes == [0, 0], output
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        assert payload["ok"] is True, payload.get("error")
        assert payload["result"]["final_iteration"] == 3
        assert payload["result"]["applied_updates"] == 3
        saves = [event for event in _main_events(workdir, rank) if event["kind"] == "save"]
        assert [event["iteration"] for event in saves] == [2, 3], saves
    index = read_json(workdir / "run" / "checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 3
    assert index["latest"]["checkpoint"] == "checkpoints/iter000003"
    periodic = _read_manifest(workdir / "run" / "checkpoints/iter000002")
    final = _read_manifest(workdir / "run" / "checkpoints/iter000003")
    assert final["iteration"] == 3 and periodic["iteration"] == 2
    assert (
        final["rank_shards"][0]["continuation_state_hash"]
        != periodic["rank_shards"][0]["continuation_state_hash"]
    )


# ---------------------------------------------------------------------------
# Scenario 5: same iteration, different state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["weights", "rng", "optimizer"])
def test_same_iteration_different_state_is_rejected_by_every_rank(tmp_path, variant):
    workdir = write_fixture(tmp_path / "case")
    # Same iteration, same state version: only one continuation component differs.
    codes, output = _run(
        workdir, "--mode", "direct-save",
        "--pre-iteration", "1", "--pre-state-version", "1",
        "--iteration", "1", "--state-version", "1", "--variant", variant,
    )
    assert codes != [0, 0], output
    assert_joint_failure(workdir, stage="validate", contains="CheckpointStateMismatch")
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        # The old complete recovery is byte-identical before and after the rejection.
        assert payload["watched_before"] == payload["watched_after"]
        assert payload["watched_before"]
    index = read_json(workdir / "run" / "checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 1
    _read_manifest(workdir / "run" / "checkpoints/iter000001")


# ---------------------------------------------------------------------------
# Scenario 6: primary mkdir / publish / index failure
# ---------------------------------------------------------------------------
def test_manifest_publish_failure_fails_every_rank_and_keeps_the_old_recovery(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save",
        "--pre-iteration", "1", "--pre-state-version", "1",
        "--iteration", "2", "--state-version", "2", "--inject", "publish",
    )
    assert codes != [0, 0], output
    assert_joint_failure(workdir, stage="publish", contains="injected failure while publishing")
    assert not (workdir / "run" / "checkpoints/iter000002").exists()
    assert not (workdir / "run" / "checkpoints/iter000002.tmp").exists()
    for rank in (0, 1):
        assert result_of(workdir, rank)["watched_before"] == result_of(workdir, rank)["watched_after"]
    old = _read_manifest(workdir / "run" / "checkpoints/iter000001")
    assert old["iteration"] == 1
    assert read_json(workdir / "run/checkpoint_manifest.json")["latest"]["iteration"] == 1


def test_index_failure_fails_every_rank_and_is_repaired_on_retry(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save",
        "--pre-iteration", "1", "--pre-state-version", "1",
        "--iteration", "2", "--state-version", "2", "--inject", "index",
    )
    assert codes != [0, 0], output
    assert_joint_failure(workdir, stage="index", contains="injected failure while updating the checkpoint index")
    run = workdir / "run"
    # The complete directory was published but the index still names iteration 1.
    manifest = _read_manifest(run / "checkpoints/iter000002")
    index = read_json(run / "checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 1
    assert index["latest"]["checkpoint"] == "checkpoints/iter000001"
    assert all(
        row["checkpoint"] != "checkpoints/iter000002" for row in index["history"]
    )
    saved_manifest_sha = file_sha(run / "checkpoints/iter000002/manifest.json")

    # A new pair of processes reuses the complete state and repairs the index
    # without rewriting the recovery (same manifest bytes).
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
    )
    assert codes == [0, 0], output
    repaired = read_json(run / "checkpoint_manifest.json")
    assert repaired["latest"]["iteration"] == 2
    assert repaired["latest"]["checkpoint"] == "checkpoints/iter000002"
    assert repaired["latest"]["manifest_sha256"] == saved_manifest_sha
    assert file_sha(run / "checkpoints/iter000002/manifest.json") == saved_manifest_sha
    assert manifest["iteration"] == 2


def test_primary_mkdir_failure_is_bounded_and_keeps_the_old_recovery(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save",
        "--pre-iteration", "1", "--pre-state-version", "1",
        "--iteration", "2", "--state-version", "2", "--inject", "mkdir",
    )
    assert codes != [0, 0], output
    assert_joint_failure(workdir, stage="prepare")
    for rank in (0, 1):
        assert result_of(workdir, rank)["watched_before"] == result_of(workdir, rank)["watched_after"]
    _read_manifest(workdir / "run" / "checkpoints/iter000001")
    assert read_json(workdir / "run/checkpoint_manifest.json")["latest"]["iteration"] == 1


# ---------------------------------------------------------------------------
# Scenario 7: non-primary shard write failure
# ---------------------------------------------------------------------------
def test_non_primary_shard_write_failure_fails_both_ranks_without_a_complete_checkpoint(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save",
        "--pre-iteration", "1", "--pre-state-version", "1",
        "--iteration", "2", "--state-version", "2", "--inject", "shard-rank1",
    )
    assert codes != [0, 0], output
    assert_joint_failure(workdir, stage="shard", contains="injected failure while writing the rank 1 shard")
    run = workdir / "run"
    assert not (run / "checkpoints/iter000002").exists()
    assert not (run / "checkpoints/iter000002.tmp").exists()
    index = read_json(run / "checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 1
    _read_manifest(run / "checkpoints/iter000001")


# ---------------------------------------------------------------------------
# Scenario 8: half-written / bad hash / wrong world size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("injection", "needle"),
    [
        ("half-written", "manifest.json"),
        ("bad-hash", "shard missing or corrupt"),
        ("wrong-world-size", "world size"),
    ],
)
def test_corrupt_destination_is_collectively_rejected(tmp_path, injection, needle):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save",
        "--pre-iteration", "1", "--pre-state-version", "1",
        "--iteration", "2", "--state-version", "2", "--inject", injection,
    )
    assert codes != [0, 0], output
    assert_joint_failure(workdir, stage="preflight", contains=needle)
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        assert payload["watched_before"] == payload["watched_after"]
    # The last valid recovery is preserved and the broken directory is not
    # mistaken for a complete one.
    old = _read_manifest(workdir / "run" / "checkpoints/iter000001")
    assert old["iteration"] == 1
    index = read_json(workdir / "run/checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 1


# ---------------------------------------------------------------------------
# Scenario 9: absolute / relative run_dir
# ---------------------------------------------------------------------------
def test_relative_run_dir_still_writes_a_portable_index(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "1", "--state-version", "1",
        "--relative-run",
    )
    assert codes == [0, 0], output
    run = workdir / "run"
    index = read_json(run / "checkpoint_manifest.json")
    assert index["latest"]["checkpoint"] == "checkpoints/iter000001"
    assert (run / index["latest"]["checkpoint"]).resolve().is_relative_to(run.resolve())
    _read_manifest(run / index["latest"]["checkpoint"])


def test_index_rejects_a_checkpoint_dir_outside_the_run_root(tmp_path):
    from guandan_marvel.kaggle.checkpoints import update_checkpoint_index

    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = {"iteration": 1, "manifest_hash": "0" * 64, "created_at": "test-only"}
    with pytest.raises(ValueError):
        update_checkpoint_index(run, checkpoint_dir=outside, ddp_manifest=manifest)
    inside = run / "checkpoints/iter000001"
    inside.mkdir(parents=True)
    (inside / "manifest.json").write_text("{}", encoding="utf-8")
    # The old serial-compare caller passed a relative path; the helper must keep
    # rejecting it so the fix has to normalise the caller instead.
    with pytest.raises(ValueError):
        update_checkpoint_index(run, checkpoint_dir=inside.relative_to(run), ddp_manifest=manifest)


# ---------------------------------------------------------------------------
# Scenario 10: world_size=1 (old single-process behavior)
# ---------------------------------------------------------------------------
class _SingleProcessTrainer:
    world_size = 1
    is_primary = True
    distributed = None
    opponent_pool = None

    def __init__(self, iteration: int):
        self.iteration = int(iteration)
        self.learner_updates = int(iteration)

    def save_checkpoint(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"single-process-checkpoint-bytes")


def test_single_process_save_never_opens_a_process_group(tmp_path, monkeypatch):
    import torch.distributed as dist

    from serial_compare import training

    def forbidden(*_args, **_kwargs):
        raise AssertionError("world_size=1 must not call a distributed collective")

    monkeypatch.setattr(dist, "all_reduce", forbidden)
    monkeypatch.setattr(dist, "all_gather_object", forbidden)
    monkeypatch.setattr(dist, "barrier", forbidden)
    monkeypatch.chdir(tmp_path)
    path = training._save_recovery(_SingleProcessTrainer(2), Path("run"))
    assert Path(path).is_file()
    index = read_json(tmp_path / "run/checkpoint_manifest.json")
    assert index["latest"]["checkpoint"] == "checkpoints/iter000002.pt"
    assert (tmp_path / "run" / index["latest"]["checkpoint"]).is_file()
    assert index["latest"]["sha256"] == file_sha(Path(path))


def test_single_process_save_writes_a_portable_index(tmp_path):
    from serial_compare import training

    run_dir = (tmp_path / "absolute-run").resolve()
    training._save_recovery(_SingleProcessTrainer(1), run_dir)
    training._save_recovery(_SingleProcessTrainer(2), run_dir)
    index = read_json(run_dir / "checkpoint_manifest.json")
    assert index["latest"]["checkpoint"] == "checkpoints/iter000002.pt"
    assert [row["checkpoint"] for row in index["history"]] == [
        "checkpoints/iter000001.pt", "checkpoints/iter000002.pt",
    ]
    assert (run_dir / index["latest"]["checkpoint"]).is_file()


# ---------------------------------------------------------------------------
# Extra: same iteration / same state reuse (direct save twice)
# ---------------------------------------------------------------------------
def test_two_rank_repeated_save_of_the_same_state_reuses_the_recovery(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "1", "--state-version", "1",
        "--saves", "2",
    )
    assert codes == [0, 0], output
    for rank in (0, 1):
        saves = [event for event in _main_events(workdir, rank) if event["kind"] == "save"]
        assert len(saves) == 2, saves
        assert saves[0]["sha256"] == saves[1]["sha256"]
        assert saves[0]["mtime_ns"] == saves[1]["mtime_ns"]
    index = read_json(workdir / "run/checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 1
    assert [row["iteration"] for row in index["history"]] == [1]
    _read_manifest(workdir / "run/checkpoints/iter000001")


# ---------------------------------------------------------------------------
# Repair round 4: the versioned continuation identity closes the AMP-scaler /
# AMP-state / curriculum-offset idempotency gap.  Every case below runs the real
# two-rank save chain; nothing is faked.
# ---------------------------------------------------------------------------
def _stored_payload(checkpoint: Path, rank: int) -> dict:
    from guandan_marvel.training.checkpoints import TrainingCheckpointStore

    return dict(TrainingCheckpointStore.load(checkpoint / f"rank{rank:03d}.pt", device="cpu"))


def _rewrite_manifest_body(checkpoint: Path, mutate) -> None:
    """Rewrite a manifest the way an older writer did, recomputing its hash."""
    path = checkpoint / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    mutate(body)
    body["manifest_hash"] = _canonical(body)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _strip_versioned_identity(checkpoint: Path) -> None:
    """Simulate a pre-round-4 (legacy) manifest that never stored the identity."""

    def mutate(body: dict) -> None:
        for row in body["rank_shards"]:
            row.pop("continuation_identity", None)

    _rewrite_manifest_body(checkpoint, mutate)


def _assert_original_recovery_intact(workdir: Path, checkpoint: Path) -> dict:
    index = read_json(workdir / "run" / "checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 2
    assert index["latest"]["checkpoint"] == "checkpoints/iter000002"
    assert index["latest"]["manifest_sha256"] == file_sha(checkpoint / "manifest.json")
    manifest = _read_manifest(checkpoint)
    assert manifest["iteration"] == 2
    return manifest


def test_same_iteration_only_amp_scaler_changes_is_rejected_on_both_ranks(tmp_path):
    """Codex counterexample 1: a different AMP scaler must not be silently reused.

    Requested second save: scale 2048.  Stored (and preserved) scale: 1024.
    """
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
        "--saves", "2", "--mutate-field", "scaler",
        "--scaler-scale", "1024", "--curriculum-offset", "375",
    )
    assert codes != [0, 0], output
    zero, one = assert_joint_failure(
        workdir, stage="validate", contains="CheckpointStateMismatch",
    )
    assert "continuation_identity" in zero["error"], zero["error"]
    checkpoint = workdir / "run/checkpoints/iter000002"
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        # The requested state really differed from the published state.
        assert payload["requested_state"]["scaler_scale"] == 2048.0, payload
        assert payload["requested_state"]["iteration"] == 2
        # Nothing under the destination changed and the old recovery is readable.
        assert payload["watched_before"] == payload["watched_after"]
        assert payload["watched_before"]
        stored = _stored_payload(checkpoint, rank)
        assert stored["grad_scaler_state"]["scale"] == 1024.0
        assert stored["curriculum_offset"] == 375
    manifest = _assert_original_recovery_intact(workdir, checkpoint)
    for row in manifest["rank_shards"]:
        assert row["continuation_identity"]["schema"] == (
            "guandan-marvel-continuation-identity-v2"
        )
    assert one["error_stage"] == "validate" and one["message_sha256"] == zero["message_sha256"]


def test_same_iteration_only_curriculum_offset_changes_is_rejected_on_both_ranks(tmp_path):
    """Codex counterexample 2: a different curriculum origin must not be reused.

    Requested second save: offset 376.  Stored (and preserved) offset: 375.
    """
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
        "--saves", "2", "--mutate-field", "curriculum",
        "--scaler-scale", "1024", "--curriculum-offset", "375",
    )
    assert codes != [0, 0], output
    zero, one = assert_joint_failure(
        workdir, stage="validate", contains="CheckpointStateMismatch",
    )
    assert "continuation_identity" in zero["error"], zero["error"]
    checkpoint = workdir / "run/checkpoints/iter000002"
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        assert payload["requested_state"]["curriculum_offset"] == 376, payload
        assert payload["requested_state"]["iteration"] == 2
        assert payload["watched_before"] == payload["watched_after"]
        assert payload["watched_before"]
        stored = _stored_payload(checkpoint, rank)
        assert stored["curriculum_offset"] == 375
        assert stored["grad_scaler_state"]["scale"] == 1024.0
    manifest = _assert_original_recovery_intact(workdir, checkpoint)
    for row in manifest["rank_shards"]:
        assert row["continuation_identity"]["schema"] == (
            "guandan-marvel-continuation-identity-v2"
        )
    assert one["error_stage"] == "validate" and one["message_sha256"] == zero["message_sha256"]


def test_same_iteration_only_amp_bookkeeping_changes_is_rejected_on_both_ranks(tmp_path):
    """The third newly covered field: AMP bookkeeping (skipped_steps)."""
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
        "--saves", "2", "--mutate-field", "amp",
        "--scaler-scale", "1024", "--curriculum-offset", "375", "--amp-skipped-steps", "3",
    )
    assert codes != [0, 0], output
    zero, _ = assert_joint_failure(
        workdir, stage="validate", contains="CheckpointStateMismatch",
    )
    assert "continuation_identity" in zero["error"], zero["error"]
    checkpoint = workdir / "run/checkpoints/iter000002"
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        assert payload["requested_state"]["amp_skipped_steps"] == 4, payload
        assert payload["watched_before"] == payload["watched_after"]
        assert _stored_payload(checkpoint, rank)["amp"]["skipped_steps"] == 3
    _assert_original_recovery_intact(workdir, checkpoint)


def test_identical_amp_scaler_and_curriculum_state_is_reused_without_rewriting(tmp_path):
    """Positive control: identical state *including* the newly covered fields."""
    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
        "--saves", "2", "--scaler-scale", "1024", "--curriculum-offset", "375",
        "--amp-skipped-steps", "7",
    )
    assert codes == [0, 0], output
    for rank in (0, 1):
        payload = result_of(workdir, rank)
        assert payload["ok"] is True, payload.get("error")
        saves = [event for event in _main_events(workdir, rank) if event["kind"] == "save"]
        assert len(saves) == 2, saves
        assert saves[0]["sha256"] == saves[1]["sha256"]
        assert saves[0]["mtime_ns"] == saves[1]["mtime_ns"]
        assert saves[-1]["curriculum_offset"] == 375
    index = read_json(workdir / "run/checkpoint_manifest.json")
    assert index["latest"]["checkpoint"] == "checkpoints/iter000002"
    assert [row["iteration"] for row in index["history"]] == [2]
    checkpoint = workdir / "run/checkpoints/iter000002"
    _read_manifest(checkpoint)
    from guandan_marvel.kaggle.checkpoints import load_rank_checkpoint_for_resume

    payload, _ = load_rank_checkpoint_for_resume(
        checkpoint, rank=0, expected_world_size=2, expected_source_commit="f" * 40,
    )
    assert payload["grad_scaler_state"]["scale"] == 1024.0
    assert payload["curriculum_offset"] == 375


def test_identical_state_with_deleted_index_rebuilds_it_without_rewriting(tmp_path):
    workdir = write_fixture(tmp_path / "case")
    args = (
        "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
        "--scaler-scale", "1024", "--curriculum-offset", "375", "--amp-skipped-steps", "7",
    )
    codes, output = _run(workdir, *args)
    assert codes == [0, 0], output
    checkpoint = workdir / "run/checkpoints/iter000002"
    manifest_sha = file_sha(checkpoint / "manifest.json")
    (workdir / "run" / "checkpoint_manifest.json").unlink()

    codes, output = _run(workdir, *args)
    assert codes == [0, 0], output
    for rank in (0, 1):
        assert result_of(workdir, rank)["ok"] is True, result_of(workdir, rank).get("error")
    index = read_json(workdir / "run/checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 2
    assert index["latest"]["checkpoint"] == "checkpoints/iter000002"
    assert index["latest"]["manifest_sha256"] == manifest_sha
    # The recovery itself was reused, not rewritten.
    assert file_sha(checkpoint / "manifest.json") == manifest_sha
    _read_manifest(checkpoint)


def test_legacy_manifest_without_versioned_identity_is_never_silently_reused(tmp_path):
    """A pre-round-4 manifest cannot prove the full state, so it fails closed."""
    from guandan_marvel.kaggle.checkpoints import load_rank_checkpoint_for_resume

    workdir = write_fixture(tmp_path / "case")
    args = (
        "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
        "--scaler-scale", "1024", "--curriculum-offset", "375",
    )
    codes, output = _run(workdir, *args)
    assert codes == [0, 0], output
    checkpoint = workdir / "run/checkpoints/iter000002"
    index_manifest_sha = read_json(workdir / "run/checkpoint_manifest.json")["latest"]["manifest_sha256"]
    assert index_manifest_sha == file_sha(checkpoint / "manifest.json")
    _strip_versioned_identity(checkpoint)
    legacy_manifest_sha = file_sha(checkpoint / "manifest.json")
    destination_before = {
        path.name: file_sha(path) for path in sorted(checkpoint.iterdir()) if path.is_file()
    }
    # Old-format compatibility: the legacy manifest still loads for resume.
    payload, manifest = load_rank_checkpoint_for_resume(
        checkpoint, rank=0, expected_world_size=2, expected_source_commit="f" * 40,
    )
    assert payload["iteration"] == 2
    assert "continuation_identity" not in manifest["rank_shards"][0]
    _read_manifest(checkpoint)

    # Re-saving the identical state must NOT be silently treated as reusable.
    codes, output = _run(workdir, *args)
    assert codes != [0, 0], output
    zero, one = assert_joint_failure(workdir, stage="validate", contains="legacy recovery")
    assert "guandan-marvel-continuation-identity-v2" in zero["error"], zero["error"]
    assert one["message_sha256"] == zero["message_sha256"]
    # The legacy recovery is neither rewritten nor overwritten.
    assert file_sha(checkpoint / "manifest.json") == legacy_manifest_sha
    destination_after = {
        path.name: file_sha(path) for path in sorted(checkpoint.iterdir()) if path.is_file()
    }
    assert destination_after == destination_before
    index = read_json(workdir / "run/checkpoint_manifest.json")
    assert index["latest"]["iteration"] == 2
    assert index["latest"]["checkpoint"] == "checkpoints/iter000002"
    # The rejected save did not touch the index either.
    assert index["latest"]["manifest_sha256"] == index_manifest_sha


def test_manifest_with_a_tampered_continuation_identity_fails_resume(tmp_path):
    """The new identity is verified against the shard, not merely trusted."""
    from guandan_marvel.kaggle.checkpoints import load_rank_checkpoint_for_resume

    workdir = write_fixture(tmp_path / "case")
    codes, output = _run(
        workdir, "--mode", "direct-save", "--iteration", "2", "--state-version", "2",
        "--scaler-scale", "1024", "--curriculum-offset", "375",
    )
    assert codes == [0, 0], output
    checkpoint = workdir / "run/checkpoints/iter000002"

    def mutate(body: dict) -> None:
        body["rank_shards"][0]["continuation_identity"]["hash"] = "0" * 64

    _rewrite_manifest_body(checkpoint, mutate)
    with pytest.raises(ValueError, match="continuation_identity"):
        load_rank_checkpoint_for_resume(
            checkpoint, rank=0, expected_world_size=2, expected_source_commit="f" * 40,
        )


def test_versioned_continuation_identity_is_deterministic_and_covers_new_fields():
    import torch

    from guandan_marvel.kaggle.checkpoints import (
        CONTINUATION_IDENTITY_KEYS,
        CONTINUATION_IDENTITY_SCHEMA,
        continuation_identity,
    )

    for key in (
        "grad_scaler_state", "amp", "curriculum_offset",
        "agent_resume_state_dict", "rng_state", "sampler_state", "iteration", "counters",
    ):
        assert key in CONTINUATION_IDENTITY_KEYS
    base = {
        "agent_resume_state_dict": {"layer.weight": torch.tensor([[1.0]], dtype=torch.float32)},
        "agent_state_dict": {"layer.weight": torch.tensor([[1.0]], dtype=torch.float32)},
        "agent_optimizer_state_dict": {"state": {}, "param_groups": []},
        "replay_buffer_state": None,
        "sampler_state": {"marker": 1},
        "rng_state": {"marker": 1},
        "schedule_state": {"marker": 1},
        "merge_state": {"update_count": 1},
        "iteration": 2,
        "durable_updates": 2,
        "learner_updates": 2,
        "durable_learner_updates": 2,
        "counters": {"total_env_steps": 20},
        "grad_scaler_state": {"scale": 1024.0, "growth_factor": 2.0},
        "amp": {"enabled": False, "skipped_steps": 0},
        "curriculum_offset": 375,
    }
    identity = continuation_identity(base)
    assert identity["schema"] == CONTINUATION_IDENTITY_SCHEMA
    # Fixed field order: insertion order of the payload cannot change the digest.
    reordered = dict(reversed(list(base.items())))
    assert continuation_identity(reordered) == identity
    assert continuation_identity(base) == identity

    def variant(**changes):
        payload = dict(base)
        payload.update(changes)
        return continuation_identity(payload)

    assert variant(grad_scaler_state={"scale": 2048.0, "growth_factor": 2.0}) != identity
    assert variant(curriculum_offset=376) != identity
    assert variant(amp={"enabled": False, "skipped_steps": 1}) != identity
    # A legacy payload without the new keys computes explicitly and never raises.
    legacy = {key: value for key, value in base.items()
              if key not in ("grad_scaler_state", "amp", "curriculum_offset")}
    legacy_identity = continuation_identity(legacy)
    assert legacy_identity["schema"] == CONTINUATION_IDENTITY_SCHEMA
    # Absence is distinguishable from a stored value (e.g. an explicit zero).
    assert legacy_identity != variant(curriculum_offset=0)
    assert variant(curriculum_offset=0) != identity
