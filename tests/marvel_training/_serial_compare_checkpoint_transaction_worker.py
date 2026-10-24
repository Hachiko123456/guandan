"""Two-rank worker that drives the REAL checkpoint save transaction.

Unlike ``_serial_compare_ddp_loop_worker.py`` this worker never replaces
``training._save_recovery`` with a fake: every save executes the production
``_save_recovery`` -> ``save_distributed_checkpoint`` -> ``TrainingCheckpointStore.save``
-> manifest -> index chain.  Only the trainer's forward/backward math is a static
fixture; the checkpoint payload is a schema-complete mapping of small tensors.

Failure injection is limited to precise filesystem/stage boundaries (a wrong
directory, a corrupted shard, a blocked rename); the positive save path never
disables a hash/identity check.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

SCHEMA = "guandan-marvel-training-checkpoint-v5"


def canonical(value: object) -> str:
    """Local canonical hash; deliberately not the production helper."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hashed(body: dict) -> dict:
    return {**body, "hash": canonical(body)}


def load_identity(workdir: Path) -> dict:
    return json.loads((workdir / "fixture.json").read_text(encoding="utf-8"))


def build_payload(
    identity: dict, *, iteration: int, state_version: int, variant: str = "none",
    scaler_scale: float = 1024.0, curriculum_offset: int = 0, amp_skipped_steps: int = 0,
) -> dict:
    """A minimal but schema-complete training checkpoint payload.

    ``variant`` isolates one continuation-state component so a same-iteration
    save can differ *only* in the weights, *only* in the RNG/sampler state, or
    *only* in the optimizer summary.  ``grad_scaler_state`` and ``amp`` are
    always present and ``curriculum_offset`` is written only when non-zero,
    mirroring ``LocalTrainerBase._checkpoint_common`` exactly.
    """
    value = float(state_version) + 0.5 + (1.0 if variant == "weights" else 0.0)
    tensor = torch.tensor([[value, 1.0]], dtype=torch.float32)
    rng_state = {"marker": int(state_version)}
    sampler_state_state = {"marker": int(state_version)}
    if variant == "rng":
        rng_state["nonce"] = "different-rng"
        sampler_state_state["nonce"] = "different-rng"
    optimizer_state: dict = {"state": {}, "param_groups": []}
    if variant == "optimizer":
        optimizer_state = {"state": {"p0": {"step": 1}}, "param_groups": [{"lr": 1e-4}]}
    return {
        "schema": SCHEMA,
        "algorithm": "ippo",
        "architecture": "fixture",
        "protocol_version": "fixture-v2",
        "backend": identity["backend_identity"]["backend"],
        "backend_identity": deepcopy(identity["backend_identity"]),
        "semantic_contract": deepcopy(identity["semantic_contract"]),
        "training_semantics_profile": "fixture",
        "policy_regularizer_mode": "fixture",
        "reward_semantics": "fixture",
        "reward_scale": 1.0,
        "source_fingerprint": deepcopy(identity["source_fingerprint"]),
        "source_hash": identity["source_hash"],
        "source_parent_commit": "f" * 40,
        "resolved_config": deepcopy(identity["resolved_config"]),
        "resolved_config_hash": identity["resolved_config_hash"],
        "iteration": int(iteration),
        "durable_updates": int(iteration),
        "learner_updates": int(iteration),
        "durable_learner_updates": int(iteration),
        "iteration_status": "completed",
        "counters": {
            "total_env_steps": int(iteration) * 10,
            "total_game_steps": int(iteration) * 20,
            "completed_games_total": int(iteration),
        },
        # Real production field names, as persisted by _checkpoint_common.
        "grad_scaler_state": {
            "scale": float(scaler_scale), "growth_factor": 2.0,
            "backoff_factor": 0.5, "growth_interval": 2000, "_growth_tracker": 0,
        },
        "amp": {
            "enabled": False, "device_type": "cpu", "autocast_dtype": "torch.float32",
            "scaler_enabled": False, "skipped_steps": int(amp_skipped_steps),
        },
        **(
            {"curriculum_offset": int(curriculum_offset)}
            if int(curriculum_offset) else {}
        ),
        "rng_state": rng_state,
        "sampler_state": sampler_state_state,
        "schedule_state": {"marker": int(state_version)},
        "merged_policy": {"identity": None, "beta": 1.0, "not_smoothed_when_beta_zero": False},
        "training_data_provenance": deepcopy(identity["training_data_provenance"]),
        "agent_resume_state_dict": {"layer.weight": tensor.clone()},
        "agent_state_dict": {"layer.weight": tensor.clone()},
        "agent_optimizer_state_dict": optimizer_state,
        "merge_state": {"last_iteration": int(iteration), "update_count": int(iteration), "total_weight": 1.0},
    }


class FixtureTrainer:
    """Static fixture trainer; only iteration/state bookkeeping changes."""

    world_size = 2
    opponent_pool = None

    def __init__(self, identity: dict, context, *, iteration: int, state_version: int,
                 seconds_per_iteration: float = 0.0, clock: list | None = None,
                 variant: str = "none", scaler_scale: float = 1024.0,
                 curriculum_offset: int = 0, amp_skipped_steps: int = 0):
        self.identity = identity
        self.distributed = context
        self.is_primary = bool(context.is_primary)
        self.iteration = int(iteration)
        self.state_version = int(state_version)
        self.learner_updates = int(iteration)
        self.seconds_per_iteration = float(seconds_per_iteration)
        self.clock = clock
        self.variant = str(variant)
        self.scaler_scale = float(scaler_scale)
        self.curriculum_offset = int(curriculum_offset)
        self.amp_skipped_steps = int(amp_skipped_steps)
        self.cfg = None

    def checkpoint_payload(self) -> dict:
        return build_payload(
            self.identity, iteration=self.iteration, state_version=self.state_version,
            variant=self.variant, scaler_scale=self.scaler_scale,
            curriculum_offset=self.curriculum_offset,
            amp_skipped_steps=self.amp_skipped_steps,
        )

    def apply_mutation(self, field: str) -> None:
        """Change exactly one persisted continuation field (counterexample helper)."""
        if field == "scaler":
            self.scaler_scale = self.scaler_scale * 2.0
        elif field == "curriculum":
            self.curriculum_offset += 1
        elif field == "amp":
            self.amp_skipped_steps += 1
        else:
            raise ValueError(f"unknown mutation {field!r}")

    def load_checkpoint_payload(self, payload):
        self.identity = {
            key: deepcopy(payload[key]) for key in (
                "resolved_config", "resolved_config_hash", "semantic_contract",
                "backend_identity", "source_fingerprint", "source_hash",
                "training_data_provenance",
            )
        }
        self.iteration = int(payload["iteration"])
        self.learner_updates = int(payload.get("learner_updates", payload["iteration"]))
        self.state_version = int(payload["rng_state"]["marker"])
        self.variant = "none"
        return "exact_resume"

    def train_iteration(self):
        # A real collective keeps the process group aligned, exactly like the
        # production iteration; only the numeric work is a static fixture.
        tensor = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(tensor)
        if self.clock is not None:
            self.clock[0] += self.seconds_per_iteration
        self.iteration += 1
        self.learner_updates += 1
        self.state_version += 1
        return {"statistics/learner_updates": 1, "macro/value_ev": 0.5}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_dir():
        return {}
    return {
        item.relative_to(path).as_posix(): sha(item)
        for item in sorted(path.rglob("*")) if item.is_file()
    }


def manifest_snapshot(destination: Path | None) -> dict | None:
    if destination is None:
        return None
    manifest = Path(destination) / "manifest.json"
    if not manifest.is_file():
        return None
    return {
        "sha256": sha(manifest),
        "mtime_ns": manifest.stat().st_mtime_ns,
    }


class Injection:
    """Precise stage/file failure injection; no hash or identity check is weakened."""

    def __init__(self, name: str | None, *, run_dir: Path, destination: Path, rank: int):
        self.name = name
        self.run_dir = Path(run_dir).resolve()
        self.destination = Path(destination).resolve()
        self.rank = int(rank)
        self.restore_actions: list = []

    # Filesystem preconditions are created once and then published by a barrier;
    # per-rank I/O failure hooks are installed independently on each rank.
    PREP_INJECTIONS = ("mkdir", "half-written", "bad-hash", "wrong-world-size")

    def apply(self, context) -> None:
        if not self.name:
            return
        if self.name in self.PREP_INJECTIONS:
            if self.rank == 0:
                self._apply_prep()
            context.barrier()
            return
        self._apply_local()

    def _apply_prep(self) -> None:
        name = self.name
        if name == "mkdir":
            temporary = self.destination.with_name(self.destination.name + ".tmp")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(b"blocking file: a directory cannot be created here\n")
            return
        if name == "half-written":
            self.destination.mkdir(parents=True, exist_ok=True)
            (self.destination / "rank000.pt").mkdir(parents=True, exist_ok=True)
            return
        if name == "bad-hash":
            self._copy_valid_then(self.destination, self._corrupt_shard)
            return
        if name == "wrong-world-size":
            self._copy_valid_then(self.destination, self._rewrite_world_size)
            return
        raise ValueError(f"unknown injection {name!r}")

    def _apply_local(self) -> None:
        name = self.name
        if name == "publish":
            self._patch_os_replace(
                lambda src, dst: Path(src).resolve() == self.destination.with_name(
                    self.destination.name + ".tmp"
                ) and Path(dst).resolve() == self.destination,
                OSError("injected failure while publishing the checkpoint manifest"),
            )
            return
        if name == "index":
            self._patch_os_replace(
                lambda src, dst: Path(dst).name == "checkpoint_manifest.json"
                and Path(dst).parent.resolve() == self.run_dir,
                OSError("injected failure while updating the checkpoint index"),
            )
            return
        if name == "shard-rank1":
            if self.rank == 1:
                self._patch_torch_save()
            return
        raise ValueError(f"unknown injection {name!r}")

    # -- helpers ---------------------------------------------------------
    def _valid_source(self) -> Path:
        candidates = sorted(
            path for path in (self.run_dir / "checkpoints").glob("iter[0-9]*")
            if path.is_dir() and (path / "manifest.json").is_file()
        )
        candidates = [path for path in candidates if path.resolve() != self.destination]
        if not candidates:
            raise RuntimeError("injection needs a pre-existing valid recovery")
        return candidates[-1]

    def _copy_valid_then(self, destination: Path, mutate) -> None:
        source = self._valid_source()
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        mutate(destination)

    def _corrupt_shard(self, destination: Path) -> None:
        shard = destination / "rank000.pt"
        shard.write_bytes(shard.read_bytes() + b"corrupted-after-the-fact")

    def _rewrite_world_size(self, destination: Path) -> None:
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in manifest["rank_shards"]]
        extra = dict(rows[1])
        extra["rank"] = 2  # same shard file, third rank: a genuinely wrong world size
        rows.append(extra)
        body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        body["rank_shards"] = rows
        body["world_size"] = 3
        body["manifest_hash"] = canonical(body)
        manifest_path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")

    def _patch_os_replace(self, predicate, error: OSError) -> None:
        import guandan_marvel.kaggle.checkpoints as checkpoints

        original = checkpoints.os.replace

        def guarded(src, dst):
            if predicate(src, dst):
                raise error
            return original(src, dst)

        checkpoints.os.replace = guarded
        self.restore_actions.append(lambda: setattr(checkpoints.os, "replace", original))

    def _patch_torch_save(self) -> None:
        import guandan_marvel.training.checkpoints as store

        original = store.torch.save

        def guarded(obj, path, *args, **kwargs):
            if Path(path).name.startswith("rank001.pt.tmp-"):
                raise OSError("injected failure while writing the rank 1 shard")
            return original(obj, path, *args, **kwargs)

        store.torch.save = guarded
        self.restore_actions.append(lambda: setattr(store.torch, "save", original))

    def restore(self) -> None:
        for action in reversed(self.restore_actions):
            try:
                action()
            except Exception:  # noqa: BLE001
                pass
        self.restore_actions.clear()


def resolve_run_dir(workdir: Path, relative: bool) -> Path:
    if relative:
        os.chdir(str(workdir))
        return Path("run")
    return workdir / "run"


def build_plan_file(workdir: Path, *, target_iterations: int | None = None):
    from serial_compare.common import build_plan, sha as common_sha

    source = workdir / "policy_375.pt"
    plan = build_plan(
        source={key: "x" * 64 for key in (
            "archive_sha256", "content_manifest_hash", "core_source_hash",
            "backend_hash", "native_sha256")}
        | {"git_commit": "f" * 40, "runtime_path": "/tmp", "python": "3.13.15",
           "torch": "fake", "native_name": "n"},
        models={name: {"sha256": "1" * 64, "iteration": 1, "training_resume_allowed": False}
                for name in ("p375", "p250", "p500")},
    )
    plan["models"]["p375"]["sha256"] = common_sha(source)
    if target_iterations is not None:
        plan["arms"]["control"]["max_new_iterations"] = int(target_iterations)
    return plan


class FakeConfig:
    seed = 570_700
    training_data_generator_version = "checkpoint-transaction-worker"
    training_context_schedule = {"schedule": "fixture"}
    training_context_curriculum = {"curriculum": "fixture"}

    def to_dict(self):
        return {
            "training_context_schedule": self.training_context_schedule,
            "training_context_curriculum": self.training_context_curriculum,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct-save", "run-loop", "at-target"), required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--state-version", type=int, default=1)
    parser.add_argument("--pre-iteration", type=int, default=None)
    parser.add_argument("--pre-state-version", type=int, default=None)
    parser.add_argument("--saves", type=int, default=1)
    parser.add_argument("--mutate-between-saves", type=int, default=0)
    parser.add_argument("--variant", choices=("none", "weights", "rng", "optimizer"), default="none")
    parser.add_argument("--scaler-scale", type=float, default=1024.0)
    parser.add_argument("--curriculum-offset", type=int, default=0)
    parser.add_argument("--amp-skipped-steps", type=int, default=0)
    parser.add_argument(
        "--mutate-field", choices=("none", "scaler", "curriculum", "amp"), default="none",
    )
    parser.add_argument("--inject", type=str, default=None)
    parser.add_argument("--relative-run", action="store_true")
    parser.add_argument("--target-iterations", type=int, default=2)
    parser.add_argument("--save-interval", type=float, default=0.0)
    parser.add_argument("--seconds-per-iteration", type=float, default=1.0)
    parser.add_argument("--pg-timeout", type=float, default=12.0)
    args = parser.parse_args(argv)
    rank = int(args.rank)

    dist.init_process_group("gloo", timeout=timedelta(seconds=float(args.pg_timeout)))
    from guandan_marvel.runtime.distributed import DistributedContext

    context = DistributedContext(
        True, rank == 0, int(os.environ["WORLD_SIZE"]), rank, rank, torch.device("cpu"),
    )
    workdir = args.workdir
    identity = load_identity(workdir)
    run_dir = resolve_run_dir(workdir, bool(args.relative_run))
    events: list[dict] = []
    payload: dict = {"ok": False, "mode": args.mode, "rank": rank}

    from serial_compare import training

    injection: Injection | None = None
    watched: Path | None = None
    watched_before: dict[str, str] = {}
    requested_state: dict | None = None
    try:
        if args.mode == "direct-save":
            checkpoint_dir = run_dir / "checkpoints"
            if args.pre_iteration is not None:
                watched = checkpoint_dir / f"iter{int(args.pre_iteration):06d}"
            elif int(args.saves) > 1:
                # Repeated save of the same iteration: watch the destination so a
                # rejection can prove it was left byte-identical.
                watched = checkpoint_dir / f"iter{int(args.iteration):06d}"
            else:
                watched = None
            trainer = FixtureTrainer(
                identity, context,
                iteration=int(args.pre_iteration if args.pre_iteration is not None else args.iteration),
                state_version=int(
                    args.pre_state_version if args.pre_state_version is not None else args.state_version
                ),
                scaler_scale=float(args.scaler_scale),
                curriculum_offset=int(args.curriculum_offset),
                amp_skipped_steps=int(args.amp_skipped_steps),
            )
            if watched is not None and args.pre_iteration is not None:
                path = training._save_recovery(trainer, run_dir)
                events.append({"kind": "save", "phase": "pre", "rank": rank,
                               "iteration": trainer.iteration, **manifest_snapshot(Path(path))})
                watched_before = directory_hashes(watched)
            trainer.iteration = int(args.iteration)
            trainer.state_version = int(args.state_version)
            trainer.learner_updates = int(args.iteration)
            trainer.variant = str(args.variant)
            injection = Injection(
                args.inject, run_dir=run_dir,
                destination=(run_dir / "checkpoints" / f"iter{int(args.iteration):06d}"), rank=rank,
            )
            injection.apply(context)
            for index in range(max(int(args.saves), 1)):
                if index:
                    if args.mutate_between_saves:
                        trainer.state_version += int(args.mutate_between_saves)
                    if args.mutate_field != "none":
                        trainer.apply_mutation(args.mutate_field)
                requested_state = {
                    "iteration": int(trainer.iteration),
                    "scaler_scale": trainer.scaler_scale,
                    "curriculum_offset": trainer.curriculum_offset,
                    "amp_skipped_steps": trainer.amp_skipped_steps,
                }
                path = training._save_recovery(trainer, run_dir)
                snapshot = manifest_snapshot(Path(path))
                events.append({"kind": "save", "phase": "main", "seq": index, "rank": rank,
                               "iteration": trainer.iteration, "state_version": trainer.state_version,
                               "scaler_scale": trainer.scaler_scale,
                               "curriculum_offset": trainer.curriculum_offset,
                               "amp_skipped_steps": trainer.amp_skipped_steps,
                               **snapshot})
                if index == 0 and watched is not None and not watched_before:
                    # Snapshot the published recovery before the second attempt.
                    watched_before = directory_hashes(watched)
            payload["manifest"] = manifest_snapshot(
                run_dir / "checkpoints" / f"iter{int(args.iteration):06d}"
            )
        elif args.mode == "run-loop":
            from serial_compare.common import sha as common_sha

            plan = build_plan_file(workdir, target_iterations=int(args.target_iterations))
            clock = [0.0]
            real_save = training._save_recovery

            def recorded(trainer, save_dir):
                path = real_save(trainer, save_dir)
                snapshot = manifest_snapshot(Path(path))
                events.append({
                    "kind": "save", "rank": rank, "iteration": int(trainer.iteration),
                    "state_version": int(trainer.state_version), **snapshot,
                })
                return path

            training._save_recovery = recorded
            try:
                result = training.run_training_arm(
                    plan=plan, arm="control", run_dir=run_dir, source_payload={},
                    source_model_path=workdir / "policy_375.pt", model_paths={},
                    game=object(), device="cpu", ddp_enabled=True, world_size=2,
                    clock=lambda: clock[0], max_seconds=100.0,
                    save_interval_seconds=float(args.save_interval),
                    trainer_factory=lambda cfg: FixtureTrainer(
                        identity, context, iteration=0, state_version=0,
                        seconds_per_iteration=float(args.seconds_per_iteration), clock=clock,
                    ),
                    import_policy=lambda *a, **k: {"hash": "import"},
                    initial_state_fingerprint=lambda trainer: {"hash": "fp"},
                    config_override=FakeConfig(), pool_entries_override=(), rank=rank,
                    policy_exporter=lambda trainer, path, iteration: Path(path).write_bytes(b"policy"),
                )
            finally:
                training._save_recovery = real_save
            payload["result"] = result
            payload["expected_final_state_version"] = int(result["final_iteration"])
        elif args.mode == "at-target":
            plan = build_plan_file(workdir)
            recovery = run_dir / "checkpoints" / f"iter{int(args.iteration):06d}"
            watched = recovery
            watched_before = directory_hashes(watched)
            result = training.run_training_arm(
                plan=plan, arm="control", run_dir=run_dir, source_payload={},
                source_model_path=workdir / "policy_375.pt", model_paths={},
                game=object(), device="cpu", ddp_enabled=True, world_size=2,
                clock=lambda: 0.0, max_seconds=100.0, save_interval_seconds=0.0,
                trainer_factory=lambda cfg: FixtureTrainer(
                    identity, context, iteration=0, state_version=0,
                ),
                import_policy=lambda *a, **k: {"hash": "import"},
                initial_state_fingerprint=lambda trainer: {"hash": "fp"},
                config_override=FakeConfig(), pool_entries_override=(), rank=rank,
                resume_from=recovery, iteration_limit=0,
                policy_exporter=lambda trainer, path, iteration: Path(path).write_bytes(b"policy"),
            )
            payload["result"] = result
        payload["ok"] = True
    except BaseException as error:  # noqa: BLE001 - recorded honestly
        message = str(error)
        payload.update({
            "ok": False,
            "error_type": type(error).__name__,
            "error": message,
            "error_stage": getattr(error, "stage", None),
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "traceback": traceback.format_exc(),
        })
        events.append({
            "kind": "failure", "rank": rank,
            "error_type": type(error).__name__, "stage": getattr(error, "stage", None),
            "t": round(time.monotonic(), 3),
        })
    finally:
        if injection is not None:
            injection.restore()
        if requested_state is not None:
            # Requested values of the *last attempted* save, recorded even when
            # that save was rejected, so tests can compare them with what the
            # published shard actually stores.
            payload["requested_state"] = requested_state
        if watched is not None:
            payload["watched_before"] = watched_before
            payload["watched_after"] = directory_hashes(watched)
        payload["events"] = events
        events_path = args.events.with_name(args.events.stem + f".rank{rank}" + args.events.suffix)
        events_path.write_text(
            "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
            encoding="utf-8",
        )
        result_path = args.result.with_name(args.result.stem + f".rank{rank}" + args.result.suffix)
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
