# SPDX-License-Identifier: MIT
"""Production wiring of the serial comparison for the Kaggle notebook.

Every heavy step is a *separate process* (two DDP ranks for an arm, two sharded
evaluation workers) so a phase never inherits the previous phase's model, cache
or RNG.  Locally this module is imported for its contracts only; without two T4
GPUs and the original Linux native it is never executed.
"""
from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import sys
import time
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .budget import BudgetLedger
from .common import (
    PlanError, atomic_json, canonical, read_json, sha, validate_deal_manifest, validate_plan,
)
from .evaluation import (
    collect_records, confirmation_tasks, development_tasks, merge_records,
    pending_tasks, schedule, summarize_confirmation, summarize_development,
)
from .orchestrator import PhaseContext, SerialCompareOrchestrator
from .reporting import ensure_minimal_result, write_reports

INPUTS_SCHEMA = "guandan-serial-compare-inputs-v1"
FINALS_SCHEMA = "guandan-serial-compare-finals-v1"
PREFLIGHT_SCHEMA = "guandan-serial-compare-preflight-v1"
PREFLIGHT_RESUME_ITERATION_WINDOW = 3

if PREFLIGHT_RESUME_ITERATION_WINDOW <= 0:
    raise ValueError("preflight resume iteration window must be positive")


def latest_recovery_files(run_dir: str | Path) -> list[Path]:
    """Files that constitute an arm's latest recovery (index + one checkpoint).

    Used both when declaring the training phase's outputs (so deleting a
    recovery invalidates the completion marker) and when bundling the result
    ZIP.  Older checkpoints are deliberately excluded.
    """
    run_dir = Path(run_dir)
    index = run_dir / "checkpoint_manifest.json"
    if not index.is_file():
        return []
    payload = read_json(index)
    latest = dict(payload.get("latest") or {})
    value = latest.get("checkpoint")
    if not value:
        raise PlanError("checkpoint index exists but declares no latest checkpoint")
    target = (run_dir / str(value)).resolve()
    if not target.is_relative_to(run_dir.resolve()) or not target.exists():
        raise PlanError(f"latest recovery checkpoint is missing: {target}")
    files: list[Path] = [index]
    if target.is_dir():
        files.extend(sorted(path for path in target.rglob("*") if path.is_file()))
    else:
        files.append(target)
    return files


def _bounded_excerpt(value: str, limit: int = 6000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[excerpt truncated; see full log]"


def _log_excerpt(path: Path, *, start_offset: int = 0) -> str:
    """Stream one log, retaining bounded head, first error and tail excerpts."""
    head: list[str] = []
    first_error: list[str] = []
    tail: deque[str] = deque(maxlen=40)
    found_error = False
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            stream.seek(start_offset)
            # Limit each read as well: even an enormous single line must not
            # cause the parent to allocate an unbounded diagnostic message.
            for line in iter(lambda: stream.readline(2000), ""):
                line = line.rstrip("\r\n")
                if len(head) < 12:
                    head.append(line)
                tail.append(line)
                if not found_error and (
                    "Traceback (most recent call last)" in line
                    or "Error:" in line or "Exception:" in line
                ):
                    found_error = True
                if found_error and len(first_error) < 40:
                    first_error.append(line)
    except OSError as error:
        return f"log unavailable: {error}"
    tail_text = "\n".join(tail) or "[empty]"
    if len(tail_text) > 6000:
        tail_text = "[earlier tail omitted; see full log]\n" + tail_text[-6000:]
    return "\n".join((
        "log head:\n" + _bounded_excerpt("\n".join(head) or "[empty]"),
        "first traceback/exception:\n" + _bounded_excerpt("\n".join(first_error) or "[none found]"),
        "log tail:\n" + tail_text,
    ))


def _worker_error_directory(command: Sequence[str], cwd: Path) -> Path | None:
    run_dir = None
    for index, part in enumerate(command):
        if part == "--run-dir" and index + 1 < len(command):
            run_dir = Path(command[index + 1])
            break
        if str(part).startswith("--run-dir="):
            run_dir = Path(str(part).split("=", 1)[1])
            break
    if run_dir is None:
        return None
    return (cwd / run_dir / "worker_errors").resolve()


def _worker_error_snapshot(command: Sequence[str], cwd: Path) -> dict[str, tuple]:
    directory = _worker_error_directory(command, cwd)
    if directory is None:
        return {}
    result = {}
    for path in sorted(islice(directory.glob("rank*.json"), 4)):
        try:
            if path.is_symlink():
                continue
            stat = path.stat()
            with path.open("rb") as stream:
                raw = stream.read(65537)
            result[str(path)] = (stat.st_mtime_ns, stat.st_size, hashlib.sha256(raw).hexdigest())
        except OSError:
            continue
    return result


def _worker_error_excerpt(command: Sequence[str], cwd: Path, *, previous_errors: Mapping[str, tuple] | None = None) -> str:
    """Read current-attempt errors only in the explicit arm run directory."""
    directory = _worker_error_directory(command, cwd)
    if directory is None:
        return ""
    excerpts: list[str] = []
    try:
        # No recursive glob, only a few bounded JSON files.  Unknown fields
        # (in particular any environment dump) are deliberately omitted.
        for path in sorted(islice(directory.glob("rank*.json"), 4)):
            if path.is_symlink():
                continue
            try:
                with path.open("rb") as stream:
                    raw = stream.read(65537)
                if len(raw) > 65536:
                    continue
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size, hashlib.sha256(raw).hexdigest())
                if previous_errors is not None and previous_errors.get(str(path)) == signature:
                    continue
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    continue
                fields = {
                    name: payload[name] for name in (
                        "rank", "local_rank", "world_size", "stage", "type", "message", "traceback",
                    ) if name in payload
                }
                if fields:
                    excerpts.append(f"{path}:\n" + _bounded_excerpt(
                        json.dumps(fields, ensure_ascii=False, indent=2),
                    ))
            except (OSError, ValueError, UnicodeError):
                continue  # A partial/malformed sidecar must not hide the log.
    except OSError:
        pass
    return _bounded_excerpt("\n".join(excerpts), 12000)


def _failure_diagnostics(log: Path, command: Sequence[str], cwd: Path, *, start_offset: int = 0, previous_errors: Mapping[str, tuple] | None = None) -> str:
    parts = [f"full log: {log.resolve()}"]
    worker_error = _worker_error_excerpt(command, cwd, previous_errors=previous_errors)
    if worker_error:
        parts.append("worker errors:\n" + worker_error)
    parts.append(_log_excerpt(log, start_offset=start_offset))
    return "\n".join(parts)


class ProductionActions:
    """Real actions; the notebook constructs exactly one of these."""

    def __init__(
        self,
        *,
        plan: Mapping[str, object],
        root: str | Path,
        dataset: str | Path,
        source: str | Path,
        python: str | None = None,
        ranks: int = 2,
        clock: Callable[[], float] = time.monotonic,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.plan = plan
        self.root = Path(root)
        self.dataset = Path(dataset)
        self.source = Path(source)
        self.python = python or sys.executable
        self.ranks = int(ranks)
        self.clock = clock
        self.runner = runner or subprocess.run
        self.orchestrator: SerialCompareOrchestrator | None = None

    # -- process helpers --------------------------------------------------
    def _run(self, command: Sequence[str], *, log: Path, timeout: float, cwd: Path | None = None) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        print(f"[SERIAL-COMPARE PROCESS START] log={log.resolve()}", flush=True)
        started = self.clock()
        limit = max(1.0, float(timeout))
        start_offset = log.stat().st_size if log.exists() else 0
        previous_errors = _worker_error_snapshot(command, cwd or self.source)
        try:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"[cmd] {' '.join(str(part) for part in command)}\n")
                stream.flush()
                if self.runner is subprocess.run:
                    # POSIX: put torchrun and its workers in a new session so a
                    # timeout can kill the whole process group (no orphan GPU
                    # processes).  Windows has no process groups; kill the child.
                    process = subprocess.Popen(
                        list(command), stdout=stream, stderr=subprocess.STDOUT,
                        cwd=str(cwd or self.source), env=env,
                        start_new_session=(os.name != "nt"),
                    )
                    try:
                        process.communicate(timeout=limit)
                    except subprocess.TimeoutExpired:
                        if os.name != "nt":
                            try:
                                os.killpg(
                                    os.getpgid(process.pid),
                                    getattr(signal, "SIGKILL", signal.SIGTERM),
                                )
                            except (ProcessLookupError, PermissionError):
                                pass
                        process.kill()
                        process.communicate()
                        raise
                    completed = subprocess.CompletedProcess(list(command), process.returncode)
                else:
                    # Injected runner (tests) keeps the simple contract.
                    completed = self.runner(
                        list(command), stdout=stream, stderr=subprocess.STDOUT,
                        timeout=limit, cwd=str(cwd or self.source), env=env,
                    )
        except Exception as error:
            diagnostics = _failure_diagnostics(log, command, cwd or self.source,
                                               start_offset=start_offset, previous_errors=previous_errors)
            error.add_note(diagnostics)
            # Keep TimeoutExpired (and injected-runner exceptions) intact for
            # callers, including evaluation's budget-stop handling.  Notes are
            # shown by tracebacks; stderr also covers callers that catch it.
            print(f"{type(error).__name__}: {error}\n{diagnostics}", file=sys.stderr, flush=True)
            raise
        elapsed = self.clock() - started
        if int(getattr(completed, "returncode", 0)) != 0:
            raise RuntimeError(
                f"subprocess failed with exit {completed.returncode} after {elapsed:.1f}s; "
                f"{_failure_diagnostics(log, command, cwd or self.source, start_offset=start_offset, previous_errors=previous_errors)}"
            )
        print(f"[SERIAL-COMPARE PROCESS COMPLETE] seconds={elapsed:.1f} log={log.resolve()}", flush=True)

    def _arm_command(self, arm: str, run_dir: Path, result: Path, *, max_seconds: float,
                     iteration_limit: int | None, resume_from: Path | None,
                     seed_offset: int = 0, hard_reserve_seconds: float = 0.0) -> list[str]:
        command = [
            self.python, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={self.ranks}", "--max-restarts=0", "--monitor-interval=2",
            "-m", "serial_compare.arm_worker",
            "--plan", str(self.dataset / "plan.json"),
            "--arm", arm,
            "--dataset", str(self.dataset),
            "--source", str(self.source),
            "--run-dir", str(run_dir),
            "--result", str(result),
            "--max-seconds", str(float(max_seconds)),
        ]
        if iteration_limit is not None:
            command += ["--iteration-limit", str(int(iteration_limit))]
        if resume_from is not None:
            command += ["--resume-from", str(resume_from)]
        if int(seed_offset):
            command += ["--seed-offset", str(int(seed_offset))]
        # The arm's ``max_seconds`` is already capped at its quota, which the
        # budget gate guarantees excludes the evaluation/finalise reserve; the
        # CLI value is forwarded for callers that additionally protect time
        # *inside* the quota (see the documented value passed by ``train``).
        command += ["--hard-reserve-seconds", str(float(hard_reserve_seconds))]
        return command

    def _latest_recovery(self, run_dir: Path) -> Path | None:
        """Locate the latest recovery, failing closed on any inconsistency.

        An *absent* index means "this arm never trained" (fresh import).  An
        index that exists but points at a missing or corrupted checkpoint is a
        broken run: silently returning ``None`` would retrain from 375 and
        overwrite the run directory, destroying recoverable progress.
        """
        index = run_dir / "checkpoint_manifest.json"
        if not index.is_file():
            return None
        payload = read_json(index)
        if not isinstance(payload, Mapping):
            raise PlanError("checkpoint index is not an object")
        latest = dict(payload.get("latest") or {})
        value = latest.get("checkpoint")
        if not value:
            raise PlanError("checkpoint index exists but declares no latest checkpoint")
        candidate = (run_dir / str(value)).resolve()
        if not candidate.is_relative_to(run_dir.resolve()):
            raise PlanError("checkpoint index points outside the arm run directory")
        if not candidate.exists():
            raise PlanError(f"latest recovery checkpoint is missing: {candidate}")
        if candidate.is_dir():
            declared = str(latest.get("manifest_sha256") or "")
            manifest_path = candidate / "manifest.json"
            if not manifest_path.is_file():
                raise PlanError(f"latest recovery directory has no manifest: {candidate}")
            if declared and sha(manifest_path) != declared:
                raise PlanError("latest recovery manifest hash changed since it was indexed")
            from guandan_marvel.kaggle.checkpoints import load_ddp_manifest

            try:
                load_ddp_manifest(candidate, verify_shards=True)
            except Exception as error:  # noqa: BLE001 - surface as a domain error
                raise PlanError(f"latest recovery checkpoint is corrupt: {error}") from error
        else:
            if candidate.stat().st_size == 0:
                raise PlanError(f"latest recovery checkpoint is empty: {candidate}")
            declared = str(latest.get("sha256") or "")
            if declared and sha(candidate) != declared:
                raise PlanError("latest recovery checkpoint hash changed since it was indexed")
        return candidate

    # -- actions ----------------------------------------------------------
    def input_check(self, context: PhaseContext) -> Mapping[str, object]:
        validate_plan(self.plan)
        models = {}
        for name in ("p375", "p250", "p500"):
            expected = str(self.plan["models"][name]["sha256"])
            path = context.model_paths[name]
            if not path.is_file():
                raise PlanError(f"model file is missing: {path}")
            actual = sha(path)
            if actual != expected:
                raise PlanError(f"model {name!r} hash mismatch: {actual} != {expected}")
            models[name] = {"path": str(path), "sha256": actual}
        deals = {}
        manifests = {
            domain: read_json(self.dataset / f"deals_{domain}.json")
            for domain in ("preflight", "development", "confirmation")
        }
        for domain, manifest in manifests.items():
            validate_deal_manifest(manifest, expected_domain=domain)
            deals[domain] = {"hash": manifest["hash"], "deals": manifest["deal_count"]}
        from .runtime_validation import validate_runtime_deals

        runtime_deals = validate_runtime_deals(manifests)
        print(f"[SERIAL-COMPARE INPUTS] {runtime_deals['rules_configs_checked']} rule configs "
              f"and {runtime_deals['native_resets']} native resets validated before training", flush=True)
        from .common import assert_domains_disjoint, assert_training_seeds_disjoint

        disjoint = assert_domains_disjoint(manifests)
        # Compare the *physical* first deal of every training rank against every
        # evaluation deal; comparing seed names alone is not enough (spec 8.1).
        isolation = assert_training_seeds_disjoint(
            self.plan, manifests, ranks=self.ranks, physical=True,
        )
        payload = {
            "schema": INPUTS_SCHEMA,
            "plan_hash": self.plan["plan_hash"],
            "models": models,
            "deals": deals,
            "runtime_deals": runtime_deals,
            "domains": disjoint,
            "isolation": isolation,
            "source": str(self.source),
            "fabledan": "NOT_RUN",
            "final_holdout": "NOT_TOUCHED",
        }
        payload["hash"] = canonical(payload)
        path = context.phase_dir / "inputs_identity.json"
        atomic_json(path, payload)
        ensure_minimal_result(self.root, plan_hash=str(self.plan["plan_hash"]))
        return {"status": "complete", "outputs": [path], "metadata": {"inputs_hash": payload["hash"]}}

    def preflight(self, context: PhaseContext, arm: str) -> Mapping[str, object]:
        phase_dir = context.phase_dir
        phase_dir.mkdir(parents=True, exist_ok=True)
        run_dir = phase_dir / "run"
        result_path = phase_dir / "arm_result.json"
        allowance = float(context.budget.remaining_for_phase(context.phase))
        if allowance <= 60.0:
            raise RuntimeError(f"preflight/{arm} has no budget left")
        smoke_seconds = min(300.0, max(60.0, allowance / 4.0))
        # Preflight *training* must live in its own seed namespace: reusing the
        # formal seed would replay the same early deals before the real arm.
        from .common import DOMAIN_BASES

        preflight_offset = int(DOMAIN_BASES["preflight"]) - int(self.plan["arms"][arm]["seed"])
        self._run(
            self._arm_command(arm, run_dir, result_path, max_seconds=smoke_seconds,
                              iteration_limit=2, resume_from=None, seed_offset=preflight_offset),
            log=phase_dir / "train.console.log", timeout=smoke_seconds + 240.0,
        )
        first = read_json(result_path)
        first_start_iteration = int(first.get("start_iteration", 0))
        first_final_iteration = int(first.get("final_iteration", 0))
        if first.get("status") != "complete" or first_final_iteration - first_start_iteration != 2:
            raise RuntimeError("preflight initial training did not complete its two-iteration window")
        recovery = Path(str(first["recovery"]))
        if not recovery.exists():
            raise RuntimeError("preflight did not produce a recoverable checkpoint")
        resume_dir = phase_dir / "resume"
        resume_result = phase_dir / "resume_result.json"
        self._run(
            self._arm_command(arm, resume_dir, resume_result, max_seconds=smoke_seconds,
                              iteration_limit=PREFLIGHT_RESUME_ITERATION_WINDOW,
                              resume_from=recovery, seed_offset=preflight_offset),
            log=phase_dir / "resume.console.log", timeout=smoke_seconds + 240.0,
        )
        second = read_json(resume_result)
        if int(second.get("start_iteration", 0)) <= int(first.get("start_iteration", 0)):
            raise RuntimeError("preflight resume did not restore the trained iteration")
        resume_start_iteration = int(second.get("start_iteration", 0))
        resume_final_iteration = int(second.get("final_iteration", 0))
        if (
            second.get("status") != "complete"
            or resume_final_iteration - resume_start_iteration != PREFLIGHT_RESUME_ITERATION_WINDOW
        ):
            raise RuntimeError(
                "preflight resume did not complete its bounded iteration window"
            )
        resume_iterations = [
            item for item in second.get("iterations", []) if isinstance(item, Mapping)
        ]
        # Small complete evaluation on the preflight namespace: 2 deals x 2
        # assignments against the frozen 375 policy.
        manifest = read_json(self.dataset / "deals_preflight.json")
        from .evaluation import tasks_for

        deals = validate_deal_manifest(manifest, expected_domain="preflight")[:2]
        tasks = tasks_for(deals, ["control_vs_375" if arm == "control" else "candidate_vs_375"])
        task_file = phase_dir / "tasks.json"
        atomic_json(task_file, tasks)
        games_dir = phase_dir / "games"
        games_dir.mkdir(parents=True, exist_ok=True)
        finals = phase_dir / "preflight_models.json"
        atomic_json(finals, {
            "control_final": str(second["policy"]),
            "candidate_final": str(second["policy"]),
        })
        self._run_eval(
            tasks, games_dir, finals_path=finals, log=phase_dir / "eval.console.log",
            timeout=min(600.0, max(60.0, allowance / 2.0)),
        )
        records, ignored = collect_records(games_dir)
        from .evaluation import validate_preflight_records

        models = dict(self.plan["models"])
        for name in ("control_final", "candidate_final"):
            models[name] = {"sha256": sha(Path(str(second["policy"])))}
        records = validate_preflight_records(records, tasks, models=models, ignored_partial_lines=ignored)
        payload = {
            "schema": PREFLIGHT_SCHEMA,
            "arm": arm,
            "status": "pass",
            "train_updates": int(first.get("applied_updates", 0)),
            "resume_updates": int(second.get("applied_updates", 0)),
            "iteration_after_resume": resume_final_iteration,
            "resume_iteration_window": PREFLIGHT_RESUME_ITERATION_WINDOW,
            "resume_new_iterations": resume_final_iteration - resume_start_iteration,
            "resume_kl_stops": sum(bool(item.get("kl_stopped")) for item in resume_iterations),
            "evaluated_games": len(records),
            "initial_state_fingerprint": second.get("initial_state_fingerprint"),
            "opponent_pool_manifest_hash": second.get("opponent_pool_manifest_hash"),
            "kl_stop_path_exercised": any(bool(item.get("kl_stopped")) for item in resume_iterations),
        }
        if payload["train_updates"] <= 0 or payload["resume_updates"] <= 0:
            raise RuntimeError(f"preflight/{arm} produced no effective update")
        payload["hash"] = canonical(payload)
        path = phase_dir / "preflight.json"
        atomic_json(path, payload)
        atomic_json(phase_dir / "skipped_games.json", {"ignored_lines": ignored})
        return {
            "status": "pass",
            "outputs": [path],
            "metadata": {
                "preflight_hash": payload["hash"],
                "initial_state_fingerprint": payload["initial_state_fingerprint"],
                "pool_manifest_hash": payload["opponent_pool_manifest_hash"],
            },
        }

    def freeze_budget(self, context: PhaseContext, resolved: Mapping[str, object]) -> Mapping[str, object]:
        path = context.phase_dir / "resolved_budget.json"
        atomic_json(path, dict(resolved))
        if not bool(resolved.get("frozen")):
            raise RuntimeError("the resolved budget must be frozen before training")
        return {"status": "complete", "outputs": [path], "metadata": {"resolved": dict(resolved)}}

    def train(self, context: PhaseContext, arm: str, *, max_seconds: float) -> Mapping[str, object]:
        run_dir = context.phase_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = context.phase_dir / "arm_result.json"
        resumed = self._latest_recovery(run_dir)
        # The orchestrator's ``max_seconds`` is this arm's *quota* (already
        # excluding the evaluation/finalise reserve).  Give the worker the
        # global safe allowance as a wall-clock cap but pass the hard reserve
        # so its per-iteration p95 check stops before touching that reserve.
        reserve = float(context.budget.hard_reserve_remaining())
        remaining_total = float(context.budget.remaining_total())
        if float(max_seconds) <= 0.0:
            raise RuntimeError(f"{arm} has no remaining arm quota")
        if remaining_total <= reserve:
            raise RuntimeError(
                f"{arm} cannot start without eating the evaluation/finalise reserve "
                f"(remaining {remaining_total:.0f}s <= reserve {reserve:.0f}s)"
            )
        effective_max = min(remaining_total, float(max_seconds) + reserve)
        if effective_max <= 0.0:
            raise RuntimeError(f"{arm} has no safe wall-clock allowance left")
        self._run(
            self._arm_command(
                arm, run_dir, result_path, max_seconds=effective_max,
                iteration_limit=None, resume_from=resumed,
                hard_reserve_seconds=reserve,
            ),
            log=context.phase_dir / "train.console.log", timeout=effective_max + 600.0,
        )
        result = read_json(result_path)
        applied = int(result.get("applied_updates", 0))
        if applied <= 0 and not bool(result.get("resumed_at_target")):
            raise RuntimeError(f"arm {arm!r} reported no applied optimizer updates")
        policy = Path(str(result.get("policy") or ""))
        if not policy.is_file():
            raise RuntimeError(f"arm {arm!r} did not export a final policy")
        # Bind the latest complete recovery into the phase marker: deleting or
        # corrupting it invalidates "train complete" instead of silently
        # resuming from nothing.
        recovery = latest_recovery_files(run_dir)
        if not recovery:
            raise RuntimeError(f"arm {arm!r} finished without a recoverable checkpoint")
        atomic_json(context.phase_dir / "arm_summary.json", result)
        outputs = [result_path, policy, context.phase_dir / "arm_summary.json", *recovery]
        return {"status": "complete", "outputs": outputs, "metadata": dict(result)}

    def freeze_finals(self, context: PhaseContext) -> Mapping[str, object]:
        from .training import verify_arm_pair

        control = context.arm_result("control")
        candidate = context.arm_result("candidate")
        pair = verify_arm_pair(control, candidate)
        finals_dir = context.phase_dir / "finals"
        finals_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = {"schema": FINALS_SCHEMA, "pair": pair}
        for arm, payload in (("control", control), ("candidate", candidate)):
            source = Path(str(payload.get("policy") or ""))
            if not source.is_file():
                raise RuntimeError(f"{arm} final policy is missing")
            target = finals_dir / f"{arm}_final.pt"
            target.write_bytes(source.read_bytes())
            digest = sha(target)
            if digest != payload.get("policy_sha256"):
                raise RuntimeError(f"{arm} final policy changed after export")
            record[f"{arm}_final"] = str(target)
            record[f"{arm}_final_sha256"] = digest
        record["hash"] = canonical(record)
        path = context.phase_dir / "finals.json"
        atomic_json(path, record)
        return {"status": "complete", "outputs": [path], "metadata": dict(record)}

    def _run_eval(
        self, tasks: Sequence[Mapping[str, object]], games_dir: Path, *,
        finals_path: Path, log: Path, timeout: float,
    ) -> None:
        shards = schedule(tasks, ranks=self.ranks)
        for rank, shard in enumerate(shards):
            if not shard:
                continue
            task_file = games_dir / f"tasks.rank{rank}.json"
            atomic_json(task_file, [dict(task) for task in shard])
            out = games_dir / f"games.rank{rank}.jsonl"
            command = [
                self.python, "-m", "serial_compare.eval_worker",
                "--plan", str(self.dataset / "plan.json"),
                "--dataset", str(self.dataset),
                "--source", str(self.source),
                "--tasks", str(task_file),
                "--finals", str(finals_path),
                "--out", str(out),
                "--device", f"cuda:{rank}",
            ]
            self._run(
                command, log=games_dir / f"worker.rank{rank}.console.log",
                timeout=float(timeout),
            )

    def evaluate(self, context: PhaseContext, phase: str, tasks: list[dict[str, object]]) -> Mapping[str, object]:
        domain = "confirmation" if phase.endswith("confirmation_primary") else "development"
        manifest = read_json(self.dataset / f"deals_{domain}.json")
        expected = confirmation_tasks(manifest) if domain == "confirmation" else development_tasks(manifest)
        if tasks:
            if {task["task_id"] for task in tasks} != {task["task_id"] for task in expected}:
                raise RuntimeError("pre-registered task set differs from the deal manifest")
        tasks = expected
        if context.orchestrator is not None:
            context.orchestrator.record_tasks(phase, tasks)
        games_dir = context.phase_dir / "games"
        games_dir.mkdir(parents=True, exist_ok=True)
        finals = context.phase_dir / "evaluation_models.json"
        finals_payload = {
            "control_final": str(context.final_paths().get("control_final")),
            "candidate_final": str(context.final_paths().get("candidate_final")),
        }
        atomic_json(finals, finals_payload)

        existing, ignored = collect_records(games_dir)
        models = dict(self.plan["models"])
        for name in ("control_final", "candidate_final"):
            value = finals_payload.get(name)
            if value and Path(str(value)).is_file():
                models[name] = {"sha256": sha(Path(str(value)))}
        merged, stats = merge_records(existing, [], tasks=tasks, models=models)
        pending = pending_tasks(tasks, merged)
        allowance = float(context.budget.remaining_for_phase(phase))
        status = "complete"
        timed_out = False
        if pending:
            if allowance <= 60.0:
                status = "incomplete"
            else:
                try:
                    self._run_eval(
                        pending, games_dir, finals_path=finals,
                        log=games_dir / "eval.console.log", timeout=allowance - 30.0,
                    )
                except subprocess.TimeoutExpired:
                    # Per-game persistence means the completed games survive;
                    # a timeout is a budget stop, not a lost shard and not a
                    # technical failure of the run.
                    timed_out = True
                fresh, fresh_ignored = collect_records(games_dir)
                ignored = ignored + fresh_ignored
                merged, stats = merge_records(existing, fresh, tasks=tasks, models=models)
                pending = pending_tasks(tasks, merged)
                status = "complete" if not pending and not timed_out else "incomplete"
        summary = (
            summarize_confirmation(merged, tasks) if domain == "confirmation"
            else summarize_development(merged, tasks)
        )
        # A truncated game is a persisted result but not a completed one: the
        # phase must say "incomplete" even though there is nothing left to
        # schedule, otherwise a formally incomplete evaluation would be marked
        # complete at the phase level.
        if not bool(summary.get("complete")):
            status = "incomplete"
        summary_path = context.phase_dir / f"{domain}_summary.json"
        atomic_json(summary_path, summary)
        games_path = context.phase_dir / f"{domain}_games.jsonl"
        with games_path.open("w", encoding="utf-8") as stream:
            for record in merged:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        atomic_json(context.phase_dir / "ignored_lines.json", {"ignored": ignored})
        outputs = [summary_path, games_path]
        return {
            "status": status,
            "outputs": outputs,
            "summary": summary,
            "metadata": {
                "domain": domain,
                "expected_games": len(tasks),
                "completed_games": len(merged),
                "pending_games": len(pending),
                "timed_out": timed_out,
                "duplicate_identical_records": stats.get("duplicate_identical_records", 0),
                "verdict": dict(summary.get("verdict") or {}),
            },
        }

    def summarize(self, context: PhaseContext) -> Mapping[str, object]:
        orchestrator = context.orchestrator
        if orchestrator is not None:
            # The orchestrator records this phase *after* the action returns, so
            # without a provisional entry RESULT.json would never mention
            # ``summarize_and_package``.  The real entry replaces it afterwards.
            phases = dict(orchestrator.state.get("phases") or {})
            phases["summarize_and_package"] = {
                "status": "complete", "seconds": None, "marker_hash": None,
                "metadata": {"provisional": True},
            }
            orchestrator.state["phases"] = phases
        result = orchestrator.result() if orchestrator is not None else {}
        artifacts = self.root
        artifacts.mkdir(parents=True, exist_ok=True)
        # The bundle carries each arm's latest recovery plus a complete
        # ``input/`` tree (plan/deals/models/source/native/drone) so extracting
        # it yields a valid RESUME_RUN_ROOT for the bootstrap.
        written = write_reports(artifacts, result, resume_inputs=self.dataset)
        archive = Path(str(written["archive"]))
        # Keep the historical download location, byte-identical to the bound copy.
        download = self.root.parent / archive.name
        if download.resolve() != archive.resolve():
            download.parent.mkdir(parents=True, exist_ok=True)
            download.write_bytes(archive.read_bytes())
        outputs = [
            written["result"], artifacts / "REPORT.md", artifacts / "report.html",
            artifacts / "recovery_index.json", archive,
        ]
        return {
            "status": "complete",
            "outputs": outputs,
            "metadata": {
                key: written[key] for key in ("result_sha256", "archive", "archive_sha256")
            } | {"download_archive": str(download) if download.is_file() else None},
        }


def run_notebook(
    *,
    dataset: str | Path,
    run_root: str | Path,
    source: str | Path,
    clock: Callable[[], float] = time.monotonic,
    ranks: int = 2,
) -> dict[str, object]:
    dataset = Path(dataset)
    plan = read_json(dataset / "plan.json")
    validate_plan(plan)
    model_paths = {
        "p375": dataset / "policy_375.pt",
        "p250": dataset / "anchor_250.pt",
        "p500": dataset / "policy_500.pt",
    }
    run_root = Path(run_root)
    actions = ProductionActions(
        plan=plan, root=run_root / "artifacts", dataset=dataset, source=source,
        ranks=ranks, clock=clock,
    )
    orchestrator = SerialCompareOrchestrator(
        plan, root=run_root / "artifacts", actions=actions,
        model_paths=model_paths, clock=clock,
    )
    actions.orchestrator = orchestrator
    return orchestrator.run()


__all__ = ["FINALS_SCHEMA", "INPUTS_SCHEMA", "PREFLIGHT_SCHEMA", "ProductionActions", "run_notebook"]
