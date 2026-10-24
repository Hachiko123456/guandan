# SPDX-License-Identifier: MIT
"""MARVEL v14 phase-4 local validation (no long training, no Kaggle job).

Runs the real IPPO macro trainer on the C++ backend:

1. policy-only warm start from an evaluation checkpoint;
2. two full iterations, checkpoint;
3. build a recovery bundle (checkpoint + provenance + manifest);
4. resume for two more iterations with exact-resume classification;
5. assert macro/curriculum/context/reward continuity, RNG/optimizer/scaler and
   completed-game budget exactness, source-identity match and rejection of an
   incompatible recovery.

Evidence (JSON + metrics JSONL + recovery zip + sha256) is written under
``--output``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import torch

from guandan_marvel.config import (
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE, MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
    ModelConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.context_schedule import build_rank_leader_schedule
from guandan_marvel.training.checkpoints import canonical_json_hash, classify_resume
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.smoke import CppSmokeGame

CURRICULUM = {
    "schema": "guandan-marvel-context-curriculum-v1",
    "old_context": {"level_rank": 15, "leader_seat": 0, "previous_result": None},
    "warmup_iterations": 0,
    "blend_iterations": 2,
    "resolution": 52,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config(run_dir: Path, *, logging_dir: Path | None = None) -> tuple[TrainConfig, CppSmokeGame]:
    game = CppSmokeGame(512)
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name="ippo",
        logging_dir=str(logging_dir if logging_dir is not None else run_dir / "logging"),
        run_name="phase4-validation",
        seed=9137,
        rollout_batch_size=2,
        collect_batch_size=2,
        replay_buffer_scale=1,
        mini_batches=1,
        total_iterations=4,
        max_sampler_bytes_per_rank=1 << 30,
        amp_enabled=False,
        checkpoint_replay_buffer=True,
        training_context_schedule=build_rank_leader_schedule(),
        training_context_curriculum=CURRICULUM,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            agent_epoch=1,
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
        ),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


def _build_recovery(
    *, run_dir: Path, checkpoint: Path, payload: dict[str, Any], destination: Path,
) -> dict[str, object]:
    """Atomic single-process recovery bundle (checkpoint + provenance + manifest)."""
    manifest_body = {
        "schema": "guandan-marvel-phase4-local-recovery-v1",
        "iteration": int(payload["iteration"]),
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "semantic_contract_hash": canonical_json_hash(payload["semantic_contract"]),
        "source_hash": str(payload["source_hash"]),
        "sampler_state_schema": str(payload["sampler_state"]["schema"]),
        "training_data_provenance_hash": str(payload["training_data_provenance"]["hash"]),
        "resume_classification": str(payload["resume_classification"]),
    }
    manifest = {**manifest_body, "manifest_hash": canonical_json_hash(manifest_body)}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr("recovery_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive.write(checkpoint, f"checkpoint/{checkpoint.name}")
    temporary.replace(destination)
    return manifest


def _require(condition: bool, message: str, evidence: dict[str, object]) -> None:
    if not condition:
        raise AssertionError(message)
    evidence.setdefault("assertions", []).append(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/marvel_v14_phase4/validation")
    parser.add_argument(
        "--warm-start",
        default="runs/marvel_v14_phase3/ab-smoke/synthetic-v13-warm-start.pt",
    )
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        shutil.rmtree(output)
    (output / "run").mkdir(parents=True)
    (output / "resumed").mkdir(parents=True)
    started = time.perf_counter()
    evidence: dict[str, object] = {"schema": "guandan-marvel-phase4-local-validation-v1"}

    cfg, game = _config(output / "run")
    schedule_hash = cfg.training_data_manifest_hash
    evidence["manifest_hash"] = schedule_hash
    evidence["reward_semantics"] = cfg.reward_semantics
    evidence["reward_scale"] = cfg.reward_scale
    evidence["curriculum"] = dict(CURRICULUM)

    trainer = IPPOTrainer(cfg, game=game, run_dir=output / "run")
    warm_payload = torch.load(Path(args.warm_start), map_location="cpu", weights_only=False)
    warm = trainer.load_warm_start_policy(warm_payload)
    evidence["warm_start"] = warm

    metrics: list[dict[str, object]] = []
    for _ in range(2):
        metrics.append(trainer.train_iteration())
    checkpoint = trainer.save_checkpoint(output / "run" / "phase4-iter000002.pt")
    payload = trainer.checkpoint_payload()
    evidence["first_run"] = {
        "iterations": [int(m["statistics/iterations"]) for m in metrics],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    recovery = _build_recovery(
        run_dir=output / "run", checkpoint=checkpoint, payload=payload,
        destination=output / "phase4-recovery.zip",
    )
    evidence["recovery"] = {**recovery, "zip_sha256": _sha256(output / "phase4-recovery.zip")}

    # ---- continuity assertions over the first two iterations -----------------
    for metric in metrics:
        _require(metric["reward/semantics"] == "guandan_rank_points", "reward semantics recorded", evidence)
        _require(metric["reward/training_scale"] == 1.0 / 3.0, "training scale is 1/3", evidence)
        raw = metric["guandan/raw_team0_reward_mean"]
        train_return = metric["statistics/return_team0_mean"]
        _require(raw is not None and math.isfinite(float(raw)), "raw team reward recorded", evidence)
        rank_points = metric["guandan/mean_rank_point_reward"]
        _require(rank_points is not None and round(float(rank_points)) in (1, 2, 3),
                 "mean rank-point reward is one of 1/2/3", evidence)
        _require(abs(float(train_return) - float(raw) / 3.0) < 1e-6, "train return equals raw/3", evidence)
        _require(metric["macro/profile"] == MACRO_DECISION_TRAINING_SEMANTICS_PROFILE, "macro profile continuous", evidence)
        _require(math.isfinite(float(metric["macro/pg_loss"])), "macro pg_loss finite", evidence)
        _require(math.isfinite(float(metric["macro/value_loss"])), "macro value_loss finite", evidence)
    provenance = payload["training_data_provenance"]
    _require(provenance["curriculum"]["active"] is True, "curriculum recorded in provenance", evidence)
    _require(provenance["curriculum"]["hash"], "curriculum hash recorded", evidence)
    _require(provenance["declared_context_coverage"]["context_count"] == 52, "52-context schedule declared", evidence)
    _require(provenance["observed_context_coverage"]["deal_count"] > 0, "observed deal coverage present", evidence)
    _require(
        all(0 < int(row["level_rank"]) <= 15 and 0 <= int(row["leader_seat"]) <= 3
            for row in payload["sampler_state"]["training_deal_records"]),
        "every deal row records its actual level/leader",
        evidence,
    )
    _require(
        all("context_sha256" in row for row in payload["sampler_state"]["training_deal_records"]),
        "every deal row records its context hash",
        evidence,
    )
    _require(
        payload["semantic_contract"]["semantic_config"]["reward"]["semantics"] == "guandan_rank_points",
        "checkpoint semantic contract records reward semantics",
        evidence,
    )
    _require(
        payload["semantic_contract"]["semantic_config"]["reward"]["scale"] == 1.0 / 3.0,
        "checkpoint semantic contract records reward scale",
        evidence,
    )
    _require(payload["grad_scaler_state"] is not None, "GradScaler state persisted", evidence)
    _require(payload["rng_state"] is not None, "RNG state persisted", evidence)
    _require(payload["sampler_state"]["schema"].endswith("v2"), "sampler state schema persisted", evidence)
    completed_first = int(payload["counters"]["completed_games_total"])
    expected_first = sum(int(m["statistics/trajectories"]) for m in metrics)
    _require(completed_first == expected_first, "completed-game budget is exact", evidence)

    # ---- resume --------------------------------------------------------------
    resumed_cfg, resumed_game = _config(
        output / "resumed", logging_dir=output / "run" / "logging",
    )
    resumed = IPPOTrainer(resumed_cfg, game=resumed_game, run_dir=output / "resumed")
    target = torch.load(checkpoint, map_location="cpu", weights_only=False)
    decision = resumed.load_checkpoint_payload(target)
    evidence["resume_decision"] = {"kind": decision.kind, "reasons": list(decision.reasons)}
    _require(decision.kind == "exact_resume", "source identity strictly matches (exact_resume)", evidence)
    _require(
        str(resumed.source_hash) == str(payload["source_hash"]),
        "resumed source fingerprint equals checkpoint source fingerprint",
        evidence,
    )
    _require(resumed.iteration == 2, "resume restores iteration 2", evidence)
    _require(
        resumed.sampler.training_data_provenance()["hash"] == payload["training_data_provenance"]["hash"],
        "sampler training provenance restored exactly",
        evidence,
    )

    resumed_metrics: list[dict[str, object]] = []
    for _ in range(2):
        resumed_metrics.append(resumed.train_iteration())
    evidence["resumed_run"] = {
        "iterations": [int(m["statistics/iterations"]) for m in resumed_metrics],
        "completed_games_total": int(resumed.completed_games_total),
    }
    _require([int(m["statistics/iterations"]) for m in resumed_metrics] == [3, 4], "iteration numbering continues", evidence)
    _require(
        int(resumed.completed_games_total) == completed_first + sum(int(m["statistics/trajectories"]) for m in resumed_metrics),
        "completed-game budget continues exactly",
        evidence,
    )
    final_payload = resumed.checkpoint_payload()
    final_provenance = final_payload["training_data_provenance"]
    _require(
        final_provenance["curriculum"]["iteration"] == 3,
        "curriculum iteration tracks the resumed trainer",
        evidence,
    )
    _require(
        final_provenance["observed_context_coverage"]["deal_count"] > provenance["observed_context_coverage"]["deal_count"],
        "context provenance accumulates across resume",
        evidence,
    )

    # ---- incompatible recovery is rejected ----------------------------------
    tampered = dict(target)
    tampered["source_fingerprint"] = {**dict(target["source_fingerprint"]), "hash": "0" * 64}
    incompatible_cfg, incompatible_game = _config(output / "incompatible")
    incompatible = IPPOTrainer(incompatible_cfg, game=incompatible_game, run_dir=output / "incompatible")
    rejected = False
    try:
        incompatible.load_checkpoint_payload(tampered)
    except Exception as error:  # noqa: BLE001 - the rejection type is the evidence
        rejected = True
        evidence["incompatible_recovery_error"] = f"{type(error).__name__}: {error}"
    _require(rejected, "incompatible source recovery is rejected", evidence)
    _require(incompatible.failed, "trainer is poisoned after a rejected recovery", evidence)

    evidence["wall_seconds"] = time.perf_counter() - started
    evidence["status"] = "pass"
    evidence["sha256"] = {
        "checkpoint": _sha256(checkpoint),
        "recovery_zip": _sha256(output / "phase4-recovery.zip"),
    }
    (output / "validation_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({"status": evidence["status"], "wall_seconds": evidence["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
