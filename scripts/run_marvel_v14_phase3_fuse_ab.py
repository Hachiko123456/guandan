#!/usr/bin/env python
"""MARVEL v1.4 phase-3 actor-fuse A/B harness (Control vs Treatment).

Both arms start from the *same* v1.3 evaluation-only warm-start policy, with the
same seed, the same data-generator manifest (identical deals / games / rank seed
stride) and the same macro-PPO + macro-GAE training semantics.  The only
difference between the two arms is ``model.architecture``:

* **Control**  -- ``v1.3``                     (fuse off, legacy multiply path)
* **Treatment** -- ``v1.4-macro-fuse``          (identical, plus the residual fuse)

The harness deliberately has *no* resume-from-control path: a Treatment arm can
only be obtained by warm-starting the fused architecture from the shared v1.3
policy, which structurally prevents "train a Control, then flip the fuse on and
call it an A/B".

By default the run is executed with AMP **off**, so the measured difference is
the architecture effect alone; ``--amp`` is available for a follow-up run.

The formal Control/Treatment training run is NOT executed by this phase.  Use
``--smoke`` to validate the harness end-to-end on a synthetic v1.3 policy and a
reduced C++ smoke game (cheap, no M09 checkpoint needed).

Formal run (once authorised)::

    python scripts/run_marvel_v14_phase3_fuse_ab.py \\
        --warm-start /path/to/m09-v1.3-evaluation-checkpoint.pt \\
        --output-dir runs/marvel_v14_phase3/ab \\
        --game cpp --dims checkpoint --iterations 20 --seed 1234

``--game cpp`` selects the production C++ game (full horizon) and
``--dims checkpoint`` reuses the M09 policy's own architecture/batch dimensions,
so the two arms differ only in ``model.architecture``.  Add ``--amp`` for the
follow-up AMP run; the architecture-isolation run must stay AMP-off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from guandan_marvel.config import (  # noqa: E402
    LEGACY_POLICY_REGULARIZER_MODE, MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_FUSE_MODEL_ARCHITECTURE, ModelConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.evaluation.checkpoints import freeze_merged_policy  # noqa: E402
from guandan_marvel.training.checkpoints import TrainingCheckpointStore, canonical_json_hash  # noqa: E402
from guandan_marvel.training.local import IPPOTrainer  # noqa: E402
from guandan_marvel.training.smoke import CppSmokeGame  # noqa: E402

AB_SCHEMA = "guandan-marvel-v14-phase3-fuse-ab-v1"
CONTROL_ARCHITECTURE = "v1.3"
TREATMENT_ARCHITECTURE = MACRO_FUSE_MODEL_ARCHITECTURE
# Non-semantic resolved-config keys that legitimately differ between two arms.
NON_SEMANTIC_ARM_KEYS = ("run_name", "logging_dir")
METRIC_KEYS = (
    "macro/pg_loss", "macro/value_loss_last", "macro/legacy_token_kl", "macro/macro_kl",
    "macro/clip_frac", "macro/raw_entropy", "macro/normalized_entropy", "agent/grad_norm",
    "macro/macro_count_global", "macro/macro_length_mean",
    "runtime/amp_scaler_scale", "runtime/amp_skipped_steps",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_config(
    architecture: str,
    *,
    seed: int,
    total_iterations: int,
    amp_enabled: bool,
    game,
    dims: Mapping[str, int],
    run_name: str,
) -> TrainConfig:
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name="ippo",
        run_name=run_name,
        logging_dir=f"runs/marvel_v14_phase3/ab/{run_name}",
        seed=seed,
        rollout_batch_size=dims["rollout_batch_size"],
        collect_batch_size=dims["collect_batch_size"],
        replay_buffer_scale=1,
        mini_batches=dims["mini_batches"],
        total_iterations=total_iterations,
        max_sampler_bytes_per_rank=dims["max_sampler_bytes_per_rank"],
        amp_enabled=amp_enabled,
        checkpoint_replay_buffer=True,
        model=ModelConfig(
            architecture=architecture,
            history_dim=dims["history_dim"],
            feature_dim=dims["feature_dim"],
            decoder_layers=dims["decoder_layers"],
            decoder_heads=dims["decoder_heads"],
            state_layers=dims["state_layers"],
            action_layers=dims["action_layers"],
            fuse_layers=dims["fuse_layers"],
        ),
        ppo=PPOConfig(
            agent_epoch=dims["agent_epoch"],
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=LEGACY_POLICY_REGULARIZER_MODE,
        ),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg


def _arm_difference(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> list[str]:
    """Resolved-config paths that differ between the two arms (order-independent)."""
    differences: list[str] = []

    def walk(left: Any, right: Any, prefix: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key))
            return
        if left != right:
            differences.append(prefix)

    for key in sorted(set(control) | set(treatment)):
        if key in NON_SEMANTIC_ARM_KEYS:
            continue
        walk(control.get(key), treatment.get(key), str(key))
    return differences


def _metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS}


def run_arm(
    architecture: str,
    *,
    warm_start_payload: Mapping[str, Any],
    game,
    run_dir: Path,
    iterations: int,
    seed: int,
    amp_enabled: bool,
    dims: Mapping[str, int],
    run_name: str,
) -> dict[str, Any]:
    cfg = arm_config(
        architecture, seed=seed, total_iterations=iterations, amp_enabled=amp_enabled,
        game=game, dims=dims, run_name=run_name,
    )
    trainer = IPPOTrainer(cfg, game=game, run_dir=run_dir)
    try:
        if architecture == TREATMENT_ARCHITECTURE:
            assert trainer.agent_resume.actor_fuse is not None
            assert int(torch.count_nonzero(trainer.agent_resume.actor_fuse.output_proj.weight)) == 0
        else:
            assert trainer.agent_resume.actor_fuse is None
        migration = trainer.load_warm_start_policy(warm_start_payload)
        series: list[dict[str, Any]] = []
        for _ in range(int(iterations)):
            series.append(_metric_snapshot(trainer.train_iteration()))
        checkpoint = trainer.save_checkpoint(run_dir / f"{run_name}-iter{trainer.iteration:06d}.pt")
        return {
            "architecture": architecture,
            "run_name": run_name,
            "resolved_config": serialize_cfg(cfg),
            "resolved_config_hash": canonical_json_hash(serialize_cfg(cfg)),
            "data_manifest_hash": cfg.training_data_manifest_hash,
            "seed": int(seed),
            "iterations": int(trainer.iteration),
            "durable_updates": int(trainer.durable_updates),
            "amp_enabled": bool(cfg.amp_enabled),
            "amp_autocast_dtype": str(trainer.amp_dtype),
            "amp_scaler_enabled": bool(trainer.amp_scaler_enabled),
            "fuse_layers": int(cfg.model.fuse_layers),
            "parameter_count": int(sum(p.numel() for p in trainer.agent_resume.parameters())),
            "warm_start_migration": migration["migration"],
            "metrics": series,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
        }
    finally:
        trainer.close()


def serialize_cfg(cfg: TrainConfig) -> dict[str, Any]:
    return cfg.to_dict()


def run_fuse_ab(
    *,
    warm_start_payload: Mapping[str, Any],
    game,
    output_dir: Path,
    iterations: int,
    seed: int,
    amp_enabled: bool = False,
    dims: Mapping[str, int],
    run_tag: str = "ab",
    extra_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    warm_architecture = str(warm_start_payload.get("architecture"))
    if warm_architecture != CONTROL_ARCHITECTURE:
        raise ValueError(
            "the A/B harness requires the shared M09 v1.3 evaluation-only warm start; "
            f"got architecture {warm_architecture!r}"
        )
    if warm_start_payload.get("schema") != "guandan-marvel-evaluation-checkpoint-v1":
        raise ValueError("the A/B warm start must be an evaluation-only checkpoint")

    control = run_arm(
        CONTROL_ARCHITECTURE, warm_start_payload=warm_start_payload, game=game,
        run_dir=output_dir / f"{run_tag}-control", iterations=iterations, seed=seed,
        amp_enabled=amp_enabled, dims=dims, run_name=f"{run_tag}-control",
    )
    treatment = run_arm(
        TREATMENT_ARCHITECTURE, warm_start_payload=warm_start_payload, game=game,
        run_dir=output_dir / f"{run_tag}-treatment", iterations=iterations, seed=seed,
        amp_enabled=amp_enabled, dims=dims, run_name=f"{run_tag}-treatment",
    )

    differences = _arm_difference(control["resolved_config"], treatment["resolved_config"])
    if differences != ["model.architecture"]:
        raise ValueError(
            "the two A/B arms may differ only in model.architecture; found: "
            + ", ".join(differences)
        )
    if control["data_manifest_hash"] != treatment["data_manifest_hash"]:
        raise ValueError("the two A/B arms must share one training-data manifest (same deals)")
    if control["seed"] != treatment["seed"]:
        raise ValueError("the two A/B arms must share one seed")
    if treatment["parameter_count"] <= control["parameter_count"]:
        raise ValueError("the treatment arm must carry the extra fuse parameters")

    report = {
        "schema": AB_SCHEMA,
        "status": "completed",
        "executed": True,
        "amp_enabled": bool(amp_enabled),
        "seed": int(seed),
        "iterations": int(iterations),
        "warm_start": {
            "schema": warm_start_payload.get("schema"),
            "architecture": warm_architecture,
            "iteration": int(warm_start_payload.get("iteration", -1)),
            "resolved_config_hash": warm_start_payload.get("resolved_config_hash"),
            "source_hash": warm_start_payload.get("source_hash"),
        },
        "arm_configuration_differences": differences,
        "control": control,
        "treatment": treatment,
        "parameter_delta": treatment["parameter_count"] - control["parameter_count"],
        "isolation": {
            "same_warm_start": True,
            "same_seed": True,
            "same_data_manifest": control["data_manifest_hash"] == treatment["data_manifest_hash"],
            "amp_off": not amp_enabled,
            "resume_from_control_supported": False,
        },
    }
    report.update(dict(extra_report or {}))
    report["hash"] = canonical_json_hash({k: v for k, v in report.items() if k != "hash"})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ab_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8",
    )
    return report


DEFAULT_DIMS = {
    "rollout_batch_size": 2,
    "collect_batch_size": 2,
    "mini_batches": 1,
    "max_sampler_bytes_per_rank": 256 * 2**20,
    "history_dim": 16,
    "feature_dim": 24,
    "decoder_layers": 1,
    "decoder_heads": 4,
    "state_layers": 1,
    "action_layers": 1,
    "fuse_layers": 1,
    "agent_epoch": 1,
}


def _synthetic_warm_start(path: Path, game) -> Path:
    """A cheap, synthetic v1.3 evaluation-only policy for harness smoke runs."""
    cfg = arm_config(
        CONTROL_ARCHITECTURE, seed=4242, total_iterations=1, amp_enabled=False,
        game=game, dims=DEFAULT_DIMS, run_name="ab-synthetic-warm-start",
    )
    trainer = IPPOTrainer(cfg, game=game, run_dir=path.parent / "synthetic-warm-start")
    try:
        return freeze_merged_policy(trainer, path)
    finally:
        trainer.close()


def dims_from_checkpoint(payload: Mapping[str, Any]) -> dict[str, int]:
    """Reuse the warm-start checkpoint's own architecture/batch dimensions.

    For the formal A/B this is what guarantees the arms match the M09 v1.3 policy
    exactly, with ``model.architecture`` as the single difference.
    """
    cfg = payload["resolved_config"]
    model = cfg["model"]
    return {
        "rollout_batch_size": int(cfg["rollout_batch_size"]),
        "collect_batch_size": int(cfg["collect_batch_size"]),
        "mini_batches": int(cfg["mini_batches"]),
        "max_sampler_bytes_per_rank": int(cfg["max_sampler_bytes_per_rank"]),
        "history_dim": int(model["history_dim"]),
        "feature_dim": int(model["feature_dim"]),
        "decoder_layers": int(model["decoder_layers"]),
        "decoder_heads": int(model["decoder_heads"]),
        "state_layers": int(model["state_layers"]),
        "action_layers": int(model["action_layers"]),
        "fuse_layers": int(model["fuse_layers"]),
        "agent_epoch": int(cfg["ppo"]["agent_epoch"]),
    }


def resolve_game(name: str, horizon: int):
    if name == "cpp":
        from guandan_marvel import cpp_game
        return cpp_game
    return CppSmokeGame(max_game_horizon=int(horizon))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start", help="M09 v1.3 evaluation-only checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--horizon", type=int, default=512, help="smoke-game horizon")
    parser.add_argument("--amp", action="store_true", help="run with AMP (default: off)")
    parser.add_argument(
        "--game", choices=("smoke", "cpp"), default="smoke",
        help="smoke = reduced-horizon CppSmokeGame (harness validation); "
             "cpp = the production C++ game (formal A/B run)",
    )
    parser.add_argument(
        "--dims", choices=("smoke", "checkpoint"), default="smoke",
        help="smoke = tiny dims (harness validation); "
             "checkpoint = reuse the warm-start checkpoint's own dimensions (formal run)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="validate the harness cheaply with a synthetic warm start and reduced iterations",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    game = resolve_game(args.game, args.horizon)
    iterations = int(args.iterations)
    if args.smoke:
        iterations = min(iterations, 2)
        warm_start_path = output_dir / "synthetic-v13-warm-start.pt"
        output_dir.mkdir(parents=True, exist_ok=True)
        _synthetic_warm_start(warm_start_path, game)
        dims = DEFAULT_DIMS
    else:
        if not args.warm_start:
            raise SystemExit(
                "the formal A/B run requires --warm-start <M09 v1.3 evaluation checkpoint>; "
                "use --smoke for a cheap harness validation"
            )
        warm_start_path = Path(args.warm_start)
        dims = None

    payload = TrainingCheckpointStore.load(warm_start_path, device="cpu")
    if dims is None:
        dims = DEFAULT_DIMS if args.dims == "smoke" else dims_from_checkpoint(payload)
    game_identity = {
        "module": str(getattr(game, "__name__", type(game).__name__)),
        "identifier": str(getattr(game, "identifier", getattr(game, "GAME_IDENTIFIER", "unknown"))),
        "max_game_horizon": int(getattr(game, "max_game_horizon", args.horizon)),
        "backend": str(getattr(game, "BACKEND", "unknown")),
    }
    report = run_fuse_ab(
        warm_start_payload=payload, game=game, output_dir=output_dir,
        iterations=iterations, seed=int(args.seed), amp_enabled=bool(args.amp),
        dims=dims, run_tag="smoke" if args.smoke else "ab",
        extra_report={"game_identity": game_identity, "dims": dict(dims),
                      "dims_profile": "smoke" if args.smoke else args.dims},
    )
    print(json.dumps({
        "schema": report["schema"],
        "smoke": bool(args.smoke),
        "iterations": report["iterations"],
        "amp_enabled": report["amp_enabled"],
        "control_parameters": report["control"]["parameter_count"],
        "treatment_parameters": report["treatment"]["parameter_count"],
        "parameter_delta": report["parameter_delta"],
        "arm_configuration_differences": report["arm_configuration_differences"],
        "control_final_pg_loss": report["control"]["metrics"][-1]["macro/pg_loss"],
        "treatment_final_pg_loss": report["treatment"]["metrics"][-1]["macro/pg_loss"],
        "report": str(output_dir / "ab_report.json"),
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
