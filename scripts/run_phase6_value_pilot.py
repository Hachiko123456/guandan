# SPDX-License-Identifier: MIT
"""Bounded phase-6 GPU value-qualification pilot (≤16 deals, ≤600 s).

    D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase6_value_pilot.py --device cuda

The pre-registered gate requires >=64 independent deals, >=512 samples, two
distinct checkpoints and >=64 counterfactual ranking points.  A local pilot
cannot satisfy those thresholds; the expected verdict is INCONCLUSIVE and the
search is explicitly NOT RUN.  The script never edits thresholds or increases
the budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from guandan_marvel.config import (  # noqa: E402
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE, ModelConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.evaluation.checkpoints import (  # noqa: E402
    freeze_merged_policy, load_v2_policy,
)
from guandan_marvel.evaluation.common import file_sha256  # noqa: E402
from guandan_marvel.evaluation.manifests import (  # noqa: E402
    build_deals_manifest, deal_game_config, deals_for,
)
from guandan_marvel.phase6.value_gate import (  # noqa: E402
    _deal_hands, assert_qualification_isolation, build_qualification_deals,
    canonical_physical_deal, evaluate_value_gate, load_value_gate_registration,
    qualification_producer_source_hash, run_value_qualification,
    write_value_gate_registration,
)
from guandan_marvel.training.local import IPPOTrainer  # noqa: E402
from guandan_marvel.training.smoke import CppSmokeGame  # noqa: E402
from guandan_marvel import cpp_game  # noqa: E402


def _cfg(game) -> TrainConfig:
    cfg = TrainConfig(
        game=game.spec, device="cuda", algo_name="ippo", seed=31337,
        run_name="phase6-value-pilot",
        rollout_batch_size=1, collect_batch_size=1, mini_batches=1,
        max_sampler_bytes_per_rank=1024 * 2**20, amp_enabled=False,
        checkpoint_replay_buffer=False,
        model=ModelConfig(
            architecture="v1.4-macro-fuse", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
        ),
    )
    cfg.ppo.agent_epoch = 1
    cfg.optimizer.use_muon = False
    return cfg


def _previous_results() -> list[dict[str, object] | None]:
    # TEAM_OF = (0, 1, 0, 1); ranking[seat] is the finishing rank of that seat.
    return [
        None,
        {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": False},
        {"ranking": [2, 1, 4, 3], "winner_team": 1, "outcome_class": "HEAD_THIRD", "tribute_cancelled": False},
        {"ranking": [1, 2, 4, 3], "winner_team": 0, "outcome_class": "HEAD_LAST", "tribute_cancelled": False},
        {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": True},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--deals", type=int, default=16)
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    parser.add_argument("--max-ranking-points", type=int, default=2)
    parser.add_argument(
        "--out-dir", default="runs/phase6/value_gate",
    )
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("value qualification requires an explicit available CUDA device")
    if args.deals < 1 or args.deals > 32:
        raise SystemExit("pilot deals must be within 1..32")
    if args.wall_seconds <= 0 or args.wall_seconds > 1200:
        raise SystemExit("pilot wall budget must be >0 and <=1200 seconds")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    registration_path = output / "registration.json"
    producer_source_hash = qualification_producer_source_hash()
    game = CppSmokeGame(1024)
    cfg = _cfg(game)
    started = time.perf_counter()
    trainer = IPPOTrainer(cfg, game=game, run_dir=output / "pilot-trainer")
    trainer.train_iteration()
    checkpoint = freeze_merged_policy(trainer, output / "pilot-eval.pt")
    artifact = load_v2_policy(checkpoint, device="cpu", expected_algorithm="ippo")
    if registration_path.exists():
        registration = load_value_gate_registration(registration_path)
        if registration.get("source_hash") not in ("", producer_source_hash):
            raise SystemExit(
                "existing registration has a different qualification producer source; "
                "use a new evidence dir"
            )
        if registration.get("config_hash") not in ("", trainer.resolved_config_hash):
            raise SystemExit(
                "existing registration has a different real config hash; use a new evidence dir"
            )
    else:
        registration = write_value_gate_registration(
            registration_path,
            source_hash=producer_source_hash,
            config_hash=trainer.resolved_config_hash,
        )
    # Real protected data: materialize the actual training/selection/final deal
    # records and their canonical hashes; never a placeholder seed list.
    protected_manifest = build_deals_manifest(
        counts={"training": 512, "selection": 8, "final": 8},
    )
    protected_rows = [
        dict(row)
        for domain in ("training", "selection", "final")
        for row in deals_for(protected_manifest, domain)
    ]
    protected_hashes = []
    for row in protected_rows:
        protected_env = cpp_game.EnvBatch(
            1, configs=[deal_game_config(row)], auto_reset=False, max_horizon=8,
        )
        protected_env.reset(int(row["seed"]))
        protected_hashes.append(canonical_physical_deal(_deal_hands(protected_env)))
    deals = build_qualification_deals(
        args.deals, base_seed=2_000_000_000, previous_results=_previous_results(),
    )
    isolation = assert_qualification_isolation(
        deals,
        protected_rows=protected_rows,
        protected_canonical_hashes=protected_hashes,
    )
    result = run_value_qualification(
        checkpoint_path=checkpoint,
        registration_path=registration_path,
        deals=deals,
        device="cuda",
        max_deals=args.deals,
        wall_seconds=args.wall_seconds,
        max_ranking_points=args.max_ranking_points,
        protected_canonical_hashes=protected_hashes,
    )
    gate = evaluate_value_gate(
        result,
        registration=registration,
        checkpoint_identities=[artifact.identity],
    )
    samples_path = output / "samples.json"
    samples_path.write_text(
        json.dumps({
            "samples": [sample.payload() for sample in result.samples],
            "ranking_points": [point.payload() for point in result.ranking_points],
            "deal_records": result.deal_records,
            "metadata": result.metadata,
        }, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    gate_path = output / "qualification.json"
    evidence = {
        "schema": "guandan-marvel-phase6-value-pilot-evidence-v1",
        "python_executable": sys.executable,
        "gpu_name": torch.cuda.get_device_name(0),
        "checkpoint_sha256": file_sha256(checkpoint),
        "registration_path": str(registration_path.resolve()),
        "samples_path": str(samples_path.resolve()),
        "isolation": isolation,
        "source_hash": trainer.source_hash,
        "config_hash": trainer.resolved_config_hash,
        "deals_manifest_hash": protected_manifest["hash"],
        "protected_deal_count": len(protected_rows),
        "protected_canonical_hash_count": len(protected_hashes),
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "budget": {
            "deals": args.deals,
            "wall_seconds_cap": args.wall_seconds,
            "max_ranking_points": args.max_ranking_points,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    gate_path.write_text(
        json.dumps(gate, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "VALUE_GATE": gate["verdict"],
        "SEARCH_STATUS": gate["search_status"],
        "independent_deals": gate["sufficiency"]["independent_deals"],
        "samples": gate["sufficiency"]["sample_count"],
        "ranking_points": gate["sufficiency"]["ranking_points"],
        "reasons": gate["reasons"],
        "output": str(gate_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
