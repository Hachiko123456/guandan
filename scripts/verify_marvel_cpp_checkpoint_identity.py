"""Bounded verification of C++ backend checkpoint provenance and resume policy."""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from guandan_marvel import cpp_game, python_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.training.checkpoints import (
    CheckpointCompatibilityError, TrainingCheckpointStore,
    canonical_json_hash,
)
from guandan_marvel.training.local import IPPOTrainer


def config(game, root: Path, *, logging_suffix: str) -> TrainConfig:
    cfg = TrainConfig(
        game=game.GAME_SPEC, device="cpu", algo_name="ippo",
        logging_dir=str(root / logging_suffix), run_name="m05-cpp-checkpoint",
        seed=93_001, rollout_batch_size=1, collect_batch_size=1,
        replay_buffer_scale=1, mini_batches=1, total_iterations=2,
        max_sampler_bytes_per_rank=512 * 2**20,
        checkpoint_replay_buffer=False, amp_enabled=False,
        model=ModelConfig(
            architecture="v1.3", history_dim=8, feature_dim=8,
            decoder_layers=1, decoder_heads=1,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.ppo.agent_epoch = 1
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg


def poisoned(trainer: IPPOTrainer) -> bool:
    return bool(trainer.failed and trainer.sampler.failed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    cpp_cfg = config(cpp_game, root, logging_suffix="cpp-source")
    source = IPPOTrainer(cpp_cfg, game=cpp_game, run_dir=root / "cpp-source")
    first_metrics = source.train_iteration()
    checkpoint = source.save_checkpoint(root / "cpp-backend-iter000001-v4.pt")
    payload = dict(TrainingCheckpointStore.load(checkpoint))
    identity = payload["backend_identity"]
    implementation = identity["implementation"]
    source_identity = {
        "backend": payload["backend"],
        "backend_identity_hash": identity["hash"],
        "source_hash": payload["source_hash"],
        "extension_path": implementation["extension_path"],
        "extension_sha256": implementation["extension_sha256"],
        "embedded_source_sha256": implementation["embedded_source_sha256"],
        "source_matches_loaded_binary": implementation["source_matches_loaded_binary"],
    }
    del source
    gc.collect()

    exact = IPPOTrainer(cpp_cfg, game=cpp_game, run_dir=root / "cpp-exact")
    exact_decision = exact.load_checkpoint(checkpoint)
    second_metrics = exact.train_iteration()
    del exact
    gc.collect()

    compatible_cfg = config(cpp_game, root, logging_suffix="different-log-dir")
    compatible_cfg.run_name = "different-run-name"
    compatible = IPPOTrainer(compatible_cfg, game=cpp_game, run_dir=root / "cpp-compatible")
    compatible_decision = compatible.load_checkpoint(checkpoint)
    del compatible
    gc.collect()

    python_cfg = config(python_game, root, logging_suffix="python-cross-backend")
    cross = IPPOTrainer(python_cfg, game=python_game, run_dir=root / "python-cross-backend")
    cross_error = None
    try:
        cross.load_checkpoint(checkpoint)
    except CheckpointCompatibilityError as error:
        cross_error = str(error)
    cross_poisoned = poisoned(cross)
    del cross
    gc.collect()

    tampered_payload = copy.deepcopy(payload)
    tampered_identity = tampered_payload["backend_identity"]
    tampered_identity["implementation"]["extension_sha256"] = "0" * 64
    tampered_identity["hash"] = canonical_json_hash({
        key: value for key, value in tampered_identity.items() if key != "hash"
    })
    tampered = root / "cpp-backend-tampered-build.pt"
    torch.save(tampered_payload, tampered)
    build_target = IPPOTrainer(cpp_cfg, game=cpp_game, run_dir=root / "cpp-build-mismatch")
    build_error = None
    try:
        build_target.load_checkpoint(tampered)
    except CheckpointCompatibilityError as error:
        build_error = str(error)
    build_poisoned = poisoned(build_target)

    valid = bool(
        payload["backend"] == "cpp"
        and implementation["source_matches_loaded_binary"] is True
        and exact_decision.kind == "exact_resume"
        and compatible_decision.kind == "compatible_resume"
        and cross_error and cross_poisoned
        and build_error and build_poisoned
        and int(second_metrics["statistics/iterations"]) == 2
    )
    result = {
        "schema": "guandan-marvel-m05-cpp-checkpoint-verification-v1",
        "checkpoint": str(checkpoint),
        "source_identity": source_identity,
        "first_iteration": {
            "iteration": int(first_metrics["statistics/iterations"]),
            "backend": first_metrics["runtime/backend"],
        },
        "same_backend_same_build": {
            "decision": exact_decision.kind,
            "continued_iteration": int(second_metrics["statistics/iterations"]),
        },
        "logging_only_change": {"decision": compatible_decision.kind},
        "cross_backend": {"rejected": bool(cross_error), "error": cross_error, "poisoned": cross_poisoned},
        "build_identity_change": {"rejected": bool(build_error), "error": build_error, "poisoned": build_poisoned},
        "valid": valid,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": valid, "exact": exact_decision.kind,
        "compatible": compatible_decision.kind,
        "cross_backend_rejected": bool(cross_error),
        "build_mismatch_rejected": bool(build_error),
        "output": str(output),
    }, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
