"""Generate M03 repair evidence for interruption and semantic resume contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.training.checkpoints import CheckpointCompatibilityError
from guandan_marvel.training.local import IPPOTrainer, RUNTIME_SOURCE_COMPONENTS
from guandan_marvel.training.smoke import PythonReferenceSmokeGame


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(root: Path, *, gamma: float = 1.0, heads: int = 4, logging_suffix: str = "configured"):
    game = PythonReferenceSmokeGame(512)
    cfg = TrainConfig(
        game=game.spec, device="cpu", algo_name="ippo",
        logging_dir=str(root / logging_suffix), run_name="m03-repair-ippo",
        seed=4101, rollout_batch_size=1, collect_batch_size=1,
        replay_buffer_scale=1, mini_batches=1, total_iterations=3,
        max_sampler_bytes_per_rank=64 * 2**20, amp_enabled=False,
        checkpoint_replay_buffer=True, discount_gamma=gamma,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=heads,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.ppo.agent_epoch = 2
    cfg.vrpo.aux_epoch = 2
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


def rejected(callable_) -> dict[str, object]:
    try:
        callable_()
    except BaseException as error:
        return {"rejected": True, "error_type": type(error).__name__, "error": str(error)}
    return {"rejected": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cfg, game = config(root)
    source = IPPOTrainer(cfg, game=game, run_dir=root / "source")
    source.train_iteration()
    checkpoint = source.save_checkpoint(root / "guandan-marvel-v2-ippo-iter000001-v3.pt")
    checkpoint_hash = sha256(checkpoint)

    interrupted = IPPOTrainer(cfg, game=game, run_dir=root / "interrupted")
    interrupted.load_checkpoint(checkpoint)
    before_weights = {key: value.detach().clone() for key, value in interrupted.agent_resume.state_dict().items()}
    original_step = interrupted._optimizer_step
    calls = 0

    def stop_after_step(*a, **kw):
        nonlocal calls
        value = original_step(*a, **kw)
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("verification interrupt after optimizer step")
        return value

    interrupted._optimizer_step = stop_after_step
    interrupt_error = None
    try:
        interrupted.train_iteration()
    except KeyboardInterrupt as error:
        interrupt_error = {"type": type(error).__name__, "message": str(error)}
    changed_weights = sum(
        not torch.equal(before_weights[key], interrupted.agent_resume.state_dict()[key])
        for key in before_weights
    )
    interrupt_result = {
        "exception": interrupt_error,
        "changed_weight_tensors": changed_weights,
        "iteration": interrupted.iteration,
        "durable_updates": interrupted.durable_updates,
        "learner_updates_observed": interrupted.learner_updates,
        "durable_learner_updates": interrupted.durable_learner_updates,
        "iteration_status": interrupted.iteration_status,
        "trainer_failed": interrupted.failed,
        "sampler_failed": interrupted.sampler.failed,
        "failure": interrupted.failure,
        "checkpoint_payload": rejected(interrupted.checkpoint_payload),
        "save_checkpoint": rejected(lambda: interrupted.save_checkpoint(root / "forbidden.pt")),
        "retry_iteration": rejected(interrupted.train_iteration),
        "retry_sampler": rejected(lambda: interrupted.sampler.collect(agent_model=interrupted.agent_model)),
        "previous_checkpoint_unchanged": sha256(checkpoint) == checkpoint_hash,
    }

    recovered = IPPOTrainer(cfg, game=game, run_dir=root / "recovered")
    recovery_decision = recovered.load_checkpoint(checkpoint)
    recovered.train_iteration()
    recovery = {
        "decision": recovery_decision.kind,
        "iteration": recovered.iteration,
        "durable_updates": recovered.durable_updates,
        "completed_games": recovered.completed_games_total,
    }

    semantic = {}
    for name, changed_cfg in (
        ("discount_gamma", config(root, gamma=0.5)[0]),
        ("decoder_heads", config(root, heads=2)[0]),
    ):
        trainer = IPPOTrainer(changed_cfg, game=game, run_dir=root / f"semantic-{name}")
        try:
            trainer.load_checkpoint(checkpoint)
            semantic[name] = {"loaded": True}
        except CheckpointCompatibilityError as error:
            semantic[name] = {
                "loaded": False,
                "error": str(error),
                "trainer_failed": trainer.failed,
                "sampler_failed": trainer.sampler.failed,
                "save_rejected": rejected(lambda: trainer.save_checkpoint(root / f"{name}-forbidden.pt")),
            }

    compatible_cfg, _ = config(root, logging_suffix="other-logs")
    compatible_cfg.run_name = "other-run-name"
    compatible = IPPOTrainer(compatible_cfg, game=game, run_dir=root / "compatible")
    compatible_decision = compatible.load_checkpoint(checkpoint)

    report = {
        "schema": "guandan-marvel-m03-repair-verification-v1",
        "date": "2026-10-09",
        "original_implementation_commit": "4968840679ea031ec7b69132cebd862efa302c17",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash, "bytes": checkpoint.stat().st_size},
        "interrupt": interrupt_result,
        "recovery_from_last_complete_checkpoint": recovery,
        "semantic_changes": semantic,
        "compatible_logging_change": {
            "decision": compatible_decision.kind,
            "reasons": list(compatible_decision.reasons),
        },
        "source_fingerprint": {
            "hash": source.source_fingerprint["hash"],
            "components": {
                name: {
                    "hash": payload["hash"],
                    "files": [item["path"] for item in payload["files"]],
                }
                for name, payload in source.source_fingerprint["components"].items()
            },
            "declared_inventory": {
                name: list(paths) for name, paths in RUNTIME_SOURCE_COMPONENTS.items()
            },
        },
        "execution_fingerprint": source.execution_fingerprint,
        "semantic_configuration": source.semantic_configuration,
        "semantic_contract": source.semantic_contract,
        "model_strength_claim": False,
    }
    output = Path(args.evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "interrupt_failed": interrupt_result["trainer_failed"],
        "payload_rejected": interrupt_result["checkpoint_payload"]["rejected"],
        "recovery_iteration": recovery["iteration"],
        "gamma_loaded": semantic["discount_gamma"]["loaded"],
        "heads_loaded": semantic["decoder_heads"]["loaded"],
        "logging_change": compatible_decision.kind,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
