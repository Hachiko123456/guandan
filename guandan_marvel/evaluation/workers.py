# SPDX-License-Identifier: MIT
"""Process-isolated evaluation workers used by the M06 hook."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Mapping

import torch

from .checkpoints import load_v2_policy


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_policy_identity_work(*, snapshot: Path, iteration: int, config: Mapping[str, object]) -> dict[str, object]:
    """Real loader work unit used by repair tests and lightweight hook audits."""
    delay = float(config.get("delay_seconds", 0.0))
    if delay > 0:
        time.sleep(delay)
    artifact = load_v2_policy(snapshot, device=str(config.get("device", "cpu")))
    return {
        "schema": "guandan-marvel-evaluation-worker-result-v1",
        "iteration": int(iteration),
        "identity": artifact.identity,
        "model_state_sha256": _state_digest(artifact.model.state_dict()),
    }


def failing_work(*, snapshot: Path, iteration: int, config: Mapping[str, object]) -> dict[str, object]:
    raise RuntimeError(str(config.get("message", f"injected evaluation failure at iteration {iteration}")))


def fail_on_iterations_work(*, snapshot: Path, iteration: int, config: Mapping[str, object]) -> dict[str, object]:
    failures = {int(value) for value in config.get("fail_iterations", [])}
    if int(iteration) in failures:
        raise RuntimeError(f"injected worker failure at iteration {iteration}")
    return load_policy_identity_work(snapshot=snapshot, iteration=iteration, config=config)


def selection_evaluation_work(*, snapshot: Path, iteration: int, config: Mapping[str, object]) -> dict[str, object]:
    """Run a complete selection evaluation from immutable paths/config only."""
    from .common import load_json
    from .evaluator import EvaluationRunner
    from .pool import OpponentPoolStore
    from .promotion import ChampionStore, PromotionPolicy

    output_dir = Path(str(config["output_dir"])) / f"iteration-{int(iteration):06d}"
    runner = EvaluationRunner(
        output_dir=output_dir,
        champion_store=ChampionStore(str(config["champion_manifest"])),
        opponent_pool=OpponentPoolStore(str(config["opponent_pool_manifest"])),
        legacy_checkpoint=str(config["legacy_checkpoint"]),
        device=str(config.get("device", "cpu")),
        temperature=float(config.get("temperature", 0.0)),
        promotion_policy=PromotionPolicy.from_mapping(config.get("promotion_policy")),
        provenance_policy=str(config.get("provenance_policy", "formal")),
    )
    result = runner.run_selection(
        deals_manifest=load_json(str(config["deals_manifest"])),
        candidate_checkpoint=snapshot,
    )
    return {
        "schema": "guandan-marvel-evaluation-worker-result-v1",
        "iteration": int(iteration),
        "result_path": str((output_dir / "result.json").resolve()),
        "status": result["status"],
        "promotion_status": None if result.get("promotion_decision") is None else result["promotion_decision"]["status"],
        "evaluation_hash": result["evaluation_hash"],
    }


__all__ = ["fail_on_iterations_work", "failing_work", "load_policy_identity_work", "selection_evaluation_work"]
