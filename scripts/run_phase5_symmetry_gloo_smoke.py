# SPDX-License-Identifier: MIT
"""CPU dual-rank gloo check of the symmetry loss's DDP aggregation.

Validates, without any NCCL/T4 claim:

* the ``all_reduce`` global token count is identical on every rank;
* a rank with zero local active tokens contributes an exact, graph-connected
  zero and no spurious gradient;
* the differentiable local loss is ``world_size / global_count * local_sum``,
  so summing rank losses reproduces the single-process concatenated reference.

Run with the guarded phase-5 runner; the script exits non-zero on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29555")

import torch

ROOT = Path(__file__).resolve().parents[1]


def _tiny_agent():
    from guandan_marvel import python_game
    from guandan_marvel.config import ModelConfig, TrainConfig
    from guandan_marvel.model import Agent

    cfg = TrainConfig(
        game=python_game.GAME_SPEC, device="cpu",
        rollout_batch_size=1, collect_batch_size=1,
        max_sampler_bytes_per_rank=64 * 2**20,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.optimizer.use_muon = False
    torch.manual_seed(7)
    return Agent(cfg, enable_value_head=True, enable_ntp_head=False)


def _write_fixture(root: Path):
    from guandan_marvel.phase5.design import phase5_design_hash
    from guandan_marvel.phase5.symmetry_training import canonical_hash

    agent = _tiny_agent()
    checkpoint = root / "init.pt"
    checkbox_bytes = b"phase5-gloo-initial-checkpoint"
    checkpoint.write_bytes(checkbox_bytes)
    document = {
        "schema": "guandan-phase5-symmetry-diagnostic-run-v1",
        "status": "COMPLETED", "training": False, "truncated": False,
        "deals_requested": 16, "design_hash": phase5_design_hash(),
        "checkpoint_sha256": hashlib.sha256(checkbox_bytes).hexdigest(),
        "source_policy_identity": {
            "architecture": "v1.3", "algorithm": "ippo",
            "history_dim": 16, "feature_dim": 24,
            "parameter_count": sum(v.numel() for v in agent.state_dict().values()),
            "state_dict_keys": len(agent.state_dict()),
        },
        "decision": {
            "status": "SIGNIFICANT", "augmentation_recommended": True,
            "sample_count": 1536, "deal_count": 16, "non_finite_kl_count": 0,
            "mean_kl": 0.02857, "argmax_disagreement_rate": 0.11263,
            "thresholds": {
                "min_macro_samples": 256, "min_deals": 8,
                "mean_restricted_kl_threshold": 0.02,
                "argmax_disagreement_threshold": 0.10,
            },
        },
        "diagnostic": {
            "transforms": ["copy_swap"],
            "aggregate": {
                "mean_kl": 0.02857, "median_kl": 0.02, "max_kl": 0.2,
                "non_finite_kl_count": 0,
            },
        },
    }
    artifact = root / "artifact.json"
    artifact.write_text(json.dumps(document), encoding="utf-8")
    seal = {
        "schema": "guandan-marvel-phase5-symmetry-artifact-seal-v1",
        "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "content_hash": canonical_hash(document),
        "design_hash": document["design_hash"],
        "checkpoint_sha256": document["checkpoint_sha256"],
        "source_policy_identity": document["source_policy_identity"],
        "transforms": ["copy_swap"],
    }
    seal["seal_hash"] = canonical_hash(seal)
    seal_path = root / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return seal_path, checkpoint


def _rollout(*, active: bool):
    from guandan_marvel.model import TensorDict

    b, t, p, h, c, a, l = 1, 1, 4, 48, 512, 3, 8
    obs_tokens = torch.zeros((b, t, p, h), dtype=torch.long)
    obs_tokens[..., 0] = 1
    obs_tokens[..., 1] = 256
    obs_tokens[..., 2] = 2
    obs_tokens[..., 3] = 26
    obs_tokens[..., 4] = 38
    obs_tokens[..., 5] = 308
    obs_channels = torch.zeros((b, t, p, c), dtype=torch.uint8)
    for observer in range(p):
        obs_channels[:, :, observer, 364 + observer] = 1
        obs_channels[:, :, observer, 384] = 1
        obs_channels[:, :, observer, 388 + 4] = 1
        obs_channels[:, :, observer, 405 + (observer % 2)] = 1
        obs_channels[:, :, observer, 0 + observer] = 1
    rows = torch.zeros((b, t, a, l), dtype=torch.long)
    for index, token in enumerate((6, 64 + 3, 64 + 57)):
        rows[:, :, index] = torch.tensor([262, 2, 263, 296, 304, token, 306, 307])
    weights = torch.zeros((b, t, a))
    weights[..., :3] = 1.0
    return TensorDict({
        "num_steps": torch.tensor([1], dtype=torch.long),
        "player_ids": torch.tensor([[0 if active else -1]], dtype=torch.long),
        "obs_tokens": obs_tokens, "obs_channels": obs_channels,
        "avail_tokens": rows, "avail_weights": weights,
    }, batch_size=[b])


def _worker(rank: int, world_size: int, seal: str, checkpoint: str, init_file: str, queue) -> None:
    from datetime import timedelta

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        from guandan_marvel.phase5.symmetry_training import (
            SymmetryConsistency, load_sealed_symmetry_artifact,
        )

        agent = _tiny_agent()
        contract = load_sealed_symmetry_artifact(seal)
        consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
        # rank 0 holds one active decision, rank 1 holds none.
        active = rank == 0
        rollout = _rollout(active=active)
        output = agent(rollout)
        result = consistency.loss(
            rollout, base_log_probs=output.avail_log_probs,
            forward=lambda batch: agent(batch), world_size=world_size,
        )
        if result.token_count_global != 1:
            queue.put({"status": "FAIL", "rank": rank, "error": "global token count"})
            return
        if active:
            result.loss.backward()
            grads = {
                name: (None if p.grad is None else p.grad.detach().clone())
                for name, p in agent.named_parameters()
            }
            # The single-process reference must not join the two-rank collectives:
            # tear the group down first (the peer has already finished).
            torch.distributed.destroy_process_group()
            # Single-process reference: same decision, world 1, global 1.
            ref_agent = _tiny_agent()
            ref_agent.load_state_dict(agent.state_dict())
            ref_rollout = _rollout(active=True)
            ref_output = ref_agent(ref_rollout)
            ref = consistency.loss(
                ref_rollout, base_log_probs=ref_output.avail_log_probs,
                forward=lambda batch: ref_agent(batch), world_size=1,
            )
            ref.loss.backward()
            expected = ref.loss.detach().item() * world_size
            if not math.isclose(result.loss.detach().item(), expected, rel_tol=1e-6, abs_tol=1e-8):
                queue.put({"status": "FAIL", "rank": rank, "error": "weighted reduction"})
                return
            for name, parameter in ref_agent.named_parameters():
                reference = (
                    parameter.grad.detach()
                    if parameter.grad is not None
                    else torch.zeros_like(parameter)
                )
                actual = (
                    grads[name] if grads[name] is not None
                    else torch.zeros_like(parameter)
                )
                if not torch.allclose(
                    actual, reference * world_size, rtol=1e-5, atol=1e-8,
                ):
                    queue.put({"status": "FAIL", "rank": rank, "error": f"gradient {name}"})
                    return
            nonzero_transformed = sum(
                1 for value in grads.values()
                if value is not None and float(value.abs().sum()) > 0.0
            )
            if nonzero_transformed == 0:
                queue.put({
                    "status": "FAIL", "rank": rank,
                    "error": "transformed side produced no non-zero gradient",
                })
                return
            queue.put({
                "status": "PASS", "rank": rank,
                "loss": result.loss.detach().item(),
                "reference_loss": ref.loss.detach().item(),
                "mean_kl_global": result.mean_kl_global,
                "token_count_global": result.token_count_global,
            })
        else:
            if abs(result.loss.detach().item()) != 0.0:
                queue.put({"status": "FAIL", "rank": rank, "error": "zero-rank loss"})
                return
            result.loss.backward()
            if any(
                p.grad is not None and float(p.grad.abs().sum()) != 0.0
                for p in agent.parameters()
            ):
                queue.put({"status": "FAIL", "rank": rank, "error": "zero-rank gradient"})
                return
            queue.put({
                "status": "PASS", "rank": rank, "loss": 0.0,
                "token_count_global": result.token_count_global,
            })
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> int:
    import tempfile
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        seal, checkpoint = _write_fixture(root)
        context = mp.get_context("spawn")
        queue = context.Queue()
        init_file = str((root / "gloo_init").resolve())
        workers = [
            context.Process(
                target=_worker,
                args=(rank, 2, str(seal), str(checkpoint), init_file, queue),
            )
            for rank in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=150)
            if worker.exitcode != 0:
                print(json.dumps({"status": "FAIL", "error": f"worker exit {worker.exitcode}"}))
                return 1
        results = [queue.get(timeout=10) for _ in range(2)]
    passed = all(item.get("status") == "PASS" for item in results)
    print(json.dumps({"status": "PASS" if passed else "FAIL", "ranks": results}, indent=2))
    return 0 if passed else 1


def _run_parent(world_size: int, seal: str, checkpoint: str, port: int, queue) -> None:
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    workers = [
        context.Process(target=_worker, args=(rank, world_size, seal, checkpoint, port, queue))
        for rank in range(world_size)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=150)
        if worker.exitcode != 0:
            raise SystemExit(worker.exitcode)


if __name__ == "__main__":
    raise SystemExit(main())
