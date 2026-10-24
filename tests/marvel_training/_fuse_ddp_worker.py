"""Two-rank gloo worker for the phase-3 actor-fuse DDP invariants.

Modes (``--mode``):

``fuse_ddp``
    Builds a real v1.4 ``IPPOTrainer`` under CPU gloo DDP with
    ``find_unused_parameters=False`` and a bitwise-identical deterministic init on
    every rank.  Rank-specific probe data is used so the all-reduce is actually
    observable, then it captures:

    1. the per-rank **local** (pre-all-reduce) gradients,
    2. the first DDP backward (post-all-reduce) gradients,
    3. the parameter hash after one real optimizer step,
    4. the second DDP backward gradients,
    5. one full production ``train_iteration()`` (Profile C macro learner).

Each rank dumps its own JSON so the test can compare across ranks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from guandan_marvel.training.local import IPPOTrainer  # noqa: E402
from tests.marvel_training._fuse_fixtures import (  # noqa: E402
    FUSE_ARCH, game_rollout, smoke_trainer_config,
)

INIT_SEED = 0xC0FFEE
PROBE_PARAMS = (
    "actor_fuse.output_proj.weight",
    "actor_fuse.input_proj.weight",
    "actor_fuse.trunk.layers.0.mlp.down_proj.weight",
    "actor_fuse.trunk.layers.0.norm.weight",
    "actor_fuse.trunk.output_norm.weight",
    "logit_head.weight",
)
TRUNK_PROBES = (
    "actor_fuse.input_proj.weight",
    "actor_fuse.trunk.layers.0.mlp.down_proj.weight",
    "actor_fuse.trunk.layers.0.norm.weight",
    "actor_fuse.trunk.output_norm.weight",
)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    tensor = tensor.detach().cpu().contiguous()
    return {
        "hash": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "sum": float(tensor.sum()),
        "absmax": float(tensor.abs().max()),
        "abs_sum": float(tensor.abs().sum()),
    }


def _grad_stats(module: torch.nn.Module, names=PROBE_PARAMS) -> dict[str, object]:
    parameters = dict(module.named_parameters())
    payload: dict[str, object] = {}
    for name in names:
        grad = parameters[name].grad
        payload[name] = {"present": grad is not None}
        if grad is not None:
            payload[name].update(_tensor_stats(grad))
    return payload


def _param_digest(module: torch.nn.Module, names=PROBE_PARAMS) -> str:
    parameters = dict(module.named_parameters())
    digest = hashlib.sha256()
    for name in names:
        tensor = parameters[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _deterministic_init(trainer: IPPOTrainer) -> None:
    """Bitwise-identical init on every rank.

    The residual gate ``actor_fuse.output_proj.weight`` is deliberately kept at
    its strict zero initialisation: that is the migrated-v1.3 state, and it is the
    precondition for the "first backward only produces a W_out gradient" gate.
    """
    generator = torch.Generator().manual_seed(INIT_SEED)
    with torch.no_grad():
        for name, parameter in trainer.agent_resume.named_parameters():
            if name == "actor_fuse.output_proj.weight":
                parameter.zero_()
                continue
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    trainer.agent_merged.load_state_dict(trainer.agent_resume.state_dict(), strict=True)
    if trainer.agent_optimizer is not None:
        trainer.agent_optimizer.zero_grad(set_to_none=True)


def _parameter_hash(module: torch.nn.Module, prefix: str = "actor_fuse.") -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not name.startswith(prefix):
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _make_loss(output) -> torch.Tensor:
    weights = torch.linspace(0.1, 0.9, output.avail_probs.shape[-1])
    return (output.avail_probs * weights).sum() + output.player_state_value.square().sum()


def run_fuse_ddp(rank: int, out_path: Path) -> None:
    if not dist.is_initialized():
        dist.init_process_group("gloo")

    cfg, game = smoke_trainer_config(
        FUSE_ARCH, seed=777, ddp=True, run_name=f"fuse-ddp-{rank}",
        logging_dir=str(out_path.parent / f"rank-{rank}"),
    )
    trainer = IPPOTrainer(cfg, game=game, run_dir=out_path.parent / f"trainer-rank-{rank}")
    payload: dict[str, object] = {"rank": rank, "world_size": int(trainer.world_size)}

    _deterministic_init(trainer)
    payload["init_fuse_hash"] = _parameter_hash(trainer.agent_resume)
    payload["init_probe_digest"] = _param_digest(trainer.agent_resume)
    payload["init_output_proj_absmax"] = float(
        trainer.agent_resume.actor_fuse.output_proj.weight.detach().abs().max()
    )
    payload["find_unused_parameters"] = bool(cfg.ddp.find_unused_parameters)
    payload["amp_enabled"] = bool(trainer.amp_enabled)
    payload["scenario"] = "fuse_ddp"

    # Rank-specific probe data: rank 0 and rank 1 must genuinely differ, otherwise
    # an all-reduce would be indistinguishable from a local-only backward.
    rollout = game_rollout(game, batch=1, steps=3, seed=5 + rank)

    # ---- 1. per-rank local (pre-all-reduce) gradients
    trainer.agent_resume.zero_grad(set_to_none=True)
    output = trainer.agent_resume(rollout=rollout)
    _make_loss(output).backward()
    payload["local_grad"] = _grad_stats(trainer.agent_resume)

    # ---- 2. first DDP backward (post-all-reduce)
    trainer.agent_resume.zero_grad(set_to_none=True)
    output = trainer._forward_agent(rollout)
    _make_loss(output).backward()
    payload["first_post_grad"] = _grad_stats(trainer.agent_resume)

    # ---- 3. one real optimizer step, then a parameter digest
    trainer.agent_optimizer.step()
    payload["post_step_fuse_hash"] = _parameter_hash(trainer.agent_resume)
    payload["post_step_probe_digest"] = _param_digest(trainer.agent_resume)
    payload["output_proj_nonzero_after_step"] = int(
        torch.count_nonzero(trainer.agent_resume.actor_fuse.output_proj.weight)
    )

    # ---- 4. second DDP backward
    trainer.agent_resume.zero_grad(set_to_none=True)
    output = trainer._forward_agent(rollout)
    _make_loss(output).backward()
    payload["second_post_grad"] = _grad_stats(trainer.agent_resume)

    # ---- 5. one full production macro iteration (find_unused_parameters=False)
    metrics = trainer.train_iteration()
    payload["iteration"] = int(trainer.iteration)
    payload["durable_updates"] = int(trainer.durable_updates)
    payload["grad_norm"] = float(metrics["agent/grad_norm"])
    payload["macro_count_global"] = float(metrics["macro_initial/macro_count_global"])
    payload["pg_loss"] = float(metrics["macro_initial/pg_loss"])
    payload["ddp_enabled"] = bool(metrics["runtime/ddp_enabled"])
    payload["profile"] = str(metrics["macro/profile"])

    trainer.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    out_file = out_path.parent / f"{out_path.name}.rank{rank}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


MODES = {"fuse_ddp": run_fuse_ddp}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    MODES[args.mode](args.rank, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
