"""Two-rank gloo worker for the phase-2 macro DDP checks.

Modes:
- ``synthetic``: rank-specific macro counts; runs the production macro PPO/value/KL
  functions with ``world_size=2`` all-reduce statistics, backward through a shared
  scalar parameter and dump the gradient/loss/counts to JSON.
- ``trainer``: one real IPPOTrainer iteration (Profile C) under CPU gloo DDP.
- ``empty_rank``: one *real* IPPOTrainer iteration with an injected deterministic
  rollout, driving Profile B/C through the production ``_macro_train_iteration``.
  Under two-rank DDP rank 0 holds zero macros (``n_0 = 0``) while rank 1 holds eight
  (``N = 8``); a single-process run (``WORLD_SIZE`` unset) is the reference over
  rank 1's samples only.  The worker dumps per-rank local (pre-all-reduce) gradients,
  post-backward gradients, the exact macro count and the local metric record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from guandan_marvel.config import (  # noqa: E402
    DDPConfig, LEGACY_POLICY_REGULARIZER_MODE, MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE, ModelConfig, PPOConfig,
    SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE, TrainConfig,
)
from guandan_marvel.model.tensordict import TensorDict  # noqa: E402
from guandan_marvel.training.local import IPPOTrainer  # noqa: E402
from guandan_marvel.training.macro_actions import (  # noqa: E402
    ACTION_KIND_BASE, ACTION_KIND_NAMES, ACTION_PAYLOAD, ACTION_ROW_BEGIN, ACTION_ROW_END,
)
from guandan_marvel.training.macro_ppo import (  # noqa: E402
    build_macro_batch_view, compute_macro_normalized_kl_mean, compute_macro_pg_loss,
    compute_macro_value_loss,
)
from guandan_marvel.training.smoke import CppSmokeGame  # noqa: E402
from guandan_marvel.training.validation import validate_complete_trajectories  # noqa: E402
from tests.marvel_training._macro_fixtures import make_case  # noqa: E402

RANK_CASES = {
    0: {
        "players": [0, 1],
        "commits": [True, True],
        "rows": [[0.45, 0.25, 0.15, 0.15], [0.4, 0.3, 0.2, 0.1]],
    },
    1: {
        "players": [0, 0, 1, 2, 3, 1, 2, 3, 0],
        "commits": [False, True, True, True, True, True, True, True, True],
        "rows": [
            [0.2, 0.5, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
            [0.5, 0.2, 0.2, 0.1],
            [0.35, 0.35, 0.2, 0.1],
            [0.3, 0.3, 0.3, 0.1],
            [0.4, 0.2, 0.2, 0.2],
            [0.35, 0.25, 0.25, 0.15],
            [0.45, 0.25, 0.2, 0.1],
            [0.5, 0.2, 0.2, 0.1],
        ],
    },
}


def synthetic_case(rows_spec):
    return make_case(  # noqa: F821 - imported at module scope
        rows_spec["players"], rows_spec["commits"],
        horizon=12,
        avail_probs_rows=rows_spec["rows"],
    )


def stack_cases(cases):
    from guandan_marvel.model.tensordict import TensorDict
    from guandan_marvel.training.macro_actions import segment_macro_actions

    length = max(case[0]["player_ids"].shape[1] for case in cases)
    fields = {}
    for name in cases[0][0].keys():
        tensors = []
        for rollout, _values, _seg, _probs in cases:
            tensor = rollout[name]
            if tensor.dim() == 1:
                tensors.append(tensor)
                continue
            if tensor.shape[1] < length:
                padding = torch.zeros(
                    (tensor.shape[0], length - tensor.shape[1], *tensor.shape[2:]),
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, padding], dim=1)
            tensors.append(tensor)
        fields[name] = torch.cat(tensors, dim=0)
    rollout = TensorDict(fields, batch_size=[len(cases)])
    raw_values = [case[1] for case in cases]
    value_length = max(value.shape[1] for value in raw_values)
    padded_values = []
    for value in raw_values:
        if value.shape[1] < value_length:
            value = torch.cat([
                value,
                torch.zeros(
                    (value.shape[0], value_length - value.shape[1], *value.shape[2:]),
                    dtype=value.dtype,
                ),
            ], dim=1)
        padded_values.append(value)
    values = torch.cat(padded_values, dim=0)
    segmentation = segment_macro_actions(
        num_steps=rollout["num_steps"],
        player_ids=rollout["player_ids"],
        action_taken=rollout["action_taken"],
        avail_is_commit=rollout["avail_is_commit"],
        behavior_action_prob=rollout["behavior_action_prob"],
        is_terminal=rollout["is_terminal"],
    )
    probabilities = torch.cat([case[3] for case in cases], dim=0)
    return rollout, values, segmentation, probabilities


def macro_inputs(segmentation):
    """Deterministic advantage/value targets and a start-index prediction basis."""
    advantage = torch.zeros((segmentation.batch_size, segmentation.max_macros), dtype=torch.float32)
    target = torch.zeros_like(advantage)
    prediction_base = torch.zeros(
        (segmentation.batch_size, segmentation.horizon), dtype=torch.float32,
    )
    for b in range(segmentation.batch_size):
        for m in range(segmentation.max_macros):
            if not bool(segmentation.macro_valid[b, m]):
                continue
            start = float(int(segmentation.macro_start_index[b, m]))
            sign = 1.0 if m % 2 == 0 else -1.0
            advantage[b, m] = sign * (start + 1.0) * 0.5
            target[b, m] = 0.3 * (start + 1.0) - 0.2
            prediction_base[b, int(start)] = target[b, m]
    return advantage, target, prediction_base


def _scaled_log_probs(rollout, probabilities, w):
    """Scale the legal-action log-probs while keeping padding rows finite.

    Padding rows are masked to zero log-probability *before* the scaling so the
    autograd graph never contains ``-inf * w`` (a genuine NaN source), exactly
    like a real model whose softmax always sees at least one masked-legal action.
    """
    steps = rollout["num_steps"].unsqueeze(1)
    time = torch.arange(probabilities.shape[1]).unsqueeze(0)
    active = time < steps
    base = torch.where(
        active.unsqueeze(-1), probabilities.clamp_min(1e-6).log(), torch.zeros_like(probabilities),
    )
    return base * w


def synthetic_losses(rollout, segmentation, probabilities, advantage, target, prediction_base, w, world_size):
    view = build_macro_batch_view(segmentation, torch.arange(segmentation.batch_size))
    scaled_log_probs = _scaled_log_probs(rollout, probabilities, w)
    scaled_probs = torch.softmax(scaled_log_probs, dim=-1)
    pg = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=scaled_log_probs,
        macro_advantage=advantage, clip_coef=0.2, world_size=world_size,
    )
    value = compute_macro_value_loss(
        player_state_value_pred=prediction_base * w,
        view=view, macro_value_target=target, world_size=world_size,
    )
    kl = compute_macro_normalized_kl_mean(
        rollout, view=view, avail_probs_new=scaled_probs, world_size=world_size,
    )
    return pg, value, kl


def per_sample_mean(loss_value: float, *, global_count: int, local_count: int, world_size: int) -> float:
    """Mirror the production log representation: DDP-scaled loss -> per-sample mean."""
    if local_count <= 0:
        return 0.0
    return float(loss_value) * (float(global_count) / (float(world_size) * float(local_count)))


def run_synthetic(rank: int, out_path: Path) -> None:
    import torch.distributed as dist_module

    if not dist_module.is_initialized():
        dist_module.init_process_group("gloo")
    rollout, _values, segmentation, probabilities = synthetic_case(RANK_CASES[rank])
    advantage, target, prediction_base = macro_inputs(segmentation)
    w = torch.tensor(0.7, requires_grad=True)
    world_size = dist_module.get_world_size()
    pg, value, kl = synthetic_losses(
        rollout, segmentation, probabilities, advantage, target, prediction_base, w, world_size,
    )
    total = pg.pg_loss_mean + 0.5 * value.macro_value_loss_mean + 0.25 * kl
    total.backward()
    active_tokens = int((segmentation.token_to_macro_id >= 0).sum().item())
    payload = {
        "rank": rank,
        "world_size": world_size,
        "grad": float(w.grad.item()),
        "pg_loss": float(pg.pg_loss_mean.detach().item()),
        "value_loss": float(value.macro_value_loss_mean.detach().item()),
        "kl": float(kl.detach().item()),
        "total": float(total.detach().item()),
        "macro_count_local": pg.macro_count_local,
        "macro_count_global": pg.macro_count_global,
        "value_count_global": value.macro_count_global,
        "adv_norm_mean": pg.adv_norm_mean,
        "pg_clip_frac": pg.clip_frac,
        "local_macro_count": int(segmentation.num_macros.sum().item()),
        # Mirrors the production per-rank metric record (local.py _macro_train_iteration).
        "local_record": {
            "macro_initial/macro_count_local": float(pg.macro_count_local),
            "macro_initial/macro_count_global": float(pg.macro_count_global),
            "macro_initial/token_count_local": float(active_tokens),
            "macro_initial/pg_loss": per_sample_mean(
                float(pg.pg_loss_mean.detach().item()),
                global_count=pg.macro_count_global,
                local_count=pg.macro_count_local,
                world_size=world_size,
            ),
            "macro_initial/value_loss": per_sample_mean(
                float(value.macro_value_loss_mean.detach().item()),
                global_count=value.macro_count_global,
                local_count=value.macro_count_local,
                world_size=world_size,
            ),
            "macro_initial/adv_norm_mean": pg.adv_norm_mean,
            "macro_initial/adv_norm_mean_global": pg.adv_norm_mean_global,
            "macro_initial/adv_count_global": float(pg.adv_count_global),
            "macro/value_sample_unit": "macro_start",
        },
    }
    out_file = out_path.parent / f"{out_path.name}.rank{rank}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    out_file.write_text(json.dumps(payload), encoding="utf-8")
    dist_module.destroy_process_group()


def run_trainer(rank: int, out_path: Path) -> None:
    game = CppSmokeGame(max_game_horizon=512)
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name="ippo",
        logging_dir=str(out_path.parent / f"rank-{rank}"),
        run_name=f"ddp-macro-{rank}",
        seed=777,
        rollout_batch_size=2,
        collect_batch_size=2,
        replay_buffer_scale=1,
        mini_batches=2,
        total_iterations=1,
        max_sampler_bytes_per_rank=256 * 2**20,
        amp_enabled=False,
        checkpoint_replay_buffer=True,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            agent_epoch=1,
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=LEGACY_POLICY_REGULARIZER_MODE,
        ),
        ddp=DDPConfig(enabled=True, backend="gloo", require_cuda=False),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    trainer = IPPOTrainer(cfg, game=game, run_dir=out_path.parent / f"trainer-rank-{rank}")
    metrics = trainer.train_iteration()
    payload = {
        "rank": int(trainer.rank),
        "world_size": int(trainer.world_size),
        "iteration": int(trainer.iteration),
        "durable_updates": int(trainer.durable_updates),
        "learner_updates": int(trainer.learner_updates),
        "profile": metrics["macro/profile"],
        "regularizer_mode": metrics["macro/regularizer_mode"],
        "macro_count_global": metrics["macro_initial/macro_count_global"],
        "macro_count_local": metrics["macro_initial/macro_count_local"],
        "adv_raw_mean_global": metrics["macro_initial/adv_raw_mean_global"],
        "adv_raw_std_global": metrics["macro_initial/adv_raw_std_global"],
        "adv_norm_mean_global": metrics["macro_initial/adv_norm_mean_global"],
        "adv_norm_std_global": metrics["macro_initial/adv_norm_std_global"],
        "adv_count_global": metrics["macro_initial/adv_count_global"],
        "pg_loss": metrics["macro_initial/pg_loss"],
        "value_loss": metrics["macro_initial/value_loss"],
        "legacy_token_kl": metrics["macro_initial/legacy_token_kl"],
        "segment_violations": metrics["macro/diagnostic_player_switch_violation_count"],
        "incomplete": metrics["macro/diagnostic_incomplete_count"],
        "grad_norm": metrics["agent/grad_norm"],
        "ddp_enabled": bool(metrics["runtime/ddp_enabled"]),
        # The uncompressed per-rank local metric record (post-fix log representation).
        "local_record": {
            str(key): value for key, value in metrics.items()
            if key.startswith(("macro/", "macro_initial/"))
        },
    }
    trainer.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.parent / f"{out_path.name}.rank{rank}.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    out_file.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# empty_rank mode: real trainer iteration with an empty local macro batch
# ---------------------------------------------------------------------------

# rank 0 (empty): every step is a chance step (player_ids < 0) so the phase-1
# segmentation yields zero macros -> max_macros == 0 and a constant macro loss.
_EMPTY_LOCAL_CASE = {
    "players": [-1, -1],
    "commits": [True, True],
    "rows": [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
}
# rank 1 (8 macros in a single trajectory); matches RANK_CASES[1].
_EIGHT_MACRO_CASE = {
    "players": [0, 0, 1, 2, 3, 1, 2, 3, 0],
    "commits": [False, True, True, True, True, True, True, True, True],
    "rows": [[0.2, 0.5, 0.2, 0.1]] * 9,
}
_EMPTY_RANK_HORIZON = 12
_EMPTY_RANK_SEED = 777
# Deterministic, environment-independent parameter init so the DDP rank and the
# single-process reference start from bitwise-identical weights.
_EMPTY_RANK_INIT_SEED = 0xC0FFEE


def build_macro_ready_rollout(spec, game, *, horizon: int = _EMPTY_RANK_HORIZON) -> TensorDict:
    """A deterministic rollout the real ``Agent`` forward can consume.

    ``make_case`` supplies the segmentation-relevant fields (player_ids,
    action_taken, avail_is_commit, behavior_action_prob, policy_probs,
    on_policy_valid, is_terminal, rewards, num_steps); the model also needs
    ``obs_tokens``/``obs_channels``/``avail_tokens``.  The obs tensors are
    synthesised (only shape/dtype matter for the forward).  ``avail_tokens`` is
    filled with a *valid* GD-ACTION-0.2 framing row on the selected action so the
    phase-1 strict decoder accepts it, with COMMIT/FAMILY kinds agreeing with
    ``avail_is_commit`` (the decoded-kind contract).
    """
    rollout, _values, _segmentation, _probs = make_case(
        spec["players"], spec["commits"], horizon=horizon, avail_probs_rows=spec["rows"],
        dtype=torch.float32,
    )
    batch, steps = rollout["player_ids"].shape
    players = int(game.num_players)
    channels = int(game.num_channels)
    action_tokens = int(game.max_action_tokens)
    actions = int(rollout["avail_weights"].shape[-1])
    fields = {name: rollout[name] for name in rollout.keys()}
    fields["obs_tokens"] = torch.ones((batch, steps, players, 4), dtype=torch.long)
    fields["obs_channels"] = torch.zeros((batch, steps, players, channels), dtype=torch.long)

    commit_kind = ACTION_KIND_BASE + ACTION_KIND_NAMES.index("COMMIT")
    family_kind = ACTION_KIND_BASE + ACTION_KIND_NAMES.index("FAMILY")
    avail_tokens = torch.zeros((batch, steps, actions, action_tokens), dtype=torch.long)
    num_steps = rollout["num_steps"].to(torch.long)
    action_taken = rollout["action_taken"].to(torch.long)
    for b in range(batch):
        for t in range(int(num_steps[b])):
            chosen = int(action_taken[b, t])
            kind = commit_kind if bool(spec["commits"][t]) else family_kind
            avail_tokens[b, t, chosen, :8] = torch.tensor(
                [ACTION_ROW_BEGIN, 0, 0, kind, ACTION_PAYLOAD, 0, 0, ACTION_ROW_END],
                dtype=torch.long,
            )
    fields["avail_tokens"] = avail_tokens
    return TensorDict(fields, batch_size=[batch])


class _InjectedMacroTrainer(IPPOTrainer):
    """IPPOTrainer whose rollout comes from a deterministic spec, not the sampler.

    The full production learner (``_macro_train_iteration``: model forward, macro
    PPO/value/KL, backward, DDP, optimizer step) runs unchanged; only the rollout
    source is replaced so the per-rank macro counts are controlled.
    """

    def __init__(self, *args, spec, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._injected_spec = spec

    def _collect(self, agent_model):
        self._emit_phase("rollout")
        started = time.perf_counter()
        rollout = build_macro_ready_rollout(self._injected_spec, self.game)
        seconds = time.perf_counter() - started
        validation = validate_complete_trajectories(rollout, require_on_policy=True)
        infos = [
            {"outcome_class": "UNKNOWN"} for _ in range(int(rollout.batch_size[0]))
        ]
        self.last_trajectory_keys = tuple(sorted(rollout.keys()))
        self.last_trajectory_num_steps = tuple(int(value) for value in rollout["num_steps"].tolist())
        return rollout, validation, infos, seconds


def _deterministic_init(trainer) -> None:
    generator = torch.Generator().manual_seed(_EMPTY_RANK_INIT_SEED)
    with torch.no_grad():
        for _name, parameter in trainer.agent_resume.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)


def _apply_rank_averaged_reference(trainer, profile, regularizer_mode, world_size: int) -> None:
    """Make a single process reproduce the DDP gradient average of an empty world.

    Under DDP every rank's gradient is the mean over ``world_size`` ranks.  With
    one non-empty rank and ``world_size - 1`` empty ranks that factor is exactly
    ``1 / world_size``.  The world-size-weighted macro terms already carry it
    (``compute_macro_pg_loss``/``compute_macro_value_loss`` scale by
    ``world_size / global_count``), but the *rank-local-mean* terms - the legacy
    token KL regularizer and the Profile B token value loss - are plain means, so
    the reference must divide their coefficients by ``world_size`` to match.
    """
    scale = 1.0 / float(world_size)
    if profile == SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE:
        # Profile B token value loss is an unweighted mean (world_size-independent).
        trainer.cfg.value_loss_coef = trainer.cfg.value_loss_coef * scale
    if regularizer_mode == LEGACY_POLICY_REGULARIZER_MODE:
        # The legacy token-KL regularizer is an unweighted mean as well.
        import dataclasses

        import guandan_marvel.training.local as local_module

        base_schedule = local_module.compute_agent_schedule

        def _scaled_schedule(cfg, *, iteration):
            schedule = base_schedule(cfg, iteration=iteration)
            return dataclasses.replace(schedule, reg_coef=schedule.reg_coef * scale)

        local_module.compute_agent_schedule = _scaled_schedule


def empty_rank_game_config(profile: str, regularizer_mode: str, world_size: int):
    """Config/game for the empty-rank scenario at a given world size (reused by tests)."""
    game = CppSmokeGame(max_game_horizon=_EMPTY_RANK_HORIZON)
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name="ippo",
        logging_dir="empty-rank",
        run_name="empty-rank",
        seed=_EMPTY_RANK_SEED,
        rollout_batch_size=world_size,
        collect_batch_size=world_size,
        replay_buffer_scale=1,
        mini_batches=1,
        total_iterations=1,
        max_sampler_bytes_per_rank=256 * 2**20,
        amp_enabled=False,
        checkpoint_replay_buffer=True,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            agent_epoch=1,
            training_semantics_profile=profile,
            policy_regularizer_mode=regularizer_mode,
        ),
        ddp=DDPConfig(enabled=world_size > 1, backend="gloo", require_cuda=False),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


def run_empty_rank(
    profile: str, regularizer_mode: str, out_path: Path, *, emulate_world_size: int = 1,
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    ddp_enabled = world_size > 1
    if not ddp_enabled and dist.is_initialized():  # pragma: no cover - defensive
        raise RuntimeError("single-process reference must not join a process group")

    cfg, game = empty_rank_game_config(profile, regularizer_mode, world_size)

    # The empty rank under DDP holds no macro; the single-process reference holds
    # only the eight-macro samples (N = 8) so it matches the DDP global average.
    spec = _EMPTY_LOCAL_CASE if (ddp_enabled and rank == 0) else _EIGHT_MACRO_CASE
    spec_name = "empty" if spec is _EMPTY_LOCAL_CASE else "eight"
    trainer = _InjectedMacroTrainer(
        cfg, game=game, run_dir=out_path.parent / f"empty-rank-{rank}", spec=spec,
    )
    _deterministic_init(trainer)
    if emulate_world_size > 1:
        _apply_rank_averaged_reference(trainer, profile, regularizer_mode, emulate_world_size)

    parameter_names = [name for name, _ in trainer.agent_resume.named_parameters()]
    local_grad: dict[str, float] = {}

    def _record(name):
        def hook(grad):
            local_grad[name] = float(grad.detach().float().abs().sum().item())
        return hook

    for name, parameter in trainer.agent_resume.named_parameters():
        parameter.register_hook(_record(name))

    metrics = trainer.train_iteration()

    post_grad = {
        name: (None if parameter.grad is None else float(parameter.grad.detach().float().abs().sum().item()))
        for name, parameter in trainer.agent_resume.named_parameters()
    }
    post_norm = float(torch.sqrt(sum(
        (parameter.grad.detach().float().pow(2).sum() if parameter.grad is not None else 0.0)
        for _, parameter in trainer.agent_resume.named_parameters()
    )).item())

    local_record = {
        str(key): value for key, value in metrics.items()
        if key.startswith(("macro/", "macro_initial/"))
    }
    payload = {
        "rank": int(trainer.rank),
        "world_size": int(trainer.world_size),
        "ddp_enabled": ddp_enabled,
        "profile": metrics["macro/profile"],
        "regularizer_mode": metrics["macro/regularizer_mode"],
        "spec": spec_name,
        "iteration": int(trainer.iteration),
        "learner_updates": int(trainer.learner_updates),
        "macro_count_local": metrics["macro_initial/macro_count_local"],
        "macro_count_global": metrics["macro_initial/macro_count_global"],
        "grad_norm": metrics["agent/grad_norm"],
        "local_record": local_record,
        "parameter_names": parameter_names,
        # Hook values are the rank-local gradients captured *before* the DDP
        # all-reduce; post_grad is the post-backward (DDP-averaged) gradient.
        "local_grad": local_grad,
        "post_grad": post_grad,
        "post_grad_norm": post_norm,
    }
    trainer.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.parent / f"{out_path.name}.rank{rank}.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    out_file.write_text(json.dumps(payload), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "trainer", "empty_rank"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--emulate-world-size", type=int, default=1,
        help="single-process reference only: reproduce the DDP average of an empty world",
    )
    parser.add_argument(
        "--profile",
        choices=(SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE, MACRO_DECISION_TRAINING_SEMANTICS_PROFILE),
        default=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    )
    parser.add_argument(
        "--regularizer-mode",
        choices=(LEGACY_POLICY_REGULARIZER_MODE, MACRO_NORMALIZED_POLICY_REGULARIZER_MODE),
        default=LEGACY_POLICY_REGULARIZER_MODE,
    )
    args = parser.parse_args(argv)
    out_path = Path(args.out)
    if args.mode == "synthetic":
        run_synthetic(args.rank, out_path)
    elif args.mode == "trainer":
        run_trainer(args.rank, out_path)
    else:
        run_empty_rank(
            args.profile, args.regularizer_mode, out_path,
            emulate_world_size=args.emulate_world_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
