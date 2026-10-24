"""Shared builders for the phase-3 actor-fuse / AMP / checkpoint tests.

Nothing here calls the production fuse helpers to derive an expected value: the
builders only construct *inputs* (rollouts, configs, plain tensors).  Every
assertion in the tests is then computed from those inputs with explicit scalar /
dense-torch references.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import torch

from guandan_marvel.config import (
    DDPConfig, LEGACY_POLICY_REGULARIZER_MODE, MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_FUSE_MODEL_ARCHITECTURE, ModelConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.model import Agent, TensorDict
from guandan_marvel.testing.fake_game import FAKE_GAME_SPEC
from guandan_marvel.training.smoke import CppSmokeGame

FUSE_ARCH = MACRO_FUSE_MODEL_ARCHITECTURE
LEGACY_ARCH = "v1.3"


# ---------------------------------------------------------------------------
# Model-level fixtures (tiny fake game, no trainer, no sampler)
# ---------------------------------------------------------------------------


def tiny_config(
    architecture: str = FUSE_ARCH,
    *,
    feature_dim: int = 24,
    history_dim: int = 16,
    fuse_layers: int = 2,
    state_layers: int = 1,
    action_layers: int = 1,
) -> TrainConfig:
    cfg = TrainConfig(
        game = FAKE_GAME_SPEC,
        device = "cpu",
        algo_name = "ippo",
        run_name = "fuse-unit",
        logging_dir = "runs/marvel_v14_phase3/unit",
        rollout_batch_size = 2,
        collect_batch_size = 2,
        max_sampler_bytes_per_rank = 256 * 2**20,
        model = ModelConfig(
            architecture = architecture,
            history_dim = history_dim,
            feature_dim = feature_dim,
            decoder_layers = 1,
            decoder_heads = 4,
            state_layers = state_layers,
            action_layers = action_layers,
            fuse_layers = fuse_layers,
        ),
    )
    cfg.compile.enabled = False
    cfg.optimizer.use_muon = False
    return cfg


def tiny_rollout(
    *,
    batch: int = 2,
    steps: int = 3,
    seed: int = 11,
    legal_per_time: Sequence[Sequence[int]] | None = None,
    weights: Sequence[Sequence[Sequence[float]]] | None = None,
    drop_last_time: bool = True,
) -> TensorDict:
    """A rollout for ``FAKE_GAME_SPEC`` with an exactly controllable legal mask.

    ``legal_per_time[t]`` lists the legal action indices at time ``t`` (the same
    for every batch row); ``weights`` overrides the per-(b, t, a) weights.  When
    ``drop_last_time`` is set the last batch row's final step is masked out
    entirely (``player_ids == -1``), so it exercises the "whole row illegal"
    branch of the sparse fuse.
    """
    spec = FAKE_GAME_SPEC
    players, actions = spec.num_players, spec.max_available_actions
    generator = torch.Generator().manual_seed(seed)
    obs_tokens = torch.randint(
        1, spec.num_tokens, (batch, steps, players, spec.max_observation_tokens), generator = generator,
    )
    obs_channels = torch.randint(
        0, 2, (batch, steps, players, spec.num_channels), dtype = torch.uint8, generator = generator,
    )
    avail_tokens = torch.randint(
        1, spec.num_tokens, (batch, steps, actions, spec.max_action_tokens), generator = generator,
    )
    player_ids = torch.arange(steps).remainder(players).expand(batch, -1).clone()
    num_steps = torch.full((batch,), steps, dtype = torch.long)
    if drop_last_time and batch > 1:
        num_steps[-1] = max(1, steps - 1)
        player_ids[-1, num_steps[-1]:] = -1

    avail_weights = torch.zeros((batch, steps, actions), dtype = torch.float32)
    if legal_per_time is None:
        avail_weights.fill_(1.0)
    else:
        for t, legal in enumerate(legal_per_time):
            if t >= steps:
                break
            for action in legal:
                avail_weights[:, t, action] = 1.0
    if weights is not None:
        for b, rows in enumerate(weights):
            for t, row in enumerate(rows):
                avail_weights[b, t] = torch.tensor(row, dtype = torch.float32)

    return TensorDict(
        {
            "num_steps": num_steps,
            "player_ids": player_ids,
            "obs_tokens": obs_tokens,
            "obs_channels": obs_channels,
            "avail_tokens": avail_tokens,
            "avail_weights": avail_weights,
        },
        batch_size = [batch],
    )


def deterministic_agent(
    cfg: TrainConfig,
    *,
    seed: int = 1234,
    enable_value_head: bool = True,
    enable_ntp_head: bool = False,
) -> Agent:
    torch.manual_seed(seed)
    agent = Agent(cfg, enable_value_head = enable_value_head, enable_ntp_head = enable_ntp_head)
    return agent


def migrate_legacy_into_fuse(target: Agent, source: Agent) -> tuple[list[str], list[str]]:
    """Load a v1.3 actor_state into a v1.4 model; returns (missing, unexpected)."""
    missing, unexpected = target.load_state_dict(source.state_dict(), strict = False)
    return sorted(missing), sorted(unexpected)


def capture_logits(model: Agent, rollout: TensorDict) -> tuple[object, torch.Tensor]:
    """Run the agent and capture the *pre-softmax* logits the model uses.

    The hook fires on ``logit_head`` (shape ``[B, T, A, 1]``) and squeezes the
    trailing singleton so the captured tensor is exactly the ``avail_logits`` the
    forward pass builds *before* masking / the ``log(weight)`` correction.
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        captured["logits"] = output.detach().clone().squeeze(-1)

    handle = model.logit_head.register_forward_hook(_hook)
    try:
        output = model(rollout = rollout)
    finally:
        handle.remove()
    return output, captured["logits"]


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(parameters: Mapping[str, torch.nn.Parameter] | None = None, *, module=None) -> str:
    digest = hashlib.sha256()
    source = (
        dict(module.named_parameters()) if module is not None else dict(parameters or {})
    )
    for name in sorted(source):
        tensor = source[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def dense_fuse_reference(
    fuse_module: torch.nn.Module,
    player_state: torch.Tensor,
    action_feats: torch.Tensor,
) -> torch.Tensor:
    """Dense (no gather/scatter) reference for the residual branch.

    Computed straight from the raw four-way concatenation, never via the
    production gather helper, so it independently pins ``W_out * trunk(W_in u)``.
    """
    batch, steps, _ = player_state.shape
    actions = action_feats.shape[2]
    state = player_state.unsqueeze(-2).expand(batch, steps, actions, player_state.shape[-1])
    features = torch.cat(
        (state, action_feats, state * action_feats, (state - action_feats).abs()), dim = -1,
    )
    return fuse_module(features)


# ---------------------------------------------------------------------------
# Trainer-level fixtures (CppSmokeGame, real sampler)
# ---------------------------------------------------------------------------


def game_rollout(game, *, batch: int = 1, steps: int = 3, seed: int = 5) -> TensorDict:
    """A shape-correct rollout for an arbitrary game contract (forward-only probe).

    Only the tensor capacities matter for an ``Agent`` forward pass; no phase-1
    action decoding happens here.  The last batch row's final step is masked out
    (``player_ids == -1``) so the probe exercises the masked-row branch too.
    """
    num_tokens = int(game.num_tokens)
    players = int(game.num_players)
    channels = int(game.num_channels)
    obs_len = int(game.max_observation_tokens)
    action_len = int(game.max_action_tokens)
    actions = int(game.max_available_actions)
    generator = torch.Generator().manual_seed(seed)
    obs_tokens = torch.randint(1, num_tokens, (batch, steps, players, obs_len), generator = generator)
    obs_channels = torch.randint(
        0, 2, (batch, steps, players, channels), dtype = torch.uint8, generator = generator,
    )
    avail_tokens = torch.randint(1, num_tokens, (batch, steps, actions, action_len), generator = generator)
    avail_weights = torch.full((batch, steps, actions), 0.5, dtype = torch.float32)
    player_ids = torch.arange(steps).remainder(players).expand(batch, -1).clone()
    num_steps = torch.full((batch,), steps, dtype = torch.long)
    if batch > 1:
        num_steps[-1] = max(1, steps - 1)
        player_ids[-1, num_steps[-1]:] = -1
        avail_weights[-1, num_steps[-1]:, :] = 0.0
    return TensorDict(
        {
            "num_steps": num_steps,
            "player_ids": player_ids,
            "obs_tokens": obs_tokens,
            "obs_channels": obs_channels,
            "avail_tokens": avail_tokens,
            "avail_weights": avail_weights,
        },
        batch_size = [batch],
    )


def smoke_trainer_config(
    architecture: str = FUSE_ARCH,
    *,
    seed: int = 4242,
    horizon: int = 512,
    amp_enabled: bool = False,
    fuse_layers: int = 1,
    ddp: bool = False,
    run_name: str = "phase3-fuse",
    logging_dir: str = "runs/marvel_v14_phase3/smoke",
) -> tuple[TrainConfig, CppSmokeGame]:
    game = CppSmokeGame(max_game_horizon = horizon)
    cfg = TrainConfig(
        game = game.spec,
        device = "cpu",
        algo_name = "ippo",
        run_name = run_name,
        logging_dir = logging_dir,
        seed = seed,
        rollout_batch_size = 2 if ddp else 1,
        collect_batch_size = 2 if ddp else 1,
        replay_buffer_scale = 1,
        mini_batches = 1,
        total_iterations = 4,
        max_sampler_bytes_per_rank = 256 * 2**20,
        amp_enabled = amp_enabled,
        checkpoint_replay_buffer = True,
        model = ModelConfig(
            architecture = architecture,
            history_dim = 16,
            feature_dim = 24,
            decoder_layers = 1,
            decoder_heads = 4,
            state_layers = 1,
            action_layers = 1,
            fuse_layers = fuse_layers,
        ),
        ppo = PPOConfig(
            agent_epoch = 1,
            training_semantics_profile = MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode = LEGACY_POLICY_REGULARIZER_MODE,
        ),
        ddp = DDPConfig(enabled = ddp, backend = "gloo", require_cuda = False),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


def optimizer_param_ids(optimizer) -> set[int]:
    """Identity set of every parameter reachable from an optimizer / MultiOptimizer."""
    from guandan_marvel.optim.common import MultiOptimizer

    optimizers = (
        list(optimizer.optimizers.values()) if isinstance(optimizer, MultiOptimizer) else [optimizer]
    )
    ids: set[int] = set()
    for entry in optimizers:
        for group in entry.param_groups:
            for parameter in group["params"]:
                ids.add(id(parameter))
    return ids


def group_of(optimizer, parameter: torch.nn.Parameter) -> tuple[str, float]:
    """Return ``(optimizer_name, weight_decay)`` for the group holding *parameter*."""
    from guandan_marvel.optim.common import MultiOptimizer

    entries = (
        dict(optimizer.optimizers) if isinstance(optimizer, MultiOptimizer) else {"adamw": optimizer}
    )
    for name, entry in entries.items():
        for group in entry.param_groups:
            if any(candidate is parameter for candidate in group["params"]):
                return name, float(group.get("weight_decay", 0.0))
    return "<missing>", float("nan")


__all__ = [
    "FUSE_ARCH", "LEGACY_ARCH",
    "capture_logits", "dense_fuse_reference", "deterministic_agent", "game_rollout",
    "group_of", "migrate_legacy_into_fuse", "optimizer_param_ids", "parameter_hash",
    "smoke_trainer_config", "state_hash", "tiny_config", "tiny_rollout",
]
