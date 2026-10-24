"""Phase-3 MARVEL v1.4 actor residual fuse: architecture, sparsity and gradients.

Every expected value is derived independently (dense torch reference or plain
scalar arithmetic); no assertion is computed by calling the production fuse
helper it is supposed to check.
"""
from __future__ import annotations

import math

import pytest
import torch

from guandan_marvel.config import OptimizerConfig
from guandan_marvel.model import Agent
from guandan_marvel.model.base import ResGatedMLP
from guandan_marvel.optim.common import MultiOptimizer
from tests.marvel_training._fuse_fixtures import (
    FUSE_ARCH, LEGACY_ARCH, capture_logits, dense_fuse_reference, deterministic_agent,
    group_of, migrate_legacy_into_fuse, optimizer_param_ids, smoke_trainer_config,
    tiny_config, tiny_rollout,
)

FUSE_KEY_PREFIX = "actor_fuse."


def fuse_parameter_names(model: Agent) -> list[str]:
    return [name for name, _ in model.named_parameters() if name.startswith(FUSE_KEY_PREFIX)]


# ---------------------------------------------------------------------------
# 1. Architecture: fuse branch exists, is driven by fuse_layers, adds exactly
#    the declared parameters
# ---------------------------------------------------------------------------


def test_v14_actor_builds_fuse_branch_while_v13_actor_does_not():
    legacy = deterministic_agent(tiny_config(LEGACY_ARCH))
    fused = deterministic_agent(tiny_config(FUSE_ARCH))
    assert legacy.actor_fuse is None
    assert fused.actor_fuse is not None
    assert fused.actor_fuse_enabled is True and legacy.actor_fuse_enabled is False
    # The legacy multiply path is still present as an attribute-free code path:
    # the v1.4 model keeps every v1.3 parameter *and* gains only fuse keys.
    legacy_keys = set(dict(legacy.named_parameters()))
    fused_keys = set(dict(fused.named_parameters()))
    assert legacy_keys < fused_keys
    assert {key for key in fused_keys - legacy_keys} == set(fuse_parameter_names(fused))
    assert all(key.startswith(FUSE_KEY_PREFIX) for key in fused_keys - legacy_keys)


def test_fuse_trunk_layer_count_is_driven_by_fuse_layers():
    for n_layers in (1, 2, 3):
        cfg = tiny_config(FUSE_ARCH, fuse_layers=n_layers)
        model = deterministic_agent(cfg)
        trunk = model.actor_fuse.trunk
        assert len(trunk.layers) == n_layers
        # The v1.3 trunk sizes must be untouched by fuse_layers.
        legacy = deterministic_agent(tiny_config(LEGACY_ARCH, fuse_layers=n_layers))
        assert len(legacy.state_mlp) == len(deterministic_agent(tiny_config(LEGACY_ARCH)).state_mlp)


def test_fuse_parameter_count_matches_hand_computed_v13_delta():
    """The production-size fuse branch adds exactly ``11 * D^2 + 4 * D`` weights."""
    import guandan_marvel.game as game
    from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig

    def build(architecture: str) -> Agent:
        cfg = TrainConfig(
            game = GameSpec.from_game(game), device = "cpu", algo_name = "ippo",
            run_name = "param-check", logging_dir = ".", seed = 1,
            rollout_batch_size = 2, collect_batch_size = 2, mini_batches = 1,
            total_iterations = 1, ntp_weight = 0.0,
            model = ModelConfig(
                architecture = architecture, history_dim = 128, feature_dim = 256,
                decoder_layers = 4, decoder_heads = 4, state_layers = 3,
                action_layers = 2, fuse_layers = 2,
            ),
        )
        return Agent(cfg, enable_value_head = True)

    legacy_total = sum(p.numel() for p in build(LEGACY_ARCH).parameters())
    fused_total = sum(p.numel() for p in build(FUSE_ARCH).parameters())
    assert legacy_total == 2_061_314
    dim = 256
    layers = 2
    expected_fuse = (
        4 * dim * dim + dim            # input projection weight + bias
        + layers * (dim + 3 * dim * dim)  # per-layer norm + gate/up/down (ratio 1)
        + dim                          # trunk output norm
        + dim * dim                    # bias-free output projection
    )
    assert expected_fuse == 11 * dim * dim + 4 * dim == 721_920
    assert fused_total == legacy_total + expected_fuse == 2_783_234
    # Only fuse keys account for the delta.
    fused_agent = build(FUSE_ARCH)
    assert sum(p.numel() for p in fused_agent.actor_fuse.parameters()) == expected_fuse


# ---------------------------------------------------------------------------
# 2. FP32 zero-residual initialisation equivalence (bitwise)
# ---------------------------------------------------------------------------


def test_fp32_fuse_migration_is_bitwise_identical_no_autocast():
    torch.manual_seed(7)
    legacy = deterministic_agent(tiny_config(LEGACY_ARCH), seed = 99).eval()
    fused = deterministic_agent(tiny_config(FUSE_ARCH), seed = 99).eval()
    missing, unexpected = migrate_legacy_into_fuse(fused, legacy)
    assert unexpected == []
    assert missing and all(key.startswith(FUSE_KEY_PREFIX) for key in missing)
    assert set(missing) == set(fuse_parameter_names(fused))
    # The residual gate really is an independent, strictly zero projection.
    assert torch.equal(
        fused.actor_fuse.output_proj.weight.detach(),
        torch.zeros_like(fused.actor_fuse.output_proj.weight),
    )

    rollout = tiny_rollout(batch = 2, steps = 3)
    with torch.no_grad():
        old_out, old_logits = capture_logits(legacy, rollout)
        new_out, new_logits = capture_logits(fused, rollout)
    assert torch.equal(old_logits, new_logits)                       # torch.equal logits
    assert torch.equal(old_out.avail_probs, new_out.avail_probs)     # torch.equal probs
    assert torch.equal(
        torch.argmax(old_out.avail_probs, dim = -1),
        torch.argmax(new_out.avail_probs, dim = -1),
    )                                                                # torch.equal argmax
    assert torch.equal(old_out.avail_log_probs, new_out.avail_log_probs)
    assert torch.equal(old_out.player_state_value, new_out.player_state_value)


def test_legacy_multiply_path_is_preserved_and_the_fuse_is_purely_additive():
    """``z`` is still ``player_state * action_feats``; the fuse only adds to it."""
    torch.manual_seed(3)
    fused = deterministic_agent(tiny_config(FUSE_ARCH), seed = 5).eval()
    rollout = tiny_rollout(batch = 2, steps = 3)
    with torch.no_grad():
        _, with_fuse_zero = capture_logits(fused, rollout)
        assert torch.count_nonzero(fused.actor_fuse.output_proj.weight) == 0

        # Disable the branch entirely: this is structurally the pre-phase-3 actor
        # (no fuse branch), i.e. the legacy multiply path on its own.
        module = fused.actor_fuse
        fused.actor_fuse = None
        try:
            _, without_fuse = capture_logits(fused, rollout)
        finally:
            fused.actor_fuse = module
        # Independent recomputation of the legacy path (pre-mask logits) from the
        # model's own public feature encoders.
        state, action_feats, avail_mask = _state_and_action_features(fused, rollout)
        manual = fused.logit_head(state.unsqueeze(-2) * action_feats).squeeze(-1)

    assert torch.equal(without_fuse, manual)
    assert torch.equal(with_fuse_zero, manual)

    # With a non-zero gate the logits must change, i.e. the branch is live.
    with torch.no_grad():
        fused.actor_fuse.output_proj.weight.normal_(mean = 0.0, std = 0.05)
    with torch.no_grad():
        _, live = capture_logits(fused, rollout)
    assert not torch.equal(live, manual)
    assert float((live - manual).abs().max()) > 0.0


def _state_and_action_features(model: Agent, rollout):
    """Recompute (player_state, action_feats, avail_mask) with public helpers."""
    num_steps, player_ids = rollout["num_steps"], rollout["player_ids"]
    obs_tokens, obs_channels = rollout["obs_tokens"], rollout["obs_channels"]
    avail_tokens, avail_weights = rollout["avail_tokens"], rollout["avail_weights"]
    device = obs_tokens.device
    B, T, P, _ = obs_tokens.shape
    time_mask = model._build_time_mask(num_steps, T, device = device)
    flat_time_mask = time_mask.repeat_interleave(P, dim = 0)
    flat_player_ids = torch.arange(P, device = device).expand(B, P).reshape(-1, 1).expand(-1, T)
    flat_obs_tokens = model._flatten_players(obs_tokens, name = "obs_tokens")
    decoded, _ = model._encode_history(flat_obs_tokens, flat_player_ids, flat_time_mask)
    decoded_by_player = model._unflatten_players(decoded, batch_size = B, name = "decoded")
    acting = player_ids.clamp(min = 0)
    mask = player_ids >= 0
    player_mask = mask & time_mask
    index = acting.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, decoded_by_player.shape[-1])
    acting_decoded = decoded_by_player.gather(2, index).squeeze(2)
    channel_index = acting.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, obs_channels.shape[-1])
    acting_channels = obs_channels.gather(2, channel_index).squeeze(2)
    player_state = model._encode_state_features(acting_decoded, acting_channels, player_mask)
    action_feats, avail_mask = model._encode_action_features(avail_tokens, avail_weights, player_mask)
    return player_state, action_feats, avail_mask


# ---------------------------------------------------------------------------
# 3. Sparse gather/scatter
# ---------------------------------------------------------------------------


def test_sparse_fuse_runs_the_mlp_only_on_legal_candidates():
    torch.manual_seed(17)
    model = deterministic_agent(tiny_config(FUSE_ARCH), seed = 21).eval()
    legal_per_time = [[0, 1], [2], [0, 1, 2, 3]]
    rollout = tiny_rollout(batch = 2, steps = 3, legal_per_time = legal_per_time)
    weights = rollout["avail_weights"]
    time_mask = torch.arange(3).unsqueeze(0) < rollout["num_steps"].unsqueeze(1)
    avail_mask = (weights != 0) & time_mask.unsqueeze(-1)
    # The fixture really contains a partially legal row and a fully masked row:
    # row 0 has (2, 1, 4) legal candidates, row 1 has (2, 1, 0).
    assert int(avail_mask.sum()) == 2 + 1 + 4 + 2 + 1 + 0 == 10
    assert bool((avail_mask.sum(dim = -1) == 0).any())
    assert bool((avail_mask.sum(dim = -1) == 1).any())

    seen: list[tuple[int, ...]] = []
    handle = model.actor_fuse.input_proj.register_forward_hook(
        lambda _m, inputs, _o: seen.append(tuple(inputs[0].shape)),
    )
    try:
        with torch.no_grad():
            model(rollout = rollout)
    finally:
        handle.remove()
    assert len(seen) == 1
    assert seen[0][0] == int(avail_mask.sum())


def test_fuse_delta_matches_dense_reference_and_is_zero_off_the_legal_set():
    torch.manual_seed(23)
    model = deterministic_agent(tiny_config(FUSE_ARCH), seed = 31)
    with torch.no_grad():
        model.actor_fuse.output_proj.weight.normal_(mean = 0.0, std = 0.2)

    B, T, A, D = 2, 3, 4, model.feature_dim
    player_state = torch.randn(B, T, D, generator = torch.Generator().manual_seed(5))
    action_feats = torch.randn(B, T, A, D, generator = torch.Generator().manual_seed(6))
    avail_mask = torch.zeros(B, T, A, dtype = torch.bool)
    avail_mask[:, :, 0] = True
    avail_mask[0, 1, 2] = True
    # A whole row (row 1, time 2) is illegal.
    avail_mask[1, 2, :] = False

    fused = player_state.unsqueeze(-2) * action_feats
    out = model._apply_actor_fuse(fused, player_state, action_feats, avail_mask)
    delta = out - fused

    reference = dense_fuse_reference(model.actor_fuse, player_state, action_feats)
    # Same values, but a different gather layout/GEMV batch size, so BLAS blocking
    # may reorder the accumulation: compare with a tight tolerance, not bitwise.
    assert torch.allclose(delta[avail_mask], reference[avail_mask], rtol = 1e-5, atol = 1e-6)
    assert float((delta[avail_mask] - reference[avail_mask]).detach().abs().max()) < 1e-6
    assert float(reference[avail_mask].detach().abs().max()) > 0.0
    # Illegal candidates are never touched: an exactly-zero residual, not a
    # masked-out non-zero value.
    assert torch.equal(delta[~avail_mask], torch.zeros_like(delta[~avail_mask]))
    assert int(torch.count_nonzero(delta[~avail_mask])) == 0


# ---------------------------------------------------------------------------
# 4. Gradient gating at the three documented time points
# ---------------------------------------------------------------------------


def _fuse_trunk_parameters(model: Agent):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(FUSE_KEY_PREFIX) and name != f"{FUSE_KEY_PREFIX}output_proj.weight"
    }


def test_gradient_gating_first_backward_first_step_second_backward():
    torch.manual_seed(41)
    model = deterministic_agent(tiny_config(FUSE_ARCH), seed = 47)
    model.train()
    rollout = tiny_rollout(batch = 2, steps = 3)
    weights = torch.linspace(0.1, 0.9, 4)
    optimizer = model.configure_optimizer(lr = 1e-2, optimizer = OptimizerConfig(use_muon = False))

    def forward_backward():
        model.zero_grad(set_to_none = True)
        out = model(rollout = rollout)
        loss = (out.avail_probs * weights).sum() + out.player_state_value.square().sum()
        loss.backward()
        return out, loss

    gate = model.actor_fuse.output_proj.weight
    trunk_params = _fuse_trunk_parameters(model)
    assert set(trunk_params) == {
        "actor_fuse.input_proj.weight", "actor_fuse.input_proj.bias",
        "actor_fuse.trunk.layers.0.norm.weight",
        "actor_fuse.trunk.layers.0.mlp.gate_proj.weight",
        "actor_fuse.trunk.layers.0.mlp.up_proj.weight",
        "actor_fuse.trunk.layers.0.mlp.down_proj.weight",
        "actor_fuse.trunk.layers.1.norm.weight",
        "actor_fuse.trunk.layers.1.mlp.gate_proj.weight",
        "actor_fuse.trunk.layers.1.mlp.up_proj.weight",
        "actor_fuse.trunk.layers.1.mlp.down_proj.weight",
        "actor_fuse.trunk.output_norm.weight",
    }

    # --- time point 1: the first backward -------------------------------------
    forward_backward()
    assert gate.grad is not None
    first_gate_grad = float(gate.grad.abs().sum())
    assert first_gate_grad > 0.0
    for name, parameter in trunk_params.items():
        grad = parameter.grad
        assert grad is None or float(grad.abs().sum()) == 0.0, name
    # The legacy heads still learn at the first step.
    assert float(model.logit_head.weight.grad.abs().sum()) > 0.0

    # --- time point 2: after the first effective optimizer step ----------------
    optimizer.step()
    assert int(torch.count_nonzero(gate.detach())) > 0
    assert float(gate.detach().abs().sum()) > 0.0

    # --- time point 3: the second backward ------------------------------------
    forward_backward()
    assert float(gate.grad.abs().sum()) > 0.0
    for name, parameter in trunk_params.items():
        assert parameter.grad is not None, name
        assert float(parameter.grad.abs().sum()) > 0.0, name


def test_naive_zeroed_down_projection_is_not_a_zero_residual():
    """The documented trap: zeroing the trunk's last down_proj is *not* identity.

    A naive "zero residual" implementation would zero the final SwiGLU/down
    projection of the residual trunk and use the trunk output directly as the
    residual.  The residual identity path (``x = x + MLP(norm(x))`` plus the
    trailing ``output_norm``) keeps the trunk output non-zero, so the residual
    would *not* be zero.  Only an independent zero-initialised output projection
    can guarantee ``W_out h == 0`` for arbitrary ``h``.
    """
    torch.manual_seed(53)
    trunk = ResGatedMLP(24, n_layers = 2, expansion_ratio = 1)
    with torch.no_grad():
        trunk.layers[-1]["mlp"].down_proj.weight.zero_()
    x = torch.randn(5, 24)
    naive_residual = trunk(x)
    assert not torch.equal(naive_residual, torch.zeros_like(naive_residual))
    assert float(naive_residual.detach().abs().max()) > 0.0
    # ... and it is not the identity either, so it cannot be a zero residual.
    assert not torch.equal(naive_residual, trunk.output_norm(x))
    assert float((naive_residual - trunk.output_norm(x)).detach().abs().max()) > 0.0


# ---------------------------------------------------------------------------
# 5. Optimizer grouping
# ---------------------------------------------------------------------------


def test_optimizer_places_every_fuse_parameter_and_exempts_output_proj():
    torch.manual_seed(61)
    model = deterministic_agent(tiny_config(FUSE_ARCH), seed = 67)
    optimizer = model.configure_optimizer(lr = 1e-3, optimizer = OptimizerConfig(use_muon = False))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer_param_ids(optimizer) == {id(p) for p in model.parameters()}

    name, wd = group_of(optimizer, model.actor_fuse.output_proj.weight)
    assert (name, wd) == ("adamw", 0.0)
    assert model.actor_fuse.output_proj.weight in {
        p for group in optimizer.param_groups if group["weight_decay"] == 0.0 for p in group["params"]
    }

    expected_wd = {
        "actor_fuse.input_proj.weight",
        "actor_fuse.trunk.layers.0.mlp.gate_proj.weight",
        "actor_fuse.trunk.layers.0.mlp.up_proj.weight",
        "actor_fuse.trunk.layers.0.mlp.down_proj.weight",
        "actor_fuse.trunk.layers.1.mlp.gate_proj.weight",
        "actor_fuse.trunk.layers.1.mlp.up_proj.weight",
        "actor_fuse.trunk.layers.1.mlp.down_proj.weight",
    }
    expected_no_wd = {
        "actor_fuse.input_proj.bias",
        "actor_fuse.trunk.layers.0.norm.weight",
        "actor_fuse.trunk.layers.1.norm.weight",
        "actor_fuse.trunk.output_norm.weight",
        "actor_fuse.output_proj.weight",
    }
    names = dict(model.named_parameters())
    for key in expected_wd:
        assert key in model.weight_decay_param_names, key
        assert group_of(optimizer, names[key])[1] == model.cfg.optimizer.weight_decay
    for key in expected_no_wd:
        assert key not in model.weight_decay_param_names, key
        assert group_of(optimizer, names[key])[1] == 0.0
    assert set(expected_wd) | set(expected_no_wd) == set(fuse_parameter_names(model))


def test_optimizer_muon_path_keeps_the_fuse_grouped_and_covered():
    torch.manual_seed(71)
    model = deterministic_agent(tiny_config(FUSE_ARCH), seed = 73)
    optimizer = model.configure_optimizer(
        lr = 1e-2, optimizer = OptimizerConfig(use_muon = True, weight_decay = 0.01),
    )
    assert isinstance(optimizer, MultiOptimizer)
    assert set(optimizer.optimizers) == {"muon", "adamw"}
    assert optimizer_param_ids(optimizer) == {id(p) for p in model.parameters()}

    names = dict(model.named_parameters())
    # Large fuse matrices go to Muon (weight decay); the zero gate and the norms
    # / biases go to AdamW with weight decay disabled.
    for key in (
        "actor_fuse.input_proj.weight",
        "actor_fuse.trunk.layers.0.mlp.gate_proj.weight",
        "actor_fuse.trunk.layers.0.mlp.down_proj.weight",
        "actor_fuse.trunk.layers.1.mlp.up_proj.weight",
    ):
        assert group_of(optimizer, names[key]) == ("muon", 0.01), key
    for key in (
        "actor_fuse.output_proj.weight",
        "actor_fuse.input_proj.bias",
        "actor_fuse.trunk.output_norm.weight",
        "actor_fuse.trunk.layers.1.norm.weight",
    ):
        name, wd = group_of(optimizer, names[key])
        assert (name, wd) == ("adamw", 0.0), key

    # A real step through MultiOptimizer must not raise and must move the gate.
    model.train()
    rollout = tiny_rollout(batch = 1, steps = 2)
    model.zero_grad(set_to_none = True)
    out = model(rollout = rollout)
    loss = (out.avail_probs * torch.linspace(0.2, 0.8, 4)).sum() + out.player_state_value.square().sum()
    loss.backward()
    gate_before = model.actor_fuse.output_proj.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(model.actor_fuse.output_proj.weight.detach(), gate_before)
    assert math.isfinite(float(model.actor_fuse.output_proj.weight.detach().abs().sum()))


# ---------------------------------------------------------------------------
# 6. Trainer-level smoke: the fuse really trains through the macro learner
# ---------------------------------------------------------------------------


def test_v14_macro_trainer_runs_and_moves_the_fuse_gate(tmp_path):
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name = "fuse-smoke")
    from guandan_marvel.training.local import IPPOTrainer

    trainer = IPPOTrainer(cfg, game = game, run_dir = tmp_path / "run")
    try:
        assert trainer.agent_resume.actor_fuse is not None
        assert torch.count_nonzero(trainer.agent_resume.actor_fuse.output_proj.weight) == 0
        metrics = trainer.train_iteration()
        assert math.isfinite(float(metrics["agent/grad_norm"]))
        assert float(metrics["agent/grad_norm"]) > 0.0
        assert math.isfinite(float(metrics["macro_initial/pg_loss"]))
        assert int(torch.count_nonzero(trainer.agent_resume.actor_fuse.output_proj.weight)) > 0
    finally:
        trainer.close()


def test_v13_macro_trainer_has_no_fuse_branch(tmp_path):
    cfg, game = smoke_trainer_config(LEGACY_ARCH, run_name = "legacy-smoke")
    from guandan_marvel.training.local import IPPOTrainer

    trainer = IPPOTrainer(cfg, game = game, run_dir = tmp_path / "run")
    try:
        assert trainer.agent_resume.actor_fuse is None
        assert not any(
            name.startswith(FUSE_KEY_PREFIX) for name, _ in trainer.agent_resume.named_parameters()
        )
        metrics = trainer.train_iteration()
        assert math.isfinite(float(metrics["agent/grad_norm"]))
    finally:
        trainer.close()


def test_v14_vrpo_trainer_builds_the_fuse_and_runs_one_iteration(tmp_path):
    """The other algorithm keeps working on the fused architecture."""
    from guandan_marvel.config import LEGACY_TRAINING_SEMANTICS_PROFILE
    from guandan_marvel.training.local import VRPOTrainer

    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name = "vrpo-smoke")
    cfg.algo_name = "vrpo"
    cfg.ppo.training_semantics_profile = LEGACY_TRAINING_SEMANTICS_PROFILE
    trainer = VRPOTrainer(cfg, game = game, run_dir = tmp_path / "run")
    try:
        assert trainer.agent_resume.actor_fuse is not None
        assert trainer.agent_resume.value_head is None
        assert torch.count_nonzero(trainer.agent_resume.actor_fuse.output_proj.weight) == 0
        metrics = trainer.train_iteration()
        assert math.isfinite(float(metrics["agent/grad_norm"]))
        assert math.isfinite(float(metrics["critic/grad_norm"]))
        assert int(torch.count_nonzero(trainer.agent_resume.actor_fuse.output_proj.weight)) > 0
    finally:
        trainer.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
