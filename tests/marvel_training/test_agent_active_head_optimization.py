from __future__ import annotations

import copy

import torch

from guandan_marvel import python_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.model import Agent, TensorDict


def _config() -> TrainConfig:
    cfg = TrainConfig(
        game=python_game.GAME_SPEC, device="cpu",
        rollout_batch_size=2, collect_batch_size=1,
        max_sampler_bytes_per_rank=512 * 2**20,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.optimizer.use_muon = False
    return cfg


def _rollout() -> TensorDict:
    torch.manual_seed(123)
    B, T, P, H, C, A, L = 2, 3, 4, 48, 512, 256, 8
    num_steps = torch.tensor([3, 2], dtype=torch.long)
    player_ids = torch.tensor([[0, 1, 2], [3, 0, -1]], dtype=torch.long)
    obs_tokens = torch.randint(1, 309, (B, T, P, H), dtype=torch.long)
    obs_channels = torch.randint(0, 2, (B, T, P, C), dtype=torch.uint8)
    avail_tokens = torch.randint(1, 309, (B, T, A, L), dtype=torch.long)
    avail_weights = torch.zeros(B, T, A)
    avail_weights[..., :7] = torch.rand(B, T, 7) + 0.1
    return TensorDict({
        "num_steps": num_steps, "player_ids": player_ids,
        "obs_tokens": obs_tokens, "obs_channels": obs_channels,
        "avail_tokens": avail_tokens, "avail_weights": avail_weights,
    }, batch_size=[B])


def _legacy_heads(agent: Agent, rollout: TensorDict):
    agent._validate_rollout(rollout)
    num_steps, player_ids = rollout["num_steps"], rollout["player_ids"]
    obs_tokens, obs_channels = rollout["obs_tokens"], rollout["obs_channels"]
    avail_tokens, avail_weights = rollout["avail_tokens"], rollout["avail_weights"]
    device = obs_tokens.device
    B, T, P, _ = obs_tokens.shape
    _, _, A, L = avail_tokens.shape
    time_mask = agent._build_time_mask(num_steps, T, device=device)
    flat_time_mask = time_mask.repeat_interleave(P, dim=0)
    flat_player_ids = torch.arange(P, device=device).expand(B, P).reshape(-1, 1).expand(-1, T)
    flat_obs_tokens = agent._flatten_players(obs_tokens, name="obs_tokens")
    player_range = agent.player_index.to(device=device)
    active_mask = (player_ids.unsqueeze(-1) == player_range) & (player_ids.unsqueeze(-1) >= 0)
    flat_active_mask = active_mask.permute(0, 2, 1).reshape(B * P, T).bool()
    flat_obs_channels = agent._flatten_players(obs_channels, name="obs_channels")
    flat_avail_tokens = avail_tokens.repeat_interleave(P, dim=0).reshape(B * P, T, A, L)
    flat_avail_weights = avail_weights.repeat_interleave(P, dim=0).reshape(B * P, T, A) * flat_time_mask.unsqueeze(-1)
    decoded, _ = agent._encode_history(flat_obs_tokens, flat_player_ids, flat_time_mask)
    fused_state = agent._encode_state_features(decoded, flat_obs_channels, flat_time_mask)
    player_state = agent._apply_mask(fused_state, flat_active_mask)
    action_feats, avail_mask = agent._encode_action_features(flat_avail_tokens, flat_avail_weights, flat_time_mask)
    avail_mask = avail_mask & flat_active_mask.unsqueeze(-1)
    logits = agent.logit_head(player_state.unsqueeze(-2) * action_feats).squeeze(-1)
    logits = logits.masked_fill(~avail_mask, torch.finfo(logits.dtype).min)
    logits = logits + avail_mask.to(logits.dtype) * flat_avail_weights.clamp(min=1e-9).log()
    flat_probs = torch.softmax(logits.float(), dim=-1)
    flat_probs = agent._apply_mask(flat_probs, avail_mask).to(logits.dtype)
    full = agent._unflatten_players(flat_probs, batch_size=B, name="flat_probs")
    acting = player_ids.clamp(min=0);mask=player_ids>=0;player_mask=mask & time_mask
    idx = acting.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, A)
    agent_probs = full.gather(2, idx).squeeze(2)
    sums=avail_weights.sum(-1,keepdim=True);uniform=torch.full_like(avail_weights,1.0/A)
    env_probs=torch.where(sums>0,avail_weights/sums.clamp(min=1e-9),uniform)
    probs=torch.where(mask.unsqueeze(-1),agent_probs,env_probs)
    state_values=player_value=None
    if agent.value_head is not None:
        values=agent.value_head(player_state).squeeze(-1)
        values=agent._apply_mask(values,flat_active_mask)
        state_values=agent._unflatten_players(values,batch_size=B,name="values") * time_mask.unsqueeze(-1)
        player_value=state_values.gather(2,acting.unsqueeze(-1)).squeeze(-1)*player_mask
    return probs,state_values,player_value


def test_active_only_heads_match_legacy_probabilities_and_values():
    torch.manual_seed(456)
    agent = Agent(_config(), enable_value_head=True).eval()
    rollout = _rollout()
    expected = _legacy_heads(agent, rollout)
    actual = agent(rollout=rollout)
    torch.testing.assert_close(actual.avail_probs, expected[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.state_values, expected[1], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.player_state_value, expected[2], rtol=1e-6, atol=1e-7)


def test_active_only_heads_match_legacy_loss_and_gradients():
    torch.manual_seed(789)
    optimized = Agent(_config(), enable_value_head=True).train()
    legacy = copy.deepcopy(optimized).train()
    rollout = _rollout()
    weights = torch.linspace(0.1, 1.0, rollout["avail_weights"].shape[-1])
    out = optimized(rollout=rollout)
    optimized_loss = (out.avail_probs * weights).sum() + out.player_state_value.square().sum()
    optimized_loss.backward()
    probs, _states, values = _legacy_heads(legacy, rollout)
    legacy_loss = (probs * weights).sum() + values.square().sum()
    legacy_loss.backward()
    torch.testing.assert_close(optimized_loss, legacy_loss, rtol=1e-5, atol=1e-6)
    for (name_a, param_a), (name_b, param_b) in zip(optimized.named_parameters(), legacy.named_parameters(), strict=True):
        assert name_a == name_b
        if param_a.grad is None or param_b.grad is None:
            assert param_a.grad is None and param_b.grad is None
        else:
            torch.testing.assert_close(param_a.grad, param_b.grad, rtol=1e-5, atol=1e-6)
