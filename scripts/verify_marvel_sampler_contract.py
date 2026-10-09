"""Generate machine-readable sampler probability and failure-state evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from guandan.action_state import COMMIT_TOKEN, TokenCodec
from guandan.combos import CombinationKind
from guandan_marvel import python_game as game
from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig
from guandan_marvel.model import Agent
from guandan_marvel.model.types import AgentOutput
from guandan_marvel.runtime.sampler import Sampler, SamplerConfig, SamplerFailedStateError
from guandan_marvel.training.losses import compute_clipped_pg_loss


class DecoderConfig:
    num_key_value_heads = 1
    num_attention_heads = 1
    hidden_size = 4


class Decoder:
    config = DecoderConfig()


class NonOneHotAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.decoder = Decoder()
        self.decoder_n_layers = 1

    def forward(self, *, rollout, **_kwargs):
        weights = rollout["avail_weights"]
        legal = weights > 0
        scores = torch.arange(1, weights.shape[-1] + 1, device=weights.device, dtype=torch.float32).view(1, 1, -1).expand_as(weights.float())
        probs = scores * legal
        return AgentOutput(probs / probs.sum(dim=-1, keepdim=True))


def game_type(horizon: int, env_horizon: int):
    class Game:
        identifier = f"sampler-contract-{horizon}-{env_horizon}"
        num_players = game.num_players
        num_tokens = game.num_tokens
        pad_token = game.pad_token
        num_channels = game.num_channels
        max_game_horizon = horizon
        max_observation_tokens = game.max_observation_tokens
        max_action_tokens = game.max_action_tokens
        max_available_actions = game.max_available_actions
        protocol_version = game.protocol_version
        actions_from_np = staticmethod(game.actions_from_np)

        @staticmethod
        def EnvBatch(batch_size: int):
            return game.EnvBatch(batch_size, max_horizon=env_horizon, auto_reset=False)
    return Game


def make_sampler(Game, *, purpose: str, temperature: float, collect: int = 0, agent=None):
    sampler = Sampler(config=SamplerConfig(
        game=Game, device="cpu", rollout_batch_size=1,
        collect_batch_size=collect, sampling_temperature=temperature,
        purpose=purpose, env_seed=0,
    ))
    model = agent or NonOneHotAgent()
    sampler.prepare_agent_cache(agent_model=model)
    return sampler, model


def index_for_token(sampler, token: int) -> int:
    rows = sampler.parsed_envs_step["avail_tokens"][0]
    weights = sampler.parsed_envs_step["avail_weights"][0]
    return int((((rows[:, 5] == token) & (weights > 0)).nonzero().flatten())[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Game = game_type(16, 16)
    agent = NonOneHotAgent()

    on_policy, _ = make_sampler(Game, purpose="on_policy", temperature=1.0, agent=agent)
    on_policy.rollout_step(agent_model=agent)
    taken = int(on_policy.rollout_buffer["action_taken"][0, 0])
    raw = float(on_policy.rollout_buffer["policy_probs"][0, 0, taken])
    behavior = float(on_policy.rollout_buffer["behavior_action_prob"][0, 0])

    rejected_temperatures = {}
    for temperature in (0.0, 0.5, 2.0):
        try:
            SamplerConfig(game=Game, device="cpu", rollout_batch_size=1, collect_batch_size=0, sampling_temperature=temperature, purpose="on_policy")
            rejected_temperatures[str(temperature)] = False
        except ValueError:
            rejected_temperatures[str(temperature)] = True

    evaluation = {}
    for temperature in (0.0, 0.5, 2.0):
        torch.manual_seed(17)
        sampler, _ = make_sampler(Game, purpose="evaluation", temperature=temperature, agent=agent)
        sampler.rollout_step(agent_model=agent)
        index = int(sampler.rollout_buffer["action_taken"][0, 0])
        evaluation[str(temperature)] = {
            "action": index,
            "policy_probability": float(sampler.rollout_buffer["policy_probs"][0, 0, index]),
            "behavior_action_probability": float(sampler.rollout_buffer["behavior_action_prob"][0, 0]),
            "on_policy_valid": bool(sampler.rollout_buffer["on_policy_valid"][0, 0]),
        }

    forced, _ = make_sampler(Game, purpose="test", temperature=1.0, agent=agent)
    forced.rollout_step(agent_model=agent, forced_action_indices=torch.tensor([0]))
    forced_rollout = forced.rollout_buffer[:1]
    training_rejected = False
    try:
        compute_clipped_pg_loss(
            forced_rollout,
            clip_coef=0.2,
            avail_probs_new=forced_rollout["policy_probs"].clone(),
            player_advantage=torch.ones_like(forced_rollout["behavior_action_prob"]),
        )
    except ValueError:
        training_rejected = True

    FailureGame = game_type(16, 3)
    cfg = TrainConfig(
        game=GameSpec.from_game(FailureGame), device="cpu",
        model=ModelConfig(architecture="v1.3", history_dim=8, feature_dim=8, decoder_layers=1, decoder_heads=1, state_layers=1, action_layers=1, fuse_layers=1),
    )
    real_agent = Agent(cfg)
    failed, _ = make_sampler(FailureGame, purpose="test", temperature=1.0, agent=real_agent)
    failed.rollout_step(agent_model=real_agent, forced_action_indices=torch.tensor([index_for_token(failed, TokenCodec.family(CombinationKind.SINGLE))]))
    rows = failed.parsed_envs_step["avail_tokens"][0]
    weights = failed.parsed_envs_step["avail_weights"][0]
    card_index = int(((((rows[:, 3] - game.ACTION_KIND_BASE) == 6) & (weights > 0)).nonzero().flatten())[0])
    failed.rollout_step(agent_model=real_agent, forced_action_indices=torch.tensor([card_index]))
    before = {
        "env_steps": failed.total_env_steps,
        "game_steps": failed.total_game_steps,
        "buffer_steps": int(failed.rollout_buffer["num_steps"][0]),
        "cache_seq_lens": failed.agent_kv._seq_lens.tolist(),
        "environment": failed.envs.serialize().decode("utf-8"),
    }
    error_type = None
    try:
        failed.rollout_step(agent_model=real_agent, forced_action_indices=torch.tensor([index_for_token(failed, COMMIT_TOKEN)]))
    except Exception as error:
        error_type = type(error).__name__
    after = {
        "env_steps": failed.total_env_steps,
        "game_steps": failed.total_game_steps,
        "buffer_steps": int(failed.rollout_buffer["num_steps"][0]),
        "cache_seq_lens": failed.agent_kv._seq_lens.tolist(),
        "environment_unchanged": failed.envs.serialize().decode("utf-8") == before["environment"],
        "failed": failed.failed,
        "failure_info": failed.failure_info,
    }
    retry_rejected = False
    try:
        failed.rollout_step(agent_model=real_agent)
    except SamplerFailedStateError:
        retry_rejected = True
    del before["environment"]

    report = {
        "schema": "guandan-marvel-sampler-contract-verification-v2",
        "stage": "M02_guandan_adapter_v2_repair",
        "on_policy_temperature_one": {
            "selected_policy_probability": raw,
            "selected_behavior_probability": behavior,
            "match": abs(raw - behavior) < 1e-7,
            "on_policy_valid": bool(on_policy.rollout_buffer["on_policy_valid"][0, 0]),
        },
        "on_policy_nonunit_temperature_rejected": rejected_temperatures,
        "evaluation_modes": evaluation,
        "forced_test_mode": {
            "behavior_action_probability": float(forced_rollout["behavior_action_prob"][0, 0]),
            "on_policy_valid": bool(forced_rollout["on_policy_valid"][0, 0]),
            "ppo_entry_rejected": training_rejected,
        },
        "failure_transaction": {
            "exception": error_type,
            "before": before,
            "after": after,
            "completed_counters_unchanged": before["env_steps"] == after["env_steps"] and before["game_steps"] == after["game_steps"] and before["buffer_steps"] == after["buffer_steps"],
            "cache_advanced_before_failure": after["cache_seq_lens"] != before["cache_seq_lens"],
            "subsequent_retry_rejected": retry_rejected,
            "strategy": "poison_on_failure",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "on_policy_match": report["on_policy_temperature_one"]["match"],
        "nonunit_rejected": all(rejected_temperatures.values()),
        "ppo_entry_rejected": training_rejected,
        "failure_counters_unchanged": report["failure_transaction"]["completed_counters_unchanged"],
        "retry_rejected": retry_rejected,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
