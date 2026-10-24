# SPDX-License-Identifier: MIT
"""Observation-only agents for the C++ V2 evaluation host."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from guandan.evaluation_agents import SnapshotAgent
from guandan_marvel.model import KVCache, TensorDict

from .checkpoints import LegacyV1OpponentArtifact, V2PolicyArtifact

ACTION_TOKEN_COLUMN = 5
ACTION_COMMIT_COLUMN = 6
PASS_TOKEN = 6
COMMIT_TOKEN = 7
FAMILY_PRIORITY = (14, 13, 11, 12, 10, 9, 8, 15, 16, 17)


def legal_action_indices(step) -> np.ndarray:
    weights = np.asarray(step.available_actions.weights[0])
    indices = np.flatnonzero(weights > 0)
    if indices.size == 0:
        raise RuntimeError("evaluation host received no legal V2 action")
    return indices


def action_index_for_v1_token(step, token_id: int) -> int:
    legal = legal_action_indices(step)
    rows = np.asarray(step.available_actions.tokens[0])
    matches = [int(index) for index in legal if int(rows[index, ACTION_TOKEN_COLUMN]) == int(token_id)]
    if len(matches) != 1:
        raise RuntimeError(f"legacy token {token_id} does not map to exactly one legal V2 action row")
    return matches[0]


class V2PolicySession:
    """Fresh KV/RNG state for one frozen V2 policy in one game."""

    def __init__(self, artifact: V2PolicyArtifact, *, seed: int, temperature: float, device: str | torch.device = "cpu"):
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        self.artifact = artifact
        self.model = artifact.model
        self.device = torch.device(device)
        self.temperature = float(temperature)
        decoder_config = self.model.decoder.config
        kv_heads = decoder_config.num_key_value_heads or decoder_config.num_attention_heads
        head_dim = decoder_config.hidden_size // decoder_config.num_attention_heads
        dtype = next(self.model.parameters()).dtype
        self.cache = KVCache(
            self.model.decoder_n_layers,
            4,
            max_ctx_len=int(self.model.cfg.game.max_game_horizon),
            num_heads=kv_heads,
            head_dim=head_dim,
            value_heads=kv_heads,
            value_head_dim=head_dim,
            dtype=dtype,
            device=self.device,
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.steps = 0

    @torch.inference_mode()
    def observe_and_score(self, step) -> torch.Tensor:
        return self.observe_score_and_value(step)[0]

    @torch.inference_mode()
    def observe_score_and_value(self, step) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return policy probabilities and the acting player's value head output.

        The value is the IPPO state value for the acting player at this state;
        it is exactly what the phase-6 value qualification target is compared
        against (same player, same state, same checkpoint).
        """
        rollout = TensorDict({
            "num_steps": torch.ones(1, dtype=torch.long, device=self.device),
            "player_ids": torch.as_tensor(step.player_ids, dtype=torch.long, device=self.device).unsqueeze(1),
            "obs_tokens": torch.as_tensor(step.observations.tokens, dtype=torch.long, device=self.device).unsqueeze(1),
            "obs_channels": torch.as_tensor(step.observations.channels, dtype=torch.uint8, device=self.device).unsqueeze(1),
            "avail_tokens": torch.as_tensor(step.available_actions.tokens, dtype=torch.long, device=self.device).unsqueeze(1),
            "avail_weights": torch.as_tensor(step.available_actions.weights, dtype=torch.float32, device=self.device).unsqueeze(1),
        }, batch_size=[1], device=self.device)
        output = self.model(rollout=rollout, use_cache=True, past_key_values=self.cache)
        probabilities = output.avail_probs[0, 0].float()
        legal = torch.as_tensor(step.available_actions.weights[0] > 0, dtype=torch.bool, device=self.device)
        if probabilities.shape != legal.shape or not bool(torch.isfinite(probabilities).all()):
            raise FloatingPointError("V2 policy returned invalid probabilities")
        if not bool((probabilities[~legal] == 0).all()):
            raise RuntimeError("V2 policy assigned probability to an illegal action")
        value = None
        if output.player_state_value is not None:
            value = output.player_state_value[0, 0].float()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("V2 policy returned a non-finite state value")
        self.steps += 1
        return probabilities, value

    def select_index(self, probabilities: torch.Tensor, step) -> int:
        weights = torch.as_tensor(step.available_actions.weights[0], dtype=torch.float32, device=self.device)
        legal = weights > 0
        log_prior = weights.clamp(min=1e-9).log()
        raw_logits = probabilities.clamp(min=1e-9).log() - log_prior
        neg_inf = torch.finfo(raw_logits.dtype).min
        if self.temperature == 0.0:
            index = int(torch.argmax(raw_logits.masked_fill(~legal, neg_inf)).item())
        else:
            logits = raw_logits / self.temperature + log_prior
            distribution = torch.softmax(logits.masked_fill(~legal, neg_inf), dim=-1)
            index = int(torch.multinomial(distribution, 1, generator=self.generator).item())
        if not bool(legal[index]):
            raise RuntimeError("V2 policy selected an illegal action; no fallback")
        return index


class RandomV2Agent:
    def __init__(self, seed: int):
        self.generator = np.random.default_rng(int(seed))
        self.metadata = {"kind": "random", "version": "v2-row-uniform-v1", "seed": int(seed)}

    def select_index(self, step) -> int:
        return int(self.generator.choice(legal_action_indices(step)))


class RuleV2Agent:
    """Deterministic legal-row heuristic using only actor-visible V2 inputs."""

    metadata = {"kind": "rule", "version": "v2-public-row-heuristic-v1"}

    def select_index(self, step) -> int:
        legal = legal_action_indices(step)
        rows = np.asarray(step.available_actions.tokens[0])
        tokens = {int(rows[index, ACTION_TOKEN_COLUMN]): int(index) for index in legal}
        if COMMIT_TOKEN in tokens:
            return tokens[COMMIT_TOKEN]
        for family in FAMILY_PRIORITY:
            if family in tokens:
                return tokens[family]
        non_pass = [(token, index) for token, index in tokens.items() if token != PASS_TOKEN]
        return min(non_pass or list(tokens.items()))[1]


@dataclass
class LegacyV1OpponentSession:
    artifact: LegacyV1OpponentArtifact
    seed: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.agent = SnapshotAgent(self.artifact.path, seed=int(self.seed), device=self.device)

    def select_index(self, step, observation) -> int:
        token = self.agent.select_token(observation)
        return action_index_for_v1_token(step, token)


__all__ = [
    "LegacyV1OpponentSession", "RandomV2Agent", "RuleV2Agent", "V2PolicySession",
    "action_index_for_v1_token", "legal_action_indices",
]
