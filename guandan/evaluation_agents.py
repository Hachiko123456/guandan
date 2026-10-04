"""A06 observation-only token agents; snapshot means real frozen model weights."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

from .action_state import COMMIT_TOKEN, PASS_TOKEN, TokenCodec
from .cards import Card, Rank, effective_rank_value
from .environment_types import Observation
from .model import PolicyValueNet
from .training.checkpoint import CheckpointError, load_checkpoint
from .training.runtime import file_sha256


def legal_tokens(observation: Observation) -> np.ndarray:
    if not isinstance(observation, Observation):
        raise TypeError('agents only accept the policy-facing Observation')
    mask = observation.legal_next_mask
    if mask.dtype != np.bool_ or mask.shape != observation.legal_next_tokens.shape:
        raise ValueError('malformed legal token mask')
    tokens = observation.legal_next_tokens[mask]
    if not len(tokens) or len(set(tokens.tolist())) != len(tokens) or np.any((tokens <= 0) | (tokens >= 256)):
        raise ValueError('missing/duplicate/invalid legal token IDs')
    return tokens


class RandomAgent:
    def __init__(self, seed: int):
        self.generator = np.random.default_rng(seed)
        self.metadata = {'kind': 'random', 'version': 'token-uniform-0.1', 'seed': int(seed)}

    def select_token(self, observation: Observation) -> int:
        return int(self.generator.choice(legal_tokens(observation)))


class RuleAgent:
    """Deterministic self/public heuristic, not an oracle or a strength claim.

    Prefer shedding larger ordinary families, delay bombs, and follow when
    possible. Never removes tokens from the environment's complete action set.
    """
    metadata = {'kind': 'rule', 'version': 'public-token-heuristic-0.1'}
    family_priority = (14, 13, 11, 12, 10, 9, 8, 15, 16, 17)

    def select_token(self, observation: Observation) -> int:
        tokens = legal_tokens(observation).tolist()
        if COMMIT_TOKEN in tokens:
            return COMMIT_TOKEN
        for family in self.family_priority:
            if family in tokens:
                return family
        cards = [token for token in tokens if 64 <= token < 172]
        if cards:
            level = Rank(int(observation.state_channels[3]))
            return min(cards, key=lambda token: (effective_rank_value(Card(token-64).rank, level), token))
        ranks = [token for token in tokens if 18 <= token < 33]
        if ranks:
            level = Rank(int(observation.state_channels[3]))
            return min(ranks, key=lambda token: effective_rank_value(TokenCodec.rank_from_token(token), level))
        # Only pass when no playable branch remains.
        return min([token for token in tokens if token != PASS_TOKEN] or tokens)


class SnapshotAgent:
    """An actual A05 actor checkpoint; missing/incompatible weights fail closed."""
    def __init__(self, checkpoint: str | Path, *, seed: int = 0, device: str = 'cpu', deterministic: bool = False):
        self.path = Path(checkpoint).resolve()
        loaded = load_checkpoint(self.path, map_location='cpu')
        config = loaded.model_config
        if config.get('architecture') != 'mean-pooled-token-MLP':
            raise CheckpointError('unsupported actor architecture')
        hidden = config.get('hidden_dim')
        if type(hidden) is not int or hidden <= 0:
            raise CheckpointError('snapshot hidden_dim missing/invalid')
        if config.get('vocabulary') != 256 or config.get('channels') != 256:
            raise CheckpointError('snapshot observation contract differs')
        weights = {key[len('actor.'):]: value for key,value in loaded.model_state.items() if key.startswith('actor.')}
        if not weights:
            raise CheckpointError('snapshot has no actor state; no fallback')
        self.device = torch.device(device)
        # Construction must not reset or advance the caller's torch RNG.
        with torch.random.fork_rng(devices=[]):
            self.model = PolicyValueNet(hidden_dim=hidden)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.deterministic = deterministic
        self.training_deals = deepcopy(loaded.training_state.get('engine', {}).get('collector', {}).get('deal_records', []))
        self.metadata = {
            'kind': 'snapshot', 'checkpoint': str(self.path), 'sha256': file_sha256(self.path),
            'algorithm': loaded.metadata.algorithm, 'update': loaded.metadata.update_count,
            'model_version': loaded.metadata.model_version, 'training_version': loaded.metadata.training_version,
            'protocol_versions': loaded.metadata.protocol_versions,
            'model_config': deepcopy(config), 'seed': seed, 'deterministic': deterministic,
            'device': str(self.device),
        }

    @torch.no_grad()
    def logits(self, observation: Observation) -> torch.Tensor:
        legal_tokens(observation)
        obs_ids = observation.observation_tokens
        valid_history = np.flatnonzero(obs_ids != 0)
        length = int(valid_history[-1])+1 if len(valid_history) else 1
        count = int(observation.legal_next_mask.sum())
        # Remove only trailing PAD rows/tokens, not live actions/history.
        logits, _ = self.model(
            torch.as_tensor(obs_ids[None, :length].copy(), dtype=torch.long, device=self.device),
            torch.as_tensor(observation.state_channels[None].copy(), dtype=torch.float32, device=self.device),
            torch.as_tensor(observation.legal_next_tokens[None, :count].copy(), dtype=torch.long, device=self.device),
            torch.as_tensor(observation.legal_next_mask[None, :count].copy(), dtype=torch.bool, device=self.device),
        )
        probabilities = self.model.distribution(logits).probs
        if not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(-1), torch.ones(1, device=self.device), atol=1e-6):
            raise FloatingPointError('snapshot probabilities invalid')
        return logits.cpu()

    def select_token(self, observation: Observation) -> int:
        logits = self.logits(observation)
        index = int(logits.argmax(-1).item()) if self.deterministic else int(torch.multinomial(logits.softmax(-1), 1, generator=self.generator).item())
        token = int(observation.legal_next_tokens[index])
        if token not in legal_tokens(observation):
            raise RuntimeError('snapshot selected illegal token; no fallback')
        return token


def make_agent(name: str, *, seed: int = 0, checkpoint=None, device='cpu'):
    if name == 'random':
        return RandomAgent(seed)
    if name == 'rule':
        return RuleAgent()
    if name == 'snapshot':
        if checkpoint is None:
            raise ValueError('snapshot requires explicit checkpoint')
        return SnapshotAgent(checkpoint, seed=seed, device=device)
    raise ValueError(f'unknown agent {name!r}; no fallback')
