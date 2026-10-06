"""Observation-only evaluation agents with session-scoped frozen policy resources."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np
import torch

from .action_state import COMMIT_TOKEN, PASS_TOKEN, TokenCodec
from .cards import Card, Rank, effective_rank_value
from .environment_types import Observation
from .model import PolicyValueNet
from .training.checkpoint import CheckpointError
from .training.runtime import file_sha256
from .training.snapshot import load_policy_snapshot


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
    """Deterministic self/public heuristic, not an oracle or a strength claim."""
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
        return min([token for token in tokens if token != PASS_TOKEN] or tokens)


@dataclass(frozen=True, slots=True)
class _SnapshotResources:
    model: PolicyValueNet
    device: torch.device
    metadata: dict
    training_deals: list


_SNAPSHOT_CACHE: dict[tuple[Path, torch.device], _SnapshotResources] = {}
_SNAPSHOT_CACHE_LOCK = Lock()


def clear_snapshot_cache() -> None:
    with _SNAPSHOT_CACHE_LOCK:
        _SNAPSHOT_CACHE.clear()


def reset_snapshot_cache() -> None:
    clear_snapshot_cache()


clear_cache = clear_snapshot_cache
reset_cache = reset_snapshot_cache


def _snapshot_device(device: torch.device) -> torch.device:
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(f'requested snapshot device {device} is unavailable')
        return torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    return torch.device('cpu') if device.type == 'cpu' else device


def _checkpoint_digest(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise CheckpointError(f'could not read checkpoint {path}: {exc}') from exc


def _snapshot_resources(path: Path, device: torch.device) -> _SnapshotResources:
    key = (path, device)
    with _SNAPSHOT_CACHE_LOCK:
        digest = _checkpoint_digest(path)
        cached = _SNAPSHOT_CACHE.get(key)
        if cached is not None and cached.metadata['sha256'] == digest:
            return cached
        _SNAPSHOT_CACHE.pop(key, None)
        artifact = load_policy_snapshot(path)
        config = artifact.model_config
        if not isinstance(config, dict):
            raise CheckpointError('snapshot model_config must be a dictionary')
        metadata = artifact.metadata
        profile = metadata.get('profile')
        algorithm = metadata.get('algorithm')
        update = metadata.get('update')
        if type(profile) is not str or not profile:
            raise CheckpointError('snapshot profile missing/invalid')
        if type(algorithm) is not str or algorithm not in {'ippo', 'vrpo'}:
            raise CheckpointError('snapshot algorithm missing/invalid')
        if type(update) is not int or update < 0:
            raise CheckpointError('snapshot update missing/invalid')
        if config.get('architecture') != 'mean-pooled-token-MLP':
            raise CheckpointError('unsupported actor architecture')
        hidden = config.get('hidden_dim')
        if type(hidden) is not int or hidden <= 0:
            raise CheckpointError('snapshot hidden_dim missing/invalid')
        if any(type(config.get(name)) is not int or config[name] != 256 for name in ('vocabulary', 'channels')):
            raise CheckpointError('snapshot observation contract differs')
        weights = dict(artifact.model_state)
        if not weights:
            raise CheckpointError('snapshot has no actor state; no fallback')
        with torch.random.fork_rng(devices=[]):
            model = PolicyValueNet(hidden_dim=hidden)
        model.load_state_dict(weights, strict=True)
        model.to(device).eval()
        model.requires_grad_(False)
        enriched = {
            'kind': 'snapshot', 'checkpoint': str(path), 'sha256': digest,
            'profile': profile, 'algorithm': algorithm, 'update': update,
            'model_version': metadata.get('model_version'), 'training_version': metadata.get('training_version'),
            'protocol_versions': deepcopy(metadata.get('protocol_versions', {})),
            'model_config': deepcopy(config),
            'semantic_contract': deepcopy(metadata.get('semantic_contract')),
            'source_fingerprint': metadata.get('source_fingerprint'),
            'implementation_fingerprint': metadata.get('implementation_fingerprint'),
        }
        if _checkpoint_digest(path) != digest:
            raise CheckpointError('snapshot checkpoint changed while loading; no fallback')
        resources = _SnapshotResources(model, device, enriched, deepcopy(artifact.training_deals))
        _SNAPSHOT_CACHE[key] = resources
        return resources


class SnapshotAgent:
    """Frozen model wrapper with an independent CPU RNG stream."""
    @classmethod
    def clear_cache(cls) -> None:
        clear_snapshot_cache()

    @classmethod
    def reset_cache(cls) -> None:
        clear_snapshot_cache()

    def __init__(self, checkpoint: str | Path, *, seed: int = 0, device: str = 'cpu', deterministic: bool = False):
        self.path = Path(checkpoint).resolve()
        self.device = torch.device(device)
        resources = _snapshot_resources(self.path, _snapshot_device(self.device))
        self._resources = resources
        self.model = resources.model
        self._model_device = resources.device
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.deterministic = deterministic
        self.training_deals = deepcopy(resources.training_deals)
        self.metadata = deepcopy(resources.metadata)
        self.metadata.update(seed=seed, deterministic=deterministic, device=str(self.device))

    @staticmethod
    def _shape(observation: Observation) -> tuple[int, int]:
        legal_tokens(observation)
        ids = observation.observation_tokens
        valid = np.flatnonzero(ids != 0)
        length = int(valid[-1]) + 1 if len(valid) else 1
        return length, int(observation.legal_next_mask.sum())

    @torch.no_grad()
    def logits_batch(self, observations: Sequence[Observation]) -> torch.Tensor:
        if not observations:
            raise ValueError('logits_batch requires at least one observation')
        shapes = [self._shape(observation) for observation in observations]
        max_length = max(shape[0] for shape in shapes)
        max_count = max(shape[1] for shape in shapes)
        count = len(observations)
        ids = np.zeros((count, max_length), dtype=np.int64)
        channels = np.zeros((count, observations[0].state_channels.shape[0]), dtype=np.float32)
        candidates = np.zeros((count, max_count), dtype=np.int64)
        masks = np.zeros((count, max_count), dtype=np.bool_)
        for row, (observation, (length, legal_count)) in enumerate(zip(observations, shapes)):
            ids[row, :length] = observation.observation_tokens[:length]
            channels[row] = observation.state_channels
            candidates[row, :legal_count] = observation.legal_next_tokens[:legal_count]
            masks[row, :legal_count] = observation.legal_next_mask[:legal_count]
        logits, _ = self.model(
            torch.as_tensor(ids, dtype=torch.long, device=self._model_device),
            torch.as_tensor(channels, dtype=torch.float32, device=self._model_device),
            torch.as_tensor(candidates, dtype=torch.long, device=self._model_device),
            torch.as_tensor(masks, dtype=torch.bool, device=self._model_device),
        )
        probabilities = self.model.distribution(logits).probs
        if not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(-1), torch.ones(count, device=self._model_device), atol=1e-6):
            raise FloatingPointError('snapshot probabilities invalid')
        return logits.cpu()

    @torch.no_grad()
    def logits(self, observation: Observation) -> torch.Tensor:
        return self.logits_batch([observation])

    @classmethod
    def select_tokens_batch(cls, requests: Sequence[tuple['SnapshotAgent', Observation]]) -> list[int]:
        if not requests:
            return []
        results: list[int | None] = [None] * len(requests)
        groups: dict[tuple[int, torch.device], list[tuple[int, SnapshotAgent, Observation]]] = {}
        for index, (agent, observation) in enumerate(requests):
            if not isinstance(agent, cls):
                raise TypeError('batch requests must contain SnapshotAgent instances')
            groups.setdefault((id(agent.model), agent._model_device), []).append((index, agent, observation))
        for group in groups.values():
            logits = group[0][1].logits_batch([item[2] for item in group])
            for row, (index, agent, observation) in enumerate(group):
                count = int(observation.legal_next_mask.sum())
                row_logits = logits[row, :count]
                index_in_candidates = int(row_logits.argmax().item()) if agent.deterministic else int(torch.multinomial(row_logits.softmax(-1), 1, generator=agent.generator).item())
                token = int(observation.legal_next_tokens[index_in_candidates])
                if token not in legal_tokens(observation):
                    raise RuntimeError('snapshot selected illegal token; no fallback')
                results[index] = token
        return [int(value) for value in results]

    def select_token(self, observation: Observation) -> int:
        return self.select_tokens_batch([(self, observation)])[0]


class EvaluationSession:
    """Own fixed inference resources for one evaluation session."""
    def __init__(self, candidate_checkpoint, snapshot_checkpoint, *, device='cpu'):
        self.candidate_checkpoint = Path(candidate_checkpoint).resolve()
        self.snapshot_checkpoint = Path(snapshot_checkpoint).resolve()
        self.device = device
        self._candidate = SnapshotAgent(self.candidate_checkpoint, seed=0, device=device)
        self._snapshot = SnapshotAgent(self.snapshot_checkpoint, seed=0, device=device)

    @property
    def metadata(self) -> dict:
        return {'candidate': deepcopy(self._candidate.metadata), 'snapshot_opponent': deepcopy(self._snapshot.metadata)}

    def make_agent(self, name: str, *, seed: int = 0, checkpoint=None):
        if name == 'snapshot':
            if checkpoint is None:
                raise ValueError('snapshot requires explicit checkpoint')
            return SnapshotAgent(checkpoint, seed=seed, device=self.device)
        return make_agent(name, seed=seed, checkpoint=checkpoint, device=self.device)

    def select_tokens_batch(self, requests):
        return SnapshotAgent.select_tokens_batch(requests)


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


__all__ = [
    'RandomAgent', 'RuleAgent', 'SnapshotAgent', 'EvaluationSession', 'make_agent',
    'legal_tokens', 'clear_snapshot_cache', 'reset_snapshot_cache', 'clear_cache', 'reset_cache',
]
