"""Transactional batches of independent policy-facing environments.

Transport types are shared with ``guandan.environment_types``.  The single-env
import is lazy so that ``guandan.environment`` can re-export this batch API.
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Sequence, TypeVar

import numpy as np

from .action_state import EpisodeTerminatedError, ProtocolError
from .environment_types import (
    BatchObservation,
    BatchStepResult,
    GameConfig,
    Observation,
    ObservationSpec,
    StepResult,
    strict_int,
)
from .state import IllegalActionError

if TYPE_CHECKING:
    from .environment import GuandanEnv

_T = TypeVar("_T")

# V1 policy tensors; inactive rows keep their zero-filled allocations.
_ARRAY_FIELDS = (
    ("observation_tokens", np.int32, "max_observation_tokens"),
    ("state_channels", np.float32, "num_state_channels"),
    ("legal_next_tokens", np.int32, "max_legal_next_tokens"),
    ("legal_next_mask", np.bool_, "max_legal_next_tokens"),
    ("legal_next_is_commit", np.bool_, "max_legal_next_tokens"),
    ("legal_next_kinds", np.int32, "max_legal_next_tokens"),
    ("legal_next_values", np.int32, "max_legal_next_tokens"),
)


class GuandanEnvBatch:
    """A fixed-size collection with explicit reset and all-or-nothing writes.

    ``envs`` exposes a tuple of single environments.  Successful writes replace
    affected slots with trial instances; failed writes preserve every original
    instance, including its RNG.  Retained references are therefore snapshots,
    not stable handles to slots.  Inactive slots accept only the PAD no-op.
    """

    def __init__(
        self,
        configs: Sequence[GameConfig],
        *,
        auto_reset: bool = False,
        observation_spec: ObservationSpec | None = None,
    ) -> None:
        from .environment import GuandanEnv

        configs = tuple(configs)
        if not configs:
            raise ValueError("batch requires at least one config")
        if any(not isinstance(config, GameConfig) for config in configs):
            raise TypeError("configs must contain only GameConfig values")
        if not isinstance(auto_reset, bool):
            raise TypeError("auto_reset must be bool")
        if observation_spec is not None and not isinstance(observation_spec, ObservationSpec):
            raise TypeError("observation_spec must be ObservationSpec or None")
        self.observation_spec = observation_spec if observation_spec is not None else ObservationSpec()
        self.auto_reset = auto_reset
        self._envs = tuple(
            GuandanEnv(config, observation_spec=self.observation_spec) for config in configs
        )

    @property
    def envs(self) -> tuple[GuandanEnv, ...]:
        """Current independent slots; reading this property never resets them."""
        return self._envs

    def _items(self, values: Sequence[_T], name: str) -> tuple[_T, ...]:
        if isinstance(values, (str, bytes, bytearray, memoryview)):
            raise TypeError(f"{name} must be a sequence of slot values")
        if len(values) != len(self._envs):
            raise ValueError(f"{name} length must equal batch size")
        return tuple(values)

    @staticmethod
    def _reset_arguments(seed: int | None, initial_state: bytes | None) -> None:
        if seed is not None:
            strict_int(seed, "seed")
        if initial_state is not None and not isinstance(initial_state, bytes):
            raise TypeError("initial_state must be bytes or None")
        if seed is not None and initial_state is not None:
            raise ValueError("provide seed or initial_state for a slot, not both")

    def _stack(
        self,
        envs: Sequence[GuandanEnv],
        observations: Sequence[Observation | None],
    ) -> BatchObservation:
        """Build owned arrays before committing a transaction's trial slots."""
        size = len(envs)
        arrays = {
            name: np.zeros((size, getattr(self.observation_spec, capacity)), dtype=dtype)
            for name, dtype, capacity in _ARRAY_FIELDS
        }
        player_id = np.full(size, -1, dtype=np.int32)
        team_id = np.full(size, -1, dtype=np.int32)
        token_step = np.array([env.token_step for env in envs], dtype=np.int64)
        committed_step = np.array([env.committed_step for env in envs], dtype=np.int64)
        present = np.zeros(size, dtype=np.bool_)
        phases: list[str] = []
        for index, (env, observation) in enumerate(zip(envs, observations)):
            if observation is None:
                phases.append("TRUNCATED" if env.truncated else "TERMINAL" if env.done else "UNINITIALIZED")
                continue
            phases.append(observation.phase)
            present[index] = True
            player_id[index] = observation.player_id
            team_id[index] = observation.team_id
            token_step[index] = observation.token_step
            committed_step[index] = observation.committed_step
            for name, destination in arrays.items():
                source = getattr(observation, name)
                if (
                    not isinstance(source, np.ndarray)
                    or source.shape != destination.shape[1:]
                    or source.dtype != destination.dtype
                ):
                    raise ProtocolError(f"slot {index}: {name} does not match ObservationSpec/dtype")
                destination[index] = source
        return BatchObservation(
            player_id=player_id,
            team_id=team_id,
            phase=np.array(phases, dtype=np.str_),
            token_step=token_step,
            committed_step=committed_step,
            **arrays,
            private_hand_card_ids=None,
            present=present,
        )

    def reset(
        self,
        *,
        seeds: Sequence[int | None] | None = None,
        initial_states: Sequence[bytes | None] | None = None,
    ) -> BatchObservation:
        """Reset every slot atomically, including all slot RNG streams.

        Seeds and initial states may be combined across *different* slots, but
        supplying both for the same slot is an error, as in ``GuandanEnv``.
        """
        seeds = (None,) * len(self._envs) if seeds is None else self._items(seeds, "seeds")
        states = (None,) * len(self._envs) if initial_states is None else self._items(initial_states, "initial_states")
        for seed, state in zip(seeds, states):
            self._reset_arguments(seed, state)
        trials = tuple(env.clone() for env in self._envs)
        observations = [
            env.reset(seed=seed, initial_state=state)
            for env, seed, state in zip(trials, seeds, states)
        ]
        if any(observation is None for observation in observations):
            raise ProtocolError("reset must return a live Observation")
        observation = self._stack(trials, observations)
        self._envs = trials
        return observation

    def reset_at(
        self, index: int, *, seed: int | None = None, initial_state: bytes | None = None
    ) -> Observation:
        """Reset just one slot and return an independent single observation."""
        index = strict_int(index, "index")
        if index >= len(self._envs):
            raise IndexError("batch index out of range")
        self._reset_arguments(seed, initial_state)
        trial = self._envs[index].clone()
        observation = trial.reset(seed=seed, initial_state=initial_state)
        if observation is None:
            raise ProtocolError("reset must return a live Observation")
        self._stack((trial,), (observation,))  # Validate before replacing a slot.
        observation = deepcopy(observation)
        self._envs = self._envs[:index] + (trial,) + self._envs[index + 1:]
        return observation

    def observe(self) -> BatchObservation:
        """Stack live observations and zero-fill absent/inactive rows."""
        return self._stack(self._envs, [env.observe() for env in self._envs])

    @staticmethod
    def _inactive_result(env: GuandanEnv) -> StepResult:
        info = {"inactive": True}
        if env.truncated:
            info["termination_reason"] = "max_token_steps"
        return StepResult(
            observation=None,
            rewards=np.zeros(4, dtype=np.float32),
            done=bool(env.done),
            truncated=bool(env.truncated),
            is_commit=False,
            committed_action=None,
            token_step=env.token_step,
            committed_step=env.committed_step,
            info=info,
        )

    def step(self, token_ids: np.ndarray) -> BatchStepResult:
        """Consume a signed NumPy integer vector without coercing input types.

        With auto-reset, result flags/counters/info describe the transition
        that ended, while its observation row describes the next reset hand.
        A failure anywhere, even in auto-reset or stacking, publishes no slots.
        """
        if not isinstance(token_ids, np.ndarray):
            raise TypeError("token_ids must be a signed NumPy integer vector")
        if token_ids.shape != (len(self._envs),) or token_ids.dtype.kind != "i":
            raise ValueError("token_ids must have shape (B,) and a signed integer dtype")
        inactive = tuple(env.done or env.truncated for env in self._envs)
        for env, ended, token in zip(self._envs, inactive, token_ids):
            if ended:
                if token != self.observation_spec.pad_token_id:
                    raise EpisodeTerminatedError("inactive batch slots require PAD=0")
            elif env.observe() is None:
                raise RuntimeError("every batch slot must be reset before step")
            elif not 0 < token < self.observation_spec.token_vocab_size:
                raise IllegalActionError("token is outside the legal vocabulary or is PAD")

        # Inactive originals are deliberately retained, not reset or stepped.
        trials = tuple(env if ended else env.clone() for env, ended in zip(self._envs, inactive))
        results: list[StepResult] = []
        observations: list[Observation | None] = []
        for env, ended, token in zip(trials, inactive, token_ids):
            result = self._inactive_result(env) if ended else env.step(int(token))
            # Snapshot info/reward before reset can mutate any single-env data.
            result = deepcopy(result)
            observation = result.observation
            if self.auto_reset and not ended and (result.done or result.truncated):
                final_info = deepcopy(result.info)
                result.info["auto_reset"] = True
                result.info["final_info"] = final_info
                observation = env.reset()
                if observation is None:
                    raise ProtocolError("auto-reset must return a live Observation")
            results.append(result)
            observations.append(observation)

        rewards = np.stack([result.rewards for result in results]).astype(np.float32, copy=True)
        if rewards.shape != (len(trials), 4):
            raise ProtocolError("single-environment rewards must have shape (4,)")
        output = BatchStepResult(
            observation=self._stack(trials, observations),
            rewards=rewards,
            done=np.array([result.done for result in results], dtype=np.bool_),
            truncated=np.array([result.truncated for result in results], dtype=np.bool_),
            is_commit=np.array([result.is_commit for result in results], dtype=np.bool_),
            committed_action=tuple(result.committed_action for result in results),
            token_step=np.array([result.token_step for result in results], dtype=np.int64),
            committed_step=np.array([result.committed_step for result in results], dtype=np.int64),
            info=tuple(result.info for result in results),
        )
        self._envs = trials
        return output

    def serialize(self) -> list[bytes]:
        """Return one independent single-environment payload per slot."""
        return [env.serialize() for env in self._envs]

    def load_serialized(self, payloads: Sequence[bytes]) -> None:
        """Restore slots atomically; never reset them or change auto-reset."""
        from .environment import GuandanEnv

        payloads = self._items(payloads, "payloads")
        if any(not isinstance(payload, bytes) for payload in payloads):
            raise TypeError("payloads must contain serialized bytes")
        trials = tuple(GuandanEnv.deserialize(payload) for payload in payloads)
        if any(env.observation_spec != self.observation_spec for env in trials):
            raise ProtocolError("serialized ObservationSpec differs from batch")
        self._stack(trials, [env.observe() for env in trials])
        self._envs = trials


__all__ = ["BatchObservation", "BatchStepResult", "GuandanEnvBatch"]
