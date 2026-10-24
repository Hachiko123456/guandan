# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/runtime/sampler.py.
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch
from guandan_marvel.model.tensordict import TensorDict

from guandan_marvel.buffer import BufferConfig, FieldSpecLike, TrajectoryBuffer, trajectory_field_specs
from guandan_marvel.context_schedule import (
    context_hash, curriculum_expanded_contexts, curriculum_provenance,
    expand_context_schedule, schedule_coverage, validate_context_curriculum,
    validate_context_schedule,
)
from guandan_marvel.env_batch import EnvStepBatch
from guandan_marvel.model import KVCache
from guandan_marvel.reward import TRAINING_REWARD_SCALE
from guandan_marvel.runtime.trajectory_layout import trim_completed_trajectory_padding
from guandan_marvel.utils import unwrap_distributed_module

SAMPLER_CONTRACT_SCHEMA = "guandan-marvel-sampler-contract-v2"
SAMPLER_STATE_SCHEMA = "guandan-marvel-sampler-state-v2"
PRE_PROVENANCE_SAMPLER_STATE_SCHEMA = "guandan-marvel-sampler-state-v1"
TRAINING_DATA_PROVENANCE_SCHEMA = "guandan-marvel-training-data-provenance-v1"
_SAMPLER_PURPOSES = frozenset({"on_policy", "evaluation", "test"})


class SamplerFailedStateError(RuntimeError):
    """The sampler observed a non-transactional failure and cannot continue."""


@dataclass(frozen = True)
class SamplerConfig:
    """Configuration subset required by the sampler and trajectory buffers."""

    game: Any
    device: torch.device | str
    rollout_batch_size: int
    collect_batch_size: int
    sampling_temperature: float = 1.0
    purpose: str = "on_policy"
    cache_dtype: torch.dtype | None = None
    field_specs: Optional[Mapping[str, FieldSpecLike]] = None
    env_seed: int | None = 0
    state_logger: Optional[Callable[[Any, int], None]] = None
    timing_callback: Optional[Callable[[str, float], None]] = None
    data_domain: str | None = None
    data_manifest_hash: str | None = None
    data_generator_version: str = "guandan-marvel-env-batch-continuous-prng-v1"
    context_schedule: Mapping[str, object] | None = None
    context_curriculum: Mapping[str, object] | None = None
    reward_scale: float = TRAINING_REWARD_SCALE
    trim_completed_padding: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.trim_completed_padding, bool):
            raise TypeError("trim_completed_padding must be bool")
        object.__setattr__(self, "device", torch.device(self.device))
        temperature = float(self.sampling_temperature)
        purpose = str(self.purpose).strip().lower()
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("sampling_temperature must be finite and non-negative.")
        if purpose not in _SAMPLER_PURPOSES:
            raise ValueError(f"purpose must be one of {sorted(_SAMPLER_PURPOSES)}.")
        if purpose == "on_policy" and temperature != 1.0:
            raise ValueError("on_policy sampling requires sampling_temperature=1.0 exactly.")
        object.__setattr__(self, "sampling_temperature", temperature)
        object.__setattr__(self, "purpose", purpose)
        if self.data_domain is not None and str(self.data_domain) != "training":
            raise ValueError("sampler data_domain must be 'training' or None")
        if self.data_manifest_hash is not None:
            value = str(self.data_manifest_hash)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise ValueError("data_manifest_hash must be a 64-character hexadecimal digest")
            object.__setattr__(self, "data_manifest_hash", value.lower())
        if not str(self.data_generator_version).strip():
            raise ValueError("data_generator_version must be non-empty")
        scale = float(self.reward_scale)
        if not math.isfinite(scale) or scale != TRAINING_REWARD_SCALE:
            raise ValueError(
                "reward_scale must be exactly the rank-point training scale 1/3"
            )
        object.__setattr__(self, "reward_scale", scale)
        if self.field_specs is None:
            specs = trajectory_field_specs(self.game)
            object.__setattr__(self, "field_specs", specs)


class Sampler:
    """Collect trajectories from the environment and manage commit/no-commit rollout modes."""

    def __init__(
        self,
        *,
        config: SamplerConfig,
        envs: Any | None = None,
        rollout_buffer: TrajectoryBuffer | None = None,
        parsed_envs_step: EnvStepBatch | None = None,
        policy_router: Any | None = None,
    ) -> None:
        self.config = config
        self.game = config.game
        self.device = config.device
        self._cache_dtype = config.cache_dtype
        self._sampling_temperature = float(config.sampling_temperature)
        self.purpose = config.purpose
        if policy_router is not None:
            # The pool router consumes the same on-policy temperature-1 path and
            # emits explicit owner/train masks.  No other purpose is supported.
            if config.purpose != "on_policy":
                raise ValueError("a policy router is supported only for on_policy sampling")
            for method in ("prepare", "policy_probabilities", "on_commit", "state_dict"):
                if not callable(getattr(policy_router, method, None)):
                    raise TypeError(f"policy router must define {method}()")
        self.policy_router = policy_router
        self._env_seed = config.env_seed
        self._data_domain = config.data_domain
        self._data_manifest_hash = config.data_manifest_hash
        self._data_generator_version = str(config.data_generator_version)
        self._training_deal_records: list[dict[str, object]] = []
        self._training_deal_hashes: set[str] = set()

        def game_attr(name: str) -> int:
            if hasattr(self.game, name):
                return int(getattr(self.game, name))
            upper_name = name.upper()
            if hasattr(self.game, upper_name):
                return int(getattr(self.game, upper_name))
            raise AttributeError(f"Game module missing expected attribute '{name}'/'{upper_name}'.")

        self.T = game_attr("max_game_horizon")
        self.P = game_attr("num_players")
        self.rollout_batch_size = config.rollout_batch_size
        self.collect_batch_size = config.collect_batch_size
        if self.collect_batch_size < 0:
            raise ValueError("collect_batch_size must be non-negative.")
        self._use_commit_buffer = self.collect_batch_size > 0

        specs = dict(config.field_specs)
        buffer_config = BufferConfig(
            game = self.game,
            device = self.device,
            field_specs = specs,
        )
        self._buffer_config = buffer_config

        self.envs = envs if envs is not None else self.game.EnvBatch(self.rollout_batch_size)
        self._context_schedule = config.context_schedule
        self._context_curriculum = config.context_curriculum
        self._curriculum_iteration = 0
        self._last_context_configuration: tuple[int, str] | None = None
        if self._context_schedule is not None:
            validate_context_schedule(dict(self._context_schedule))
        if self._context_curriculum is not None:
            validate_context_curriculum(dict(self._context_curriculum))
            if self._context_schedule is None:
                raise ValueError("a context curriculum requires a target context schedule")
        if envs is None and self._context_schedule is not None:
            self._configure_contexts(0)
        if parsed_envs_step is None:
            seed = self._env_seed if self._env_seed is not None else 0
            reset_result = self.envs.reset(seed)
            parsed_envs_step = EnvStepBatch.from_env_step(reset_result, config)
        self.parsed_envs_step = parsed_envs_step
        self._episode_identities: dict[int, dict[str, object]] = {}
        if self.purpose == "on_policy" and self._data_domain == "training":
            # Capture the in-flight deals now; each is recorded once, at the
            # moment it completes (see _record_training_deals).
            self._capture_episode_identities(range(self.rollout_batch_size))

        self.rollout_buffer = rollout_buffer or TrajectoryBuffer(
            self.rollout_batch_size,
            buffer_config,
        )
        self.commit_buffer: TrajectoryBuffer | None = None
        if self._use_commit_buffer:
            self.commit_buffer = TrajectoryBuffer(
                self.rollout_batch_size + self.collect_batch_size,
                buffer_config,
            )

        self.commit_buffer_size = 0
        self.total_env_steps = 0
        self.total_game_steps = 0
        self.last_step_rewards: torch.Tensor | None = None
        self.last_step_is_terminal: torch.Tensor | None = None
        self.attempted_env_steps = 0
        self.failed_env_steps = 0
        self.rejected_rollout_steps = 0
        self._failed = False
        self._failure_info: dict[str, object] | None = None
        self._terminal_info_queue: list[dict[str, object]] = []

        self.agent_kv: KVCache | None = None
        self._reset_mask_full: torch.Tensor | None = None
        self._micro_buffer_range = torch.arange(
            self.rollout_batch_size,
            dtype = torch.long,
            device = self.device,
        )

    @property
    def failed(self) -> bool:
        return self._failed

    def _expanded_contexts(self, iteration: int) -> list[dict[str, object]]:
        if self._context_curriculum is not None:
            return curriculum_expanded_contexts(
                self._context_curriculum, self._context_schedule, iteration,
            )
        return expand_context_schedule(self._context_schedule)

    def _configure_contexts(self, iteration: int) -> None:
        """Install the iteration's context mixture and deterministic offset.

        The environment consumes a global per-deal counter that restarts at
        ``offset = (iteration * collect_batch_size) % len(expanded)`` exactly
        once per collection.  Consecutive slots consume consecutive positions,
        the counter advances by one per deal, and the offset is a pure function
        of the iteration and collect size (never wall-clock time).  The
        configuration is applied only when ``(offset, expanded hash)`` changes,
        so a repeated collect in the same iteration never clears the counter.
        """
        configure = getattr(self.envs, "configure_context_schedule", None)
        if not callable(configure):
            raise TypeError("runtime EnvBatch does not support context schedules")
        expanded = [dict(row) for row in self._expanded_contexts(int(iteration))]
        offset = (int(iteration) * int(self.collect_batch_size)) % len(expanded)
        configuration = (offset, self._canonical_hash(expanded))
        if configuration == self._last_context_configuration:
            return
        try:
            configure(expanded, offset=offset)
        except TypeError:
            # Backend without offset support is rejected rather than silently
            # falling back to a non-deterministic assignment.
            raise TypeError("runtime EnvBatch.configure_context_schedule must accept offset=")
        self._last_context_configuration = configuration
        self._curriculum_iteration = int(iteration)

    def set_curriculum_iteration(self, iteration: int) -> None:
        """Record the upcoming iteration; the mixture is installed in collect()."""
        if self._context_schedule is None:
            return
        value = int(iteration)
        if value < 0:
            raise ValueError("curriculum iteration must be non-negative")
        self._curriculum_iteration = value

    @property
    def failure_info(self) -> dict[str, object] | None:
        return None if self._failure_info is None else dict(self._failure_info)

    def _require_healthy(self) -> None:
        if self._failed:
            info = self._failure_info or {}
            raise SamplerFailedStateError(
                "sampler is unusable after a failed rollout; construct or restore a new sampler "
                f"from a known-consistent state (failure={info})"
            )

    def _mark_failed(self, *, stage: str, error: BaseException) -> None:
        self._failed = True
        self._failure_info = {
            "schema": SAMPLER_CONTRACT_SCHEMA,
            "strategy": "poison_on_failure",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "exact_resume_allowed": False,
            "completed_env_steps": self.total_env_steps,
            "completed_game_steps": self.total_game_steps,
            "attempted_env_steps": self.attempted_env_steps,
            "failed_env_steps": self.failed_env_steps,
        }

    def invalidate(self, *, stage: str, error: BaseException) -> None:
        """Permanently disable this sampler after an owning trainer becomes inconsistent."""
        if not self._failed:
            self.failed_env_steps += self.rollout_batch_size
            self.rejected_rollout_steps += 1
            self._mark_failed(stage=stage, error=error)

    def prepare_agent_cache(self, *, agent_model: Any) -> None:
        """Ensure the decoder KV cache is initialized and aligned with rollout history."""
        self._require_healthy()
        with torch.inference_mode():
            self._ensure_agent_cache(agent_model)
            self._rebuild_agent_cache(agent_model)
            if self.policy_router is not None:
                self.policy_router.prepare(rollout_buffer=self.rollout_buffer)

    def collect(
        self,
        *,
        agent_model: Any,
    ) -> TensorDict:
        """Roll out trajectories until `collect_batch_size` games finish and return that batch."""
        self._require_healthy()
        if not self._use_commit_buffer or self.commit_buffer is None:
            raise RuntimeError(
                "Sampler.collect() requires collect_batch_size > 0 (commit mode). "
                "Use rollout_step() and read sampler.envs/parsed_envs_step directly in no-commit mode."
            )
        if self._context_schedule is not None:
            self._configure_contexts(self._curriculum_iteration)
        # Inference performs no gradient synchronization. Bypass DDP's forward
        # bookkeeping for independently progressing environment streams; learner
        # forwards/backwards still use the synchronized DDP module.
        if self.config.trim_completed_padding:
            agent_model = unwrap_distributed_module(agent_model)
        with torch.inference_mode():
            cache_started = time.perf_counter() if self.config.timing_callback is not None else None
            self.prepare_agent_cache(agent_model = agent_model)
            if cache_started is not None:
                self.config.timing_callback("cache_prefill_and_allocation", time.perf_counter() - cache_started)
            while self.commit_buffer_size < self.collect_batch_size:
                self.rollout_step(agent_model = agent_model)

        collect_batch = self.commit_buffer[: self.collect_batch_size]
        if self.config.trim_completed_padding:
            collect_batch = trim_completed_trajectory_padding(collect_batch)
        collect_batch = collect_batch.clone()
        self.consume(self.collect_batch_size)
        return collect_batch

    @staticmethod
    def _canonical_hash(value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _debug_game_state(self, index: int) -> Mapping[str, object]:
        debug = getattr(self.envs, "debug_state", None)
        if callable(debug):
            payload = debug(int(index))
            return payload
        env_values = getattr(self.envs, "envs", None)
        if env_values is not None:
            env = env_values[int(index)]
            full = env.full_state()
            state = full["round_state"]
            return {
                "config": {
                    "seed": getattr(env.config, "seed", None),
                    "level_rank": state.get("level_rank", getattr(env.config, "level_rank", None)),
                    "leader_seat": state.get("leader_seat", getattr(env.config, "leader_seat", None)),
                    "previous_result": state.get("previous_result"),
                },
                "state": state,
                "reset_count": int(getattr(env, "_reset_count", 0)),
            }
        raise TypeError("training provenance requires an environment debug/full-state API")

    def _episode_identity(self, index: int) -> dict[str, object]:
        payload = self._debug_game_state(int(index))
        state = payload["state"]
        config = payload.get("config", {})
        previous = state.get("previous_result")
        level = int(state["level_rank"])
        leader = int(state["leader_seat"])
        hands = [sorted(int(card) for card in hand) for hand in state["hands"]]
        identity = {"hands": hands, "level_rank": level, "leader_seat": leader, "previous_result": previous}
        return {
            "deal_sha256": self._canonical_hash(identity),
            "context_sha256": context_hash(
                {"level_rank": level, "leader_seat": leader, "previous_result": previous},
            ),
            "level_rank": level,
            "leader_seat": leader,
            "previous_result": previous,
            "source_seed": config.get("seed", self._env_seed),
            "reset_count": int(payload.get("reset_count", 0)),
        }

    def _capture_episode_identities(self, indices) -> None:
        """Record the identity of each slot's *current* (post-reset) game."""
        for index in indices:
            self._episode_identities[int(index)] = self._episode_identity(int(index))

    def _record_training_deals(self, indices) -> None:
        """Emit one provenance record per just-completed game.

        ``rollout_step`` calls this after ``envs.step``; with auto-reset the
        environment has already dealt the *next* game, so the completed game's
        identity is taken from the per-slot store captured when it started, and
        the store is then refreshed from the new post-reset state.
        """
        for index in indices:
            index = int(index)
            identity = self._episode_identities.get(index)
            if identity is None:
                identity = self._episode_identity(index)
            record = {
                "slot": index,
                "reset_count": int(identity["reset_count"]),
                "deal_sha256": identity["deal_sha256"],
                "context_sha256": identity["context_sha256"],
                "source_seed": identity["source_seed"],
                "level_rank": int(identity["level_rank"]),
                "leader_seat": int(identity["leader_seat"]),
                "previous_result": identity["previous_result"],
            }
            self._training_deal_records.append(record)
            self._training_deal_hashes.add(str(record["deal_sha256"]))
        self._capture_episode_identities(indices)

    def training_data_provenance(self) -> dict[str, object]:
        observed_levels = sorted({int(row["level_rank"]) for row in self._training_deal_records})
        observed_leaders = sorted({int(row["leader_seat"]) for row in self._training_deal_records})
        observed_modes = sorted({
            "none" if row.get("previous_result") is None else (
                "cancelled" if row["previous_result"].get("tribute_cancelled") else str(row["previous_result"]["outcome_class"]).lower()
            )
            for row in self._training_deal_records
        })
        declared_coverage = None
        if self._context_schedule is not None:
            declared_coverage = schedule_coverage(self._context_schedule)
        base = {
            "schema": TRAINING_DATA_PROVENANCE_SCHEMA,
            "domain": self._data_domain,
            "manifest_hash": self._data_manifest_hash,
            "generator_version": self._data_generator_version,
            "seed_semantics": "EnvBatch initial seed plus slot, followed by each slot's serialized continuous PRNG reset stream; subsequent deals are not represented as seed+counter",
            "initial_env_seed": self._env_seed,
            "rollout_batch_size": self.rollout_batch_size,
            "deal_count": len(self._training_deal_records),
            "deal_hashes": sorted(self._training_deal_hashes),
            "records_hash": self._canonical_hash(self._training_deal_records),
            "declared_context_coverage": declared_coverage,
            "observed_context_coverage": {
                "level_ranks": observed_levels,
                "leader_seats": observed_leaders,
                "tribute_modes": observed_modes,
                "deal_count": len(self._training_deal_records),
                "context_hashes": sorted({str(row["context_sha256"]) for row in self._training_deal_records}),
            },
            "curriculum": curriculum_provenance(
                self._context_curriculum, self._context_schedule, self._curriculum_iteration,
            ),
            "complete": bool(
                self.purpose == "on_policy"
                and self._data_domain == "training"
                and self._data_manifest_hash
                and self._training_deal_records
            ),
        }
        return {**base, "hash": self._canonical_hash(base)}

    def consume(self, batch_size: int) -> None:
        """Drop the oldest `batch_size` trajectories from the commit buffer."""
        if self.commit_buffer is None:
            raise RuntimeError("consume() is unavailable when collect_batch_size == 0.")
        if batch_size <= 0:
            return
        if batch_size > self.commit_buffer_size:
            raise ValueError("Cannot consume more trajectories than are stored in the commit buffer.")

        self.commit_buffer[:-batch_size] = self.commit_buffer[batch_size:].clone()
        self.commit_buffer[-batch_size:] = self.commit_buffer.empty(batch_size = batch_size)
        self.commit_buffer_size -= batch_size

    def pop_terminal_infos(self, count: int) -> list[dict[str, object]]:
        self._require_healthy()
        value = int(count)
        if value < 0 or value > len(self._terminal_info_queue):
            raise ValueError("terminal info count is outside the queued range")
        result = self._terminal_info_queue[:value]
        del self._terminal_info_queue[:value]
        return [dict(item) for item in result]

    def state_dict(self) -> dict[str, object]:
        self._require_healthy()
        return {
            "schema": SAMPLER_STATE_SCHEMA,
            "purpose": self.purpose,
            "sampling_temperature": self._sampling_temperature,
            "rollout_batch_size": self.rollout_batch_size,
            "collect_batch_size": self.collect_batch_size,
            "envs": self.envs.serialize(),
            "rollout_buffer": self.rollout_buffer.state_dict(),
            "commit_buffer": None if self.commit_buffer is None else self.commit_buffer.state_dict(),
            "commit_buffer_size": self.commit_buffer_size,
            "total_env_steps": self.total_env_steps,
            "total_game_steps": self.total_game_steps,
            "attempted_env_steps": self.attempted_env_steps,
            "failed_env_steps": self.failed_env_steps,
            "rejected_rollout_steps": self.rejected_rollout_steps,
            "terminal_info_queue": [dict(item) for item in self._terminal_info_queue],
            "training_deal_records": [dict(item) for item in self._training_deal_records],
            "episode_identities": {str(key): dict(value) for key, value in sorted(self._episode_identities.items())},
            "curriculum_iteration": int(self._curriculum_iteration),
            "training_data_provenance": self.training_data_provenance(),
            "policy_router": None if self.policy_router is None else self.policy_router.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != SAMPLER_STATE_SCHEMA:
            raise ValueError(
                f"unsupported sampler state schema {state.get('schema')!r}; "
                f"expected {SAMPLER_STATE_SCHEMA}; {PRE_PROVENANCE_SAMPLER_STATE_SCHEMA} lacks training provenance"
            )
        expected = {
            "purpose": self.purpose,
            "sampling_temperature": self._sampling_temperature,
            "rollout_batch_size": self.rollout_batch_size,
            "collect_batch_size": self.collect_batch_size,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"sampler state {key} mismatch")
        env_type = type(self.envs)
        deserialize = getattr(env_type, "deserialize", None)
        if not callable(deserialize):
            raise TypeError("environment batch type does not support deserialize()")
        restored_envs = deserialize(state["envs"])
        observed = restored_envs.observe()
        parsed = EnvStepBatch.from_env_step(observed, self.config)
        self.rollout_buffer.load_state_dict(state["rollout_buffer"])
        raw_commit = state.get("commit_buffer")
        if self.commit_buffer is None:
            if raw_commit is not None:
                raise ValueError("checkpoint contains a commit buffer but sampler is no-commit")
        else:
            if raw_commit is None:
                raise ValueError("checkpoint is missing the configured commit buffer")
            self.commit_buffer.load_state_dict(raw_commit)
        self.envs = restored_envs
        self.parsed_envs_step = parsed
        self.commit_buffer_size = int(state["commit_buffer_size"])
        self.total_env_steps = int(state["total_env_steps"])
        self.total_game_steps = int(state["total_game_steps"])
        self.attempted_env_steps = int(state.get("attempted_env_steps", self.total_env_steps))
        self.failed_env_steps = int(state.get("failed_env_steps", 0))
        self.rejected_rollout_steps = int(state.get("rejected_rollout_steps", 0))
        self._terminal_info_queue = [dict(item) for item in state.get("terminal_info_queue", [])]
        records = state.get("training_deal_records", [])
        if not isinstance(records, list):
            raise ValueError("sampler training_deal_records must be a list")
        self._training_deal_records = [dict(item) for item in records]
        self._training_deal_hashes = {str(item["deal_sha256"]) for item in self._training_deal_records}
        raw_identities = state.get("episode_identities", {})
        if not isinstance(raw_identities, Mapping):
            raise ValueError("sampler episode_identities must be a mapping")
        self._episode_identities = {
            int(key): dict(value) for key, value in raw_identities.items()
        }
        self._last_context_configuration = None
        self._curriculum_iteration = int(state.get("curriculum_iteration", 0))
        raw_router = state.get("policy_router")
        if self.policy_router is None:
            if raw_router is not None:
                raise ValueError(
                    "checkpoint contains opponent-pool router state but the sampler is pool-disabled"
                )
        else:
            if not isinstance(raw_router, Mapping):
                raise ValueError(
                    "sampler is pool-enabled but the checkpoint lacks opponent-pool router state"
                )
            self.policy_router.load_state_dict(raw_router)
        declared_provenance = state.get("training_data_provenance")
        if declared_provenance != self.training_data_provenance():
            raise ValueError("sampler training data provenance mismatch")
        if not 0 <= self.commit_buffer_size <= (0 if self.commit_buffer is None else self.commit_buffer.batch_size):
            raise ValueError("invalid restored commit_buffer_size")
        self._failed = False
        self._failure_info = None
        self.agent_kv = None
        self._reset_mask_full = None
        self.last_step_rewards = None
        self.last_step_is_terminal = None

    def _ensure_agent_cache(self, agent_model: Any) -> None:
        agent_model = unwrap_distributed_module(agent_model)
        if agent_model is None:
            raise TypeError("agent_model must be a torch.nn.Module.")
        if self.agent_kv is not None:
            expected_batch = self.rollout_batch_size * self.P
            if self.agent_kv.batch_size == expected_batch and self.agent_kv.max_ctx_len == self.T:
                return

        decoder_config = agent_model.decoder.config
        kv_heads = (
            decoder_config.num_key_value_heads
            if decoder_config.num_key_value_heads is not None
            else decoder_config.num_attention_heads
        )
        head_dim = decoder_config.hidden_size // decoder_config.num_attention_heads
        cache_dtype = self._cache_dtype or next(agent_model.parameters()).dtype

        player_batch_size = self.rollout_batch_size * self.P
        self.agent_kv = KVCache(
            agent_model.decoder_n_layers,
            player_batch_size,
            max_ctx_len = self.T,
            num_heads = kv_heads,
            head_dim = head_dim,
            value_heads = kv_heads,
            value_head_dim = head_dim,
            dtype = cache_dtype,
            device = self.device,
        )
        self._reset_mask_full = torch.ones(
            self.agent_kv.batch_size,
            dtype = torch.bool,
            device = self.device,
        )

    def _rebuild_agent_cache(self, agent_model: Any) -> None:
        """Reconstruct the decoder KV cache from partial trajectories.

        Uses a single batched forward pass through the decoder instead of
        stepping one token at a time, replacing up to T sequential forward
        calls with one padded-sequence forward.
        """
        if self.agent_kv is None or self._reset_mask_full is None:
            raise RuntimeError("Agent KV cache must be initialized before rebuilding.")
        self.agent_kv.reset(self._reset_mask_full)

        num_steps = self.rollout_buffer["num_steps"]
        if not torch.any(num_steps > 0):
            return

        # Use the full horizon so the tensor shape is static across calls,
        # avoiding torch.compile recompilation on varying T.  The attention
        # mask zeros out padding positions, so correctness is preserved.
        rollout = TensorDict(
            {
                "num_steps": num_steps,
                "player_ids": self.rollout_buffer["player_ids"],
                "obs_tokens": self.rollout_buffer["obs_tokens"],
                "obs_channels": self.rollout_buffer["obs_channels"],
                "avail_tokens": self.rollout_buffer["avail_tokens"],
                "avail_weights": self.rollout_buffer["avail_weights"],
            },
            batch_size = [self.rollout_batch_size],
            device = self.device,
        )
        if self.config.trim_completed_padding:
            # Cache capacity and real histories stay unchanged. Only avoid prefill
            # computation on a globally padded suffix (compile is off in this pilot).
            end = int(num_steps.max().item())
            rollout = TensorDict(
                {name: (value[:, :end] if value.dim() >= 2 else value)
                 for name, value in rollout.items()},
                batch_size=rollout.batch_size, device=rollout.device,
            )
        agent_model(
            rollout = rollout,
            use_cache = True,
            past_key_values = self.agent_kv,
            cache_only = True,
        )

    def _commit_done(self, mask: torch.Tensor) -> None:
        mask = mask.to(dtype = torch.bool, device = self.device)
        num_done = int(mask.sum().item())
        if self.agent_kv is None:
            raise RuntimeError("Agent KV cache must be initialized before committing rollouts.")

        if self.commit_buffer is not None:
            completed = self.rollout_buffer[mask]
            self.commit_buffer[self.commit_buffer_size: self.commit_buffer_size + num_done] = completed
            self.commit_buffer_size += num_done

        self.rollout_buffer.reset(mask)
        self.agent_kv.reset(mask.repeat_interleave(self.P))
        if self.policy_router is not None:
            self.policy_router.on_commit(mask)

    @torch.inference_mode()
    def rollout_step(
        self,
        *,
        agent_model: Any,
        forced_action_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one transactional rollout step.

        ``on_policy`` mode accepts only temperature 1 and never accepts forced
        actions. Evaluation/test modes may use greedy, temperature transforms,
        or forced legal actions, but their transitions carry
        ``on_policy_valid=False`` and are rejected by PPO loss entry points.

        Failure strategy B is used: any exception after agent/cache execution
        poisons the sampler. Completed counters are not incremented, and later
        rollout/collect/cache calls fail with ``SamplerFailedStateError``.
        """
        self._require_healthy()
        if self.agent_kv is None or self._reset_mask_full is None:
            raise RuntimeError(
                "Sampler cache is not initialized. "
                "Call sampler.prepare_agent_cache(agent_model=...) before rollout_step()."
            )
        if forced_action_indices is not None and self.purpose == "on_policy":
            raise ValueError("forced_action_indices are forbidden for on_policy sampling.")

        forced: torch.Tensor | None = None
        if forced_action_indices is not None:
            forced = torch.as_tensor(forced_action_indices, dtype=torch.long, device=self.device)
            if tuple(forced.shape) != (self.rollout_batch_size,):
                raise ValueError(
                    "forced_action_indices must have shape [rollout_batch_size]: "
                    f"expected {(self.rollout_batch_size,)}, got {tuple(forced.shape)}."
                )
            action_count = self.parsed_envs_step["avail_weights"].shape[-1]
            if bool(((forced < 0) | (forced >= action_count)).any()):
                raise ValueError("forced_action_indices contains an out-of-range action index.")
            forced_legal = self.parsed_envs_step["avail_weights"][self._micro_buffer_range, forced] > 0
            if not bool(forced_legal.all()):
                raise ValueError("forced_action_indices must select legal actions in every slot.")

        if self.config.state_logger is not None:
            self.config.state_logger(self.envs, self.total_env_steps)

        self.attempted_env_steps += self.rollout_batch_size
        stage = "agent_forward"
        step_fields: dict[str, torch.Tensor] | None = None
        router_on_policy: torch.Tensor | None = None
        try:
            agent_started = time.perf_counter()
            step_rollout = TensorDict(
                {
                    "num_steps": torch.ones(self.rollout_batch_size, dtype=torch.long, device=self.device),
                    "player_ids": self.parsed_envs_step["player_ids"].unsqueeze(1),
                    "obs_tokens": self.parsed_envs_step["obs_tokens"].unsqueeze(1),
                    "obs_channels": self.parsed_envs_step["obs_channels"].unsqueeze(1),
                    "avail_tokens": self.parsed_envs_step["avail_tokens"].unsqueeze(1),
                    "avail_weights": self.parsed_envs_step["avail_weights"].unsqueeze(1),
                },
                batch_size=[self.rollout_batch_size],
                device=self.device,
            )
            per_env_output = agent_model(
                rollout=step_rollout,
                use_cache=True,
                past_key_values=self.agent_kv,
            )
            policy_probs = per_env_output.avail_probs.squeeze(1).float()
            if self.policy_router is not None:
                policy_probs, router_on_policy, step_fields = self.policy_router.policy_probabilities(
                    step_rollout=step_rollout,
                    parsed_envs_step=self.parsed_envs_step,
                    current_probabilities=policy_probs,
                )
            if self.config.timing_callback is not None:
                self.config.timing_callback(
                    "model_forward_with_cache", time.perf_counter() - agent_started,
                )
            sampling_started = time.perf_counter()
            avail_weights_f = self.parsed_envs_step["avail_weights"].float()
            avail_mask = avail_weights_f > 0
            if tuple(policy_probs.shape) != tuple(avail_weights_f.shape):
                raise ValueError("agent policy probability shape does not match available actions.")
            if not bool(torch.isfinite(policy_probs).all()):
                raise FloatingPointError("agent policy probabilities contain non-finite values.")

            log_probs = policy_probs.clamp(min=1e-9).log()
            log_prior = avail_weights_f.clamp(min=1e-9).log()
            raw_logits = log_probs - log_prior
            neg_inf = torch.finfo(raw_logits.dtype).min
            if forced is not None:
                action_taken = forced
                behavior_probs = torch.zeros_like(policy_probs).scatter_(1, action_taken.unsqueeze(1), 1.0)
            elif self._sampling_temperature == 0.0:
                action_taken = torch.argmax(raw_logits.masked_fill(~avail_mask, neg_inf), dim=1)
                behavior_probs = torch.zeros_like(policy_probs).scatter_(1, action_taken.unsqueeze(1), 1.0)
            elif self._sampling_temperature == 1.0:
                behavior_probs = policy_probs
                action_taken = torch.multinomial(behavior_probs, 1).squeeze(1)
            else:
                scaled_logits = (
                    raw_logits / self._sampling_temperature + log_prior
                ).masked_fill(~avail_mask, neg_inf)
                behavior_probs = torch.softmax(scaled_logits, dim=1)
                action_taken = torch.multinomial(behavior_probs, 1).squeeze(1)

            player_ids_step = self.parsed_envs_step["player_ids"].to(dtype=torch.long)
            chance_mask = player_ids_step < 0
            if torch.any(chance_mask):
                chance_weights = self.parsed_envs_step["avail_weights"].float()
                norm = chance_weights[chance_mask].sum(dim=1, keepdim=True).clamp(min=1e-9)
                chance_probs = chance_weights[chance_mask] / norm
                sampled = torch.multinomial(chance_probs, 1).squeeze(1)
                action_taken = action_taken.clone()
                action_taken[chance_mask] = sampled
                behavior_probs = behavior_probs.clone()
                behavior_probs[chance_mask] = chance_probs

            behavior_action_prob = behavior_probs[
                self._micro_buffer_range, action_taken
            ].to(dtype=torch.float32)
            on_policy_valid = torch.full(
                (self.rollout_batch_size,),
                self.purpose == "on_policy" and forced is None,
                dtype=torch.bool,
                device=self.device,
            )
            if router_on_policy is not None:
                on_policy_valid = on_policy_valid & router_on_policy.to(torch.bool)
            action_is_commit = self.parsed_envs_step["avail_is_commit"][
                self._micro_buffer_range, action_taken
            ].to(torch.bool)
            action_tokens = self.parsed_envs_step["avail_tokens"][
                self._micro_buffer_range, action_taken
            ]
            actions = self.game.actions_from_np(action_tokens.detach().cpu().numpy())
            self.parsed_envs_step.record_agent_outputs(
                policy_probs=policy_probs,
                behavior_action_prob=behavior_action_prob,
                on_policy_valid=on_policy_valid,
                action_taken=action_taken,
                step_fields=step_fields,
            )

            if self.config.timing_callback is not None:
                self.config.timing_callback(
                    "sampling_and_action_marshalling",
                    time.perf_counter() - sampling_started,
                )
            stage = "environment_step"
            environment_started = time.perf_counter()
            envs_step = self.envs.step(actions)
            if self.config.timing_callback is not None:
                self.config.timing_callback(
                    "environment_step", time.perf_counter() - environment_started,
                )
            conversion_started = time.perf_counter()
            next_parsed_envs_step = EnvStepBatch.from_env_step(envs_step, self.config)
            if self.config.timing_callback is not None:
                self.config.timing_callback(
                    "observation_conversion", time.perf_counter() - conversion_started,
                )
            bookkeeping_started = time.perf_counter()
            outcome = next_parsed_envs_step.outcome
            step_done_mask = outcome["is_terminal"].to(dtype=torch.bool)
            transition = self.parsed_envs_step.build_transition(outcome)

            stage = "rollout_buffer_write"
            self.rollout_buffer.add_step(self._micro_buffer_range, transition)
            if bool(step_done_mask.any().item()):
                stage = "completed_trajectory_commit"
                self._commit_done(step_done_mask)
                completed_indices = step_done_mask.nonzero().flatten().tolist()
                for index in completed_indices:
                    self._terminal_info_queue.append(dict(envs_step.infos[index]))
                if self.purpose == "on_policy" and self._data_domain == "training":
                    self._record_training_deals(completed_indices)

            if self.config.timing_callback is not None:
                self.config.timing_callback(
                    "trajectory_buffer_and_commit",
                    time.perf_counter() - bookkeeping_started,
                )
            stage = "publish"
            self.parsed_envs_step = next_parsed_envs_step
            self.last_step_rewards = outcome["rewards"].to(
                dtype=torch.float32, device=self.device
            ).clone()
            self.last_step_is_terminal = step_done_mask.clone()
            self.total_env_steps += self.rollout_batch_size
            self.total_game_steps += int(action_is_commit.sum().item())
            return step_done_mask
        except BaseException as error:
            self.failed_env_steps += self.rollout_batch_size
            self.rejected_rollout_steps += 1
            self._mark_failed(stage=stage, error=error)
            raise


__all__ = [
    "SAMPLER_CONTRACT_SCHEMA",
    "SAMPLER_STATE_SCHEMA",
    "PRE_PROVENANCE_SAMPLER_STATE_SCHEMA",
    "TRAINING_DATA_PROVENANCE_SCHEMA",
    "Sampler",
    "SamplerConfig",
    "SamplerFailedStateError",
]
