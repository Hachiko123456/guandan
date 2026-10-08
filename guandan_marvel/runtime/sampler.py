# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/runtime/sampler.py.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch
from guandan_marvel.model.tensordict import TensorDict

from guandan_marvel.buffer import BufferConfig, FieldSpecLike, TrajectoryBuffer, trajectory_field_specs
from guandan_marvel.env_batch import EnvStepBatch
from guandan_marvel.model import KVCache
from guandan_marvel.utils import unwrap_distributed_module


@dataclass(frozen = True)
class SamplerConfig:
    """Configuration subset required by the sampler and trajectory buffers."""

    game: Any
    device: torch.device | str
    rollout_batch_size: int
    collect_batch_size: int
    sampling_temperature: float = 1.0
    cache_dtype: torch.dtype | None = None
    field_specs: Optional[Mapping[str, FieldSpecLike]] = None
    env_seed: int | None = 0
    state_logger: Optional[Callable[[Any, int], None]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        temperature = float(self.sampling_temperature)
        if not math.isfinite(temperature):
            raise ValueError("sampling_temperature must be finite.")
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
    ) -> None:
        self.config = config
        self.game = config.game
        self.device = config.device
        self._cache_dtype = config.cache_dtype
        self._sampling_temperature = float(config.sampling_temperature)
        self._env_seed = config.env_seed

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
        if parsed_envs_step is None:
            seed = self._env_seed if self._env_seed is not None else 0
            reset_result = self.envs.reset(seed)
            parsed_envs_step = EnvStepBatch.from_env_step(reset_result, config)
        self.parsed_envs_step = parsed_envs_step

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

        self.agent_kv: KVCache | None = None
        self._reset_mask_full: torch.Tensor | None = None
        self._micro_buffer_range = torch.arange(
            self.rollout_batch_size,
            dtype = torch.long,
            device = self.device,
        )

    def prepare_agent_cache(self, *, agent_model: Any) -> None:
        """Ensure the decoder KV cache is initialized and aligned with rollout history."""
        with torch.inference_mode():
            self._ensure_agent_cache(agent_model)
            self._rebuild_agent_cache(agent_model)

    def collect(
        self,
        *,
        agent_model: Any,
    ) -> TensorDict:
        """Roll out trajectories until `collect_batch_size` games finish and return that batch."""
        if not self._use_commit_buffer or self.commit_buffer is None:
            raise RuntimeError(
                "Sampler.collect() requires collect_batch_size > 0 (commit mode). "
                "Use rollout_step() and read sampler.envs/parsed_envs_step directly in no-commit mode."
            )
        with torch.inference_mode():
            self.prepare_agent_cache(agent_model = agent_model)
            while self.commit_buffer_size < self.collect_batch_size:
                self.rollout_step(agent_model = agent_model)

        collect_batch = self.commit_buffer[: self.collect_batch_size].clone()
        self.consume(self.collect_batch_size)
        return collect_batch

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

    def rollout_step(
        self,
        *,
        agent_model: Any,
        forced_action_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance environments by one step and return the per-env is_terminal mask for that step.

        Invariant (critical): ``self.parsed_envs_step`` MUST represent the game state at
        trajectory position ``self.rollout_buffer["num_steps"]`` — i.e. the next slot to
        be written. If the env has been stepped PAST this position before handoff (e.g.
        via an external ``envs.step(...)`` between prefill and the first ``rollout_step``),
        buffer positions drift by one and every downstream policy query operates on a
        misaligned state. This was the root cause of the 3-pt search-rollout calibration
        gap against the eval path; see ``kagami/search/rollout_session.py``.

        Args:
            agent_model: trained MARVEL agent module.
            forced_action_indices: optional tensor of shape ``[rollout_batch_size]`` with
                per-env action indices into ``parsed_envs_step["avail_tokens"]``. When
                provided, player-node action selection bypasses the sampling-temperature
                logic and uses these indices directly. Chance-node sampling is still
                performed via ``torch.multinomial`` on ``avail_weights`` (forced indices
                do NOT apply at chance nodes). Intended for driving a predetermined root
                action from within the rollout loop itself — the only correct way to
                preserve the prefill↔parsed_envs_step invariant when a root action needs
                to be applied before the main policy-driven rollout.
        """
        if self.agent_kv is None or self._reset_mask_full is None:
            raise RuntimeError(
                "Sampler cache is not initialized. "
                "Call sampler.prepare_agent_cache(agent_model=...) before rollout_step()."
            )

        if self.config.state_logger is not None:
            self.config.state_logger(self.envs, self.total_env_steps)

        self.total_env_steps += self.rollout_batch_size

        step_rollout = TensorDict(
            {
                "num_steps": torch.ones(self.rollout_batch_size, dtype = torch.long, device = self.device),
                "player_ids": self.parsed_envs_step["player_ids"].unsqueeze(1),
                "obs_tokens": self.parsed_envs_step["obs_tokens"].unsqueeze(1),
                "obs_channels": self.parsed_envs_step["obs_channels"].unsqueeze(1),
                "avail_tokens": self.parsed_envs_step["avail_tokens"].unsqueeze(1),
                "avail_weights": self.parsed_envs_step["avail_weights"].unsqueeze(1),
            },
            batch_size = [self.rollout_batch_size],
            device = self.device,
        )
        per_env_output = agent_model(
            rollout = step_rollout,
            use_cache = True,
            past_key_values = self.agent_kv,
        )

        avail_probs = per_env_output.avail_probs.squeeze(1).float()

        # Prior-preserving temperature: softmax(raw_logits / temp + log_prior).
        # Reconstruct raw logits (up to constant) by subtracting the log-prior,
        # scale only the learned component by temperature, then re-add the prior.
        # At temp=1 this reduces to the original probs; as temp→∞ it converges
        # to the prior distribution (avail_weights / sum) rather than flat uniform.
        avail_weights_f = self.parsed_envs_step["avail_weights"].float()
        avail_mask = avail_weights_f > 0
        log_probs = avail_probs.clamp(min = 1e-9).log()
        log_prior = avail_weights_f.clamp(min = 1e-9).log()
        raw_logits = log_probs - log_prior
        neg_inf = torch.finfo(raw_logits.dtype).min
        if forced_action_indices is not None:
            forced = forced_action_indices.to(dtype = torch.long, device = self.device)
            if tuple(forced.shape) != (self.rollout_batch_size,):
                raise ValueError(
                    "forced_action_indices must have shape [rollout_batch_size]: "
                    f"expected {(self.rollout_batch_size,)}, got {tuple(forced.shape)}."
                )
            action_taken = forced
        elif self._sampling_temperature <= 0.0:
            action_taken = torch.argmax(raw_logits.masked_fill(~avail_mask, neg_inf), dim = 1)
        else:
            scaled_logits = (raw_logits / self._sampling_temperature + log_prior).masked_fill(~avail_mask, neg_inf)
            sampling_probs = torch.softmax(scaled_logits, dim = 1)
            action_taken = torch.multinomial(sampling_probs, 1).squeeze(1)

        # Chance / environment steps are not agent decisions: sample from the provided
        # action weights (which represent a probability mass function).
        player_ids_step = self.parsed_envs_step["player_ids"].to(dtype = torch.long)
        chance_mask = player_ids_step < 0
        avail_probs_used = avail_probs
        if torch.any(chance_mask):
            chance_weights = self.parsed_envs_step["avail_weights"].float()
            sampled = torch.multinomial(chance_weights[chance_mask], 1).squeeze(1)
            action_taken = action_taken.clone()
            action_taken[chance_mask] = sampled

            # Record the normalized chance distribution (useful for downstream metrics/debugging).
            norm = chance_weights[chance_mask].sum(dim = 1, keepdim = True).clamp(min = 1e-9)
            avail_probs_used = avail_probs.clone()
            avail_probs_used[chance_mask] = chance_weights[chance_mask] / norm

        action_is_commit = self.parsed_envs_step["avail_is_commit"][
            self._micro_buffer_range,
            action_taken,
        ].to(torch.bool)
        self.total_game_steps += int(action_is_commit.sum().item())
        action_tokens = self.parsed_envs_step["avail_tokens"][self._micro_buffer_range, action_taken]
        actions = self.game.actions_from_np(action_tokens.detach().cpu().numpy())
        self.parsed_envs_step.record_agent_outputs(
            avail_probs = avail_probs_used,
            action_taken = action_taken,
        )

        envs_step = self.envs.step(actions)
        next_parsed_envs_step = EnvStepBatch.from_env_step(envs_step, self.config)
        outcome = next_parsed_envs_step.outcome

        step_done_mask = outcome["is_terminal"].to(dtype = torch.bool)
        self.last_step_rewards = outcome["rewards"].to(dtype = torch.float32, device = self.device).clone()
        self.last_step_is_terminal = step_done_mask.clone()
        transition = self.parsed_envs_step.build_transition(outcome)
        self.rollout_buffer.add_step(self._micro_buffer_range, transition)
        if bool(step_done_mask.any().item()):
            self._commit_done(step_done_mask)

        self.parsed_envs_step = next_parsed_envs_step
        return step_done_mask
