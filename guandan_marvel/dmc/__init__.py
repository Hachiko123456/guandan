# SPDX-License-Identifier: MIT
"""DMC (Deep Monte-Carlo) training package.

Stage 1 only registers the versioned runtime configuration and its semantic
identifiers.  The candidate enumerator, action features, Q head, replay buffer,
sampler and learner arrive in later stages; nothing here trains or samples.
"""

from __future__ import annotations

from dataclasses import dataclass

# ``algo_name="dmc"`` must always be paired with this semantics profile and
# these two version strings.  They are defined here (rather than in
# ``guandan_marvel.config``) so the config module can import them without a
# cycle; ``config`` re-exports them as the public access point.
DMC_TRAINING_SEMANTICS_PROFILE = "dmc_v1"
DMC_ACTION_FEATURE_VERSION = "GD-DMC-ACTION-1"
DMC_CHECKPOINT_SCHEMA = "dmc-v1"


@dataclass
class DMCRuntimeConfig:
    """DMC runtime hyperparameters, data-only for stage 1.

    The defaults are the locked values from the DMC migration plan
    (``docs/dmc_prompts/00_DMC_MIGRATION_PLAN.md`` section 2).  DMC never reads
    ``PPOConfig``: there is no advantage estimator, clipping, entropy bonus or
    policy regularizer, so none of those fields apply here.
    """

    buffer_capacity: int = 500_000
    batch_size: int = 1024
    #: Stage 07B: ``batch_size`` is the optimizer-step batch (locked 1024);
    #: ``accumulation_steps`` splits it into that many micro-batches (4 x 256)
    #: so the forward peak memory stays at one micro-batch.
    accumulation_steps: int = 4
    #: Stage 07B observation window split: head 128 + tail 896 = 1024 tokens.
    obs_head_tokens: int = 128
    games_per_cycle: int = 256
    gradient_replay_ratio: float = 1.0
    epsilon_start: float = 0.10
    epsilon_final: float = 0.02
    epsilon_decay_samples: int = 5_000_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    action_feature_version: str = DMC_ACTION_FEATURE_VERSION
    checkpoint_schema: str = DMC_CHECKPOINT_SCHEMA


__all__ = [
    "DMC_ACTION_FEATURE_VERSION",
    "DMC_CHECKPOINT_SCHEMA",
    "DMC_TRAINING_SEMANTICS_PROFILE",
    "DMCRuntimeConfig",
]
