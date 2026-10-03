"""Training-facing import path for the A05 policy/value model.

This package intentionally exports the model layer only; PPO trainers and
checkpointing are outside the scope of this module.
"""
from __future__ import annotations

from ..model import (
    Policy,
    PolicyValueNet,
    PROTOCOL_MAX_LEGAL_NEXT_TOKENS,
    PROTOCOL_MAX_OBSERVATION_TOKENS,
    PROTOCOL_PAD_TOKEN_ID,
    PROTOCOL_STATE_CHANNELS,
    PROTOCOL_TOKEN_VOCAB_SIZE,
)

__all__ = [
    "Policy",
    "PolicyValueNet",
    "PROTOCOL_MAX_LEGAL_NEXT_TOKENS",
    "PROTOCOL_MAX_OBSERVATION_TOKENS",
    "PROTOCOL_PAD_TOKEN_ID",
    "PROTOCOL_STATE_CHANNELS",
    "PROTOCOL_TOKEN_VOCAB_SIZE",
]
