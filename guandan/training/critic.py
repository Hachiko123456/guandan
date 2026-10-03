"""Separate centralized action-value critic; never an actor input adapter."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..model import PROTOCOL_STATE_CHANNELS, PROTOCOL_TOKEN_VOCAB_SIZE


OWNERSHIP_CHANNELS = 4 * 108
CRITIC_STATE_DIM = OWNERSHIP_CHANNELS + PROTOCOL_STATE_CHANNELS  # 432 + 256 = 688


class CentralQCritic(nn.Module):
    """Candidate-conditioned team-zero Q, expanded as [q, -q, q, -q].

    ``critic_state[B,688]`` contains four 108-card ownership one-hot vectors
    followed by 256 policy channels. This training-only privileged state must
    never be sent to PolicyValueNet. This module shares no actor parameters.
    ``legal_tokens[B,A]`` are candidate token IDs (padding is allowed); the
    caller excludes invalid candidates with the policy's separate legal mask.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be an integer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.state_dim = CRITIC_STATE_DIM
        self.state_encoder = nn.Sequential(
            nn.Linear(CRITIC_STATE_DIM, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.action_embedding = nn.Embedding(PROTOCOL_TOKEN_VOCAB_SIZE, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, critic_state: Tensor, legal_tokens: Tensor) -> Tensor:
        """Evaluate all candidate actions, returning finite Q[B,A,4]."""
        if not isinstance(critic_state, Tensor) or not critic_state.is_floating_point():
            raise TypeError("critic_state must be a floating-point tensor")
        if critic_state.ndim != 2 or critic_state.shape[-1] != CRITIC_STATE_DIM:
            raise ValueError("critic_state must have shape [B, 688]")
        if not isinstance(legal_tokens, Tensor) or legal_tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("legal_tokens must be an int32 or int64 tensor")
        if (legal_tokens.ndim != 2 or legal_tokens.shape[0] != critic_state.shape[0]
                or legal_tokens.shape[1] == 0 or critic_state.shape[0] == 0):
            raise ValueError("legal_tokens must have nonempty shape [B, A] matching critic_state")
        if legal_tokens.device != critic_state.device:
            raise ValueError("critic_state and legal_tokens must share a device")
        if ((legal_tokens < 0) | (legal_tokens >= PROTOCOL_TOKEN_VOCAB_SIZE)).any():
            raise ValueError("legal_tokens must be inside the token vocabulary")
        if not torch.isfinite(critic_state).all():
            raise ValueError("critic_state must be finite")
        state = self.state_encoder(critic_state).unsqueeze(1)
        actions = self.action_embedding(legal_tokens.long())
        joint = torch.cat((state.expand(-1, actions.shape[1], -1), actions), dim=-1)
        q = self.q_head(joint).squeeze(-1)
        return torch.stack((q, -q, q, -q), dim=-1)


__all__ = ["CentralQCritic", "CRITIC_STATE_DIM", "OWNERSHIP_CHANNELS"]
