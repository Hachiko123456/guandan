"""Small token-level policy/value model for the A04 GuanDan observation contract.

This module intentionally contains only the neural model layer.  It does not
implement a rollout collector, PPO update, or checkpoint format.

The model consumes the seven-field A04 policy projection as four tensors:

* ``observation_tokens``: ``long``/integer ``[B, T]`` token history;
* ``state_channels``: floating point ``[B, C]`` channels;
* ``legal_next_tokens``: ``long``/integer ``[B, A]`` candidate token IDs; and
* ``legal_mask``: boolean ``[B, A]`` validity mask.

The default dimensions are the GD-ENV-0.1 dimensions (vocabulary 256,
channels 256, up to 4096 history tokens and 256 legal rows).  ``hidden_dim``
can be reduced for a quick local smoke test without changing that wire
contract.
"""
from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Categorical


PROTOCOL_TOKEN_VOCAB_SIZE: Final = 256
PROTOCOL_PAD_TOKEN_ID: Final = 0
PROTOCOL_MAX_OBSERVATION_TOKENS: Final = 4096
PROTOCOL_MAX_LEGAL_NEXT_TOKENS: Final = 256
PROTOCOL_STATE_CHANNELS: Final = 256


class PolicyValueNet(nn.Module):
    """A compact candidate-scoring policy with a scalar value head.

    ``forward`` returns ``(masked_logits, values)``.  ``masked_logits`` has
    shape ``[B, A]`` and contains ``-inf`` at every invalid legal-row
    position; ``values`` has shape ``[B]``.  Consequently, constructing a
    categorical distribution from the returned logits cannot assign mass to an
    invalid candidate (provided each row has at least one legal candidate, as
    guaranteed by a live A04 observation).

    ``sample_actions`` returns actual token IDs, not candidate indices.  In
    training mode it samples from the masked categorical distribution; in eval
    mode its default is deterministic argmax.  To calculate the sampled
    token's PPO log probability, call ``log_prob(logits, actions,
    legal_next_tokens=legal_next_tokens)``.  ``log_prob`` also accepts
    candidate indices when ``legal_next_tokens`` is omitted.
    """

    def __init__(
        self,
        *,
        token_vocab_size: int = PROTOCOL_TOKEN_VOCAB_SIZE,
        num_state_channels: int = PROTOCOL_STATE_CHANNELS,
        hidden_dim: int = 128,
        pad_token_id: int = PROTOCOL_PAD_TOKEN_ID,
        max_observation_tokens: int = PROTOCOL_MAX_OBSERVATION_TOKENS,
        max_legal_next_tokens: int = PROTOCOL_MAX_LEGAL_NEXT_TOKENS,
        # Friendly aliases keep the small public API convenient while the
        # protocol-oriented names above remain the canonical ones.
        vocab_size: int | None = None,
        state_dim: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size is not None:
            if token_vocab_size != PROTOCOL_TOKEN_VOCAB_SIZE and token_vocab_size != vocab_size:
                raise ValueError("token_vocab_size and vocab_size disagree")
            token_vocab_size = vocab_size
        if state_dim is not None:
            if num_state_channels != PROTOCOL_STATE_CHANNELS and num_state_channels != state_dim:
                raise ValueError("num_state_channels and state_dim disagree")
            num_state_channels = state_dim
        for name, value in (
            ("token_vocab_size", token_vocab_size),
            ("num_state_channels", num_state_channels),
            ("hidden_dim", hidden_dim),
            ("pad_token_id", pad_token_id),
            ("max_observation_tokens", max_observation_tokens),
            ("max_legal_next_tokens", max_legal_next_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0 and name not in {"pad_token_id"}:
                raise ValueError(f"{name} must be positive")
        if not 0 <= pad_token_id < token_vocab_size:
            raise ValueError("pad_token_id must be inside the token vocabulary")

        self.token_vocab_size = token_vocab_size
        self.vocab_size = token_vocab_size
        self.num_state_channels = num_state_channels
        self.state_dim = num_state_channels
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.max_observation_tokens = max_observation_tokens
        self.max_legal_next_tokens = max_legal_next_tokens

        self.token_embedding = nn.Embedding(
            token_vocab_size,
            hidden_dim,
            padding_idx=pad_token_id,
        )
        self.history_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.channel_projection = nn.Sequential(
            nn.LayerNorm(num_state_channels),
            nn.Linear(num_state_channels, hidden_dim),
            nn.Tanh(),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.logit_scale = hidden_dim ** -0.5

        # A zero embedding is useful for the padded wire tail.  The embedding
        # module's padding_idx already prevents that row receiving gradients.
        with torch.no_grad():
            self.token_embedding.weight[pad_token_id].zero_()

    def _validate_inputs(
        self,
        obs_tokens: Tensor,
        state_channels: Tensor,
        legal_next_tokens: Tensor,
        legal_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tensors = {
            "obs_tokens": obs_tokens,
            "state_channels": state_channels,
            "legal_next_tokens": legal_next_tokens,
            "legal_mask": legal_mask,
        }
        for name, value in tensors.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.device != obs_tokens.device:
                raise ValueError("all model inputs must be on the same device")
        if obs_tokens.ndim != 2:
            raise ValueError("obs_tokens must have shape [B, T]")
        if state_channels.ndim != 2:
            raise ValueError("state_channels must have shape [B, C]")
        if legal_next_tokens.ndim != 2:
            raise ValueError("legal_next_tokens must have shape [B, A]")
        if legal_mask.ndim != 2:
            raise ValueError("legal_mask must have shape [B, A]")
        batch = obs_tokens.shape[0]
        if state_channels.shape[0] != batch or legal_next_tokens.shape[0] != batch or legal_mask.shape[0] != batch:
            raise ValueError("all inputs must have the same batch dimension")
        if state_channels.shape[1] != self.num_state_channels:
            raise ValueError(
                f"state_channels must have width {self.num_state_channels}, "
                f"got {state_channels.shape[1]}"
            )
        if not 1 <= obs_tokens.shape[1] <= self.max_observation_tokens:
            raise ValueError("obs_tokens has an unsupported history length")
        if not 1 <= legal_next_tokens.shape[1] <= self.max_legal_next_tokens:
            raise ValueError("legal_next_tokens has an unsupported candidate width")
        if legal_mask.shape != legal_next_tokens.shape:
            raise ValueError("legal_mask must have the same shape as legal_next_tokens")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if obs_tokens.dtype not in integer_dtypes:
            raise TypeError("obs_tokens must use an integer dtype")
        if legal_next_tokens.dtype not in integer_dtypes:
            raise TypeError("legal_next_tokens must use an integer dtype")
        if not state_channels.is_floating_point():
            raise TypeError("state_channels must use a floating point dtype")
        if legal_mask.dtype != torch.bool:
            raise TypeError("legal_mask must use torch.bool")

        obs_tokens = obs_tokens.to(dtype=torch.long)
        legal_next_tokens = legal_next_tokens.to(dtype=torch.long)
        if torch.any(obs_tokens < 0) or torch.any(obs_tokens >= self.token_vocab_size):
            raise ValueError("obs_tokens contains an ID outside the token vocabulary")
        if torch.any(legal_next_tokens < 0) or torch.any(legal_next_tokens >= self.token_vocab_size):
            raise ValueError("legal_next_tokens contains an ID outside the token vocabulary")
        state_channels = state_channels.to(dtype=self.token_embedding.weight.dtype)
        return obs_tokens, state_channels, legal_next_tokens, legal_mask

    def forward(
        self,
        obs_tokens: Tensor,
        state_channels: Tensor,
        legal_next_tokens: Tensor,
        legal_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score the legal-token rows and estimate the state value.

        Args:
            obs_tokens: Batched A04 ``observation_tokens``, shaped ``[B, T]``.
                PAD tokens are excluded from mean pooling.
            state_channels: A04 ``state_channels``, shaped ``[B, C]``.
            legal_next_tokens: A04 candidate token IDs, shaped ``[B, A]``.
            legal_mask: A04 ``legal_next_mask``, shaped ``[B, A]``.

        Returns:
            ``(masked_logits, values)`` with shapes ``[B, A]`` and ``[B]``.
            Invalid candidates are exactly ``-inf`` in ``masked_logits``.
            Inactive A04 batch rows (all-false masks) may pass through forward,
            but must be filtered using ``BatchObservation.present`` before
            calling sampling/log-probability/entropy helpers, which reject them.
        """
        obs_tokens, state_channels, legal_next_tokens, legal_mask = self._validate_inputs(
            obs_tokens, state_channels, legal_next_tokens, legal_mask
        )

        token_embeddings = self.token_embedding(obs_tokens)
        history_mask = obs_tokens.ne(self.pad_token_id).unsqueeze(-1)
        history_count = history_mask.sum(dim=1).clamp_min(1).to(token_embeddings.dtype)
        pooled_history = (token_embeddings * history_mask).sum(dim=1) / history_count
        history = self.history_projection(pooled_history)
        channels = self.channel_projection(state_channels)
        context = self.context_projection(history + channels)

        candidates = self.candidate_projection(self.token_embedding(legal_next_tokens))
        query = self.policy_query(context).unsqueeze(1)
        raw_logits = (candidates * query).sum(dim=-1) * self.logit_scale
        masked_logits = raw_logits.masked_fill(~legal_mask, -torch.inf)
        values = self.value_head(context).squeeze(-1)
        return masked_logits, values

    @staticmethod
    def _validate_logits(logits: Tensor) -> None:
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError("logits must have shape [B, A]")
        if logits.shape[1] < 1:
            raise ValueError("logits must contain at least one candidate")

    @staticmethod
    def _require_legal(logits: Tensor) -> None:
        if torch.any(torch.isnan(logits) | torch.isposinf(logits)):
            raise ValueError("logits must be finite or negative infinity")
        if torch.any(~torch.isfinite(logits).any(dim=1)):
            raise ValueError("each batch row must contain at least one legal candidate")

    @classmethod
    def distribution(cls, logits: Tensor) -> Categorical:
        """Build a safe categorical distribution from masked forward logits."""
        cls._validate_logits(logits)
        cls._require_legal(logits)
        return Categorical(logits=logits)

    def sample_actions(
        self,
        obs_tokens: Tensor,
        state_channels: Tensor,
        legal_next_tokens: Tensor,
        legal_mask: Tensor,
        *,
        deterministic: bool | None = None,
    ) -> Tensor:
        """Sample or greedily select one *token ID* per batch row.

        ``deterministic=None`` follows the module mode: training mode samples,
        while ``eval()`` selects the highest masked logit.  Passing an explicit
        boolean overrides that behavior.  The return shape is ``[B]`` and each
        value is copied from a ``legal_next_tokens`` position whose mask is
        true, never from an invalid row.
        """
        logits, _ = self.forward(obs_tokens, state_channels, legal_next_tokens, legal_mask)
        self._require_legal(logits)
        if deterministic is None:
            deterministic = not self.training
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be bool or None")
        indices = logits.argmax(dim=-1) if deterministic else self.distribution(logits).sample()
        return legal_next_tokens.to(dtype=torch.long).gather(1, indices.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _token_indices(
        actions: Tensor,
        legal_next_tokens: Tensor,
    ) -> Tensor:
        if actions.ndim != 1 or legal_next_tokens.ndim != 2 or actions.shape[0] != legal_next_tokens.shape[0]:
            raise ValueError("actions must have shape [B] and match legal_next_tokens")
        action_tokens = actions.to(dtype=legal_next_tokens.dtype).unsqueeze(1)
        matches = legal_next_tokens.eq(action_tokens)
        if torch.any(matches.sum(dim=1) != 1):
            raise ValueError("each action token must identify exactly one legal candidate")
        return matches.to(dtype=torch.long).argmax(dim=1)

    def log_prob(
        self,
        masked_logits: Tensor,
        actions: Tensor,
        *,
        legal_next_tokens: Tensor | None = None,
    ) -> Tensor:
        """Return PPO log probabilities with shape ``[B]``.

        By default ``actions`` are candidate indices in ``0..A-1``.  Pass the
        A04 ``legal_next_tokens`` array to interpret ``actions`` as the actual
        sampled token IDs returned by ``sample_actions``.  Invalid candidate
        indices and masked-out rows raise ``ValueError`` instead of silently
        assigning probability to an illegal action.
        """
        self._validate_logits(masked_logits)
        self._require_legal(masked_logits)
        if not isinstance(actions, Tensor) or actions.ndim != 1 or actions.shape[0] != masked_logits.shape[0]:
            raise ValueError("actions must have shape [B] matching masked_logits")
        if actions.dtype not in (torch.int32, torch.int64):
            raise TypeError("actions must use torch.int32 or torch.int64")
        if actions.device != masked_logits.device:
            raise ValueError("actions and masked_logits must share a device")
        if legal_next_tokens is None:
            indices = actions.to(dtype=torch.long)
        else:
            if not isinstance(legal_next_tokens, Tensor) or legal_next_tokens.shape != masked_logits.shape:
                raise ValueError("legal_next_tokens must match masked_logits shape [B, A]")
            if legal_next_tokens.dtype not in (torch.int32, torch.int64):
                raise TypeError("legal_next_tokens must use torch.int32 or torch.int64")
            if legal_next_tokens.device != masked_logits.device:
                raise ValueError("legal_next_tokens and masked_logits must share a device")
            indices = self._token_indices(actions, legal_next_tokens.to(dtype=torch.long))
        if torch.any(indices < 0) or torch.any(indices >= masked_logits.shape[1]):
            raise ValueError("action index is outside the candidate dimension")
        selected = masked_logits.gather(1, indices.unsqueeze(-1)).squeeze(-1)
        if torch.any(~torch.isfinite(selected)):
            raise ValueError("action selects an invalid masked candidate")
        return F.log_softmax(masked_logits, dim=-1).gather(1, indices.unsqueeze(-1)).squeeze(-1)

    @classmethod
    def entropy(cls, masked_logits: Tensor) -> Tensor:
        """Return masked categorical entropy with shape ``[B]``."""
        cls._validate_logits(masked_logits)
        cls._require_legal(masked_logits)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()
        # Mask BEFORE multiplication: 0 * -inf is NaN, including in backward.
        finite_log_probs = log_probs.masked_fill(torch.isneginf(log_probs), 0.0)
        return -(probs * finite_log_probs).sum(dim=-1)


# A short alias is useful in tiny local training scripts without changing the
# explicit PolicyValueNet name used by callers and tests.
Policy = PolicyValueNet


__all__ = [
    "Policy",
    "PolicyValueNet",
    "PROTOCOL_MAX_LEGAL_NEXT_TOKENS",
    "PROTOCOL_MAX_OBSERVATION_TOKENS",
    "PROTOCOL_PAD_TOKEN_ID",
    "PROTOCOL_STATE_CHANNELS",
    "PROTOCOL_TOKEN_VOCAB_SIZE",
]
