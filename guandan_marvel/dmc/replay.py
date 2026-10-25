# SPDX-License-Identifier: MIT
"""DMC flat FIFO replay buffer (stage 3).

This is deliberately *not* :class:`guandan_marvel.buffer.TrajectoryBuffer`: DMC
stores one candidate-level decision per row instead of a dense
``[max_horizon, players, ...]`` trajectory.  Rows are pre-allocated once, the
write pointer wraps, and the oldest row is overwritten silently by design
(FIFO eviction), exactly like DouZero/FableDan's flat ring.

Row layout (all arrays are pre-allocated, length ``capacity``):

====================  =========================  ===================
field                 shape / dtype              notes
====================  =========================  ===================
``obs_tokens``        ``int32[capacity, T_max]`` packed token row, padded
``obs_token_lengths`` ``int32[capacity]``         true token length per row
``obs_channels``      ``uint8[capacity, C]``     actor state channels
``action_feats``      ``float32[capacity, 64]``  GD-DMC-ACTION-1, chosen action
``z``                 ``float32[capacity]``      team rank score / 3, gamma=1
``seat``              ``int8[capacity]``         acting seat (provenance)
``game_id``           ``int64[capacity]``        game provenance
====================  =========================  ===================

Sampling is uniform without replacement from the live prefix (``len``), driven
by an explicit ``numpy.random.Generator`` so a seed reproduces the same index
sequence.  There is no prioritisation and no target network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from guandan_marvel.dmc.candidates import ACTION_FEATURE_DIM

OBS_TOKENS_DTYPE = np.int32
OBS_TOKEN_LENGTHS_DTYPE = np.int32
OBS_CHANNELS_DTYPE = np.uint8
ACTION_FEATS_DTYPE = np.float32
Z_DTYPE = np.float32
SEAT_DTYPE = np.int8
GAME_ID_DTYPE = np.int64

#: The locked stage-3 default state-channel width.  ``DMCAgent`` derives its
#: width from ``cfg.game.num_channels``; stage 4 must construct the replay with
#: the same width as the game adapter it feeds (the shipped Guandan adapter is
#: 512-wide, so this default is a stage-3 specification value, not a claim that
#: the real game is 256-wide).
DEFAULT_CHANNEL_DIM = 256


def _as_integral(value: Any, name: str) -> int:
    """Return ``value`` as a Python int without ever truncating.

    ``int(2.9)`` and ``np.asarray((2**40,), dtype=np.int32)`` both silently
    change the value, so every integral field is checked in its original
    domain first.  Bools are rejected (``isinstance(True, int)`` is true);
    ``2.0`` is accepted because it is exactly integral, while ``2.9``,
    ``nan`` and ``inf`` are not.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number) or number % 1.0 != 0.0:
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(number)
    raise ValueError(f"{name} must be an integer, got {type(value).__name__}")


def _as_finite_float(value: Any, name: str) -> float:
    """Convert to a finite Python float, rejecting overflow and non-numbers."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _as_finite_float32(value: Any, name: str) -> np.float32:
    """Narrow to float32 only after proving the narrowed value stays finite."""
    number = _as_finite_float(value, name)
    with np.errstate(over="ignore"):
        narrowed = np.float32(number)
    if not np.isfinite(narrowed):
        raise ValueError(f"{name} must be finite and representable in float32")
    return narrowed


@dataclass(frozen=True)
class DMCTransition:
    """One candidate-level decision.

    ``obs_tokens`` is the *packed* observation token row (length 1..T_max), not
    a padded ``[T]`` vector.  The remaining fields describe the single action
    that was actually played; the learner scores exactly that candidate.
    """

    obs_tokens: tuple[int, ...]
    obs_channels: np.ndarray
    action_feats: np.ndarray
    z: float
    seat: int
    game_id: int


@dataclass(frozen=True)
class DMCReplayBatch:
    """A sampled batch; arrays are copies owned by the caller."""

    obs_tokens: np.ndarray
    obs_token_lengths: np.ndarray
    obs_channels: np.ndarray
    action_feats: np.ndarray
    z: np.ndarray
    seat: np.ndarray
    game_id: np.ndarray

    def __len__(self) -> int:
        return int(self.z.shape[0])


class DMCReplay:
    """Pre-allocated flat FIFO replay buffer for DMC decisions."""

    def __init__(
        self,
        capacity: int,
        *,
        max_obs_tokens: int,
        channel_dim: int = DEFAULT_CHANNEL_DIM,
        action_feature_dim: int = ACTION_FEATURE_DIM,
        num_players: int = 4,
        pad_token: int = 0,
        num_tokens: int | None = None,
        seed: int | None = None,
    ) -> None:
        capacity = int(capacity)
        max_obs_tokens = int(max_obs_tokens)
        channel_dim = int(channel_dim)
        action_feature_dim = int(action_feature_dim)
        num_players = int(num_players)
        pad_token = int(pad_token)
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_obs_tokens <= 0:
            raise ValueError("max_obs_tokens must be positive")
        if channel_dim <= 0:
            raise ValueError("channel_dim must be positive")
        if action_feature_dim != ACTION_FEATURE_DIM:
            raise ValueError(
                f"action_feature_dim must be {ACTION_FEATURE_DIM} (GD-DMC-ACTION-1); "
                f"got {action_feature_dim}"
            )
        if num_players <= 0:
            raise ValueError("num_players must be positive")
        if num_players > int(np.iinfo(SEAT_DTYPE).max) + 1:
            raise ValueError("num_players must be <= 128 (int8 seat)")
        if pad_token < 0:
            raise ValueError("pad_token must be non-negative")
        if pad_token > int(np.iinfo(OBS_TOKENS_DTYPE).max):
            raise ValueError("pad_token must be <= 2**31-1 (int32 token ids)")
        if max_obs_tokens > int(np.iinfo(OBS_TOKEN_LENGTHS_DTYPE).max):
            raise ValueError("max_obs_tokens must be <= 2**31-1 (int32 lengths)")
        if num_tokens is not None:
            num_tokens = int(num_tokens)
            if num_tokens <= pad_token:
                raise ValueError("num_tokens must be greater than pad_token")
            if num_tokens > int(np.iinfo(OBS_TOKENS_DTYPE).max) + 1:
                raise ValueError("num_tokens must be <= 2**31 (int32 token ids)")

        self.capacity = capacity
        self.max_obs_tokens = max_obs_tokens
        self.channel_dim = channel_dim
        self.action_feature_dim = action_feature_dim
        self.num_players = num_players
        self.pad_token = pad_token
        self.num_tokens = num_tokens

        self._obs_tokens = np.full(
            (capacity, max_obs_tokens), pad_token, dtype=OBS_TOKENS_DTYPE,
        )
        self._obs_token_lengths = np.zeros(capacity, dtype=OBS_TOKEN_LENGTHS_DTYPE)
        self._obs_channels = np.zeros((capacity, channel_dim), dtype=OBS_CHANNELS_DTYPE)
        self._action_feats = np.zeros((capacity, action_feature_dim), dtype=ACTION_FEATS_DTYPE)
        self._z = np.zeros(capacity, dtype=Z_DTYPE)
        self._seat = np.zeros(capacity, dtype=SEAT_DTYPE)
        self._game_id = np.zeros(capacity, dtype=GAME_ID_DTYPE)

        self._write_index = 0
        self._size = 0
        self._total_pushed = 0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Basic state
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def total_pushed(self) -> int:
        """Monotonic count of rows ever pushed (provenance; not capped)."""
        return self._total_pushed

    @property
    def write_position(self) -> int:
        return self._write_index

    def nbytes(self) -> int:
        """Bytes currently owned by the pre-allocated arrays."""
        return int(
            self._obs_tokens.nbytes
            + self._obs_token_lengths.nbytes
            + self._obs_channels.nbytes
            + self._action_feats.nbytes
            + self._z.nbytes
            + self._seat.nbytes
            + self._game_id.nbytes
        )

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------
    def _transition_from_mapping(self, sample: Mapping[str, Any]) -> DMCTransition:
        required = ("obs_tokens", "obs_channels", "action_feats", "z", "seat", "game_id")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"DMC transition is missing fields: {', '.join(missing)}")
        raw_tokens = sample["obs_tokens"]
        if isinstance(raw_tokens, (str, bytes)):
            raise ValueError("obs_tokens must be a sequence of integer token ids")
        try:
            tokens = tuple(
                _as_integral(token, f"obs_tokens[{index}]")
                for index, token in enumerate(raw_tokens)
            )
        except TypeError as exc:
            raise ValueError("obs_tokens must be a sequence of integer token ids") from exc
        return DMCTransition(
            obs_tokens=tokens,
            obs_channels=np.asarray(sample["obs_channels"]),
            action_feats=np.asarray(sample["action_feats"]),
            z=_as_finite_float(sample["z"], "z"),
            seat=_as_integral(sample["seat"], "seat"),
            game_id=_as_integral(sample["game_id"], "game_id"),
        )

    def _normalise(self, samples) -> list[DMCTransition]:
        if isinstance(samples, DMCTransition):
            return [samples]
        if isinstance(samples, Mapping):
            tokens = samples.get("obs_tokens")
            if tokens is None:
                raise KeyError("batched DMC samples require 'obs_tokens'")
            token_array = np.asarray(tokens)
            if token_array.ndim == 1:
                raise ValueError(
                    "batched obs_tokens must be 2D [B, L]; pass a sequence of "
                    "DMCTransition for single samples"
                )
            if token_array.ndim != 2:
                raise ValueError("obs_tokens must be [B, L] or [B, L, ...] packed rows")
            batch = int(token_array.shape[0])
            lengths = samples.get("obs_token_lengths")
            if lengths is None:
                lengths = np.full(batch, token_array.shape[1], dtype=np.int64)
                if token_array.shape[1] and self.pad_token is not None:
                    # Trailing pad tokens are not part of the true length.
                    non_pad = np.ones_like(token_array, dtype=bool)
                    non_pad &= token_array != self.pad_token
                    has_token = non_pad.any(axis=1)
                    last = np.where(
                        has_token,
                        token_array.shape[1] - 1 - np.argmax(non_pad[:, ::-1], axis=1),
                        -1,
                    )
                    lengths = (last + 1).astype(np.int64)
            lengths = np.asarray(lengths)
            if lengths.shape != (batch,):
                raise ValueError(f"obs_token_lengths must be [{batch}], got {lengths.shape}")
            transitions = []
            for row in range(batch):
                length = _as_integral(lengths[row], f"obs_token_lengths[{row}]")
                if not 1 <= length <= token_array.shape[1]:
                    raise ValueError(
                        f"obs_token_lengths[{row}]={length} must lie in 1..{token_array.shape[1]}"
                    )
                transitions.append(
                    self._transition_from_mapping({
                        "obs_tokens": token_array[row, :length],
                        "obs_channels": np.asarray(samples["obs_channels"])[row],
                        "action_feats": np.asarray(samples["action_feats"])[row],
                        "z": np.asarray(samples["z"])[row],
                        "seat": np.asarray(samples["seat"])[row],
                        "game_id": np.asarray(samples["game_id"])[row],
                    })
                )
            return transitions
        if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
            transitions = []
            for sample in samples:
                if isinstance(sample, DMCTransition):
                    transitions.append(sample)
                elif isinstance(sample, Mapping):
                    transitions.append(self._transition_from_mapping(sample))
                else:
                    raise TypeError(
                        "each DMC sample must be a DMCTransition or a mapping, "
                        f"got {type(sample).__name__}"
                    )
            return transitions
        raise TypeError(
            "samples must be a DMCTransition, a mapping of batched arrays, or a sequence "
            f"of transitions; got {type(samples).__name__}"
        )

    def _validate_tokens(self, raw: Any) -> np.ndarray:
        """Validate token ids in their original domain, then narrow to int32.

        The range check happens *before* ``np.asarray(..., dtype=np.int32)``:
        casting first would wrap ``2**40`` to ``0`` and ``2**32 + 3`` to ``3``.
        """
        if isinstance(raw, (str, bytes)):
            raise ValueError("obs_tokens must be a 1D sequence of integer token ids")
        if isinstance(raw, np.ndarray):
            if raw.ndim != 1:
                raise ValueError("obs_tokens must be a 1D packed row")
            values = raw.tolist()
        else:
            try:
                values = list(raw)
            except TypeError as exc:
                raise ValueError("obs_tokens must be a 1D sequence of integer token ids") from exc
        length = len(values)
        if not 1 <= length <= self.max_obs_tokens:
            raise ValueError(
                f"obs_tokens length {length} must lie in 1..{self.max_obs_tokens}"
            )
        upper = self.num_tokens if self.num_tokens is not None else 2**31
        # Never let ``np.asarray(..., dtype=int32)`` raise OverflowError: cap the
        # accepted range at what the storage can represent and report ValueError.
        upper = min(upper, int(np.iinfo(OBS_TOKENS_DTYPE).max) + 1)
        checked: list[int] = []
        for index, value in enumerate(values):
            token = _as_integral(value, f"obs_tokens[{index}]")
            if not 0 <= token < upper:
                raise ValueError(
                    f"obs_tokens[{index}]={token} must lie in 0..{upper - 1}"
                )
            checked.append(token)
        return np.asarray(checked, dtype=OBS_TOKENS_DTYPE)

    def _validate_channels(self, raw: Any) -> np.ndarray:
        channels = np.asarray(raw)
        if channels.shape != (self.channel_dim,):
            raise ValueError(
                f"obs_channels must be [{self.channel_dim}], got {channels.shape}"
            )
        if not np.all(np.isfinite(channels)):
            raise ValueError("obs_channels must be finite")
        if not np.issubdtype(channels.dtype, np.integer):
            if not np.all(np.equal(np.mod(channels, 1), 0)):
                raise ValueError("obs_channels must be integral (uint8 channels)")
        if bool(((channels < 0) | (channels > 255)).any()):
            raise ValueError("obs_channels values must lie in 0..255")
        return channels.astype(OBS_CHANNELS_DTYPE, copy=True)

    def _validate_action_feats(self, raw: Any) -> np.ndarray:
        action_feats = np.asarray(raw)
        if action_feats.shape != (self.action_feature_dim,):
            raise ValueError(
                f"action_feats must be [{self.action_feature_dim}], got {action_feats.shape}"
            )
        if not np.all(np.isfinite(action_feats)):
            raise ValueError("action_feats must be finite")
        narrowed = action_feats.astype(ACTION_FEATS_DTYPE, copy=True)
        if not np.all(np.isfinite(narrowed)):
            # A finite float64 value outside the float32 range would otherwise
            # be stored as inf.
            raise ValueError("action_feats must be representable in float32")
        return narrowed

    def _validate(self, transition: DMCTransition):
        tokens = self._validate_tokens(transition.obs_tokens)
        channels = self._validate_channels(transition.obs_channels)
        action_feats = self._validate_action_feats(transition.action_feats)
        z = _as_finite_float32(transition.z, "z")
        seat = _as_integral(transition.seat, "seat")
        if not 0 <= seat < self.num_players:
            raise ValueError(f"seat {seat} must lie in 0..{self.num_players - 1}")
        if seat > int(np.iinfo(SEAT_DTYPE).max):
            raise ValueError(
                f"seat {seat} must be <= {int(np.iinfo(SEAT_DTYPE).max)} (int8)"
            )
        game_id = _as_integral(transition.game_id, "game_id")
        if game_id < 0:
            raise ValueError("game_id must be non-negative")
        if game_id > int(np.iinfo(GAME_ID_DTYPE).max):
            raise ValueError(
                f"game_id {game_id} must be <= {int(np.iinfo(GAME_ID_DTYPE).max)} (int64)"
            )
        return tokens, channels, action_feats, z, seat, game_id

    def _try_push_batched_arrays(self, samples) -> int | None:
        """Vectorised validation for batched-numpy pushes; ``None`` = fall back.

        Handles exactly the layout the sampler pushes (2-D integer token rows
        with explicit lengths, plus per-field arrays).  Every precondition that
        cannot be checked vectorised returns ``None`` so :meth:`_normalise`
        raises its canonical error instead.  Value-level rejections replicate
        the per-row messages, and with several defective rows the reported
        error is the row-major-first one, matching the per-row scan order.
        """
        if not isinstance(samples, Mapping):
            return None
        try:
            token_array = samples["obs_tokens"]
            lengths = samples["obs_token_lengths"]
            if "obs_channels" not in samples or "action_feats" not in samples \
                    or "z" not in samples or "seat" not in samples or "game_id" not in samples:
                return None
            if not isinstance(token_array, np.ndarray) or token_array.ndim != 2 \
                    or token_array.dtype == np.bool_ \
                    or not np.issubdtype(token_array.dtype, np.integer):
                return None
            if not isinstance(lengths, np.ndarray) or lengths.shape != (token_array.shape[0],) \
                    or lengths.dtype == np.bool_ \
                    or not np.issubdtype(lengths.dtype, np.integer) \
                    or lengths.dtype == np.uint64:
                return None
            channels = np.asarray(samples["obs_channels"])
            feats = np.asarray(samples["action_feats"])
            z = np.asarray(samples["z"])
            seat = np.asarray(samples["seat"])
            game_id = np.asarray(samples["game_id"])
            batch = int(token_array.shape[0])
            width = int(token_array.shape[1])
            if channels.shape != (batch, self.channel_dim) or channels.dtype == np.bool_ \
                    or not np.issubdtype(channels.dtype, np.integer):
                return None
            if feats.shape != (batch, self.action_feature_dim) \
                    or not np.issubdtype(feats.dtype, np.floating):
                return None
            if z.shape != (batch,) or not np.issubdtype(z.dtype, np.floating):
                return None
            if seat.shape != (batch,) or seat.dtype == np.bool_ \
                    or not np.issubdtype(seat.dtype, np.integer) or seat.dtype == np.uint64:
                return None
            if game_id.shape != (batch,) or game_id.dtype == np.bool_ \
                    or not np.issubdtype(game_id.dtype, np.integer) \
                    or game_id.dtype == np.uint64:
                return None
        except (TypeError, KeyError, IndexError):
            return None
        if batch == 0:
            return 0
        if batch > self.capacity:
            raise ValueError(
                f"cannot push {batch} rows into capacity {self.capacity}"
            )

        # Row-major-first defect selection: each field reports its first bad
        # row; the smallest such row wins, ties broken by the per-row scan
        # order, so a single-defect batch raises the identical message.
        defects: list[tuple[int, int, Callable[[], ValueError]]] = []

        length_values = lengths.astype(np.int64, copy=False)
        bad_width = np.flatnonzero((length_values < 1) | (length_values > width))
        if bad_width.size:
            row = int(bad_width[0])
            defects.append((row, 0, lambda: ValueError(
                f"obs_token_lengths[{row}]={int(length_values[row])} must lie in 1..{width}"
            )))
        bad_window = np.flatnonzero(length_values > self.max_obs_tokens)
        if bad_window.size:
            row = int(bad_window[0])
            defects.append((row, 1, lambda: ValueError(
                f"obs_tokens length {int(length_values[row])} must lie in "
                f"1..{self.max_obs_tokens}"
            )))

        upper = self.num_tokens if self.num_tokens is not None else 2**31
        upper = min(upper, int(np.iinfo(OBS_TOKENS_DTYPE).max) + 1)
        prefixes = np.arange(width, dtype=np.int64)[None, :] < length_values[:, None]
        with np.errstate(invalid="ignore"):
            bad_tokens = prefixes & ((token_array < 0) | (token_array >= upper))
        bad_token_flat = np.flatnonzero(bad_tokens)
        if bad_token_flat.size:
            row, col = (int(bad_token_flat[0] // width), int(bad_token_flat[0] % width))
            defects.append((row, 2, lambda: ValueError(
                f"obs_tokens[{col}]={int(token_array[row, col])} must lie in 0..{upper - 1}"
            )))

        bad_channels = np.flatnonzero((channels < 0).any(axis=1) | (channels > 255).any(axis=1))
        if bad_channels.size:
            defects.append((int(bad_channels[0]), 3, lambda: ValueError(
                "obs_channels values must lie in 0..255"
            )))

        if not bool(np.isfinite(feats).all()):
            bad_feats = np.flatnonzero(~np.isfinite(feats).all(axis=1))
            defects.append((int(bad_feats[0]), 4, lambda: ValueError(
                "action_feats must be finite"
            )))
        else:
            narrowed = feats.astype(ACTION_FEATS_DTYPE, copy=False)
            if not bool(np.isfinite(narrowed).all()):
                bad_feats = np.flatnonzero(~np.isfinite(narrowed).all(axis=1))
                defects.append((int(bad_feats[0]), 4, lambda: ValueError(
                    "action_feats must be representable in float32"
                )))

        if not bool(np.isfinite(z).all()):
            defects.append((0, 5, lambda: ValueError("z must be a finite number")))
        else:
            z_narrowed = z.astype(Z_DTYPE, copy=False)
            if not bool(np.isfinite(z_narrowed).all()):
                defects.append((0, 5, lambda: ValueError(
                    "z must be representable in float32"
                )))

        seat_values = seat.astype(np.int64, copy=False)
        bad_seats = np.flatnonzero((seat_values < 0) | (seat_values >= self.num_players))
        if bad_seats.size:
            row = int(bad_seats[0])
            defects.append((row, 6, lambda: ValueError(
                f"seat {int(seat_values[row])} must lie in 0..{self.num_players - 1}"
            )))

        gid_values = game_id.astype(np.int64, copy=False)
        bad_gid = np.flatnonzero(gid_values < 0)
        if bad_gid.size:
            defects.append((int(bad_gid[0]), 7, lambda: ValueError(
                "game_id must be non-negative"
            )))

        if defects:
            defects.sort(key=lambda item: (item[0], item[1]))
            raise defects[0][2]()

        for row in range(batch):
            index = self._write_index
            row_buffer = self._obs_tokens[index]
            row_buffer[:] = self.pad_token
            length = int(length_values[row])
            row_buffer[:length] = token_array[row, :length]
            self._obs_token_lengths[index] = length
            self._obs_channels[index] = channels[row]
            self._action_feats[index] = feats[row].astype(ACTION_FEATS_DTYPE, copy=False)
            self._z[index] = z[row]
            self._seat[index] = seat[row]
            self._game_id[index] = gid_values[row]
            self._write_index = (index + 1) % self.capacity
            self._size = min(self.capacity, self._size + 1)
        self._total_pushed += batch
        return batch

    def push_batch(self, samples) -> int:
        """Append a batch of decisions; returns the number of rows written.

        Every row is validated before anything is written, so a malformed batch
        leaves the buffer untouched.  When the ring is full the oldest rows are
        overwritten (FIFO).

        The sampler's batched-numpy pushes take a vectorised validation path
        (:meth:`_try_push_batched_arrays`) whose accept/reject set and error
        messages are identical to the per-row path; anything it does not
        recognise falls back to :meth:`_normalise` unchanged.
        """
        fast = self._try_push_batched_arrays(samples)
        if fast is not None:
            return fast
        transitions = self._normalise(samples)
        if not transitions:
            return 0
        if len(transitions) > self.capacity:
            raise ValueError(
                f"cannot push {len(transitions)} rows into capacity {self.capacity}"
            )
        validated = [self._validate(transition) for transition in transitions]

        for tokens, channels, action_feats, z, seat, game_id in validated:
            index = self._write_index
            row = self._obs_tokens[index]
            row[:] = self.pad_token
            row[: tokens.shape[0]] = tokens
            self._obs_token_lengths[index] = tokens.shape[0]
            self._obs_channels[index] = channels
            self._action_feats[index] = action_feats
            self._z[index] = z
            self._seat[index] = seat
            self._game_id[index] = game_id
            self._write_index = (index + 1) % self.capacity
            self._size = min(self.capacity, self._size + 1)
        self._total_pushed += len(validated)
        return len(validated)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------
    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> DMCReplayBatch:
        """Uniform sample **without replacement** from the live prefix.

        ``rng`` must be a ``numpy.random.Generator``; the same Generator state
        reproduces the same index sequence.  Sampling an empty buffer or asking
        for more than ``len(self)`` raises instead of silently returning fewer
        rows.
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise ValueError("cannot sample an empty DMC replay buffer")
        if batch_size > self._size:
            raise ValueError(
                f"cannot sample {batch_size} rows without replacement from {self._size}"
            )
        generator = self._rng if rng is None else rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        indices = generator.choice(self._size, size=batch_size, replace=False)
        lengths = self._obs_token_lengths[indices]
        max_length = int(lengths.max())
        return DMCReplayBatch(
            obs_tokens=self._obs_tokens[indices, :max_length].copy(),
            obs_token_lengths=lengths.copy(),
            obs_channels=self._obs_channels[indices].copy(),
            action_feats=self._action_feats[indices].copy(),
            z=self._z[indices].copy(),
            seat=self._seat[indices].copy(),
            game_id=self._game_id[indices].copy(),
        )


def dmc_replay_memory_budget(
    capacity: int,
    *,
    max_obs_tokens: int,
    channel_dim: int = DEFAULT_CHANNEL_DIM,
    action_feature_dim: int = ACTION_FEATURE_DIM,
) -> dict[str, object]:
    """Pre-allocation byte estimate for :class:`DMCReplay`.

    Mirrors the style of ``buffer.trajectory_memory_budget``: it returns the
    per-field element count, dtype and bytes without allocating the arrays.
    Excludes the model, optimizer, activations and temporary sample copies.

    The channel dtype is fixed to :data:`OBS_CHANNELS_DTYPE` (``uint8``) so the
    estimate is identical to ``DMCReplay.nbytes()`` for every supported shape.
    A caller cannot request a different width here and silently get a budget
    that disagrees with the allocated buffer.
    """
    capacity = int(capacity)
    max_obs_tokens = int(max_obs_tokens)
    channel_dim = int(channel_dim)
    action_feature_dim = int(action_feature_dim)
    if capacity <= 0 or max_obs_tokens <= 0 or channel_dim <= 0:
        raise ValueError("capacity, max_obs_tokens and channel_dim must be positive")
    if action_feature_dim != ACTION_FEATURE_DIM:
        raise ValueError(
            f"action_feature_dim must be {ACTION_FEATURE_DIM}; got {action_feature_dim}"
        )
    layouts = {
        "obs_tokens": ((capacity, max_obs_tokens), OBS_TOKENS_DTYPE),
        "obs_token_lengths": ((capacity,), OBS_TOKEN_LENGTHS_DTYPE),
        "obs_channels": ((capacity, channel_dim), OBS_CHANNELS_DTYPE),
        "action_feats": ((capacity, action_feature_dim), ACTION_FEATS_DTYPE),
        "z": ((capacity,), Z_DTYPE),
        "seat": ((capacity,), SEAT_DTYPE),
        "game_id": ((capacity,), GAME_ID_DTYPE),
    }
    fields = []
    total = 0
    for name, (shape, dtype) in layouts.items():
        elements = 1
        for size in shape:
            elements *= int(size)
        bytes_value = elements * np.dtype(dtype).itemsize
        total += bytes_value
        fields.append({
            "name": name,
            "shape": list(shape),
            "dtype": np.dtype(dtype).name,
            "elements": elements,
            "bytes": bytes_value,
        })
    return {
        "schema": "guandan-marvel-dmc-replay-memory-budget-v1",
        "capacity": capacity,
        "max_obs_tokens": max_obs_tokens,
        "channel_dim": channel_dim,
        "action_feature_dim": action_feature_dim,
        "fields": fields,
        "total_bytes": total,
        "total_mib": total / 2**20,
        "total_gib": total / 2**30,
        "total_gb": total / 1e9,
        "excludes": ["model", "optimizer", "activations", "sampled_batch_copies", "grad_scaler"],
    }


__all__ = [
    "DMCReplay",
    "DMCReplayBatch",
    "DMCTransition",
    "DEFAULT_CHANNEL_DIM",
    "dmc_replay_memory_budget",
]
