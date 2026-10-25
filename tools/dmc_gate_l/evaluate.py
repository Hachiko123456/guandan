# SPDX-License-Identifier: MIT
"""ε=0 greedy DMC evaluation against the shipped V2 row agents (Gate L).

Lane semantics, pre-declared in ``docs/dmc_prompts/05_gate_l_local.md``:

* the two DMC seats form one partnership (seats ``{0, 2}`` or ``{1, 3}``,
  alternating per game so every deal is played from both rotations) and play
  the **Q-argmax full legal candidate** — the locked table's ``ε=0 贪心
  argmax`` — committed token-by-token exactly like the training sampler;
* the other two seats are the shipped ``RuleV2Agent`` / ``RandomV2Agent``
  heuristics selecting one legal V2 row per step: the established rule/random
  lane behaviour on this adapter, never a local reimplementation;
* evaluation pushes nothing into a replay buffer and never advances the ε
  clock — it must not perturb the training run it is embedded in.

Observation streams mirror ``DMCSampler`` (cumulative per-seat event rows,
head+tail window of ``max_obs_tokens`` — stage 07B's 128/896 split), so the Q
values scored here are the same function the sampler trains; every window trim
is counted, never silent.

Failure policy is fail-loud: an illegal row, a horizon overrun or a non-finite
Q value raises instead of being absorbed into a counter, so a clean summary
implies ``illegal = truncated = nonfinite = 0`` by construction.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

import numpy as np
import torch

from guandan_marvel.backends import get_game_backend
from guandan_marvel.dmc.candidates import (
    encode_action_features,
    enumerate_candidates,
)
from guandan_marvel.dmc.learner import DMCNonFiniteError
from guandan_marvel.dmc.sampler import DEFAULT_OBS_HEAD_TOKENS, trim_obs_window
from guandan_marvel.evaluation.agents import (
    ACTION_TOKEN_COLUMN,
    RandomV2Agent,
    RuleV2Agent,
)

EVAL_SUMMARY_SCHEMA = "guandan-marvel-dmc-eval-summary-v1"
#: The fixed 2v2 partnerships; a game is scored from the first DMC seat.
ROTATION_SEATS = ((0, 2), (1, 3))


def _event_row(row: np.ndarray) -> list[int]:
    """True (non-padded) event tokens of one observation row — as the sampler."""
    return [int(token) for token in row.tolist() if token != 0]


def _action_digest(action_tokens: list[int]) -> str:
    return hashlib.sha256(
        b",".join(str(token).encode("ascii") for token in action_tokens),
    ).hexdigest()


class _EvalGame:
    """One single-env evaluation game (batch=1, ``auto_reset=False``)."""

    def __init__(
        self,
        env: Any,
        q_agent: torch.nn.Module,
        *,
        device: torch.device,
        amp: bool,
        opponent: Any,
        dmc_seats: tuple[int, int],
        game_seed: int,
        max_obs_tokens: int,
        pad_token: int,
        obs_head_tokens: int = DEFAULT_OBS_HEAD_TOKENS,
    ) -> None:
        self.env = env
        self.q_agent = q_agent
        self.device = device
        self.amp = amp
        self.opponent = opponent
        self.dmc_seats = dmc_seats
        self.game_seed = int(game_seed)
        self.max_obs_tokens = int(max_obs_tokens)
        self.obs_head_tokens = int(obs_head_tokens)
        self.pad_token = int(pad_token)

    def run(self) -> dict[str, Any]:
        step = self.env.reset(self.game_seed)
        streams = [
            _event_row(step.observations.tokens[0][seat]) for seat in range(4)
        ]
        pending: list[int] | None = []
        action_tokens: list[int] = []
        head_drops = 0
        tail_drops = 0
        steps = 0
        while True:
            if not pending:
                protocol = self.env.envs[0]._require_protocol()
                seat = int(protocol.round_state.active_seat)
                if seat in self.dmc_seats:
                    candidates = enumerate_candidates(protocol)
                    if not candidates:
                        raise RuntimeError(
                            f"seed {self.game_seed}: DMC decision point with no legal candidate"
                        )
                    q_row = self._score(seat, streams[seat], step, candidates, protocol)
                    # ε=0 greedy: pure argmax, tie-breaks to the first index
                    # (torch.argmax), never an RNG draw.
                    choice = int(torch.argmax(q_row).item())
                    pending = list(candidates[choice].tokens)
                else:
                    pending = None
            if pending:
                token = pending.pop(0)
                row = self._row_for_token(step, token)
            else:
                index = self.opponent.select_index(step)
                row = np.asarray(
                    step.available_actions.tokens[0][index], dtype=np.int64, copy=True,
                )
                token = int(row[ACTION_TOKEN_COLUMN])
            action_tokens.append(int(token))
            step = self.env.step(np.asarray([row], dtype=np.int64))
            steps += 1
            if bool(step.is_terminal[0]):
                rewards = np.asarray(step.rewards[0], dtype=np.float64)
                return {
                    "status": "complete",
                    "seed": self.game_seed,
                    "steps": steps,
                    "rewards": [float(value) for value in rewards],
                    "dmc_seats": list(self.dmc_seats),
                    # Partners share the terminal team rank points, so the
                    # first DMC seat's reward is the team result (±3/±2/±1).
                    "dmc_team_reward": float(rewards[self.dmc_seats[0]]),
                    "win": bool(rewards[self.dmc_seats[0]] > 0),
                    # Kept for the pre-existing Gate L row schema: the total
                    # window trim, now split into its two stage-07B parts.
                    "obs_window_truncations": head_drops + tail_drops,
                    "obs_head_drops": head_drops,
                    "obs_tail_drops": tail_drops,
                    "action_sha256": _action_digest(action_tokens),
                }
            for seat in range(4):
                stream = streams[seat]
                stream.extend(_event_row(step.observations.tokens[0][seat]))
                dropped_head, dropped_tail = trim_obs_window(
                    stream,
                    max_tokens=self.max_obs_tokens,
                    head_tokens=self.obs_head_tokens,
                )
                head_drops += dropped_head
                tail_drops += dropped_tail

    def _row_for_token(self, step: Any, token: int) -> np.ndarray:
        rows = step.available_actions.tokens[0]
        legal = step.available_actions.weights[0] > 0
        matches = np.flatnonzero(legal & (rows[:, ACTION_TOKEN_COLUMN] == int(token)))
        if len(matches) != 1:
            raise RuntimeError(
                f"seed {self.game_seed}: token {token} does not map to exactly one legal row"
            )
        return np.asarray(rows[matches[0]], dtype=np.int64, copy=True)

    def _score(
        self,
        seat: int,
        stream: list[int],
        step: Any,
        candidates: tuple[Any, ...],
        protocol: Any,
    ) -> torch.Tensor:
        obs = list(stream)[-self.max_obs_tokens :]
        if not obs:
            raise RuntimeError(
                f"seed {self.game_seed}: empty observation stream at a DMC eval decision"
            )
        tokens = np.full((1, len(obs)), self.pad_token, dtype=np.int64)
        tokens[0] = obs
        lengths = np.array([len(obs)], dtype=np.int64)
        channels = np.asarray(step.observations.channels[0][seat], dtype=np.uint8)[None]
        feats = np.stack(
            [encode_action_features(candidate, protocol.round_state) for candidate in candidates],
        ).astype(np.float32)[None]
        masks = np.ones((1, len(candidates)), dtype=bool)
        tokens_t = torch.from_numpy(tokens).to(self.device)
        lengths_t = torch.from_numpy(lengths).to(self.device)
        channels_t = torch.from_numpy(channels).to(self.device)
        feats_t = torch.from_numpy(feats).to(self.device)
        masks_t = torch.from_numpy(masks).to(self.device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.inference_mode(), autocast:
            q = self.q_agent.q_values(
                tokens_t, channels_t, feats_t, masks_t, token_lengths=lengths_t,
            )
        q_row = q.detach().to("cpu")[0]
        if not bool(torch.isfinite(q_row).all()):
            raise DMCNonFiniteError(
                f"seed {self.game_seed}: non-finite Q value in DMC evaluation"
            )
        return q_row


def play_eval_games(
    q_agent: torch.nn.Module,
    *,
    opponent_kind: str,
    games: int,
    base_seed: int,
    device: str = "cpu",
    max_obs_tokens: int = 1024,
    obs_head_tokens: int = DEFAULT_OBS_HEAD_TOKENS,
    max_horizon: int | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Play ``games`` ε=0 greedy games against one opponent kind.

    Deal ``i`` uses seed ``base_seed + i // 2`` and rotation ``i % 2``: every
    deal is played from both partnerships, halving the independent-deal count
    relative to the game count (the paired-rotation convention of the shipped
    pilot evaluation).
    """
    kind = str(opponent_kind)
    if kind not in ("rule", "random"):
        raise ValueError(f"opponent_kind must be 'rule' or 'random', got {kind!r}")
    count = int(games)
    if count <= 0:
        raise ValueError("games must be positive")
    game = get_game_backend("python_reference")
    horizon = int(game.max_game_horizon) if max_horizon is None else int(max_horizon)
    env = game.EnvBatch(
        1, backend="python_reference", auto_reset=False, max_horizon=horizon,
    )
    opponent = RuleV2Agent() if kind == "rule" else RandomV2Agent(seed=int(base_seed))
    device_t = torch.device(device)
    amp = device_t.type == "cuda"
    pad_token = int(game.pad_token)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(count):
        deal_seed = int(base_seed) + index // 2
        rotation = index % 2
        row = _EvalGame(
            env,
            q_agent,
            device=device_t,
            amp=amp,
            opponent=opponent,
            dmc_seats=ROTATION_SEATS[rotation],
            game_seed=deal_seed,
            max_obs_tokens=max_obs_tokens,
            pad_token=pad_token,
            obs_head_tokens=obs_head_tokens,
        ).run()
        row.update({
            "game_index": index,
            "opponent": kind,
            "rotation": rotation,
            "deal_seed": deal_seed,
        })
        rows.append(row)
        if log is not None and (index + 1) % 16 == 0:
            log(
                f"[DMC-EVAL {kind}] {index + 1}/{count} games "
                f"elapsed={time.perf_counter() - started:.0f}s"
            )
    wins = [bool(row["win"]) for row in rows]
    rewards = [float(row["dmc_team_reward"]) for row in rows]
    elapsed = time.perf_counter() - started
    ordered_steps = sorted(int(row["steps"]) for row in rows)
    summary = {
        "schema": EVAL_SUMMARY_SCHEMA,
        "opponent": kind,
        "games": len(rows),
        "independent_deals": (len(rows) + 1) // 2,
        "win_rate": (sum(wins) / len(wins)) if rows else None,
        "mean_team_reward": (sum(rewards) / len(rewards)) if rows else None,
        # A failed game raises (fail-loud); a returned summary therefore has a
        # structurally clean health record.
        "illegal": 0,
        "truncated": 0,
        "nonfinite": 0,
        "obs_window_truncations": sum(int(row["obs_window_truncations"]) for row in rows),
        "obs_head_drops": sum(int(row["obs_head_drops"]) for row in rows),
        "obs_tail_drops": sum(int(row["obs_tail_drops"]) for row in rows),
        "median_steps": ordered_steps[len(ordered_steps) // 2] if rows else None,
        "wall_seconds": elapsed,
        "games_per_second": (len(rows) / elapsed) if elapsed > 0 else None,
    }
    return {"schema": "guandan-marvel-dmc-eval-result-v1", "summary": summary, "rows": rows}


__all__ = ["EVAL_SUMMARY_SCHEMA", "ROTATION_SEATS", "play_eval_games"]
