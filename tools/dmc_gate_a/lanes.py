# SPDX-License-Identifier: MIT
"""Gate A evaluation lanes: ε=0 greedy DMC against four pre-declared opponents.

``docs/dmc_prompts/06_gate_a_kaggle.md`` (2026-10-26) pre-declares the matrix

===========  =====  ==========================================================
lane         games  pairing
===========  =====  ==========================================================
rule           512  paired deals, 2 rotations (the Gate L convention)
random         512  paired deals, 2 rotations
legacy         512  paired deals, 2 rotations, legacy V1 anchor + mirror env
m09           512  paired deals, **4 rotations** (128 deals x 4)
===========  =====  ==========================================================

Two different drivers, and the reason is not cosmetic:

* ``rule`` / ``random`` / ``legacy`` run on the stage-07C thin driver
  (:class:`guandan_marvel.dmc.env_fast.DMCEnvBatch`, ``expose_action_rows``)
  — the same driver the collector trains on, so those lanes measure the
  policy on its own observation distribution at full speed.
* ``m09`` runs on the **reference** adapter
  (:class:`guandan_marvel.game.EnvBatch`).  The thin driver encodes only the
  acting seat's 512 channels and leaves the other three rows zero (stage 07C,
  lever 1); a V2 token-level policy consumes all four seats' channels, so it
  would see a distribution it was never trained on.  The reference adapter
  encodes every seat, which is exactly the surface
  :class:`guandan_marvel.evaluation.agents.V2PolicySession` expects from the
  official host.  The DMC agent is unaffected by the choice: it only reads its
  own seat's row, and stage 07C's differential suite pins the acting-seat
  tokens/channels byte-identical between the two drivers.

The m09 seats are driven by the **official** loader/session pair
(``load_v2_policy`` + ``V2PolicySession``), never by a local reimplementation:
``observe_and_score`` is called on *every* token step (the session's KV cache
must consume the whole game stream, exactly as
``guandan_marvel.evaluation.host.play_game`` does), and only the acting seat's
row is committed.

Health is recorded, not swallowed: a truncated game, an illegal row or a
non-finite Q is captured per game and counted, and the run's health gate then
fails.  A single bad game can never destroy an eight-hour session, and a clean
gate can never hide one.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from guandan.action_state import StepwiseActionState
from guandan.cards import decode_cards
from guandan.environment import GuandanEnv
from guandan.state import RoundState
from guandan_marvel.backends import get_game_backend
from guandan_marvel.dmc.candidates import (
    encode_action_features,
    enumerate_candidates,
)
from guandan_marvel.dmc.env_fast import DMCEnvBatch
from guandan_marvel.dmc.learner import DMCNonFiniteError
from guandan_marvel.dmc.sampler import DEFAULT_OBS_HEAD_TOKENS, trim_obs_window
from guandan_marvel.evaluation.agents import (
    ACTION_TOKEN_COLUMN,
    LegacyV1OpponentSession,
    RandomV2Agent,
    RuleV2Agent,
    V2PolicySession,
)
from guandan_marvel.game import EnvBatch

LANE_SUMMARY_SCHEMA = "guandan-marvel-dmc-gate-a-lane-summary-v1"
LANE_RESULT_SCHEMA = "guandan-marvel-dmc-gate-a-lane-result-v1"

#: 2-rotation pairing (Gate L convention): every deal is played from both
#: partnerships, ``deal_seed = base_seed + index // 2``.
ROTATION_SEATS_2 = ((0, 2), (1, 3))
#: 4-rotation pairing for the m09 lane, identical to
#: ``guandan_marvel.evaluation.host.seat_assignment`` for rotations 0..3:
#: both partnerships x both hand assignments, ``deal_seed = base + index // 4``.
ROTATION_SEATS_4 = ((0, 2), (1, 3), (2, 0), (3, 1))

#: Which driver each lane runs on (declared, never inferred).
LANE_DRIVERS = {
    "rule": "dmc_env_fast",
    "random": "dmc_env_fast",
    "legacy": "dmc_env_fast",
    "m09": "env_batch_reference",
}
LANE_ROTATIONS = {"rule": ROTATION_SEATS_2, "random": ROTATION_SEATS_2,
                  "legacy": ROTATION_SEATS_2, "m09": ROTATION_SEATS_4}


class LaneError(RuntimeError):
    """A lane invariant was violated (never absorbed into a counter)."""


def _event_row(row: np.ndarray) -> list[int]:
    """True (non-padded) event tokens of one observation row — as the sampler."""
    return [int(token) for token in row.tolist() if token != 0]


def _action_digest(action_tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        b",".join(str(token).encode("ascii") for token in action_tokens),
    ).hexdigest()


def _row_for_token(step: Any, token: int, *, deal_seed: int) -> np.ndarray:
    rows = step.available_actions.tokens[0]
    legal = step.available_actions.weights[0] > 0
    matches = np.flatnonzero(legal & (rows[:, ACTION_TOKEN_COLUMN] == int(token)))
    if len(matches) != 1:
        raise LaneError(
            f"seed {deal_seed}: token {token} does not map to exactly one legal row"
        )
    return np.asarray(rows[matches[0]], dtype=np.int64, copy=True)


class QScorer:
    """ε=0 greedy scoring of the DMC candidate set (training-time window)."""

    def __init__(
        self,
        q_agent: torch.nn.Module,
        *,
        device: torch.device,
        amp: bool,
        pad_token: int,
        max_obs_tokens: int,
        obs_head_tokens: int = DEFAULT_OBS_HEAD_TOKENS,
    ) -> None:
        self.q_agent = q_agent
        self.device = device
        self.amp = amp
        self.pad_token = int(pad_token)
        self.max_obs_tokens = int(max_obs_tokens)
        self.obs_head_tokens = int(obs_head_tokens)

    def choose(self, seat: int, stream: list[int], step: Any, protocol: Any,
               *, deal_seed: int) -> list[int]:
        candidates = enumerate_candidates(protocol)
        if not candidates:
            raise LaneError(f"seed {deal_seed}: DMC decision point with no legal candidate")
        obs = list(stream)[-self.max_obs_tokens:]
        if not obs:
            raise LaneError(f"seed {deal_seed}: empty observation stream at a DMC decision")
        tokens = np.full((1, len(obs)), self.pad_token, dtype=np.int64)
        tokens[0] = obs
        lengths = np.array([len(obs)], dtype=np.int64)
        channels = np.asarray(step.observations.channels[0][seat], dtype=np.uint8)[None]
        feats = np.stack(
            [encode_action_features(candidate, protocol.round_state) for candidate in candidates],
        ).astype(np.float32)[None]
        masks = np.ones((1, len(candidates)), dtype=bool)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.inference_mode(), autocast:
            q = self.q_agent.q_values(
                torch.from_numpy(tokens).to(self.device),
                torch.from_numpy(channels).to(self.device),
                torch.from_numpy(feats).to(self.device),
                torch.from_numpy(masks).to(self.device),
                token_lengths=torch.from_numpy(lengths).to(self.device),
            )
        q_row = q.detach().to("cpu")[0]
        if not bool(torch.isfinite(q_row).all()):
            raise DMCNonFiniteError(f"seed {deal_seed}: non-finite Q value in the {seat} seat")
        choice = int(torch.argmax(q_row).item())  # ε=0: pure argmax, never an RNG draw
        return list(candidates[choice].tokens)


class _Lane:
    """Opponent under test: one object per lane, one game at a time."""

    kind = "abstract"

    def begin(self, env: Any, *, deal_seed: int) -> None:
        raise NotImplementedError

    def observe(self, step: Any) -> Any:
        """Called once per token step, before the acting seat is decided."""
        return None

    def select(self, step: Any, context: Any) -> int:
        raise NotImplementedError

    def advance(self, token: int) -> None:
        """Called after the env committed ``token`` (mirror bookkeeping)."""

    def close(self) -> None:
        """Release per-lane resources (model handles)."""

    def metadata(self) -> dict[str, Any]:
        return {"kind": self.kind}


class _RowLane(_Lane):
    """rule / random: one legal V2 row per step from the shipped heuristics."""

    def __init__(self, kind: str, agent: Any) -> None:
        self.kind = kind
        self.agent = agent

    def begin(self, env: Any, *, deal_seed: int) -> None:
        return None

    def select(self, step: Any, context: Any) -> int:
        return int(self.agent.select_index(step))

    def metadata(self) -> dict[str, Any]:
        return {"kind": self.kind, "agent": dict(getattr(self.agent, "metadata", {}))}


class _LegacyLane(_Lane):
    """legacy: the frozen V1 anchor, driven through a mirror GuandanEnv.

    The mirror is built from the *live* environment's own
    ``full_state()['round_state']`` (never from a re-deal), then advanced by the
    same V1 token the batch committed.  The terminal ranking is cross-checked
    against the batch, so a silently diverged mirror cannot score a game.
    """

    kind = "legacy"

    def __init__(self, artifact: Any, *, seed: int, device: str) -> None:
        self.artifact = artifact
        self.seed = int(seed)
        self.device = str(device)
        self.session: LegacyV1OpponentSession | None = None
        self.mirror: GuandanEnv | None = None
        self.observation: Any = None

    def begin(self, env: Any, *, deal_seed: int) -> None:
        inner = env.envs[0]
        state = inner.full_state()["round_state"]
        config = inner.config
        mirror_state = RoundState.from_hands(
            [list(decode_cards(hand)) for hand in state["hands"]],
            level_rank=config.level_rank,
            leader_seat=config.leader_seat,
            previous_result=config.previous_result,
        )
        self.mirror = GuandanEnv(config)
        self.observation = self.mirror.reset(
            initial_state=StepwiseActionState(mirror_state).serialize(),
        )
        self.session = LegacyV1OpponentSession(self.artifact, seed=self.seed, device=self.device)

    def select(self, step: Any, context: Any) -> int:
        if self.session is None or self.observation is None:
            raise LaneError("legacy lane used before begin()")
        seat = int(step.player_ids[0])
        if int(self.observation.player_id) != seat:
            # The mirror must be at the same seat as the batch; a silent
            # divergence would let the anchor play from a stale observation.
            raise LaneError(
                f"legacy mirror active seat {self.observation.player_id} != batch seat {seat}"
            )
        return int(self.session.select_index(step, self.observation))

    def advance(self, token: int) -> None:
        if self.mirror is None:
            raise LaneError("legacy lane advanced before begin()")
        self.observation = self.mirror.step(int(token)).observation

    def close(self) -> None:
        self.session = None
        self.mirror = None
        self.observation = None

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "checkpoint_sha256": getattr(self.artifact, "sha256", None),
            "protocol_versions": dict(getattr(self.artifact, "protocol_versions", {})),
            "mirror": "RoundState.from_hands(env.full_state()) + GuandanEnv(initial_state)",
        }


class _M09Lane(_Lane):
    """m09: the frozen iter2000 V2 policy through the official loader/session."""

    kind = "m09"

    def __init__(self, artifact: Any, *, seed: int, device: str) -> None:
        self.artifact = artifact
        self.seed = int(seed)
        self.device = str(device)
        self.session: V2PolicySession | None = None

    def begin(self, env: Any, *, deal_seed: int) -> None:
        self.session = V2PolicySession(
            self.artifact, seed=self.seed, temperature=0.0, device=self.device,
        )

    def observe(self, step: Any) -> torch.Tensor:
        if self.session is None:
            raise LaneError("m09 lane used before begin()")
        return self.session.observe_and_score(step)

    def select(self, step: Any, context: Any) -> int:
        if self.session is None:
            raise LaneError("m09 lane used before begin()")
        return int(self.session.select_index(context, step))

    def close(self) -> None:
        self.session = None

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "checkpoint_sha256": getattr(self.artifact, "sha256", None),
            "checkpoint_iteration": getattr(self.artifact, "iteration", None),
            "architecture": getattr(self.artifact, "architecture", None),
            "algorithm": getattr(self.artifact, "algorithm", None),
            "temperature": 0.0,
            "loader": "guandan_marvel.evaluation.checkpoints.load_v2_policy",
            "session": "guandan_marvel.evaluation.agents.V2PolicySession",
        }


def build_lane(
    kind: str,
    *,
    seed: int,
    device: str,
    legacy_artifact: Any = None,
    m09_artifact: Any = None,
) -> _Lane:
    if kind == "rule":
        return _RowLane("rule", RuleV2Agent())
    if kind == "random":
        return _RowLane("random", RandomV2Agent(seed=int(seed)))
    if kind == "legacy":
        if legacy_artifact is None:
            raise LaneError("the legacy lane requires the frozen V1 anchor artifact")
        return _LegacyLane(legacy_artifact, seed=seed, device=device)
    if kind == "m09":
        if m09_artifact is None:
            raise LaneError("the m09 lane requires the frozen iter2000 artifact")
        return _M09Lane(m09_artifact, seed=seed, device=device)
    raise LaneError(f"unsupported lane kind {kind!r}")


def _make_env(kind: str, *, horizon: int) -> Any:
    if LANE_DRIVERS[kind] == "dmc_env_fast":
        # Rows stay exposed because the row-selecting opponents pick by index.
        return DMCEnvBatch(1, auto_reset=False, max_horizon=horizon, expose_action_rows=True)
    return EnvBatch(1, auto_reset=False, max_horizon=horizon)


def _play_one(
    env: Any,
    scorer: QScorer,
    lane: _Lane,
    *,
    deal_seed: int,
    dmc_seats: tuple[int, int],
    max_obs_tokens: int,
    obs_head_tokens: int,
) -> dict[str, Any]:
    step = env.reset(deal_seed)
    lane.begin(env, deal_seed=deal_seed)
    streams = [_event_row(step.observations.tokens[0][seat]) for seat in range(4)]
    pending: list[int] = []
    action_tokens: list[int] = []
    head_drops = 0
    tail_drops = 0
    steps = 0
    while True:
        seat = int(step.player_ids[0])
        if seat not in range(4):
            raise LaneError(f"seed {deal_seed}: nonterminal frame has no active seat")
        # The V2 session must consume *every* token step, whoever acts.
        context = lane.observe(step)
        if not pending:
            if seat in dmc_seats:
                protocol = env.envs[0]._require_protocol()
                if int(protocol.round_state.active_seat) != seat:
                    raise LaneError(
                        f"seed {deal_seed}: protocol active seat "
                        f"{protocol.round_state.active_seat} != frame seat {seat}"
                    )
                pending = scorer.choose(seat, streams[seat], step, protocol, deal_seed=deal_seed)
                token = pending.pop(0)
                row = _row_for_token(step, token, deal_seed=deal_seed)
            else:
                index = lane.select(step, context)
                row = np.asarray(
                    step.available_actions.tokens[0][index], dtype=np.int64, copy=True,
                )
                token = int(row[ACTION_TOKEN_COLUMN])
        else:
            token = pending.pop(0)
            row = _row_for_token(step, token, deal_seed=deal_seed)
        action_tokens.append(int(token))
        step = env.step(np.asarray([row], dtype=np.int64))
        lane.advance(int(token))
        steps += 1
        if bool(step.is_terminal[0]):
            rewards = np.asarray(step.rewards[0], dtype=np.float64)
            info = dict(step.infos[0])
            ranking = [int(value) for value in info["ranking"]]
            if lane.kind == "legacy":
                mirror_ranking = [
                    int(value) for value in lane.mirror.full_state()["round_state"]["ranking"]
                ]
                if mirror_ranking != ranking:
                    raise LaneError(
                        f"seed {deal_seed}: legacy mirror ranking {mirror_ranking} "
                        f"!= batch ranking {ranking}"
                    )
            return {
                "status": "complete",
                "seed": deal_seed,
                "steps": steps,
                "rewards": [float(value) for value in rewards],
                "ranking": ranking,
                "outcome_class": str(info["outcome_class"]),
                "dmc_seats": list(dmc_seats),
                "dmc_team_reward": float(rewards[dmc_seats[0]]),
                "win": bool(rewards[dmc_seats[0]] > 0),
                "obs_window_truncations": head_drops + tail_drops,
                "obs_head_drops": head_drops,
                "obs_tail_drops": tail_drops,
                "action_sha256": _action_digest(action_tokens),
            }
        for seat_index in range(4):
            stream = streams[seat_index]
            stream.extend(_event_row(step.observations.tokens[0][seat_index]))
            dropped_head, dropped_tail = trim_obs_window(
                stream, max_tokens=max_obs_tokens, head_tokens=obs_head_tokens,
            )
            head_drops += dropped_head
            tail_drops += dropped_tail


def play_lane_games(
    q_agent: torch.nn.Module,
    *,
    kind: str,
    games: int,
    base_seed: int,
    device: str = "cpu",
    max_obs_tokens: int = 1024,
    obs_head_tokens: int = DEFAULT_OBS_HEAD_TOKENS,
    max_horizon: int | None = None,
    legacy_artifact: Any = None,
    m09_artifact: Any = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Play ``games`` ε=0 greedy games for one lane.

    ``deal_seed = base_seed + index // rotations`` and
    ``dmc_seats = LANE_ROTATIONS[kind][index % rotations]``: every deal is
    played from every pre-declared rotation, so the lane's game count is
    ``independent_deals x rotations`` by construction.
    """
    if kind not in LANE_DRIVERS:
        raise LaneError(f"unsupported lane kind {kind!r}")
    count = int(games)
    if count <= 0:
        raise LaneError("games must be positive")
    rotations = LANE_ROTATIONS[kind]
    game = get_game_backend("python_reference")
    horizon = int(game.max_game_horizon) if max_horizon is None else int(max_horizon)
    device_t = torch.device(device)
    scorer = QScorer(
        q_agent,
        device=device_t,
        amp=device_t.type == "cuda",
        pad_token=int(game.pad_token),
        max_obs_tokens=max_obs_tokens,
        obs_head_tokens=obs_head_tokens,
    )
    env = _make_env(kind, horizon=horizon)
    lane = build_lane(
        kind, seed=int(base_seed) + 11, device=str(device_t),
        legacy_artifact=legacy_artifact, m09_artifact=m09_artifact,
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index in range(count):
            deal_seed = int(base_seed) + index // len(rotations)
            rotation = index % len(rotations)
            try:
                row = _play_one(
                    env, scorer, lane,
                    deal_seed=deal_seed,
                    dmc_seats=rotations[rotation],
                    max_obs_tokens=max_obs_tokens,
                    obs_head_tokens=obs_head_tokens,
                )
            except Exception as error:  # recorded, then the health gate decides
                row = {
                    "status": "failed",
                    "seed": deal_seed,
                    "error": f"{type(error).__name__}: {error}",
                    "win": None,
                    "dmc_team_reward": None,
                    "steps": None,
                    "obs_window_truncations": 0,
                    "obs_head_drops": 0,
                    "obs_tail_drops": 0,
                }
            row.update({
                "game_index": index,
                "lane": kind,
                "rotation": rotation,
                "deal_seed": deal_seed,
                "driver": LANE_DRIVERS[kind],
            })
            rows.append(row)
            if log is not None and (index + 1) % 32 == 0:
                log(
                    f"[DMC-GATE-A {kind}] {index + 1}/{count} games "
                    f"elapsed={time.perf_counter() - started:.0f}s"
                )
    finally:
        lane.close()

    complete = [row for row in rows if row["status"] == "complete"]
    failed = [row for row in rows if row["status"] != "complete"]
    wins = [bool(row["win"]) for row in complete]
    rewards = [float(row["dmc_team_reward"]) for row in complete]
    elapsed = time.perf_counter() - started
    ordered_steps = sorted(int(row["steps"]) for row in complete)
    summary = {
        "schema": LANE_SUMMARY_SCHEMA,
        "lane": kind,
        "driver": LANE_DRIVERS[kind],
        "games": len(rows),
        "games_completed": len(complete),
        "games_failed": len(failed),
        "independent_deals": (len(rows) + len(rotations) - 1) // len(rotations),
        "rotations": [list(seats) for seats in rotations],
        "win_rate": (sum(wins) / len(wins)) if wins else None,
        "mean_team_reward": (sum(rewards) / len(rewards)) if rewards else None,
        # Failures are counted, never hidden: the gate reads these fields.
        "illegal": sum(1 for row in failed if "NonFinite" not in str(row.get("error"))),
        "truncated": sum(1 for row in failed if "Horizon" in str(row.get("error"))),
        "nonfinite": sum(1 for row in failed if "NonFinite" in str(row.get("error"))),
        "obs_window_truncations": sum(int(row["obs_window_truncations"]) for row in rows),
        "obs_head_drops": sum(int(row["obs_head_drops"]) for row in rows),
        "obs_tail_drops": sum(int(row["obs_tail_drops"]) for row in rows),
        "median_steps": ordered_steps[len(ordered_steps) // 2] if ordered_steps else None,
        "wall_seconds": elapsed,
        "games_per_second": (len(rows) / elapsed) if elapsed > 0 else None,
        "opponent": lane.metadata(),
    }
    return {"schema": LANE_RESULT_SCHEMA, "summary": summary, "rows": rows}


__all__ = [
    "LANE_DRIVERS",
    "LANE_RESULT_SCHEMA",
    "LANE_ROTATIONS",
    "LANE_SUMMARY_SCHEMA",
    "ROTATION_SEATS_2",
    "ROTATION_SEATS_4",
    "LaneError",
    "QScorer",
    "build_lane",
    "play_lane_games",
]
