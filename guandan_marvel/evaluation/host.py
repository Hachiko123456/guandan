# SPDX-License-Identifier: MIT
"""GuanDan C++ evaluation host with fixed deals and four-seat policy rotation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from guandan.action_state import StepwiseActionState
from guandan.cards import Rank, decode_cards
from guandan.environment import GuandanEnv
from guandan.state import RoundState
from guandan_marvel import cpp_game

from .agents import LegacyV1OpponentSession, RandomV2Agent, RuleV2Agent, V2PolicySession
from .checkpoints import LegacyV1OpponentArtifact, V2PolicyArtifact
from .common import canonical_json_bytes
from .manifests import deal_game_config

GAME_RECORD_SCHEMA = "guandan-marvel-evaluation-game-v1"
OPPONENT_KINDS = {"v2", "rule", "random", "legacy_v1"}


def seat_assignment(rotation: int, opponent_label: str) -> tuple[list[int], list[str]]:
    if isinstance(rotation, bool) or rotation not in range(4):
        raise ValueError("rotation must be one of 0,1,2,3")
    mapping = [(logical + rotation) % 4 for logical in range(4)]
    policies = [""] * 4
    for logical, physical in enumerate(mapping):
        policies[physical] = "candidate" if logical % 2 == 0 else opponent_label
    return mapping, policies


def _base_deal_hash(env) -> str:
    state = env.debug_state(0)["state"]
    hands = [sorted(int(card) for card in hand) for hand in state["hands"]]
    if len(hands) != 4 or any(len(hand) != 27 for hand in hands):
        raise RuntimeError("evaluation did not produce a complete 27x4 deal")
    if sorted(card for hand in hands for card in hand) != list(range(108)):
        raise RuntimeError("evaluation base deal violates 108-card conservation")
    return hashlib.sha256(canonical_json_bytes(hands)).hexdigest()


def _training_overlap(artifact: LegacyV1OpponentArtifact, *, seed: int, deal_hash: str) -> None:
    for row in artifact.training_deals:
        if row.get("seed") == seed or row.get("deal_sha256") == deal_hash:
            raise ValueError("selection deal overlaps legacy opponent training data")


def play_game(
    *,
    deal: Mapping[str, object],
    rotation: int,
    opponent_id: str,
    opponent_kind: str,
    candidate: V2PolicyArtifact,
    opponent_v2: V2PolicyArtifact | None = None,
    legacy: LegacyV1OpponentArtifact | None = None,
    temperature: float = 0.0,
    device: str = "cpu",
    max_steps: int = 4096,
) -> dict[str, object]:
    if opponent_kind not in OPPONENT_KINDS:
        raise ValueError(f"unsupported opponent kind {opponent_kind!r}")
    if opponent_kind == "v2" and opponent_v2 is None:
        raise ValueError("v2 opponent requires a checkpoint")
    if opponent_kind == "legacy_v1" and legacy is None:
        raise ValueError("legacy_v1 opponent requires a dedicated legacy artifact")
    config = deal_game_config(deal)
    env = cpp_game.EnvBatch(1, configs=[config], auto_reset=False, max_horizon=max_steps)
    step = env.reset(int(deal["seed"]))
    deal_hash = _base_deal_hash(env)
    mapping, policies = seat_assignment(rotation, opponent_id)
    candidate_team = rotation % 2
    policy_seed_base = int(deal["seed"]) + 10_000 * (rotation + 1)
    candidate_session = V2PolicySession(candidate, seed=policy_seed_base + 1, temperature=temperature, device=device)
    opponent_session = None
    if opponent_kind == "v2":
        opponent_session = V2PolicySession(
            opponent_v2, seed=policy_seed_base + 2, temperature=temperature, device=device,
        )
    random_agent = RandomV2Agent(policy_seed_base + 3) if opponent_kind == "random" else None
    rule_agent = RuleV2Agent() if opponent_kind == "rule" else None
    legacy_session = LegacyV1OpponentSession(legacy, seed=policy_seed_base + 4, device=device) if legacy is not None else None
    mirror = None
    mirror_observation = None
    if legacy_session is not None:
        debug_state = env.debug_state(0)["state"]
        mirror_state = RoundState.from_hands(
            [list(decode_cards(hand)) for hand in debug_state["hands"]],
            level_rank=Rank(int(deal["level_rank"])),
            leader_seat=int(deal["leader_seat"]),
            previous_result=config.previous_result,
        )
        mirror = GuandanEnv(config)
        mirror_observation = mirror.reset(initial_state=StepwiseActionState(mirror_state).serialize())
    if legacy is not None:
        _training_overlap(legacy, seed=int(deal["seed"]), deal_hash=deal_hash)

    started = time.perf_counter()
    action_digest = hashlib.sha256()
    decisions = [0, 0, 0, 0]
    committed_steps = 0
    illegal_count = 0
    terminal_step = None
    for token_step in range(1, max_steps + 1):
        player = int(step.player_ids[0])
        if player not in range(4):
            raise RuntimeError("nonterminal evaluation frame has no active player")
        candidate_probs = candidate_session.observe_and_score(step)
        opponent_probs = opponent_session.observe_and_score(step) if opponent_session is not None else None
        role = policies[player]
        try:
            if role == "candidate":
                action_index = candidate_session.select_index(candidate_probs, step)
            elif opponent_kind == "v2":
                action_index = opponent_session.select_index(opponent_probs, step)
            elif opponent_kind == "random":
                action_index = random_agent.select_index(step)
            elif opponent_kind == "rule":
                action_index = rule_agent.select_index(step)
            else:
                if mirror_observation is None or int(mirror_observation.player_id) != player:
                    raise RuntimeError("legacy mirror active seat diverged from C++ host")
                action_index = legacy_session.select_index(step, mirror_observation)
        except Exception:
            illegal_count += 1
            raise
        row = np.asarray(step.available_actions.tokens[0, action_index], dtype=np.int64)
        token_id = int(row[5])
        is_commit = bool(row[6])
        decisions[player] += 1
        committed_steps += int(is_commit)
        action_digest.update(canonical_json_bytes([player, token_id, row.tolist()]))
        step = env.step(row.reshape(1, -1))
        if mirror is not None:
            mirror_result = mirror.step(token_id)
            mirror_observation = mirror_result.observation
        if bool(step.is_terminal[0]):
            terminal_step = token_step
            break
    runtime_seconds = time.perf_counter() - started
    if terminal_step is None:
        return {
            "schema": GAME_RECORD_SCHEMA,
            "status": "incomplete",
            "opponent_id": opponent_id,
            "opponent_kind": opponent_kind,
            "deal_id": deal["deal_id"],
            "deal_seed": deal["seed"],
            "base_deal_sha256": deal_hash,
            "seat_rotation": rotation,
            "candidate_team": candidate_team,
            "illegal_count": illegal_count,
            "truncated_count": 1,
            "runtime_seconds": runtime_seconds,
            "token_steps": max_steps,
        }
    info = dict(step.infos[0])
    ranking = [int(value) for value in info["ranking"]]
    winner_team = ranking.index(1) % 2
    rewards = [float(value) for value in np.asarray(step.rewards[0]).tolist()]
    if mirror is not None:
        if not mirror.done or mirror.truncated:
            raise RuntimeError("legacy mirror did not reach terminal with C++ host")
        mirror_state = mirror.full_state()["round_state"]
        if list(mirror_state["ranking"]) != ranking or mirror_state["outcome_class"] != info["outcome_class"]:
            raise RuntimeError("legacy mirror terminal semantics diverged from C++ host")
    return {
        "schema": GAME_RECORD_SCHEMA,
        "status": "complete",
        "opponent_id": opponent_id,
        "opponent_kind": opponent_kind,
        "deal_id": deal["deal_id"],
        "deal_seed": int(deal["seed"]),
        "level_rank": int(deal["level_rank"]),
        "leader_seat": int(deal["leader_seat"]),
        "base_deal_sha256": deal_hash,
        "seat_rotation": rotation,
        "logical_to_physical": mapping,
        "policies_by_seat": policies,
        "candidate_team": candidate_team,
        "winner_team": winner_team,
        "candidate_win": winner_team == candidate_team,
        "ranking": ranking,
        "outcome_class": str(info["outcome_class"]),
        "rewards": rewards,
        "candidate_team_reward": float(rewards[candidate_team]),
        "token_steps": terminal_step,
        "committed_steps": committed_steps,
        "decisions_by_seat": decisions,
        "illegal_count": illegal_count,
        "truncated_count": 0,
        "runtime_seconds": runtime_seconds,
        "actions_per_second": terminal_step / runtime_seconds if runtime_seconds > 0 else None,
        "action_trace_sha256": action_digest.hexdigest(),
        "candidate_checkpoint_sha256": candidate.sha256,
        "opponent_checkpoint_sha256": opponent_v2.sha256 if opponent_v2 is not None else (legacy.sha256 if legacy is not None else None),
        "temperature": float(temperature),
        "backend": cpp_game.BACKEND,
        "protocol_version": cpp_game.protocol_version,
    }


__all__ = ["GAME_RECORD_SCHEMA", "play_game", "seat_assignment"]
