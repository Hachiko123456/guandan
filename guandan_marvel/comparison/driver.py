# SPDX-License-Identifier: MIT
"""Comparison game driver: M09 / FableDan / rule / random in one MARVEL arena."""
from __future__ import annotations

import hashlib
import time
from typing import Mapping

import numpy as np

from guandan.action_state import StepwiseActionState
from guandan.cards import Rank, decode_cards
from guandan.environment import GuandanEnv
from guandan.state import RoundState
from guandan_marvel import cpp_game
from guandan_marvel.evaluation.agents import (
    RandomV2Agent, RuleV2Agent, V2PolicySession, action_index_for_v1_token,
)
from guandan_marvel.evaluation.checkpoints import V2PolicyArtifact
from guandan_marvel.evaluation.common import canonical_json_bytes
from guandan_marvel.evaluation.host import seat_assignment, validated_commit_flag
from guandan_marvel.evaluation.manifests import deal_game_config

from .cards_map import marvel_to_fd_level
from .fabledan_guest import FableDanGuestSession

GAME_RECORD_SCHEMA = "guandan-marvel-comparison-game-v1"
M09_KINDS = {"m09", "m09_champion"}
BOOTCAMP_FINAL_KIND = "bootcamp_final"
V2_KINDS = M09_KINDS | {BOOTCAMP_FINAL_KIND}
OTHER_KINDS = {"fabledan", "rule", "random"}


def _base_deal_hash(env) -> str:
    state = env.debug_state(0)["state"]
    hands = [sorted(int(card) for card in hand) for hand in state["hands"]]
    if len(hands) != 4 or any(len(hand) != 27 for hand in hands):
        raise RuntimeError("comparison did not produce a complete 27x4 deal")
    if sorted(card for hand in hands for card in hand) != list(range(108)):
        raise RuntimeError("comparison base deal violates 108-card conservation")
    return hashlib.sha256(canonical_json_bytes(hands)).hexdigest()


class _M09Agent:
    def __init__(self, artifact: V2PolicyArtifact, *, seed: int, temperature: float, device: str):
        self.session = V2PolicySession(artifact, seed=seed, temperature=temperature, device=device)

    def select_index(self, step, mirror_protocol=None) -> int:
        probabilities = self.session.observe_and_score(step)
        return self.session.select_index(probabilities, step)


class _FableDanAgent:
    def __init__(self, session: FableDanGuestSession, mirror: GuandanEnv):
        self.session = session
        self.mirror = mirror

    def select_index(self, step, mirror_protocol=None) -> int:
        protocol = self.mirror._require_protocol()
        token = self.session.next_token(protocol)
        return action_index_for_v1_token(step, token)


def _resolve_kind(kind: str, *, m09_artifacts: Mapping[str, V2PolicyArtifact], fabledan_model, fabledan_level: int, mirror: GuandanEnv | None, seed: int, temperature: float, device: str):
    if kind in V2_KINDS:
        artifact = m09_artifacts[kind]
        return _M09Agent(artifact, seed=seed, temperature=temperature, device=device)
    if kind == "fabledan":
        if fabledan_model is None or mirror is None:
            raise ValueError("fabledan agent requires the guest model and a mirror environment")
        return _FableDanAgent(FableDanGuestSession(fabledan_model, fabledan_level), mirror)
    if kind == "rule":
        return RuleV2Agent()
    if kind == "random":
        return RandomV2Agent(seed)
    raise ValueError(f"unsupported comparison kind {kind!r}")


def play_comparison_game(
    *,
    arm_id: str,
    task_id: str,
    candidate_kind: str,
    opponent_kind: str,
    candidate_label: str,
    opponent_label: str,
    deal: Mapping[str, object],
    rotation: int,
    m09_artifacts: Mapping[str, V2PolicyArtifact],
    fabledan_model=None,
    temperature: float = 0.0,
    device: str = "cpu",
    deep_verify: bool = False,
    max_steps: int = 4096,
) -> dict[str, object]:
    if candidate_kind not in V2_KINDS | {"fabledan"}:
        raise ValueError(f"unsupported candidate kind {candidate_kind!r}")
    if opponent_kind not in OTHER_KINDS | V2_KINDS:
        raise ValueError(f"unsupported opponent kind {opponent_kind!r}")
    needs_fd = candidate_kind == "fabledan" or opponent_kind == "fabledan"
    config = deal_game_config(deal)
    env = cpp_game.EnvBatch(1, configs=[config], auto_reset=False, max_horizon=max_steps)
    step = env.reset(int(deal["seed"]))
    deal_hash = _base_deal_hash(env)
    mapping, policies = seat_assignment(rotation, opponent_label)
    candidate_team = rotation % 2

    mirror: GuandanEnv | None = None
    if needs_fd or deep_verify:
        debug_state = env.debug_state(0)["state"]
        mirror_state = RoundState.from_hands(
            [list(decode_cards(hand)) for hand in debug_state["hands"]],
            level_rank=Rank(int(deal["level_rank"])),
            leader_seat=int(deal["leader_seat"]),
            previous_result=config.previous_result,
        )
        mirror = GuandanEnv(config)
        mirror.reset(initial_state=StepwiseActionState(mirror_state).serialize())

    fabledan_level = marvel_to_fd_level(int(deal["level_rank"]))
    policy_seed_base = int(deal["seed"]) + 10_000 * (rotation + 1)
    candidate_agent = _resolve_kind(
        candidate_kind, m09_artifacts=m09_artifacts, fabledan_model=fabledan_model,
        fabledan_level=fabledan_level, mirror=mirror,
        seed=policy_seed_base + 1, temperature=temperature, device=device,
    )
    opponent_agent = _resolve_kind(
        opponent_kind, m09_artifacts=m09_artifacts, fabledan_model=fabledan_model,
        fabledan_level=fabledan_level, mirror=mirror,
        seed=policy_seed_base + 2, temperature=temperature, device=device,
    )
    fd_session = None
    for agent in (candidate_agent, opponent_agent):
        if isinstance(agent, _FableDanAgent):
            fd_session = agent.session

    started = time.perf_counter()
    action_digest = hashlib.sha256()
    decisions = [0, 0, 0, 0]
    committed_steps = 0
    deep_verified = 0
    terminal_step = None
    for token_step in range(1, max_steps + 1):
        player = int(step.player_ids[0])
        if player not in range(4):
            raise RuntimeError("nonterminal comparison frame has no active player")
        role = policies[player]
        agent = candidate_agent if role == "candidate" else opponent_agent
        action_index = agent.select_index(step)
        row = np.asarray(step.available_actions.tokens[0, action_index], dtype=np.int64)
        token_id = int(row[5])
        is_commit = validated_commit_flag(step, action_index)
        decisions[player] += 1
        committed_steps += int(is_commit)
        action_digest.update(canonical_json_bytes([player, token_id, row.tolist()]))
        step = env.step(row.reshape(1, -1))
        if mirror is not None:
            mirror_result = mirror.step(token_id)
            if mirror_result.is_commit and fd_session is not None:
                fd_session.record_commit(mirror_result.committed_action)
            if deep_verify and mirror_result.is_commit:
                engine_history = env.debug_state(0)["state"]["history"]
                engine_cards = sorted(int(card) for card in engine_history[-1]["card_ids"])
                mirror_cards = [] if mirror_result.committed_action is None else sorted(int(c) for c in mirror_result.committed_action.card_ids)
                if engine_cards != mirror_cards:
                    raise RuntimeError("mirror committed cards diverged from the C++ engine")
                deep_verified += 1
        if bool(step.is_terminal[0]):
            terminal_step = token_step
            break
    runtime_seconds = time.perf_counter() - started

    base_record: dict[str, object] = {
        "schema": GAME_RECORD_SCHEMA,
        "arm_id": arm_id,
        "task_id": task_id,
        "candidate_kind": candidate_kind,
        "opponent_kind": opponent_kind,
        "candidate_label": candidate_label,
        "opponent_label": opponent_label,
        "deal_id": deal["deal_id"],
        "deal_seed": int(deal["seed"]),
        "level_rank": int(deal["level_rank"]),
        "leader_seat": int(deal["leader_seat"]),
        "base_deal_sha256": deal_hash,
        "seat_rotation": rotation,
        "logical_to_physical": mapping,
        "policies_by_seat": policies,
        "candidate_team": candidate_team,
        "temperature": float(temperature),
        "backend": cpp_game.BACKEND,
        "protocol_version": cpp_game.protocol_version,
        "token_steps": terminal_step if terminal_step is not None else max_steps,
        "committed_steps": committed_steps,
        "decisions_by_seat": decisions,
        "illegal_count": 0,
        "runtime_seconds": runtime_seconds,
        "action_trace_sha256": action_digest.hexdigest(),
        "fd_diagnostics": None if fd_session is None else dict(fd_session.diagnostics),
        "deep_verified_commits": deep_verified,
    }

    if terminal_step is None:
        base_record.update({"status": "incomplete", "truncated_count": 1, "candidate_win": None, "ranking": None})
        return base_record

    info = dict(step.infos[0])
    terminal_debug = env.debug_state(0)["state"]
    finish_committed_step = int(terminal_debug["finish_committed_step"])
    finish_token_step = int(terminal_debug["finish_token_step"])
    if committed_steps != finish_committed_step:
        raise RuntimeError("comparison committed-step count disagrees with engine finish_committed_step")
    if terminal_step != finish_token_step:
        raise RuntimeError("comparison token-step count disagrees with engine finish_token_step")
    ranking = [int(value) for value in info["ranking"]]
    winner_team = ranking.index(1) % 2
    rewards = [float(value) for value in np.asarray(step.rewards[0]).tolist()]
    candidate_team_reward = float(rewards[candidate_team])
    # candidate_win is derived from the *sign* of the candidate team's raw rank
    # point reward, never from an exact +/-1 comparison: the phase-4 reward is
    # +3/+2/+1, so sign(reward) is the only scale-free win criterion.
    candidate_win = candidate_team_reward > 0.0
    if candidate_win != (winner_team == candidate_team):
        raise RuntimeError("candidate_win sign disagrees with winner_team")
    base_record.update({
        "status": "complete",
        "truncated_count": 0,
        "winner_team": winner_team,
        "candidate_win": candidate_win,
        "ranking": ranking,
        "outcome_class": str(info["outcome_class"]),
        "rewards": rewards,
        "candidate_team_reward": candidate_team_reward,
        "engine_finish_token_step": finish_token_step,
        "engine_finish_committed_step": finish_committed_step,
        "actions_per_second": terminal_step / runtime_seconds if runtime_seconds > 0 else None,
    })
    return base_record


__all__ = ["BOOTCAMP_FINAL_KIND", "GAME_RECORD_SCHEMA", "OTHER_KINDS", "V2_KINDS", "play_comparison_game"]
