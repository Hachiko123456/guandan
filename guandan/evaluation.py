"""Deterministic, profile-driven A06 evaluation baselines."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np

from .cards import Rank, effective_rank_value
from .combos import CombinationKind, can_beat, require_combination
from .execution_profiles import load_execution_profile
from .round import apply_action
from .training.runtime import EVALUATION_SEED_BASE
from .state import CommittedAction, RoundState, TEAM_OF


class EvaluationIncomplete(RuntimeError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(json.dumps(report, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class GameRecord:
    opponent: str
    deal_group: int
    seat_rotation: int
    policy_seat: int
    seed: int
    done: bool
    ranking: tuple[int, int, int, int]
    winner_team: int
    outcome_class: str
    rewards: tuple[float, float, float, float]
    team_rewards: tuple[float, float]
    committed_steps: int
    terminal: bool


@dataclass(frozen=True)
class EvaluationResult:
    profile: str
    status: str
    target_games_per_opponent: int
    completed_games_per_opponent: dict[str, int]
    total_games: int
    records: tuple[GameRecord, ...]
    runtime_seconds: float
    seed_namespace: int
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        data = asdict(self)
        data["records"] = [asdict(record) for record in self.records]
        return data


def _single_actions(state: RoundState) -> list[CommittedAction]:
    if state.phase.value != "play" or state.done:
        return []
    actions = []
    for card in state.hands[state.active_seat]:
        declaration = require_combination((card,), level_rank=state.level_rank, kind=CombinationKind.SINGLE)
        if state.current_winning is None or can_beat(declaration, state.current_winning):
            actions.append(CommittedAction(state.active_seat, CombinationKind.SINGLE, (card,), declaration))
    return actions


def _choose_action(state: RoundState, strategy: str, rng: random.Random) -> CommittedAction:
    legal = _single_actions(state)
    if state.current_winning is not None:
        legal.append(CommittedAction(state.active_seat, "pass"))
    if not legal:
        raise RuntimeError("baseline found no legal action")
    plays = [action for action in legal if not action.is_pass]
    if strategy == "random":
        return rng.choice(plays if plays else legal)
    if strategy == "rule":
        if plays:
            return min(plays, key=lambda action: (effective_rank_value(action.cards[0].rank, state.level_rank), action.cards[0].card_id))
        return legal[0]
    if strategy == "snapshot":
        if plays:
            return max(plays, key=lambda action: (effective_rank_value(action.cards[0].rank, state.level_rank), -action.cards[0].card_id))
        return legal[0]
    raise ValueError(f"unknown baseline strategy: {strategy}")


def _rotate_state(state: RoundState, rotation: int) -> RoundState:
    rotation = int(rotation) % 4
    hands = [list(state.hands[(seat - rotation) % 4]) for seat in range(4)]
    return RoundState.from_hands(hands, level_rank=state.level_rank, leader_seat=(state.leader_seat + rotation) % 4)


def play_game(*, opponent: str, deal_group: int, seat_rotation: int, seed: int, max_steps: int = 4096) -> GameRecord:
    base = RoundState.deal(seed=seed, level_rank=Rank.FIVE, leader_seat=0)
    state = _rotate_state(base, seat_rotation)
    policy_seat = int(seat_rotation) % 4
    rng = random.Random(seed ^ (hash(opponent) & 0xFFFFFFFF) ^ (seat_rotation << 20))
    for _ in range(max_steps):
        if state.done:
            break
        if state.active_seat == policy_seat:
            strategy = "snapshot"
        elif TEAM_OF[state.active_seat] != TEAM_OF[policy_seat]:
            strategy = opponent
        else:
            strategy = "rule"
        action = _choose_action(state, strategy, rng)
        apply_action(state, action)
        state.check_invariants()
    if not state.done:
        raise RuntimeError(f"game failed to terminate within {max_steps} committed steps")
    return GameRecord(
        opponent=opponent, deal_group=int(deal_group), seat_rotation=int(seat_rotation), policy_seat=policy_seat,
        seed=int(seed), done=bool(state.done), ranking=tuple(state.ranking), winner_team=int(state.winner_team),
        outcome_class=str(state.outcome_class), rewards=tuple(float(x) for x in state.rewards),
        team_rewards=tuple(float(x) for x in state.team_rewards), committed_steps=int(state.committed_step), terminal=True,
    )


def evaluate_profile(*, profile: str = "local_fast", seed: int = EVALUATION_SEED_BASE, max_hours: float | None = None, evidence_path: str | Path | None = None) -> EvaluationResult:
    resolved = load_execution_profile(Path(__file__).resolve().parents[1], profile)
    config = resolved["config"]["evaluation"]
    groups = int(config["deal_groups_per_pairing"]); rotations = tuple(config["seat_rotations"]); opponents = tuple(config["opponents"])
    target = groups * len(rotations)
    budget = float(config["max_hours"] if max_hours is None else max_hours)
    started = time.monotonic(); records: list[GameRecord] = []; errors: list[str] = []
    for opponent in opponents:
        for group in range(groups):
            for rotation in rotations:
                elapsed = time.monotonic() - started
                if elapsed >= budget * 3600:
                    raise EvaluationIncomplete({"status": "incomplete", "profile": profile, "completed": len(records), "target": target * len(opponents), "errors": ["max_hours"]})
                game_seed = int(seed + group * 100 + rotation)
                try:
                    record = play_game(opponent=opponent, deal_group=group, seat_rotation=rotation, seed=game_seed)
                    records.append(record)
                except Exception as exc:
                    errors.append(f"{opponent}/{group}/{rotation}: {type(exc).__name__}: {exc}")
                    raise
    counts = {opponent: sum(record.opponent == opponent for record in records) for opponent in opponents}
    result = EvaluationResult(profile, "complete", target, counts, len(records), tuple(records), time.monotonic() - started, int(seed), tuple(errors))
    if any(value != target for value in counts.values()) or len(records) != target * len(opponents):
        raise EvaluationIncomplete(result.as_dict())
    if evidence_path is not None:
        path = Path(evidence_path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["EvaluationIncomplete", "EvaluationResult", "GameRecord", "evaluate_profile", "play_game", "EVALUATION_SEED_BASE"]
