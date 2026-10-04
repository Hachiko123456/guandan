"""Auditable complete-round regression runner for the A02 defect fixes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guandan.cards import Rank
from guandan.round import apply_action, return_cards_for, tribute_card_for
from guandan.state import CommittedAction, Phase, PreviousHandResult, RoundState, TEAM_OF

PREVIOUS_HEAD_THIRD = PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD")


def _partition_ok(state: RoundState) -> bool:
    hand_ids = [card.card_id for hand in state.hands for card in hand]
    played_ids = [card.card_id for action in state.history if getattr(action.kind, "value", action.kind) not in {"tribute", "return", "pass"} for card in action.cards]
    escrow_ids = [card.card_id for card in state.tribute_escrow]
    all_ids = hand_ids + played_ids + escrow_ids
    return len(all_ids) == len(set(all_ids)) and set(all_ids) == set(range(108))


def _settlement_checks(state: RoundState, previous_result: PreviousHandResult | None) -> dict[str, Any]:
    tribute_actions = [action for action in state.history if action.kind == "tribute"]
    return_actions = [action for action in state.history if action.kind == "return"]
    if previous_result is None or state.anti_tribute:
        tribute_ok = not tribute_actions and not return_actions and not state.tribute_transfers and not state.return_transfers
    else:
        expected = len(state.tribute_donors)
        tribute_ok = len(tribute_actions) == expected and len(return_actions) == expected and len(state.tribute_transfers) == expected and len(state.return_transfers) == expected and not state.tribute_escrow
    ranking_ok = state.done and sorted(state.ranking) == [1, 2, 3, 4]
    winner_ok = ranking_ok and state.winner_team == TEAM_OF[state.ranking.index(1)]
    reward_ok = winner_ok and list(state.rewards) == [1.0 if TEAM_OF[seat] == state.winner_team else -1.0 for seat in range(4)] and list(state.team_rewards) == [1.0 if team == state.winner_team else -1.0 for team in range(2)]
    return {"tribute_return_ok": tribute_ok, "ranking_ok": ranking_ok, "reward_ok": reward_ok}


def run_game(seed: int, *, level_rank: Rank = Rank.FIVE, previous_result: PreviousHandResult | None = PREVIOUS_HEAD_THIRD, max_steps: int = 512) -> dict[str, Any]:
    state = RoundState.deal(seed=seed, level_rank=level_rank, previous_result=previous_result)
    illegal_actions: list[str] = []
    step = 0
    while not state.done and step < max_steps:
        if state.phase is Phase.TRIBUTE:
            card = tribute_card_for(state, state.active_seat)
            if card is None:
                illegal_actions.append("tribute candidate unexpectedly empty")
                break
            action = CommittedAction(state.active_seat, "tribute", (card,))
        elif state.phase is Phase.RETURN:
            candidates = return_cards_for(state, state.active_seat)
            if not candidates:
                illegal_actions.append("return candidate unexpectedly empty")
                break
            action = CommittedAction(state.active_seat, "return", (candidates[0],))
        elif state.current_winning is not None:
            action = CommittedAction(state.active_seat, "pass")
        else:
            if not state.hands[state.active_seat]:
                illegal_actions.append("active seat has no cards while leading")
                break
            action = CommittedAction(state.active_seat, "single", (state.hands[state.active_seat][0],))
        try:
            apply_action(state, action)
            state.check_invariants()
        except Exception as exc:  # pragma: no cover - release evidence guard
            illegal_actions.append(f"{type(exc).__name__}: {exc}")
            break
        step += 1
    terminal = bool(state.done)
    truncated = not terminal
    checks = {"terminal_or_controlled_truncation": terminal or (truncated and step >= max_steps), "no_illegal_actions": not illegal_actions, "card_partition_ok": _partition_ok(state), **_settlement_checks(state, previous_result)}
    return {"seed": seed, "level_rank": level_rank.name, "steps": step, "terminal": terminal, "truncated": truncated, "truncation_reason": None if terminal else ("max_steps" if step >= max_steps else "policy_error"), "illegal_actions": illegal_actions, "final_hand_sizes": [len(hand) for hand in state.hands], "ranking": list(state.ranking), "winner_team": state.winner_team, "outcome_class": state.outcome_class, "rewards": [float(value) for value in state.rewards], "team_rewards": [float(value) for value in state.team_rewards], "committed_steps": state.committed_step, "tribute_actions": sum(action.kind == "tribute" for action in state.history), "return_actions": sum(action.kind == "return" for action in state.history), "anti_tribute": bool(state.anti_tribute), "checks": checks, "all_checks_passed": all(checks.values())}


def run_regression(count: int, *, start_seed: int = 0, level_rank: Rank = Rank.FIVE, previous_result: PreviousHandResult | None = PREVIOUS_HEAD_THIRD, max_steps: int = 512) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    games = [run_game(start_seed + offset, level_rank=level_rank, previous_result=previous_result, max_steps=max_steps) for offset in range(count)]
    failed = [game for game in games if not game["all_checks_passed"]]
    return {"scope": "complete_single_hand_games", "count_requested": count, "count_completed": sum(game["terminal"] for game in games), "count_truncated": sum(game["truncated"] for game in games), "count_with_illegal_actions": sum(bool(game["illegal_actions"]) for game in games), "count_failed_checks": len(failed), "seed_range": [start_seed, start_seed + count - 1], "level_rank": level_rank.name, "previous_result": None if previous_result is None else {"ranking": list(previous_result.ranking), "winner_team": previous_result.winner_team, "outcome_class": previous_result.outcome_class}, "max_steps": max_steps, "games": games, "passed": not failed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_regression(args.count, start_seed=args.start_seed, max_steps=args.max_steps)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in ("count_requested", "count_completed", "count_truncated", "count_with_illegal_actions", "count_failed_checks", "passed")}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
