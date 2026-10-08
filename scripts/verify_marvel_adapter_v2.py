"""Generate compact machine-readable M02 adapter verification evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from guandan.action_state import FAMILY_TOKEN_BASE, TokenCodec
from guandan.cards import NORMAL_RANKS, Rank
from guandan.combos import CombinationKind
from guandan.environment_types import GameConfig
from guandan.state import Phase, PreviousHandResult
from guandan_marvel import game

PREVIOUS = PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN")


def choose_row(step) -> np.ndarray:
    mask = step.available_actions.weights[0] > 0
    rows = step.available_actions.tokens[0, mask]
    commits = step.available_actions.is_commit[0, mask]
    return rows[int(np.flatnonzero(commits)[0])] if commits.any() else rows[0]


def same_step(left, right) -> None:
    np.testing.assert_array_equal(left.rewards, right.rewards)
    np.testing.assert_array_equal(left.is_terminal, right.is_terminal)
    np.testing.assert_array_equal(left.player_ids, right.player_ids)
    np.testing.assert_array_equal(left.observations.tokens, right.observations.tokens)
    np.testing.assert_array_equal(left.observations.channels, right.observations.channels)
    np.testing.assert_array_equal(left.available_actions.tokens, right.available_actions.tokens)
    np.testing.assert_array_equal(left.available_actions.weights, right.available_actions.weights)
    np.testing.assert_array_equal(left.available_actions.is_commit, right.available_actions.is_commit)


def run_case(level: Rank, seed: int, previous: PreviousHandResult | None, verify_resume: bool) -> dict:
    config = GameConfig(level_rank=level, previous_result=previous, max_token_steps=game.max_game_horizon)
    batch = game.EnvBatch(1, configs=[config], auto_reset=False)
    step = batch.reset(seed)
    restored = None
    max_candidates = max_event_tokens = max_prefix_stage = commits = 0
    action_rows_checked = 0
    candidate_families: set[str] = set()
    executed_families: dict[str, int] = {}
    steps = 0
    started = time.perf_counter()
    while not bool(step.is_terminal[0]):
        mask = step.available_actions.weights[0] > 0
        rows = step.available_actions.tokens[0, mask]
        decoded = game.actions_from_np(rows)
        legal_v1 = tuple(int(token) for token in batch.envs[0].legal_tokens())
        if tuple(action.token_id for action in decoded) != legal_v1:
            raise AssertionError("V1/V2 legal-next action set/order mismatch")
        if len({action.row for action in decoded}) != len(decoded):
            raise AssertionError("structured action row collision")
        action_rows_checked += len(decoded)
        for candidate in decoded:
            if FAMILY_TOKEN_BASE <= candidate.token_id < FAMILY_TOKEN_BASE + len(CombinationKind):
                candidate_families.add(TokenCodec.family_from_token(candidate.token_id).value)
        max_candidates = max(max_candidates, len(decoded))
        max_prefix_stage = max(max_prefix_stage, *(action.prefix_stage for action in decoded))
        max_event_tokens = max(max_event_tokens, int(np.count_nonzero(step.observations.tokens, axis=-1).max()))
        row = choose_row(step)
        action = game.actions_from_np(row)[0]
        commits += int(action.is_commit)
        before_committed = batch.envs[0].committed_step
        step = batch.step((action,))
        steps += 1
        if batch.envs[0].committed_step > before_committed:
            last = batch.envs[0]._require_protocol().round_state.history[-1]
            if isinstance(last.kind, CombinationKind):
                executed_families[last.kind.value] = executed_families.get(last.kind.value, 0) + 1
        if verify_resume and steps == 40:
            restored = game.EnvBatch.deserialize(batch.serialize())
            same_step(batch.observe(), restored.observe())
        if restored is not None:
            restored_step = restored.step(game.actions_from_np(row.copy())) if steps > 40 else restored.observe()
            if steps > 40:
                same_step(step, restored_step)
                if batch.serialize() != restored.serialize():
                    raise AssertionError("serialized continuation diverged")
        if steps > game.max_game_horizon:
            raise AssertionError("game exceeded configured horizon")

    state = batch.envs[0]._require_protocol().round_state
    state.check_invariants()
    remaining = [card.card_id for hand in state.hands for card in hand]
    played = [
        card.card_id for action in state.history
        if action.kind not in ("pass", "tribute", "return") for card in action.cards
    ]
    if len(remaining) + len(played) != 108 or len(set(remaining + played)) != 108:
        raise AssertionError("terminal physical-card conservation failed")
    return {
        "level_rank": int(level),
        "seed": seed,
        "previous_result": previous is not None,
        "steps": steps,
        "commits": commits,
        "action_rows_checked": action_rows_checked,
        "candidate_families": sorted(candidate_families),
        "executed_families": dict(sorted(executed_families.items())),
        "max_candidates": max_candidates,
        "max_event_tokens": max_event_tokens,
        "max_prefix_stage": max_prefix_stage,
        "ranking": list(state.ranking),
        "winner_team": state.winner_team,
        "outcome_class": state.outcome_class,
        "reward": step.rewards[0].tolist(),
        "real_terminal": bool(state.done and state.phase is Phase.TERMINAL and step.is_terminal[0]),
        "card_conservation": True,
        "resume_continuation_checked": verify_resume,
        "duration_seconds": round(time.perf_counter() - started, 6),
    }


def git(*args: str, cwd: str = ".") -> str:
    return subprocess.check_output(["git", "-C", cwd, *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cases = [run_case(level, index, None, index == 0) for index, level in enumerate(NORMAL_RANKS)]
    cases.extend([
        run_case(Rank.SIX, 101, PREVIOUS, False),
        run_case(Rank.TWO, 102, PREVIOUS, False),
    ])
    evidence = {
        "schema": "guandan-marvel-m02-verification-v1",
        "stage": "M02_guandan_adapter_v2",
        "status": "passed",
        "backend": game.BACKEND,
        "protocols": {
            "action": game.ACTION_VERSION,
            "environment": game.ENV_VERSION,
            "encoding": game.ENCODING_VERSION,
        },
        "game_spec": {
            "identifier": game.GAME_IDENTIFIER,
            "num_players": game.num_players,
            "num_tokens": game.num_tokens,
            "num_channels": game.num_channels,
            "max_game_horizon": game.max_game_horizon,
            "max_available_actions": game.max_available_actions,
            "max_observation_tokens": game.max_observation_tokens,
            "max_action_tokens": game.max_action_tokens,
        },
        "capacity_basis": {
            "available_actions": "one row per unique V1 legal token; V1 vocabulary has 256 token IDs",
            "action_row": "fixed eight-field reversible schema",
            "observation_event": "max(32-token V1 action + 5 framing, four 5-token exchange records + framing) <= 48",
            "horizon": "inherits V1 max_token_steps=4096 and hard-fails before nonterminal truncation",
        },
        "summary": {
            "complete_games": len(cases),
            "all_13_level_ranks": len({case["level_rank"] for case in cases}) == 13,
            "exchange_games": sum(case["previous_result"] for case in cases),
            "total_steps": sum(case["steps"] for case in cases),
            "total_commits": sum(case["commits"] for case in cases),
            "action_rows_checked": sum(case["action_rows_checked"] for case in cases),
            "candidate_families": sorted({family for case in cases for family in case["candidate_families"]}),
            "executed_families": {
                family: sum(case["executed_families"].get(family, 0) for case in cases)
                for family in sorted({family for case in cases for family in case["executed_families"]})
            },
            "max_observed_candidates": max(case["max_candidates"] for case in cases),
            "max_observed_event_tokens": max(case["max_event_tokens"] for case in cases),
            "max_observed_prefix_stage": max(case["max_prefix_stage"] for case in cases),
            "min_steps": min(case["steps"] for case in cases),
            "max_steps": max(case["steps"] for case in cases),
            "all_real_terminal": all(case["real_terminal"] for case in cases),
            "all_card_conservation": all(case["card_conservation"] for case in cases),
            "resume_continuation_checked": any(case["resume_continuation_checked"] for case in cases),
        },
        "cases": cases,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "source": {
            "branch": git("branch", "--show-current"),
            "head_before_stage_commit": git("rev-parse", "HEAD"),
            "reference_commit": git("rev-parse", "HEAD", cwd=r"D:\project\doudizhu"),
            "reference_worktree_clean": git("status", "--short", cwd=r"D:\project\doudizhu") == "",
        },
        "duration_seconds": round(time.perf_counter() - started, 6),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
