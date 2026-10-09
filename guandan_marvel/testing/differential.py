"""M05 Python/C++ differential helpers.

The Python adapter remains the rules oracle.  This module only constructs
matched deals, compares the versioned V2 wire, and writes replay data; it does
not change A01/A02 rule semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from guandan.action_state import COMMIT_TOKEN, PASS_TOKEN, StepwiseActionState
from guandan.cards import PRINTED_RANKS, Rank, decode_cards
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan.state import Phase, PreviousHandResult, RoundState
from guandan_marvel import python_game
from nanopokerlib import guandan as cpp_game

CORPUS_SCHEMA = "guandan-marvel-m05-state-action-corpus-v1"
FIXTURE_SCHEMA = "guandan-marvel-m05-mismatch-fixture-v1"
SUMMARY_SCHEMA = "guandan-marvel-m05-differential-summary-v1"
BASE_SEED = 5_050_000
POLICY_SEED = 5_051_000

_PREVIOUS_RESULTS: tuple[PreviousHandResult | None, ...] = (
    None,
    PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN"),
    PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD"),
    PreviousHandResult((1, 2, 4, 3), 0, "HEAD_LAST"),
    PreviousHandResult((3, 1, 4, 2), 1, "DOUBLE_DOWN"),
    PreviousHandResult((2, 1, 4, 3), 1, "HEAD_THIRD"),
    PreviousHandResult((2, 1, 3, 4), 1, "HEAD_LAST"),
    PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN", tribute_cancelled=True),
)


class DifferentialMismatch(AssertionError):
    """A replayable semantic mismatch between the two backends."""

    def __init__(self, category: str, step_index: int, details: Mapping[str, Any]):
        super().__init__(f"{category} mismatch at state {step_index}")
        self.category = category
        self.step_index = int(step_index)
        self.details = dict(details)


@dataclass(frozen=True, slots=True)
class DifferentialCase:
    game_index: int
    seed: int
    level_rank: int
    leader_seat: int
    previous_result: dict[str, Any] | None
    deal: tuple[tuple[int, ...], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "game_index": self.game_index,
            "seed": self.seed,
            "level_rank": self.level_rank,
            "leader_seat": self.leader_seat,
            "previous_result": self.previous_result,
            "deal": [list(hand) for hand in self.deal],
        }


def previous_result_to_dict(value: PreviousHandResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "ranking": list(value.ranking),
        "winner_team": int(value.winner_team),
        "outcome_class": value.outcome_class,
        "tribute_cancelled": bool(value.tribute_cancelled),
    }


def previous_result_from_dict(value: Mapping[str, Any] | None) -> PreviousHandResult | None:
    if value is None:
        return None
    return PreviousHandResult(
        tuple(int(item) for item in value["ranking"]),
        int(value["winner_team"]),
        str(value["outcome_class"]),
        bool(value.get("tribute_cancelled", False)),
    )


def build_case(game_index: int, *, base_seed: int = BASE_SEED) -> DifferentialCase:
    game_index = int(game_index)
    seed = int(base_seed) + game_index
    level_rank = PRINTED_RANKS[game_index % len(PRINTED_RANKS)]
    leader_seat = game_index % 4
    previous = _PREVIOUS_RESULTS[game_index % len(_PREVIOUS_RESULTS)]
    state = RoundState.deal(
        seed=seed,
        level_rank=level_rank,
        leader_seat=leader_seat,
        previous_result=previous,
    )
    return DifferentialCase(
        game_index=game_index,
        seed=seed,
        level_rank=int(level_rank),
        leader_seat=leader_seat,
        previous_result=previous_result_to_dict(previous),
        deal=tuple(tuple(card.card_id for card in hand) for hand in state.hands),
    )


def round_state_from_case(case: DifferentialCase) -> RoundState:
    return RoundState.from_hands(
        [list(decode_cards(hand)) for hand in case.deal],
        level_rank=Rank(case.level_rank),
        leader_seat=case.leader_seat,
        previous_result=previous_result_from_dict(case.previous_result),
    )


def python_batch_from_case(case: DifferentialCase, *, auto_reset: bool = False):
    state = round_state_from_case(case)
    env = GuandanEnv(GameConfig(
        level_rank=case.level_rank,
        leader_seat=case.leader_seat,
        previous_result=previous_result_from_dict(case.previous_result),
        max_token_steps=python_game.max_game_horizon,
    ))
    env.reset(initial_state=StepwiseActionState(state).serialize())
    batch = python_game.EnvBatch.from_envs([env], auto_reset=auto_reset)
    return batch, batch.observe()


def cpp_batch_from_case(case: DifferentialCase, *, auto_reset: bool = False):
    batch = cpp_game.EnvBatch(1, auto_reset=auto_reset)
    step = batch.reset_from_deals(
        [[list(hand) for hand in case.deal]],
        level_ranks=[case.level_rank],
        leader_seats=[case.leader_seat],
        previous_results=[case.previous_result],
    )
    return batch, step


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array_mismatch_details(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    details: dict[str, Any] = {
        "python_shape": list(expected.shape),
        "cpp_shape": list(actual.shape),
        "python_dtype": expected.dtype.str,
        "cpp_dtype": actual.dtype.str,
        "python_sha256": _array_digest(expected),
        "cpp_sha256": _array_digest(actual),
    }
    if expected.shape == actual.shape:
        locations = np.argwhere(expected != actual)
        examples = []
        for location in locations[:8]:
            index = tuple(int(value) for value in location)
            examples.append({
                "index": list(index),
                "python": np.asarray(expected[index]).item(),
                "cpp": np.asarray(actual[index]).item(),
            })
        details["difference_count"] = int(locations.shape[0])
        details["examples"] = examples
    return details


def _assert_array(category: str, step_index: int, expected: np.ndarray, actual: np.ndarray) -> None:
    if expected.dtype != actual.dtype or expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise DifferentialMismatch(category, step_index, _array_mismatch_details(expected, actual))


def _step_arrays(step: Any, critic: Any) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("rewards", np.asarray(step.rewards)),
        ("is_terminal", np.asarray(step.is_terminal)),
        ("player_ids", np.asarray(step.player_ids)),
        ("observations.tokens", np.asarray(step.observations.tokens)),
        ("observations.channels", np.asarray(step.observations.channels)),
        ("available_actions.tokens", np.asarray(step.available_actions.tokens)),
        ("available_actions.weights", np.asarray(step.available_actions.weights)),
        ("available_actions.is_commit", np.asarray(step.available_actions.is_commit)),
        ("critic.ownership", np.asarray(critic.ownership)),
        ("critic.escrow", np.asarray(critic.escrow)),
        ("critic.played", np.asarray(critic.played)),
    )


def compare_step(
    python_step: Any,
    cpp_step: Any,
    python_critic: Any,
    cpp_critic: Any,
    *,
    step_index: int,
) -> tuple[str, str]:
    python_arrays = _step_arrays(python_step, python_critic)
    cpp_arrays = _step_arrays(cpp_step, cpp_critic)
    python_digest = hashlib.sha256()
    cpp_digest = hashlib.sha256()
    for (name, expected), (cpp_name, actual) in zip(python_arrays, cpp_arrays, strict=True):
        if name != cpp_name:
            raise AssertionError("internal differential field order mismatch")
        _assert_array(name, step_index, expected, actual)
        for digest, value in ((python_digest, expected), (cpp_digest, actual)):
            digest.update(name.encode("utf-8"))
            digest.update(_array_digest(value).encode("ascii"))
    return python_digest.hexdigest(), cpp_digest.hexdigest()


def _terminal_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "termination_reason": info.get("termination_reason"),
        "ranking": list(info.get("ranking", ())),
        "winner_team": info.get("winner_team"),
        "outcome_class": info.get("outcome_class"),
        "team_reward": [float(value) for value in info.get("team_reward", ())],
        "finish_token_step": info.get("finish_token_step"),
        "finish_committed_step": info.get("finish_committed_step"),
    }


def _coverage_from_state(state: RoundState) -> dict[str, Any]:
    families: dict[str, int] = {}
    natural_wild = 0
    substituted_wild = 0
    for action in state.history:
        kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
        families[kind] = families.get(kind, 0) + 1
        if action.declaration is not None:
            natural_wild += len(action.declaration.natural_wild_ids)
            substituted_wild += sum(
                card_id not in action.declaration.natural_wild_ids
                for card_id in action.declaration.wild_assignments
            )
    return {
        "families": families,
        "natural_wild_assignments": natural_wild,
        "substituted_wild_assignments": substituted_wild,
        "anti_tribute": int(state.anti_tribute),
        "tribute_actions": families.get("tribute", 0),
        "return_actions": families.get("return", 0),
    }


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _write_mismatch_fixture(
    path: Path,
    *,
    case: DifferentialCase,
    steps: Sequence[Sequence[int]],
    restore_step: int,
    mismatch: DifferentialMismatch | BaseException,
) -> None:
    if isinstance(mismatch, DifferentialMismatch):
        category = mismatch.category
        step_index = mismatch.step_index
        details = mismatch.details
    else:
        category = "exception"
        step_index = len(steps)
        details = {"type": type(mismatch).__name__, "message": str(mismatch)}
    payload = {
        "schema": FIXTURE_SCHEMA,
        "case": case.to_json(),
        "restore_step": restore_step,
        "steps": [list(step) for step in steps],
        "mismatch": {
            "category": category,
            "state_index": step_index,
            "details": details,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_game(
    case: DifferentialCase,
    *,
    intermediate_quota: int,
    mismatch_dir: Path | None = None,
    policy_seed: int = POLICY_SEED,
) -> dict[str, Any]:
    python_batch, python_step = python_batch_from_case(case, auto_reset=False)
    cpp_batch, cpp_step = cpp_batch_from_case(case, auto_reset=False)
    rng = np.random.default_rng(int(policy_seed) + case.game_index)
    restore_step = 7 + (case.game_index * 17) % 53
    transitions: list[list[int]] = []
    state_digests: list[str] = []
    python_trajectory = hashlib.sha256()
    cpp_trajectory = hashlib.sha256()
    contact_count = 0
    serialization_roundtrips = 0
    compared_states = 0

    try:
        state_index = 0
        while True:
            python_critic = python_batch.critic_state()
            cpp_critic = cpp_batch.critic_state()
            python_digest, cpp_digest = compare_step(
                python_step,
                cpp_step,
                python_critic,
                cpp_critic,
                step_index=state_index,
            )
            compared_states += 1
            python_trajectory.update(python_digest.encode("ascii"))
            cpp_trajectory.update(cpp_digest.encode("ascii"))

            terminal = bool(python_step.is_terminal[0])
            if terminal != bool(cpp_step.is_terminal[0]):
                raise DifferentialMismatch("terminal", state_index, {
                    "python": terminal,
                    "cpp": bool(cpp_step.is_terminal[0]),
                })
            if terminal:
                python_terminal = _terminal_summary(python_step.infos[0])
                cpp_terminal = _terminal_summary(cpp_step.infos[0])
                if python_terminal != cpp_terminal:
                    raise DifferentialMismatch("terminal_info", state_index, {
                        "python": python_terminal,
                        "cpp": cpp_terminal,
                    })
                break

            state_digests.append(python_digest)
            legal = np.flatnonzero(python_step.available_actions.weights[0] > 0)
            if legal.size == 0:
                raise DifferentialMismatch("legal_actions_empty", state_index, {})
            action_index = int(rng.choice(legal))
            row = python_step.available_actions.tokens[0, action_index].copy()
            prefix = tuple(python_batch.envs[0]._require_protocol()._prefix_tokens)
            transitions.append([int(row[5]), *(int(token) for token in prefix)])
            python_trajectory.update(np.ascontiguousarray(row).tobytes())
            cpp_trajectory.update(np.ascontiguousarray(row).tobytes())

            before_state = python_batch.envs[0]._require_protocol().round_state
            contact_candidate = (
                int(row[5]) == PASS_TOKEN
                and before_state.current_winner_seat in before_state.emptied_seats
            )
            python_step = python_batch.step(row[None, :])
            cpp_step = cpp_batch.step(row[None, :])
            after_state = python_batch.envs[0]._require_protocol().round_state
            if contact_candidate and after_state.phase is Phase.PLAY and after_state.current_winning is None:
                contact_count += 1

            if len(transitions) == restore_step and not bool(python_step.is_terminal[0]):
                python_payload = python_batch.serialize()
                cpp_payload = cpp_batch.serialize()
                restored_python = python_game.EnvBatch.deserialize(python_payload)
                restored_cpp = cpp_game.EnvBatch.deserialize(cpp_payload)
                if restored_python.serialize() != python_payload:
                    raise DifferentialMismatch("python_serialization_roundtrip", state_index, {})
                if restored_cpp.serialize() != cpp_payload:
                    raise DifferentialMismatch("cpp_serialization_roundtrip", state_index, {})
                python_batch, cpp_batch = restored_python, restored_cpp
                python_step, cpp_step = python_batch.observe(), cpp_batch.observe()
                serialization_roundtrips += 1
            state_index += 1
    except BaseException as error:
        if mismatch_dir is not None:
            name = f"game-{case.game_index:05d}-state-{len(transitions):04d}.json"
            _write_mismatch_fixture(
                mismatch_dir / name,
                case=case,
                steps=transitions,
                restore_step=restore_step,
                mismatch=error,
            )
        raise

    if python_trajectory.hexdigest() != cpp_trajectory.hexdigest():
        mismatch = DifferentialMismatch("trajectory_digest", len(transitions), {
            "python": python_trajectory.hexdigest(),
            "cpp": cpp_trajectory.hexdigest(),
        })
        if mismatch_dir is not None:
            _write_mismatch_fixture(
                mismatch_dir / f"game-{case.game_index:05d}-trajectory.json",
                case=case,
                steps=transitions,
                restore_step=restore_step,
                mismatch=mismatch,
            )
        raise mismatch

    if intermediate_quota < 0 or intermediate_quota > len(state_digests):
        raise ValueError("intermediate_quota must fit the completed trajectory")
    sample_rng = np.random.default_rng(int(policy_seed) + 10_000_000 + case.game_index)
    sample_indices = sorted(int(value) for value in sample_rng.choice(
        len(state_digests), size=intermediate_quota, replace=False,
    )) if intermediate_quota else []
    sample_digest = hashlib.sha256()
    for index in sample_indices:
        sample_digest.update(f"{index}:{state_digests[index]}\n".encode("ascii"))

    final_state = python_batch.envs[0]._require_protocol().round_state
    coverage = _coverage_from_state(final_state)
    coverage["contact"] = contact_count
    return {
        "case": case.to_json(),
        "restore_step": restore_step,
        "serialization_roundtrips": serialization_roundtrips,
        "steps": transitions,
        "state_count": compared_states,
        "intermediate_sample_count": len(sample_indices),
        "intermediate_sample_digest": sample_digest.hexdigest(),
        "trajectory_digest": python_trajectory.hexdigest(),
        "terminal": _terminal_summary(python_step.infos[0]),
        "coverage": coverage,
    }


def corpus_record(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CORPUS_SCHEMA,
        **result["case"],
        "restore_step": result["restore_step"],
        "steps": result["steps"],
        "trajectory_digest": result["trajectory_digest"],
        "terminal": result["terminal"],
    }


def replay_corpus_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("schema") != CORPUS_SCHEMA:
        raise ValueError("unsupported M05 corpus schema")
    case = DifferentialCase(
        game_index=int(record["game_index"]),
        seed=int(record["seed"]),
        level_rank=int(record["level_rank"]),
        leader_seat=int(record["leader_seat"]),
        previous_result=record.get("previous_result"),
        deal=tuple(tuple(int(card) for card in hand) for hand in record["deal"]),
    )
    python_batch, python_step = python_batch_from_case(case, auto_reset=False)
    cpp_batch, cpp_step = cpp_batch_from_case(case, auto_reset=False)
    trajectory = hashlib.sha256()
    for state_index, encoded in enumerate(record["steps"]):
        python_critic = python_batch.critic_state()
        cpp_critic = cpp_batch.critic_state()
        python_digest, _ = compare_step(
            python_step, cpp_step, python_critic, cpp_critic, step_index=state_index,
        )
        trajectory.update(python_digest.encode("ascii"))
        action_token, *expected_prefix = [int(value) for value in encoded]
        actual_prefix = list(python_batch.envs[0]._require_protocol()._prefix_tokens)
        if actual_prefix != expected_prefix:
            raise DifferentialMismatch("corpus_prefix", state_index, {
                "expected": expected_prefix,
                "actual": actual_prefix,
            })
        legal = python_step.available_actions.weights[0] > 0
        matches = np.flatnonzero(legal & (python_step.available_actions.tokens[0, :, 5] == action_token))
        if len(matches) != 1:
            raise DifferentialMismatch("corpus_action", state_index, {
                "token_id": action_token,
                "matches": matches.tolist(),
            })
        row = python_step.available_actions.tokens[0, int(matches[0])].copy()
        trajectory.update(np.ascontiguousarray(row).tobytes())
        python_step = python_batch.step(row[None, :])
        cpp_step = cpp_batch.step(row[None, :])
    final_index = len(record["steps"])
    python_digest, _ = compare_step(
        python_step, cpp_step, python_batch.critic_state(), cpp_batch.critic_state(),
        step_index=final_index,
    )
    trajectory.update(python_digest.encode("ascii"))
    if not bool(python_step.is_terminal[0]):
        raise DifferentialMismatch("corpus_not_terminal", final_index, {})
    if trajectory.hexdigest() != record["trajectory_digest"]:
        raise DifferentialMismatch("corpus_trajectory_digest", final_index, {
            "expected": record["trajectory_digest"],
            "actual": trajectory.hexdigest(),
        })
    return {
        "game_index": case.game_index,
        "steps": len(record["steps"]),
        "trajectory_digest": trajectory.hexdigest(),
    }


def merge_coverage(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    totals = {
        "natural_wild_assignments": 0,
        "substituted_wild_assignments": 0,
        "anti_tribute": 0,
        "tribute_actions": 0,
        "return_actions": 0,
        "contact": 0,
    }
    for result in results:
        coverage = result["coverage"]
        _merge_counts(families, coverage["families"])
        for name in totals:
            totals[name] += int(coverage.get(name, 0))
    return {"families": families, **totals}


__all__ = [
    "BASE_SEED", "CORPUS_SCHEMA", "DifferentialCase", "DifferentialMismatch",
    "FIXTURE_SCHEMA", "POLICY_SEED", "SUMMARY_SCHEMA", "build_case",
    "compare_step", "corpus_record", "cpp_batch_from_case", "merge_coverage",
    "previous_result_from_dict", "previous_result_to_dict", "python_batch_from_case",
    "replay_corpus_record", "round_state_from_case", "run_game",
]
