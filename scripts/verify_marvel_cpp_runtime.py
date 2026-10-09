"""Generate machine-readable M04 C++ runtime verification evidence.

This is a local functional/smoke verifier. It deliberately does not implement
or claim the M05 10k/100k differential and performance gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from guandan.action_state import StepwiseActionState, canonical_tokens
from guandan.cards import Card, Rank, effective_rank_value
from guandan.combos import recognize_combinations
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan.state import RoundState
from guandan_marvel import python_game
from guandan_marvel.testing.complex_actions import DIRECTED_ACTION_CASES, build_directed_action
from nanopokerlib import guandan as cpp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_for_token(step, token_id: int) -> np.ndarray:
    live = step.available_actions.weights[0] > 0
    matches = np.flatnonzero(live & (step.available_actions.tokens[0, :, 5] == token_id))
    if len(matches) != 1:
        raise AssertionError((token_id, matches.tolist()))
    return step.available_actions.tokens[[0], matches]


def signature_python(declaration):
    return (
        declaration.kind.value,
        tuple(card.card_id for card in declaration.cards),
        int(declaration.comparison_rank),
        tuple(declaration.comparison_key),
        tuple((card_id, int(rank), int(suit), card_id in declaration.natural_wild_ids)
              for card_id, (rank, suit) in sorted(declaration.wild_assignments.items())),
        tuple(declaration.natural_wild_ids),
    )


def signature_cpp(declaration):
    return (
        declaration.kind,
        tuple(declaration.card_ids),
        declaration.comparison_rank,
        tuple(declaration.comparison_key),
        tuple((item.card_id, int(item.rank), int(item.suit), item.natural)
              for item in declaration.wild_assignments),
        tuple(declaration.natural_wild_ids),
    )


def directed_verification() -> list[dict[str, object]]:
    records = []
    for case in DIRECTED_ACTION_CASES:
        py_values = {
            signature_python(item)
            for item in recognize_combinations(tuple(Card(card_id) for card_id in case.card_ids), level_rank=Rank.FOUR)
        }
        cpp_values = {
            signature_cpp(item)
            for item in cpp.recognize_combinations(list(case.card_ids), int(Rank.FOUR))
        }
        if py_values != cpp_values:
            raise AssertionError(f"combination mismatch: {case.name}")
        state, action = build_directed_action(case)
        env = GuandanEnv(GameConfig(level_rank=Rank.FOUR, leader_seat=0, max_token_steps=4096))
        env.reset(initial_state=StepwiseActionState(state).serialize())
        oracle = python_game.EnvBatch.from_envs([env], auto_reset=False)
        expected = oracle.observe()
        runtime = cpp.EnvBatch(1, auto_reset=False)
        actual = runtime.reset_from_deals(
            [[[card.card_id for card in hand] for hand in state.hands]],
            level_ranks=[int(Rank.FOUR)], leader_seats=[0],
        )
        maximum = 0
        sequence = canonical_tokens(action)
        for token_id in sequence:
            maximum = max(maximum, int(actual.available_actions.weights.sum()))
            for name, left, right in (
                ("actions", actual.available_actions.tokens, expected.available_actions.tokens),
                ("channels", actual.observations.channels, expected.observations.channels),
                ("events", actual.observations.tokens, expected.observations.tokens),
            ):
                if not np.array_equal(left, right):
                    raise AssertionError(f"{case.name}: {name} mismatch")
            row = row_for_token(actual, token_id)
            expected = oracle.step(row)
            actual = runtime.step(row)
        records.append({
            "name": case.name,
            "family": case.family.value,
            "canonical_tokens": list(sequence),
            "wild_mode": case.wild_mode,
            "max_legal_next_rows": maximum,
            "cpp_python_equal": True,
        })
    return records


def full_game_smokes() -> list[dict[str, object]]:
    records = []
    for index, (seed, level) in enumerate(((234, Rank.NINE), (991, Rank.THREE), (77, Rank.TWO))):
        state = RoundState.deal(seed=seed, level_rank=level)
        deal = [[card.card_id for card in hand] for hand in state.hands]
        env = GuandanEnv(GameConfig(level_rank=level, leader_seat=0, max_token_steps=4096))
        env.reset(initial_state=StepwiseActionState(state).serialize())
        oracle = python_game.EnvBatch.from_envs([env], auto_reset=False)
        expected = oracle.observe()
        runtime = cpp.EnvBatch(1, auto_reset=False)
        actual = runtime.reset_from_deals([deal], level_ranks=[int(level)], leader_seats=[0])
        rng = np.random.default_rng(1000 + index)
        max_actions = 0
        for token_step in range(4096):
            for name, left, right in (
                ("actions", actual.available_actions.tokens, expected.available_actions.tokens),
                ("weights", actual.available_actions.weights, expected.available_actions.weights),
                ("commit", actual.available_actions.is_commit, expected.available_actions.is_commit),
                ("channels", actual.observations.channels, expected.observations.channels),
                ("events", actual.observations.tokens, expected.observations.tokens),
                ("players", actual.player_ids, expected.player_ids),
            ):
                if not np.array_equal(left, right):
                    raise AssertionError(f"full game {index} step {token_step}: {name} mismatch")
            legal = np.flatnonzero(actual.available_actions.weights[0] > 0)
            max_actions = max(max_actions, len(legal))
            row = actual.available_actions.tokens[[0], [int(rng.choice(legal))]]
            expected = oracle.step(row)
            actual = runtime.step(row)
            if not np.array_equal(actual.rewards, expected.rewards) or not np.array_equal(actual.is_terminal, expected.is_terminal):
                raise AssertionError(f"full game {index} outcome mismatch")
            if bool(actual.is_terminal[0]):
                info = actual.infos[0]
                records.append({
                    "seed": seed,
                    "level_rank": int(level),
                    "token_steps": token_step + 1,
                    "ranking": list(info["ranking"]),
                    "outcome_class": info["outcome_class"],
                    "max_legal_next_rows": max_actions,
                    "cpp_python_equal": True,
                })
                break
        else:
            raise AssertionError("full game did not terminate")
    return records


def batch_smoke(batch_size: int) -> dict[str, object]:
    batch = cpp.EnvBatch(batch_size, auto_reset=True)
    step = batch.reset(20261009)
    rng = np.random.default_rng(42)
    terminal_counts = np.zeros(batch_size, dtype=np.int64)
    token_transitions = 0
    started = time.perf_counter()
    for batch_step in range(4096):
        rows = []
        for slot in range(batch_size):
            legal = np.flatnonzero(step.available_actions.weights[slot] > 0)
            rows.append(step.available_actions.tokens[slot, int(rng.choice(legal))])
        step = batch.step(np.asarray(rows, dtype=np.int64))
        terminal_counts += step.is_terminal.astype(np.int64)
        token_transitions += batch_size
        if np.all(terminal_counts > 0):
            break
    else:
        raise AssertionError("batch smoke did not terminate every slot")
    elapsed = time.perf_counter() - started
    critic = batch.critic_state()
    if not np.all(critic.ownership.sum(axis=1) + critic.escrow + critic.played == 1):
        raise AssertionError("batch card conservation failed")
    payload = batch.serialize()
    restored = cpp.EnvBatch.deserialize(payload)
    if restored.serialize() != payload:
        raise AssertionError("batch serialization is not byte-stable")
    return {
        "batch_size": batch_size,
        "batch_steps": batch_step + 1,
        "token_transitions": token_transitions,
        "elapsed_seconds": elapsed,
        "token_transitions_per_second": token_transitions / elapsed,
        "terminal_counts": terminal_counts.tolist(),
        "serialization_bytes": len(payload),
        "serialization_byte_stable": True,
        "card_conservation": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import nanopokerlib._guandan_cpp as extension_module
    extension_path = Path(extension_module.__file__).resolve()
    started = time.time()
    directed = directed_verification()
    games = full_game_smokes()
    batch = batch_smoke(args.batch_size)
    payload = {
        "schema": "guandan-marvel-m04-cpp-runtime-verification-v1",
        "stage": "M04_cpp_runtime",
        "timestamp_unix": started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "extension_path": str(extension_path),
            "extension_bytes": extension_path.stat().st_size,
            "extension_sha256": sha256(extension_path),
        },
        "protocol": {
            "backend": cpp.BACKEND,
            "identifier": cpp.GAME_IDENTIFIER,
            "action": cpp.ACTION_VERSION,
            "environment": cpp.ENV_VERSION,
            "encoding": cpp.ENCODING_VERSION,
            "serialization": cpp.SERIALIZATION_SCHEMA,
            "game_spec": asdict(cpp.GAME_SPEC),
        },
        "cards_and_levels": {
            "physical_cards": 108,
            "level_contexts": 13,
            "all_values_checked": all(
                cpp.effective_rank_value(int(rank), int(level)) == effective_rank_value(rank, level)
                for level in tuple(Rank(value) for value in range(3, 16))
                for rank in Rank
            ),
        },
        "directed_actions": directed,
        "full_game_smokes": games,
        "batch_smoke": batch,
        "scope_note": "Local M04 functional/smoke evidence only; not the M05 10k/100k differential or >=2x performance gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "directed_cases": len(directed),
        "full_games": len(games),
        "batch_terminal_counts": batch["terminal_counts"],
        "token_transitions_per_second": batch["token_transitions_per_second"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
