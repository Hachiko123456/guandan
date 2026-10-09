"""Generate focused M04 repair verification evidence.

This verifier covers the repaired public action interface, strict deserialization,
real MARVEL Sampler integration, and RNG continuation through auto-reset. It is
not the M05 large-scale differential/performance gate.
"""
from __future__ import annotations

import argparse
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
import pybind11
import torch

from guandan_marvel import cpp_game
from guandan_marvel.buffer import trajectory_memory_budget
from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig
from guandan_marvel.model import Agent
from guandan_marvel.runtime.sampler import Sampler, SamplerConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_bytes(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _first_rows(step) -> np.ndarray:
    result = []
    for slot in range(step.rewards.shape[0]):
        legal = np.flatnonzero(step.available_actions.weights[slot] > 0)
        result.append(step.available_actions.tokens[slot, int(legal[0])])
    return np.asarray(result, dtype=np.int64)


def _commit_first_row(step) -> np.ndarray:
    legal = step.available_actions.weights[0] > 0
    commits = legal & step.available_actions.is_commit[0]
    index = int(np.flatnonzero(commits if commits.any() else legal)[0])
    return step.available_actions.tokens[[0], [index]]


def _step_digest(step) -> str:
    digest = hashlib.sha256()
    for array in (
        step.rewards,
        step.is_terminal,
        step.player_ids,
        step.observations.tokens,
        step.observations.channels,
        step.available_actions.tokens,
        step.available_actions.weights,
        step.available_actions.is_commit,
    ):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def verify_action_interface() -> dict[str, object]:
    records = []
    for batch_size in (1, 3):
        source = cpp_game.EnvBatch(batch_size, auto_reset=False)
        initial = source.reset(700 + batch_size)
        payload = source.serialize()
        raw_env = cpp_game.EnvBatch.deserialize(payload)
        decoded_env = cpp_game.EnvBatch.deserialize(payload)
        rows = _first_rows(initial)
        raw = raw_env.step(rows)
        decoded_actions = cpp_game.actions_from_np(rows)
        decoded = decoded_env.step(decoded_actions)
        if raw_env.serialize() != decoded_env.serialize() or _step_digest(raw) != _step_digest(decoded):
            raise AssertionError("raw and decoded action transitions differ")
        records.append({
            "batch_size": batch_size,
            "decoded_action_count": len(decoded_actions),
            "transition_sha256": _step_digest(raw),
            "serialized_equal": True,
        })

    atomic = cpp_game.EnvBatch(3, auto_reset=False)
    step = atomic.reset(811)
    rows = _first_rows(step)
    before = atomic.serialize()
    forged = rows.copy()
    forged[1, 2] += 1
    try:
        atomic.step(forged)
    except cpp_game.ProtocolError as error:
        rejection = {"type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("forged multi-slot action was accepted")
    if atomic.serialize() != before:
        raise AssertionError("multi-slot failure committed partial state")
    return {"equivalence": records, "forged_multi_slot_rejection": rejection, "failure_atomic": True}


def verify_deserialization() -> dict[str, object]:
    env = cpp_game.EnvBatch(1, auto_reset=False)
    env.reset(901)
    base = json.loads(env.serialize())
    valid_rng = base["games"][0]["rng_state"]
    cases = {
        "empty_rng": lambda data: data["games"][0].__setitem__("rng_state", ""),
        "invalid_rng": lambda data: data["games"][0].__setitem__("rng_state", "NOT_A_VALID_RNG_STATE"),
        "truncated_rng": lambda data: data["games"][0].__setitem__("rng_state", valid_rng[: len(valid_rng) // 2]),
        "trailing_rng_garbage": lambda data: data["games"][0].__setitem__("rng_state", valid_rng + " GARBAGE"),
        "negative_trajectory": lambda data: data.__setitem__("trajectory_steps", [-1]),
        "over_horizon_trajectory": lambda data: data.__setitem__("trajectory_steps", [4097]),
        "mismatched_trajectory": lambda data: data.__setitem__("trajectory_steps", [123]),
        "batch_length": lambda data: data.__setitem__("events", []),
        "game_horizon": lambda data: data["games"][0]["config"].__setitem__("max_token_steps", 512),
    }
    rejected = {}
    for name, mutate in cases.items():
        data = json.loads(json.dumps(base))
        mutate(data)
        try:
            cpp_game.EnvBatch.deserialize(_json_bytes(data))
        except Exception as error:
            rejected[name] = {"type": type(error).__name__, "message": str(error)}
        else:
            raise AssertionError(f"corruption accepted: {name}")

    whitespace = json.loads(json.dumps(base))
    whitespace["games"][0]["rng_state"] += " \n\t"
    if cpp_game.EnvBatch.deserialize(_json_bytes(whitespace)).serialize() != env.serialize():
        raise AssertionError("legal RNG trailing whitespace did not canonicalize")
    return {"rejected": rejected, "legal_rng_whitespace": True}


def verify_rng_auto_reset() -> dict[str, object]:
    left = cpp_game.EnvBatch(1, auto_reset=True)
    step = left.reset(1001)
    step = left.step(_commit_first_row(step))
    right = cpp_game.EnvBatch.deserialize(left.serialize())
    right_step = right.observe()
    for token_steps in range(1, 4097):
        row = _commit_first_row(step)
        step = left.step(row)
        right_step = right.step(cpp_game.actions_from_np(row))
        if _step_digest(step) != _step_digest(right_step):
            raise AssertionError("restored transition diverged before auto-reset")
        if bool(step.is_terminal[0]):
            break
    else:
        raise AssertionError("RNG continuation game did not terminate")
    left_state = left.debug_state(0)
    right_state = right.debug_state(0)
    if left_state["state"]["hands"] != right_state["state"]["hands"]:
        raise AssertionError("post-auto-reset deal diverged")
    if left.serialize() != right.serialize():
        raise AssertionError("post-auto-reset serialized states diverged")
    hand_bytes = json.dumps(left_state["state"]["hands"], separators=(",", ":")).encode()
    return {
        "terminal_token_steps_after_restore": token_steps,
        "post_reset_token_step": left_state["token_step"],
        "post_reset_hands_sha256": hashlib.sha256(hand_bytes).hexdigest(),
        "serialized_equal": True,
    }


class _CppSamplerGame:
    identifier = "m04-repair-verifier-cpp-sampler"
    GAME_IDENTIFIER = identifier
    num_players = cpp_game.num_players
    num_tokens = cpp_game.num_tokens
    pad_token = cpp_game.pad_token
    num_channels = cpp_game.num_channels
    max_game_horizon = 512
    max_observation_tokens = cpp_game.max_observation_tokens
    max_action_tokens = cpp_game.max_action_tokens
    max_available_actions = cpp_game.max_available_actions
    protocol_version = cpp_game.protocol_version
    actions_from_np = staticmethod(cpp_game.actions_from_np)

    @staticmethod
    def EnvBatch(batch_size: int):
        return cpp_game.EnvBatch(batch_size, max_horizon=512, auto_reset=True)


def _agent() -> Agent:
    config = TrainConfig(
        game=GameSpec.from_game(_CppSamplerGame),
        device="cpu",
        model=ModelConfig(
            architecture="v1.3", history_dim=8, feature_dim=8,
            decoder_layers=1, decoder_heads=1,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    return Agent(config)


def _trajectory_record(trajectory, index: int) -> dict[str, object]:
    steps = int(trajectory["num_steps"][index])
    terminal = trajectory["is_terminal"][index, :steps]
    if steps <= 0 or steps >= 512 or int(terminal.sum()) != 1 or not bool(terminal[-1]):
        raise AssertionError("trajectory is not one complete real terminal")
    if not bool(trajectory["on_policy_valid"][index, :steps].all()):
        raise AssertionError("real on-policy trajectory was marked invalid")
    return {
        "steps": steps,
        "terminal_count": int(terminal.sum()),
        "terminal_reward": trajectory["rewards"][index, steps - 1].tolist(),
        "policy_finite": bool(torch.isfinite(trajectory["policy_probs"][index, :steps]).all()),
        "behavior_probability_finite": bool(torch.isfinite(trajectory["behavior_action_prob"][index, :steps]).all()),
        "on_policy_valid": True,
    }


def verify_sampler() -> dict[str, object]:
    budget = trajectory_memory_budget(
        GameSpec.from_game(_CppSamplerGame), rollout_batch_size=2, collect_batch_size=2, world_size=1,
    )
    if int(budget["sampler_bytes_per_rank"]) >= 128 * 2**20:
        raise AssertionError("sampler smoke exceeds memory budget")
    torch.manual_seed(20261012)
    agent = _agent()
    sampler = Sampler(config=SamplerConfig(
        game=_CppSamplerGame, device="cpu", rollout_batch_size=2,
        collect_batch_size=2, sampling_temperature=1.0,
        purpose="on_policy", env_seed=1202,
    ))
    first = sampler.collect(agent_model=agent)
    first_records = [_trajectory_record(first, index) for index in range(2)]
    after_first = {"total_env_steps": sampler.total_env_steps, "total_game_steps": sampler.total_game_steps}
    second = sampler.collect(agent_model=agent)
    second_records = [_trajectory_record(second, index) for index in range(2)]
    if sampler.failed or sampler.total_env_steps <= after_first["total_env_steps"] or sampler.total_game_steps <= after_first["total_game_steps"]:
        raise AssertionError("sampler did not continue cleanly after auto-reset")
    return {
        "memory_budget": budget,
        "first_collect": first_records,
        "second_collect": second_records,
        "first_collect_staggered": len({item["steps"] for item in first_records}) > 1,
        "counters_after_first": after_first,
        "counters_after_second": {"total_env_steps": sampler.total_env_steps, "total_game_steps": sampler.total_game_steps},
        "sampler_failed": sampler.failed,
        "live_player_ids_after_auto_reset": sampler.parsed_envs_step["player_ids"].tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import nanopokerlib._guandan_cpp as extension
    extension_path = Path(extension.__file__).resolve()
    started = time.perf_counter()
    payload = {
        "schema": "guandan-marvel-m04-cpp-runtime-repair-verification-v1",
        "stage": "M04_cpp_runtime_repair",
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pybind11": pybind11.__version__,
            "extension_path": str(extension_path),
            "extension_bytes": extension_path.stat().st_size,
            "extension_sha256": _sha256(extension_path),
        },
        "action_interface": verify_action_interface(),
        "deserialization": verify_deserialization(),
        "rng_auto_reset_continuation": verify_rng_auto_reset(),
        "sampler": verify_sampler(),
        "scope_note": "M04 repair verification only; M05 10k/100k differential and >=2x gate are not claimed.",
    }
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "elapsed_seconds": payload["elapsed_seconds"],
        "first_collect_steps": [item["steps"] for item in payload["sampler"]["first_collect"]],
        "second_collect_steps": [item["steps"] for item in payload["sampler"]["second_collect"]],
        "corruptions_rejected": len(payload["deserialization"]["rejected"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
