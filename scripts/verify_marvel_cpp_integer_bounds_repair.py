"""Focused evidence generator for the M04 action-row integer-bounds repair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from guandan_marvel import cpp_game, game as python_game


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reset_rows(game, batch_size: int = 1, seed: int = 800):
    env = game.EnvBatch(batch_size, auto_reset=False)
    step = env.reset(seed)
    rows = []
    for slot in range(batch_size):
        index = int(np.flatnonzero(step.available_actions.weights[slot] > 0)[0])
        rows.append(step.available_actions.tokens[slot, index])
    return env, np.asarray(rows)


def reject_cpp(rows: np.ndarray, batch_size: int = 1) -> dict[str, object]:
    results = {}
    for entry in ("decoded", "raw"):
        env, _ = reset_rows(cpp_game, batch_size=batch_size)
        before = env.serialize()
        try:
            if entry == "decoded":
                env.step(cpp_game.actions_from_np(rows))
            else:
                env.step(rows)
        except Exception as error:
            results[entry] = {
                "rejected": True,
                "type": type(error).__name__,
                "message": str(error),
                "state_unchanged": env.serialize() == before,
            }
        else:
            raise AssertionError(f"C++ accepted invalid row through {entry}")
    return results


def field_coverage() -> list[dict[str, object]]:
    _env, legal = reset_rows(cpp_game)
    records = []
    for field in range(cpp_game.max_action_tokens):
        for dtype, delta, label in (
            (np.int64, 2**32, "signed_plus_2p32"),
            (np.int64, -(2**32), "signed_minus_2p32"),
            (np.uint64, 2**32, "unsigned_plus_2p32"),
        ):
            rows = legal.astype(dtype, copy=True)
            value = int(legal[0, field]) + delta
            rows[0, field] = value
            if int(rows[0, field]) != value:
                raise AssertionError("test array overflowed before C++ validation")
            result = reject_cpp(rows)
            records.append({
                "field": field,
                "dtype": np.dtype(dtype).name,
                "case": label,
                "raw_value": value,
                "decoded_rejected": result["decoded"]["rejected"],
                "raw_rejected": result["raw"]["rejected"],
                "state_unchanged": result["decoded"]["state_unchanged"] and result["raw"]["state_unchanged"],
            })
    return records


def boundary_coverage() -> list[dict[str, object]]:
    _env, legal = reset_rows(cpp_game)
    base = int(legal[0, 5])
    cases = [
        (np.int32, -1),
        (np.int32, cpp_game.num_tokens),
        (np.int32, np.iinfo(np.int32).min),
        (np.int32, np.iinfo(np.int32).max),
        (np.int64, np.iinfo(np.int64).min),
        (np.int64, np.iinfo(np.int64).max),
        (np.int64, base + 2**32),
        (np.int64, base - 2**32),
        (np.uint32, cpp_game.num_tokens),
        (np.uint32, np.iinfo(np.uint32).max),
        (np.uint64, np.iinfo(np.int64).max + 1),
        (np.uint64, np.iinfo(np.uint64).max),
        (np.uint64, base + 2**32),
    ]
    records = []
    for dtype, value in cases:
        rows = legal.astype(dtype, copy=True)
        rows[0, 5] = value
        if int(rows[0, 5]) != int(value):
            raise AssertionError("boundary test value was not preserved")
        result = reject_cpp(rows)
        records.append({
            "dtype": np.dtype(dtype).name,
            "raw_value": int(value),
            "decoded_rejected": result["decoded"]["rejected"],
            "raw_rejected": result["raw"]["rejected"],
            "state_unchanged": result["decoded"]["state_unchanged"] and result["raw"]["state_unchanged"],
        })
    return records


def legal_dtype_compatibility() -> list[dict[str, object]]:
    source, rows = reset_rows(cpp_game, batch_size=2, seed=900)
    payload = source.serialize()
    records = []
    for dtype in (np.int32, np.int64, np.uint32, np.uint64):
        typed = rows.astype(dtype, copy=True)
        raw_env = cpp_game.EnvBatch.deserialize(payload)
        decoded_env = cpp_game.EnvBatch.deserialize(payload)
        raw = raw_env.step(typed)
        decoded = decoded_env.step(cpp_game.actions_from_np(typed))
        if raw_env.serialize() != decoded_env.serialize():
            raise AssertionError("legal raw/decoded transitions differ")
        records.append({
            "dtype": np.dtype(dtype).name,
            "serialized_equal": True,
            "player_ids": raw.player_ids.tolist(),
            "action_shape": list(decoded.available_actions.tokens.shape),
        })
    return records


def python_cpp_rejection() -> list[dict[str, object]]:
    records = []
    for delta in (2**32, -(2**32)):
        for name, game in (("python", python_game), ("cpp", cpp_game)):
            env, rows = reset_rows(game, seed=1000)
            rows = rows.astype(np.int64, copy=True)
            rows[0, 5] = int(rows[0, 5]) + delta
            before = env.serialize()
            try:
                game.actions_from_np(rows)
            except Exception as error:
                records.append({"backend": name, "delta": delta, "rejected": True, "type": type(error).__name__, "state_unchanged": env.serialize() == before})
            else:
                raise AssertionError(f"{name} accepted wraparound input")
    return records


def batch_atomicity() -> dict[str, object]:
    env, rows = reset_rows(cpp_game, batch_size=2, seed=1100)
    bad = rows.astype(np.uint64, copy=True)
    bad[1, 5] = np.uint64(2**32 + int(rows[1, 5]))
    before = env.serialize()
    result = reject_cpp(bad, batch_size=2)
    # reject_cpp uses fresh equivalent environments; also verify this exact batch.
    try:
        env.step(bad)
    except cpp_game.ProtocolError as error:
        return {"rejected": True, "type": type(error).__name__, "message": str(error), "all_slots_and_rng_unchanged": env.serialize() == before, "entry_results": result}
    raise AssertionError("multi-slot out-of-range row was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import nanopokerlib._guandan_cpp as extension
    extension_path = Path(extension.__file__).resolve()
    payload = {
        "schema": "guandan-marvel-m04-cpp-integer-bounds-repair-verification-v1",
        "stage": "M04_cpp_integer_bounds_repair",
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "extension_path": str(extension_path),
            "extension_bytes": extension_path.stat().st_size,
            "extension_sha256": sha256(extension_path),
        },
        "field_coverage": field_coverage(),
        "boundary_coverage": boundary_coverage(),
        "legal_dtype_compatibility": legal_dtype_compatibility(),
        "python_cpp_rejection": python_cpp_rejection(),
        "batch_atomicity": batch_atomicity(),
        "scope_note": "Narrow M04 integer-bounds repair only; M05 is not executed or claimed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "field_cases": len(payload["field_coverage"]),
        "boundary_cases": len(payload["boundary_coverage"]),
        "legal_dtypes": [item["dtype"] for item in payload["legal_dtype_compatibility"]],
        "batch_atomic": payload["batch_atomicity"]["all_slots_and_rng_unchanged"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
