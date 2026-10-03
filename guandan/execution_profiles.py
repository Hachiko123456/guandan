"""本机/远端运行规模配置；不修改牌局或 token 协议。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

PROFILE_VERSION = "profiles-0.2"
PROFILE_NAMES = ("local_fast", "remote_full")


def _keys(value, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label}: expected keys {sorted(expected)}")


def _positive_int(value, label: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label}: expected positive integer")


def _budget(value, label: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: expected finite positive hours")


def _validate(name: str, config: dict) -> None:
    _keys(config, {"mode", "device", "algorithms", "training", "evaluation"}, name)
    expected_mode, expected_device = ("local", "auto") if name == "local_fast" else ("remote", "cuda")
    if config["mode"] != expected_mode or config["device"] != expected_device:
        raise ValueError(f"{name}: invalid mode/device")
    if config["algorithms"] != ["ippo", "vrpo"]:
        raise ValueError(f"{name}: both ippo and vrpo are required")
    training = config["training"]
    _keys(training, {"updates_per_algorithm", "rollout_envs", "token_steps_per_update",
                     "checkpoint_interval", "resume_updates", "max_hours"}, name + ".training")
    for key in training:
        if key == "max_hours":
            _budget(training[key], name + ".training." + key)
        else:
            _positive_int(training[key], name + ".training." + key)
    if training["token_steps_per_update"] % training["rollout_envs"]:
        raise ValueError(f"{name}: aggregate token steps must be divisible by rollout_envs")
    if training["checkpoint_interval"] > training["updates_per_algorithm"]:
        raise ValueError(f"{name}: checkpoint interval exceeds planned updates")
    evaluation = config["evaluation"]
    _keys(evaluation, {"deal_groups_per_pairing", "seat_rotations", "opponents", "max_hours"}, name + ".evaluation")
    _positive_int(evaluation["deal_groups_per_pairing"], name + ".evaluation.deal_groups_per_pairing")
    _budget(evaluation["max_hours"], name + ".evaluation.max_hours")
    if (not isinstance(evaluation["seat_rotations"], list)
            or any(type(x) is not int for x in evaluation["seat_rotations"])
            or evaluation["seat_rotations"] != [0, 1, 2, 3]):
        raise ValueError(f"{name}: all four seat rotations are required")
    if evaluation["opponents"] != ["random", "rule", "snapshot"]:
        raise ValueError(f"{name}: random/rule/snapshot opponents are required")


def load_execution_profile(root: Path, name: str) -> dict:
    """读取真实配置；未知 profile、缺失文件或非法值必须失败，绝不降级。"""
    if name not in PROFILE_NAMES:
        raise ValueError(f"unknown execution profile: {name!r}")
    root = Path(root).resolve()
    source = root / "configs" / "acceptance_profiles.json"
    if not source.resolve().is_relative_to(root):
        raise ValueError("profile configuration escapes project root")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read profile configuration: {exc}") from exc
    _keys(payload, {"version", "profiles"}, "profile catalog")
    if payload["version"] != PROFILE_VERSION:
        raise ValueError(f"unsupported profile version: {payload['version']!r}")
    _keys(payload["profiles"], set(PROFILE_NAMES), "profiles")
    for profile_name, value in payload["profiles"].items():
        _validate(profile_name, value)
    config = payload["profiles"][name]
    training, evaluation = config["training"], config["evaluation"]
    return {
        "name": name, "version": PROFILE_VERSION,
        "source": source.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(), "config": config,
        "derived": {
            "vector_steps_per_update": training["token_steps_per_update"] // training["rollout_envs"],
            "token_steps_per_algorithm_before_resume": training["token_steps_per_update"] * training["updates_per_algorithm"],
            "games_per_pairing": evaluation["deal_groups_per_pairing"] * len(evaluation["seat_rotations"]),
            "total_games_all_opponents": evaluation["deal_groups_per_pairing"] * len(evaluation["seat_rotations"]) * len(evaluation["opponents"]),
        },
    }
