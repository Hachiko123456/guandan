# SPDX-License-Identifier: MIT
"""Comparison control schema: deals, arms, deterministic task enumeration."""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

from guandan_marvel.evaluation.common import payload_hash
from guandan_marvel.evaluation.manifests import build_deals_manifest, dataset_identity, deals_for

CONTROL_SCHEMA = "guandan-marvel-comparison-control-v1"
SUMMARY_SCHEMA = "guandan-marvel-comparison-summary-v1"
DELIVERY_ID = "m09-fabledan-comparison-n256-v1"
DEAL_COUNT = 256
DEAL_BASE = 1600000000
LEVEL_RANKS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
LEADER_SEATS = (0, 1, 2, 3)
ROTATIONS = (0, 1, 2, 3)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261016
BOOTCAMP_FINAL_KIND = "bootcamp_final"
BOOTCAMP_FINAL_CANDIDATE_ARM = "bootcamp_final_vs_m09_iter2000"
BOOTCAMP_FINAL_MIRROR_ARM = "m09_iter2000_vs_bootcamp_final"
BOOTCAMP_FINAL_CONTROL_KEY = "bootcamp_final"
TRAINING_LEVEL_RANK = 15
TRAINING_LEADER_SEAT = 0


def build_comparison_deals(*, count: int = DEAL_COUNT, base: int = DEAL_BASE) -> dict[str, object]:
    return build_deals_manifest(
        counts={"training": 0, "selection": 0, "final": int(count)},
        bases={"training": 0, "selection": 0, "final": int(base)},
        level_ranks=LEVEL_RANKS,
        leader_seats=LEADER_SEATS,
        previous_results=(None,),
    )


def default_arms(*, with_bootcamp_final: bool = False) -> list[dict[str, object]]:
    arms = [
        {
            "id": "h2h", "priority": 0, "deals": 256,
            "candidate_kind": "m09", "opponent_kind": "fabledan",
            "candidate_label": "m09_iter2000", "opponent_label": "fabledan_numpy",
        },
        {
            "id": "m09_vs_rule", "priority": 1, "deals": 128,
            "candidate_kind": "m09", "opponent_kind": "rule",
            "candidate_label": "m09_iter2000", "opponent_label": "rule",
        },
        {
            "id": "fabledan_vs_rule", "priority": 1, "deals": 128,
            "candidate_kind": "fabledan", "opponent_kind": "rule",
            "candidate_label": "fabledan_numpy", "opponent_label": "rule",
        },
        {
            "id": "m09_vs_random", "priority": 2, "deals": 128,
            "candidate_kind": "m09", "opponent_kind": "random",
            "candidate_label": "m09_iter2000", "opponent_label": "random",
        },
        {
            "id": "fabledan_vs_random", "priority": 2, "deals": 128,
            "candidate_kind": "fabledan", "opponent_kind": "random",
            "candidate_label": "fabledan_numpy", "opponent_label": "random",
        },
        {
            "id": "champion_vs_fabledan", "priority": 3, "deals": 128,
            "candidate_kind": "m09_champion", "opponent_kind": "fabledan",
            "candidate_label": "m09_iter100_champion", "opponent_label": "fabledan_numpy",
        },
    ]
    if with_bootcamp_final:
        arms = [
            {
                "id": BOOTCAMP_FINAL_CANDIDATE_ARM, "priority": 0, "deals": 256,
                "candidate_kind": BOOTCAMP_FINAL_KIND, "opponent_kind": "m09",
                "candidate_label": "bootcamp_week1_final", "opponent_label": "m09_iter2000",
            },
            {
                "id": BOOTCAMP_FINAL_MIRROR_ARM, "priority": 0, "deals": 256,
                "candidate_kind": "m09", "opponent_kind": BOOTCAMP_FINAL_KIND,
                "candidate_label": "m09_iter2000", "opponent_label": "bootcamp_week1_final",
            },
        ] + arms
    return arms


def task_id_for(arm_id: str, deal_id: str, rotation: int) -> str:
    payload = f"{arm_id}|{deal_id}|{int(rotation)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


def enumerate_tasks(control: Mapping[str, object], deals: Mapping[str, object]) -> list[dict[str, object]]:
    rows = deals_for(deals, "final")
    tasks: list[dict[str, object]] = []
    for arm in control["arms"]:
        arm_id = str(arm["id"])
        count = int(arm["deals"])
        if count > len(rows):
            raise ValueError(f"arm {arm_id} requests more deals than the manifest holds")
        for index in range(count):
            deal = rows[index]
            for rotation in ROTATIONS:
                tasks.append({
                    "task_id": task_id_for(arm_id, str(deal["deal_id"]), rotation),
                    "arm_id": arm_id,
                    "priority": int(arm["priority"]),
                    "deal_index": index,
                    "deal_id": str(deal["deal_id"]),
                    "rotation": int(rotation),
                })
    return tasks


def expected_games(control: Mapping[str, object]) -> int:
    return sum(int(arm["deals"]) * len(ROTATIONS) for arm in control["arms"])


def final_plan_hash(control_body: Mapping[str, object]) -> str:
    return payload_hash(control_body)


__all__ = [
    "CONTROL_SCHEMA", "SUMMARY_SCHEMA", "DELIVERY_ID", "DEAL_COUNT", "DEAL_BASE",
    "ROTATIONS", "BOOTSTRAP_SAMPLES", "BOOTSTRAP_SEED",
    "BOOTCAMP_FINAL_KIND", "BOOTCAMP_FINAL_CANDIDATE_ARM", "BOOTCAMP_FINAL_MIRROR_ARM",
    "BOOTCAMP_FINAL_CONTROL_KEY", "TRAINING_LEVEL_RANK", "TRAINING_LEADER_SEAT",
    "build_comparison_deals", "default_arms", "task_id_for", "enumerate_tasks",
    "expected_games", "final_plan_hash", "dataset_identity",
]
