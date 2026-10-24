"""Serial deal contracts checked against the real Python rules, without GPU work."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import permutations, product
from pathlib import Path
import subprocess
import sys

import pytest

from guandan.cards import PRINTED_RANKS, Rank
from guandan_marvel.evaluation.manifests import deal_game_config

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from serial_compare.common import (  # noqa: E402
    DEAL_GENERATOR_VERSION,
    ENGINE_LEVEL_RANKS,
    PlanError,
    build_deal_manifest,
    canonical,
    validate_deal_manifest,
)


def _resign(manifest):
    manifest["hash"] = canonical({key: value for key, value in manifest.items() if key != "hash"})
    return manifest


@pytest.mark.parametrize(
    "domain,count,base",
    [("preflight", 8, 1_600_000_000), ("development", 32, 1_610_000_000),
     ("confirmation", 128, 1_620_000_000)],
)
def test_every_generated_deal_uses_real_rules(domain, count, base):
    manifest = build_deal_manifest(domain)
    assert manifest == build_deal_manifest(domain)
    rows = validate_deal_manifest(manifest, expected_domain=domain)
    assert len(rows) == count
    assert manifest["seeds"] == list(range(base, base + count))
    assert manifest["deal_ids"] == [f"{domain}-{index:04d}" for index in range(count)]
    assert ENGINE_LEVEL_RANKS == tuple(range(3, 16))
    assert set(ENGINE_LEVEL_RANKS) == {int(rank) for rank in PRINTED_RANKS}
    assert ENGINE_LEVEL_RANKS[-1] == int(Rank.TWO)
    contexts = Counter()
    for index, row in enumerate(rows):
        # This is the production parser, including actual GameConfig and
        # PreviousHandResult constructors; no backend/config replacement.
        config = deal_game_config(row)
        assert config.seed == base + index
        assert config.level_rank == 3 + index % 13
        assert config.leader_seat == index % 4
        previous = config.previous_result
        mode = index % 4
        if mode == 0:
            assert previous is None
            contexts["none"] += 1
        else:
            assert previous is not None
            assert previous.winner_team == 0
            assert previous.seat_with_rank(1) == 0
            assert previous.tribute_cancelled is (mode == 1)
            if mode in (1, 2):
                assert previous.ranking == (1, 3, 2, 4)
                assert previous.outcome_class == "DOUBLE_DOWN"
                assert {previous.seat_with_rank(rank) for rank in (1, 2)} == {0, 2}
            else:
                assert previous.ranking == (1, 2, 3, 4)
                assert previous.outcome_class == "HEAD_THIRD"
                assert {previous.seat_with_rank(rank) for rank in (1, 3)} == {0, 2}
            contexts["cancelled" if mode == 1 else previous.outcome_class] += 1
    assert contexts == {"none": count // 4, "cancelled": count // 4,
                        "DOUBLE_DOWN": count // 4, "HEAD_THIRD": count // 4}
    assert {row["level_rank"] for row in rows} == set(range(3, 3 + min(count, 13)))


@pytest.mark.parametrize("rank", [2, 16, 17, 0, -1])
def test_resigned_illegal_engine_rank_is_rejected(rank):
    manifest = build_deal_manifest("preflight")
    manifest["deals"][0]["level_rank"] = rank
    _resign(manifest)
    with pytest.raises(PlanError, match="level_rank"):
        validate_deal_manifest(manifest)
    with pytest.raises(ValueError):
        deal_game_config(manifest["deals"][0])


@pytest.mark.parametrize("leader", [-1, 4, 16])
def test_resigned_illegal_leader_is_rejected(leader):
    manifest = build_deal_manifest("preflight")
    manifest["deals"][0]["leader_seat"] = leader
    _resign(manifest)
    with pytest.raises(PlanError, match="leader_seat"):
        validate_deal_manifest(manifest)
    with pytest.raises(ValueError):
        deal_game_config(manifest["deals"][0])


@pytest.mark.parametrize(
    "field,value",
    [("ranking", [1, 2, 3, 4]),  # Old DOUBLE_DOWN bug: actually HEAD_THIRD.
     ("ranking", [1, 1, 3, 4]), ("ranking", [0, 1, 2, 3]),
     ("ranking", [1, 2, 3]), ("winner_team", -1), ("winner_team", 2),
     ("winner_team", 1),  # Valid team ID, inconsistent with rank-1 seat.
     ("outcome_class", "HEAD_THIRD"), ("outcome_class", "HEAD_LAST"),
     ("outcome_class", "UNKNOWN")],
)
def test_resigned_invalid_previous_result_is_rejected_by_both_validators(field, value):
    manifest = build_deal_manifest("preflight")
    manifest["deals"][2]["previous_result"][field] = value
    _resign(manifest)
    with pytest.raises(PlanError, match="previous_result"):
        validate_deal_manifest(manifest)
    with pytest.raises(ValueError):
        deal_game_config(manifest["deals"][2])


def test_resigned_old_head_third_ranking_is_rejected():
    manifest = build_deal_manifest("preflight")
    manifest["deals"][3]["previous_result"]["ranking"] = [1, 3, 2, 4]
    _resign(manifest)
    with pytest.raises(PlanError, match="outcome_class"):
        validate_deal_manifest(manifest)
    with pytest.raises(ValueError):
        deal_game_config(manifest["deals"][3])


@pytest.mark.parametrize("field,value", [
    ("level_rank", True), ("level_rank", 3.5), ("level_rank", "3"),
    ("leader_seat", False), ("leader_seat", 1.5), ("leader_seat", "1"),
    ("previous_result", []), ("previous_result", {}),
])
def test_resigned_malformed_fields_fail_closed(field, value):
    manifest = build_deal_manifest("preflight")
    manifest["deals"][0][field] = value
    with pytest.raises(PlanError):
        validate_deal_manifest(_resign(manifest))


@pytest.mark.parametrize("field,value", [
    ("ranking", [True, 3, 2, 4]), ("ranking", [1, 3.0, 2, 4]),
    ("winner_team", False), ("winner_team", "0"),
    ("tribute_cancelled", "false"), ("tribute_cancelled", 0),
])
def test_resigned_previous_result_rejects_coerced_values(field, value):
    manifest = build_deal_manifest("preflight")
    manifest["deals"][2]["previous_result"][field] = value
    with pytest.raises(PlanError, match="previous_result"):
        validate_deal_manifest(_resign(manifest))


@pytest.mark.parametrize("version", [None, "guandan-serial-compare-deal-seed-generator-v1", "unknown"])
def test_resigned_legacy_or_unknown_generator_is_rejected(version):
    manifest = build_deal_manifest("preflight")
    assert DEAL_GENERATOR_VERSION == "guandan-serial-compare-deal-seed-generator-v2"
    manifest["generator_version"] = version
    with pytest.raises(PlanError, match="generator version"):
        validate_deal_manifest(_resign(manifest))


def test_lightweight_previous_validation_matches_all_real_rankings():
    template = build_deal_manifest("preflight", count=1)
    accepted = rejected = 0
    for ranking, winner, outcome, cancelled in product(
        permutations((1, 2, 3, 4)), (0, 1),
        ("DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST"), (False, True),
    ):
        manifest = deepcopy(template)
        manifest["deals"][0]["previous_result"] = {
            "ranking": list(ranking), "winner_team": winner,
            "outcome_class": outcome, "tribute_cancelled": cancelled,
        }
        _resign(manifest)
        try:
            deal_game_config(manifest["deals"][0])
        except ValueError:
            with pytest.raises(PlanError, match="previous_result"):
                validate_deal_manifest(manifest)
            rejected += 1
        else:
            assert validate_deal_manifest(manifest) == manifest["deals"]
            accepted += 1
    # Every rank permutation has exactly one winning team/outcome combination.
    assert (accepted, rejected) == (48, 240)


def test_offline_isolated_common_requires_only_standard_library(tmp_path):
    # Copy only common.py: -I -S disables cwd/PYTHONPATH/site-packages. Both
    # generation and semantic rejection must work without the workspace.
    standalone = tmp_path / "common.py"
    standalone.write_bytes((REPO_ROOT / "tools/serial_compare/common.py").read_bytes())
    script = """
import runpy, sys
c = runpy.run_path(sys.argv[1])
assert not any(name.split('.')[0] in {'guandan', 'guandan_marvel', 'torch', 'numpy', 'nanopokerlib'}
               for name in sys.modules)
for domain in ('preflight', 'development', 'confirmation'):
    manifest = c['build_deal_manifest'](domain)
    c['validate_deal_manifest'](manifest)
    manifest['deals'][0]['level_rank'] = 2
    manifest['hash'] = c['canonical']({k: v for k, v in manifest.items() if k != 'hash'})
    try:
        c['validate_deal_manifest'](manifest)
    except c['PlanError']:
        pass
    else:
        raise AssertionError('offline validator accepted rank 2')
print('isolated generation and validation passed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(standalone)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated generation and validation passed" in result.stdout
