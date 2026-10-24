"""Real rules parsing before any training, independent of native availability."""
import copy
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from serial_compare.common import build_deal_manifest, canonical, PlanError
from serial_compare.runtime_validation import validate_runtime_deals


def test_all_runtime_rule_configurations_are_checked():
    manifests = {d: build_deal_manifest(d) for d in ("preflight", "development", "confirmation")}
    result = validate_runtime_deals(manifests, reset_native=False)
    assert result["rules_configs_checked"] == 168 and result["native_resets"] == 0
    assert result["coverage"]["confirmation"]["level_ranks"] == list(range(3, 16))


@pytest.mark.parametrize("field,value", [("level_rank", 2), ("previous_result", {"ranking": [1, 2, 3, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN"})])
def test_resigned_invalid_config_cannot_reach_training(field, value):
    manifest = build_deal_manifest("preflight")
    manifest["deals"][0][field] = value
    manifest["hash"] = canonical({k: v for k, v in manifest.items() if k != "hash"})
    with pytest.raises(PlanError):
        validate_runtime_deals({"preflight": manifest}, reset_native=False)


def test_input_check_wires_rules_before_physical_seed_probe():
    import ast
    from serial_compare.pipeline import ProductionActions
    import inspect
    text = inspect.getsource(ProductionActions.input_check)
    assert text.index("validate_runtime_deals(manifests)") < text.index("assert_training_seeds_disjoint(")
