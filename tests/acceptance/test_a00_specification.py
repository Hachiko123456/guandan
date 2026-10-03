from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_a00_required_spec_documents_exist() -> None:
    required = [
        ROOT / "docs" / "completion_criteria.md",
        ROOT / "docs" / "roadmap.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "observation_protocol.md",
        ROOT / "docs" / "training_protocol.md",
        ROOT / "docs" / "rules.md",
        ROOT / "docs" / "environment_contract.md",
        ROOT / "docs" / "action_protocol.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, f"missing A00 specification files: {missing}"


def test_a00_documents_have_versioned_contract_markers_and_fixed_v1_choices() -> None:
    rules = (ROOT / "docs" / "rules.md").read_text(encoding="utf-8")
    environment = (ROOT / "docs" / "environment_contract.md").read_text(encoding="utf-8")
    action = (ROOT / "docs" / "action_protocol.md").read_text(encoding="utf-8")

    assert "GD-RULES-0.1" in rules
    assert "GD-ENV-0.1" in environment
    assert "GD-ACTION-0.1" in action
    assert "FOUR_KINGS" in rules and "exactly two small jokers and two big jokers" in rules
    assert "exactly 3 consecutive rank pairs" in rules
    assert "exactly 2 consecutive rank triples" in rules
    assert "256" in environment and "max_action_tokens=32" in environment
    assert "token-level PPO/GAE" in action
    assert "COMMIT" in action
    assert "unresolved variants remain open decisions" not in rules


def test_a00_does_not_reference_fabledan_as_runtime_dependency() -> None:
    for path in (ROOT / "guandan").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "fabledan" not in text, f"runtime dependency reference in {path}"
