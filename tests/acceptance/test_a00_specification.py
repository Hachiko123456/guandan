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


def test_a00_does_not_reference_fabledan_as_runtime_dependency() -> None:
    for path in (ROOT / "guandan").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "fabledan" not in text, f"runtime dependency reference in {path}"
