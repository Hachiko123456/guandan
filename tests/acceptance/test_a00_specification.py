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
        ROOT / "docs" / "environment_api.md",
        ROOT / "docs" / "encoding_layout.md",
        ROOT / "docs" / "action_protocol.md",
        ROOT / "docs" / "MASTER_EXECUTION_PLAN.md",
        ROOT / "docs" / "AGENT_EXECUTION_PROTOCOL.md",
        ROOT / "docs" / "CONTINUE_PROMPT.md",
        ROOT / "project_status" / "STATUS_SCHEMA.md",
        ROOT / "configs" / "acceptance_profiles.json",
        ROOT / "docs" / "acceptance" / "A00_specification.md",
        ROOT / "docs" / "acceptance" / "A01_cards_and_combinations.md",
        ROOT / "docs" / "acceptance" / "A02_round_rules_engine.md",
        ROOT / "docs" / "acceptance" / "A03_stepwise_action_protocol.md",
        ROOT / "docs" / "acceptance" / "A04_observation_and_batch_environment.md",
        ROOT / "docs" / "acceptance" / "A05_training_integration.md",
        ROOT / "docs" / "acceptance" / "A06_evaluation.md",
        ROOT / "docs" / "acceptance" / "A07_kaggle.md",
        ROOT / "docs" / "acceptance" / "A08_belief_search.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, f"missing A00 specification files: {missing}"


def test_a00_documents_have_versioned_contract_markers_and_fixed_v1_choices() -> None:
    rules = (ROOT / "docs" / "rules.md").read_text(encoding="utf-8")
    environment = (ROOT / "docs" / "environment_contract.md").read_text(encoding="utf-8")
    action = (ROOT / "docs" / "action_protocol.md").read_text(encoding="utf-8")
    profiles = (ROOT / "configs" / "acceptance_profiles.json").read_text(encoding="utf-8")

    assert "GD-RULES-0.1" in rules
    assert "GD-ENV-0.1" in environment
    assert "GD-ACTION-0.1" in action
    assert "local_fast" in profiles and "remote_full" in profiles
    assert "FOUR_KINGS" in rules and "两张小王" in rules and "两张大王" in rules
    assert "三连对" in rules and "恰好" in rules
    assert "钢板" in rules and "两组" in rules
    assert "256" in environment and "max_action_tokens=32" in environment
    assert "token" in action and "PPO/GAE" in action
    assert "COMMIT" in action
    assert "GD-RULES-0.1" in rules and "单副" in rules


def test_a00_master_plan_covers_all_stages_and_continuation() -> None:
    plan = (ROOT / "docs" / "MASTER_EXECUTION_PLAN.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "AGENT_EXECUTION_PROTOCOL.md").read_text(encoding="utf-8")
    prompt = (ROOT / "docs" / "CONTINUE_PROMPT.md").read_text(encoding="utf-8")
    for stage in ("A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"):
        assert stage in plan
    assert "accepted=false" in prompt or "accepted=false" in plan
    assert "差异" in protocol
    assert "完整测试套件" in protocol


def test_a00_does_not_reference_fabledan_as_runtime_dependency() -> None:
    for path in (ROOT / "guandan").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "fabledan" not in text, f"runtime dependency reference in {path}"


def test_project_documentation_uses_chinese_headings_and_continuation() -> None:
    documents = [ROOT / "README.md", ROOT / "project_status" / "STATUS_SCHEMA.md"]
    documents.extend(sorted((ROOT / "docs").rglob("*.md")))
    for document in documents:
        text = document.read_text(encoding="utf-8-sig")
        headings = [line for line in text.splitlines() if line.startswith("#")]
        assert headings, f"文档缺少标题：{document}"
        assert any("\u4e00" <= char <= "\u9fff" for char in headings[0]), (
            f"文档主标题应使用中文：{document}"
        )
        assert "\ufffd" not in text, f"文档包含乱码替换字符：{document}"

    prompt = (ROOT / "docs" / "CONTINUE_PROMPT.md").read_text(encoding="utf-8")
    assert "继续执行" in prompt
    assert "accepted=false" in prompt
    assert "主代理" in prompt and "子代理" in prompt
    assert "完整测试套件" in prompt
