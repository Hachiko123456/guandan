from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import uuid

import pytest

import tools.export_guance_declaration_contract as exporter


ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "guance_phase2_declaration_handoff"
RULES_PATH = HANDOFF / "guandan_declaration_rules_v1.json"
VECTORS_PATH = HANDOFF / "guandan_declaration_golden_vectors_v1.json"


def _load() -> tuple[dict, dict]:
    return (
        json.loads(RULES_PATH.read_text(encoding="utf-8")),
        json.loads(VECTORS_PATH.read_text(encoding="utf-8")),
    )


def test_all_legal_and_invalid_levels_are_fixed_in_matrix():
    rules, vectors = _load()
    expected = [rank.name for rank in exporter.LEGAL_LEVELS]
    assert expected == [
        "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
        "TEN", "JACK", "QUEEN", "KING", "ACE", "TWO",
    ]
    for payload in (rules, vectors):
        matrix = payload["level_parity"]
        assert matrix["levels_tested"] == expected
        assert matrix["invalid_levels_rejected"] == ["SMALL_JOKER", "BIG_JOKER"]
        assert matrix["matrix_case_count"] == 13 * 26
        assert matrix["cases_per_level"] == 26
        assert all(item["passed"] for item in matrix["levels"])
        assert all(item["python_recognizer_rejected"] and item["native_recognizer_rejected"] and item["environment_config_rejected"]
                   for item in matrix["invalid_level_results"])


def test_every_level_covers_six_natural_families_and_all_wild_modes():
    _, vectors = _load()
    expected_families = {
        "FULL_HOUSE", "STRAIGHT", "STRAIGHT_FLUSH",
        "PAIR_SEQUENCE", "TRIPLE_SEQUENCE", "FOUR_KINGS",
    }
    wildcard_families = expected_families - {"FOUR_KINGS"}
    matrix = vectors["level_parity"]
    assert set(matrix["families_tested"]) == expected_families
    for level in matrix["levels"]:
        cases = level["cases"]
        natural = {case["family"] for case in cases if "matrix_natural" in case["coverage"]}
        natural_wild = {case["family"] for case in cases if "matrix_wild_natural" in case["coverage"]}
        substituted = {case["family"] for case in cases if "matrix_wild_substituted" in case["coverage"]}
        substituted_level = {case["family"] for case in cases if "matrix_wild_substituted_level_rank" in case["coverage"]}
        maximum = {case["family"] for case in cases if "matrix_wild_max" in case["coverage"]}
        assert natural == expected_families
        assert natural_wild == wildcard_families
        assert substituted == wildcard_families
        assert substituted_level == wildcard_families
        assert maximum == wildcard_families
        assert all(case["wildcard_count"] == 0 for case in cases if "matrix_natural" in case["coverage"])
        assert all(case["wildcard_count"] == 1 for case in cases if "matrix_wild_natural" in case["coverage"])
        assert all(case["wildcard_count"] == 1 for case in cases if "matrix_wild_substituted" in case["coverage"])
        assert all(case["wildcard_count"] == 1 for case in cases if "matrix_wild_substituted_level_rank" in case["coverage"])
        for case in cases:
            if "matrix_wild_substituted_level_rank" in case["coverage"]:
                assert all(item["mode"] == "substituted" for item in case["wild_assignments"])
                assert all(item["target_rank"]["value"] == level["level"]["value"] for item in case["wild_assignments"])
        assert all(case["wildcard_count"] == 2 for case in cases if "matrix_wild_max" in case["coverage"])
        assert level["natural_family_count"] == 6
        assert level["natural_wildcard_family_count"] == 5
        assert level["substituted_wildcard_family_count"] == 5
        assert level["substituted_level_rank_wildcard_family_count"] == 5
        assert level["maximum_wildcard_family_count"] == 5
        four = next(case for case in cases if case["family"] == "FOUR_KINGS")
        assert four["four_kings_contract"] == {
            "wildcard_rejected": True,
            "forbidden_declaration_tokens_absent": True,
            "canonical_action_card_ids": [52, 53, 106, 107],
            "declaration_storage_card_ids": [52, 106, 53, 107],
        }


def test_level_parity_booleans_are_backed_by_case_checks():
    _, vectors = _load()
    matrix = vectors["level_parity"]
    cases = [case for level in matrix["levels"] for case in level["cases"]]
    assert matrix["natural_cases_passed"] == all(case["passed"] for case in cases if "matrix_natural" in case["coverage"])
    assert matrix["natural_wildcard_cases_passed"] == all(case["passed"] for case in cases if "matrix_wild_natural" in case["coverage"])
    assert matrix["substituted_wildcard_cases_passed"] == all(case["passed"] for case in cases if "matrix_wild_substituted" in case["coverage"])
    assert matrix["substituted_level_rank_wildcard_cases_passed"] == all(case["passed"] for case in cases if "matrix_wild_substituted_level_rank" in case["coverage"])
    assert matrix["maximum_wildcard_cases_passed"] == all(case["passed"] for case in cases if "matrix_wild_max" in case["coverage"])
    assert matrix["python_native_recognizer_parity"] == all(case["checks"]["python_recognizer_equals_native"] for case in cases)
    assert matrix["python_native_token_parity"] == all(case["checks"]["python_tokens_equal_native_tokens"] for case in cases)
    assert matrix["python_native_action_row_parity"] == all(case["checks"]["python_rows_equal_native_rows"] for case in cases)
    assert matrix["python_native_decode_parity"] == all(case["checks"]["python_row_decode_equal"] and case["checks"]["native_row_decode_equal"] for case in cases)
    assert matrix["python_native_environment_commit_parity"] == all(case["checks"]["python_environment_commit_equal"] and case["checks"]["native_environment_commit_equal"] for case in cases)


def test_three_ace_two_special_boundary_vectors_are_auditable():
    _, vectors = _load()
    by_id = {vector["id"]: vector for vector in vectors["vectors"]}
    required = {
        "special_three_full_house_wild_natural",
        "special_three_full_house_wild_substituted",
        "special_three_full_house_wild_substituted_level_rank",
        "special_three_straight_wild_substituted_level_rank",
        "special_three_straight_flush_wild_substituted_level_rank",
        "special_three_pair_sequence_wild_substituted_level_rank",
        "special_three_triple_sequence_wild_substituted_level_rank",
        "special_three_straight_wild_natural",
        "special_three_straight_wild_substituted",
        "special_three_full_house_level_pair",
        "special_ace_full_house_wild_natural",
        "special_ace_full_house_wild_substituted",
        "special_ace_full_house_wild_substituted_level_rank",
        "special_ace_straight_wild_substituted_level_rank",
        "special_ace_straight_flush_wild_substituted_level_rank",
        "special_ace_pair_sequence_wild_substituted_level_rank",
        "special_ace_triple_sequence_wild_substituted_level_rank",
        "special_ace_full_house_level_pair",
        "special_ace_straight_a2345",
        "special_ace_straight_flush_hearts_a2345",
        "special_ace_straight_high_10jqka",
        "special_ace_invalid_jqka2",
        "special_two_full_house_wild_natural",
        "special_two_full_house_wild_substituted",
        "special_two_full_house_wild_substituted_level_rank",
        "special_two_straight_wild_substituted_level_rank",
        "special_two_straight_flush_wild_substituted_level_rank",
        "special_two_pair_sequence_wild_substituted_level_rank",
        "special_two_triple_sequence_wild_substituted_level_rank",
        "special_two_full_house_level_pair",
        "special_two_straight_a2345",
        "special_two_straight_flush_hearts_a2345",
        "special_two_straight_high_10jqka",
        "special_two_pair_sequence_a23",
        "special_two_triple_sequence_a2",
        "special_two_invalid_pair_sequence_ka2",
    }
    assert required <= set(by_id)
    for vector_id in required:
        vector = by_id[vector_id]
        assert vector["level_rank"]["name"] in {"THREE", "ACE", "TWO"}
        if vector["polarity"] == "positive":
            assert vector["expected"]["result"] == "legal"
            assert vector["expected"]["complete_token_sequence"]
            assert vector["expected"]["marvel_v2_action_rows"]
            assert all(vector["expected"]["parity"].values())
        else:
            assert vector["expected"]["result"] == "invalid"
            assert all(vector["expected"]["authority_checks"].values())


def test_source_hash_scope_and_git_blob_identity_are_explicit():
    rules, _ = _load()
    assert rules["authority_source_commit"] == rules["guandan_source_commit"]
    assert rules["handoff_artifact_commit"] is None
    assert rules["handoff_artifact_git_status"] == "untracked"
    records = rules["authority"]["source_files"]
    assert len(records) == len(exporter.AUTHORITATIVE_SOURCES)
    for record in records:
        assert record["working_tree"]["hash_scope"] == "raw_checkout_bytes"
        assert record["working_tree"]["line_endings"] in {"CRLF", "LF", "MIXED", "NONE"}
        assert record["normalized_lf"]["normalization"] == "CRLF and bare CR converted to LF"
        assert record["git_blob"]["commit"] == rules["authority_source_commit"]
        assert len(record["git_blob"]["oid"]) == 40
        assert record["matches_head_after_normalization"] is True
        assert record["normalized_lf"]["bytes"] == record["git_blob"]["bytes"]
        assert record["normalized_lf"]["sha256"] == record["git_blob"]["sha256"]
    assert rules["authority"]["native_build_identity"]["source_matches_loaded_binary"] is True


def test_semantic_hashes_and_exporter_hash_are_current():
    rules, vectors = _load()
    exporter._assert_semantic_hash(rules, "rules")
    exporter._assert_semantic_hash(vectors, "vectors")
    current = exporter._sha256_file(exporter.Path(exporter.__file__).resolve())
    assert rules["exporter_sha256"] == current
    assert vectors["exporter_sha256"] == current


def test_dirty_authority_source_is_fail_closed(monkeypatch):
    class Result:
        returncode = 1
    monkeypatch.setattr(exporter.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(AssertionError, match="authoritative source files differ from HEAD"):
        exporter._assert_authority_clean()


def _temporary_handoff() -> Path:
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    target = build / f"handoff-test-{uuid.uuid4().hex}"
    shutil.copytree(HANDOFF, target)
    return target


def _prepare_fast_check(monkeypatch, target: Path) -> dict[str, bytes]:
    expected = {name: (target / name).read_bytes() for name in exporter.OUTPUT_NAMES}
    (target / "SHA256SUMS").write_bytes(exporter._manifest_bytes(
        [target / name for name in exporter.OUTPUT_NAMES] + [Path(exporter.__file__).resolve()]
    ))
    monkeypatch.setattr(exporter, "_build_artifacts", lambda generated_at, output_dir: expected)
    return expected


@pytest.mark.parametrize("tamper", ["rules_source_hash", "vectors", "manifest", "exporter_manifest"])
def test_check_detects_artifact_source_exporter_or_manifest_tampering(monkeypatch, tamper):
    target = _temporary_handoff()
    try:
        _prepare_fast_check(monkeypatch, target)
        if tamper == "rules_source_hash":
            rules = json.loads((target / exporter.OUTPUT_NAMES[0]).read_text(encoding="utf-8"))
            rules["authority"]["source_files"][0]["git_blob"]["sha256"] = "0" * 64
            rules["integrity"]["semantic_sha256"] = exporter._semantic_digest(rules)
            (target / exporter.OUTPUT_NAMES[0]).write_bytes(exporter._json_bytes(rules))
        elif tamper == "vectors":
            path = target / exporter.OUTPUT_NAMES[1]
            path.write_bytes(path.read_bytes() + b" ")
        elif tamper == "manifest":
            path = target / "SHA256SUMS"
            path.write_bytes(path.read_bytes() + b"tamper\n")
        else:
            lines = (target / "SHA256SUMS").read_text(encoding="ascii").splitlines()
            lines[-1] = f"{'0' * 64}  tools/export_guance_declaration_contract.py"
            (target / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
        with pytest.raises((AssertionError, json.JSONDecodeError)):
            exporter._check(target)
    finally:
        shutil.rmtree(target, ignore_errors=True)


def test_scope_guards_and_authority_allowlist_exclude_forbidden_data():
    rules, _ = _load()
    guards = rules["scope_guards"]
    assert guards == {
        "checkpoint_read": False,
        "real_capture_read": False,
        "complete_real_game_read": False,
        "hidden_hand_data_read": False,
        "training_executed": False,
        "model_inference_executed": False,
        "guance_repository_accessed": False,
        "fixtures": "synthetic physical-card vectors only",
    }
    forbidden = ("runs/", "logs/", "checkpoint", "capture", "d:/project/guance")
    for path in exporter.AUTHORITATIVE_SOURCES:
        assert not any(item in path.lower() for item in forbidden)






def test_check_rebuilds_authority_sources_level_matrix_and_vectors(monkeypatch):
    calls = {"authority_clean": 0, "source_records": 0, "level_matrix": 0, "vectors": 0}
    original_clean = exporter._assert_authority_clean
    original_sources = exporter._source_records
    original_matrix = exporter._build_level_parity_matrix
    original_vectors = exporter._build_vectors

    def checked_clean():
        calls["authority_clean"] += 1
        return original_clean()

    def checked_sources(*args, **kwargs):
        calls["source_records"] += 1
        return original_sources(*args, **kwargs)

    def checked_matrix(*args, **kwargs):
        calls["level_matrix"] += 1
        return original_matrix(*args, **kwargs)

    def checked_vectors(*args, **kwargs):
        calls["vectors"] += 1
        return original_vectors(*args, **kwargs)

    monkeypatch.setattr(exporter, "_assert_authority_clean", checked_clean)
    monkeypatch.setattr(exporter, "_source_records", checked_sources)
    monkeypatch.setattr(exporter, "_build_level_parity_matrix", checked_matrix)
    monkeypatch.setattr(exporter, "_build_vectors", checked_vectors)
    result = exporter._check(HANDOFF)
    assert result["exit_code"] == 0
    assert all(count > 0 for count in calls.values())
