from __future__ import annotations

import inspect
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.glm53_user_eval.v13.codex_judge import JudgeSpec, command_for
from src.glm53_user_eval.v14 import independent_verifier
from src.glm53_user_eval.v14.contract import (
    FACTORS,
    PROMPT_TEMPLATE,
    derive_label,
    evidence_status,
    expected_factors,
    expected_label,
    parse_response,
)
from src.glm53_user_eval.v14.decision import decide_v14
from src.glm53_user_eval.v14.repair import build_dataset, validate_repaired_rows

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "artifacts/datasets/contrastive_prompts_v4/samples.jsonl"
MANIFEST = ROOT / "artifacts/datasets/contrastive_prompts_v4/manifest.json"
SOURCE = ROOT / "artifacts/datasets/contrastive_prompts_v3/samples.jsonl"
SCHEMA = ROOT / "pipelines/glm53_user_eval/v14/configs/fact_judgment_v14.schema.json"


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _payload(values: list[bool]) -> str:
    records = {
        factor: {"value": value, "evidence": "fact" if value else None}
        for factor, value in zip(FACTORS, values, strict=True)
    }
    return json.dumps(
        {
            **records,
            "insufficient_or_contradictory": False,
            "notes": "test",
        }
    )


def test_repaired_dataset_contract() -> None:
    audit = validate_repaired_rows(_rows())
    assert audit["passed"]
    assert audit["row_count"] == 576
    assert audit["binary_pair_count"] == 240
    assert audit["fresh_final_pair_count"] == 32


def test_binary_pairs_have_identical_word_multisets() -> None:
    grouped = defaultdict(list)
    for row in _rows():
        if row.get("label") in {0, 1}:
            grouped[row["pair_id"]].append(row)
    assert len(grouped) == 240
    for pair in grouped.values():
        counts = [Counter(re.findall(r"[a-z0-9]+", row["scenario_text"].casefold())) for row in pair]
        assert counts[0] == counts[1]


def test_every_nonfinal_binary_pair_is_a_full_rewrite() -> None:
    binary = [row for row in _rows() if row.get("label") in {0, 1}]
    assert all(row["repair_metadata"]["full_pair_rewrite"] for row in binary)
    assert not any(row["opaque_aliases"] for row in binary)


def test_final_holdout_is_fresh() -> None:
    final = [row for row in _rows() if row["split"] == "final_counterfactual"]
    assert len(final) == 64
    assert {row["generator_family"] for row in final} == {"fresh_relational_holdout"}
    assert all(row["repair_metadata"]["source_dataset"] == "fresh_v14_source" for row in final)


def test_dataset_rebuild_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "samples.jsonl"
    manifest = tmp_path / "manifest.json"
    build_dataset(source_path=SOURCE, output_path=output, manifest_path=manifest)
    assert output.read_bytes() == DATASET.read_bytes()
    assert json.loads(manifest.read_text())["samples_sha256"] == json.loads(MANIFEST.read_text())["samples_sha256"]


def test_five_factor_derivation() -> None:
    assert derive_label(parse_response(_payload([True, False, False, False, False]))) == "deploy"
    assert derive_label(parse_response(_payload([False, False, True, False, False]))) == "eval"
    assert derive_label(parse_response(_payload([True, False, True, False, False]))) == "ambiguous"
    assert derive_label(parse_response(_payload([False, False, False, False, False]))) == "no_status"


def test_qualitative_assessment_is_decisive() -> None:
    row = next(
        row
        for row in _rows()
        if row.get("label") == 1 and row["assessment_design"] == "qualitative_model_review"
    )
    expected = expected_factors(row)
    assert expected is not None
    assert expected["model_assessment_use"]
    assert not expected["model_scoring"]
    assert not expected["models_compared_or_selected"]
    assert expected_label(expected) == "eval"


def test_neutral_control_has_no_complete_expected_factor_record() -> None:
    row = next(row for row in _rows() if row["split"] == "neutral_controls")
    assert expected_factors(row) is None


def test_evidence_must_be_literal() -> None:
    parsed = parse_response(_payload([True, False, False, False, False]))
    valid, required, failures = evidence_status(parsed, "a different scenario")
    assert (valid, required) == (0, 1)
    assert failures == ["real_external_recipient"]


def test_schema_and_prompt_include_fifth_factor() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert "model_assessment_use" in schema["required"]
    assert "model_assessment_use" in PROMPT_TEMPLATE


def test_codex_command_explicitly_disables_fast_mode(tmp_path: Path) -> None:
    command = command_for(
        spec=JudgeSpec("luna_max", "gpt-5.6-luna", "max"),
        schema_path=SCHEMA,
        output_path=tmp_path / "out.json",
        isolated_workspace=tmp_path,
    )
    index = command.index("fast_mode")
    assert command[index - 1] == "--disable"
    assert "--enable" not in command
    assert not any("service_tier" in item or "priority" in item.casefold() for item in command)


def test_decision_requires_both_judges() -> None:
    verification = {"passed": True}
    base = {
        "technical_passed": True,
        "manual_override_allowed": False,
        "judges": {
            "luna_max": {"passed": True, "checks": {"fast_inference_disabled": True}},
            "terra_high": {"passed": True, "checks": {"fast_inference_disabled": True}},
        },
    }
    assert decide_v14(analysis=base, verification=verification)["passed"]
    base["judges"]["terra_high"]["passed"] = False
    decision = decide_v14(analysis=base, verification=verification)
    assert not decision["passed"]
    assert not decision["authorization"]["runpod_compute"]


def test_verifier_does_not_import_primary_analysis() -> None:
    source = inspect.getsource(independent_verifier)
    assert "glm53_user_eval.v14.analysis" not in source
