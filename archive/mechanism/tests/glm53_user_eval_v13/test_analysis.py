from __future__ import annotations

import json
from pathlib import Path

from src.glm53_user_eval.v12.fact_validation import FACTORS
from src.glm53_user_eval.v13.analysis import analyze_cohort, analyze_judge
from src.glm53_user_eval.v13.codex_judge import judge_specs
from src.glm53_user_eval.v13.decision import decide_v13
from src.glm53_user_eval.v13.failure_audit import build_failure_audit
from src.glm53_user_eval.v13.independent_verifier import verify_v13

from .conftest import judgment


def test_perfect_judge_passes_all_frozen_thresholds(
    frozen_rows: list[dict], schema_path: Path
) -> None:
    spec = judge_specs()[0]
    rows = [judgment(row, spec=spec, schema_path=schema_path) for row in frozen_rows]
    result = analyze_judge(frozen_rows, rows, spec=spec, schema_path=schema_path)
    assert result["passed"] is True
    assert result["factor_accuracy"]["accuracy"] == 1.0
    assert result["derived_labels"]["clean_binary"]["accuracy"] == 1.0


def test_one_factor_regression_fails(
    frozen_rows: list[dict], schema_path: Path
) -> None:
    spec = judge_specs()[0]
    rows = [judgment(row, spec=spec, schema_path=schema_path) for row in frozen_rows]
    for record in rows[:100]:
        current = record["parsed"]["real_external_recipient"]["value"] is True
        record["parsed"]["real_external_recipient"] = {
            "value": not current,
            "evidence": str(frozen_rows[0]["scenario_text"])[:1] if not current else None,
        }
    result = analyze_judge(frozen_rows, rows, spec=spec, schema_path=schema_path)
    assert result["passed"] is False
    assert result["checks"]["each_factor_accuracy_ge_092"] is False


def _write_cohort(
    root: Path, frozen_rows: list[dict], schema_path: Path
) -> None:
    for spec in judge_specs():
        row_root = root / spec.judge_id / "rows"
        row_root.mkdir(parents=True)
        for row in frozen_rows:
            record = judgment(row, spec=spec, schema_path=schema_path)
            (row_root / f"{row['sample_id']}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )


def test_cohort_and_independent_verifier_agree(
    tmp_path: Path,
    frozen_rows: list[dict],
    schema_path: Path,
    repo_root: Path,
) -> None:
    _write_cohort(tmp_path, frozen_rows, schema_path)
    v12 = json.loads(
        (repo_root / "artifacts/glm53_user_eval/v12/semantic_validation/primary_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    analysis = analyze_cohort(
        frozen_rows,
        output_root=tmp_path,
        schema_path=schema_path,
        v12_primary=v12,
    )
    verification = verify_v13(
        dataset=frozen_rows,
        output_root=tmp_path,
        schema_path=schema_path,
        primary=analysis,
    )
    assert analysis["passed"] is True
    assert verification["passed"] is True
    decision = decide_v13(analysis=analysis, verification=verification)
    assert decision["decision"] == "unchanged_bank_validated_by_both_codex_judges"
    assert decision["authorization"]["exact_fp8_source_extraction"] is True


def test_failure_audit_requires_paired_repair(
    tmp_path: Path,
    frozen_rows: list[dict],
    schema_path: Path,
    repo_root: Path,
) -> None:
    _write_cohort(tmp_path, frozen_rows, schema_path)
    first = frozen_rows[0]
    for spec in judge_specs():
        path = tmp_path / spec.judge_id / "rows" / f"{first['sample_id']}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        for factor in FACTORS:
            record["parsed"][factor] = {"value": False, "evidence": None}
        path.write_text(json.dumps(record), encoding="utf-8")
    v12 = json.loads(
        (repo_root / "artifacts/glm53_user_eval/v12/semantic_validation/primary_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    analysis = analyze_cohort(
        frozen_rows,
        output_root=tmp_path,
        schema_path=schema_path,
        v12_primary=v12,
    )
    analysis["passed"] = False
    audit = build_failure_audit(frozen_rows, analysis)
    assert audit["diagnostic_row_count"] >= 1
    assert audit["repair_constraints"]["matched_pair_or_generator_level_changes_required"] is True
    assert audit["repair_constraints"]["fresh_untouched_holdout_required_after_any_final_holdout_edit"] is True
