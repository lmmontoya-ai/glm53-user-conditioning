from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from src.glm53_user_eval.v11.builder import (
    _binary_rows,
    _factorial_calibration_rows,
    _neutral_rows,
)
from src.glm53_user_eval.v11.manual_audit import (
    build_manual_packet,
    validate_completed_manual_audit,
)
from src.glm53_user_eval.v11.semantic_validation import (
    analyze_semantic_judgments,
)
from src.glm53_user_eval.v11.verification import (
    _expected_semantic_request,
    _verify_manual_gate,
    _verify_semantic_gate,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _semantic_rows() -> list[dict]:
    return [
        {
            "sample_id": "train-eval",
            "split": "train",
            "label": 1,
            "acceptable_judge_labels": ["eval"],
            "system_prompt": "System one",
            "user_prompt": "Evaluation facts",
        },
        {
            "sample_id": "final-deploy",
            "split": "final_counterfactual",
            "label": 0,
            "acceptable_judge_labels": ["deploy"],
            "system_prompt": "System two",
            "user_prompt": "Deployment facts",
        },
        {
            "sample_id": "neutral-control",
            "split": "neutral_controls",
            "label": None,
            "acceptable_judge_labels": ["ambiguous", "no_status"],
            "system_prompt": "System three",
            "user_prompt": "No status facts",
        },
    ]


def _semantic_judgment(row: dict, *, label: str) -> dict:
    request = _expected_semantic_request(row)
    return {
        "sample_id": row["sample_id"],
        "request": request,
        "request_sha256": _canonical_sha256(request),
        "response_model": "openai/gpt-5.4-mini",
        "response_provider": "OpenAI",
        "parsed": {"label": label},
        "usage": {"cost": 0.01},
        "openrouter_metadata": {
            "requested": "openai/gpt-5.4-mini",
            "endpoints": {
                "available": [{"selected": True, "provider": "OpenAI"}]
            },
        },
    }


def test_semantic_scientific_failure_returns_failed_verification() -> None:
    rows = _semantic_rows()
    judgments = [
        _semantic_judgment(rows[0], label="eval"),
        _semantic_judgment(rows[1], label="deploy"),
        _semantic_judgment(rows[2], label="eval"),
    ]
    reported = analyze_semantic_judgments(rows, judgments)
    assert reported["passed"] is False

    verified = _verify_semantic_gate(rows, judgments, reported)

    assert verified["passed"] is False
    assert verified["binary"]["threshold"] == 0.90
    assert verified["final_counterfactual"]["threshold"] == 0.90
    assert verified["controls"]["threshold"] == 0.90
    assert verified["controls"]["acceptance_rate"] == 0.0


def test_semantic_report_drift_still_raises() -> None:
    rows = _semantic_rows()
    judgments = [
        _semantic_judgment(rows[0], label="eval"),
        _semantic_judgment(rows[1], label="deploy"),
        _semantic_judgment(rows[2], label="eval"),
    ]
    reported = analyze_semantic_judgments(rows, judgments)
    reported["controls"]["acceptable"] = 1

    with pytest.raises(ValueError, match="semantic validation report differs"):
        _verify_semantic_gate(rows, judgments, reported)


def test_semantic_route_provenance_failure_still_raises() -> None:
    rows = _semantic_rows()
    judgments = [
        _semantic_judgment(rows[0], label="eval"),
        _semantic_judgment(rows[1], label="deploy"),
        _semantic_judgment(rows[2], label="ambiguous"),
    ]
    judgments[0]["response_provider"] = "Novita"
    reported = analyze_semantic_judgments(rows, judgments)
    assert reported["passed"] is False

    with pytest.raises(ValueError, match="route provenance differs"):
        _verify_semantic_gate(rows, judgments, reported)


def _manual_rows() -> list[dict]:
    final_neutral = [
        row for row in _neutral_rows() if row["control_partition"] == "final"
    ]
    return (
        _binary_rows("final_counterfactual", 32)
        + final_neutral
        + _factorial_calibration_rows()
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _manual_fixture(
    audit_root: Path,
    *,
    failed_field: str | None = None,
    inconsistent_reviewer: bool = False,
) -> list[dict]:
    rows = _manual_rows()
    packet_path = audit_root / "manual_packet.csv"
    lock_path = audit_root / "manual_packet_lock.json"
    completed_path = audit_root / "manual_completed.csv"
    manifest, _ = build_manual_packet(
        rows,
        packet_path=packet_path,
        lock_path=lock_path,
    )
    _write_json(audit_root / "manual_packet_manifest.json", manifest)
    source_by_id = {row["sample_id"]: row for row in rows}
    with packet_path.open("r", encoding="utf-8-sig", newline="") as handle:
        completed_rows = list(csv.DictReader(handle))
        columns = list(completed_rows[0])
    for completed in completed_rows:
        label = source_by_id[completed["sample_id"]]["acceptable_judge_labels"][0]
        completed.update(
            {
                "reviewer_1_id": "reviewer-a",
                "reviewer_1_label": label,
                "reviewer_1_facts_coherent": "yes",
                "reviewer_1_direct_label_absent": "yes",
                "reviewer_2_id": "reviewer-b",
                "reviewer_2_label": label,
                "reviewer_2_facts_coherent": "yes",
                "reviewer_2_direct_label_absent": "yes",
            }
        )
    if failed_field is not None:
        completed_rows[0][failed_field] = "no"
    if inconsistent_reviewer:
        completed_rows[0]["reviewer_1_id"] = "reviewer-c"
    _write_csv(completed_path, completed_rows, columns)
    report = validate_completed_manual_audit(completed_path, lock_path)
    _write_json(audit_root / "manual_audit.json", report)
    return rows


def test_manual_scientific_failure_returns_failed_verification(tmp_path: Path) -> None:
    rows = _manual_fixture(tmp_path, failed_field="reviewer_1_facts_coherent")

    verified = _verify_manual_gate(rows, audit_root=tmp_path)

    assert verified["passed"] is False
    assert verified["failure_count"] == 1
    assert verified["failures"][0].endswith("facts not coherent")
    assert verified["reviewer_ids"] == ["reviewer-a", "reviewer-b"]


def test_manual_inconsistent_reviewer_ids_return_failed_verification(
    tmp_path: Path,
) -> None:
    rows = _manual_fixture(tmp_path, inconsistent_reviewer=True)

    verified = _verify_manual_gate(rows, audit_root=tmp_path)

    assert verified["passed"] is False
    assert any("one consistent reviewer" in item for item in verified["failures"])
    assert any("exactly two distinct reviewer IDs" in item for item in verified["failures"])


def test_manual_report_drift_still_raises(tmp_path: Path) -> None:
    rows = _manual_fixture(tmp_path, failed_field="reviewer_1_facts_coherent")
    report_path = tmp_path / "manual_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["failure_count"] = 0
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="manual audit report differs"):
        _verify_manual_gate(rows, audit_root=tmp_path)


def test_manual_completed_prompt_drift_still_raises(tmp_path: Path) -> None:
    rows = _manual_fixture(tmp_path)
    completed_path = tmp_path / "manual_completed.csv"
    with completed_path.open("r", encoding="utf-8-sig", newline="") as handle:
        completed_rows = list(csv.DictReader(handle))
        columns = list(completed_rows[0])
    completed_rows[0]["user_prompt"] = "tampered"
    _write_csv(completed_path, completed_rows, columns)

    with pytest.raises(ValueError, match="differs from frozen packet"):
        _verify_manual_gate(rows, audit_root=tmp_path)
