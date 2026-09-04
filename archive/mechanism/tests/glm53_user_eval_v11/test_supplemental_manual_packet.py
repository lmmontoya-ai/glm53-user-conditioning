from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from src.glm53_user_eval.v11.supplemental_manual_packet import (
    PACKET_COLUMNS,
    build_supplemental_disagreement_packet,
    select_omitted_semantic_disagreements,
    validate_completed_supplemental_review,
    write_supplemental_review_report,
)


def _sample(index: int, *, acceptable: str = "deploy") -> dict:
    return {
        "sample_id": f"sample-{index:03d}",
        "split": "train",
        "system_prompt": f"System {index}",
        "user_prompt": f"User {index}",
        "acceptable_judge_labels": [acceptable],
    }


def _judgment(row: dict, *, label: str | None = None) -> dict:
    return {
        "sample_id": row["sample_id"],
        "parsed": {"label": label or row["acceptable_judge_labels"][0]},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_packet(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        writer.writerows({"sample_id": row["sample_id"]} for row in rows)


def _fixture(tmp_path: Path) -> dict[str, Path | list[dict]]:
    rows = [_sample(index) for index in range(12)]
    judgments = [
        _judgment(row, label="eval" if index < 9 else "deploy")
        for index, row in enumerate(rows)
    ]
    samples_path = tmp_path / "samples.jsonl"
    _write_jsonl(samples_path, rows)
    judgment_dir = tmp_path / "judgments"
    judgment_dir.mkdir()
    for judgment in judgments:
        (judgment_dir / f"{judgment['sample_id']}.json").write_text(
            json.dumps(judgment, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    original_packet = tmp_path / "manual_packet.csv"
    _write_packet(original_packet, rows[:3])
    semantic_validation = tmp_path / "semantic_validation.json"
    semantic_validation.write_text(
        json.dumps(
            {
                "disagreement_sample_ids": [
                    row["sample_id"] for row in rows[:9]
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows,
        "judgments": judgments,
        "samples_path": samples_path,
        "judgment_dir": judgment_dir,
        "original_packet": original_packet,
        "semantic_validation": semantic_validation,
    }


def _built_fixture(tmp_path: Path) -> dict[str, Path | list[dict]]:
    fixture = _fixture(tmp_path)
    packet = tmp_path / "supplemental.csv"
    manifest = tmp_path / "manifest.json"
    digest = tmp_path / "manifest.sha256"
    build_supplemental_disagreement_packet(
        samples_path=fixture["samples_path"],
        judgment_rows_dir=fixture["judgment_dir"],
        original_packet_path=fixture["original_packet"],
        semantic_validation_path=fixture["semantic_validation"],
        packet_path=packet,
        manifest_path=manifest,
        manifest_digest_path=digest,
    )
    fixture.update({"packet": packet, "manifest": manifest, "digest": digest})
    return fixture


def _complete_packet(
    frozen: Path,
    completed: Path,
    *,
    mismatch_source_on_first: bool = False,
    reviewer_disagreement_on_first: bool = False,
    adjudicate: bool = True,
) -> None:
    with frozen.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    for index, row in enumerate(rows):
        label_1 = "eval" if mismatch_source_on_first and index == 0 else "deploy"
        label_2 = "eval" if reviewer_disagreement_on_first and index == 0 else label_1
        row.update(
            {
                "reviewer_1_id": "reviewer-a",
                "reviewer_1_label": label_1,
                "reviewer_1_facts_coherent": "yes",
                "reviewer_1_direct_label_absent": "yes",
                "reviewer_2_id": "reviewer-b",
                "reviewer_2_label": label_2,
                "reviewer_2_facts_coherent": "yes",
                "reviewer_2_direct_label_absent": "yes",
                "adjudicated_label": (
                    "deploy" if reviewer_disagreement_on_first and adjudicate else ""
                ),
                "adjudication_notes": (
                    "Resolved from the stated use."
                    if reviewer_disagreement_on_first and adjudicate
                    else ""
                ),
            }
        )
    with completed.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_selection_computes_disagreements_then_subtracts_original_ids() -> None:
    rows = [_sample(index) for index in range(12)]
    judgments = [
        _judgment(row, label="eval" if index < 9 else "deploy")
        for index, row in enumerate(rows)
    ]
    selected, counts = select_omitted_semantic_disagreements(
        rows,
        judgments,
        {row["sample_id"] for row in rows[:3]},
    )
    assert {row["sample_id"] for row in selected} == {
        row["sample_id"] for row in rows[3:9]
    }
    assert counts == {
        "dataset_rows": 12,
        "semantic_disagreements": 9,
        "disagreements_in_original_packet": 3,
        "omitted_disagreements": 6,
    }


def test_selection_fails_closed_unless_exactly_six_rows_remain() -> None:
    rows = [_sample(index) for index in range(7)]
    judgments = [_judgment(row, label="eval") for row in rows]
    with pytest.raises(ValueError, match="exactly 6.*observed 7"):
        select_omitted_semantic_disagreements(rows, judgments, set())


def test_packet_is_blinded_non_gating_and_bound_by_hashes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    packet = tmp_path / "supplemental.csv"
    manifest_path = tmp_path / "manifest.json"
    digest_path = tmp_path / "manifest.sha256"
    manifest = build_supplemental_disagreement_packet(
        samples_path=fixture["samples_path"],
        judgment_rows_dir=fixture["judgment_dir"],
        original_packet_path=fixture["original_packet"],
        semantic_validation_path=fixture["semantic_validation"],
        packet_path=packet,
        manifest_path=manifest_path,
        manifest_digest_path=digest_path,
    )

    with packet.open("r", encoding="utf-8-sig", newline="") as handle:
        packet_rows = list(csv.DictReader(handle))
    assert len(packet_rows) == 6
    assert tuple(packet_rows[0]) == PACKET_COLUMNS
    assert "acceptable_judge_labels" not in packet_rows[0]
    assert "semantic_judge_label" not in packet_rows[0]
    assert all(not row["reviewer_1_label"] for row in packet_rows)
    assert all(not row["reviewer_2_label"] for row in packet_rows)
    assert manifest["scientific_role"] == "supplemental_non_gating_human_review"
    assert manifest["changes_preregistered_gate"] is False
    assert manifest["changes_semantic_metrics"] is False
    assert manifest["changes_paid_authorization"] is False
    assert manifest["packet_sha256"] == hashlib.sha256(packet.read_bytes()).hexdigest()
    expected_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest_path.read_text(encoding="ascii") == (
        f"{expected_digest}  {manifest_path.name}\n"
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_builder_rejects_a_semantic_summary_that_differs_from_rows(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["semantic_validation"].write_text(
        json.dumps({"disagreement_sample_ids": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="do not match recomputation"):
        build_supplemental_disagreement_packet(
            samples_path=fixture["samples_path"],
            judgment_rows_dir=fixture["judgment_dir"],
            original_packet_path=fixture["original_packet"],
            semantic_validation_path=fixture["semantic_validation"],
            packet_path=tmp_path / "supplemental.csv",
            manifest_path=tmp_path / "manifest.json",
            manifest_digest_path=tmp_path / "manifest.sha256",
        )


def test_completed_review_validates_and_writes_a_non_gating_report(
    tmp_path: Path,
) -> None:
    fixture = _built_fixture(tmp_path)
    completed = tmp_path / "completed.csv"
    _complete_packet(fixture["packet"], completed)
    report = validate_completed_supplemental_review(
        completed_path=completed,
        frozen_packet_path=fixture["packet"],
        packet_manifest_path=fixture["manifest"],
        packet_manifest_digest_path=fixture["digest"],
        samples_path=fixture["samples_path"],
    )
    assert report["passed_review_integrity"] is True
    assert report["reviewer_ids"] == ["reviewer-a", "reviewer-b"]
    assert report["source_contract_comparison"]["match_count"] == 6
    assert report["changes_semantic_metrics"] is False
    assert report["changes_paid_authorization"] is False

    report_path = tmp_path / "report.json"
    report_manifest_path = tmp_path / "report_manifest.json"
    report_manifest = write_supplemental_review_report(
        report,
        report_path=report_path,
        report_manifest_path=report_manifest_path,
    )
    assert report_manifest["report_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert report_manifest["changes_preregistered_gate"] is False
    assert not list(tmp_path.glob("*.tmp"))


def test_source_label_mismatch_is_reported_but_does_not_fail_integrity(
    tmp_path: Path,
) -> None:
    fixture = _built_fixture(tmp_path)
    completed = tmp_path / "completed.csv"
    _complete_packet(fixture["packet"], completed, mismatch_source_on_first=True)
    report = validate_completed_supplemental_review(
        completed_path=completed,
        frozen_packet_path=fixture["packet"],
        packet_manifest_path=fixture["manifest"],
        packet_manifest_digest_path=fixture["digest"],
        samples_path=fixture["samples_path"],
    )
    assert report["passed_review_integrity"] is True
    assert report["source_contract_comparison"]["mismatch_count"] == 1
    assert report["source_contract_comparison"]["gating"] is False


def test_validator_rejects_prompt_edits_and_overwriting_the_frozen_packet(
    tmp_path: Path,
) -> None:
    fixture = _built_fixture(tmp_path)
    completed = tmp_path / "completed.csv"
    _complete_packet(fixture["packet"], completed)
    with completed.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    rows[0]["user_prompt"] += " changed"
    with completed.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="user_prompt differs"):
        validate_completed_supplemental_review(
            completed_path=completed,
            frozen_packet_path=fixture["packet"],
            packet_manifest_path=fixture["manifest"],
            packet_manifest_digest_path=fixture["digest"],
            samples_path=fixture["samples_path"],
        )
    with pytest.raises(ValueError, match="must not overwrite"):
        validate_completed_supplemental_review(
            completed_path=fixture["packet"],
            frozen_packet_path=fixture["packet"],
            packet_manifest_path=fixture["manifest"],
            packet_manifest_digest_path=fixture["digest"],
            samples_path=fixture["samples_path"],
        )


def test_review_disagreement_requires_adjudication_and_ids_are_consistent(
    tmp_path: Path,
) -> None:
    fixture = _built_fixture(tmp_path)
    completed = tmp_path / "completed.csv"
    _complete_packet(
        fixture["packet"],
        completed,
        reviewer_disagreement_on_first=True,
        adjudicate=False,
    )
    with completed.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    rows[-1]["reviewer_1_id"] = "reviewer-c"
    with completed.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    report = validate_completed_supplemental_review(
        completed_path=completed,
        frozen_packet_path=fixture["packet"],
        packet_manifest_path=fixture["manifest"],
        packet_manifest_digest_path=fixture["digest"],
        samples_path=fixture["samples_path"],
    )
    assert report["passed_review_integrity"] is False
    assert any("consistent reviewer" in failure for failure in report["failures"])
    assert any("lacks adjudication" in failure for failure in report["failures"])
