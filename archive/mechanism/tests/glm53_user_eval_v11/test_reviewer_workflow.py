from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from pipelines.glm53_user_eval.v11.run import COMMANDS
from src.glm53_user_eval.v11.builder import (
    _binary_rows,
    _factorial_calibration_rows,
    _neutral_rows,
)
from src.glm53_user_eval.v11.manual_audit import (
    PACKET_COLUMNS,
    build_manual_packet,
    validate_completed_manual_audit,
)
from src.glm53_user_eval.v11.reviewer_workflow import (
    ADJUDICATION_COLUMNS,
    REVIEWER_COLUMNS,
    authenticate_source_packet,
    merge_adjudication,
    merge_independent_reviews,
    prepare_review_assignment,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_rows() -> list[dict]:
    final_neutral = [
        row for row in _neutral_rows() if row["control_partition"] == "final"
    ]
    return (
        _binary_rows("final_counterfactual", 32)
        + final_neutral
        + _factorial_calibration_rows()
    )


def _write_packet(path: Path, rows: list[dict[str, str]], columns=PACKET_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_packet(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_authorities(tmp_path: Path) -> tuple[Path, Path]:
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    primary = audit_root / "manual_packet.csv"
    lock = audit_root / "manual_packet_lock.json"
    primary_manifest, _ = build_manual_packet(
        _audit_rows(), packet_path=primary, lock_path=lock
    )
    (audit_root / "manual_packet_manifest.json").write_text(
        json.dumps(primary_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    primary_rows = _read_packet(primary)
    supplemental_rows = [dict(row) for row in primary_rows[:6]]
    for index, row in enumerate(supplemental_rows, start=1):
        row["packet_index"] = str(index)
    supplemental = audit_root / "supplemental_semantic_disagreements.csv"
    _write_packet(supplemental, supplemental_rows)
    supplemental_manifest = {
        "schema_version": "contrastive_prompts_v3_supplemental_manual_packet_v1",
        "scientific_role": "supplemental_non_gating_human_review",
        "row_count": 6,
        "selected_sample_ids": [row["sample_id"] for row in supplemental_rows],
        "packet_sha256": _sha256(supplemental),
        "source_hashes": {"original_manual_packet": _sha256(primary)},
        "changes_preregistered_gate": False,
        "changes_semantic_metrics": False,
        "changes_paid_authorization": False,
    }
    manifest_path = audit_root / "supplemental_semantic_disagreements_manifest.json"
    manifest_path.write_text(
        json.dumps(supplemental_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest_path = audit_root / "supplemental_semantic_disagreements_manifest.sha256"
    digest_path.write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n", encoding="ascii"
    )
    return audit_root, tmp_path / "human_review"


def _complete_reviewer(
    template: Path,
    destination: Path,
    *,
    label: str = "deploy",
    first_label: str | None = None,
) -> None:
    rows = _read_packet(template)
    for index, row in enumerate(rows):
        row["label"] = first_label if index == 0 and first_label else label
        row["facts_coherent"] = "yes"
        row["direct_label_absent"] = "yes"
        row["notes"] = f"Independent note {index + 1}"
    _write_packet(destination, rows, REVIEWER_COLUMNS)


def _prepare(tmp_path: Path, kind: str = "primary") -> tuple[Path, Path, dict]:
    audit_root, review_root = _build_authorities(tmp_path)
    manifest = prepare_review_assignment(
        audit_root=audit_root,
        review_root=review_root,
        kind=kind,
        reviewer_1_id="reviewer-alpha",
        reviewer_2_id="reviewer-beta",
    )
    return audit_root, review_root, manifest


def test_private_sheets_show_only_the_assigned_reviewer_fields(tmp_path: Path) -> None:
    audit_root, review_root, manifest = _prepare(tmp_path)
    reviewer_1 = review_root / "to_reviewer_1/primary_review.csv"
    reviewer_2 = review_root / "to_reviewer_2/primary_review.csv"
    rows_1 = _read_packet(reviewer_1)
    rows_2 = _read_packet(reviewer_2)

    assert tuple(rows_1[0]) == REVIEWER_COLUMNS
    assert tuple(rows_2[0]) == REVIEWER_COLUMNS
    assert {row["reviewer_id"] for row in rows_1} == {"reviewer-alpha"}
    assert {row["reviewer_id"] for row in rows_2} == {"reviewer-beta"}
    assert all(not row["label"] and not row["notes"] for row in rows_1 + rows_2)
    serialized = reviewer_1.read_text(encoding="utf-8-sig") + json.dumps(manifest)
    assert "reviewer_2_label" not in serialized
    assert "acceptable_judge_labels" not in serialized
    assert "semantic_judge_outputs" not in rows_1[0]
    assert manifest["contains_acceptable_labels"] is False
    assert manifest["contains_semantic_judge_outputs"] is False
    assert authenticate_source_packet(audit_root, "primary")[1] == manifest[
        "source_binding"
    ]


def test_supervisor_exposes_the_three_review_workflow_commands() -> None:
    assert {
        "prepare-human-review",
        "merge-human-reviews",
        "merge-human-adjudication",
    } <= set(COMMANDS)


def test_assignment_is_write_once_and_reviewer_ids_are_exact(tmp_path: Path) -> None:
    audit_root, review_root, _ = _prepare(tmp_path)
    with pytest.raises(ValueError, match="non-identical workflow artifact"):
        prepare_review_assignment(
            audit_root=audit_root,
            review_root=review_root,
            kind="primary",
            reviewer_1_id="different-person",
            reviewer_2_id="reviewer-beta",
        )
    with pytest.raises(ValueError, match="distinct IDs"):
        prepare_review_assignment(
            audit_root=audit_root,
            review_root=tmp_path / "other",
            kind="primary",
            reviewer_1_id="Same Person",
            reviewer_2_id="same person",
        )


def test_source_packet_and_authority_tampering_fail_closed(tmp_path: Path) -> None:
    audit_root, review_root, _ = _prepare(tmp_path)
    packet = audit_root / "manual_packet.csv"
    packet.write_bytes(packet.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash differs"):
        prepare_review_assignment(
            audit_root=audit_root,
            review_root=review_root / "new",
            kind="primary",
            reviewer_1_id="r1",
            reviewer_2_id="r2",
        )


def test_merge_rejects_prompt_edit_row_reorder_and_wrong_reviewer_id(
    tmp_path: Path,
) -> None:
    audit_root, review_root, _ = _prepare(tmp_path)
    template_1 = review_root / "to_reviewer_1/primary_review.csv"
    template_2 = review_root / "to_reviewer_2/primary_review.csv"
    completed_1 = tmp_path / "reviewer_1_completed.csv"
    completed_2 = tmp_path / "reviewer_2_completed.csv"
    _complete_reviewer(template_1, completed_1)
    _complete_reviewer(template_2, completed_2)

    rows = _read_packet(completed_1)
    rows[0]["user_prompt"] += " changed"
    _write_packet(completed_1, rows, REVIEWER_COLUMNS)
    with pytest.raises(ValueError, match="user_prompt differs"):
        merge_independent_reviews(
            audit_root=audit_root,
            review_root=review_root,
            kind="primary",
            reviewer_1_completed=completed_1,
            reviewer_2_completed=completed_2,
        )

    _complete_reviewer(template_1, completed_1)
    rows = _read_packet(completed_1)
    rows[0], rows[1] = rows[1], rows[0]
    _write_packet(completed_1, rows, REVIEWER_COLUMNS)
    with pytest.raises(ValueError, match="packet_index differs"):
        merge_independent_reviews(
            audit_root=audit_root,
            review_root=review_root,
            kind="primary",
            reviewer_1_completed=completed_1,
            reviewer_2_completed=completed_2,
        )

    _complete_reviewer(template_1, completed_1)
    rows = _read_packet(completed_1)
    rows[0]["reviewer_id"] = "Reviewer-Alpha"
    _write_packet(completed_1, rows, REVIEWER_COLUMNS)
    with pytest.raises(ValueError, match="exact assignment"):
        merge_independent_reviews(
            audit_root=audit_root,
            review_root=review_root,
            kind="primary",
            reviewer_1_completed=completed_1,
            reviewer_2_completed=completed_2,
        )


def test_merge_creates_canonical_packet_and_only_disagreements_for_adjudication(
    tmp_path: Path,
) -> None:
    audit_root, review_root, _ = _prepare(tmp_path)
    completed_1 = tmp_path / "reviewer_1_completed.csv"
    completed_2 = tmp_path / "reviewer_2_completed.csv"
    _complete_reviewer(
        review_root / "to_reviewer_1/primary_review.csv", completed_1, label="deploy"
    )
    _complete_reviewer(
        review_root / "to_reviewer_2/primary_review.csv",
        completed_2,
        label="deploy",
        first_label="eval",
    )
    report = merge_independent_reviews(
        audit_root=audit_root,
        review_root=review_root,
        kind="primary",
        reviewer_1_completed=completed_1,
        reviewer_2_completed=completed_2,
    )
    merged = _read_packet(audit_root / "manual_completed.csv")
    adjudication = _read_packet(
        review_root / "to_adjudicator/primary_adjudication.csv"
    )

    assert report["disagreement_count"] == 1
    assert report["ready_for_validation"] is False
    assert len(merged) == 128
    assert merged[0]["reviewer_1_id"] == "reviewer-alpha"
    assert merged[0]["reviewer_2_id"] == "reviewer-beta"
    assert merged[0]["adjudicated_label"] == ""
    assert tuple(adjudication[0]) == ADJUDICATION_COLUMNS
    assert len(adjudication) == 1
    assert adjudication[0]["adjudicator_id"] == ""
    assert "acceptable_judge_labels" not in json.dumps(report)
    assert report["changes_paid_authorization"] is False


def test_primary_merge_output_is_accepted_by_the_existing_validator(
    tmp_path: Path,
) -> None:
    audit_root, review_root, _ = _prepare(tmp_path)
    lock = json.loads((audit_root / "manual_packet_lock.json").read_text())
    completed_paths = [tmp_path / "reviewer_1.csv", tmp_path / "reviewer_2.csv"]
    templates = [
        review_root / "to_reviewer_1/primary_review.csv",
        review_root / "to_reviewer_2/primary_review.csv",
    ]
    for template, completed in zip(templates, completed_paths, strict=True):
        rows = _read_packet(template)
        for row in rows:
            row["label"] = lock["expected"][row["sample_id"]][
                "acceptable_labels"
            ][0]
            row["facts_coherent"] = "yes"
            row["direct_label_absent"] = "yes"
        _write_packet(completed, rows, REVIEWER_COLUMNS)
    merge_independent_reviews(
        audit_root=audit_root,
        review_root=review_root,
        kind="primary",
        reviewer_1_completed=completed_paths[0],
        reviewer_2_completed=completed_paths[1],
    )
    report = validate_completed_manual_audit(
        audit_root / "manual_completed.csv",
        audit_root / "manual_packet_lock.json",
    )
    assert report["passed"] is True
    assert report["reviewer_disagreement_count"] == 0


def test_third_human_adjudication_is_distinct_bound_and_idempotent(
    tmp_path: Path,
) -> None:
    audit_root, review_root, _ = _prepare(tmp_path)
    completed_1 = tmp_path / "reviewer_1_completed.csv"
    completed_2 = tmp_path / "reviewer_2_completed.csv"
    _complete_reviewer(
        review_root / "to_reviewer_1/primary_review.csv", completed_1, label="deploy"
    )
    _complete_reviewer(
        review_root / "to_reviewer_2/primary_review.csv",
        completed_2,
        label="deploy",
        first_label="eval",
    )
    merge_independent_reviews(
        audit_root=audit_root,
        review_root=review_root,
        kind="primary",
        reviewer_1_completed=completed_1,
        reviewer_2_completed=completed_2,
    )
    template = review_root / "to_adjudicator/primary_adjudication.csv"
    completed_adjudication = tmp_path / "adjudication_completed.csv"
    rows = _read_packet(template)
    rows[0]["adjudicator_id"] = "reviewer-gamma"
    rows[0]["adjudicated_label"] = "ambiguous"
    rows[0]["adjudication_notes"] = "Both purposes are material."
    _write_packet(completed_adjudication, rows, ADJUDICATION_COLUMNS)

    with pytest.raises(ValueError, match="distinct IDs"):
        merge_adjudication(
            audit_root=audit_root,
            review_root=review_root,
            kind="primary",
            completed_adjudication=completed_adjudication,
            adjudicator_id="reviewer-alpha",
        )
    report = merge_adjudication(
        audit_root=audit_root,
        review_root=review_root,
        kind="primary",
        completed_adjudication=completed_adjudication,
        adjudicator_id="reviewer-gamma",
    )
    repeated = merge_adjudication(
        audit_root=audit_root,
        review_root=review_root,
        kind="primary",
        completed_adjudication=completed_adjudication,
        adjudicator_id="reviewer-gamma",
    )
    final_rows = _read_packet(audit_root / "manual_completed.csv")
    assert report == repeated
    assert report["adjudicated_row_count"] == 1
    assert report["ready_for_validation"] is True
    assert final_rows[0]["adjudicated_label"] == "ambiguous"
    assert final_rows[0]["adjudication_notes"] == "Both purposes are material."


def test_supplemental_workflow_writes_only_non_gating_canonical_path(
    tmp_path: Path,
) -> None:
    audit_root, review_root, _ = _prepare(tmp_path, kind="supplemental")
    completed_1 = tmp_path / "supplemental_1.csv"
    completed_2 = tmp_path / "supplemental_2.csv"
    _complete_reviewer(
        review_root / "to_reviewer_1/supplemental_review.csv", completed_1
    )
    _complete_reviewer(
        review_root / "to_reviewer_2/supplemental_review.csv", completed_2
    )
    report = merge_independent_reviews(
        audit_root=audit_root,
        review_root=review_root,
        kind="supplemental",
        reviewer_1_completed=completed_1,
        reviewer_2_completed=completed_2,
    )
    assert report["row_count"] == 6
    assert report["ready_for_validation"] is True
    assert report["source_binding"]["scientific_role"] == (
        "supplemental_non_gating_human_review"
    )
    assert (
        audit_root / "supplemental_semantic_disagreements_completed.csv"
    ).is_file()
    assert not (audit_root / "manual_completed.csv").exists()


def test_adjudication_rejects_modified_reviewer_evidence(tmp_path: Path) -> None:
    audit_root, review_root, _ = _prepare(tmp_path)
    completed_1 = tmp_path / "reviewer_1_completed.csv"
    completed_2 = tmp_path / "reviewer_2_completed.csv"
    _complete_reviewer(
        review_root / "to_reviewer_1/primary_review.csv", completed_1
    )
    _complete_reviewer(
        review_root / "to_reviewer_2/primary_review.csv",
        completed_2,
        first_label="eval",
    )
    merge_independent_reviews(
        audit_root=audit_root,
        review_root=review_root,
        kind="primary",
        reviewer_1_completed=completed_1,
        reviewer_2_completed=completed_2,
    )
    template = review_root / "to_adjudicator/primary_adjudication.csv"
    completed = tmp_path / "adjudication.csv"
    rows = _read_packet(template)
    rows[0]["reviewer_1_notes"] = "Changed evidence"
    rows[0]["adjudicator_id"] = "reviewer-gamma"
    rows[0]["adjudicated_label"] = "deploy"
    rows[0]["adjudication_notes"] = "Decision"
    _write_packet(completed, rows, ADJUDICATION_COLUMNS)
    with pytest.raises(ValueError, match="reviewer_1_notes differs"):
        merge_adjudication(
            audit_root=audit_root,
            review_root=review_root,
            kind="primary",
            completed_adjudication=completed,
            adjudicator_id="reviewer-gamma",
        )


def test_adjudication_rejects_a_prefilled_or_modified_canonical_packet(
    tmp_path: Path,
) -> None:
    audit_root, review_root, _ = _prepare(tmp_path)
    completed_1 = tmp_path / "reviewer_1_completed.csv"
    completed_2 = tmp_path / "reviewer_2_completed.csv"
    _complete_reviewer(
        review_root / "to_reviewer_1/primary_review.csv", completed_1
    )
    _complete_reviewer(
        review_root / "to_reviewer_2/primary_review.csv",
        completed_2,
        first_label="eval",
    )
    merge_independent_reviews(
        audit_root=audit_root,
        review_root=review_root,
        kind="primary",
        reviewer_1_completed=completed_1,
        reviewer_2_completed=completed_2,
    )
    adjudication_template = review_root / "to_adjudicator/primary_adjudication.csv"
    adjudication_completed = tmp_path / "adjudication.csv"
    adjudication_rows = _read_packet(adjudication_template)
    adjudication_rows[0]["adjudicator_id"] = "reviewer-gamma"
    adjudication_rows[0]["adjudicated_label"] = "deploy"
    adjudication_rows[0]["adjudication_notes"] = "Independent decision"
    _write_packet(adjudication_completed, adjudication_rows, ADJUDICATION_COLUMNS)

    canonical = audit_root / "manual_completed.csv"
    canonical_rows = _read_packet(canonical)
    canonical_rows[0]["reviewer_1_notes"] = "Modified after merge"
    canonical_rows[0]["adjudicated_label"] = "deploy"
    canonical_rows[0]["adjudication_notes"] = "Independent decision"
    _write_packet(canonical, canonical_rows, PACKET_COLUMNS)
    with pytest.raises(ValueError, match="changed after independent merge"):
        merge_adjudication(
            audit_root=audit_root,
            review_root=review_root,
            kind="primary",
            completed_adjudication=adjudication_completed,
            adjudicator_id="reviewer-gamma",
        )
