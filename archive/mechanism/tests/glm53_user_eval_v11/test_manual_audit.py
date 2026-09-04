from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest
from pipelines.glm53_user_eval.v11 import run as supervisor
from src.glm53_user_eval.v11.builder import (
    _binary_rows,
    _factorial_calibration_rows,
    _neutral_rows,
)
from src.glm53_user_eval.v11.manual_audit import (
    build_manual_packet,
    validate_completed_manual_audit,
)


def audit_rows() -> list[dict]:
    final_neutral = [
        row for row in _neutral_rows() if row["control_partition"] == "final"
    ]
    return (
        _binary_rows("final_counterfactual", 32)
        + final_neutral
        + _factorial_calibration_rows()
    )


def completed_packet_rows(packet: Path) -> tuple[list[dict[str, str]], list[str]]:
    source_by_id = {row["sample_id"]: row for row in audit_rows()}
    with packet.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    for row in rows:
        label = source_by_id[row["sample_id"]]["acceptable_judge_labels"][0]
        row.update(
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
    return rows, columns


def write_packet(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_packet_contains_every_final_and_calibration_row_without_labels(tmp_path) -> None:
    packet = tmp_path / "packet.csv"
    lock = tmp_path / "lock.json"
    manifest, _ = build_manual_packet(audit_rows(), packet_path=packet, lock_path=lock)
    assert manifest["row_count"] == 128
    assert manifest["final_counterfactual_rows"] == 64
    assert manifest["factorial_calibration_rows"] == 32
    assert manifest["final_neutral_rows"] == 32
    with packet.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 128
    assert "label" not in rows[0]
    assert "factors" not in rows[0]


def test_completed_two_reviewer_packet_passes_when_labels_match(tmp_path) -> None:
    packet = tmp_path / "manual_packet.csv"
    completed = tmp_path / "manual_completed.csv"
    lock = tmp_path / "lock.json"
    build_manual_packet(audit_rows(), packet_path=packet, lock_path=lock)
    rows, columns = completed_packet_rows(packet)
    write_packet(completed, rows, columns)
    report = validate_completed_manual_audit(completed, lock)
    assert report["passed"] is True
    assert report["reviewed_fraction"] == 1.0
    assert report["reviewer_ids"] == ["reviewer-a", "reviewer-b"]
    assert report["reviewer_roles"] == {
        "reviewer_1_id": "reviewer-a",
        "reviewer_2_id": "reviewer-b",
    }


def test_disagreement_requires_adjudication(tmp_path) -> None:
    packet = tmp_path / "manual_packet.csv"
    completed = tmp_path / "manual_completed.csv"
    lock = tmp_path / "lock.json"
    build_manual_packet(audit_rows(), packet_path=packet, lock_path=lock)
    with packet.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    for row in rows:
        row.update(
            {
                "reviewer_1_id": "reviewer-a",
                "reviewer_1_label": "eval",
                "reviewer_1_facts_coherent": "yes",
                "reviewer_1_direct_label_absent": "yes",
                "reviewer_2_id": "reviewer-b",
                "reviewer_2_label": "deploy",
                "reviewer_2_facts_coherent": "yes",
                "reviewer_2_direct_label_absent": "yes",
            }
        )
    write_packet(completed, rows, columns)
    report = validate_completed_manual_audit(completed, lock)
    assert report["passed"] is False
    assert any("lacks adjudication" in failure for failure in report["failures"])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("packet_index", "999"),
        ("split", "train"),
        ("system_prompt", "altered system prompt"),
        ("user_prompt", "altered user prompt"),
    ],
)
def test_completed_packet_rejects_changed_frozen_fields(
    tmp_path,
    field: str,
    replacement: str,
) -> None:
    packet = tmp_path / "manual_packet.csv"
    completed = tmp_path / "manual_completed.csv"
    lock = tmp_path / "manual_packet_lock.json"
    build_manual_packet(audit_rows(), packet_path=packet, lock_path=lock)
    rows, columns = completed_packet_rows(packet)
    rows[0][field] = replacement
    write_packet(completed, rows, columns)
    with pytest.raises(ValueError, match=f"completed {field} differs from frozen packet"):
        validate_completed_manual_audit(completed, lock)


def test_completed_packet_rejects_a_third_reviewer_identity(tmp_path) -> None:
    packet = tmp_path / "manual_packet.csv"
    completed = tmp_path / "manual_completed.csv"
    lock = tmp_path / "manual_packet_lock.json"
    build_manual_packet(audit_rows(), packet_path=packet, lock_path=lock)
    rows, columns = completed_packet_rows(packet)
    rows[0]["reviewer_1_id"] = "reviewer-c"
    write_packet(completed, rows, columns)
    report = validate_completed_manual_audit(completed, lock)
    assert report["passed"] is False
    assert any("reviewer_1_id must identify one consistent reviewer" in failure for failure in report["failures"])
    assert any("exactly two distinct reviewer IDs" in failure for failure in report["failures"])


def test_completed_packet_rejects_swapped_reviewer_roles(tmp_path) -> None:
    packet = tmp_path / "manual_packet.csv"
    completed = tmp_path / "manual_completed.csv"
    lock = tmp_path / "manual_packet_lock.json"
    build_manual_packet(audit_rows(), packet_path=packet, lock_path=lock)
    rows, columns = completed_packet_rows(packet)
    rows[0]["reviewer_1_id"] = "reviewer-b"
    rows[0]["reviewer_2_id"] = "reviewer-a"
    write_packet(completed, rows, columns)
    report = validate_completed_manual_audit(completed, lock)
    assert report["passed"] is False
    assert any("reviewer_1_id must identify one consistent reviewer" in failure for failure in report["failures"])
    assert any("reviewer_2_id must identify one consistent reviewer" in failure for failure in report["failures"])


def test_completed_packet_requires_nonempty_reviewer_ids_on_every_row(tmp_path) -> None:
    packet = tmp_path / "manual_packet.csv"
    completed = tmp_path / "manual_completed.csv"
    lock = tmp_path / "manual_packet_lock.json"
    build_manual_packet(audit_rows(), packet_path=packet, lock_path=lock)
    rows, columns = completed_packet_rows(packet)
    rows[0]["reviewer_2_id"] = " "
    write_packet(completed, rows, columns)
    report = validate_completed_manual_audit(completed, lock)
    assert report["passed"] is False
    assert any("two distinct reviewer IDs required" in failure for failure in report["failures"])


def test_supervisor_accepts_completed_packet_at_canonical_path(
    tmp_path,
    monkeypatch,
) -> None:
    canonical = tmp_path / "manual_completed.csv"
    canonical.write_text("completed packet\n", encoding="utf-8")
    lock = tmp_path / "manual_packet_lock.json"
    lock.write_text("{}\n", encoding="utf-8")

    def fake_validate(completed_path, lock_path):
        assert completed_path == canonical
        assert lock_path == lock
        return {"passed": True}

    monkeypatch.setattr(supervisor, "validate_completed_manual_audit", fake_validate)
    supervisor.command_validate_manual(
        Namespace(completed_manual_audit=canonical, audit_root=tmp_path)
    )
    assert json.loads((tmp_path / "manual_audit.json").read_text(encoding="utf-8")) == {
        "passed": True
    }
