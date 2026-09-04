"""Independent-review handoff and deterministic merge workflow for V11.

This module never reads source labels or semantic-judge outputs. It authenticates
the two frozen blinded packets, makes one-column-set copies for each reviewer,
and merges only completed human fields back into the canonical packet schema.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.glm53_user_eval.v11.manual_audit import (
    IMMUTABLE_PACKET_FIELDS,
    PACKET_COLUMNS,
    VALID_LABELS,
    sha256_text,
)

PacketKind = Literal["primary", "supplemental"]

REVIEWER_COLUMNS = (
    *IMMUTABLE_PACKET_FIELDS,
    "reviewer_id",
    "label",
    "facts_coherent",
    "direct_label_absent",
    "notes",
)

ADJUDICATION_COLUMNS = (
    *IMMUTABLE_PACKET_FIELDS,
    "reviewer_1_label",
    "reviewer_1_facts_coherent",
    "reviewer_1_direct_label_absent",
    "reviewer_1_notes",
    "reviewer_2_label",
    "reviewer_2_facts_coherent",
    "reviewer_2_direct_label_absent",
    "reviewer_2_notes",
    "adjudicator_id",
    "adjudicated_label",
    "adjudication_notes",
)

_PRIMARY_SCHEMA = "contrastive_prompts_v3_manual_audit_lock_v1"
_PRIMARY_MANIFEST_SCHEMA = "contrastive_prompts_v3_manual_packet_v1"
_SUPPLEMENTAL_SCHEMA = "contrastive_prompts_v3_supplemental_manual_packet_v1"
_ASSIGNMENT_SCHEMA = "contrastive_prompts_v3_review_assignment_v1"
_MERGE_SCHEMA = "contrastive_prompts_v3_independent_review_merge_v1"
_FINAL_SCHEMA = "contrastive_prompts_v3_adjudication_merge_v1"


@dataclass(frozen=True)
class WorkflowPaths:
    """Resolved paths for one packet kind."""

    kind: PacketKind
    source_packet: Path
    canonical_completed: Path
    reviewer_1_template: Path
    reviewer_2_template: Path
    assignment_manifest: Path
    assignment_digest: Path
    merge_manifest: Path
    merge_digest: Path
    adjudication_template: Path
    final_manifest: Path
    final_digest: Path


def _paths(audit_root: Path, review_root: Path, kind: PacketKind) -> WorkflowPaths:
    if kind == "primary":
        source_packet = audit_root / "manual_packet.csv"
        canonical_completed = audit_root / "manual_completed.csv"
    elif kind == "supplemental":
        source_packet = audit_root / "supplemental_semantic_disagreements.csv"
        canonical_completed = (
            audit_root / "supplemental_semantic_disagreements_completed.csv"
        )
    else:  # pragma: no cover, enforced by argparse and type checkers
        raise ValueError(f"unknown packet kind: {kind}")
    return WorkflowPaths(
        kind=kind,
        source_packet=source_packet,
        canonical_completed=canonical_completed,
        reviewer_1_template=review_root / "to_reviewer_1" / f"{kind}_review.csv",
        reviewer_2_template=review_root / "to_reviewer_2" / f"{kind}_review.csv",
        assignment_manifest=review_root / "admin" / f"{kind}_assignment.json",
        assignment_digest=review_root / "admin" / f"{kind}_assignment.sha256",
        merge_manifest=review_root / "admin" / f"{kind}_review_merge.json",
        merge_digest=review_root / "admin" / f"{kind}_review_merge.sha256",
        adjudication_template=(
            review_root / "to_adjudicator" / f"{kind}_adjudication.csv"
        ),
        final_manifest=review_root / "admin" / f"{kind}_adjudication_merge.json",
        final_digest=review_root / "admin" / f"{kind}_adjudication_merge.sha256",
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_once_or_identical(path: Path, payload: bytes) -> None:
    """Create a workflow artifact once, allowing only byte-identical replays."""

    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to replace non-identical workflow artifact: {path}")
        return
    _atomic_bytes(path, payload)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_manifest(path: Path, digest_path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_bytes(value)
    digest_payload = f"{_sha256_bytes(payload)}  {path.name}\n".encode("ascii")
    _write_once_or_identical(path, payload)
    _write_once_or_identical(digest_path, digest_payload)


def _load_bound_manifest(path: Path, digest_path: Path) -> dict[str, Any]:
    expected = f"{_sha256_file(path)}  {path.name}\n"
    if digest_path.read_text(encoding="ascii") != expected:
        raise ValueError(f"manifest digest differs: {digest_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"manifest is not an object: {path}")
    return value


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return b"\xef\xbb\xbf" + stream.getvalue().encode("utf-8")


def _read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"CSV columns differ: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"CSV has overflow columns: {path}")
    if any(value is None for row in rows for value in row.values()):
        raise ValueError(f"CSV has missing columns: {path}")
    return rows


def _validate_reviewer_id(value: str, *, field: str) -> str:
    if value != value.strip() or not value:
        raise ValueError(f"{field} must be nonempty and have no surrounding whitespace")
    if len(value) > 120 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} contains an invalid control character or is too long")
    return value


def _assert_distinct_people(*values: str) -> None:
    normalized = [value.casefold() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("reviewers and adjudicator must have distinct IDs")


def _validate_frozen_rows(rows: Sequence[Mapping[str, str]], *, expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"frozen packet row count differs: expected {expected_count}, observed {len(rows)}"
        )
    sample_ids = [row["sample_id"] for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(rows):
        raise ValueError("frozen packet sample IDs are blank or duplicated")
    expected_indices = [str(index) for index in range(1, len(rows) + 1)]
    if [row["packet_index"] for row in rows] != expected_indices:
        raise ValueError("frozen packet indices or row order differ")
    mutable = set(PACKET_COLUMNS) - set(IMMUTABLE_PACKET_FIELDS)
    for row in rows:
        if any(row[field] for field in mutable):
            raise ValueError("frozen packet already contains human-review values")


def _authenticate_primary(audit_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    packet_path = audit_root / "manual_packet.csv"
    lock_path = audit_root / "manual_packet_lock.json"
    manifest_path = audit_root / "manual_packet_manifest.json"
    rows = _read_csv(packet_path, PACKET_COLUMNS)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != _PRIMARY_SCHEMA:
        raise ValueError("primary packet lock schema differs")
    if manifest.get("schema_version") != _PRIMARY_MANIFEST_SCHEMA:
        raise ValueError("primary packet manifest schema differs")
    expected_count = int(lock.get("row_count", -1))
    if expected_count != 128 or int(manifest.get("row_count", -1)) != expected_count:
        raise ValueError("primary packet authorities do not bind 128 rows")
    packet_sha256 = _sha256_file(packet_path)
    if packet_sha256 != lock.get("packet_sha256"):
        raise ValueError("primary packet hash differs from its lock")
    if packet_sha256 != manifest.get("packet_sha256"):
        raise ValueError("primary packet hash differs from its manifest")
    _validate_frozen_rows(rows, expected_count=expected_count)
    expected = lock.get("expected")
    if not isinstance(expected, dict) or set(expected) != {row["sample_id"] for row in rows}:
        raise ValueError("primary packet IDs differ from its lock")
    for row in rows:
        authority = expected[row["sample_id"]]
        prompt_sha256 = sha256_text(
            f"{row['system_prompt']}\n<USER>\n{row['user_prompt']}"
        )
        if row["split"] != authority.get("split"):
            raise ValueError(f"{row['sample_id']}: primary split differs from its lock")
        if prompt_sha256 != authority.get("prompt_sha256"):
            raise ValueError(f"{row['sample_id']}: primary prompt differs from its lock")
    binding = {
        "kind": "primary",
        "row_count": expected_count,
        "packet_sha256": packet_sha256,
        "sample_order_sha256": _sha256_bytes(
            _canonical_json([row["sample_id"] for row in rows]).encode("utf-8")
        ),
        "authority_sha256": {
            "manual_packet_lock.json": _sha256_file(lock_path),
            "manual_packet_manifest.json": _sha256_file(manifest_path),
        },
    }
    return rows, binding


def _authenticate_supplemental(
    audit_root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    packet_path = audit_root / "supplemental_semantic_disagreements.csv"
    manifest_path = audit_root / "supplemental_semantic_disagreements_manifest.json"
    digest_path = audit_root / "supplemental_semantic_disagreements_manifest.sha256"
    manifest = _load_bound_manifest(manifest_path, digest_path)
    if manifest.get("schema_version") != _SUPPLEMENTAL_SCHEMA:
        raise ValueError("supplemental packet manifest schema differs")
    if (
        manifest.get("scientific_role") != "supplemental_non_gating_human_review"
        or manifest.get("changes_preregistered_gate") is not False
        or manifest.get("changes_semantic_metrics") is not False
        or manifest.get("changes_paid_authorization") is not False
    ):
        raise ValueError("supplemental packet authority is not diagnostic-only")
    rows = _read_csv(packet_path, PACKET_COLUMNS)
    expected_count = int(manifest.get("row_count", -1))
    if expected_count != 6:
        raise ValueError("supplemental packet authority does not bind six rows")
    packet_sha256 = _sha256_file(packet_path)
    if packet_sha256 != manifest.get("packet_sha256"):
        raise ValueError("supplemental packet hash differs from its manifest")
    _validate_frozen_rows(rows, expected_count=expected_count)
    sample_ids = [row["sample_id"] for row in rows]
    if sample_ids != manifest.get("selected_sample_ids"):
        raise ValueError("supplemental packet row order differs from its manifest")
    primary_sha256 = _sha256_file(audit_root / "manual_packet.csv")
    source_hashes = manifest.get("source_hashes") or {}
    if primary_sha256 != source_hashes.get("original_manual_packet"):
        raise ValueError("supplemental authority does not bind the frozen primary packet")
    binding = {
        "kind": "supplemental",
        "row_count": expected_count,
        "packet_sha256": packet_sha256,
        "sample_order_sha256": _sha256_bytes(
            _canonical_json(sample_ids).encode("utf-8")
        ),
        "scientific_role": "supplemental_non_gating_human_review",
        "authority_sha256": {
            "supplemental_semantic_disagreements_manifest.json": _sha256_file(
                manifest_path
            ),
            "supplemental_semantic_disagreements_manifest.sha256": _sha256_file(
                digest_path
            ),
        },
    }
    return rows, binding


def authenticate_source_packet(
    audit_root: Path, kind: PacketKind
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Authenticate one frozen blinded packet without returning source labels."""

    if kind == "primary":
        return _authenticate_primary(audit_root)
    if kind == "supplemental":
        return _authenticate_supplemental(audit_root)
    raise ValueError(f"unknown packet kind: {kind}")


def _reviewer_rows(
    source_rows: Sequence[Mapping[str, str]], reviewer_id: str
) -> list[dict[str, str]]:
    return [
        {
            **{field: row[field] for field in IMMUTABLE_PACKET_FIELDS},
            "reviewer_id": reviewer_id,
            "label": "",
            "facts_coherent": "",
            "direct_label_absent": "",
            "notes": "",
        }
        for row in source_rows
    ]


def prepare_review_assignment(
    *,
    audit_root: Path,
    review_root: Path,
    kind: PacketKind,
    reviewer_1_id: str,
    reviewer_2_id: str,
) -> dict[str, Any]:
    """Create separate, write-once sheets for two independent reviewers."""

    reviewer_1_id = _validate_reviewer_id(reviewer_1_id, field="reviewer_1_id")
    reviewer_2_id = _validate_reviewer_id(reviewer_2_id, field="reviewer_2_id")
    _assert_distinct_people(reviewer_1_id, reviewer_2_id)
    paths = _paths(audit_root, review_root, kind)
    source_rows, source_binding = authenticate_source_packet(audit_root, kind)
    reviewer_1_payload = _csv_bytes(
        REVIEWER_COLUMNS, _reviewer_rows(source_rows, reviewer_1_id)
    )
    reviewer_2_payload = _csv_bytes(
        REVIEWER_COLUMNS, _reviewer_rows(source_rows, reviewer_2_id)
    )
    manifest: dict[str, Any] = {
        "schema_version": _ASSIGNMENT_SCHEMA,
        "packet_kind": kind,
        "row_count": len(source_rows),
        "source_binding": source_binding,
        "reviewers": {
            "reviewer_1": {
                "reviewer_id": reviewer_1_id,
                "template_sha256": _sha256_bytes(reviewer_1_payload),
                "visible_human_role": "reviewer_1_only",
            },
            "reviewer_2": {
                "reviewer_id": reviewer_2_id,
                "template_sha256": _sha256_bytes(reviewer_2_payload),
                "visible_human_role": "reviewer_2_only",
            },
        },
        "labels_blinded": True,
        "reviewers_cannot_see_each_others_fields": True,
        "contains_acceptable_labels": False,
        "contains_semantic_judge_outputs": False,
        "changes_preregistered_gate": False,
        "changes_paid_authorization": False,
    }
    _write_once_or_identical(paths.reviewer_1_template, reviewer_1_payload)
    _write_once_or_identical(paths.reviewer_2_template, reviewer_2_payload)
    _write_manifest(paths.assignment_manifest, paths.assignment_digest, manifest)
    return manifest


def _load_assignment(
    *, audit_root: Path, review_root: Path, kind: PacketKind
) -> tuple[WorkflowPaths, dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    paths = _paths(audit_root, review_root, kind)
    assignment = _load_bound_manifest(paths.assignment_manifest, paths.assignment_digest)
    if assignment.get("schema_version") != _ASSIGNMENT_SCHEMA:
        raise ValueError("review assignment schema differs")
    if assignment.get("packet_kind") != kind:
        raise ValueError("review assignment packet kind differs")
    if (
        assignment.get("labels_blinded") is not True
        or assignment.get("reviewers_cannot_see_each_others_fields") is not True
        or assignment.get("contains_acceptable_labels") is not False
        or assignment.get("contains_semantic_judge_outputs") is not False
        or assignment.get("changes_preregistered_gate") is not False
        or assignment.get("changes_paid_authorization") is not False
    ):
        raise ValueError("review assignment privacy or scientific-role flags differ")
    source_rows, source_binding = authenticate_source_packet(audit_root, kind)
    if assignment.get("source_binding") != source_binding:
        raise ValueError("review assignment does not bind the current frozen packet")
    if int(assignment.get("row_count", -1)) != len(source_rows):
        raise ValueError("review assignment row count differs")
    reviewers = assignment.get("reviewers") or {}
    for role, template in (
        ("reviewer_1", paths.reviewer_1_template),
        ("reviewer_2", paths.reviewer_2_template),
    ):
        expected = (reviewers.get(role) or {}).get("template_sha256")
        if _sha256_file(template) != expected:
            raise ValueError(f"{role} frozen template hash differs")
    return paths, assignment, source_rows, source_binding


def _validate_completed_reviewer_sheet(
    *,
    completed_path: Path,
    source_rows: Sequence[Mapping[str, str]],
    expected_reviewer_id: str,
) -> list[dict[str, str]]:
    rows = _read_csv(completed_path, REVIEWER_COLUMNS)
    if len(rows) != len(source_rows):
        raise ValueError(f"completed reviewer row count differs: {completed_path}")
    normalized: list[dict[str, str]] = []
    for source, row in zip(source_rows, rows, strict=True):
        sample_id = source["sample_id"]
        for field in IMMUTABLE_PACKET_FIELDS:
            if row[field] != source[field]:
                raise ValueError(f"{sample_id}: completed {field} differs from frozen packet")
        if row["reviewer_id"] != expected_reviewer_id:
            raise ValueError(f"{sample_id}: reviewer_id differs from exact assignment")
        label = row["label"].strip().casefold()
        if label not in VALID_LABELS:
            raise ValueError(f"{sample_id}: reviewer label is missing or invalid")
        booleans: dict[str, str] = {}
        for field in ("facts_coherent", "direct_label_absent"):
            value = row[field].strip().casefold()
            if value not in {"yes", "no"}:
                raise ValueError(f"{sample_id}: {field} must be yes or no")
            booleans[field] = value
        normalized.append(
            {
                **{field: source[field] for field in IMMUTABLE_PACKET_FIELDS},
                "reviewer_id": expected_reviewer_id,
                "label": label,
                **booleans,
                "notes": row["notes"],
            }
        )
    return normalized


def _adjudication_rows(
    source_rows: Sequence[Mapping[str, str]],
    reviewer_1_rows: Sequence[Mapping[str, str]],
    reviewer_2_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source, reviewer_1, reviewer_2 in zip(
        source_rows, reviewer_1_rows, reviewer_2_rows, strict=True
    ):
        if reviewer_1["label"] == reviewer_2["label"]:
            continue
        rows.append(
            {
                **{field: source[field] for field in IMMUTABLE_PACKET_FIELDS},
                "reviewer_1_label": reviewer_1["label"],
                "reviewer_1_facts_coherent": reviewer_1["facts_coherent"],
                "reviewer_1_direct_label_absent": reviewer_1[
                    "direct_label_absent"
                ],
                "reviewer_1_notes": reviewer_1["notes"],
                "reviewer_2_label": reviewer_2["label"],
                "reviewer_2_facts_coherent": reviewer_2["facts_coherent"],
                "reviewer_2_direct_label_absent": reviewer_2[
                    "direct_label_absent"
                ],
                "reviewer_2_notes": reviewer_2["notes"],
                "adjudicator_id": "",
                "adjudicated_label": "",
                "adjudication_notes": "",
            }
        )
    return rows


def merge_independent_reviews(
    *,
    audit_root: Path,
    review_root: Path,
    kind: PacketKind,
    reviewer_1_completed: Path,
    reviewer_2_completed: Path,
) -> dict[str, Any]:
    """Validate two private sheets and merge them into the canonical CSV."""

    if reviewer_1_completed.resolve() == reviewer_2_completed.resolve():
        raise ValueError("reviewer completion files must be distinct")
    paths, assignment, source_rows, source_binding = _load_assignment(
        audit_root=audit_root, review_root=review_root, kind=kind
    )
    reviewer_1_id = assignment["reviewers"]["reviewer_1"]["reviewer_id"]
    reviewer_2_id = assignment["reviewers"]["reviewer_2"]["reviewer_id"]
    _assert_distinct_people(reviewer_1_id, reviewer_2_id)
    reviewer_1_rows = _validate_completed_reviewer_sheet(
        completed_path=reviewer_1_completed,
        source_rows=source_rows,
        expected_reviewer_id=reviewer_1_id,
    )
    reviewer_2_rows = _validate_completed_reviewer_sheet(
        completed_path=reviewer_2_completed,
        source_rows=source_rows,
        expected_reviewer_id=reviewer_2_id,
    )
    merged_rows: list[dict[str, str]] = []
    for source, reviewer_1, reviewer_2 in zip(
        source_rows, reviewer_1_rows, reviewer_2_rows, strict=True
    ):
        merged_rows.append(
            {
                **{field: source[field] for field in IMMUTABLE_PACKET_FIELDS},
                "reviewer_1_id": reviewer_1_id,
                "reviewer_1_label": reviewer_1["label"],
                "reviewer_1_facts_coherent": reviewer_1["facts_coherent"],
                "reviewer_1_direct_label_absent": reviewer_1[
                    "direct_label_absent"
                ],
                "reviewer_1_notes": reviewer_1["notes"],
                "reviewer_2_id": reviewer_2_id,
                "reviewer_2_label": reviewer_2["label"],
                "reviewer_2_facts_coherent": reviewer_2["facts_coherent"],
                "reviewer_2_direct_label_absent": reviewer_2[
                    "direct_label_absent"
                ],
                "reviewer_2_notes": reviewer_2["notes"],
                "adjudicated_label": "",
                "adjudication_notes": "",
            }
        )
    adjudication_rows = _adjudication_rows(
        source_rows, reviewer_1_rows, reviewer_2_rows
    )
    merged_payload = _csv_bytes(PACKET_COLUMNS, merged_rows)
    adjudication_payload = _csv_bytes(ADJUDICATION_COLUMNS, adjudication_rows)
    _write_once_or_identical(paths.canonical_completed, merged_payload)
    _write_once_or_identical(paths.adjudication_template, adjudication_payload)
    merge_manifest: dict[str, Any] = {
        "schema_version": _MERGE_SCHEMA,
        "packet_kind": kind,
        "row_count": len(merged_rows),
        "source_binding": source_binding,
        "assignment_manifest_sha256": _sha256_file(paths.assignment_manifest),
        "reviewer_ids": {
            "reviewer_1": reviewer_1_id,
            "reviewer_2": reviewer_2_id,
        },
        "completed_review_sha256": {
            "reviewer_1": _sha256_file(reviewer_1_completed),
            "reviewer_2": _sha256_file(reviewer_2_completed),
        },
        "canonical_unadjudicated_sha256": _sha256_bytes(merged_payload),
        "adjudication_template_sha256": _sha256_bytes(adjudication_payload),
        "disagreement_count": len(adjudication_rows),
        "disagreement_sample_ids": [row["sample_id"] for row in adjudication_rows],
        "ready_for_validation": not adjudication_rows,
        "contains_acceptable_labels": False,
        "contains_semantic_judge_outputs": False,
        "changes_preregistered_gate": False,
        "changes_paid_authorization": False,
    }
    _write_manifest(paths.merge_manifest, paths.merge_digest, merge_manifest)
    return merge_manifest


def _adjudication_template_rows(
    paths: WorkflowPaths, merge_manifest: Mapping[str, Any]
) -> list[dict[str, str]]:
    if _sha256_file(paths.adjudication_template) != merge_manifest.get(
        "adjudication_template_sha256"
    ):
        raise ValueError("adjudication template hash differs from review merge")
    rows = _read_csv(paths.adjudication_template, ADJUDICATION_COLUMNS)
    if [row["sample_id"] for row in rows] != merge_manifest.get(
        "disagreement_sample_ids"
    ):
        raise ValueError("adjudication template rows differ from review merge")
    return rows


def merge_adjudication(
    *,
    audit_root: Path,
    review_root: Path,
    kind: PacketKind,
    completed_adjudication: Path,
    adjudicator_id: str,
) -> dict[str, Any]:
    """Merge a third human's decisions without exposing frozen answers."""

    adjudicator_id = _validate_reviewer_id(adjudicator_id, field="adjudicator_id")
    paths, assignment, source_rows, source_binding = _load_assignment(
        audit_root=audit_root, review_root=review_root, kind=kind
    )
    merge_manifest = _load_bound_manifest(paths.merge_manifest, paths.merge_digest)
    if merge_manifest.get("schema_version") != _MERGE_SCHEMA:
        raise ValueError("review merge schema differs")
    if merge_manifest.get("packet_kind") != kind:
        raise ValueError("review merge packet kind differs")
    if merge_manifest.get("source_binding") != source_binding:
        raise ValueError("review merge does not bind the current frozen packet")
    if merge_manifest.get("assignment_manifest_sha256") != _sha256_file(
        paths.assignment_manifest
    ):
        raise ValueError("review merge does not bind the current assignment")
    reviewer_1_id = assignment["reviewers"]["reviewer_1"]["reviewer_id"]
    reviewer_2_id = assignment["reviewers"]["reviewer_2"]["reviewer_id"]
    _assert_distinct_people(reviewer_1_id, reviewer_2_id, adjudicator_id)
    disagreement_count = int(merge_manifest.get("disagreement_count", -1))
    if disagreement_count <= 0:
        raise ValueError("reviewers agree on every row; adjudication is not required")
    template_rows = _adjudication_template_rows(paths, merge_manifest)
    completed_rows = _read_csv(completed_adjudication, ADJUDICATION_COLUMNS)
    if len(completed_rows) != disagreement_count:
        raise ValueError("completed adjudication row count differs")
    decisions: dict[str, tuple[str, str]] = {}
    for template, completed in zip(template_rows, completed_rows, strict=True):
        sample_id = template["sample_id"]
        for field in ADJUDICATION_COLUMNS:
            if field in {"adjudicator_id", "adjudicated_label", "adjudication_notes"}:
                continue
            if completed[field] != template[field]:
                raise ValueError(
                    f"{sample_id}: completed adjudication {field} differs from template"
                )
        if completed["adjudicator_id"] != adjudicator_id:
            raise ValueError(f"{sample_id}: adjudicator_id differs from exact assignment")
        label = completed["adjudicated_label"].strip().casefold()
        notes = completed["adjudication_notes"]
        if label not in VALID_LABELS:
            raise ValueError(f"{sample_id}: adjudicated label is missing or invalid")
        if not notes.strip():
            raise ValueError(f"{sample_id}: adjudication notes are required")
        decisions[sample_id] = (label, notes)

    current_sha256 = _sha256_file(paths.canonical_completed)
    unadjudicated_sha256 = str(merge_manifest["canonical_unadjudicated_sha256"])
    if current_sha256 != unadjudicated_sha256:
        if not paths.final_manifest.is_file() or not paths.final_digest.is_file():
            raise ValueError("canonical completed packet changed after independent merge")
        existing_final = _load_bound_manifest(paths.final_manifest, paths.final_digest)
        existing_checks = {
            "schema": existing_final.get("schema_version") == _FINAL_SCHEMA,
            "kind": existing_final.get("packet_kind") == kind,
            "source": existing_final.get("source_binding") == source_binding,
            "assignment": existing_final.get("assignment_manifest_sha256")
            == _sha256_file(paths.assignment_manifest),
            "merge": existing_final.get("review_merge_manifest_sha256")
            == _sha256_file(paths.merge_manifest),
            "completed_adjudication": existing_final.get(
                "completed_adjudication_sha256"
            )
            == _sha256_file(completed_adjudication),
            "adjudicator": existing_final.get("adjudicator_id") == adjudicator_id,
            "reviewers": existing_final.get("reviewer_ids")
            == {"reviewer_1": reviewer_1_id, "reviewer_2": reviewer_2_id},
            "row_count": existing_final.get("adjudicated_row_count")
            == disagreement_count,
            "canonical": existing_final.get("canonical_completed_sha256")
            == current_sha256,
            "ready": existing_final.get("ready_for_validation") is True,
            "labels_blinded": existing_final.get("contains_acceptable_labels")
            is False,
            "judge_outputs_absent": existing_final.get(
                "contains_semantic_judge_outputs"
            )
            is False,
            "paid_locked": existing_final.get("changes_paid_authorization") is False,
        }
        if not all(existing_checks.values()):
            failed = sorted(name for name, passed in existing_checks.items() if not passed)
            raise ValueError(f"existing adjudication merge differs: {failed}")
        return existing_final

    unadjudicated_rows = _read_csv(paths.canonical_completed, PACKET_COLUMNS)
    if len(unadjudicated_rows) != len(source_rows):
        raise ValueError("canonical completed packet row count differs")
    final_rows: list[dict[str, str]] = []
    for source, row in zip(source_rows, unadjudicated_rows, strict=True):
        sample_id = source["sample_id"]
        for field in IMMUTABLE_PACKET_FIELDS:
            if row[field] != source[field]:
                raise ValueError(f"{sample_id}: canonical {field} differs from source")
        final_row = dict(row)
        if sample_id in decisions:
            label, notes = decisions[sample_id]
            final_row["adjudicated_label"] = label
            final_row["adjudication_notes"] = notes
        elif row["adjudicated_label"] or row["adjudication_notes"]:
            raise ValueError(f"{sample_id}: agreement row contains adjudication")
        final_rows.append(final_row)
    if set(decisions) != set(merge_manifest["disagreement_sample_ids"]):
        raise ValueError("completed adjudication IDs differ from all disagreements")
    final_payload = _csv_bytes(PACKET_COLUMNS, final_rows)
    final_sha256 = _sha256_bytes(final_payload)
    final_manifest: dict[str, Any] = {
        "schema_version": _FINAL_SCHEMA,
        "packet_kind": kind,
        "row_count": len(final_rows),
        "source_binding": source_binding,
        "assignment_manifest_sha256": _sha256_file(paths.assignment_manifest),
        "review_merge_manifest_sha256": _sha256_file(paths.merge_manifest),
        "completed_adjudication_sha256": _sha256_file(completed_adjudication),
        "adjudicator_id": adjudicator_id,
        "reviewer_ids": {
            "reviewer_1": reviewer_1_id,
            "reviewer_2": reviewer_2_id,
        },
        "adjudicated_row_count": len(decisions),
        "canonical_completed_sha256": final_sha256,
        "ready_for_validation": True,
        "contains_acceptable_labels": False,
        "contains_semantic_judge_outputs": False,
        "changes_preregistered_gate": False,
        "changes_paid_authorization": False,
    }
    # Write the recovery record first. If the process stops before the CSV
    # replacement, an identical retry can finish from the still-authenticated
    # unadjudicated packet. A changed retry cannot overwrite either artifact.
    _write_manifest(paths.final_manifest, paths.final_digest, final_manifest)
    _atomic_bytes(paths.canonical_completed, final_payload)
    return final_manifest


__all__ = [
    "ADJUDICATION_COLUMNS",
    "REVIEWER_COLUMNS",
    "authenticate_source_packet",
    "merge_adjudication",
    "merge_independent_reviews",
    "prepare_review_assignment",
]
