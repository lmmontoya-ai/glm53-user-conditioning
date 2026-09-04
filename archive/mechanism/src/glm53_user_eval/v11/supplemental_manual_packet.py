"""Build a non-gating review packet for omitted semantic disagreements."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

VALID_LABELS = {"eval", "deploy", "ambiguous", "no_status"}
EXPECTED_OMITTED_DISAGREEMENTS = 6
PACKET_COLUMNS = (
    "packet_index",
    "sample_id",
    "split",
    "system_prompt",
    "user_prompt",
    "reviewer_1_id",
    "reviewer_1_label",
    "reviewer_1_facts_coherent",
    "reviewer_1_direct_label_absent",
    "reviewer_1_notes",
    "reviewer_2_id",
    "reviewer_2_label",
    "reviewer_2_facts_coherent",
    "reviewer_2_direct_label_absent",
    "reviewer_2_notes",
    "adjudicated_label",
    "adjudication_notes",
)
IMMUTABLE_PACKET_FIELDS = (
    "packet_index",
    "sample_id",
    "split",
    "system_prompt",
    "user_prompt",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_bytes(path, payload)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"JSONL file contains a non-object row: {path}")
    return rows


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{source} row lacks sample_id")
        if sample_id in indexed:
            raise ValueError(f"{source} contains duplicate sample_id {sample_id}")
        indexed[sample_id] = row
    return indexed


def _original_packet_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("original manual packet contains a blank sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("original manual packet contains duplicate sample IDs")
    return set(sample_ids)


def _read_packet(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PACKET_COLUMNS:
            raise ValueError(f"supplemental packet columns differ: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"supplemental packet contains overflow columns: {path}")
    if any(value is None for row in rows for value in row.values()):
        raise ValueError(f"supplemental packet contains missing columns: {path}")
    return rows


def _as_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"{field} must be yes/no")


def _normalized_reviewer_id(value: str) -> str:
    return value.strip().casefold()


def select_omitted_semantic_disagreements(
    rows: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    original_packet_ids: set[str],
    *,
    expected_count: int = EXPECTED_OMITTED_DISAGREEMENTS,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Select judge disagreements not already present in the frozen packet."""

    by_id = _index_unique(rows, source="dataset")
    judgments_by_id = _index_unique(judgments, source="semantic judgments")
    if set(judgments_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(judgments_by_id))
        extra = sorted(set(judgments_by_id) - set(by_id))
        raise ValueError(
            "semantic judgment IDs differ from the dataset; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    if not original_packet_ids <= set(by_id):
        extra = sorted(original_packet_ids - set(by_id))
        raise ValueError(f"original packet IDs are absent from the dataset: {extra[:5]}")

    disagreement_ids: set[str] = set()
    for sample_id, row in by_id.items():
        acceptable = row.get("acceptable_judge_labels")
        if (
            not isinstance(acceptable, list)
            or not acceptable
            or not set(acceptable) <= VALID_LABELS
        ):
            raise ValueError(f"dataset row {sample_id} has invalid acceptable labels")
        parsed = judgments_by_id[sample_id].get("parsed")
        predicted = parsed.get("label") if isinstance(parsed, Mapping) else None
        if predicted not in VALID_LABELS:
            raise ValueError(f"semantic judgment {sample_id} has an invalid parsed label")
        if predicted not in set(acceptable):
            disagreement_ids.add(sample_id)

    omitted_ids = disagreement_ids - original_packet_ids
    if len(omitted_ids) != expected_count:
        raise ValueError(
            "supplemental packet requires exactly "
            f"{expected_count} omitted disagreements, observed {len(omitted_ids)}"
        )
    selected = [by_id[sample_id] for sample_id in omitted_ids]
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"v11-supplemental-disagreement|{row['sample_id']}".encode()
        ).hexdigest()
    )
    counts = {
        "dataset_rows": len(by_id),
        "semantic_disagreements": len(disagreement_ids),
        "disagreements_in_original_packet": len(
            disagreement_ids & original_packet_ids
        ),
        "omitted_disagreements": len(omitted_ids),
    }
    return selected, counts


def _packet_bytes(selected: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=PACKET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for index, row in enumerate(selected, start=1):
        writer.writerow(
            {
                "packet_index": index,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "system_prompt": row["system_prompt"],
                "user_prompt": row["user_prompt"],
                "reviewer_1_id": "",
                "reviewer_1_label": "",
                "reviewer_1_facts_coherent": "",
                "reviewer_1_direct_label_absent": "",
                "reviewer_1_notes": "",
                "reviewer_2_id": "",
                "reviewer_2_label": "",
                "reviewer_2_facts_coherent": "",
                "reviewer_2_direct_label_absent": "",
                "reviewer_2_notes": "",
                "adjudicated_label": "",
                "adjudication_notes": "",
            }
        )
    return b"\xef\xbb\xbf" + stream.getvalue().encode("utf-8")


def build_supplemental_disagreement_packet(
    *,
    samples_path: Path,
    judgment_rows_dir: Path,
    original_packet_path: Path,
    semantic_validation_path: Path,
    packet_path: Path,
    manifest_path: Path,
    manifest_digest_path: Path,
) -> dict[str, Any]:
    """Build and bind the six-row supplemental packet without changing any gate."""

    rows = _load_jsonl(samples_path)
    judgment_paths = sorted(judgment_rows_dir.glob("*.json"))
    judgments = [json.loads(path.read_text(encoding="utf-8")) for path in judgment_paths]
    judgments_by_id = _index_unique(judgments, source="semantic judgments")
    original_ids = _original_packet_ids(original_packet_path)
    selected, counts = select_omitted_semantic_disagreements(
        rows,
        judgments,
        original_ids,
    )

    semantic_validation = json.loads(semantic_validation_path.read_text(encoding="utf-8"))
    recomputed_disagreement_ids = {
        str(row["sample_id"])
        for row in rows
        if judgments_by_id[str(row["sample_id"])]["parsed"]["label"]
        not in set(row["acceptable_judge_labels"])
    }
    recorded_disagreement_ids = set(semantic_validation.get("disagreement_sample_ids") or [])
    if recomputed_disagreement_ids != recorded_disagreement_ids:
        raise ValueError("semantic validation disagreement IDs do not match recomputation")

    packet_payload = _packet_bytes(selected)
    _atomic_bytes(packet_path, packet_payload)
    packet_sha256 = sha256_bytes(packet_payload)
    judgment_hash_rows = sorted(
        (
            {
                "sample_id": str(judgment["sample_id"]),
                "sha256": sha256_file(path),
            }
            for path, judgment in zip(judgment_paths, judgments, strict=True)
        ),
        key=lambda item: item["sample_id"],
    )
    selected_ids = [str(row["sample_id"]) for row in selected]
    split_counts = Counter(str(row["split"]) for row in selected)
    manifest = {
        "schema_version": "contrastive_prompts_v3_supplemental_manual_packet_v1",
        "scientific_role": "supplemental_non_gating_human_review",
        "selection_rule": (
            "semantic parsed label not in acceptable_judge_labels, then subtract "
            "all sample IDs in the frozen primary manual packet"
        ),
        "row_count": len(selected),
        "counts": counts,
        "split_counts": dict(sorted(split_counts.items())),
        "selected_sample_ids": selected_ids,
        "selected_sample_ids_sha256": sha256_bytes(
            canonical_json(selected_ids).encode("utf-8")
        ),
        "labels_blinded": True,
        "two_distinct_reviewers_required": True,
        "adjudication_required_on_disagreement": True,
        "changes_preregistered_gate": False,
        "changes_semantic_metrics": False,
        "changes_paid_authorization": False,
        "packet_sha256": packet_sha256,
        "source_hashes": {
            "samples_jsonl": sha256_file(samples_path),
            "original_manual_packet": sha256_file(original_packet_path),
            "semantic_validation": sha256_file(semantic_validation_path),
            "semantic_judgment_set": sha256_bytes(
                canonical_json(judgment_hash_rows).encode("utf-8")
            ),
        },
    }
    _atomic_json(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    digest_payload = f"{manifest_sha256}  {manifest_path.name}\n".encode("ascii")
    _atomic_bytes(manifest_digest_path, digest_payload)
    return manifest


def validate_completed_supplemental_review(
    *,
    completed_path: Path,
    frozen_packet_path: Path,
    packet_manifest_path: Path,
    packet_manifest_digest_path: Path,
    samples_path: Path,
) -> dict[str, Any]:
    """Validate a completed packet and report diagnostic source-label agreement."""

    if completed_path.resolve() == frozen_packet_path.resolve():
        raise ValueError("completed review must not overwrite the frozen supplemental packet")
    manifest = json.loads(packet_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != (
        "contrastive_prompts_v3_supplemental_manual_packet_v1"
    ):
        raise ValueError("supplemental packet manifest schema differs")
    if (
        manifest.get("scientific_role") != "supplemental_non_gating_human_review"
        or manifest.get("changes_preregistered_gate") is not False
        or manifest.get("changes_semantic_metrics") is not False
        or manifest.get("changes_paid_authorization") is not False
    ):
        raise ValueError("supplemental packet manifest is not diagnostic-only")
    expected_manifest_digest = (
        f"{sha256_file(packet_manifest_path)}  {packet_manifest_path.name}\n"
    )
    if packet_manifest_digest_path.read_text(encoding="ascii") != expected_manifest_digest:
        raise ValueError("supplemental packet manifest digest differs")
    if sha256_file(frozen_packet_path) != manifest.get("packet_sha256"):
        raise ValueError("frozen supplemental packet hash differs from its manifest")
    source_hashes = manifest.get("source_hashes") or {}
    if sha256_file(samples_path) != source_hashes.get("samples_jsonl"):
        raise ValueError("source dataset hash differs from the supplemental manifest")

    frozen_rows = _read_packet(frozen_packet_path)
    completed_rows = _read_packet(completed_path)
    expected_count = int(manifest.get("row_count", -1))
    if expected_count != EXPECTED_OMITTED_DISAGREEMENTS:
        raise ValueError("supplemental manifest does not bind exactly six rows")
    if len(frozen_rows) != expected_count or len(completed_rows) != expected_count:
        raise ValueError("supplemental packet row count differs")
    frozen_ids = [row["sample_id"] for row in frozen_rows]
    completed_ids = [row["sample_id"] for row in completed_rows]
    if len(set(frozen_ids)) != expected_count or frozen_ids != completed_ids:
        raise ValueError("completed supplemental packet sample IDs or order differ")
    if frozen_ids != manifest.get("selected_sample_ids"):
        raise ValueError("frozen packet sample IDs differ from its manifest")
    for frozen, completed in zip(frozen_rows, completed_rows, strict=True):
        for field in IMMUTABLE_PACKET_FIELDS:
            if completed[field] != frozen[field]:
                raise ValueError(
                    f"{frozen['sample_id']}: completed {field} differs from frozen packet"
                )

    source_by_id = _index_unique(_load_jsonl(samples_path), source="dataset")
    for row in frozen_rows:
        source = source_by_id.get(row["sample_id"])
        if source is None:
            raise ValueError(f"frozen sample {row['sample_id']} is absent from the dataset")
        for field in ("split", "system_prompt", "user_prompt"):
            if row[field] != str(source[field]):
                raise ValueError(
                    f"{row['sample_id']}: frozen {field} differs from source dataset"
                )

    failures: list[str] = []
    reviewer_1_ids = {
        _normalized_reviewer_id(row["reviewer_1_id"])
        for row in completed_rows
        if _normalized_reviewer_id(row["reviewer_1_id"])
    }
    reviewer_2_ids = {
        _normalized_reviewer_id(row["reviewer_2_id"])
        for row in completed_rows
        if _normalized_reviewer_id(row["reviewer_2_id"])
    }
    if len(reviewer_1_ids) != 1:
        failures.append("reviewer_1_id must identify one consistent reviewer across all rows")
    if len(reviewer_2_ids) != 1:
        failures.append("reviewer_2_id must identify one consistent reviewer across all rows")
    if len(reviewer_1_ids | reviewer_2_ids) != 2 or reviewer_1_ids & reviewer_2_ids:
        failures.append("exactly two distinct reviewer IDs are required across the packet")

    reviewer_disagreements = 0
    comparison_rows: list[dict[str, Any]] = []
    final_label_counts: Counter[str] = Counter()
    for row in completed_rows:
        sample_id = row["sample_id"]
        reviewer_1 = _normalized_reviewer_id(row["reviewer_1_id"])
        reviewer_2 = _normalized_reviewer_id(row["reviewer_2_id"])
        label_1 = row["reviewer_1_label"].strip().casefold()
        label_2 = row["reviewer_2_label"].strip().casefold()
        row_valid = True
        if not reviewer_1 or not reviewer_2 or reviewer_1 == reviewer_2:
            failures.append(f"{sample_id}: two distinct reviewer IDs required")
            row_valid = False
        if label_1 not in VALID_LABELS or label_2 not in VALID_LABELS:
            failures.append(f"{sample_id}: reviewer label invalid")
            row_valid = False
        for field in (
            "reviewer_1_facts_coherent",
            "reviewer_2_facts_coherent",
            "reviewer_1_direct_label_absent",
            "reviewer_2_direct_label_absent",
        ):
            try:
                value = _as_bool(row[field], field=field)
            except ValueError as exc:
                failures.append(f"{sample_id}: {exc}")
                row_valid = False
                continue
            if not value:
                message = (
                    "facts not coherent"
                    if "facts_coherent" in field
                    else "direct label statement detected"
                )
                failures.append(f"{sample_id}: {field} reports {message}")
                row_valid = False

        final_label: str | None = None
        if label_1 in VALID_LABELS and label_2 in VALID_LABELS:
            if label_1 != label_2:
                reviewer_disagreements += 1
                adjudicated = row["adjudicated_label"].strip().casefold()
                if adjudicated not in VALID_LABELS or not row[
                    "adjudication_notes"
                ].strip():
                    failures.append(f"{sample_id}: disagreement lacks adjudication")
                    row_valid = False
                else:
                    final_label = adjudicated
            else:
                final_label = label_1
        if row_valid and final_label is not None:
            acceptable = list(source_by_id[sample_id]["acceptable_judge_labels"])
            matches = final_label in set(acceptable)
            final_label_counts[final_label] += 1
            comparison_rows.append(
                {
                    "sample_id": sample_id,
                    "split": row["split"],
                    "final_manual_label": final_label,
                    "acceptable_source_labels": acceptable,
                    "matches_source_contract": matches,
                }
            )

    comparison_matches = sum(
        bool(row["matches_source_contract"]) for row in comparison_rows
    )
    comparison_count = len(comparison_rows)
    passed = not failures and comparison_count == expected_count
    return {
        "schema_version": "contrastive_prompts_v3_supplemental_manual_review_v1",
        "scientific_role": "supplemental_non_gating_diagnostic",
        "passed_review_integrity": passed,
        "row_count": len(completed_rows),
        "reviewer_disagreement_count": reviewer_disagreements,
        "reviewer_ids": sorted(reviewer_1_ids | reviewer_2_ids),
        "reviewer_roles": {
            "reviewer_1_id": (
                next(iter(reviewer_1_ids)) if len(reviewer_1_ids) == 1 else None
            ),
            "reviewer_2_id": (
                next(iter(reviewer_2_ids)) if len(reviewer_2_ids) == 1 else None
            ),
        },
        "final_label_counts": dict(sorted(final_label_counts.items())),
        "source_contract_comparison": {
            "compared_count": comparison_count,
            "match_count": comparison_matches,
            "mismatch_count": comparison_count - comparison_matches,
            "match_rate": (
                comparison_matches / comparison_count if comparison_count else None
            ),
            "rows": comparison_rows,
            "gating": False,
        },
        "failure_count": len(failures),
        "failures": failures,
        "changes_preregistered_gate": False,
        "changes_semantic_metrics": False,
        "changes_paid_authorization": False,
        "input_hashes": {
            "completed_packet": sha256_file(completed_path),
            "frozen_packet": sha256_file(frozen_packet_path),
            "packet_manifest": sha256_file(packet_manifest_path),
            "packet_manifest_digest": sha256_file(packet_manifest_digest_path),
            "samples_jsonl": sha256_file(samples_path),
        },
    }


def write_supplemental_review_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    report_manifest_path: Path,
) -> dict[str, Any]:
    """Atomically write the diagnostic report and its compact binding manifest."""

    _atomic_json(report_path, report)
    report_manifest = {
        "schema_version": "contrastive_prompts_v3_supplemental_review_manifest_v1",
        "scientific_role": "supplemental_non_gating_diagnostic",
        "report_sha256": sha256_file(report_path),
        "passed_review_integrity": bool(report.get("passed_review_integrity")),
        "row_count": int(report.get("row_count", 0)),
        "changes_preregistered_gate": False,
        "changes_semantic_metrics": False,
        "changes_paid_authorization": False,
        "input_hashes": dict(report.get("input_hashes") or {}),
    }
    _atomic_json(report_manifest_path, report_manifest)
    return report_manifest


__all__ = [
    "EXPECTED_OMITTED_DISAGREEMENTS",
    "PACKET_COLUMNS",
    "build_supplemental_disagreement_packet",
    "select_omitted_semantic_disagreements",
    "validate_completed_supplemental_review",
    "write_supplemental_review_report",
]
