"""Blinded two-reviewer manual audit for the frozen v11 holdout."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

VALID_LABELS = {"eval", "deploy", "ambiguous", "no_status"}
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


def _manual_holdout_row(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("split") in {"final_counterfactual", "factorial_calibration"}
        or (
            row.get("split") == "neutral_controls"
            and row.get("control_partition") == "final"
        )
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_packet_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PACKET_COLUMNS:
            raise ValueError(f"manual audit columns differ: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"manual audit contains overflow columns: {path}")
    if any(value is None for row in rows for value in row.values()):
        raise ValueError(f"manual audit contains missing columns: {path}")
    return rows


def _resolve_locked_packet(
    *,
    completed_path: Path,
    lock_path: Path,
    expected_sha256: str,
) -> Path:
    """Find the immutable source packet authenticated by the lock."""

    canonical = lock_path.with_name("manual_packet.csv")
    if canonical.exists():
        if _file_sha256(canonical) != expected_sha256:
            raise ValueError("frozen manual packet hash differs from its lock")
        return canonical

    matches = [
        candidate
        for candidate in lock_path.parent.glob("*.csv")
        if candidate.resolve() != completed_path.resolve()
        and _file_sha256(candidate) == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError("cannot resolve exactly one frozen manual packet from its lock")
    return matches[0]


def _normalized_reviewer_id(value: str) -> str:
    return value.strip().casefold()


def build_manual_packet(
    rows: Sequence[Mapping[str, Any]],
    *,
    packet_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [row for row in rows if _manual_holdout_row(row)]
    if len(selected) != 128:
        raise ValueError(f"manual audit requires 128 rows, observed {len(selected)}")
    if len({str(row["sample_id"]) for row in selected}) != 128:
        raise ValueError("manual audit sample IDs are not unique")
    selected.sort(key=lambda row: sha256_text(f"v11-manual|{row['sample_id']}"))
    packet_rows = [
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
        for index, row in enumerate(selected, start=1)
    ]
    columns = list(PACKET_COLUMNS)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = packet_path.with_suffix(packet_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(packet_rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, packet_path)
    packet_sha256 = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    lock = {
        "schema_version": "contrastive_prompts_v3_manual_audit_lock_v1",
        "packet_sha256": packet_sha256,
        "row_count": len(packet_rows),
        "expected": {
            str(row["sample_id"]): {
                "split": row["split"],
                "acceptable_labels": list(row["acceptable_judge_labels"]),
                "prompt_sha256": sha256_text(
                    f"{row['system_prompt']}\n<USER>\n{row['user_prompt']}"
                ),
            }
            for row in selected
        },
    }
    _atomic_text(lock_path, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    packet_manifest = {
        "schema_version": "contrastive_prompts_v3_manual_packet_v1",
        "row_count": len(packet_rows),
        "final_counterfactual_rows": sum(
            row["split"] == "final_counterfactual" for row in packet_rows
        ),
        "factorial_calibration_rows": sum(
            row["split"] == "factorial_calibration" for row in packet_rows
        ),
        "final_neutral_rows": sum(
            row["split"] == "neutral_controls" for row in packet_rows
        ),
        "packet_sha256": packet_sha256,
        "labels_blinded": True,
    }
    return packet_manifest, lock


def _as_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"{field} must be yes/no")


def validate_completed_manual_audit(
    completed_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = lock["expected"]
    packet_path = _resolve_locked_packet(
        completed_path=completed_path,
        lock_path=lock_path,
        expected_sha256=str(lock["packet_sha256"]),
    )
    packet_rows = _read_packet_csv(packet_path)
    rows = _read_packet_csv(completed_path)
    if len(rows) != lock["row_count"]:
        raise ValueError("completed manual audit row count differs")
    if len(packet_rows) != lock["row_count"]:
        raise ValueError("frozen manual packet row count differs from its lock")
    by_id = {row["sample_id"]: row for row in rows}
    packet_by_id = {row["sample_id"]: row for row in packet_rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError("completed manual audit sample IDs differ")
    if len(packet_by_id) != len(packet_rows) or set(packet_by_id) != set(expected):
        raise ValueError("frozen manual packet sample IDs differ from its lock")

    packet_indices: set[str] = set()
    for sample_id, packet_row in packet_by_id.items():
        expected_row = expected[sample_id]
        if packet_row["split"] != expected_row["split"]:
            raise ValueError(f"{sample_id}: frozen manual packet split differs from its lock")
        prompt_sha256 = sha256_text(
            f"{packet_row['system_prompt']}\n<USER>\n{packet_row['user_prompt']}"
        )
        if prompt_sha256 != expected_row["prompt_sha256"]:
            raise ValueError(f"{sample_id}: frozen manual packet prompt differs from its lock")
        packet_index = packet_row["packet_index"]
        if not packet_index.isdecimal() or not 1 <= int(packet_index) <= len(packet_rows):
            raise ValueError(f"{sample_id}: frozen manual packet index is invalid")
        packet_indices.add(packet_index)
    if len(packet_indices) != len(packet_rows):
        raise ValueError("frozen manual packet indices are not unique")

    for sample_id, row in by_id.items():
        packet_row = packet_by_id[sample_id]
        for field in IMMUTABLE_PACKET_FIELDS:
            if row[field] != packet_row[field]:
                raise ValueError(f"{sample_id}: completed {field} differs from frozen packet")

    failures: list[str] = []
    disagreements = 0
    reviewer_1_ids = {
        _normalized_reviewer_id(row["reviewer_1_id"])
        for row in rows
        if _normalized_reviewer_id(row["reviewer_1_id"])
    }
    reviewer_2_ids = {
        _normalized_reviewer_id(row["reviewer_2_id"])
        for row in rows
        if _normalized_reviewer_id(row["reviewer_2_id"])
    }
    if len(reviewer_1_ids) != 1:
        failures.append("reviewer_1_id must identify one consistent reviewer across all rows")
    if len(reviewer_2_ids) != 1:
        failures.append("reviewer_2_id must identify one consistent reviewer across all rows")
    if len(reviewer_1_ids | reviewer_2_ids) != 2 or reviewer_1_ids & reviewer_2_ids:
        failures.append("exactly two distinct reviewer IDs are required across the packet")
    for sample_id, row in by_id.items():
        reviewer_1 = _normalized_reviewer_id(row["reviewer_1_id"])
        reviewer_2 = _normalized_reviewer_id(row["reviewer_2_id"])
        if not reviewer_1 or not reviewer_2 or reviewer_1 == reviewer_2:
            failures.append(f"{sample_id}: two distinct reviewer IDs required")
            continue
        label_1 = row["reviewer_1_label"].strip().casefold()
        label_2 = row["reviewer_2_label"].strip().casefold()
        if label_1 not in VALID_LABELS or label_2 not in VALID_LABELS:
            failures.append(f"{sample_id}: reviewer label invalid")
            continue
        try:
            coherence = (
                _as_bool(row["reviewer_1_facts_coherent"], field="reviewer_1_facts_coherent")
                and _as_bool(row["reviewer_2_facts_coherent"], field="reviewer_2_facts_coherent")
            )
            direct_absent = (
                _as_bool(
                    row["reviewer_1_direct_label_absent"],
                    field="reviewer_1_direct_label_absent",
                )
                and _as_bool(
                    row["reviewer_2_direct_label_absent"],
                    field="reviewer_2_direct_label_absent",
                )
            )
        except ValueError as exc:
            failures.append(f"{sample_id}: {exc}")
            continue
        if not coherence:
            failures.append(f"{sample_id}: facts not coherent")
        if not direct_absent:
            failures.append(f"{sample_id}: direct label statement detected")
        if label_1 != label_2:
            disagreements += 1
            final_label = row["adjudicated_label"].strip().casefold()
            if final_label not in VALID_LABELS or not row["adjudication_notes"].strip():
                failures.append(f"{sample_id}: disagreement lacks adjudication")
                continue
        else:
            final_label = label_1
        if final_label not in set(expected[sample_id]["acceptable_labels"]):
            failures.append(f"{sample_id}: final manual label disagrees with frozen contract")
    return {
        "schema_version": "contrastive_prompts_v3_manual_audit_v1",
        "passed": not failures,
        "row_count": len(rows),
        "reviewed_fraction": len(rows) / lock["row_count"],
        "reviewer_disagreement_count": disagreements,
        "reviewer_ids": sorted(reviewer_1_ids | reviewer_2_ids),
        "reviewer_roles": {
            "reviewer_1_id": next(iter(reviewer_1_ids)) if len(reviewer_1_ids) == 1 else None,
            "reviewer_2_id": next(iter(reviewer_2_ids)) if len(reviewer_2_ids) == 1 else None,
        },
        "failure_count": len(failures),
        "failures": failures,
        "completed_sha256": hashlib.sha256(completed_path.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    }


__all__ = ["build_manual_packet", "validate_completed_manual_audit"]
