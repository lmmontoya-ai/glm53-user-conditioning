"""Reconcile a results ledger after accidental concurrent writers.

The original ledger is preserved byte-for-byte. A sample is retained only when exactly
one ledger row equals the atomically saved raw-response result for that sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def reconcile(run_root: Path, *, incident_id: str) -> dict[str, Any]:
    results_path = run_root / "results.jsonl"
    incident_root = run_root / "incidents" / incident_id
    backup_path = incident_root / "results_original.jsonl"
    report_path = incident_root / "reconciliation_report.json"
    if backup_path.exists() or report_path.exists():
        raise FileExistsError(f"incident destination already exists: {incident_root}")

    source_lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in source_lines]
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["sample_id"])].append((index, row))

    selected: list[tuple[int, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    for sample_id, candidates in groups.items():
        raw_path = run_root / "calls" / sample_id / "raw_response.json"
        if not raw_path.exists():
            failures.append({"sample_id": sample_id, "reason": "raw_response_missing"})
            continue
        raw_result = json.loads(raw_path.read_text(encoding="utf-8")).get("result")
        matches = [(index, row) for index, row in candidates if row == raw_result]
        if len(matches) != 1:
            failures.append(
                {
                    "sample_id": sample_id,
                    "reason": "raw_match_count_not_one",
                    "candidate_count": len(candidates),
                    "raw_match_count": len(matches),
                }
            )
            continue
        selected.append(matches[0])
    if failures:
        raise RuntimeError(f"reconciliation failed closed for {len(failures)} samples")

    selected.sort(key=lambda item: item[0])
    incident_root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(results_path, backup_path)
    with backup_path.open("r+b") as handle:
        os.fsync(handle.fileno())
    source_sha256 = _sha256(results_path)
    backup_sha256 = _sha256(backup_path)
    if source_sha256 != backup_sha256:
        raise RuntimeError("original-ledger backup hash mismatch")

    fd, temporary = tempfile.mkstemp(
        prefix=".results.jsonl.", suffix=".tmp", dir=run_root
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for _index, row in selected:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, results_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    report = {
        "schema_version": "glm53_behavior_ledger_reconciliation_v1",
        "incident_id": incident_id,
        "selection_rule": "unique_ledger_row_equal_to_atomically_saved_raw_response_result",
        "effects_inspected": False,
        "original_row_count": len(rows),
        "canonical_row_count": len(selected),
        "duplicate_sample_id_count": sum(len(items) > 1 for items in groups.values()),
        "excluded_duplicate_row_count": len(rows) - len(selected),
        "original_results_sha256": source_sha256,
        "preserved_original_sha256": backup_sha256,
        "canonical_results_sha256": _sha256(results_path),
        "preserved_original_path": str(backup_path),
    }
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--incident-id", required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.run_root, incident_id=args.incident_id), indent=2))


if __name__ == "__main__":
    main()
