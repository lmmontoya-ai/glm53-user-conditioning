"""Select a fixed, balanced transcript sample for manual roster audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = (
    "famous_coherent",
    "unknown_same_org",
    "unknown_general",
    "famous_nonai_control",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_sample(
    schedule: list[dict[str, Any]], *, seed: int, sample_size: int
) -> list[dict[str, Any]]:
    if sample_size % (len(CONDITIONS) * 5) != 0:
        raise ValueError("sample size must be divisible by 20 condition-by-block strata")
    per_stratum = sample_size // (len(CONDITIONS) * 5)
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in schedule:
        strata[(str(row["condition"]), int(row["analysis_block"]))].append(row)
    expected = {(condition, block) for condition in CONDITIONS for block in range(5)}
    if set(strata) != expected:
        raise ValueError("schedule does not contain all condition-by-block strata")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        population = sorted(strata[key], key=lambda row: str(row["sample_id"]))
        if len(population) < per_stratum:
            raise ValueError(f"stratum {key} is too small")
        selected.extend(rng.sample(population, per_stratum))
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def build_packet(
    *,
    results_path: Path,
    schedule_path: Path,
    output_csv: Path,
    output_jsonl: Path,
    manifest_path: Path,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    results = _read_jsonl(results_path)
    schedule = _read_jsonl(schedule_path)
    result_by_id = {str(row["sample_id"]): row for row in results}
    if len(result_by_id) != len(results):
        raise ValueError("results contain duplicate sample IDs")
    schedule_by_id = {str(row["sample_id"]): row for row in schedule}
    if len(schedule_by_id) != len(schedule):
        raise ValueError("schedule contains duplicate sample IDs")
    if set(result_by_id) != set(schedule_by_id):
        raise ValueError("audit packet requires the complete schedule and results")

    selected = select_sample(schedule, seed=seed, sample_size=sample_size)
    packet_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    for schedule_row in selected:
        sample_id = str(schedule_row["sample_id"])
        result = result_by_id[sample_id]
        packet_rows.append(
            {
                "sample_id": sample_id,
                "pair_index": int(schedule_row["pair_index"]),
                "condition": schedule_row["condition"],
                "analysis_block": int(schedule_row["analysis_block"]),
                "scenario_id": schedule_row["scenario_id"],
                "binary_answer": result.get("binary_answer"),
                "confidence_p": result.get("confidence_p"),
                "parse_valid": result.get("parse_valid"),
                "subject_response_main": result.get("subject_response_main"),
                "subject_response_followup": result.get("subject_response_followup"),
                "judge_text": result.get("judge_text"),
            }
        )
        log_rows.append(
            {
                "sample_id": sample_id,
                "condition": schedule_row["condition"],
                "analysis_block": int(schedule_row["analysis_block"]),
                "reviewed": "false",
                "binary_parse_correct": "",
                "confidence_parse_correct": "",
                "on_task": "",
                "notes": "",
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in packet_rows),
        encoding="utf-8",
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    manifest = {
        "schema_version": "glm53_roster_manual_audit_packet_v1",
        "seed": seed,
        "sample_size": sample_size,
        "selection": "equal_count_per_condition_by_analysis_block",
        "results_sha256": _sha256(results_path),
        "schedule_sha256": _sha256(schedule_path),
        "packet_sha256": _sha256(output_jsonl),
        "reading_log_sha256_before_review": _sha256(output_csv),
        "condition_counts": {
            condition: sum(row["condition"] == condition for row in packet_rows)
            for condition in CONDITIONS
        },
        "block_counts": {
            str(block): sum(row["analysis_block"] == block for row in packet_rows)
            for block in range(5)
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    args = parser.parse_args()
    manifest = build_packet(
        results_path=args.results,
        schedule_path=args.schedule,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        manifest_path=args.manifest,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
