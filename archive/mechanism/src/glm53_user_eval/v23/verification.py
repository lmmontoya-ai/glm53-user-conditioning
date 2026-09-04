from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_json, sha256_file


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _point(rows: list[dict[str, Any]], field: str, *, binary_choice: bool = False) -> float:
    groups = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")

    def valid(row: dict[str, Any]) -> bool:
        return row[field] in {"yes", "no"} if binary_choice else row[field] is not None

    task_center: dict[int, float] = {}
    for task in range(100):
        current = []
        for row in rows:
            if (
                row["group"] == "genpop"
                and valid(row)
                and int(str(row["stimulus_id"]).removeprefix("dd_")) == task
            ):
                current.append(
                    float(row[field] == "yes") * 100.0 if binary_choice else float(row[field])
                )
        task_center[task] = float(np.mean(current))
    values: dict[str, dict[int, list[float]]] = {
        group: {index: [] for index in range(70)} for group in groups
    }
    for row in rows:
        if row["group"] not in values or not valid(row):
            continue
        value = float(row[field] == "yes") * 100.0 if binary_choice else float(row[field])
        task = int(str(row["stimulus_id"]).removeprefix("dd_"))
        values[row["group"]][int(row["identity_index"])].append(value - task_center[task])
    means = {
        group: float(np.mean([np.mean(items) for items in people.values() if items]))
        for group, people in values.items()
    }
    return means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]


def verify(
    *, transcript_path: Path, deterministic_path: Path, annotation_path: Path, output_path: Path
) -> dict[str, Any]:
    rows = _rows(transcript_path)
    deterministic = json.loads(deterministic_path.read_text(encoding="utf-8"))
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    choice = _point(rows, "first_turn_choice", binary_choice=True)
    folded = _point(rows, "folded_confidence")
    failures = []
    if abs(choice - deterministic["first_turn_choice_interaction_pp"]["interaction"]) > 1e-10:
        failures.append("choice_point")
    if abs(folded - deterministic["folded_confidence_interaction_pp"]["interaction"]) > 1e-10:
        failures.append("folded_point")
    if annotation["judged_rows"] != 2000 or annotation["judge_records"] != 4000:
        failures.append("judge_counts")
    if annotation.get("human_review_complete") is not False:
        failures.append("human_review_claim")
    result = {
        "schema_version": "glm53_v23_independent_verification_v1",
        "passed": not failures,
        "failures": failures,
        "recomputed": {
            "choice_interaction_pp": choice,
            "folded_confidence_interaction_pp": folded,
        },
        "inputs": {
            "transcripts": sha256_file(transcript_path),
            "deterministic": sha256_file(deterministic_path),
            "annotation": sha256_file(annotation_path),
        },
        "primary_analysis_imported": False,
    }
    atomic_json(output_path, result)
    if failures:
        raise ValueError(f"independent verification failed: {failures}")
    return result
