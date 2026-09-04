"""Independent deterministic checks for V17 result tables.

This module intentionally imports no V17 analysis or decision code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

GROUPS = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_positive_control(root: Path) -> dict[str, Any]:
    rows = _jsonl(root / "positive_control_rows.jsonl")
    analysis = _json(root / "analysis.json")
    baseline = [row for row in rows if row["arm"] == "baseline"]
    accuracy = float(np.mean([row["correct"] for row in baseline]))
    lookup = {(row["sample_id"], row["codebook_id"]): row for row in baseline}
    differences: dict[str, float] = {}
    for arm in sorted({row["arm"] for row in rows} - {"baseline"}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        differences[arm] = float(
            np.mean(
                [
                    row["eval_logodds"]
                    - lookup[(row["sample_id"], row["codebook_id"])]["eval_logodds"]
                    for row in arm_rows
                ]
            )
        )
    checks = {
        "baseline_accuracy": abs(accuracy - analysis["baseline_accuracy"]) <= 1e-12,
        "all_deploy_points": all(
            abs(value - analysis["configurations"][arm.removesuffix("_deploy")]["deploy_delta"]) <= 1e-12
            for arm, value in differences.items()
            if arm.endswith("_deploy")
        ),
        "all_eval_points": all(
            abs(value - analysis["configurations"][arm.removesuffix("_eval")]["eval_delta"]) <= 1e-12
            for arm, value in differences.items()
            if arm.endswith("_eval")
        ),
        "row_keys_unique": len(rows)
        == len({(row["sample_id"], row["codebook_id"], row["arm"]) for row in rows}),
    }
    return {"schema_version": "glm53_v17_pc_verification_v1", "passed": all(checks.values()), "checks": checks}


def _interaction(rows: list[dict[str, Any]], arm: str | None = None) -> float:
    selected = rows if arm is None else [row for row in rows if row["arm"] == arm]
    task_ids = sorted({row["stimulus_id"] for row in selected})
    matrices: dict[str, np.ndarray] = {}
    index_key = "stage_index" if arm is not None else "analysis_index"
    for group in GROUPS:
        group_rows = [row for row in selected if row["group"] == group]
        identities = sorted({int(row[index_key]) for row in group_rows})
        table = {(int(row[index_key]), row["stimulus_id"]): row["expected_folded_confidence"] for row in group_rows}
        matrices[group] = np.asarray([[table.get((identity, task), np.nan) for task in task_ids] for identity in identities])
    center = np.nanmean(matrices["genpop"], axis=0)
    means = {
        group: float(np.mean(np.nanmean(matrix - center[None, :], axis=1)))
        for group, matrix in matrices.items()
    }
    return means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]


def verify_local_parity(root: Path) -> dict[str, Any]:
    rows = _jsonl(root / "raw_scores.jsonl")
    analysis = _json(root / "analysis.json")
    point = _interaction(rows)
    checks = {
        "point": abs(point - analysis["interaction_pp"]) <= 1e-10,
        "keys_unique": len(rows) == len({(row["group"], row["persona_key"], row["stimulus_id"]) for row in rows}),
        "rows": len(rows) == 6387,
    }
    return {"schema_version": "glm53_v17_parity_verification_v1", "passed": all(checks.values()), "checks": checks, "interaction_pp": point}


def verify_causal(root: Path) -> dict[str, Any]:
    rows = _jsonl(root / "all_rows.jsonl")
    analysis = _json(root / "analysis.json")
    arms = sorted({row["arm"] for row in rows})
    points = {arm: _interaction(rows, arm) for arm in arms}
    baseline = points["baseline"]
    checks = {
        "baseline": abs(baseline - analysis["baseline_interaction_pp"]) <= 1e-10,
        "arms": all(
            abs(points[arm] - analysis["arms"][arm]["interaction_pp"]) <= 1e-10
            and abs((points[arm] - baseline) - analysis["arms"][arm]["delta_pp"]) <= 1e-10
            for arm in arms
            if arm != "baseline"
        ),
        "keys_unique": len(rows)
        == len({(row["sample_id"], row["arm"]) for row in rows}),
    }
    return {"schema_version": "glm53_v17_causal_verification_v1", "passed": all(checks.values()), "checks": checks}


__all__ = ["verify_causal", "verify_local_parity", "verify_positive_control"]
