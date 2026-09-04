"""Independent recomputation for V19 result tables.

This module imports no V19 analysis or decision code.
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


def _interaction(rows: list[dict[str, Any]], *, arm: str | None, index_key: str) -> float:
    selected = rows if arm is None else [row for row in rows if row["arm"] == arm]
    tasks = sorted({row["stimulus_id"] for row in selected})
    matrices: dict[str, np.ndarray] = {}
    for group in GROUPS:
        group_rows = [row for row in selected if row["group"] == group]
        identities = sorted({int(row[index_key]) for row in group_rows})
        lookup = {
            (int(row[index_key]), row["stimulus_id"]): float(row["expected_folded_confidence"])
            for row in group_rows
        }
        matrices[group] = np.asarray(
            [[lookup.get((identity, task), np.nan) for task in tasks] for identity in identities]
        )
    center = np.nanmean(matrices["genpop"], axis=0)
    means = {
        group: float(np.mean(np.nanmean(matrix - center[None, :], axis=1)))
        for group, matrix in matrices.items()
    }
    return means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]


def verify_positive_control(root: Path) -> dict[str, Any]:
    rows = _jsonl(root / "positive_control_rows.jsonl")
    analysis = _json(root / "analysis.json")
    baseline = [row for row in rows if row["arm"] == "baseline"]
    lookup = {(row["sample_id"], row["codebook_id"]): row for row in baseline}
    checks = {
        "row_count": len(rows) == 320,
        "unique_keys": len(rows)
        == len({(row["sample_id"], row["codebook_id"], row["arm"]) for row in rows}),
        "baseline_accuracy": abs(float(np.mean([row["correct"] for row in baseline])) - analysis["baseline_accuracy"]) <= 1e-12,
    }
    for name, record in analysis["configurations"].items():
        deploy = [row for row in rows if row["arm"] == f"{name}_deploy"]
        reverse = [row for row in rows if row["arm"] == f"{name}_eval"]
        deploy_point = float(np.mean([row["eval_logodds"] - lookup[(row["sample_id"], row["codebook_id"])]["eval_logodds"] for row in deploy]))
        reverse_point = float(np.mean([row["eval_logodds"] - lookup[(row["sample_id"], row["codebook_id"])]["eval_logodds"] for row in reverse]))
        checks[f"{name}_deploy_point"] = abs(deploy_point - record["deploy_delta"]) <= 1e-12
        checks[f"{name}_eval_point"] = abs(reverse_point - record["eval_delta"]) <= 1e-12
    return {"schema_version": "glm53_v19_pc_verification_v1", "passed": all(checks.values()), "checks": checks}


def verify_local_parity(root: Path) -> dict[str, Any]:
    rows = _jsonl(root / "raw_scores.jsonl")
    analysis = _json(root / "analysis.json")
    point = _interaction(rows, arm=None, index_key="analysis_index")
    checks = {
        "point": abs(point - analysis["interaction_pp"]) <= 1e-10,
        "rows": len(rows) == 1404,
        "keys_unique": len(rows) == len({(row["group"], row["persona_key"], row["stimulus_id"]) for row in rows}),
    }
    return {"schema_version": "glm53_v19_parity_verification_v1", "passed": all(checks.values()), "checks": checks, "interaction_pp": point}


def verify_causal(root: Path) -> dict[str, Any]:
    rows = _jsonl(root / "all_rows.jsonl")
    analysis = _json(root / "analysis.json")
    arms = sorted({row["arm"] for row in rows})
    points = {arm: _interaction(rows, arm=arm, index_key="stage_index") for arm in arms}
    baseline = points["baseline"]
    checks = {
        "rows": len(rows) == 7020,
        "arms": arms == sorted(["baseline", "all16_deploy", "all16_eval", "status_deploy", "formality"]),
        "baseline": abs(baseline - analysis["baseline_interaction_pp"]) <= 1e-10,
        "points": all(
            abs(points[arm] - analysis["arms"][arm]["interaction_pp"]) <= 1e-10
            and abs(points[arm] - baseline - analysis["arms"][arm]["delta_pp"]) <= 1e-10
            for arm in arms
            if arm != "baseline"
        ),
        "keys_unique": len(rows) == len({(row["sample_id"], row["arm"]) for row in rows}),
    }
    return {"schema_version": "glm53_v19_causal_verification_v1", "passed": all(checks.values()), "checks": checks}


__all__ = ["verify_causal", "verify_local_parity", "verify_positive_control"]
