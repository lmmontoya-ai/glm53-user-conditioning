"""Fail-closed decision for V22 power planning."""

from __future__ import annotations

from typing import Any


def decide(report: dict[str, Any]) -> dict[str, Any]:
    target = float(report["target_power"])
    effect = float(report["smallest_meaningful_effect_pp"])
    key = f"{effect:.3f}"
    eligible = [
        row
        for row in report["candidate_results"]
        if float(row["power_by_effect"][key]) >= target
    ]
    selected = min(eligible, key=lambda row: int(row["dilemma_count"])) if eligible else None
    passed = selected is not None
    return {
        "schema_version": "glm53_v22_power_decision_v1",
        "project_id": report["project_id"],
        "passed": passed,
        "decision": (
            "power_gate_passed_design_selected"
            if passed
            else "stop_before_scientific_calls_insufficient_power"
        ),
        "target_power": target,
        "smallest_meaningful_effect_pp": effect,
        "selected_dilemma_count": None if selected is None else selected["dilemma_count"],
        "maximum_candidate_power": max(
            float(row["power_by_effect"][key]) for row in report["candidate_results"]
        ),
        "authorization": {
            "fresh_subject_calls": passed,
            "fresh_judge_calls": passed,
            "manipulation_check_calls": passed,
        },
        "stop_reason": (
            None
            if passed
            else "No candidate with at most 100 dilemmas reached 80% power for +0.325 pp."
        ),
    }

