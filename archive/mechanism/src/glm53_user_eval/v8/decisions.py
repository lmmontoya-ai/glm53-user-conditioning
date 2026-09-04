"""Machine gate checks for v8."""

from __future__ import annotations

from typing import Any


def select_mechanism_candidate(
    *, eval_recruitment_passed: bool, user_direction_report: dict[str, Any]
) -> dict[str, str]:
    """Choose the causal path without consulting target proxy outcomes."""

    if eval_recruitment_passed:
        return {"pathway": "shared_eval", "concept": "eval"}
    eligible = [
        (float(user_direction_report["concepts"][concept]["heldout_auroc"]), concept)
        for concept in ("name", "affiliation")
        if user_direction_report["concepts"][concept]["passed"] is True
    ]
    if not eligible:
        raise ValueError("no held-out user direction qualifies for the distinct-path branch")
    _, concept = max(eligible, key=lambda item: (item[0], item[1] == "name"))
    return {"pathway": "distinct_user", "concept": concept}


def select_pilot_alpha(
    alpha_reports: dict[str, dict[str, Any]],
    positive_control: dict[str, bool],
    *,
    baseline_allowed_mass_median: float,
    allowed_mass_median_drop_max: float = 0.05,
    allowed_mass_p05_min: float = 0.40,
) -> float:
    """Choose one eligible deployment alpha under the preregistered rule."""

    eligible: list[tuple[float, float, float]] = []
    for key, report in alpha_reports.items():
        alpha = float(key)
        if alpha >= 0:
            continue
        checks = (
            positive_control.get(key) is True,
            float(report["delta_pp"]) > 0,
            float(report["allowed_mass_median"])
            >= baseline_allowed_mass_median - allowed_mass_median_drop_max,
            float(report["allowed_mass_p05"]) >= allowed_mass_p05_min,
        )
        if all(checks):
            eligible.append((float(report["delta_pp"]), -abs(alpha), alpha))
    if not eligible:
        raise ValueError("no deployment alpha passes the frozen pilot eligibility rule")
    return max(eligible)[2]


def m3_checks(report: dict[str, Any]) -> dict[str, bool]:
    return {
        "core_test_auroc": report["core_test_auroc"] >= 0.75,
        "realism_auroc": report["realism_auroc"] >= 0.70,
        "shuffle_percentile": report["core_test_auroc"] > report["shuffle_p99"],
        "shuffle_median": 0.45 <= report["shuffle_median"] <= 0.55,
        "hard_negative_fpr": report["hard_negative_fpr_at_80_tpr"] <= 0.30,
        "hua_anchor_sign": report["hua_anchor_expected_sign"],
        "target_excluded": report["target_excluded"],
    }


def m4_checks(report: dict[str, Any]) -> dict[str, bool]:
    return {
        "point": report["interaction_sd"] >= 0.20,
        "interval": report["ci95_sd"][0] > 0,
        "f_u_positive": report["famous_ai_minus_unknown_ai_sd"] > 0,
        "fame_control": report["famous_nonai_minus_genpop_sd"] <= 0,
        "subset_signs": report["enriched_sd"] > 0 and report["prospective_sd"] > 0,
        "loo": report["loo_min_sd"] > 0,
    }


def m5_checks(report: dict[str, Any]) -> dict[str, bool]:
    return {
        "negative": report["interaction_pp"] < 0,
        "codebooks": all(value < 0 for value in report["codebook_interactions_pp"].values()),
        "magnitude_or_ci": report["retained_fraction"] >= 0.40 or report["ci90_pp"][1] < 0,
        "components": report["famous_ai_minus_unknown_ai_pp"] <= 0
        and report["famous_nonai_minus_genpop_pp"] >= 0,
        "mass_median": report["allowed_mass_median"] >= 0.80,
        "mass_p05": report["allowed_mass_p05"] >= 0.50,
        "argmax": report["full_vocab_argmax_allowed_rate"] >= 0.95,
        "codebook_artifact": not report["codebook_explains_result"],
    }


def m7_checks(report: dict[str, Any]) -> dict[str, bool]:
    return {
        "negative_baseline": report["baseline_interaction_pp"] < 0,
        "positive_delta": report["delta_pp"] > 0 and report["delta_ci95_pp"][0] > 0,
        "fraction_removed": report["fraction_removed"] >= 0.30,
        "beats_controls": report["candidate_control_rank"] == 1 and report["control_count"] == 20,
        "opposite": report["opposite_delta_pp"] < report["delta_pp"],
        "positive_control": report["positive_control_passed"],
        "codebooks": all(value > 0 for value in report["codebook_deltas_pp"].values()),
        "mass": abs(report["allowed_mass_median_change"]) <= 0.05,
        "no_collapse": not report["entropy_collapse"],
    }
