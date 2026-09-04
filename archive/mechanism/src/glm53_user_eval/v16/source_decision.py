"""Fail-closed source decision for V16."""

from __future__ import annotations

from typing import Any


def decide_source(
    analysis: dict[str, Any], permutation: dict[str, Any]
) -> dict[str, Any]:
    logistic = analysis["models"]["logistic"]
    agreement = analysis["direction_agreement"]
    factorial_checks = {
        f"{name}_{check}": bool(value)
        for name in ("logistic", "paired_mean")
        for check, value in analysis["models"][name]["factorial_calibration"]["checks"].items()
    }
    control_checks = {
        f"logistic_fresh_{check}": bool(value)
        for check, value in analysis["models"]["logistic"]["fresh_controls"]["checks"].items()
    }
    checks = {
        "ordinary_test_auroc_ge_080": logistic["ordinary_test"]["auroc"] >= 0.80,
        "final_counterfactual_auroc_ge_075": logistic["final_counterfactual"]["auroc"] >= 0.75,
        "final_counterfactual_fpr80_le_025": logistic["final_counterfactual"][
            "fpr_at_80_tpr"
        ]
        <= 0.25,
        "final_counterfactual_gap_positive": logistic["final_counterfactual"]["score_gap"] > 0,
        "direction_cosine_ge_050": agreement["raw_cosine"] >= 0.50,
        "final_score_spearman_ge_060": agreement["final_score_spearman"] >= 0.60,
        "ridge_gap_positive": agreement["logistic_score_gap"] > 0,
        "paired_gap_positive": agreement["paired_mean_score_gap"] > 0,
        "permutation_p_lt_001": permutation["add_one_empirical_p"] < 0.01,
        "stability_fifth_percentile_gt_050": analysis["paired_direction_stability"][
            "fifth_percentile_cosine"
        ]
        > 0.50,
        "leave_one_generator_positive": bool(
            analysis["leave_one_training_generator_score_gaps"]
        )
        and all(value > 0 for value in analysis["leave_one_training_generator_score_gaps"].values()),
        "final_did_not_select": analysis["selection_used_final_rows"] is False,
        "factorial_did_not_select": analysis["selection_used_factorial_rows"] is False,
        "fresh_controls_did_not_select": analysis["selection_used_fresh_controls"] is False,
        **factorial_checks,
        **control_checks,
    }
    passed = all(checks.values())
    return {
        "schema_version": "glm53_v16_source_decision_v1",
        "project_id": "glm53_user_eval_source_activation_v16",
        "passed": passed,
        "decision": "source_instrument_valid_for_frozen_transfer" if passed else "stop_before_local_parity",
        "checks": checks,
        "authorization": {
            "local_proxy_parity": passed,
            "user_recruitment": False,
            "first_cot_transfer": False,
            "steering": False,
        },
    }


__all__ = ["decide_source"]
