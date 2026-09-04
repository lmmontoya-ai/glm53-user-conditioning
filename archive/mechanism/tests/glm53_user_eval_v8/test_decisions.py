from __future__ import annotations

import pytest
from src.glm53_user_eval.v8.decisions import (
    m3_checks,
    m4_checks,
    m5_checks,
    m7_checks,
    select_mechanism_candidate,
    select_pilot_alpha,
)
from src.glm53_user_eval.v8.science import write_decision


def test_m3_pass() -> None:
    checks = m3_checks(
        {
            "core_test_auroc": 0.8,
            "realism_auroc": 0.75,
            "shuffle_p99": 0.6,
            "shuffle_median": 0.5,
            "hard_negative_fpr_at_80_tpr": 0.2,
            "hua_anchor_expected_sign": True,
            "target_excluded": True,
        }
    )
    assert all(checks.values())


def test_m3_boundary_fails() -> None:
    checks = m3_checks(
        {
            "core_test_auroc": 0.74,
            "realism_auroc": 0.75,
            "shuffle_p99": 0.6,
            "shuffle_median": 0.5,
            "hard_negative_fpr_at_80_tpr": 0.2,
            "hua_anchor_expected_sign": True,
            "target_excluded": True,
        }
    )
    assert not checks["core_test_auroc"]


def test_m4_pass() -> None:
    assert all(
        m4_checks(
            {
                "interaction_sd": 0.3,
                "ci95_sd": [0.1, 0.5],
                "famous_ai_minus_unknown_ai_sd": 0.2,
                "famous_nonai_minus_genpop_sd": -0.1,
                "enriched_sd": 0.2,
                "prospective_sd": 0.1,
                "loo_min_sd": 0.01,
            }
        ).values()
    )


def test_m5_pass() -> None:
    report = {
        "interaction_pp": -0.5,
        "codebook_interactions_pp": {"0": -0.4, "1": -0.6},
        "retained_fraction": 0.5,
        "ci90_pp": [-0.8, -0.1],
        "famous_ai_minus_unknown_ai_pp": -0.2,
        "famous_nonai_minus_genpop_pp": 0.3,
        "allowed_mass_median": 0.9,
        "allowed_mass_p05": 0.6,
        "full_vocab_argmax_allowed_rate": 0.98,
        "codebook_explains_result": False,
    }
    assert all(m5_checks(report).values())


def test_m7_pass() -> None:
    report = {
        "baseline_interaction_pp": -0.5,
        "delta_pp": 0.3,
        "delta_ci95_pp": [0.1, 0.5],
        "fraction_removed": 0.6,
        "candidate_control_rank": 1,
        "control_count": 20,
        "opposite_delta_pp": -0.1,
        "positive_control_passed": True,
        "codebook_deltas_pp": {"0": 0.2, "1": 0.3},
        "allowed_mass_median_change": -0.01,
        "entropy_collapse": False,
    }
    assert all(m7_checks(report).values())


def test_m7_random_failure() -> None:
    report = {
        "baseline_interaction_pp": -0.5,
        "delta_pp": 0.3,
        "delta_ci95_pp": [0.1, 0.5],
        "fraction_removed": 0.6,
        "candidate_control_rank": 2,
        "control_count": 20,
        "opposite_delta_pp": -0.1,
        "positive_control_passed": True,
        "codebook_deltas_pp": {"0": 0.2, "1": 0.3},
        "allowed_mass_median_change": -0.01,
        "entropy_collapse": False,
    }
    assert not m7_checks(report)["beats_controls"]


def _alpha_report(delta: float, mass: float = 0.90, p05: float = 0.70) -> dict:
    return {
        "delta_pp": delta,
        "allowed_mass_median": mass,
        "allowed_mass_p05": p05,
    }


def test_select_pilot_alpha_uses_largest_eligible_delta() -> None:
    selected = select_pilot_alpha(
        {"-1.0": _alpha_report(0.4), "-0.5": _alpha_report(0.2)},
        {"-1.0": True, "-0.5": True},
        baseline_allowed_mass_median=0.91,
    )
    assert selected == -1.0


def test_select_pilot_alpha_prefers_smaller_magnitude_on_tie() -> None:
    selected = select_pilot_alpha(
        {"-1.0": _alpha_report(0.4), "-0.5": _alpha_report(0.4)},
        {"-1.0": True, "-0.5": True},
        baseline_allowed_mass_median=0.91,
    )
    assert selected == -0.5


@pytest.mark.parametrize(
    ("reports", "controls"),
    [
        ({"-1.0": _alpha_report(0.4)}, {"-1.0": False}),
        ({"-1.0": _alpha_report(-0.1)}, {"-1.0": True}),
        ({"-1.0": _alpha_report(0.4, mass=0.80)}, {"-1.0": True}),
        ({"-1.0": _alpha_report(0.4, p05=0.30)}, {"-1.0": True}),
    ],
)
def test_select_pilot_alpha_fails_closed(reports: dict, controls: dict) -> None:
    with pytest.raises(ValueError, match="no deployment alpha"):
        select_pilot_alpha(
            reports,
            controls,
            baseline_allowed_mass_median=0.90,
        )


def test_write_decision_requires_hashed_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires hashed inputs"):
        write_decision("M6", {"x": True}, {}, tmp_path / "decision.json", inputs={})


def test_write_decision_preserves_input_lineage(tmp_path) -> None:
    inputs = {"source": {"path": "source", "sha256": "a" * 64, "size_bytes": 1}}
    result = write_decision(
        "M6",
        {"x": True},
        {"estimate": 1},
        tmp_path / "decision.json",
        inputs=inputs,
    )
    assert result["passed"] is True
    assert result["inputs"] == inputs


def test_m7_fails_when_measured_positive_control_fails() -> None:
    report = {
        "baseline_interaction_pp": -0.5,
        "delta_pp": 0.3,
        "delta_ci95_pp": [0.1, 0.5],
        "fraction_removed": 0.6,
        "candidate_control_rank": 1,
        "control_count": 20,
        "opposite_delta_pp": -0.1,
        "positive_control_passed": False,
        "codebook_deltas_pp": {"0": 0.2, "1": 0.3},
        "allowed_mass_median_change": -0.01,
        "entropy_collapse": False,
    }
    assert not m7_checks(report)["positive_control"]


def test_mechanism_candidate_prefers_shared_eval_when_recruited() -> None:
    result = select_mechanism_candidate(
        eval_recruitment_passed=True,
        user_direction_report={"concepts": {}},
    )
    assert result == {"pathway": "shared_eval", "concept": "eval"}


def test_mechanism_candidate_uses_best_heldout_distinct_direction() -> None:
    report = {
        "concepts": {
            "name": {"heldout_auroc": 0.74, "passed": True},
            "affiliation": {"heldout_auroc": 0.78, "passed": True},
        }
    }
    result = select_mechanism_candidate(
        eval_recruitment_passed=False,
        user_direction_report=report,
    )
    assert result == {"pathway": "distinct_user", "concept": "affiliation"}


def test_mechanism_candidate_fails_without_valid_direction() -> None:
    report = {
        "concepts": {
            "name": {"heldout_auroc": 0.60, "passed": False},
            "affiliation": {"heldout_auroc": 0.62, "passed": False},
        }
    }
    with pytest.raises(ValueError, match="no held-out user direction"):
        select_mechanism_candidate(
            eval_recruitment_passed=False,
            user_direction_report=report,
        )
