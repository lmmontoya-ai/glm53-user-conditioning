from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from src.glm53_user_eval.v22.decision import decide
from src.glm53_user_eval.v22.power import (
    crossed_bootstrap_draws,
    empirical_power,
    interaction,
    mde_for_power,
    read_prereg,
    repetitions_for_power,
    sha256_file,
    task_order,
    validate_parent_locks,
)

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/v22/configs/prereg_v22_information_substitution.yaml"


def test_parent_locks_and_schema() -> None:
    prereg = read_prereg(PREREG)
    hashes = validate_parent_locks(ROOT, prereg)
    assert prereg["status"]["scientific_calls_made"] == 0
    assert len(hashes) == 8


def test_task_order_is_deterministic_and_complete() -> None:
    stimuli = [f"dd_{index:04d}" for index in range(100)]
    first = task_order(stimuli, "locked")
    assert first == task_order(list(reversed(stimuli)), "locked")
    assert len(first) == len(set(first)) == 100


def test_interaction_known_fixture() -> None:
    matrices = {
        "genpop": np.full((70, 5), 70.0),
        "unknown_ai": np.full((70, 5), 70.0),
        "famous_ai": np.full((70, 5), 69.0),
        "famous_nonai": np.full((70, 5), 72.0),
    }
    assert interaction(matrices) == -3.0


def test_crossed_bootstrap_preserves_constant_interaction() -> None:
    residuals = {
        "genpop": np.zeros((70, 5)),
        "unknown_ai": np.zeros((70, 5)),
        "famous_ai": np.full((70, 5), 0.5),
        "famous_nonai": np.zeros((70, 5)),
    }
    draws = crossed_bootstrap_draws(residuals, reps=100, seed=1)
    assert np.allclose(draws, 0.5)


def test_empirical_power_and_mde_known_distribution() -> None:
    draws = np.linspace(-1.0, 1.0, 10001)
    assert empirical_power(draws, 0.0) < 0.03
    assert empirical_power(draws, 2.0) > 0.99
    assert mde_for_power(draws, 0.8) > 1.0
    assert repetitions_for_power(draws, 2.0, 0.8) == 1


def test_decision_fails_closed_when_no_candidate_passes() -> None:
    report = {
        "project_id": "fixture",
        "target_power": 0.8,
        "smallest_meaningful_effect_pp": 0.325,
        "candidate_results": [
            {"dilemma_count": 100, "power_by_effect": {"0.325": 0.5}}
        ],
    }
    result = decide(report)
    assert result["passed"] is False
    assert result["authorization"]["fresh_subject_calls"] is False


def test_decision_selects_smallest_passing_candidate() -> None:
    report = {
        "project_id": "fixture",
        "target_power": 0.8,
        "smallest_meaningful_effect_pp": 0.325,
        "candidate_results": [
            {"dilemma_count": 100, "power_by_effect": {"0.325": 0.9}},
            {"dilemma_count": 50, "power_by_effect": {"0.325": 0.81}},
        ],
    }
    result = decide(report)
    assert result["passed"] is True
    assert result["selected_dilemma_count"] == 50


def test_context_blocks_have_four_complete_families() -> None:
    path = ROOT / "pipelines/glm53_user_eval/v22/configs/context_blocks_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["families"]) == 4
    for family in payload["families"]:
        assert set(family) == {
            "family_id",
            "neutral_text",
            "operational_use",
            "model_assessment",
        }
        assert all(len(family[key].split()) >= 25 for key in family if key != "family_id")


def test_no_automated_human_review_claim() -> None:
    prereg = read_prereg(PREREG)
    review = prereg["human_and_ai_review"]
    assert review["luis_personal_review_not_claimed_by_automation"] is True
    assert review["ai_annotations_must_be_labeled_nonhuman"] is True


def test_independent_verifier_does_not_import_primary_modules() -> None:
    path = ROOT / "pipelines/glm53_user_eval/v22/verify_power_independent.py"
    source = path.read_text(encoding="utf-8")
    assert "src.glm53_user_eval.v22" not in source
    assert "from src" not in source


def test_final_evidence_hashes_resolve() -> None:
    evidence = json.loads(
        (ROOT / "artifacts/glm53_user_eval/v22/final_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["new_calls"] == {
        "subject": 0,
        "judge": 0,
        "manipulation_check": 0,
    }
    assert evidence["incremental_compute_cost_usd"] == 0.0
    for item in evidence["artifacts"].values():
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
