import pytest

from src.glm53_user_eval.audits import projected_budget_ok, reject_secret_text
from src.glm53_user_eval.decisions import build_g0_decision, build_g2_decision, build_g3_decision
from src.glm53_user_eval.statistics import (
    empirical_random_p,
    mean_paired_effect,
    percentile_interval,
    reduction_fraction,
)


HASH = "a" * 64
REVISION = "b" * 40


def test_paired_effect_matches_hand_calculation() -> None:
    assert mean_paired_effect([3.0, 5.0], [1.0, 2.0]) == 2.5


def test_percentile_interval() -> None:
    low, high = percentile_interval(list(range(101)), 0.90)
    assert low == pytest.approx(5.0)
    assert high == pytest.approx(95.0)


def test_empirical_random_p_has_plus_one_correction() -> None:
    assert empirical_random_p(3.0, [1.0, 2.0, 4.0]) == 0.5


def test_reduction_fraction_rejects_zero_baseline() -> None:
    with pytest.raises(ValueError):
        reduction_fraction(0.0, 0.0)


def test_budget_fails_above_hard_cap() -> None:
    assert projected_budget_ok(100.0, 25.0, 125.0)
    assert not projected_budget_ok(100.0, 25.01, 125.0)


def test_secret_marker_is_rejected() -> None:
    with pytest.raises(ValueError):
        reject_secret_text("ZAI_API_KEY=secret")


def _g0_checks(value: bool = True) -> dict[str, bool]:
    return {
        "source_locks_complete": value,
        "roster_counts_exact": value,
        "twin_mapping_exact": value,
        "prompt_parity_exact": value,
        "glm52_cache_reproduced": value,
        "parser_fixtures_passed": value,
        "selection_frozen": value,
        "task_splits_frozen": value,
        "prereg_committed_and_tagged": value,
        "budget_within_cap": value,
    }


def test_g0_passes_only_with_complete_checks() -> None:
    decision = build_g0_decision(
        run_id="g0",
        prereg_sha256=HASH,
        source_lock_sha256=HASH,
        model_revision=REVISION,
        checks=_g0_checks(),
        estimates={},
        inputs=(),
    )
    assert decision.passed
    assert decision.decision == "proceed_to_g2_local_runtime"


def test_g0_fails_closed_on_missing_check() -> None:
    checks = _g0_checks()
    del checks["budget_within_cap"]
    with pytest.raises(ValueError, match="missing"):
        build_g0_decision(
            run_id="g0",
            prereg_sha256=HASH,
            source_lock_sha256=HASH,
            model_revision=REVISION,
            checks=checks,
            estimates={},
            inputs=(),
        )


def _g2_checks(value: bool = True) -> dict[str, bool]:
    return {
        "official_revision_loaded": value,
        "all_weight_shards_verified": value,
        "transformers_commit_exact": value,
        "twenty_prompt_forwards": value,
        "mhc_shape_contract": value,
        "hyper_head_mean_exact": value,
        "prompt_vectors_extracted": value,
        "alpha_zero_logits_exact": value,
        "alpha_zero_generation_exact": value,
        "additive_hook_local": value,
        "hooks_removed": value,
        "deadline_respected": value,
    }


def test_g2_passes_only_with_complete_runtime_contract() -> None:
    decision = build_g2_decision(
        run_id="g2",
        prereg_sha256=HASH,
        source_lock_sha256=HASH,
        model_revision=REVISION,
        runtime_hash=HASH,
        estimates={},
        checks=_g2_checks(),
        inputs=("stage.json", "doctor.json"),
    )
    assert decision.passed
    assert decision.decision == "proceed_to_g3_local_parity"


def test_g2_fails_closed_on_missing_check() -> None:
    checks = _g2_checks()
    del checks["deadline_respected"]
    with pytest.raises(ValueError, match="G2 checks differ"):
        build_g2_decision(
            run_id="g2",
            prereg_sha256=HASH,
            source_lock_sha256=HASH,
            model_revision=REVISION,
            runtime_hash=HASH,
            estimates={},
            checks=checks,
            inputs=(),
        )


def test_g3_requires_the_complete_local_behavior_contract() -> None:
    checks = {
        "all_600_rows_present": True,
        "parse_rate_at_least_95pct": True,
        "local_runtime_metadata_complete": True,
        "name_effect_negative_90ci": True,
        "at_least_3_of_4_negative": True,
        "clean_effect_at_least_1_5pp_95ci": True,
        "missingness_spread_at_most_2pp": True,
        "no_single_block_over_half": True,
        "manual_review_complete": True,
    }
    decision = build_g3_decision(
        run_id="g3",
        prereg_sha256=HASH,
        source_lock_sha256=HASH,
        model_revision=REVISION,
        runtime_hash=HASH,
        estimates={},
        checks=checks,
        inputs=(),
    )
    assert decision.passed
    assert decision.decision == "proceed_to_g4_directions"
