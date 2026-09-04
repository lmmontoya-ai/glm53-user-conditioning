"""Fail-closed gate decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schemas import GateDecision


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_g0_decision(
    *,
    run_id: str,
    prereg_sha256: str,
    source_lock_sha256: str,
    model_revision: str,
    checks: dict[str, bool],
    estimates: dict[str, Any],
    inputs: tuple[str, ...],
) -> GateDecision:
    required = {
        "source_locks_complete",
        "roster_counts_exact",
        "twin_mapping_exact",
        "prompt_parity_exact",
        "glm52_cache_reproduced",
        "parser_fixtures_passed",
        "selection_frozen",
        "task_splits_frozen",
        "prereg_committed_and_tagged",
        "budget_within_cap",
    }
    if set(checks) != required:
        missing = sorted(required - set(checks))
        extra = sorted(set(checks) - required)
        raise ValueError(f"G0 checks differ from preregistration; missing={missing}, extra={extra}")
    passed = all(checks.values())
    failed = sorted(key for key, value in checks.items() if not value)
    return GateDecision(
        project_id="glm53_user_eval_shared_direction_v1",
        gate="G0",
        run_id=run_id,
        prereg_sha256=prereg_sha256,
        source_lock_sha256=source_lock_sha256,
        model_revision=model_revision,
        runtime_hash=None,
        inputs=inputs,
        estimates=estimates,
        checks=checks,
        passed=passed,
        decision="proceed_to_g2_local_runtime" if passed else "stop_before_model_calls",
        reason="all G0 checks passed" if passed else f"failed: {', '.join(failed)}",
        created_at=now_iso(),
    )


def build_g1_decision(
    *,
    run_id: str,
    prereg_sha256: str,
    source_lock_sha256: str,
    model_revision: str,
    estimates: dict[str, Any],
    checks: dict[str, bool],
    inputs: tuple[str, ...],
) -> GateDecision:
    required = {
        "parse_rate_at_least_97pct",
        "provider_metadata_complete",
        "enriched_effect_negative_90ci",
        "at_least_6_of_8_negative",
        "clean_effect_at_least_1_5pp_95ci",
        "fame_control_smaller",
        "missingness_spread_at_most_2pp",
        "manual_review_complete",
    }
    if set(checks) != required:
        raise ValueError("G1 checks differ from preregistration")
    passed = all(checks.values())
    failed = sorted(key for key, value in checks.items() if not value)
    return GateDecision(
        project_id="glm53_user_eval_shared_direction_v1",
        gate="G1",
        run_id=run_id,
        prereg_sha256=prereg_sha256,
        source_lock_sha256=source_lock_sha256,
        model_revision=model_revision,
        runtime_hash=None,
        inputs=inputs,
        estimates=estimates,
        checks=checks,
        passed=passed,
        decision="proceed_to_runpod_g2" if passed else "stop_whitebox_project",
        reason="all G1 checks passed" if passed else f"failed: {', '.join(failed)}",
        created_at=now_iso(),
    )


def build_g2_decision(
    *,
    run_id: str,
    prereg_sha256: str,
    source_lock_sha256: str,
    model_revision: str,
    runtime_hash: str,
    estimates: dict[str, Any],
    checks: dict[str, bool],
    inputs: tuple[str, ...],
) -> GateDecision:
    required = {
        "official_revision_loaded",
        "all_weight_shards_verified",
        "transformers_commit_exact",
        "twenty_prompt_forwards",
        "mhc_shape_contract",
        "hyper_head_mean_exact",
        "prompt_vectors_extracted",
        "alpha_zero_logits_exact",
        "alpha_zero_generation_exact",
        "additive_hook_local",
        "hooks_removed",
        "deadline_respected",
    }
    if set(checks) != required:
        missing = sorted(required - set(checks))
        extra = sorted(set(checks) - required)
        raise ValueError(f"G2 checks differ from preregistration; missing={missing}, extra={extra}")
    passed = all(checks.values())
    failed = sorted(key for key, value in checks.items() if not value)
    return GateDecision(
        project_id="glm53_user_eval_shared_direction_v1",
        gate="G2",
        run_id=run_id,
        prereg_sha256=prereg_sha256,
        source_lock_sha256=source_lock_sha256,
        model_revision=model_revision,
        runtime_hash=runtime_hash,
        inputs=inputs,
        estimates=estimates,
        checks=checks,
        passed=passed,
        decision="proceed_to_g3_local_parity" if passed else "stop_whitebox_runtime",
        reason="all G2 checks passed" if passed else f"failed: {', '.join(failed)}",
        created_at=now_iso(),
    )


def build_g3_decision(
    *,
    run_id: str,
    prereg_sha256: str,
    source_lock_sha256: str,
    model_revision: str,
    runtime_hash: str,
    estimates: dict[str, Any],
    checks: dict[str, bool],
    inputs: tuple[str, ...],
) -> GateDecision:
    required = {
        "all_600_rows_present",
        "parse_rate_at_least_95pct",
        "local_runtime_metadata_complete",
        "name_effect_negative_90ci",
        "at_least_3_of_4_negative",
        "clean_effect_at_least_1_5pp_95ci",
        "missingness_spread_at_most_2pp",
        "no_single_block_over_half",
        "manual_review_complete",
    }
    if set(checks) != required:
        missing = sorted(required - set(checks))
        extra = sorted(set(checks) - required)
        raise ValueError(f"G3 checks differ from preregistration; missing={missing}, extra={extra}")
    passed = all(checks.values())
    failed = sorted(key for key, value in checks.items() if not value)
    return GateDecision(
        project_id="glm53_user_eval_shared_direction_v1",
        gate="G3",
        run_id=run_id,
        prereg_sha256=prereg_sha256,
        source_lock_sha256=source_lock_sha256,
        model_revision=model_revision,
        runtime_hash=runtime_hash,
        inputs=inputs,
        estimates=estimates,
        checks=checks,
        passed=passed,
        decision="proceed_to_g4_directions" if passed else "stop_whitebox_mechanism_project",
        reason="all G3 checks passed" if passed else f"failed: {', '.join(failed)}",
        created_at=now_iso(),
    )


def build_g3_api_decision(
    *,
    run_id: str,
    prereg_sha256: str,
    source_lock_sha256: str,
    model_revision: str,
    estimates: dict[str, Any],
    checks: dict[str, bool],
    inputs: tuple[str, ...],
) -> GateDecision:
    required = {
        "all_600_rows_present",
        "parse_rate_at_least_95pct",
        "api_route_metadata_complete",
        "name_effect_negative_90ci",
        "at_least_3_of_4_negative",
        "clean_effect_at_least_1_5pp_95ci",
        "missingness_spread_at_most_2pp",
        "no_single_block_over_half",
        "manual_review_complete",
    }
    if set(checks) != required:
        missing = sorted(required - set(checks))
        extra = sorted(set(checks) - required)
        raise ValueError(f"G3 API checks differ from preregistration; missing={missing}, extra={extra}")
    passed = all(checks.values())
    failed = sorted(key for key, value in checks.items() if not value)
    return GateDecision(
        project_id="glm53_user_eval_shared_direction_v1",
        gate="G3-api",
        run_id=run_id,
        prereg_sha256=prereg_sha256,
        source_lock_sha256=source_lock_sha256,
        model_revision=model_revision,
        runtime_hash=None,
        inputs=inputs,
        estimates=estimates,
        checks=checks,
        passed=passed,
        decision=(
            "unlock_capped_runpod_serverless_cached_model_smoke"
            if passed
            else "stop_glm53_project_and_preserve_runpod_credit"
        ),
        reason="all G3 API checks passed" if passed else f"failed: {', '.join(failed)}",
        created_at=now_iso(),
    )


def build_roster_v5_decision(
    *,
    run_id: str,
    prereg_sha256: str,
    source_lock_sha256: str,
    model_revision: str,
    estimates: dict[str, Any],
    checks: dict[str, bool],
    decision: str,
    inputs: tuple[str, ...],
) -> GateDecision:
    required = {
        "discovery_integrity_passed",
        "confirmation_integrity_passed",
        "roster_effect_positive",
        "identity_specific_effect_positive",
        "affiliation_effect_positive",
        "clean_null_established",
    }
    if set(checks) != required:
        missing = sorted(required - set(checks))
        extra = sorted(set(checks) - required)
        raise ValueError(
            f"v5 roster checks differ from preregistration; missing={missing}, extra={extra}"
        )
    if not checks["discovery_integrity_passed"] or not checks["confirmation_integrity_passed"]:
        raise ValueError("v5 decision cannot be built from a stage with failed integrity checks")
    positive = any(
        checks[key]
        for key in (
            "roster_effect_positive",
            "identity_specific_effect_positive",
            "affiliation_effect_positive",
        )
    )
    if positive and checks["clean_null_established"]:
        raise ValueError("v5 result cannot be positive and a clean null")
    reason = {
        "roster_effect_positive_unlock_exact_checkpoint_decision": (
            "the roster-average identity effect replicated on the untouched confirmation split"
        ),
        "identity_specific_effect_positive_unlock_targeted_exact_checkpoint_decision": (
            "at least one discovery identity survived the preregistered held-out confirmation"
        ),
        "affiliation_effect_positive_unlock_affiliation_mechanism_decision": (
            "the same-name affiliation effect replicated on both untouched splits"
        ),
        "clean_roster_null_stop_glm53_user_awareness_project": (
            "the full-roster combined interval excludes a material negative name effect"
        ),
        "ambiguous_roster_result_stop_and_report_heterogeneity": (
            "no positive or clean-null rule passed"
        ),
    }.get(decision)
    if reason is None:
        raise ValueError(f"unexpected v5 decision: {decision}")
    return GateDecision(
        project_id="glm53_user_eval_shared_direction_v1",
        gate="G3-roster-v5",
        run_id=run_id,
        prereg_sha256=prereg_sha256,
        source_lock_sha256=source_lock_sha256,
        model_revision=model_revision,
        runtime_hash=None,
        inputs=inputs,
        estimates=estimates,
        checks=checks,
        passed=positive,
        decision=decision,
        reason=reason,
        created_at=now_iso(),
    )
