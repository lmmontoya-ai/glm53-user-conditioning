"""Machine-only V12 semantic decision."""

from __future__ import annotations

from typing import Any


def decide_v12(
    *,
    primary: dict[str, Any],
    verifier: dict[str, Any],
    independent: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "primary_gate_passed": primary.get("passed") is True,
        "second_pass_complete_and_route_valid": verifier.get("passed") is True,
        "independent_recomputation_passed": independent.get("passed") is True,
        "independent_scientific_classification_agrees": independent.get(
            "scientific_gate_passed"
        )
        is primary.get("passed"),
        "manual_override_absent": primary.get("manual_override_allowed") is False,
        "verifier_did_not_rescue_primary": verifier.get("scientific_role")
        == "independent_diagnostic_no_primary_rescue",
    }
    passed = all(checks.values())
    decision = (
        "fact_extracted_semantic_validation_passed_source_extraction_unlocked"
        if passed
        else "fact_extracted_semantic_validation_failed_stop_all_experiments"
    )
    return {
        "schema_version": "glm53_v12_semantic_decision_v1",
        "project_id": "glm53_user_eval_fact_validator_v12",
        "passed": passed,
        "decision": decision,
        "checks": checks,
        "authorization": {
            "runpod_compute_for_exact_fp8_source_extraction": passed,
            "exact_fp8_source_extraction": passed,
            "local_proxy_parity": False,
            "prompt_recruitment": False,
            "first_cot_transfer": False,
            "steering": False,
        },
        "compute_limits_if_passed": {
            "hard_cap_usd": 30.0,
            "balance_floor_usd": 12.0,
            "ordinary_pod_only": True,
            "subject": "official_fp8_glm53_flash",
        },
        "manual_or_ai_review_override_allowed": False,
        "interpretation": (
            "The frozen text passed the new automatic four-fact validator. Only the bounded exact-FP8 source extraction is unlocked. Local parity must pass before prompt recruitment."
            if passed
            else "The frozen text did not pass the prospective four-fact validator. No further experimental branch is authorized before the application deadline."
        ),
    }


__all__ = ["decide_v12"]
