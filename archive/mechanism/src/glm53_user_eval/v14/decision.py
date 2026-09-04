"""Fail-closed V14 machine decision."""

from __future__ import annotations

from typing import Any


def decide_v14(
    *, analysis: dict[str, Any], verification: dict[str, Any]
) -> dict[str, Any]:
    judges = analysis.get("judges") or {}
    technical = (
        analysis.get("technical_passed") is True
        and verification.get("passed") is True
    )
    luna = (judges.get("luna_max") or {}).get("passed") is True
    terra = (judges.get("terra_high") or {}).get("passed") is True
    if not technical:
        state = "invalid_technical_run"
    elif luna and terra:
        state = "balanced_repaired_bank_validated_by_both_codex_judges"
    elif luna:
        state = "final_semantic_stop_terra_failed"
    elif terra:
        state = "final_semantic_stop_luna_failed"
    else:
        state = "final_semantic_stop_both_judges_failed"
    passed = state == "balanced_repaired_bank_validated_by_both_codex_judges"
    return {
        "schema_version": "glm53_v14_balanced_repair_decision_v1",
        "project_id": "glm53_user_eval_balanced_repair_v14",
        "passed": passed,
        "decision": state,
        "checks": {
            "technical_contract_passed": technical,
            "luna_max_independently_passed": luna,
            "terra_high_independently_passed": terra,
            "verification_agreed": verification.get("passed") is True,
            "fast_inference_disabled": all(
                (result.get("checks") or {}).get("fast_inference_disabled") is True
                for result in judges.values()
            ),
            "manual_override_absent": analysis.get("manual_override_allowed") is False,
        },
        "authorization": {
            "exact_fp8_source_extraction": passed,
            "runpod_compute": passed,
            "local_proxy_parity": False,
            "prompt_recruitment": False,
            "first_cot_transfer": False,
            "steering": False,
            "further_dataset_repair_before_application": False,
        },
        "interpretation": (
            "Both stronger local Codex judges independently passed every frozen semantic threshold on the systematically repaired bank. Only bounded exact-FP8 source extraction is unlocked."
            if passed
            else "The final balanced repair did not pass both independent semantic judges. No GPU work or further pre-application dataset repair is authorized."
        ),
        "manual_or_ai_override_allowed": False,
    }


__all__ = ["decide_v14"]
