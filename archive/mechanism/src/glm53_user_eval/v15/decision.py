"""V15 terminal machine decision."""

from __future__ import annotations

from typing import Any


def decide_v15(
    *, analysis: dict[str, Any], verification: dict[str, Any]
) -> dict[str, Any]:
    judges = analysis.get("judges") or {}
    technical = analysis.get("technical_passed") is True and verification.get("passed") is True
    luna = (judges.get("luna_max") or {}).get("passed") is True
    terra = (judges.get("terra_high") or {}).get("passed") is True
    if not technical:
        state = "invalid_technical_run"
    elif luna and terra:
        state = "fresh_control_bank_validated_by_both_codex_judges"
    else:
        state = "final_semantic_stop_after_fresh_controls"
    passed = state == "fresh_control_bank_validated_by_both_codex_judges"
    return {
        "schema_version": "glm53_v15_fresh_controls_decision_v1",
        "project_id": "glm53_user_eval_fresh_controls_v15",
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
        },
        "authorization": {
            "exact_fp8_source_extraction": passed,
            "runpod_compute": passed,
            "local_proxy_parity": False,
            "prompt_recruitment": False,
            "first_cot_transfer": False,
            "steering": False,
            "further_dataset_repair": False,
        },
        "interpretation": (
            "Both independent non-fast Codex judges passed the repaired source bank with a fresh balanced control surface. Only bounded exact-FP8 source extraction is unlocked."
            if passed
            else "The fresh balanced control confirmation did not pass both judges. No GPU work or further dataset repair is authorized."
        ),
        "manual_or_ai_override_allowed": False,
    }


__all__ = ["decide_v15"]
