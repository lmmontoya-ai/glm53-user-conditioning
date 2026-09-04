"""Machine decision for the V13 local-Codex judge cohort."""

from __future__ import annotations

from typing import Any


def decide_v13(
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
        state = "unchanged_bank_validated_by_both_codex_judges"
    elif luna and not terra:
        state = "judge_sensitive_luna_only_repair_audit_required"
    elif terra and not luna:
        state = "judge_sensitive_terra_only_repair_audit_required"
    else:
        state = "both_codex_judges_failed_repair_audit_required"
    passed = state == "unchanged_bank_validated_by_both_codex_judges"
    checks = {
        "technical_contract_passed": technical,
        "luna_max_independently_passed": luna,
        "terra_high_independently_passed": terra,
        "verification_agreed": verification.get("passed") is True,
        "manual_override_absent": analysis.get("manual_override_allowed") is False,
    }
    return {
        "schema_version": "glm53_v13_codex_cohort_decision_v1",
        "project_id": "glm53_user_eval_codex_judge_cohort_v13",
        "passed": passed,
        "decision": state,
        "checks": checks,
        "authorization": {
            "offline_failure_audit": technical and not passed,
            "v14_balanced_dataset_repair": technical and not passed,
            "exact_fp8_source_extraction": passed,
            "runpod_compute": passed,
            "local_proxy_parity": False,
            "prompt_recruitment": False,
            "first_cot_transfer": False,
            "steering": False,
        },
        "interpretation": (
            "Both independently configured local Codex judges passed every frozen V12 semantic threshold on the unchanged 576-row bank. Only bounded exact-FP8 source extraction is unlocked."
            if passed
            else "The unchanged bank was not robustly validated by both local Codex judges. No GPU work is unlocked. The next allowed action is the preregistered, paired failure audit and a separately versioned repair."
        ),
        "manual_or_ai_override_allowed": False,
    }


__all__ = ["decide_v13"]
