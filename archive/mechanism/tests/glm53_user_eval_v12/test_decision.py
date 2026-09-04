from __future__ import annotations

from src.glm53_user_eval.v12.decision import decide_v12


def _inputs() -> tuple[dict, dict, dict]:
    primary = {"passed": True, "manual_override_allowed": False}
    verifier = {
        "passed": True,
        "scientific_role": "independent_diagnostic_no_primary_rescue",
    }
    independent = {"passed": True, "scientific_gate_passed": True}
    return primary, verifier, independent


def test_pass_unlocks_only_source_extraction() -> None:
    primary, verifier, independent = _inputs()
    report = decide_v12(
        primary=primary, verifier=verifier, independent=independent
    )
    assert report["passed"] is True
    assert report["authorization"] == {
        "runpod_compute_for_exact_fp8_source_extraction": True,
        "exact_fp8_source_extraction": True,
        "local_proxy_parity": False,
        "prompt_recruitment": False,
        "first_cot_transfer": False,
        "steering": False,
    }


def test_primary_failure_cannot_be_rescued() -> None:
    primary, verifier, independent = _inputs()
    primary["passed"] = False
    independent["scientific_gate_passed"] = False
    report = decide_v12(
        primary=primary, verifier=verifier, independent=independent
    )
    assert report["passed"] is False
    assert not any(report["authorization"].values())


def test_independent_disagreement_fails_closed() -> None:
    primary, verifier, independent = _inputs()
    independent["scientific_gate_passed"] = False
    report = decide_v12(
        primary=primary, verifier=verifier, independent=independent
    )
    assert report["passed"] is False


def test_manual_override_flag_fails_closed() -> None:
    primary, verifier, independent = _inputs()
    primary["manual_override_allowed"] = True
    report = decide_v12(
        primary=primary, verifier=verifier, independent=independent
    )
    assert report["passed"] is False


def test_compute_limits_are_exact() -> None:
    primary, verifier, independent = _inputs()
    report = decide_v12(
        primary=primary, verifier=verifier, independent=independent
    )
    assert report["compute_limits_if_passed"] == {
        "hard_cap_usd": 30.0,
        "balance_floor_usd": 12.0,
        "ordinary_pod_only": True,
        "subject": "official_fp8_glm53_flash",
    }
