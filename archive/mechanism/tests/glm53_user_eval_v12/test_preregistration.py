from __future__ import annotations

from pathlib import Path

from pipelines.glm53_user_eval.v12.run import validate_prereg


def test_preregistration_and_all_source_locks_validate(repo_root: Path) -> None:
    prereg = (
        repo_root
        / "pipelines/glm53_user_eval/v12/configs/prereg_v12_fact_validator.yaml"
    )
    config = validate_prereg(prereg)
    assert config["amendment"]["human_review_required"] is False
    assert config["amendment"]["ai_diagnostic_review_is_human_evidence"] is False
    assert config["dataset"]["preserve_v11_bytes"] is True
    assert config["verifier"]["can_rescue_primary"] is False


def test_preregistration_keeps_downstream_claims_locked(repo_root: Path) -> None:
    prereg = (
        repo_root
        / "pipelines/glm53_user_eval/v12/configs/prereg_v12_fact_validator.yaml"
    )
    config = validate_prereg(prereg)
    locked = set(config["decision"]["remains_locked_after_semantic_pass"])
    assert "prompt_recruitment_until_local_parity_passes" in locked
    assert "first_cot_transfer" in locked
    assert "steering" in locked
