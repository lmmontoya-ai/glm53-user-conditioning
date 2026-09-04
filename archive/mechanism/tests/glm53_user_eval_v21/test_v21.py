from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.glm53_user_eval.v21.contract import validate_v21_prereg

from src.glm53_user_eval.v21 import supervisor

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/v21/configs/prereg_v21_exploratory_continuation.yaml"


def test_prereg_and_parent_baseline_are_bound() -> None:
    report = validate_v21_prereg(ROOT, PREREG)
    assert report["passed"] is True
    assert report["baseline_rows"] == 1404
    assert report["scope"] == "exploratory_post_failed_local_parity_gate"


def test_v20_failure_is_preserved() -> None:
    prereg = __import__("yaml").safe_load(PREREG.read_text(encoding="utf-8"))
    decision = json.loads(
        (ROOT / prereg["immutable_inputs"]["baseline_decision"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert decision["decision"] == "stop_local_parity_failed"
    assert decision["authorization"] == {"direct_user_intervention": False}


def test_continuation_reuses_baseline_and_scores_only_missing_arms() -> None:
    source = inspect.getsource(supervisor.run_exploratory_continuation)
    assert 'dict(row, arm="baseline", scope="user_content")' in source
    assert 'direction=None' not in source
    assert source.count("_score_arm(") == 2


def test_frozen_arm_and_row_counts() -> None:
    prereg = __import__("yaml").safe_load(PREREG.read_text(encoding="utf-8"))
    assert prereg["intervention"]["full_arms"] == [
        "all16_deploy",
        "all16_eval",
        "status_deploy",
        "actor_deploy",
        "formality",
    ]
    assert prereg["execution"]["new_full_arm_evaluations"] == 5 * 1404
    assert prereg["execution"]["new_signflip_evaluations"] == 20 * 80
    assert prereg["execution"]["total_new_prompt_evaluations"] == 8620


def test_no_claim_upgrade_or_outcome_selection() -> None:
    prereg = __import__("yaml").safe_load(PREREG.read_text(encoding="utf-8"))
    assert prereg["status"]["confirmatory"] is False
    assert prereg["intervention"]["outcome_conditioned_selection"] is False
    assert "local_parity_passed" in prereg["status"]["prohibited_claims"]


def test_batch_size_is_frozen_to_validated_single_row_path() -> None:
    runtime = __import__("yaml").safe_load(
        (ROOT / "pipelines/glm53_user_eval/v21/configs/runtime_v21.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert runtime["forward"]["batch_size"] == 1
    assert runtime["throughput_gate"]["planned_new_prompt_evaluations"] == 8620


def test_paid_transport_is_bound_to_v21() -> None:
    launcher = (ROOT / "infra/runpod/new_glm53_v20_hua_pod.ps1").read_text(encoding="utf-8")
    bootstrap = (ROOT / "infra/runpod/bootstrap_glm53_v20.sh").read_text(encoding="utf-8")
    for text in (launcher, bootstrap):
        assert "glm53-user-eval-v21-preregistered" in text
        assert "9bec01652fe5ae1e87da238777820b4577704087" in text
        assert "glm53_user_eval_hua_exploratory_continuation_v21" in text
    assert "v20_baseline_raw_scores.jsonl" in launcher
    assert "91702099d664bf19626c1f3f944e667e26f5e6bb379a9284dbc14a7449af2ce3" in bootstrap
