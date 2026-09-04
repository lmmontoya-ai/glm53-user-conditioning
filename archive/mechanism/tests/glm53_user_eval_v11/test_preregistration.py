from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest
import yaml
from pipelines.glm53_user_eval.v11.run import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DOWNSTREAM_MANIFEST,
    DEFAULT_DOWNSTREAM_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_PREREG,
    DEFAULT_RUNTIME,
    DEFAULT_SOURCE_ROOT,
    command_extract_source,
    load_prereg,
    validate_prereg,
)

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/v11/configs/prereg_v11_source_instrument.yaml"
RUNTIME = ROOT / "pipelines/glm53_user_eval/v11/configs/runtime_v11.yaml"


def test_literal_parent_hashes_and_dataset_arithmetic_validate() -> None:
    report = validate_prereg(PREREG)
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["checks"]["v10_machine_state"] is True
    assert report["checks"]["v7_machine_state"] is True
    assert report["checks"]["checkpoint_identity"] is True
    assert report["checks"]["runtime_subject_identity"] is True
    assert report["checks"]["locked_static_files"] is True


def test_parent_evidence_counts_and_manual_quota_are_frozen() -> None:
    config = load_prereg(PREREG)
    assert set(config["behavioral_parent_v7"]["evidence"]) == {
        "final_evidence",
        "preregistration",
        "schedule_manifest",
        "raw_scores",
        "analysis",
        "verification",
        "technical_audit",
        "manual_audit",
        "decision",
        "final_report",
    }
    expected = config["dataset"]["expected"]
    assert expected["total_rows"] == 576
    assert sum(expected["split_row_counts"].values()) == 576
    semantic = config["offline_only_gate"]["semantic_validation"]
    assert semantic["semantic_judge_expected_rows"] == 576
    assert semantic["manual_review_expected_rows"] == 128
    assert semantic["manual_review_expected_final_counterfactual_rows"] == 64
    assert semantic["manual_review_expected_factorial_calibration_rows"] == 32
    assert semantic["manual_review_expected_final_neutral_rows"] == 32
    builder = config["dataset"]["deterministic_builder"]
    assert len(builder["registry"]["sha256"]) == 64
    assert {name: len(record["sha256"]) for name, record in builder["frozen_sources"].items()} == {
        "final_binary": 64,
        "final_neutral": 64,
    }


def test_paid_source_features_bind_the_passing_text_decision() -> None:
    runner = (ROOT / "pipelines/glm53_user_eval/v11/run.py").read_text(encoding="utf-8")
    extractor = (ROOT / "src/glm53_user_eval/v11/extraction.py").read_text(encoding="utf-8")
    assert '"text_decision_sha256": sha256_file(args.audit_root / "decision.json")' in runner
    assert '"text_decision_sha256"' in extractor
    assert "offline text decision input-hash set differs" in runner
    assert "expected_checks = _text_gate_checks(args)" in runner
    assert 'decision.get("checks") != expected_checks' in runner


def test_every_paid_path_override_fails_before_runtime_import(tmp_path: Path) -> None:
    frozen = {
        "prereg": DEFAULT_PREREG,
        "dataset_root": DEFAULT_DATASET_ROOT,
        "audit_root": DEFAULT_AUDIT_ROOT,
        "runtime_config": DEFAULT_RUNTIME,
        "feature_root": DEFAULT_FEATURE_ROOT,
        "source_root": DEFAULT_SOURCE_ROOT,
        "downstream_manifest": DEFAULT_DOWNSTREAM_MANIFEST,
        "downstream_root": DEFAULT_DOWNSTREAM_ROOT,
    }
    base = {
        **frozen,
        "model_path": tmp_path / "model",
        "confirm_spend": True,
    }
    for field in frozen:
        values = dict(base)
        values[field] = tmp_path / f"override-{field}"
        with pytest.raises(ValueError, match=f"path override is forbidden: {field}"):
            command_extract_source(argparse.Namespace(**values))


def test_final_holdout_and_gpu_authorization_are_fail_closed() -> None:
    config = load_prereg(PREREG)
    assert config["final_holdout"]["use_for_layer_selection"] is False
    assert config["final_holdout"]["use_for_probe_selection"] is False
    assert config["offline_only_gate"]["authorization_on_failure"]["runpod_compute"] is False
    assert config["downstream_unlock"]["steering_permitted_in_v11_paid_run"] is False


def test_source_gate_uses_untouched_counterfactuals_and_robust_controls() -> None:
    config = load_prereg(PREREG)
    checks = config["source_activation_gate"]["checks"]
    assert checks["ordinary_test_auroc_min"] == 0.80
    assert checks["final_counterfactual_auroc_min"] == 0.75
    assert checks["final_counterfactual_fpr_at_80_tpr_max"] == 0.25
    assert checks["formal_casual_neutral_fpr_max"] == 0.20
    assert checks["ai_human_neutral_fpr_max"] == 0.20
    assert config["source_readouts"]["full_pipeline_pair_permutations"]["reps"] == 1000


def test_runtime_is_exact_b300_no_volume_and_bounded() -> None:
    runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))
    runpod = runtime["runpod"]
    assert runpod["gpu_id"] == "NVIDIA B300 SXM6 AC"
    assert runpod["gpu_count"] == 2
    assert runpod["network_volume_attached"] is False
    assert runpod["compute_hard_cap_usd"] == 29.5
    assert runpod["wall_clock_hard_cap_minutes"] == 110
    assert runpod["minimum_uncommitted_balance_usd"] == 15.0
    assert runpod["storage_allowance_usd"] == 0.1
    assert runtime["throughput_gate"]["permutation_worker_candidates"] == [16, 32]
    assert runpod["pod_scoped_api_key_required"] is True
    assert runpod["independent_on_pod_deadline_delete_required"] is True
    assert runpod["normal_terminal_self_delete_required"] is True
    assert runtime["software"]["uv"] == "0.9.26"
    transport = runtime["artifact_transport"]
    assert transport["credential_session_attestation_required"] is True
    assert transport["credential_read_write_probe_required"] is True
    assert transport["credential_rotation_required_after_terminal_cleanup"] is True
    assert runtime["model"]["static_file_sha256"]["config.json"] == (
        "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
    )
    assert runtime["model"]["static_file_sha256"]["chat_template.jinja"] == (
        "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
    )


def test_downstream_human_claim_gate_is_frozen() -> None:
    config = load_prereg(PREREG)
    manual = config["downstream_unlock"]["manual_audit"]
    assert manual["completed_review_schema"] == ("glm53_v11_downstream_completed_review_v1")
    assert manual["completed_review_rows_exact"] == 85
    assert manual["exact_human_attestation_required"] is True
    assert manual["every_frozen_check_must_pass"] is True
    assert manual["source_proxy_and_recruitment_decisions_hash_bound"] is True


def test_independent_verifier_does_not_import_primary_analysis() -> None:
    source = (ROOT / "src/glm53_user_eval/v11/verification.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        module.endswith(("text_audit", "semantic_validation", "manual_audit")) for module in modules
    )
    source_verifier = (ROOT / "src/glm53_user_eval/v11/source_verification.py").read_text(
        encoding="utf-8"
    )
    source_modules = {
        node.module
        for node in ast.walk(ast.parse(source_verifier))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module.endswith(("probes", "source_decision")) for module in source_modules)
