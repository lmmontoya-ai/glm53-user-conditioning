from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pipelines.glm53_user_eval.v11 import run as supervisor
from src.glm53_user_eval.v11.offline_analysis import (
    OfflineAnalysisError,
    build_offline_analysis,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture(tmp_path: Path, *, semantic_passed: bool) -> dict[str, Path]:
    repo = tmp_path / "repo"
    dataset = repo / "dataset"
    audit = repo / "audit"
    prereg = repo / "prereg.yaml"
    output = repo / "registered" / "analysis.json"
    dataset.mkdir(parents=True)
    audit.mkdir(parents=True)
    prereg.write_text(
        yaml.safe_dump(
            {
                "schema_version": "glm53_user_eval_v11_source_prereg_v1",
                "project_id": "glm53_user_eval_source_instrument_v11",
                "dataset": {"expected": {"total_rows": 576}},
                "offline_only_gate": {
                    "analysis_path": "registered/analysis.json"
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    samples = dataset / "samples.jsonl"
    samples.write_text('{"sample_id":"one"}\n', encoding="utf-8")
    samples_sha = sha256_file(samples)
    _write_json(dataset / "manifest.json", {"samples_sha256": samples_sha})
    _write_json(
        audit / "structural_audit.json",
        {
            "schema_version": "contrastive_prompts_v3_combined_structure_audit_v1",
            "passed": True,
            "samples_sha256": samples_sha,
            "primary": {"row_count": 576},
        },
    )
    _write_json(
        dataset / "tokenizer_audit.json",
        {
            "schema_version": "glm53_v11_tokenizer_audit_v1",
            "passed": True,
            "row_count": 576,
            "samples_sha256": samples_sha,
            "pair_contract": {"checked_pair_count": 240},
        },
    )
    development_lock = "d" * 64
    _write_json(
        audit / "development_analysis.json",
        {
            "schema_version": "contrastive_prompts_v3_development_text_audit_v1",
            "development_lock_sha256": development_lock,
            "model_bundle_sha256": "m" * 64,
            "final_holdout_evaluated": False,
            "fit_splits": ["train"],
            "selection_splits": ["validation", "development_counterfactual"],
            "post_selection_report_splits": ["ordinary_test"],
        },
    )
    final_text_path = audit / "final_text_analysis.json"
    _write_json(
        final_text_path,
        {
            "schema_version": "contrastive_prompts_v3_final_text_audit_v1",
            "development_lock_sha256": development_lock,
            "evaluated_split": "final_counterfactual",
            "row_count": 64,
            "selection_performed": False,
        },
    )
    _write_json(
        audit / "FINAL_TEXT_HOLDOUT_OPENED.json",
        {
            "schema_version": "glm53_v11_final_text_holdout_open_v1",
            "opened_once": True,
            "status": "complete",
            "development_lock_sha256": development_lock,
            "final_analysis_sha256": sha256_file(final_text_path),
            "samples_sha256": samples_sha,
        },
    )
    lexical_inputs = {
        "development": sha256_file(audit / "development_analysis.json"),
        "final_text": sha256_file(final_text_path),
        "samples": samples_sha,
        "tokenizer_audit": sha256_file(dataset / "tokenizer_audit.json"),
    }
    _write_json(
        audit / "lexical_decision.json",
        {
            "schema_version": "glm53_v11_lexical_gate_decision_v1",
            "passed": True,
            "decision": "lexical_baselines_pass_semantic_review_unlocked",
            "checks": {"metadata_final_le_060": True},
            "inputs": lexical_inputs,
        },
    )
    _write_json(
        audit / "semantic_validation.json",
        {
            "schema_version": "contrastive_prompts_v3_semantic_validation_v1",
            "passed": semantic_passed,
            "row_count": 576,
            "binary": {"accuracy": 0.98},
            "final_counterfactual": {"accuracy": 0.91},
            "controls": {"acceptance_rate": 0.80 if not semantic_passed else 0.92},
            "route_validation": {"passed": True},
            "realized_cost_usd": 0.68,
        },
    )

    primary_packet = audit / "manual_packet.csv"
    primary_packet.write_text("sample_id\none\n", encoding="utf-8")
    primary_sha = sha256_file(primary_packet)
    _write_json(
        audit / "manual_packet_lock.json",
        {
            "schema_version": "contrastive_prompts_v3_manual_audit_lock_v1",
            "row_count": 128,
            "packet_sha256": primary_sha,
        },
    )
    _write_json(
        audit / "manual_packet_manifest.json",
        {
            "schema_version": "contrastive_prompts_v3_manual_packet_v1",
            "row_count": 128,
            "packet_sha256": primary_sha,
        },
    )

    supplemental_packet = audit / "supplemental_semantic_disagreements.csv"
    supplemental_packet.write_text("sample_id\ntwo\n", encoding="utf-8")
    supplemental_manifest = (
        audit / "supplemental_semantic_disagreements_manifest.json"
    )
    _write_json(
        supplemental_manifest,
        {
            "schema_version": "contrastive_prompts_v3_supplemental_manual_packet_v1",
            "scientific_role": "supplemental_non_gating_human_review",
            "changes_preregistered_gate": False,
            "changes_semantic_metrics": False,
            "changes_paid_authorization": False,
            "packet_sha256": sha256_file(supplemental_packet),
            "row_count": 6,
            "source_hashes": {"samples_jsonl": samples_sha},
        },
    )
    (audit / "supplemental_semantic_disagreements_manifest.sha256").write_text(
        f"{sha256_file(supplemental_manifest)}  {supplemental_manifest.name}\n",
        encoding="ascii",
    )
    return {
        "repo": repo,
        "dataset": dataset,
        "audit": audit,
        "prereg": prereg,
        "output": output,
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_offline_analysis(
        repo_root=paths["repo"],
        prereg_path=paths["prereg"],
        dataset_root=paths["dataset"],
        audit_root=paths["audit"],
    )


def test_pending_manual_review_is_deterministic_and_never_authorizes(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, semantic_passed=False)

    first = _build(paths)
    first_bytes = paths["output"].read_bytes()
    second = _build(paths)

    assert first == second
    assert paths["output"].read_bytes() == first_bytes
    assert first["passed"] is False
    assert first["state"] == "semantic_validation_failed_manual_review_pending"
    assert first["components"]["primary_manual_review"]["status"] == (
        "pending_two_human_review"
    )
    assert first["components"]["supplemental_manual_review"]["status"] == (
        "pending_two_human_review"
    )
    assert not any(first["authorization"].values())
    assert first["inputs"]["semantic_validation"] == sha256_file(
        paths["audit"] / "semantic_validation.json"
    )
    assert not list(paths["output"].parent.glob("*.tmp"))


def test_completed_reviews_are_bound_but_analysis_still_cannot_authorize(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, semantic_passed=True)
    audit = paths["audit"]
    completed = audit / "manual_completed.csv"
    completed.write_text("completed primary\n", encoding="utf-8")
    _write_json(
        audit / "manual_audit.json",
        {
            "schema_version": "contrastive_prompts_v3_manual_audit_v1",
            "passed": True,
            "row_count": 128,
            "failure_count": 0,
            "reviewer_disagreement_count": 3,
            "completed_sha256": sha256_file(completed),
            "lock_sha256": sha256_file(audit / "manual_packet_lock.json"),
        },
    )

    supplemental_completed = (
        audit / "supplemental_semantic_disagreements_completed.csv"
    )
    supplemental_completed.write_text("completed supplemental\n", encoding="utf-8")
    supplemental_report = audit / "supplemental_semantic_review_report.json"
    _write_json(
        supplemental_report,
        {
            "schema_version": "contrastive_prompts_v3_supplemental_manual_review_v1",
            "passed_review_integrity": True,
            "changes_preregistered_gate": False,
            "changes_semantic_metrics": False,
            "changes_paid_authorization": False,
            "input_hashes": {
                "completed_packet": sha256_file(supplemental_completed)
            },
        },
    )
    _write_json(
        audit / "supplemental_semantic_review_manifest.json",
        {
            "schema_version": "contrastive_prompts_v3_supplemental_review_manifest_v1",
            "report_sha256": sha256_file(supplemental_report),
            "passed_review_integrity": True,
            "changes_preregistered_gate": False,
            "changes_semantic_metrics": False,
            "changes_paid_authorization": False,
        },
    )

    report = _build(paths)

    assert report["passed"] is True
    assert report["state"] == "offline_analysis_complete_all_component_gates_passed"
    primary = report["components"]["primary_manual_review"]
    supplemental = report["components"]["supplemental_manual_review"]
    assert primary["status"] == "complete_passed"
    assert primary["inputs"]["manual_audit"] == sha256_file(
        audit / "manual_audit.json"
    )
    assert supplemental["status"] == "complete_integrity_passed"
    assert supplemental["gating"] is False
    assert not any(report["authorization"].values())


def test_manual_audit_must_bind_completed_review(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, semantic_passed=True)
    audit = paths["audit"]
    completed = audit / "manual_completed.csv"
    completed.write_text("completed\n", encoding="utf-8")
    _write_json(
        audit / "manual_audit.json",
        {
            "schema_version": "contrastive_prompts_v3_manual_audit_v1",
            "passed": True,
            "row_count": 128,
            "failure_count": 0,
            "reviewer_disagreement_count": 0,
            "completed_sha256": hashlib.sha256(b"wrong").hexdigest(),
            "lock_sha256": sha256_file(audit / "manual_packet_lock.json"),
        },
    )

    with pytest.raises(OfflineAnalysisError, match="does not bind manual_completed"):
        _build(paths)


def test_registered_output_path_is_enforced(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, semantic_passed=False)

    with pytest.raises(OfflineAnalysisError, match="preregistered path"):
        build_offline_analysis(
            repo_root=paths["repo"],
            prereg_path=paths["prereg"],
            dataset_root=paths["dataset"],
            audit_root=paths["audit"],
            output_path=paths["repo"] / "other.json",
        )


def test_supervisor_exposes_standalone_combined_analysis_command() -> None:
    assert supervisor.COMMANDS["build-offline-analysis"] is (
        supervisor.command_build_offline_analysis
    )
