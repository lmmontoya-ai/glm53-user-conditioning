from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from src.glm53_user_eval.v11.semantic_stop_evidence import (
    CredentialPatternError,
    EvidenceError,
    build_semantic_stop_evidence,
    sha256_file,
)

TAG = "glm53-user-eval-v11-preregistered"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Evidence Test")

    prereg = (
        repo
        / "pipelines/glm53_user_eval/v11/configs/"
        "prereg_v11_source_instrument.yaml"
    )
    prereg.parent.mkdir(parents=True)
    with prereg.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("project_id: glm53_user_eval_source_instrument_v11\n")
    _git(repo, "add", prereg.relative_to(repo).as_posix())
    _git(repo, "commit", "--quiet", "-m", "Preregister test instrument")
    _git(repo, "tag", "-a", TAG, "-m", "Frozen test preregistration")

    samples = repo / "artifacts/datasets/contrastive_prompts_v3/samples.jsonl"
    samples.parent.mkdir(parents=True)
    sample_ids = tuple(f"sample_{letter}" for letter in "abcdef")
    samples.write_text(
        "".join(
            json.dumps({"sample_id": sample_id}) + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )

    audit = repo / "artifacts/glm53_user_eval/v11/offline_audit"
    rows = audit / "semantic_judge/rows"
    for sample_id in sample_ids:
        _write_json(
            rows / f"{sample_id}.json",
            {
                "sample_id": sample_id,
                "schema_version": "contrastive_prompts_v3_semantic_judge_row_v1",
                "usage": {"cost": 0.1},
            },
        )
    attempts = audit / "semantic_judge/attempt_logs"
    attempts.mkdir(parents=True)
    (attempts / "attempt_01.log").write_text("malformed response\n", encoding="utf-8")

    _write_json(
        audit / "semantic_validation.json",
        {
            "binary": {"accuracy": 1.0, "correct": 6, "count": 6, "threshold": 0.9},
            "controls": {
                "acceptance_rate": 0.5,
                "acceptable": 3,
                "count": 6,
                "threshold": 0.9,
            },
            "final_counterfactual": {
                "accuracy": 1.0,
                "correct": 6,
                "count": 6,
                "threshold": 0.9,
            },
            "passed": False,
            "realized_cost_usd": 0.6,
            "route_validation": {"failure_count": 0, "failures": [], "passed": True},
            "row_count": 6,
        },
    )
    _write_json(
        samples.parent / "manifest.json",
        {
            "dataset_id": "contrastive_prompts_v3",
            "row_count": 6,
            "samples_sha256": sha256_file(samples),
        },
    )
    tokenizer = samples.parent / "tokenizer_audit.json"
    _write_json(
        tokenizer,
        {
            "pair_contract": {"passed": True},
            "passed": True,
            "row_count": 6,
            "samples_sha256": sha256_file(samples),
        },
    )
    _write_json(
        audit / "structural_audit.json",
        {"passed": True, "samples_sha256": sha256_file(samples)},
    )
    model_bundle = audit / "development_models.joblib"
    model_bundle.write_bytes(b"deterministic test model bundle\n")
    development_lock = "a" * 64
    development = audit / "development_analysis.json"
    _write_json(
        development,
        {
            "development_lock_sha256": development_lock,
            "final_holdout_evaluated": False,
            "model_bundle_sha256": sha256_file(model_bundle),
        },
    )
    final_text = audit / "final_text_analysis.json"
    _write_json(
        final_text,
        {
            "development_lock_sha256": development_lock,
            "evaluated_split": "final_counterfactual",
            "row_count": 64,
            "selection_performed": False,
        },
    )
    _write_json(
        audit / "FINAL_TEXT_HOLDOUT_OPENED.json",
        {
            "development_lock_sha256": development_lock,
            "final_analysis_sha256": sha256_file(final_text),
            "opened_once": True,
            "samples_sha256": sha256_file(samples),
            "status": "complete",
        },
    )
    _write_json(
        audit / "lexical_decision.json",
        {
            "decision": "lexical_baselines_pass_semantic_review_unlocked",
            "inputs": {
                "development": sha256_file(development),
                "final_text": sha256_file(final_text),
                "samples": sha256_file(samples),
                "tokenizer_audit": sha256_file(tokenizer),
            },
            "passed": True,
        },
    )

    packet = audit / "manual_packet.csv"
    packet.parent.mkdir(parents=True, exist_ok=True)
    with packet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "prompt"])
        writer.writeheader()
        writer.writerow({"sample_id": "sample_a", "prompt": "one prompt"})
    packet_sha256 = sha256_file(packet)
    _write_json(
        audit / "manual_packet_lock.json",
        {"packet_sha256": packet_sha256, "row_count": 1},
    )
    _write_json(
        audit / "manual_packet_manifest.json",
        {"packet_sha256": packet_sha256, "row_count": 1},
    )

    supplemental_packet = audit / "supplemental_semantic_disagreements.csv"
    with supplemental_packet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "prompt"])
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow({"sample_id": sample_id, "prompt": f"prompt {sample_id}"})
    judgment_hash_rows = [
        {
            "sample_id": sample_id,
            "sha256": sha256_file(rows / f"{sample_id}.json"),
        }
        for sample_id in sample_ids
    ]
    judgment_set_sha256 = hashlib.sha256(
        json.dumps(
            judgment_hash_rows,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    supplemental_manifest = audit / "supplemental_semantic_disagreements_manifest.json"
    _write_json(
        supplemental_manifest,
        {
            "changes_paid_authorization": False,
            "changes_preregistered_gate": False,
            "changes_semantic_metrics": False,
            "packet_sha256": sha256_file(supplemental_packet),
            "row_count": 6,
            "scientific_role": "supplemental_non_gating_human_review",
            "selected_sample_ids": list(sample_ids),
            "source_hashes": {
                "original_manual_packet": sha256_file(packet),
                "samples_jsonl": sha256_file(samples),
                "semantic_judgment_set": judgment_set_sha256,
                "semantic_validation": sha256_file(
                    audit / "semantic_validation.json"
                ),
            },
        },
    )
    digest = audit / "supplemental_semantic_disagreements_manifest.sha256"
    digest.write_text(
        f"{sha256_file(supplemental_manifest)}  {supplemental_manifest.name}\n",
        encoding="ascii",
    )
    semantic_path = audit / "semantic_validation.json"
    lexical_path = audit / "lexical_decision.json"
    semantic_source = json.loads(semantic_path.read_text(encoding="utf-8"))
    lexical_source = json.loads(lexical_path.read_text(encoding="utf-8"))
    _write_json(
        audit / "analysis.json",
        {
            "authorization": {
                "new_glm_forwards": False,
                "runpod_compute": False,
                "source_activation_extraction": False,
                "steering": False,
                "user_recruitment": False,
            },
            "components": {
                "lexical_decision": {
                    field: lexical_source.get(field)
                    for field in ("checks", "decision", "passed", "schema_version")
                },
                "semantic_validation": {
                    field: semantic_source.get(field)
                    for field in (
                        "binary",
                        "controls",
                        "final_counterfactual",
                        "passed",
                        "realized_cost_usd",
                        "route_validation",
                        "row_count",
                        "schema_version",
                    )
                },
            },
            "inputs": {
                "dataset_manifest": sha256_file(samples.parent / "manifest.json"),
                "development_analysis": sha256_file(development),
                "final_holdout_marker": sha256_file(
                    audit / "FINAL_TEXT_HOLDOUT_OPENED.json"
                ),
                "final_text_analysis": sha256_file(final_text),
                "lexical_decision": sha256_file(lexical_path),
                "manual_packet": sha256_file(packet),
                "manual_packet_lock": sha256_file(
                    audit / "manual_packet_lock.json"
                ),
                "manual_packet_manifest": sha256_file(
                    audit / "manual_packet_manifest.json"
                ),
                "preregistration": sha256_file(prereg),
                "samples": sha256_file(samples),
                "semantic_validation": sha256_file(semantic_path),
                "structural_audit": sha256_file(audit / "structural_audit.json"),
                "supplemental_packet": sha256_file(supplemental_packet),
                "supplemental_packet_manifest": sha256_file(
                    supplemental_manifest
                ),
                "supplemental_packet_manifest_digest": sha256_file(digest),
                "tokenizer_audit": sha256_file(tokenizer),
            },
            "passed": False,
            "state": "semantic_validation_failed_manual_review_pending",
        },
    )
    return {"repo": repo, "prereg": prereg, "samples": samples, "audit": audit}


def test_builder_emits_deterministic_atomic_summary_and_full_manifest(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result_one = build_semantic_stop_evidence(
        repo_root=paths["repo"],
        audit_root=paths["audit"],
        samples_path=paths["samples"],
        prereg_path=paths["prereg"],
        expected_row_count=6,
    )
    manifest_path = paths["repo"] / result_one["manifest_path"]
    summary_path = paths["repo"] / result_one["summary_path"]
    manifest_bytes = manifest_path.read_bytes()
    summary_bytes = summary_path.read_bytes()

    result_two = build_semantic_stop_evidence(
        repo_root=paths["repo"],
        audit_root=paths["audit"],
        samples_path=paths["samples"],
        prereg_path=paths["prereg"],
        expected_row_count=6,
    )

    assert result_one == result_two
    assert manifest_path.read_bytes() == manifest_bytes
    assert summary_path.read_bytes() == summary_bytes
    assert not list(paths["audit"].glob(".*.tmp"))

    manifest = json.loads(manifest_bytes)
    summary = json.loads(summary_bytes)
    assert manifest["semantic_rows"]["actual_count"] == 6
    assert len(manifest["semantic_rows"]["files"]) == 6
    assert manifest["semantic_rows"]["preserved_valid_row_cost_usd"] == "0.6"
    assert manifest["attempt_logs"]["count"] == 1
    assert manifest["git_lock"]["tag_object_type"] == "tag"
    assert manifest["git_lock"]["prereg_matches_tag"] is True
    assert manifest["credential_scan"]["passed"] is True
    assert manifest["credential_scan"]["scanned_file_count"] == 25
    assert manifest["supplemental_review"] == {
        "digest_matches_manifest": True,
        "manifest_sha256": sha256_file(
            paths["audit"] / "supplemental_semantic_disagreements_manifest.json"
        ),
        "non_gating": True,
        "packet_sha256": sha256_file(
            paths["audit"] / "supplemental_semantic_disagreements.csv"
        ),
        "row_count": 6,
        "source_hashes_verified": True,
        "status": "pending_two_human_review",
    }
    assert all(manifest["offline_chain"].values())
    assert manifest["combined_offline_analysis"] == {
        "all_authorizations_false": True,
        "input_hashes_verified": True,
        "lexical_component_verified": True,
        "semantic_component_verified": True,
        "state": "semantic_validation_failed_manual_review_pending",
    }
    assert [item["code"] for item in manifest["provenance_warnings"]] == [
        "initial_rows_precede_preserved_attempt_logs",
        "failed_malformed_response_cost_unavailable",
    ]
    assert summary["state"] == "semantic_gate_failed_paid_compute_locked"
    assert summary["authorization"] == {
        "new_glm_forwards": False,
        "runpod_compute": False,
    }
    assert summary["evidence_manifest"]["sha256"] == hashlib.sha256(
        manifest_bytes
    ).hexdigest()


def test_builder_rejects_high_confidence_credential_pattern(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    attempt = paths["audit"] / "semantic_judge/attempt_logs/attempt_01.log"
    attempt.write_text(
        "OPENROUTER_API_KEY=sk-or-v1-abcdefghijklmnopqrstuvwxyz123456\n",
        encoding="utf-8",
    )

    with pytest.raises(CredentialPatternError, match="credential pattern"):
        build_semantic_stop_evidence(
            repo_root=paths["repo"],
            audit_root=paths["audit"],
            samples_path=paths["samples"],
            prereg_path=paths["prereg"],
            expected_row_count=6,
        )
    assert not (paths["audit"] / "semantic_stop_summary.json").exists()
    assert not (paths["audit"] / "semantic_stop_evidence_manifest.json").exists()


def test_builder_rejects_preregistration_changed_after_tag(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["prereg"].write_text("project_id: changed_after_tag\n", encoding="utf-8")

    with pytest.raises(
        EvidenceError,
        match="input hashes differ|differs from the tagged preregistration",
    ):
        build_semantic_stop_evidence(
            repo_root=paths["repo"],
            audit_root=paths["audit"],
            samples_path=paths["samples"],
            prereg_path=paths["prereg"],
            expected_row_count=6,
        )


def test_builder_rejects_supplemental_source_hash_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest_path = (
        paths["audit"] / "supplemental_semantic_disagreements_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_hashes"]["samples_jsonl"] = "0" * 64
    _write_json(manifest_path, manifest)
    digest_path = (
        paths["audit"] / "supplemental_semantic_disagreements_manifest.sha256"
    )
    digest_path.write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n",
        encoding="ascii",
    )

    with pytest.raises(EvidenceError, match="source hashes differ"):
        build_semantic_stop_evidence(
            repo_root=paths["repo"],
            audit_root=paths["audit"],
            samples_path=paths["samples"],
            prereg_path=paths["prereg"],
            expected_row_count=6,
        )


def test_builder_rejects_supplemental_digest_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    digest_path = (
        paths["audit"] / "supplemental_semantic_disagreements_manifest.sha256"
    )
    digest_path.write_text(
        f"{'0' * 64}  supplemental_semantic_disagreements_manifest.json\n",
        encoding="ascii",
    )

    with pytest.raises(EvidenceError, match="digest does not match"):
        build_semantic_stop_evidence(
            repo_root=paths["repo"],
            audit_root=paths["audit"],
            samples_path=paths["samples"],
            prereg_path=paths["prereg"],
            expected_row_count=6,
        )
