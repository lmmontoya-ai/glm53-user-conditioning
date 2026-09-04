"""Command line supervisor for the v11 source-instrument project."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v11.builder import build_dataset
from src.glm53_user_eval.v11.dataset import load_rows, validate_rows
from src.glm53_user_eval.v11.manual_audit import (
    build_manual_packet,
    validate_completed_manual_audit,
)
from src.glm53_user_eval.v11.offline_analysis import build_offline_analysis
from src.glm53_user_eval.v11.reviewer_workflow import (
    merge_adjudication,
    merge_independent_reviews,
    prepare_review_assignment,
)
from src.glm53_user_eval.v11.semantic_validation import (
    analyze_semantic_judgments,
    load_judgment_rows,
    run_semantic_judge,
)
from src.glm53_user_eval.v11.source_decision import decide_source_instrument
from src.glm53_user_eval.v11.text_audit import (
    evaluate_final_holdout,
    fit_development_baselines,
    load_development_audit,
    save_development_audit,
    select_development_rows,
    select_final_holdout_rows,
    validate_dataset_structure,
)
from src.glm53_user_eval.v11.tokenizer_audit import audit_dataset
from src.glm53_user_eval.v11.verification import verify_offline_gate

DEFAULT_PREREG = ROOT / "pipelines/glm53_user_eval/v11/configs/prereg_v11_source_instrument.yaml"
DEFAULT_DATASET_ROOT = ROOT / "artifacts/datasets/contrastive_prompts_v3"
DEFAULT_AUDIT_ROOT = ROOT / "artifacts/glm53_user_eval/v11/offline_audit"
DEFAULT_RUNTIME = ROOT / "pipelines/glm53_user_eval/v11/configs/runtime_v11.yaml"
DEFAULT_FEATURE_ROOT = ROOT / "artifacts/glm53_user_eval/v11/features/source"
DEFAULT_SOURCE_ROOT = ROOT / "artifacts/glm53_user_eval/v11/source_readout"
DEFAULT_DOWNSTREAM_MANIFEST = (
    ROOT / "pipelines/glm53_user_eval/v11/configs/downstream_manifest_v1.json"
)
DEFAULT_DOWNSTREAM_ROOT = ROOT / "artifacts/glm53_user_eval/v11/downstream"
DEFAULT_TRANSLUCE_PERSONAS = ROOT.parent / "reference/transluce-user-awareness/core/personas2.json"
TOKENIZER_SNAPSHOT_ROOT = ROOT / "artifacts/glm53_user_eval/v11/tokenizer_snapshot"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_prereg(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["schema_version"] != "glm53_user_eval_v11_source_prereg_v1":
        raise ValueError("unexpected v11 preregistration schema")
    if config["project_id"] != "glm53_user_eval_source_instrument_v11":
        raise ValueError("unexpected v11 project ID")
    return config


def validate_prereg(path: Path) -> dict[str, Any]:
    config = load_prereg(path)
    runtime = yaml.safe_load(DEFAULT_RUNTIME.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    evidence_sections = {
        "v10": config["parent_v10"]["evidence"],
        "v7": config["behavioral_parent_v7"]["evidence"],
    }
    for parent, section in evidence_sections.items():
        for name, record in section.items():
            artifact = ROOT / record["path"]
            actual = sha256_file(artifact)
            checks[f"locked_{parent}_{name}"] = actual == record["sha256"]
    checks["v10_tag_commit"] = (
        subprocess.check_output(
            ["git", "rev-list", "-n", "1", config["parent_v10"]["final_tag"]],
            cwd=ROOT,
            text=True,
        ).strip()
        == config["parent_v10"]["final_commit"]
    )
    checks["v7_tag_commit"] = (
        subprocess.check_output(
            ["git", "rev-list", "-n", "1", config["behavioral_parent_v7"]["final_tag"]],
            cwd=ROOT,
            text=True,
        ).strip()
        == config["behavioral_parent_v7"]["final_commit"]
    )
    v10_decision = json.loads(
        (ROOT / config["parent_v10"]["evidence"]["decision"]["path"]).read_text(encoding="utf-8")
    )
    v10_verification = json.loads(
        (ROOT / config["parent_v10"]["evidence"]["verification"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    checks["v10_machine_state"] = (
        v10_decision.get("decision") == config["parent_v10"]["decision"]
        and v10_decision.get("passed") is True
        and v10_verification.get("passed") is True
        and all(
            v10_decision.get("authorization", {}).get(field) is False
            for field in ("new_paid_compute", "user_recruitment", "steering")
        )
    )
    v7_decision = json.loads(
        (ROOT / config["behavioral_parent_v7"]["evidence"]["decision"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    v7_verification = json.loads(
        (ROOT / config["behavioral_parent_v7"]["evidence"]["verification"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    checks["v7_machine_state"] = (
        v7_decision.get("decision") == config["behavioral_parent_v7"]["decision"]
        and v7_decision.get("whitebox_green_light") is True
        and v7_verification.get("passed") is True
    )
    shard = config["subject"]["shard_manifest"]
    checks["locked_shard_manifest"] = sha256_file(ROOT / shard["path"]) == shard["sha256"]
    stage_manifest = json.loads((ROOT / shard["path"]).read_text(encoding="utf-8"))
    subject = config["subject"]
    checks["checkpoint_identity"] = (
        stage_manifest.get("model_id") == subject["model_id"]
        and stage_manifest.get("revision") == subject["revision"]
        and int(stage_manifest.get("safetensor_shards", -1)) == subject["weight_shards"]
        and int(stage_manifest.get("safetensor_bytes", -1)) == subject["weight_bytes"]
        and len(stage_manifest.get("safetensor_sha256", {})) == subject["weight_shards"]
    )
    runtime_model = runtime["model"]
    checks["runtime_subject_identity"] = (
        runtime["schema_version"] == "glm53_v11_runtime_v1"
        and runtime["project_id"] == config["project_id"]
        and runtime_model["model_id"] == subject["model_id"]
        and runtime_model["revision"] == subject["revision"]
        and runtime_model["precision"] == subject["precision"]
        and runtime_model["weight_shards"] == subject["weight_shards"]
        and runtime_model["weight_bytes"] == subject["weight_bytes"]
        and runtime_model["shard_manifest_sha256"] == shard["sha256"]
        and runtime["software"]["transformers_commit"] == subject["transformers_commit"]
        and runtime_model["static_file_sha256"] == subject["static_file_sha256"]
    )
    checks["locked_static_files"] = all(
        (TOKENIZER_SNAPSHOT_ROOT / name).is_file()
        and sha256_file(TOKENIZER_SNAPSHOT_ROOT / name) == expected_sha256
        for name, expected_sha256 in subject["static_file_sha256"].items()
    )
    expected = config["dataset"]["expected"]
    checks["dataset_arithmetic"] = (
        expected["binary_rows"] + expected["neutral_rows"] + expected["factorial_calibration_rows"]
        == expected["total_rows"]
    )
    checks["binary_pair_arithmetic"] = expected["binary_pairs"] * 2 == expected["binary_rows"]
    checks["dataset_exact_counts"] = expected == {
        "total_rows": 576,
        "binary_pairs": 240,
        "binary_rows": 480,
        "neutral_rows": 64,
        "factorial_calibration_rows": 32,
        "split_pair_counts": {
            "train": 128,
            "validation": 24,
            "ordinary_test": 24,
            "development_counterfactual": 32,
            "final_counterfactual": 32,
        },
        "split_row_counts": {
            "train": 256,
            "validation": 48,
            "ordinary_test": 48,
            "development_counterfactual": 64,
            "final_counterfactual": 64,
            "neutral_controls": 64,
            "factorial_calibration": 32,
        },
    }
    checks["split_rows_sum_to_total"] = (
        sum(expected["split_row_counts"].values()) == expected["total_rows"]
    )
    checks["runtime_dataset_count"] = (
        runtime["extraction"]["expected_rows"] == expected["total_rows"]
    )
    builder_contract = config["dataset"]["deterministic_builder"]
    registry_record = builder_contract["registry"]
    registry_path = ROOT / registry_record["path"]
    checks["locked_v3_registry"] = (
        registry_path.is_file() and sha256_file(registry_path) == registry_record["sha256"]
    )
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    frozen_source_records = builder_contract["frozen_sources"]
    checks["locked_v3_final_sources"] = all(
        (ROOT / record["path"]).is_file() and sha256_file(ROOT / record["path"]) == record["sha256"]
        for record in frozen_source_records.values()
    )
    checks["registry_frozen_source_agreement"] = (
        registry["construction"]["frozen_final_binary_source"]
        == frozen_source_records["final_binary"]["path"]
        and registry["construction"]["frozen_final_binary_sha256"]
        == frozen_source_records["final_binary"]["sha256"]
        and registry["construction"]["frozen_final_neutral_source"]
        == frozen_source_records["final_neutral"]["path"]
        and registry["construction"]["frozen_final_neutral_sha256"]
        == frozen_source_records["final_neutral"]["sha256"]
    )
    semantic = config["offline_only_gate"]["semantic_validation"]
    checks["manual_review_quota"] = (
        semantic["manual_review_expected_rows"] == 128
        and semantic["manual_review_expected_final_counterfactual_rows"]
        == expected["split_row_counts"]["final_counterfactual"]
        and semantic["manual_review_expected_factorial_calibration_rows"]
        == expected["split_row_counts"]["factorial_calibration"]
        and semantic["manual_review_expected_final_neutral_rows"]
        == expected["split_row_counts"]["neutral_controls"] // 2
        and semantic["semantic_judge_expected_rows"] == expected["total_rows"]
    )
    checks["offline_gate_order"] = config["offline_only_gate"]["execution_order"] == [
        "validate_preregistration_and_parent_locks",
        "build_and_structurally_audit_all_576_rows",
        "audit_with_exact_subject_tokenizer_and_chat_template",
        "fit_text_baselines_on_development_splits",
        "open_final_text_holdout_once",
        "decide_lexical_gate",
        "run_semantic_judge_only_if_lexical_gate_passes",
        "complete_blinded_manual_review_only_if_lexical_gate_passes",
        "independently_verify_offline_artifacts",
        "write_machine_text_gate_decision",
        "permit_exact_checkpoint_source_extraction_only_if_text_gate_passes",
    ]
    checks["no_pregpu_glm"] = config["execution"]["no_gpu_work_before_text_gate"] is True
    checks["steering_forbidden"] = "steering" in config["execution"]["paid_scope"]["forbidden"]
    downstream = config["downstream_unlock"]
    downstream_lock = downstream["frozen_manifest"]
    downstream_manifest_path = ROOT / downstream_lock["path"]
    checks["locked_downstream_manifest"] = (
        downstream_manifest_path.is_file()
        and sha256_file(downstream_manifest_path) == downstream_lock["sha256"]
    )
    downstream_manifest = json.loads(downstream_manifest_path.read_text(encoding="utf-8"))
    downstream_asset_names = {
        "parent_proxy_surface": "parent_proxy_surface",
        "proxy_codebooks": "proxy_codebooks",
        "proxy_token_contract": "proxy_contract",
    }
    checks["locked_downstream_assets"] = all(
        (ROOT / record["path"]).is_file()
        and sha256_file(ROOT / record["path"]) == record["sha256"]
        and downstream_manifest["assets"][downstream_asset_names[name]]["sha256"]
        == record["sha256"]
        for name, record in downstream["frozen_assets"].items()
    )
    codebooks = json.loads(
        (ROOT / downstream["frozen_assets"]["proxy_codebooks"]["path"]).read_text(encoding="utf-8")
    )
    expected_codebook_0 = dict(zip("ABCDEFGHIJK", range(0, 101, 10), strict=True))
    expected_codebook_1 = dict(
        zip("ABCDEFGHIJK", (50, 40, 30, 20, 10, 0, 100, 80, 90, 70, 60), strict=True)
    )
    checks["downstream_codebook_contract"] = (
        codebooks["codebooks"] == {"0": expected_codebook_0, "1": expected_codebook_1}
        and codebooks["assignment"]["rule"] == "(analysis_index + frozen_task_rank) modulo 2"
    )
    parent_surface = json.loads(
        (ROOT / downstream["frozen_assets"]["parent_proxy_surface"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    checks["downstream_parent_surface"] = (
        parent_surface["passed"] is True
        and parent_surface["counts"]["pre_missing"] == 6400
        and parent_surface["counts"]["local_reconstructable"] == 6387
        and parent_surface["counts"]["api_parent_valid"] == 6375
        and parent_surface["counts"]["ineligible_empty_first_assistant"] == 13
        and parent_surface["estimate"]["interaction_pp"] == -0.7797975559008098
        and parent_surface["bootstrap"]
        == {"confidence_levels": [0.9, 0.95], "reps": 20000, "seed": 20260830}
        and parent_surface["independent_recomputation"]["passed"] is True
    )
    checks["downstream_same_process"] = (
        downstream["same_pod_continuation_required"] is True
        and downstream["same_loaded_model_process_required"] is True
        and downstream["model_reload_permitted"] is False
    )
    checks["downstream_no_cot_or_steering"] = (
        downstream["early_cot_transfer_permitted_in_v11_paid_run"] is False
        and downstream["steering_permitted_in_v11_paid_run"] is False
    )
    checks["downstream_bootstrap_contract"] = (
        downstream["local_proxy"]["bootstrap_reps"] == 20000
        and downstream["local_proxy"]["bootstrap_seed"] == 20260921
        and downstream["recruitment"]["bootstrap_reps"] == 20000
        and downstream["recruitment"]["bootstrap_seed"] == 20260922
    )
    local_proxy = downstream["local_proxy"]
    checks["downstream_proxy_measurement"] = (
        local_proxy["expected_pre_missing_rows"] == 6400
        and local_proxy["expected_local_reconstructable_rows"] == 6387
        and local_proxy["expected_api_parent_valid_rows"] == 6375
        and local_proxy["expected_ineligible_rows_for_manual_review"] == 13
        and local_proxy["parent_api_matched_interaction_pp"] == -0.7797975559008098
        and local_proxy["codebook_assignment"] == "(analysis_index + frozen_task_rank) modulo 2"
        and local_proxy["measurement"]["genpop_center"]
        == "per_dilemma_mean_of_valid_selected_genpop_scores"
        and local_proxy["measurement"]["group_summary"] == "equal_weight_across_16_people"
    )
    checks["downstream_manual_quota"] = (
        downstream["manual_audit"]["proxy_rows_exact"] == 40
        and downstream["manual_audit"]["proxy_rows_per_group"] == 10
        and downstream["manual_audit"]["proxy_rows_per_codebook"] == 20
        and downstream["manual_audit"]["recruitment_rows_exact"] == 32
        and downstream["manual_audit"]["recruitment_rows_per_group"] == 8
        and downstream["manual_audit"]["expected_proxy_technical_error_rows"] == 13
        and downstream["manual_audit"]["completed_review_rows_exact"] == 85
        and downstream["manual_audit"]["completed_review_schema"]
        == "glm53_v11_downstream_completed_review_v1"
        and downstream["manual_audit"]["human_review_required_for_final_claim"] is True
    )
    completion = downstream_manifest["manual_audit"]["completion_contract"]
    checks["locked_downstream_manual_completion"] = (
        downstream_manifest["manual_audit"]["expected_technical_error_rows"] == 13
        and completion["schema_version"] == "glm53_v11_downstream_completed_review_v1"
        and completion["human_attestation_exact"]
        == "I personally reviewed this row without automated substitution."
        and set(completion["required_checks"]) == {"proxy", "recruitment", "technical_error"}
        and completion["all_checks_must_pass"] is True
        and completion["machine_decisions_and_independent_verifications_must_be_hash_bound"] is True
    )
    paid = downstream["paid_resource_contract"]
    checks["downstream_paid_resource_contract"] = (
        paid["gpu_topology"] == "2x_NVIDIA_B300_SXM6_AC"
        and paid["aggregate_rate_cap_usd_per_hour"] == 15.78
        and paid["wall_clock_hard_cap_minutes"] == 110
        and paid["configured_compute_hard_cap_usd"] == 29.50
        and paid["minimum_uncommitted_balance_usd"] == 15.00
        and paid["source_permutation_worker_benchmarks"] == [16, 32]
        and paid["source_permutation_benchmark_reps_each"] == 32
        and paid["source_permutation_total_reps"] == 1000
        and paid["s3_credential_session_attestation_max_hours"] == 24
        and paid["s3_credential_read_write_probe_required"] is True
        and paid["s3_credentials_transient_pod_environment_only"] is True
        and paid["s3_credential_rotation_required_after_project"] is True
        and paid["runpod_injected_pod_scoped_api_key_required"] is True
        and paid["independent_on_pod_deadline_delete_required"] is True
        and paid["normal_terminal_self_delete_required"] is True
        and runtime["runpod"]["wall_clock_hard_cap_minutes"] == 110
        and runtime["runpod"]["compute_hard_cap_usd"] == 29.50
        and runtime["runpod"]["pod_scoped_api_key_required"] is True
        and runtime["runpod"]["independent_on_pod_deadline_delete_required"] is True
        and runtime["runpod"]["normal_terminal_self_delete_required"] is True
        and runtime["software"]["uv"] == "0.9.26"
        and runtime["artifact_transport"]["credential_session_attestation_required"] is True
        and runtime["artifact_transport"]["credential_read_write_probe_required"] is True
        and runtime["artifact_transport"]["credential_rotation_required_after_terminal_cleanup"]
        is True
        and runtime["throughput_gate"]["permutation_worker_candidates"] == [16, 32]
    )
    if not all(checks.values()):
        raise ValueError(f"v11 preregistration checks failed: {checks}")
    return {
        "schema_version": "glm53_v11_prereg_validation_v1",
        "passed": True,
        "prereg_sha256": sha256_file(path),
        "checks": checks,
    }


def _load_and_validate_dataset(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples_path = dataset_root / "samples.jsonl"
    rows = load_rows(samples_path)
    report = validate_rows(rows)
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["samples_sha256"] != sha256_file(samples_path):
        raise ValueError("dataset manifest does not bind samples.jsonl")
    return rows, report


def _require_pass(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("passed") is not True:
        raise ValueError(f"required artifact did not pass: {path}")
    return value


def _load_terminal_gate_artifact(path: Path) -> dict[str, Any]:
    """Load a completed gate artifact without requiring a positive result."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("passed"), bool):
        raise TypeError(f"terminal gate artifact lacks a Boolean passed field: {path}")
    return value


def _require_completed_final_text(args: argparse.Namespace) -> dict[str, Any]:
    marker_path = args.audit_root / "FINAL_TEXT_HOLDOUT_OPENED.json"
    output_path = args.audit_root / "final_text_analysis.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("opened_once") is not True or marker.get("status") != "complete":
        raise ValueError("final text holdout did not finish its one-time evaluation")
    if not output_path.is_file() or marker.get("final_analysis_sha256") != sha256_file(output_path):
        raise ValueError("final text holdout marker does not bind final_text_analysis.json")
    return marker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--downstream-manifest", type=Path, default=DEFAULT_DOWNSTREAM_MANIFEST)
    parser.add_argument("--downstream-root", type=Path, default=DEFAULT_DOWNSTREAM_ROOT)
    parser.add_argument("--transluce-personas", type=Path, default=DEFAULT_TRANSLUCE_PERSONAS)
    parser.add_argument("--completed-manual-audit", type=Path)
    parser.add_argument("--completed-downstream-review", type=Path)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument(
        "--packet-kind",
        choices=("all", "primary", "supplemental"),
        default="all",
    )
    parser.add_argument("--reviewer-1-id")
    parser.add_argument("--reviewer-2-id")
    parser.add_argument("--reviewer-1-completed", type=Path)
    parser.add_argument("--reviewer-2-completed", type=Path)
    parser.add_argument("--adjudicator-id")
    parser.add_argument("--completed-adjudication", type=Path)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--permutation-reps", type=int, default=1000)
    parser.add_argument("--permutation-workers", type=int, default=16)
    parser.add_argument("--confirm-spend", action="store_true")
    return parser


def command_validate_prereg(args: argparse.Namespace) -> None:
    print(json.dumps(validate_prereg(args.prereg), indent=2))


def command_plan(args: argparse.Namespace) -> None:
    prereg = validate_prereg(args.prereg)
    config = load_prereg(args.prereg)
    runtime = yaml.safe_load((args.prereg.parent / "runtime_v11.yaml").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "prereg_sha256": prereg["prereg_sha256"],
                "dataset_expected": config["dataset"]["expected"],
                "offline_gate": config["offline_only_gate"]["gate_id"],
                "paid_compute_unlocked": False,
                "runpod_compute_cap_usd": runtime["runpod"]["compute_hard_cap_usd"],
            },
            indent=2,
        )
    )


def command_validate_downstream(args: argparse.Namespace) -> None:
    from src.glm53_user_eval.v11.downstream import (
        atomic_json as downstream_atomic_json,
    )
    from src.glm53_user_eval.v11.downstream import (
        load_manifest,
        validate_downstream_assets,
    )

    manifest = load_manifest(args.downstream_manifest)
    personas_record = manifest["assets"]["personas"]
    target = ROOT / personas_record["target_path"]
    if not target.is_file():
        if not args.transluce_personas.is_file():
            raise ValueError("frozen Transluce personas source is absent")
        if sha256_file(args.transluce_personas) != personas_record["sha256"]:
            raise ValueError("Transluce personas source differs from downstream lock")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(args.transluce_personas, temporary)
        os.replace(temporary, target)
    report, _, _ = validate_downstream_assets(
        repo_root=ROOT,
        manifest_path=args.downstream_manifest,
    )
    downstream_atomic_json(
        ROOT / "artifacts/glm53_user_eval/v11/downstream_inputs/preflight.json",
        report,
    )
    print(json.dumps(report, indent=2))


def command_authorize_downstream_claim(args: argparse.Namespace) -> None:
    """Validate a real completed human review before authorizing the claim."""

    if args.completed_downstream_review is None:
        raise ValueError("--completed-downstream-review is required")
    from src.glm53_user_eval.v11.downstream import load_manifest
    from src.glm53_user_eval.v11.downstream_manual_review import (
        validate_completed_downstream_review,
        validate_positive_claim_machine_artifacts,
    )

    manifest = load_manifest(args.downstream_manifest)
    canonical = args.downstream_root / "completed_human_review.jsonl"
    supplied = args.completed_downstream_review.resolve()
    if not supplied.is_file():
        raise ValueError("completed downstream human review is absent")
    if canonical.is_file() and sha256_file(canonical) != sha256_file(supplied):
        raise ValueError("a different completed downstream review is already frozen")
    review_input = canonical if canonical.is_file() else supplied
    review = validate_completed_downstream_review(
        completed_path=review_input,
        packet_path=args.downstream_root / "manual_packet.jsonl",
        technical_errors_path=args.downstream_root / "technical_errors.jsonl",
        template_path=args.downstream_root / "manual_review_template.jsonl",
        status_path=args.downstream_root / "manual_audit_status.json",
        manifest=manifest,
    )
    if not canonical.is_file():
        canonical.parent.mkdir(parents=True, exist_ok=True)
        temporary = canonical.with_suffix(canonical.suffix + ".partial")
        temporary.write_bytes(supplied.read_bytes())
        os.replace(temporary, canonical)
    machine = validate_positive_claim_machine_artifacts(
        source_decision_path=args.source_root / "decision.json",
        source_verification_path=args.source_root / "verification.json",
        proxy_decision_path=args.downstream_root / "local_proxy/decision.json",
        proxy_verification_path=args.downstream_root / "local_proxy/verification.json",
        recruitment_decision_path=args.downstream_root / "recruitment/decision.json",
        recruitment_verification_path=args.downstream_root / "recruitment/verification.json",
        downstream_manifest_path=args.downstream_manifest,
        downstream_preflight_path=args.downstream_root / "preflight.json",
    )
    authorized = bool(review["passed"] and machine["passed"])
    decision = review | {
        "passed": authorized,
        "final_claim_authorized": authorized,
        "machine_claim_checks": machine["checks"],
        "inputs": review["inputs"]
        | machine["inputs"]
        | {"completed_review": sha256_file(canonical)},
    }
    atomic_json(args.downstream_root / "manual_review_decision.json", decision)
    authorization_path = args.downstream_root / "final_claim_authorization.json"
    if not authorized:
        if authorization_path.exists():
            raise ValueError("a stale positive downstream claim authorization exists")
        print(json.dumps(decision, indent=2))
        return
    authorization = {
        "schema_version": "glm53_v11_downstream_final_claim_authorization_v1",
        "passed": True,
        "human_review_completed": True,
        "final_claim_authorized": True,
        "claim_ready": True,
        "review_counts": review["review_counts"],
        "reviewer_ids": review["reviewer_ids"],
        "authorization": {
            "final_claim": True,
            "early_cot": False,
            "steering": False,
        },
        "claim_scope": (
            "Validated linear-readout recruitment only; no causal, early-CoT, "
            "steering, or API-stack-identity claim."
        ),
        "inputs": decision["inputs"]
        | {
            "manual_review_decision": sha256_file(
                args.downstream_root / "manual_review_decision.json"
            )
        },
    }
    if authorization_path.is_file():
        existing = json.loads(authorization_path.read_text(encoding="utf-8"))
        if existing != authorization:
            raise ValueError("existing downstream claim authorization differs")
    else:
        atomic_json(authorization_path, authorization)
    print(json.dumps(authorization, indent=2))


def command_build_dataset(args: argparse.Namespace) -> None:
    validate_prereg(args.prereg)
    manifest = build_dataset(args.dataset_root)
    rows, structural = _load_and_validate_dataset(args.dataset_root)
    if len(rows) != manifest["row_count"]:
        raise ValueError("built dataset row count differs")
    atomic_json(args.audit_root / "structural_audit.json", structural)
    print(json.dumps(manifest, indent=2))


def command_audit_structure(args: argparse.Namespace) -> None:
    rows, structural = _load_and_validate_dataset(args.dataset_root)
    second = validate_dataset_structure(rows)
    combined = {
        "schema_version": "contrastive_prompts_v3_combined_structure_audit_v1",
        "passed": structural["passed"] and second["passed"],
        "primary": structural,
        "independent_contract": second,
        "samples_sha256": sha256_file(args.dataset_root / "samples.jsonl"),
    }
    atomic_json(args.audit_root / "structural_audit.json", combined)
    print(json.dumps({"passed": combined["passed"], "row_count": len(rows)}, indent=2))


def command_audit_tokenizer(args: argparse.Namespace) -> None:
    if args.tokenizer_root is None:
        raise ValueError("--tokenizer-root is required")
    _require_pass(args.audit_root / "structural_audit.json")
    report = audit_dataset(
        args.dataset_root / "samples.jsonl",
        args.tokenizer_root,
        args.dataset_root / "tokenizer_audit.json",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "row_count": report["row_count"],
                "checked_pairs": report["pair_contract"]["checked_pair_count"],
            },
            indent=2,
        )
    )


def command_fit_text_development(args: argparse.Namespace) -> None:
    _require_pass(args.audit_root / "structural_audit.json")
    _require_pass(args.dataset_root / "tokenizer_audit.json")
    if (args.audit_root / "FINAL_TEXT_HOLDOUT_OPENED.json").exists():
        raise ValueError("final text holdout is already open")
    rows, _ = _load_and_validate_dataset(args.dataset_root)
    audit = fit_development_baselines(select_development_rows(rows))
    report = save_development_audit(
        audit,
        model_path=args.audit_root / "development_models.joblib",
        report_path=args.audit_root / "development_analysis.json",
    )
    print(
        json.dumps(
            {
                "development_lock_sha256": report["development_lock_sha256"],
                "model_bundle_sha256": report["model_bundle_sha256"],
            },
            indent=2,
        )
    )


def command_evaluate_text_final(args: argparse.Namespace) -> None:
    marker = args.audit_root / "FINAL_TEXT_HOLDOUT_OPENED.json"
    output = args.audit_root / "final_text_analysis.json"
    if marker.exists():
        raise ValueError("final text holdout was already opened; rerun prohibited")
    if output.exists():
        raise ValueError("final text output exists without its marker; manual recovery required")
    audit = load_development_audit(
        model_path=args.audit_root / "development_models.joblib",
        report_path=args.audit_root / "development_analysis.json",
    )
    atomic_json(
        marker,
        {
            "schema_version": "glm53_v11_final_text_holdout_open_v1",
            "opened_once": True,
            "status": "opening",
            "development_lock_sha256": audit.report["development_lock_sha256"],
            "samples_sha256": sha256_file(args.dataset_root / "samples.jsonl"),
        },
    )
    rows, _ = _load_and_validate_dataset(args.dataset_root)
    report = evaluate_final_holdout(audit, select_final_holdout_rows(rows))
    atomic_json(output, report)
    atomic_json(
        marker,
        {
            "schema_version": "glm53_v11_final_text_holdout_open_v1",
            "opened_once": True,
            "status": "complete",
            "final_analysis_sha256": sha256_file(output),
            "development_lock_sha256": report["development_lock_sha256"],
            "samples_sha256": sha256_file(args.dataset_root / "samples.jsonl"),
        },
    )
    print(json.dumps(report, indent=2))


def _lexical_gate_checks(args: argparse.Namespace) -> dict[str, bool]:
    _require_completed_final_text(args)
    structural = _require_pass(args.audit_root / "structural_audit.json")
    tokenizer = _require_pass(args.dataset_root / "tokenizer_audit.json")
    development = json.loads(
        (args.audit_root / "development_analysis.json").read_text(encoding="utf-8")
    )
    final = json.loads((args.audit_root / "final_text_analysis.json").read_text(encoding="utf-8"))
    ordinary = development["baselines"]
    heldout = final["baselines"]
    return {
        "structural": structural["passed"],
        "tokenizer": tokenizer["passed"],
        "metadata_ordinary_le_060": ordinary["structural_metadata"]["ordinary_test"]["auroc"]
        <= 0.60,
        "metadata_final_le_060": heldout["structural_metadata"]["metrics"]["auroc"] <= 0.60,
        "keyword_final_le_060": heldout["frozen_keyword"]["metrics"]["auroc"] <= 0.60,
        "word_final_le_065": heldout["word_tfidf"]["metrics"]["auroc"] <= 0.65,
        "char_final_le_065": heldout["char_3_5gram"]["metrics"]["auroc"] <= 0.65,
        "deleted_word_final_le_060": heldout["decisive_deleted_word_tfidf"]["metrics"]["auroc"]
        <= 0.60,
        "deleted_char_final_le_060": heldout["decisive_deleted_char_3_5gram"]["metrics"]["auroc"]
        <= 0.60,
    }


def command_decide_lexical(args: argparse.Namespace) -> None:
    checks = _lexical_gate_checks(args)
    passed = all(checks.values())
    decision = {
        "schema_version": "glm53_v11_lexical_gate_decision_v1",
        "passed": passed,
        "decision": (
            "lexical_baselines_pass_semantic_review_unlocked"
            if passed
            else "lexical_shortcut_detected_stop_before_judge_and_glm"
        ),
        "checks": checks,
        "authorization": {
            "semantic_judge": passed,
            "manual_packet": passed,
            "new_glm_forwards": False,
            "runpod_compute": False,
        },
        "inputs": {
            "development": sha256_file(args.audit_root / "development_analysis.json"),
            "final_text": sha256_file(args.audit_root / "final_text_analysis.json"),
            "samples": sha256_file(args.dataset_root / "samples.jsonl"),
            "tokenizer_audit": sha256_file(args.dataset_root / "tokenizer_audit.json"),
        },
    }
    atomic_json(args.audit_root / "lexical_decision.json", decision)
    print(json.dumps(decision, indent=2))


def command_build_manual_packet(args: argparse.Namespace) -> None:
    lexical = _require_pass(args.audit_root / "lexical_decision.json")
    if lexical["decision"] != "lexical_baselines_pass_semantic_review_unlocked":
        raise ValueError("lexical gate does not unlock the manual packet")
    _require_completed_final_text(args)
    rows, _ = _load_and_validate_dataset(args.dataset_root)
    manifest, _ = build_manual_packet(
        rows,
        packet_path=args.audit_root / "manual_packet.csv",
        lock_path=args.audit_root / "manual_packet_lock.json",
    )
    atomic_json(args.audit_root / "manual_packet_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def command_validate_manual(args: argparse.Namespace) -> None:
    if args.completed_manual_audit is None:
        raise ValueError("--completed-manual-audit is required")
    canonical = args.audit_root / "manual_completed.csv"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if args.completed_manual_audit.resolve() != canonical.resolve():
        shutil.copy2(args.completed_manual_audit, canonical)
    report = validate_completed_manual_audit(
        canonical,
        args.audit_root / "manual_packet_lock.json",
    )
    atomic_json(args.audit_root / "manual_audit.json", report)
    print(json.dumps(report, indent=2))


def _human_review_root(args: argparse.Namespace) -> Path:
    return args.review_root or args.audit_root / "human_review"


def _single_packet_kind(args: argparse.Namespace) -> str:
    if args.packet_kind not in {"primary", "supplemental"}:
        raise ValueError("--packet-kind must be primary or supplemental for this command")
    return str(args.packet_kind)


def command_prepare_human_review(args: argparse.Namespace) -> None:
    if not args.reviewer_1_id or not args.reviewer_2_id:
        raise ValueError("--reviewer-1-id and --reviewer-2-id are required")
    kinds = (
        ("primary", "supplemental")
        if args.packet_kind == "all"
        else (str(args.packet_kind),)
    )
    reports = {
        kind: prepare_review_assignment(
            audit_root=args.audit_root,
            review_root=_human_review_root(args),
            kind=kind,
            reviewer_1_id=args.reviewer_1_id,
            reviewer_2_id=args.reviewer_2_id,
        )
        for kind in kinds
    }
    print(json.dumps(reports, indent=2, sort_keys=True))


def command_merge_human_reviews(args: argparse.Namespace) -> None:
    kind = _single_packet_kind(args)
    if args.reviewer_1_completed is None or args.reviewer_2_completed is None:
        raise ValueError(
            "--reviewer-1-completed and --reviewer-2-completed are required"
        )
    report = merge_independent_reviews(
        audit_root=args.audit_root,
        review_root=_human_review_root(args),
        kind=kind,
        reviewer_1_completed=args.reviewer_1_completed,
        reviewer_2_completed=args.reviewer_2_completed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def command_merge_human_adjudication(args: argparse.Namespace) -> None:
    kind = _single_packet_kind(args)
    if args.completed_adjudication is None or not args.adjudicator_id:
        raise ValueError(
            "--completed-adjudication and --adjudicator-id are required"
        )
    report = merge_adjudication(
        audit_root=args.audit_root,
        review_root=_human_review_root(args),
        kind=kind,
        completed_adjudication=args.completed_adjudication,
        adjudicator_id=args.adjudicator_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def command_semantic_judge(args: argparse.Namespace) -> None:
    lexical = _require_pass(args.audit_root / "lexical_decision.json")
    if lexical["decision"] != "lexical_baselines_pass_semantic_review_unlocked":
        raise ValueError("lexical gate does not unlock semantic judging")
    _require_completed_final_text(args)
    rows, _ = _load_and_validate_dataset(args.dataset_root)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    semantic_config = load_prereg(args.prereg)["offline_only_gate"]["semantic_validation"]
    key = os.environ.get("OPENROUTER_API_KEY", "")
    results = asyncio.run(
        run_semantic_judge(
            rows,
            output_root=args.audit_root / "semantic_judge",
            api_key=key,
            model=str(semantic_config["judge_model"]),
            max_tokens=int(semantic_config["max_tokens"]),
            concurrency=args.concurrency,
            spend_cap_usd=float(semantic_config["semantic_judge_api_spend_cap_usd"]),
        )
    )
    print(json.dumps({"completed_rows": len(results)}, indent=2))


def command_analyze_semantic(args: argparse.Namespace) -> None:
    lexical = _require_pass(args.audit_root / "lexical_decision.json")
    if lexical["decision"] != "lexical_baselines_pass_semantic_review_unlocked":
        raise ValueError("lexical gate does not unlock semantic analysis")
    _require_completed_final_text(args)
    rows, _ = _load_and_validate_dataset(args.dataset_root)
    judgments = load_judgment_rows(args.audit_root / "semantic_judge")
    report = analyze_semantic_judgments(rows, judgments)
    atomic_json(args.audit_root / "semantic_validation.json", report)
    print(json.dumps(report, indent=2))


def command_build_offline_analysis(args: argparse.Namespace) -> None:
    report = build_offline_analysis(
        repo_root=ROOT,
        prereg_path=args.prereg,
        dataset_root=args.dataset_root,
        audit_root=args.audit_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def command_verify_offline(args: argparse.Namespace) -> None:
    report = verify_offline_gate(
        dataset_root=args.dataset_root,
        audit_root=args.audit_root,
        prereg_path=args.prereg,
    )
    atomic_json(args.audit_root / "verification.json", report)
    print(json.dumps(report, indent=2))


def _text_gate_checks(args: argparse.Namespace) -> dict[str, bool]:
    lexical = _load_terminal_gate_artifact(args.audit_root / "lexical_decision.json")
    semantic = _load_terminal_gate_artifact(args.audit_root / "semantic_validation.json")
    manual = _load_terminal_gate_artifact(args.audit_root / "manual_audit.json")
    verification = _load_terminal_gate_artifact(args.audit_root / "verification.json")
    lexical_checks = lexical.get("checks")
    if not isinstance(lexical_checks, dict) or not all(
        isinstance(value, bool) for value in lexical_checks.values()
    ):
        raise ValueError("lexical decision checks must be a Boolean mapping")
    return dict(lexical_checks) | {
        "lexical_decision": lexical["passed"],
        "semantic": semantic["passed"],
        "manual": manual["passed"],
        "independent_verification": verification["passed"],
    }


def _require_frozen_paid_paths(args: argparse.Namespace) -> None:
    expected = {
        "prereg": DEFAULT_PREREG,
        "dataset_root": DEFAULT_DATASET_ROOT,
        "audit_root": DEFAULT_AUDIT_ROOT,
        "runtime_config": DEFAULT_RUNTIME,
        "feature_root": DEFAULT_FEATURE_ROOT,
        "source_root": DEFAULT_SOURCE_ROOT,
        "downstream_manifest": DEFAULT_DOWNSTREAM_MANIFEST,
        "downstream_root": DEFAULT_DOWNSTREAM_ROOT,
    }
    for field, frozen in expected.items():
        supplied = Path(getattr(args, field))
        if supplied.resolve() != frozen.resolve():
            raise ValueError(f"paid v11 path override is forbidden: {field}")


def _text_decision_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Return the complete, frozen input set for the offline text decision."""
    return {
        "prereg": args.prereg,
        "samples": args.dataset_root / "samples.jsonl",
        "dataset_manifest": args.dataset_root / "manifest.json",
        "tokenizer_audit": args.dataset_root / "tokenizer_audit.json",
        "builder": ROOT / "src/glm53_user_eval/v11/builder.py",
        "spec": ROOT / "src/glm53_user_eval/v11/spec.py",
        "runtime_config": args.runtime_config,
        "structural": args.audit_root / "structural_audit.json",
        "development": args.audit_root / "development_analysis.json",
        "final_text": args.audit_root / "final_text_analysis.json",
        "final_text_marker": args.audit_root / "FINAL_TEXT_HOLDOUT_OPENED.json",
        "lexical_decision": args.audit_root / "lexical_decision.json",
        "semantic": args.audit_root / "semantic_validation.json",
        "manual": args.audit_root / "manual_audit.json",
        "verification": args.audit_root / "verification.json",
    }


def command_decide_text(args: argparse.Namespace) -> None:
    _require_frozen_paid_paths(args)
    checks = _text_gate_checks(args)
    passed = all(checks.values())
    failed_checks = sorted(name for name, value in checks.items() if value is not True)
    authorization = {
        "new_glm_forwards": passed,
        "runpod_compute": passed,
        "source_activation_extraction": passed,
        "user_recruitment": False,
        "steering": False,
    }
    if not passed and any(authorization.values()):
        raise AssertionError("failed text gate cannot authorize paid work")
    decision = {
        "schema_version": "glm53_v11_text_gate_decision_v1",
        "project_id": "glm53_user_eval_source_instrument_v11",
        "passed": passed,
        "decision": (
            "source_text_instrument_valid_for_activation_test"
            if passed
            else "source_text_instrument_invalid_stop_before_glm"
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "authorization": authorization,
        "inputs": {
            name: sha256_file(path)
            for name, path in _text_decision_input_paths(args).items()
        },
    }
    atomic_json(args.audit_root / "decision.json", decision)
    print(json.dumps(decision, indent=2))


def _require_paid_git_lock() -> str:
    tag = "glm53-user-eval-v11-preregistered"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tag_commit = subprocess.check_output(
        ["git", "rev-list", "-n", "1", tag], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != tag_commit:
        raise ValueError(f"paid extraction HEAD {head} differs from {tag} at {tag_commit}")
    if status:
        raise ValueError("paid extraction requires a clean worktree")
    return head


def _verify_text_decision_bindings(args: argparse.Namespace, decision: dict[str, Any]) -> None:
    if (
        decision.get("schema_version") != "glm53_v11_text_gate_decision_v1"
        or decision.get("project_id") != "glm53_user_eval_source_instrument_v11"
        or decision.get("decision") != "source_text_instrument_valid_for_activation_test"
        or decision.get("passed") is not True
    ):
        raise ValueError("offline text decision does not identify a passing v11 text gate")
    for field in ("new_glm_forwards", "runpod_compute", "source_activation_extraction"):
        if decision.get("authorization", {}).get(field) is not True:
            raise ValueError(f"offline text decision does not authorize {field}")
    expected_checks = _text_gate_checks(args)
    if decision.get("checks") != expected_checks or not all(expected_checks.values()):
        raise ValueError("offline text decision contains a failed or malformed check")
    expected_paths = _text_decision_input_paths(args)
    inputs = decision.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("offline text decision lacks input hashes")
    required_inputs = set(expected_paths)
    if set(inputs) != required_inputs or not all(
        isinstance(value, str) and len(value) == 64 for value in inputs.values()
    ):
        raise ValueError("offline text decision input-hash set differs from the frozen contract")
    for name, path in expected_paths.items():
        expected = inputs.get(name)
        actual = sha256_file(path)
        if not isinstance(expected, str) or len(expected) != 64 or expected != actual:
            raise ValueError(f"offline text decision binding differs for {name}")


def _paid_deadline() -> dt.datetime:
    raw = os.environ.get("GLM53_V11_DEADLINE_UTC", "").strip()
    if not raw:
        raise ValueError("paid extraction requires GLM53_V11_DEADLINE_UTC")
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        raise ValueError("GLM53_V11_DEADLINE_UTC must include a timezone")
    deadline = parsed.astimezone(dt.UTC)
    if deadline <= dt.datetime.now(dt.UTC):
        raise ValueError("paid extraction deadline has already passed")
    return deadline


def _benchmark_row_indices(
    rows: list[dict[str, Any]],
    token_audit: dict[str, Any],
    count: int,
) -> list[int]:
    token_count = {
        str(record["sample_id"]): int(record["rendered_token_count"])
        for record in token_audit["records"]
    }
    eligible = [index for index, row in enumerate(rows) if row.get("label") in {0, 1}]
    eligible.sort(key=lambda index: (token_count[str(rows[index]["sample_id"])], index))
    if count < 3 or len(eligible) < count:
        raise ValueError("throughput benchmark row count is invalid")
    positions = [round(offset * (len(eligible) - 1) / (count - 1)) for offset in range(count)]
    selected = [eligible[position] for position in positions]
    if len(set(selected)) != count:
        raise ValueError("throughput benchmark rows are not unique")
    return selected


def command_extract_source(args: argparse.Namespace, *, keep_runtime: bool = False) -> Any:
    if args.model_path is None:
        raise ValueError("--model-path is required")
    if not args.confirm_spend:
        raise ValueError("paid extraction requires --confirm-spend")
    _require_frozen_paid_paths(args)
    prereg_validation = validate_prereg(DEFAULT_PREREG)
    if prereg_validation.get("passed") is not True:
        raise ValueError("paid extraction requires a passing frozen preregistration")
    deadline = _paid_deadline()
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    if not pod_id:
        raise ValueError("paid extraction requires RUNPOD_POD_ID")
    text_decision = _require_pass(args.audit_root / "decision.json")
    if text_decision["decision"] != "source_text_instrument_valid_for_activation_test":
        raise ValueError("offline text decision does not unlock source extraction")
    _verify_text_decision_bindings(args, text_decision)
    git_commit = _require_paid_git_lock()
    runtime_config = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    from src.glm53_user_eval.v8.whitebox_runtime import verify_model_snapshot
    from src.glm53_user_eval.v11.extraction import extract_source_features
    from src.glm53_user_eval.v11.runtime import LoadedV11GLM53

    stage_manifest = json.loads(
        (ROOT / "artifacts/glm53_user_eval/runtime/g2/model_stage.json").read_text(encoding="utf-8")
    )
    snapshot = verify_model_snapshot(args.model_path, stage_manifest, full_rehash=True)
    if not snapshot["all_shards_match"]:
        raise ValueError("exact official FP8 snapshot did not verify")
    static_checks = {
        name: (
            (args.model_path / name).is_file()
            and sha256_file(args.model_path / name) == expected_sha256
        )
        for name, expected_sha256 in runtime_config["model"]["static_file_sha256"].items()
    }
    if not all(static_checks.values()):
        raise ValueError(f"exact model static-file contract failed: {static_checks}")
    rows, _ = _load_and_validate_dataset(args.dataset_root)
    token_path = args.dataset_root / "tokenizer_audit.json"
    token_audit = _require_pass(token_path)
    runtime = LoadedV11GLM53(model_path=args.model_path, config=runtime_config)
    process_started_utc = dt.datetime.now(dt.UTC).isoformat()
    source_hashes = {
        "dataset_sha256": sha256_file(args.dataset_root / "samples.jsonl"),
        "dataset_manifest_sha256": sha256_file(args.dataset_root / "manifest.json"),
        "tokenizer_audit_sha256": sha256_file(token_path),
        "prereg_sha256": sha256_file(args.prereg),
        "runtime_config_sha256": sha256_file(args.runtime_config),
        "builder_sha256": sha256_file(ROOT / "src/glm53_user_eval/v11/builder.py"),
        "spec_sha256": sha256_file(ROOT / "src/glm53_user_eval/v11/spec.py"),
        "text_decision_sha256": sha256_file(args.audit_root / "decision.json"),
        "model_revision": runtime_config["model"]["revision"],
        "paid_process_nonce": hashlib.sha256(os.urandom(32)).hexdigest(),
    }
    try:
        fp8 = runtime.fp8_scale_report()
        if not fp8["passed"]:
            raise ValueError("v11 FP8 scale audit failed")
        token_by_id = {str(record["sample_id"]): record for record in token_audit["records"]}
        diagnostic = runtime.no_op_equivalence(rows[0], token_by_id[str(rows[0]["sample_id"])])
        if not diagnostic["passed"]:
            raise ValueError("v11 no-op runtime equivalence failed")
        benchmark_count = int(runtime_config["throughput_gate"]["benchmark_rows"])
        benchmark_indices = _benchmark_row_indices(rows, token_audit, benchmark_count)
        benchmark_started = dt.datetime.now(dt.UTC)
        for index in benchmark_indices:
            feature = runtime.extract(rows[index], token_by_id[str(rows[index]["sample_id"])])
            for name in (
                "shared_task_suffix_mean",
                "prompt_final",
                "masked_prompt_mean",
                "decisive_fact_token_mean",
            ):
                if not np.isfinite(getattr(feature, name)).all():
                    raise ValueError(f"throughput diagnostic produced nonfinite {name}")
        benchmark_seconds = (dt.datetime.now(dt.UTC) - benchmark_started).total_seconds()
        prompts_per_second = benchmark_count / benchmark_seconds
        projection = (
            len(rows)
            / prompts_per_second
            * float(runtime_config["throughput_gate"]["projection_headroom_multiplier"])
        )
        reserve = int(runtime_config["throughput_gate"]["backup_reserve_seconds"])
        remaining = (deadline - dt.datetime.now(dt.UTC)).total_seconds()
        throughput_passed = projection + reserve <= remaining
        runtime_gate = {
            "schema_version": "glm53_v11_runtime_throughput_gate_v1",
            "passed": throughput_passed,
            "pod_id": pod_id,
            "deadline_utc": deadline.isoformat(),
            "fp8": fp8,
            "no_op_equivalence": diagnostic,
            "benchmark_rows": benchmark_count,
            "benchmark_seconds": benchmark_seconds,
            "prompts_per_second": prompts_per_second,
            "projected_source_seconds_with_headroom": projection,
            "backup_reserve_seconds": reserve,
            "remaining_seconds_at_decision": remaining,
            "load_seconds": runtime.load_seconds,
            "process_id_diagnostic_only": os.getpid(),
            "process_started_utc_diagnostic_only": process_started_utc,
        }
        runtime_gate_path = args.feature_root / "runtime_throughput_gate.json"
        atomic_json(runtime_gate_path, runtime_gate)
        if not throughput_passed:
            raise ValueError("v11 source extraction does not fit the frozen deadline")
        manifest = extract_source_features(
            runtime,
            rows,
            token_audit,
            output_root=args.feature_root,
            source_hashes=source_hashes,
        )
        if (deadline - dt.datetime.now(dt.UTC)).total_seconds() < reserve:
            raise ValueError("v11 extraction consumed the backup reserve")
        load_seconds = runtime.load_seconds
    finally:
        if not keep_runtime:
            runtime.close()
    report = {
        "schema_version": "glm53_v11_source_extraction_decision_v1",
        "passed": manifest["passed"],
        "git_commit": git_commit,
        "snapshot": snapshot,
        "static_file_checks": static_checks,
        "fp8_scale_report": fp8,
        "load_seconds": load_seconds,
        "feature_manifest_sha256": sha256_file(args.feature_root / "feature_manifest.json"),
        "runtime_throughput_gate_sha256": sha256_file(
            args.feature_root / "runtime_throughput_gate.json"
        ),
    }
    atomic_json(args.feature_root / "extraction_decision.json", report)
    print(json.dumps(report, indent=2))
    return runtime if keep_runtime else None


def command_fit_source_development(args: argparse.Namespace) -> None:
    from src.glm53_user_eval.v11.probes import (
        fit_source_development,
        load_partition,
        save_development_fit,
    )

    _require_pass(args.feature_root / "extraction_decision.json")
    if (args.source_root / "FINAL_SOURCE_HOLDOUT_OPENED.json").exists():
        raise ValueError("final source holdout is already open")
    features, metadata = load_partition(args.feature_root, "development")
    fit = fit_source_development(features, metadata)
    report = save_development_fit(fit, args.source_root)
    print(
        json.dumps(
            {
                "selected_layer": report["selected_layer"],
                "selected_C": report["selected_C"],
                "objective": report["objective"],
            },
            indent=2,
        )
    )


def _run_source_permutation_chunk(
    args: argparse.Namespace,
    *,
    workers: int,
    max_new_repetitions: int | None,
) -> dict[str, Any]:
    from src.glm53_user_eval.v11.probes import (
        load_development_fit,
        load_partition,
        run_full_selection_permutations,
    )

    if (args.source_root / "FINAL_SOURCE_HOLDOUT_OPENED.json").exists():
        raise ValueError("permutations must finish before final source holdout opens")
    fit = load_development_fit(args.source_root)
    features, metadata = load_partition(args.feature_root, "development")
    report = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=fit.objective,
        reps=args.permutation_reps,
        checkpoint_path=args.source_root / "permutation_rows.jsonl",
        checkpoint_binding={
            "feature_manifest_sha256": sha256_file(args.feature_root / "feature_manifest.json"),
            "readout_lock_sha256": sha256_file(args.source_root / "source_readout_lock.json"),
            "config_sha256": sha256_file(args.prereg),
        },
        workers=workers,
        max_new_repetitions=max_new_repetitions,
    )
    atomic_json(args.source_root / "permutation_analysis.json", report)
    return report


def command_source_permutations(args: argparse.Namespace) -> None:
    report = _run_source_permutation_chunk(
        args,
        workers=args.permutation_workers,
        max_new_repetitions=None,
    )
    print(json.dumps(report, indent=2))


def command_evaluate_source_final(args: argparse.Namespace) -> None:
    from src.glm53_user_eval.v11.probes import (
        evaluate_source_final,
        leave_one_generator_score_gaps,
        load_development_fit,
        load_partition,
    )

    marker = args.source_root / "FINAL_SOURCE_HOLDOUT_OPENED.json"
    output = args.source_root / "source_final_analysis.json"
    if output.exists():
        raise ValueError("completed or partial final source output may not be rerun")
    permutation = json.loads(
        (args.source_root / "permutation_analysis.json").read_text(encoding="utf-8")
    )
    if int(permutation["reps"]) != 1000 or permutation.get("complete") is not True:
        raise ValueError("final source holdout requires all 1000 frozen permutations")
    fit = load_development_fit(args.source_root)
    development_features, development_metadata = load_partition(args.feature_root, "development")
    opening = {
        "schema_version": "glm53_v11_final_source_holdout_open_v1",
        "opened_once": True,
        "status": "opening",
        "readout_lock_sha256": fit.report["readout_lock_sha256"],
        "permutation_sha256": sha256_file(args.source_root / "permutation_analysis.json"),
    }
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != opening:
            raise ValueError("final source resume marker differs from the frozen inputs")
    else:
        atomic_json(marker, opening)
    final_features, final_metadata = load_partition(args.feature_root, "final")
    calibration_features, calibration_metadata = load_partition(args.feature_root, "calibration")
    report = evaluate_source_final(
        fit,
        development_features,
        development_metadata,
        final_features,
        final_metadata,
        calibration_features,
        calibration_metadata,
    )
    leave_one = leave_one_generator_score_gaps(
        fit,
        development_features,
        development_metadata,
        final_features,
        final_metadata,
    )
    report["leave_one_training_generator_score_gaps"] = leave_one
    atomic_json(output, report)
    atomic_json(
        marker,
        {
            "schema_version": "glm53_v11_final_source_holdout_open_v1",
            "opened_once": True,
            "status": "complete",
            "analysis_sha256": sha256_file(output),
            "readout_lock_sha256": fit.report["readout_lock_sha256"],
            "permutation_sha256": sha256_file(args.source_root / "permutation_analysis.json"),
        },
    )
    print(json.dumps(report, indent=2))


def command_decide_source(args: argparse.Namespace) -> None:
    _require_pass(args.audit_root / "decision.json")
    verification = _require_pass(args.source_root / "verification.json")
    analysis_path = args.source_root / "source_final_analysis.json"
    permutation_path = args.source_root / "permutation_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    permutation = json.loads(permutation_path.read_text(encoding="utf-8"))
    decision = decide_source_instrument(
        analysis,
        permutation,
        analysis["leave_one_training_generator_score_gaps"],
    )
    if bool(verification["scientific_gate_would_pass"]) != bool(decision["passed"]):
        raise ValueError("independent verifier and primary source decision disagree")
    decision["inputs"] = {
        "analysis": sha256_file(analysis_path),
        "permutation": sha256_file(permutation_path),
        "readout_lock": sha256_file(args.source_root / "source_readout_lock.json"),
        "feature_manifest": sha256_file(args.feature_root / "feature_manifest.json"),
        "verification": sha256_file(args.source_root / "verification.json"),
    }
    decision["checks"]["independent_verification"] = verification["passed"]
    decision["passed"] = decision["passed"] and verification["passed"]
    if not decision["passed"]:
        decision["decision"] = "stop_before_user_recruitment"
        decision["authorization"]["local_proxy_parity"] = False
        decision["interpretation"] = (
            "The source result or its independent verification did not satisfy the frozen gate. "
            "V11 cannot proceed to local parity or user recruitment."
        )
    atomic_json(args.source_root / "decision.json", decision)
    print(json.dumps(decision, indent=2))


def command_verify_source(args: argparse.Namespace) -> None:
    from src.glm53_user_eval.v11.source_verification import verify_source_result

    report = verify_source_result(
        source_root=args.source_root,
        feature_root=args.feature_root,
        config_path=args.prereg,
    )
    atomic_json(args.source_root / "verification.json", report)
    print(json.dumps(report, indent=2))


def _run_source_gate_in_process(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen pure source-analysis functions without a mutable subprocess."""

    command_fit_source_development(args)
    runtime_config = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    throughput = runtime_config["throughput_gate"]
    worker_candidates = [int(value) for value in throughput["permutation_worker_candidates"]]
    benchmark_reps = int(throughput["permutation_benchmark_reps_per_candidate"])
    if worker_candidates != [16, 32] or benchmark_reps <= 0:
        raise ValueError("paid permutation benchmark contract differs")
    benchmark_reports = [
        _run_source_permutation_chunk(
            args,
            workers=workers,
            max_new_repetitions=benchmark_reps,
        )
        for workers in worker_candidates
    ]
    measured_rates = {
        int(report["workers"]): float(report["optimization"]["fitted_repetitions_per_second"])
        for report in benchmark_reports
    }
    if any(not np.isfinite(rate) or rate <= 0 for rate in measured_rates.values()):
        raise ValueError("paid permutation benchmark did not produce a valid rate")
    selected_workers = max(measured_rates, key=measured_rates.get)
    latest = benchmark_reports[-1]
    remaining_repetitions = int(latest["permutations_remaining"])
    projected_permutation_seconds = remaining_repetitions / measured_rates[selected_workers]
    conditional_seconds = float(throughput["conditional_downstream_planning_seconds"])
    source_analysis_seconds = float(throughput["source_final_analysis_allowance_seconds"])
    multiplier = float(throughput["projection_headroom_multiplier"])
    backup_seconds = int(throughput["backup_reserve_seconds"])
    projected_all_in_seconds = (
        multiplier * (projected_permutation_seconds + conditional_seconds + source_analysis_seconds)
        + backup_seconds
    )
    remaining_seconds = (_paid_deadline() - dt.datetime.now(dt.UTC)).total_seconds()
    resource_checks = {
        "both_worker_benchmarks_completed": all(
            int(report["optimization"]["repetitions_fitted"]) == benchmark_reps
            for report in benchmark_reports
        ),
        "benchmarks_count_toward_frozen_1000": int(latest["permutations_completed"])
        == benchmark_reps * len(worker_candidates),
        "full_ladder_fits_with_headroom": projected_all_in_seconds <= remaining_seconds,
    }
    resource_decision = {
        "schema_version": "glm53_v11_paid_permutation_resource_decision_v1",
        "passed": all(resource_checks.values()),
        "checks": resource_checks,
        "worker_candidates": worker_candidates,
        "benchmark_repetitions_per_candidate": benchmark_reps,
        "measured_repetitions_per_second": measured_rates,
        "selected_workers": selected_workers,
        "completed_repetitions": int(latest["permutations_completed"]),
        "remaining_repetitions": remaining_repetitions,
        "projected_remaining_permutation_seconds": projected_permutation_seconds,
        "conditional_downstream_planning_seconds": conditional_seconds,
        "source_final_analysis_allowance_seconds": source_analysis_seconds,
        "headroom_multiplier": multiplier,
        "backup_reserve_seconds": backup_seconds,
        "projected_all_in_seconds": projected_all_in_seconds,
        "remaining_seconds": remaining_seconds,
    }
    atomic_json(args.source_root / "permutation_resource_decision.json", resource_decision)
    if not resource_decision["passed"]:
        stopped = {
            "schema_version": "glm53_v11_source_resource_stop_v1",
            "passed": False,
            "decision": "source_features_saved_stop_for_insufficient_full_ladder_time",
            "authorization": {
                "local_proxy_parity": False,
                "user_recruitment": False,
                "early_cot": False,
                "steering": False,
            },
            "inputs": {
                "feature_manifest": sha256_file(args.feature_root / "feature_manifest.json"),
                "readout_lock": sha256_file(args.source_root / "source_readout_lock.json"),
                "permutation_resource_decision": sha256_file(
                    args.source_root / "permutation_resource_decision.json"
                ),
            },
        }
        atomic_json(args.source_root / "decision.json", stopped)
        return stopped
    final_permutation = _run_source_permutation_chunk(
        args,
        workers=selected_workers,
        max_new_repetitions=None,
    )
    if final_permutation.get("complete") is not True or int(final_permutation["reps"]) != 1000:
        raise ValueError("paid ladder did not complete all 1000 frozen permutations")
    command_evaluate_source_final(args)
    command_verify_source(args)
    command_decide_source(args)
    return json.loads((args.source_root / "decision.json").read_text(encoding="utf-8"))


def _downstream_binding(args: argparse.Namespace, source_decision: Path) -> dict[str, str]:
    decision = json.loads(source_decision.read_text(encoding="utf-8"))
    decision_inputs = decision.get("inputs")
    if not isinstance(decision_inputs, dict):
        raise TypeError("source decision has no immutable input binding")
    readout_hash = sha256_file(args.source_root / "source_readout_lock.json")
    feature_manifest_hash = sha256_file(args.feature_root / "feature_manifest.json")
    if decision_inputs.get("readout_lock") != readout_hash:
        raise ValueError("source readout lock differs from the source decision input")
    if decision_inputs.get("feature_manifest") != feature_manifest_hash:
        raise ValueError("source feature manifest differs from the source decision input")
    feature_manifest = json.loads(
        (args.feature_root / "feature_manifest.json").read_text(encoding="utf-8")
    )
    process_nonce = str(feature_manifest["source_hashes"]["paid_process_nonce"])
    if len(process_nonce) != 64:
        raise ValueError("source features lack a paid-process nonce")
    return {
        "downstream_manifest": sha256_file(args.downstream_manifest),
        "downstream_preflight": sha256_file(args.downstream_root / "preflight.json"),
        "source_decision": sha256_file(source_decision),
        "source_readout_lock": readout_hash,
        "source_readout_arrays": sha256_file(args.source_root / "source_readout_arrays.npz"),
        "source_feature_manifest": feature_manifest_hash,
        "paid_process_nonce": process_nonce,
    }


def _write_downstream_decision(
    *,
    output: Path,
    schema: str,
    passed: bool,
    pass_state: str,
    fail_state: str,
    checks: dict[str, bool],
    inputs: dict[str, str],
    recruitment_authorized: bool,
) -> dict[str, Any]:
    value = {
        "schema_version": schema,
        "project_id": "glm53_user_eval_source_instrument_v11",
        "passed": passed,
        "scientific_gate_passed": passed,
        "claim_ready": False,
        "decision": pass_state if passed else fail_state,
        "checks": checks,
        "authorization": {
            "user_recruitment": passed and recruitment_authorized,
            "early_cot": False,
            "steering": False,
            "final_claim": False,
        },
        "manual_audit_status": "scientific_decision_complete_manual_audit_pending",
        "inputs": inputs,
    }
    atomic_json(output, value)
    return value


def command_paid_ladder(args: argparse.Namespace) -> None:
    """Load once, decide the source gate, and conditionally run frozen transfer."""

    _require_frozen_paid_paths(args)
    prereg_validation = validate_prereg(DEFAULT_PREREG)
    if prereg_validation.get("passed") is not True:
        raise ValueError("paid ladder requires a passing frozen preregistration")

    from src.glm53_user_eval.v11.downstream import (
        analyze_local_proxy,
        analyze_recruitment,
        build_manual_audit_packet,
        calibrate_downstream_batch,
        downstream_resource_decision,
        extract_recruitment_features,
        load_frozen_source_probe,
        load_manifest,
        score_local_proxy,
        validate_downstream_assets,
        validate_runtime_proxy_token_contract,
    )
    from src.glm53_user_eval.v11.downstream import (
        atomic_jsonl as downstream_atomic_jsonl,
    )
    from src.glm53_user_eval.v11.downstream_manual_review import (
        build_downstream_review_template,
    )
    from src.glm53_user_eval.v11.downstream_verification import (
        verify_proxy,
        verify_recruitment,
    )

    runtime = command_extract_source(args, keep_runtime=True)
    if runtime is None:
        raise RuntimeError("paid ladder failed to retain its exact-checkpoint runtime")
    try:
        source_decision = _run_source_gate_in_process(args)
        required = load_manifest(args.downstream_manifest)["source_gate"]["required_decision"]
        if source_decision.get("passed") is not True or source_decision.get("decision") != required:
            print(json.dumps({"paid_ladder": "stopped_at_source_gate"}, indent=2))
            return
        if source_decision.get("authorization", {}).get("local_proxy_parity") is not True:
            raise ValueError("source decision does not authorize local proxy parity")

        preflight, proxy_rows, user_rows = validate_downstream_assets(
            repo_root=ROOT,
            manifest_path=args.downstream_manifest,
        )
        atomic_json(args.downstream_root / "preflight.json", preflight)
        manifest = load_manifest(args.downstream_manifest)
        packet, audit = build_manual_audit_packet(
            proxy_rows=proxy_rows,
            recruitment_rows=user_rows,
            manifest=manifest,
        )
        downstream_atomic_jsonl(args.downstream_root / "manual_packet.jsonl", packet)
        technical_errors = list(preflight["technical_errors"])
        downstream_atomic_jsonl(args.downstream_root / "technical_errors.jsonl", technical_errors)
        review_template = build_downstream_review_template(
            packet,
            technical_errors,
            manifest=manifest,
        )
        downstream_atomic_jsonl(
            args.downstream_root / "manual_review_template.jsonl", review_template
        )
        audit |= {
            "technical_error_rows": len(technical_errors),
            "technical_error_review_required": bool(technical_errors),
            "human_review_completed": False,
            "final_claim_authorized": False,
            "inputs": {
                "manual_packet": sha256_file(args.downstream_root / "manual_packet.jsonl"),
                "technical_errors": sha256_file(args.downstream_root / "technical_errors.jsonl"),
                "review_template": sha256_file(
                    args.downstream_root / "manual_review_template.jsonl"
                ),
            },
        }
        atomic_json(args.downstream_root / "manual_audit_status.json", audit)
        selected_layer, probe = load_frozen_source_probe(
            source_root=args.source_root,
            feature_root=args.feature_root,
        )
        label_ids = [int(value) for value in preflight["label_ids"]]
        codebook_payload = json.loads(
            (ROOT / manifest["assets"]["proxy_codebooks"]["path"]).read_text(encoding="utf-8")
        )
        token_contract = json.loads(
            (ROOT / manifest["assets"]["proxy_contract"]["path"]).read_text(encoding="utf-8")
        )
        runtime_token_validation = validate_runtime_proxy_token_contract(
            runtime.processor,
            proxy_rows=proxy_rows,
            codebook_payload=codebook_payload,
            contract=token_contract,
        )
        atomic_json(
            args.downstream_root / "runtime_proxy_token_validation.json",
            runtime_token_validation,
        )
        conditional_started = dt.datetime.now(dt.UTC)
        batch_config = manifest["execution"]["batch_calibration"]
        proxy_calibration = calibrate_downstream_batch(
            runtime,
            proxy_rows,
            selected_layer=selected_layer,
            continuation=True,
            allowed_token_ids=label_ids,
            candidate_batch_sizes=list(batch_config["candidate_batch_sizes"]),
            logits_tolerance=float(batch_config["logits_max_error"]),
            activation_tolerance=float(batch_config["activation_max_error"]),
            selected_span=False,
        )
        atomic_json(
            args.downstream_root / "proxy_batch_calibration.json",
            proxy_calibration,
        )
        if not proxy_calibration["passed"]:
            print(json.dumps({"paid_ladder": "stopped_at_proxy_batch_gate"}, indent=2))
            return
        rate = float(os.environ.get("GLM53_V11_AGGREGATE_RATE_USD", "0"))
        resource = downstream_resource_decision(
            proxy_seconds=float(proxy_calibration["selected_batch_seconds"]),
            proxy_benchmark_rows=int(proxy_calibration["selected_batch_rows"]),
            proxy_total_rows=len(proxy_rows),
            recruitment_seconds=0,
            recruitment_benchmark_rows=1,
            recruitment_total_rows=0,
            deadline_utc_seconds=_paid_deadline().timestamp(),
            hourly_rate_usd=rate,
            manifest=manifest,
            benchmark_seconds_spent=float(proxy_calibration["total_calibration_seconds"]),
        )
        atomic_json(args.downstream_root / "proxy_resource_decision.json", resource)
        if not resource["passed"]:
            print(json.dumps({"paid_ladder": "stopped_for_downstream_resources"}, indent=2))
            return

        source_decision_path = args.source_root / "decision.json"
        binding = _downstream_binding(args, source_decision_path)
        proxy_root = args.downstream_root / "local_proxy"
        proxy_run_started = dt.datetime.now(dt.UTC)
        scored = score_local_proxy(
            runtime,
            proxy_rows,
            selected_layer=selected_layer,
            label_ids=label_ids,
            output_root=proxy_root,
            binding=binding,
            checkpoint_rows=int(manifest["execution"]["checkpoint_rows"]),
            batch_size=int(proxy_calibration["selected_batch_size"]),
        )
        proxy_run_seconds = (dt.datetime.now(dt.UTC) - proxy_run_started).total_seconds()
        atomic_json(
            proxy_root / "execution.json",
            {
                "schema_version": "glm53_v11_proxy_execution_v1",
                "batch_size": int(proxy_calibration["selected_batch_size"]),
                "row_count": len(scored),
                "seconds": proxy_run_seconds,
            },
        )
        proxy_analysis = analyze_local_proxy(scored, manifest)
        proxy_analysis_path = proxy_root / "analysis.json"
        atomic_json(proxy_analysis_path, proxy_analysis)
        proxy_verification = verify_proxy(
            raw_scores_path=proxy_root / "raw_scores.jsonl",
            analysis_path=proxy_analysis_path,
            manifest=manifest,
            label_ids=label_ids,
            source_binding=binding,
            source_decision_path=source_decision_path,
            source_root=args.source_root,
            source_feature_root=args.feature_root,
            downstream_manifest_path=args.downstream_manifest,
            downstream_preflight_path=args.downstream_root / "preflight.json",
        )
        proxy_verification_path = proxy_root / "verification.json"
        atomic_json(proxy_verification_path, proxy_verification)
        proxy_pass = bool(proxy_analysis["passed"] and proxy_verification["passed"])
        proxy_decision = _write_downstream_decision(
            output=proxy_root / "decision.json",
            schema="glm53_v11_local_proxy_decision_v1",
            passed=proxy_pass,
            pass_state="local_proxy_parity_pass_user_recruitment_unlocked",
            fail_state="local_proxy_mismatch_stop_before_user_recruitment",
            checks=proxy_analysis["checks"]
            | {"independent_verification": proxy_verification["passed"]},
            inputs=binding
            | {
                "raw_scores": sha256_file(proxy_root / "raw_scores.jsonl"),
                "analysis": sha256_file(proxy_analysis_path),
                "verification": sha256_file(proxy_verification_path),
            },
            recruitment_authorized=True,
        )
        if not proxy_decision["authorization"]["user_recruitment"]:
            print(json.dumps({"paid_ladder": "stopped_at_local_proxy_gate"}, indent=2))
            return

        recruitment_calibration = calibrate_downstream_batch(
            runtime,
            user_rows,
            selected_layer=selected_layer,
            continuation=False,
            allowed_token_ids=None,
            candidate_batch_sizes=list(batch_config["candidate_batch_sizes"]),
            logits_tolerance=float(batch_config["logits_max_error"]),
            activation_tolerance=float(batch_config["activation_max_error"]),
            selected_span=True,
        )
        atomic_json(
            args.downstream_root / "recruitment_batch_calibration.json",
            recruitment_calibration,
        )
        if not recruitment_calibration["passed"]:
            print(json.dumps({"paid_ladder": "stopped_at_recruitment_batch_gate"}, indent=2))
            return
        conditional_elapsed = (dt.datetime.now(dt.UTC) - conditional_started).total_seconds()
        recruitment_resource = downstream_resource_decision(
            proxy_seconds=0,
            proxy_benchmark_rows=1,
            proxy_total_rows=0,
            recruitment_seconds=float(recruitment_calibration["selected_batch_seconds"]),
            recruitment_benchmark_rows=int(recruitment_calibration["selected_batch_rows"]),
            recruitment_total_rows=len(user_rows),
            deadline_utc_seconds=_paid_deadline().timestamp(),
            hourly_rate_usd=rate,
            manifest=manifest,
            prior_conditional_compute_cost_usd=conditional_elapsed / 3600 * rate,
        )
        atomic_json(
            args.downstream_root / "recruitment_resource_decision.json",
            recruitment_resource,
        )
        if not recruitment_resource["passed"]:
            print(json.dumps({"paid_ladder": "stopped_for_recruitment_resources"}, indent=2))
            return

        schedule_path = ROOT / manifest["assets"]["causal_schedule"]["path"]
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        recruitment_root = args.downstream_root / "recruitment"
        task_features, prompt_features, metadata = extract_recruitment_features(
            runtime,
            user_rows,
            selected_layer=selected_layer,
            output_root=recruitment_root,
            binding=binding | {"proxy_decision": sha256_file(proxy_root / "decision.json")},
            checkpoint_rows=int(manifest["execution"]["checkpoint_rows"]),
            batch_size=int(recruitment_calibration["selected_batch_size"]),
        )
        primary = analyze_recruitment(
            metadata,
            task_features,
            probe=probe,
            schedule=schedule,
            manifest=manifest,
            view="neutral_task_mean",
        )
        secondary = analyze_recruitment(
            metadata,
            prompt_features,
            probe=probe,
            schedule=schedule,
            manifest=manifest,
            view="prompt_final",
        )
        analysis = {
            "schema_version": "glm53_v11_recruitment_bundle_v1",
            "primary": primary,
            "secondary_descriptive": secondary,
        }
        primary_path = recruitment_root / "primary_analysis.json"
        atomic_json(primary_path, primary)
        atomic_json(recruitment_root / "analysis.json", analysis)
        verification = verify_recruitment(
            feature_path=recruitment_root / "features.npz",
            metadata_path=recruitment_root / "metadata.jsonl",
            analysis_path=primary_path,
            source_root=args.source_root,
            source_feature_root=args.feature_root,
            schedule_path=schedule_path,
            manifest=manifest,
            source_binding=binding,
            source_decision_path=source_decision_path,
            downstream_manifest_path=args.downstream_manifest,
            downstream_preflight_path=args.downstream_root / "preflight.json",
        )
        verification_path = recruitment_root / "verification.json"
        atomic_json(verification_path, verification)
        recruitment_pass = bool(primary["passed"] and verification["passed"])
        _write_downstream_decision(
            output=recruitment_root / "decision.json",
            schema="glm53_v11_recruitment_decision_v1",
            passed=recruitment_pass,
            pass_state="frozen_eval_readout_recruited_by_ai_specific_user_interaction",
            fail_state="no_validated_eval_readout_recruitment",
            checks=primary["checks"] | {"independent_verification": verification["passed"]},
            inputs=binding
            | {
                "proxy_decision": sha256_file(proxy_root / "decision.json"),
                "features": sha256_file(recruitment_root / "features.npz"),
                "metadata": sha256_file(recruitment_root / "metadata.jsonl"),
                "analysis": sha256_file(primary_path),
                "verification": sha256_file(verification_path),
            },
            recruitment_authorized=False,
        )
        print(
            json.dumps(
                {
                    "paid_ladder": "conditional_downstream_complete",
                    "proxy_passed": proxy_pass,
                    "recruitment_passed": recruitment_pass,
                    "manual_audit": "pending_human_review",
                },
                indent=2,
            )
        )
    finally:
        runtime.close()


COMMANDS = {
    "validate-prereg": command_validate_prereg,
    "plan": command_plan,
    "validate-downstream": command_validate_downstream,
    "authorize-downstream-claim": command_authorize_downstream_claim,
    "build-dataset": command_build_dataset,
    "audit-structure": command_audit_structure,
    "audit-tokenizer": command_audit_tokenizer,
    "fit-text-development": command_fit_text_development,
    "evaluate-text-final": command_evaluate_text_final,
    "decide-lexical": command_decide_lexical,
    "build-manual-packet": command_build_manual_packet,
    "validate-manual": command_validate_manual,
    "prepare-human-review": command_prepare_human_review,
    "merge-human-reviews": command_merge_human_reviews,
    "merge-human-adjudication": command_merge_human_adjudication,
    "semantic-judge": command_semantic_judge,
    "analyze-semantic": command_analyze_semantic,
    "build-offline-analysis": command_build_offline_analysis,
    "verify-offline": command_verify_offline,
    "decide-text": command_decide_text,
    "extract-source": command_extract_source,
    "fit-source-development": command_fit_source_development,
    "source-permutations": command_source_permutations,
    "evaluate-source-final": command_evaluate_source_final,
    "verify-source": command_verify_source,
    "decide-source": command_decide_source,
    "paid-ladder": command_paid_ladder,
}


def main() -> None:
    args = _build_parser().parse_args()
    if args.command not in COMMANDS:
        raise ValueError(f"unknown command {args.command}; choose from {sorted(COMMANDS)}")
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
