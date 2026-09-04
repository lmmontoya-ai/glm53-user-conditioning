"""Command line entry point for GLM-5.3 V16."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v16.contract import (
    MODEL_REVISION,
    canonical_sha256,
    load_yaml,
    sha256_file,
    validate_parent,
    validate_prereg,
)
from src.glm53_user_eval.v16.dataset import load_rows
from src.glm53_user_eval.v16.extraction import (
    atomic_json,
    extract_source_features,
    load_partition,
)

DEFAULT_PREREG = ROOT / "pipelines/glm53_user_eval/v16/configs/prereg_v16_source_activation.yaml"
DEFAULT_RUNTIME = ROOT / "pipelines/glm53_user_eval/v16/configs/runtime_v16.yaml"
DEFAULT_DOWNSTREAM = ROOT / "pipelines/glm53_user_eval/v16/configs/downstream_manifest_v16.json"
DEFAULT_TOKEN_AUDIT = ROOT / "artifacts/datasets/contrastive_prompts_v5/tokenizer_audit_v16.json"
DEFAULT_DATASET = ROOT / "artifacts/datasets/contrastive_prompts_v5/samples.jsonl"
DEFAULT_FEATURE_ROOT = ROOT / "artifacts/glm53_user_eval/v16/features/source"
DEFAULT_SOURCE_ROOT = ROOT / "artifacts/glm53_user_eval/v16/source_readout"
DEFAULT_DOWNSTREAM_ROOT = ROOT / "artifacts/glm53_user_eval/v16/downstream"
DEFAULT_INFRA_ROOT = ROOT / "artifacts/glm53_user_eval/v16/infrastructure"


def _atomic_json(path: Path, value: Any) -> None:
    atomic_json(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _code_sha256() -> str:
    files = sorted((ROOT / "src/glm53_user_eval/v16").glob("*.py")) + sorted(
        (ROOT / "pipelines/glm53_user_eval/v16").glob("*.py")
    )
    return canonical_sha256(
        [(path.relative_to(ROOT).as_posix(), sha256_file(path)) for path in files]
    )


def _deadline() -> dt.datetime:
    raw = os.environ.get("GLM53_V16_DEADLINE_UTC", "")
    if not raw:
        raise ValueError("GLM53_V16_DEADLINE_UTC is required")
    value = dt.datetime.fromisoformat(raw).astimezone(dt.UTC)
    if value <= dt.datetime.now(dt.UTC):
        raise ValueError("V16 paid deadline has passed")
    return value


def _paid_environment(config: dict[str, Any]) -> dict[str, Any]:
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    rate = float(os.environ.get("GLM53_V16_AGGREGATE_RATE_USD", "0"))
    balance = float(os.environ.get("GLM53_V16_LAUNCH_BALANCE_USD", "0"))
    balance_floor = float(os.environ.get("GLM53_V16_BALANCE_FLOOR_USD", "0"))
    effective_cap = balance - balance_floor
    checks = {
        "pod_id": bool(pod_id),
        "rate_positive": rate > 0,
        "rate_cap": rate <= float(config["runpod"]["aggregate_gpu_rate_cap_usd_per_hour"]),
        "balance": balance > 15.10,
        "balance_floor": balance_floor >= 15.0,
        "effective_cap": 0 < effective_cap <= 25.0,
        "deadline": _deadline() > dt.datetime.now(dt.UTC),
        "s3_transport_verified": os.environ.get("GLM53_V16_S3_TRANSPORT_VERIFIED") == "1",
        "science_process_has_no_s3_secret": not any(
            os.environ.get(name) for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"paid environment failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "pod_id": pod_id,
        "hourly_rate_usd": rate,
        "launch_balance_usd": balance,
        "effective_compute_cap_usd": effective_cap,
        "balance_floor_usd": balance_floor,
        "deadline_utc": _deadline().isoformat(),
    }


def _require_preregistered_commit() -> str:
    head = _git("rev-parse", "HEAD")
    tag_commit = _git("rev-list", "-n", "1", "glm53-user-eval-v16-preregistered-a12")
    changed_tracked = {
        line for line in _git("diff", "--name-only", "HEAD", "--").splitlines() if line
    }
    allowed_signed_inputs = {
        "artifacts/datasets/contrastive_prompts_v5/manifest.json",
        "artifacts/datasets/contrastive_prompts_v5/samples.jsonl",
        "artifacts/datasets/contrastive_prompts_v5/tokenizer_audit_v16.json",
        "artifacts/glm53_user_eval/reports/transluce_interaction_v7/analysis.json",
        "artifacts/glm53_user_eval/reports/transluce_interaction_v7/decision.json",
        "artifacts/glm53_user_eval/reports/transluce_interaction_v7/final_evidence.json",
        "artifacts/glm53_user_eval/runtime/g2/model_stage.json",
        "artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json",
        "artifacts/glm53_user_eval/v11/downstream_inputs/preflight.json",
        "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl",
        "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100_manifest.json",
        "artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json",
        "artifacts/glm53_user_eval/v15/reports/codex_cohort/verification.json",
        "pipelines/glm53_user_eval/configs/identity_selection_v1.json",
        "pipelines/glm53_user_eval/v11/configs/parent_proxy_surface_v1.json",
        "pipelines/glm53_user_eval/v11/configs/proxy_codebooks_v2.json",
        "pipelines/glm53_user_eval/v11/configs/proxy_token_contract_v2.json",
        "pipelines/glm53_user_eval/v16/configs/downstream_manifest_v16.json",
        "pipelines/glm53_user_eval/v8/configs/causal_schedule_v1.json",
        "pipelines/glm53_user_eval/v8/configs/user_prompt_templates_v1.jsonl",
    }
    untracked = {
        line
        for line in _git("ls-files", "--others", "--exclude-standard").splitlines()
        if line
    }
    if (
        head != tag_commit
        or not changed_tracked.issubset(allowed_signed_inputs)
        or not untracked.issubset(allowed_signed_inputs)
    ):
        raise ValueError("paid V16 must run from the clean preregistered commit")
    return head


def command_audit_tokenizer(args: argparse.Namespace) -> None:
    from src.glm53_user_eval.v16.tokenizer_audit import write_v16_tokenizer_audit

    tokenizer_root = args.tokenizer_root or Path(
        os.environ.get(
            "GLM53_TOKENIZER_ROOT",
            str(
                Path.home()
                / ".cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots"
                / MODEL_REVISION
            ),
        )
    )
    report = write_v16_tokenizer_audit(
        samples_path=DEFAULT_DATASET,
        manifest_path=ROOT / "artifacts/datasets/contrastive_prompts_v5/manifest.json",
        tokenizer_root=tokenizer_root,
        v4_path=ROOT / "artifacts/datasets/contrastive_prompts_v4/samples.jsonl",
        output_path=DEFAULT_TOKEN_AUDIT,
    )
    print(json.dumps({"passed": report["passed"], "rows": report["row_count"], "sha256": sha256_file(DEFAULT_TOKEN_AUDIT)}, indent=2))


def command_validate_prereg(args: argparse.Namespace) -> None:
    print(json.dumps(validate_prereg(ROOT, args.prereg), indent=2))


def command_plan_paid(args: argparse.Namespace) -> None:
    validation = validate_prereg(ROOT, args.prereg)
    parent = validate_parent(ROOT)
    rows = load_rows(DEFAULT_DATASET)
    downstream = _read_json(DEFAULT_DOWNSTREAM)
    plan = {
        "schema_version": "glm53_v16_paid_plan_v1",
        "passed": True,
        "parent": parent,
        "prereg_sha256": validation["prereg_sha256"],
        "rows": len(rows),
        "partitions": {"development": 368, "final_binary": 112, "factorial": 32, "fresh_controls": 64},
        "hardware": {"gpu_type": "NVIDIA B300 SXM6 AC", "gpu_count": 2, "fallback": False},
        "budget": {"compute_cap_usd": 25.0, "reserve_usd": 15.0, "storage_allowance_usd": 0.10},
        "conditional_rows": {
            "local_proxy": downstream["local_proxy"]["expected_eligible_rows"],
            "recruitment": downstream["recruitment"]["expected_rows"],
        },
        "forbidden": ["early_cot", "steering", "dataset_repair", "gpu_fallback"],
    }
    _atomic_json(DEFAULT_INFRA_ROOT / "paid_plan.json", plan)
    print(json.dumps(plan, indent=2))


def _source_bindings(args: argparse.Namespace, nonce: str) -> dict[str, str]:
    return {
        "dataset_sha256": sha256_file(DEFAULT_DATASET),
        "dataset_manifest_sha256": sha256_file(
            ROOT / "artifacts/datasets/contrastive_prompts_v5/manifest.json"
        ),
        "tokenizer_audit_sha256": sha256_file(DEFAULT_TOKEN_AUDIT),
        "prereg_sha256": sha256_file(args.prereg),
        "runtime_config_sha256": sha256_file(args.runtime_config),
        "downstream_manifest_sha256": sha256_file(args.downstream_manifest),
        "v15_decision_sha256": sha256_file(
            ROOT / "artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json"
        ),
        "v15_verification_sha256": sha256_file(
            ROOT / "artifacts/glm53_user_eval/v15/reports/codex_cohort/verification.json"
        ),
        "code_sha256": _code_sha256(),
        "model_revision": MODEL_REVISION,
        "paid_process_nonce": nonce,
    }


def _run_source(args: argparse.Namespace, runtime: Any, paid: dict[str, Any]) -> dict[str, Any]:
    from src.glm53_user_eval.v16.probes import (
        fit_source_development,
        run_full_selection_permutations,
        save_development_fit,
    )
    from src.glm53_user_eval.v16.source_analysis import evaluate_source_final
    from src.glm53_user_eval.v16.source_decision import decide_source
    from src.glm53_user_eval.v16.verification import verify_source

    rows = load_rows(DEFAULT_DATASET)
    audit = _read_json(DEFAULT_TOKEN_AUDIT)
    nonce = hashlib.sha256(
        f"{paid['pod_id']}|{_git('rev-parse','HEAD')}|{uuid.uuid4()}".encode()
    ).hexdigest()
    bindings = _source_bindings(args, nonce)
    extraction = extract_source_features(
        runtime,
        rows,
        audit,
        output_root=args.feature_root,
        source_hashes=bindings,
    )
    development, development_metadata = load_partition(args.feature_root, "development")
    fit = fit_source_development(development, development_metadata)
    lock = save_development_fit(fit, args.source_root)
    permutation = run_full_selection_permutations(
        development,
        development_metadata,
        observed_objective=fit.objective,
        reps=1000,
        seed=20260925,
        workers=int(os.environ.get("GLM53_V16_PERMUTATION_WORKERS", "32")),
        checkpoint_path=args.source_root / "permutation_rows.jsonl",
    )
    permutation_path = args.source_root / "permutation_analysis.json"
    _atomic_json(permutation_path, permutation)
    marker = args.source_root / "FINAL_SOURCE_HOLDOUT_OPENED.json"
    marker_value = {
        "schema_version": "glm53_v16_final_source_open_v1",
        "readout_lock_sha256": sha256_file(args.source_root / "source_readout_lock.json"),
        "permutation_sha256": sha256_file(permutation_path),
        "feature_manifest_sha256": sha256_file(args.feature_root / "feature_manifest.json"),
    }
    if marker.exists():
        if _read_json(marker) != marker_value:
            raise ValueError("existing final-source marker has different frozen inputs")
        if (args.source_root / "source_final_analysis.json").exists():
            raise ValueError("completed final source analysis may not be rerun")
    else:
        _atomic_json(marker, marker_value)
    final, final_metadata = load_partition(args.feature_root, "final_binary")
    factorial, factorial_metadata = load_partition(args.feature_root, "factorial")
    controls, control_metadata = load_partition(args.feature_root, "fresh_controls")
    analysis = evaluate_source_final(
        fit,
        development,
        development_metadata,
        final,
        final_metadata,
        factorial,
        factorial_metadata,
        controls,
        control_metadata,
        rows,
    )
    analysis_path = args.source_root / "source_final_analysis.json"
    _atomic_json(analysis_path, analysis)
    decision = decide_source(analysis, permutation)
    decision["inputs"] = {
        "analysis": sha256_file(analysis_path),
        "permutation": sha256_file(permutation_path),
        "readout_lock": sha256_file(args.source_root / "source_readout_lock.json"),
        "readout_arrays": sha256_file(args.source_root / "source_readout_arrays.npz"),
        "feature_manifest": sha256_file(args.feature_root / "feature_manifest.json"),
        "final_open_marker": sha256_file(marker),
    }
    decision_path = args.source_root / "decision.json"
    _atomic_json(decision_path, decision)
    verification = verify_source(
        feature_root=args.feature_root,
        source_root=args.source_root,
        source_rows_path=DEFAULT_DATASET,
        analysis_path=analysis_path,
        permutation_path=permutation_path,
        decision_path=decision_path,
        output_path=args.source_root / "verification.json",
    )
    if verification["passed"] is not True:
        decision["passed"] = False
        decision["decision"] = "stop_before_local_parity"
        decision["authorization"]["local_proxy_parity"] = False
    decision["checks"]["independent_verification"] = bool(verification["passed"])
    decision["inputs"]["verification"] = sha256_file(args.source_root / "verification.json")
    _atomic_json(decision_path, decision)
    return {
        "decision": decision,
        "extraction": extraction,
        "readout_lock": lock,
        "verification": verification,
    }


def _downstream_binding(args: argparse.Namespace, nonce: str) -> dict[str, str]:
    return {
        "downstream_manifest": sha256_file(args.downstream_manifest),
        "downstream_preflight": sha256_file(args.downstream_root / "preflight.json"),
        "source_decision": sha256_file(args.source_root / "decision.json"),
        "source_readout_lock": sha256_file(args.source_root / "source_readout_lock.json"),
        "source_readout_arrays": sha256_file(args.source_root / "source_readout_arrays.npz"),
        "source_feature_manifest": sha256_file(args.feature_root / "feature_manifest.json"),
        "paid_process_nonce": nonce,
    }


def _write_downstream_decision(
    path: Path,
    *,
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
        "passed": passed,
        "decision": pass_state if passed else fail_state,
        "checks": checks,
        "inputs": inputs,
        "authorization": {
            "prompt_recruitment": bool(passed and recruitment_authorized),
            "first_cot_transfer": False,
            "steering": False,
        },
    }
    _atomic_json(path, value)
    return value


def _run_downstream(args: argparse.Namespace, runtime: Any, source: dict[str, Any], paid: dict[str, Any]) -> dict[str, Any]:
    from src.glm53_user_eval.v11.downstream import (
        atomic_jsonl,
        load_manifest,
        validate_runtime_proxy_token_contract,
    )
    from src.glm53_user_eval.v11.downstream_manual_review import build_downstream_review_template
    from src.glm53_user_eval.v11.downstream_verification import verify_proxy, verify_recruitment
    from src.glm53_user_eval.v16.downstream import (
        analyze_local_proxy,
        analyze_recruitment,
        build_manual_audit_packet,
        calibrate_downstream_batch,
        downstream_resource_decision,
        extract_recruitment_features,
        load_frozen_source_probe,
        score_local_proxy,
        validate_downstream_assets,
    )

    decision = source["decision"]
    if decision.get("authorization", {}).get("local_proxy_parity") is not True:
        return {"stage": "source", "decision": decision["decision"]}
    preflight, proxy_rows, user_rows = validate_downstream_assets(
        repo_root=ROOT, manifest_path=args.downstream_manifest
    )
    _atomic_json(args.downstream_root / "preflight.json", preflight)
    manifest = load_manifest(args.downstream_manifest)
    packet, manual_status = build_manual_audit_packet(
        proxy_rows=proxy_rows, recruitment_rows=user_rows, manifest=manifest
    )
    atomic_jsonl(args.downstream_root / "manual_packet.jsonl", packet)
    technical_errors = preflight["technical_errors"]
    atomic_jsonl(args.downstream_root / "technical_errors.jsonl", technical_errors)
    review_template = build_downstream_review_template(packet, technical_errors, manifest=manifest)
    atomic_jsonl(args.downstream_root / "manual_review_template.jsonl", review_template)
    manual_status |= {
        "human_review_completed": False,
        "positive_recruitment_claim_authorized": False,
        "technical_error_rows": len(technical_errors),
    }
    _atomic_json(args.downstream_root / "manual_audit_status.json", manual_status)
    selected_layer, probe = load_frozen_source_probe(
        source_root=args.source_root, feature_root=args.feature_root
    )
    label_ids = [int(value) for value in preflight["label_ids"]]
    codebooks = _read_json(ROOT / manifest["assets"]["proxy_codebooks"]["path"])
    token_contract = _read_json(ROOT / manifest["assets"]["proxy_contract"]["path"])
    token_check = validate_runtime_proxy_token_contract(
        runtime.processor,
        proxy_rows=proxy_rows,
        codebook_payload=codebooks,
        contract=token_contract,
    )
    _atomic_json(args.downstream_root / "runtime_proxy_token_validation.json", token_check)
    batch = manifest["execution"]["batch_calibration"]
    calibration = calibrate_downstream_batch(
        runtime,
        proxy_rows,
        selected_layer=selected_layer,
        continuation=True,
        allowed_token_ids=label_ids,
        candidate_batch_sizes=list(batch["candidate_batch_sizes"]),
        logits_tolerance=float(batch["logits_max_error"]),
        activation_tolerance=float(batch["activation_max_error"]),
        selected_span=False,
    )
    _atomic_json(args.downstream_root / "proxy_batch_calibration.json", calibration)
    if not calibration["passed"]:
        return {"stage": "proxy_batch", "decision": "stop_for_numerical_or_memory_failure"}
    resource = downstream_resource_decision(
        proxy_seconds=float(calibration["selected_batch_seconds"]),
        proxy_benchmark_rows=int(calibration["selected_batch_rows"]),
        proxy_total_rows=len(proxy_rows),
        recruitment_seconds=0,
        recruitment_benchmark_rows=1,
        recruitment_total_rows=0,
        deadline_utc_seconds=_deadline().timestamp(),
        hourly_rate_usd=float(paid["hourly_rate_usd"]),
        manifest=manifest,
        benchmark_seconds_spent=float(calibration["total_calibration_seconds"]),
    )
    _atomic_json(args.downstream_root / "proxy_resource_decision.json", resource)
    if not resource["passed"]:
        return {"stage": "proxy_resource", "decision": "stop_for_budget_or_deadline"}
    nonce = _read_json(args.feature_root / "feature_manifest.json")["source_hashes"][
        "paid_process_nonce"
    ]
    binding = _downstream_binding(args, nonce)
    proxy_root = args.downstream_root / "local_proxy"
    scored = score_local_proxy(
        runtime,
        proxy_rows,
        selected_layer=selected_layer,
        label_ids=label_ids,
        output_root=proxy_root,
        binding=binding,
        checkpoint_rows=int(manifest["execution"]["checkpoint_rows"]),
        batch_size=int(calibration["selected_batch_size"]),
    )
    proxy_analysis = analyze_local_proxy(scored, manifest)
    proxy_analysis_path = proxy_root / "analysis.json"
    _atomic_json(proxy_analysis_path, proxy_analysis)
    proxy_verification = verify_proxy(
        raw_scores_path=proxy_root / "raw_scores.jsonl",
        analysis_path=proxy_analysis_path,
        manifest=manifest,
        label_ids=label_ids,
        source_binding=binding,
        source_decision_path=args.source_root / "decision.json",
        source_root=args.source_root,
        source_feature_root=args.feature_root,
        downstream_manifest_path=args.downstream_manifest,
        downstream_preflight_path=args.downstream_root / "preflight.json",
    )
    _atomic_json(proxy_root / "verification.json", proxy_verification)
    proxy_pass = bool(proxy_analysis["passed"] and proxy_verification["passed"])
    proxy_decision = _write_downstream_decision(
        proxy_root / "decision.json",
        schema="glm53_v16_local_proxy_decision_v1",
        passed=proxy_pass,
        pass_state="local_proxy_parity_pass_prompt_recruitment_unlocked",
        fail_state="local_proxy_mismatch_stop_before_prompt_recruitment",
        checks=proxy_analysis["checks"] | {"independent_verification": proxy_verification["passed"]},
        inputs=binding
        | {
            "raw_scores": sha256_file(proxy_root / "raw_scores.jsonl"),
            "analysis": sha256_file(proxy_analysis_path),
            "verification": sha256_file(proxy_root / "verification.json"),
        },
        recruitment_authorized=True,
    )
    if not proxy_pass:
        return {"stage": "local_proxy", "decision": proxy_decision["decision"]}
    recruitment_calibration = calibrate_downstream_batch(
        runtime,
        user_rows,
        selected_layer=selected_layer,
        continuation=False,
        allowed_token_ids=None,
        candidate_batch_sizes=list(batch["candidate_batch_sizes"]),
        logits_tolerance=float(batch["logits_max_error"]),
        activation_tolerance=float(batch["activation_max_error"]),
        selected_span=True,
    )
    _atomic_json(args.downstream_root / "recruitment_batch_calibration.json", recruitment_calibration)
    if not recruitment_calibration["passed"]:
        return {"stage": "recruitment_batch", "decision": "stop_for_numerical_or_memory_failure"}
    recruitment_resource = downstream_resource_decision(
        proxy_seconds=0,
        proxy_benchmark_rows=1,
        proxy_total_rows=0,
        recruitment_seconds=float(recruitment_calibration["selected_batch_seconds"]),
        recruitment_benchmark_rows=int(recruitment_calibration["selected_batch_rows"]),
        recruitment_total_rows=len(user_rows),
        deadline_utc_seconds=_deadline().timestamp(),
        hourly_rate_usd=float(paid["hourly_rate_usd"]),
        manifest=manifest,
    )
    _atomic_json(args.downstream_root / "recruitment_resource_decision.json", recruitment_resource)
    if not recruitment_resource["passed"]:
        return {"stage": "recruitment_resource", "decision": "stop_for_budget_or_deadline"}
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
    schedule_path = ROOT / manifest["assets"]["causal_schedule"]["path"]
    schedule = _read_json(schedule_path)
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
    _atomic_json(recruitment_root / "primary_analysis.json", primary)
    _atomic_json(
        recruitment_root / "analysis.json",
        {
            "schema_version": "glm53_v16_recruitment_bundle_v1",
            "primary": primary,
            "secondary_descriptive": secondary,
        },
    )
    verification = verify_recruitment(
        feature_path=recruitment_root / "features.npz",
        metadata_path=recruitment_root / "metadata.jsonl",
        analysis_path=recruitment_root / "primary_analysis.json",
        source_root=args.source_root,
        source_feature_root=args.feature_root,
        schedule_path=schedule_path,
        manifest=manifest,
        source_binding=binding,
        source_decision_path=args.source_root / "decision.json",
        downstream_manifest_path=args.downstream_manifest,
        downstream_preflight_path=args.downstream_root / "preflight.json",
    )
    _atomic_json(recruitment_root / "verification.json", verification)
    recruitment_pass = bool(primary["passed"] and verification["passed"])
    recruitment_decision = _write_downstream_decision(
        recruitment_root / "decision.json",
        schema="glm53_v16_recruitment_decision_v1",
        passed=recruitment_pass,
        pass_state="frozen_eval_readout_recruited_by_ai_specific_user_interaction",
        fail_state="no_validated_eval_readout_recruitment",
        checks=primary["checks"] | {"independent_verification": verification["passed"]},
        inputs=binding
        | {
            "proxy_decision": sha256_file(proxy_root / "decision.json"),
            "features": sha256_file(recruitment_root / "features.npz"),
            "metadata": sha256_file(recruitment_root / "metadata.jsonl"),
            "analysis": sha256_file(recruitment_root / "primary_analysis.json"),
            "verification": sha256_file(recruitment_root / "verification.json"),
        },
        recruitment_authorized=False,
    )
    return {
        "stage": "recruitment",
        "decision": recruitment_decision["decision"],
        "positive_claim_pending_human_review": recruitment_pass,
    }


def command_paid_supervisor(args: argparse.Namespace) -> None:
    if not args.confirm_spend:
        raise ValueError("paid-supervisor requires --confirm-spend")
    validate_prereg(ROOT, args.prereg)
    commit = _require_preregistered_commit()
    runtime_config = load_yaml(args.runtime_config)
    paid = _paid_environment(runtime_config)
    if sha256_file(args.runtime_config) != load_yaml(args.prereg)["runtime"]["sha256"]:
        raise ValueError("runtime config differs from preregistration")
    from src.glm53_user_eval.v8.whitebox_runtime import verify_model_snapshot
    from src.glm53_user_eval.v16.runtime import LoadedV16GLM53

    stage_manifest = _read_json(ROOT / "artifacts/glm53_user_eval/runtime/g2/model_stage.json")
    snapshot = verify_model_snapshot(args.model_path, stage_manifest, full_rehash=True)
    runtime = LoadedV16GLM53(model_path=args.model_path, config=runtime_config)
    started = time.perf_counter()
    try:
        rows = load_rows(DEFAULT_DATASET)
        token_audit = _read_json(DEFAULT_TOKEN_AUDIT)
        token_by_id = {row["sample_id"]: row for row in token_audit["records"]}
        fp8 = runtime.fp8_scale_report()
        no_op = runtime.no_op_equivalence(rows[0], token_by_id[rows[0]["sample_id"]])
        if fp8["passed"] is not True or no_op["passed"] is not True:
            raise ValueError("exact runtime algebra checks failed")
        benchmark_indices = [round(index * 575 / 31) for index in range(32)]
        benchmark_started = time.perf_counter()
        for index in benchmark_indices:
            row = rows[index]
            runtime.extract(row, token_by_id[row["sample_id"]])
        benchmark_seconds = time.perf_counter() - benchmark_started
        prompts_per_second = 32 / benchmark_seconds
        projected_source_seconds = 1.30 * 576 / prompts_per_second
        remaining = (_deadline() - dt.datetime.now(dt.UTC)).total_seconds()
        projected_all_in = projected_source_seconds + 1800 + 600
        projected_cost = projected_all_in / 3600 * paid["hourly_rate_usd"]
        throughput = {
            "schema_version": "glm53_v16_runtime_gate_v1",
            "passed": projected_all_in <= remaining and projected_cost <= paid["effective_compute_cap_usd"],
            "benchmark_rows": 32,
            "benchmark_seconds": benchmark_seconds,
            "prompts_per_second": prompts_per_second,
            "projected_source_seconds_with_headroom": projected_source_seconds,
            "analysis_allowance_seconds": 1800,
            "backup_reserve_seconds": 600,
            "projected_all_in_seconds": projected_all_in,
            "remaining_seconds": remaining,
            "projected_cost_usd": projected_cost,
            "fp8_scale_report": fp8,
            "no_op_equivalence": no_op,
            "snapshot": snapshot,
            "paid_environment": paid,
            "git_commit": commit,
        }
        _atomic_json(DEFAULT_INFRA_ROOT / "runtime_gate.json", throughput)
        if not throughput["passed"]:
            raise ValueError("V16 source stage does not fit the paid deadline or cap")
        source = _run_source(args, runtime, paid)
        downstream = _run_downstream(args, runtime, source, paid)
        terminal = {
            "schema_version": "glm53_v16_terminal_state_v1",
            "source_decision": source["decision"]["decision"],
            "source_passed": source["decision"]["passed"],
            "downstream": downstream,
            "elapsed_seconds": time.perf_counter() - started,
            "deadline_utc": _deadline().isoformat(),
            "manual_review_required_for_positive_recruitment_claim": True,
        }
        _atomic_json(DEFAULT_INFRA_ROOT / "terminal_state.json", terminal)
        from src.glm53_user_eval.v16.evidence import build_evidence

        build_evidence(
            repo_root=ROOT,
            roots=[args.feature_root, args.source_root, args.downstream_root, DEFAULT_INFRA_ROOT],
            output_path=DEFAULT_INFRA_ROOT / "final_evidence.json",
            terminal=terminal,
        )
        print(json.dumps(terminal, indent=2))
    finally:
        runtime.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-tokenizer", "validate-prereg", "plan-paid", "paid-supervisor"))
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--downstream-manifest", type=Path, default=DEFAULT_DOWNSTREAM)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--downstream-root", type=Path, default=DEFAULT_DOWNSTREAM_ROOT)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--confirm-spend", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commands = {
        "audit-tokenizer": command_audit_tokenizer,
        "validate-prereg": command_validate_prereg,
        "plan-paid": command_plan_paid,
        "paid-supervisor": command_paid_supervisor,
    }
    if args.command == "paid-supervisor" and args.model_path is None:
        raise ValueError("paid-supervisor requires --model-path")
    commands[args.command](args)


if __name__ == "__main__":
    main()
