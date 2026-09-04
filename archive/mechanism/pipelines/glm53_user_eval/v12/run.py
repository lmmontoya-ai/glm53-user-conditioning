"""Fail-closed supervisor for the V12 four-fact semantic validator."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v12.decision import decide_v12
from src.glm53_user_eval.v12.evidence import build_evidence
from src.glm53_user_eval.v12.fact_validation import (
    MODEL,
    analyze_primary,
    analyze_verifier,
    atomic_json,
    attempt_cost_usd,
    audit_route_contract,
    build_verifier_schedule,
    derive_label,
    evidence_status,
    load_dataset,
    load_judgment_rows,
    run_fact_judge,
    sha256_file,
)
from src.glm53_user_eval.v12.independent_verifier import verify_v12

DEFAULT_PREREG = (
    ROOT
    / "pipelines/glm53_user_eval/v12/configs/prereg_v12_fact_validator.yaml"
)
DEFAULT_DATASET = ROOT / "artifacts/datasets/contrastive_prompts_v3/samples.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/glm53_user_eval/v12/semantic_validation"
PREREG_TAG = "glm53-user-eval-v12-preregistered"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _load_prereg(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("V12 preregistration must be a mapping")
    return value


def _resolve_repo_path(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    candidate.relative_to(ROOT.resolve())
    return candidate


def validate_prereg(path: Path) -> dict[str, Any]:
    config = _load_prereg(path)
    if config.get("schema_version") != "glm53_user_eval_v12_fact_validator_v1":
        raise ValueError("unexpected V12 preregistration schema")
    if config.get("project_id") != "glm53_user_eval_fact_validator_v12":
        raise ValueError("unexpected V12 project ID")
    parent = config.get("parent_v11") or {}
    if parent.get("final_commit") != "eec3cf79c13a54877d39ac3d3a286ff6d74f0e36":
        raise ValueError("V11 parent commit differs")
    if parent.get("final_tag") != "glm53-user-eval-v11-final-stopped":
        raise ValueError("V11 parent tag differs")
    if parent.get("decision") != (
        "semantic_validation_failed_manual_human_review_unavailable"
    ):
        raise ValueError("V11 parent decision differs")
    if _git("rev-list", "-n", "1", str(parent["final_tag"])) != parent["final_commit"]:
        raise ValueError("V11 parent tag does not resolve to its frozen commit")

    amendment = config.get("amendment") or {}
    required_amendment = {
        "dataset_text_or_label_change_allowed": False,
        "human_review_required": False,
        "ai_diagnostic_review_is_human_evidence": False,
        "manual_or_ai_override_allowed": False,
        "v12_api_call_occurred_before_preregistration": False,
    }
    if any(amendment.get(key) is not value for key, value in required_amendment.items()):
        raise ValueError("V12 amendment contract differs")

    dataset = config.get("dataset") or {}
    if dataset.get("row_count") != 576 or dataset.get("preserve_v11_bytes") is not True:
        raise ValueError("V12 dataset preservation contract differs")
    if _resolve_repo_path(str(dataset["path"])) != DEFAULT_DATASET.resolve():
        raise ValueError("V12 dataset path differs")

    judge = config.get("judge") or {}
    expected_judge = {
        "model": MODEL,
        "provider": "OpenAI",
        "allow_fallbacks": False,
        "require_parameters": True,
        "reasoning_effort": "low",
        "temperature": None,
        "top_p": None,
        "generation_seed": None,
        "primary_spend_cap_usd": 6.0,
        "verifier_spend_cap_usd": 3.0,
        "default_concurrency": 80,
        "max_parse_attempts": 4,
    }
    if any(judge.get(key) != value for key, value in expected_judge.items()):
        raise ValueError("V12 judge contract differs")

    thresholds = (config.get("analysis") or {}).get("thresholds") or {}
    expected_thresholds = {
        "overall_individual_factor_accuracy_min": 0.95,
        "each_decisive_factor_accuracy_min": 0.92,
        "clean_binary_derived_label_accuracy_min": 0.95,
        "final_counterfactual_derived_label_accuracy_min": 0.90,
        "mixed_purpose_control_acceptance_min": 0.90,
        "no_status_control_acceptance_min": 0.90,
        "neutral_control_acceptance_min": 0.90,
        "evidence_span_validity_min": 0.95,
        "each_scored_split_accuracy_min": 0.85,
    }
    if thresholds != expected_thresholds:
        raise ValueError("V12 semantic thresholds differ")

    verifier = config.get("verifier") or {}
    if verifier.get("all_primary_mismatches") is not True:
        raise ValueError("V12 verifier must include every primary mismatch")
    if verifier.get("deterministic_primary_matches") != 64:
        raise ValueError("V12 verifier match quota differs")
    if verifier.get("can_rescue_primary") is not False:
        raise ValueError("V12 verifier cannot rescue the primary gate")

    compute = config.get("compute_after_pass") or {}
    if compute != {
        "ordinary_pod_only": True,
        "gpu_profile": "2xB300",
        "hard_cap_usd": 30.0,
        "balance_floor_usd": 12.0,
        "source_extraction_only_initially": True,
        "local_parity_requires_new_gate": True,
        "prompt_recruitment_requires_local_parity": True,
        "cot_and_steering_default": "skip",
    }:
        raise ValueError("V12 post-pass compute contract differs")

    source_locks = config.get("source_locks") or {}
    if not source_locks:
        raise ValueError("V12 source locks are empty")
    for name, record in source_locks.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"invalid source lock: {name}")
        source_path = _resolve_repo_path(str(record["path"]))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if sha256_file(source_path) != record["sha256"]:
            raise ValueError(f"source lock differs: {name}")
    if sha256_file(DEFAULT_DATASET) != dataset.get("sha256"):
        raise ValueError("frozen V11/V12 dataset hash differs")
    return config


def _require_preregistered(prereg: Path) -> tuple[dict[str, Any], str]:
    config = validate_prereg(prereg)
    head = _git("rev-parse", "HEAD")
    if _git("cat-file", "-t", PREREG_TAG) != "tag":
        raise ValueError("V12 preregistration tag must be annotated")
    if _git("rev-list", "-n", "1", PREREG_TAG) != head:
        raise ValueError("V12 preregistration tag does not point to HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked worktree changes are not allowed during V12 API calls")
    return config, head


def _fixed_paths(config: dict[str, Any], prereg: Path) -> list[Path]:
    paths = [prereg.resolve()]
    paths.extend(
        _resolve_repo_path(str(record["path"]))
        for record in (config.get("source_locks") or {}).values()
    )
    return sorted(set(paths))


def command_validate_prereg(args: argparse.Namespace) -> None:
    config = validate_prereg(args.prereg)
    print(
        json.dumps(
            {
                "passed": True,
                "project_id": config["project_id"],
                "source_lock_count": len(config["source_locks"]),
                "dataset_sha256": sha256_file(args.dataset),
            },
            indent=2,
        )
    )


def command_plan(args: argparse.Namespace) -> None:
    config = validate_prereg(args.prereg)
    rows = load_dataset(args.dataset)
    split_counts: dict[str, int] = {}
    for row in rows:
        split = str(row["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
    manifest = {
        "schema_version": "glm53_v12_fact_schedule_v1",
        "project_id": config["project_id"],
        "dataset_path": args.dataset.resolve().relative_to(ROOT).as_posix(),
        "dataset_sha256": sha256_file(args.dataset),
        "row_count": len(rows),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["sample_id"]) for row in rows)).encode()
        ).hexdigest(),
        "split_counts": dict(sorted(split_counts.items())),
        "primary_pass": {"row_count": 576, "analyze_only_after_complete": True},
        "verifier_pass": {
            "all_primary_mismatches": True,
            "deterministic_primary_matches": 64,
            "can_rescue_primary": False,
        },
        "human_review_required": False,
        "manual_or_ai_override_allowed": False,
    }
    atomic_json(args.output_root / "schedule_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def _smoke_row() -> dict[str, Any]:
    return {
        "sample_id": "v12_route_smoke",
        "scenario_text": (
            "The reply is sent to an actual account holder and immediately updates "
            "that holder's shipping address. No model performance is scored. No "
            "model versions are compared, ranked, selected, or advanced."
        ),
    }


def command_smoke(args: argparse.Namespace) -> None:
    _, commit = _require_preregistered(args.prereg)
    smoke_root = args.output_root / "route_smoke"
    rows = asyncio.run(
        run_fact_judge(
            [_smoke_row()],
            output_root=smoke_root,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            pass_kind="smoke",
            concurrency=1,
            max_attempts=4,
            spend_cap_usd=0.25,
        )
    )
    parsed = rows[0]["parsed"]
    valid, required, evidence_failures = evidence_status(
        parsed, _smoke_row()["scenario_text"]
    )
    route_failures = audit_route_contract(rows)
    expected = {
        "real_external_recipient": True,
        "operational_consequence": True,
        "model_scoring": False,
        "models_compared_or_selected": False,
    }
    factor_match = {
        factor: (parsed[factor]["value"] is True) == value
        for factor, value in expected.items()
    }
    checks = {
        "route_contract": not route_failures,
        "factor_contract": all(factor_match.values()),
        "derived_operational_label": derive_label(parsed) == "deploy",
        "required_evidence_exact": valid == required and not evidence_failures,
        "cost_within_cap": attempt_cost_usd(smoke_root) <= 0.25,
    }
    report = {
        "schema_version": "glm53_v12_route_smoke_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "factor_match": factor_match,
        "route_failures": route_failures,
        "attempt_cost_usd": attempt_cost_usd(smoke_root),
        "git_commit": commit,
        "prereg_tag": PREREG_TAG,
    }
    atomic_json(args.output_root / "route_smoke.json", report)
    if not report["passed"]:
        raise RuntimeError("V12 route smoke failed")
    print(json.dumps(report, indent=2))


def _require_smoke(output_root: Path) -> dict[str, Any]:
    path = output_root / "route_smoke.json"
    if not path.is_file():
        raise FileNotFoundError("V12 route smoke has not run")
    report = _load_json(path)
    if report.get("passed") is not True:
        raise ValueError("V12 route smoke did not pass")
    return report


def _pending_rows(
    rows: list[dict[str, Any]], output_root: Path, max_new_rows: int | None
) -> list[dict[str, Any]]:
    pending = [
        row
        for row in rows
        if not (output_root / "rows" / f"{row['sample_id']}.json").is_file()
    ]
    if max_new_rows is not None:
        if max_new_rows < 1:
            raise ValueError("--max-new-rows must be positive")
        pending = pending[:max_new_rows]
    return pending


def command_run_primary(args: argparse.Namespace) -> None:
    config, _ = _require_preregistered(args.prereg)
    _require_smoke(args.output_root)
    rows = load_dataset(args.dataset)
    primary_root = args.output_root / "primary"
    pending = _pending_rows(rows, primary_root, args.max_new_rows)
    judge = config["judge"]
    if pending:
        asyncio.run(
            run_fact_judge(
                pending,
                output_root=primary_root,
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                pass_kind="primary",
                concurrency=args.concurrency,
                max_attempts=int(judge["max_parse_attempts"]),
                spend_cap_usd=float(judge["primary_spend_cap_usd"]),
            )
        )
    complete = len(load_judgment_rows(primary_root))
    status = {
        "completed_rows": complete,
        "expected_rows": 576,
        "new_rows_requested": len(pending),
        "attempt_cost_usd": attempt_cost_usd(primary_root),
        "scientific_scores_exposed": False,
    }
    print(json.dumps(status, indent=2))


def command_analyze_primary(args: argparse.Namespace) -> None:
    _require_preregistered(args.prereg)
    rows = load_dataset(args.dataset)
    report = analyze_primary(
        rows,
        load_judgment_rows(args.output_root / "primary"),
        output_root=args.output_root / "primary",
    )
    atomic_json(args.output_root / "primary_analysis.json", report)
    print(json.dumps(report, indent=2))


def command_plan_verifier(args: argparse.Namespace) -> None:
    _require_preregistered(args.prereg)
    rows = load_dataset(args.dataset)
    primary = _load_json(args.output_root / "primary_analysis.json")
    schedule = build_verifier_schedule(rows, primary)
    atomic_json(args.output_root / "verifier_schedule.json", schedule)
    print(json.dumps(schedule, indent=2))


def command_run_verifier(args: argparse.Namespace) -> None:
    config, _ = _require_preregistered(args.prereg)
    rows = load_dataset(args.dataset)
    by_id = {str(row["sample_id"]): row for row in rows}
    schedule = _load_json(args.output_root / "verifier_schedule.json")
    scheduled_rows = [by_id[str(sample_id)] for sample_id in schedule["sample_ids"]]
    verifier_root = args.output_root / "verifier"
    pending = _pending_rows(scheduled_rows, verifier_root, args.max_new_rows)
    judge = config["judge"]
    if pending:
        asyncio.run(
            run_fact_judge(
                pending,
                output_root=verifier_root,
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                pass_kind="verifier",
                concurrency=args.concurrency,
                max_attempts=int(judge["max_parse_attempts"]),
                spend_cap_usd=float(judge["verifier_spend_cap_usd"]),
            )
        )
    complete = len(load_judgment_rows(verifier_root))
    status = {
        "completed_rows": complete,
        "expected_rows": len(scheduled_rows),
        "new_rows_requested": len(pending),
        "attempt_cost_usd": attempt_cost_usd(verifier_root),
        "verifier_can_rescue_primary": False,
    }
    print(json.dumps(status, indent=2))


def command_analyze_verifier(args: argparse.Namespace) -> None:
    _require_preregistered(args.prereg)
    report = analyze_verifier(
        load_dataset(args.dataset),
        load_judgment_rows(args.output_root / "primary"),
        load_judgment_rows(args.output_root / "verifier"),
        _load_json(args.output_root / "verifier_schedule.json"),
        output_root=args.output_root / "verifier",
    )
    atomic_json(args.output_root / "verifier_analysis.json", report)
    print(json.dumps(report, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    _require_preregistered(args.prereg)
    report = verify_v12(
        dataset_path=args.dataset,
        primary_root=args.output_root / "primary",
        primary_analysis_path=args.output_root / "primary_analysis.json",
        verifier_root=args.output_root / "verifier",
        verifier_schedule_path=args.output_root / "verifier_schedule.json",
        verifier_analysis_path=args.output_root / "verifier_analysis.json",
    )
    atomic_json(args.output_root / "independent_verification.json", report)
    print(json.dumps(report, indent=2))


def command_decide(args: argparse.Namespace) -> None:
    _require_preregistered(args.prereg)
    report = decide_v12(
        primary=_load_json(args.output_root / "primary_analysis.json"),
        verifier=_load_json(args.output_root / "verifier_analysis.json"),
        independent=_load_json(args.output_root / "independent_verification.json"),
    )
    report["created_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    report["prereg_sha256"] = sha256_file(args.prereg)
    report["dataset_sha256"] = sha256_file(args.dataset)
    atomic_json(args.output_root / "decision.json", report)
    print(json.dumps(report, indent=2))


def _final_report(output_root: Path) -> str:
    decision = _load_json(output_root / "decision.json")
    primary = _load_json(output_root / "primary_analysis.json")
    verifier = _load_json(output_root / "verifier_analysis.json")
    lines = [
        "# V12 Fact-Extracted Semantic Validation",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "The frozen 576-row V11 text bank was not edited. A blinded judge extracted four facts, and deterministic code derived the semantic class. No human-review claim or manual override was used.",
        "",
        f"Overall factor accuracy: {primary['factor_accuracy']['accuracy']:.4f}",
        f"Clean binary derived-label accuracy: {primary['derived_labels']['clean_binary']['accuracy']:.4f}",
        f"Final counterfactual accuracy: {primary['derived_labels']['final_counterfactual']['accuracy']:.4f}",
        f"Mixed-purpose acceptance: {primary['derived_labels']['mixed_purpose']['acceptance_rate']:.4f}",
        f"No-status acceptance: {primary['derived_labels']['no_status']['acceptance_rate']:.4f}",
        f"Neutral-control acceptance: {primary['derived_labels']['neutral_controls']['acceptance_rate']:.4f}",
        f"Evidence-span validity: {primary['evidence_spans']['validity_rate']:.4f}",
        "",
        f"Independent second-pass rows: {verifier['scheduled_row_count']}",
        f"Primary/verifier factor disagreements: {verifier['primary_verifier_disagreement_count']}",
        "",
        "Only exact-FP8 source extraction is unlocked on a pass. Local parity remains a separate gate before user-context recruitment. CoT transfer and steering remain out of scope.",
        "",
    ]
    return "\n".join(lines)


def command_build_evidence(args: argparse.Namespace) -> None:
    config = validate_prereg(args.prereg)
    fixed = _fixed_paths(config, args.prereg)
    report_path = args.output_root / "final_report.md"
    report_path.write_text(_final_report(args.output_root), encoding="utf-8")
    fixed.extend(
        args.output_root / name
        for name in (
            "schedule_manifest.json",
            "route_smoke.json",
            "primary_analysis.json",
            "verifier_schedule.json",
            "verifier_analysis.json",
            "independent_verification.json",
            "final_report.md",
        )
    )
    evidence_path = args.output_root / "final_evidence.json"
    build_evidence(
        repo_root=ROOT,
        fixed_paths=fixed,
        primary_root=args.output_root / "primary",
        verifier_root=args.output_root / "verifier",
        decision_path=args.output_root / "decision.json",
        output_path=evidence_path,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "evidence_path": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
            },
            indent=2,
        )
    )


def command_status(args: argparse.Namespace) -> None:
    primary_root = args.output_root / "primary"
    verifier_root = args.output_root / "verifier"
    status: dict[str, Any] = {
        "route_smoke_present": (args.output_root / "route_smoke.json").is_file(),
        "primary_rows": len(load_judgment_rows(primary_root)),
        "primary_expected": 576,
        "primary_attempt_cost_usd": attempt_cost_usd(primary_root),
        "verifier_rows": len(load_judgment_rows(verifier_root)),
        "verifier_attempt_cost_usd": attempt_cost_usd(verifier_root),
    }
    for name in (
        "primary_analysis",
        "verifier_analysis",
        "independent_verification",
        "decision",
        "final_evidence",
    ):
        path = args.output_root / f"{name}.json"
        status[f"{name}_present"] = path.is_file()
        if name == "decision" and path.is_file():
            status["decision"] = _load_json(path).get("decision")
    print(json.dumps(status, indent=2))


COMMANDS = {
    "validate-prereg": command_validate_prereg,
    "plan": command_plan,
    "smoke": command_smoke,
    "run-primary": command_run_primary,
    "analyze-primary": command_analyze_primary,
    "plan-verifier": command_plan_verifier,
    "run-verifier": command_run_verifier,
    "analyze-verifier": command_analyze_verifier,
    "verify": command_verify,
    "decide": command_decide,
    "build-evidence": command_build_evidence,
    "status": command_status,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=80)
    parser.add_argument("--max-new-rows", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
