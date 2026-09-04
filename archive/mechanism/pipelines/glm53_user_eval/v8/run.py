"""Command line entry point for the machine-gated v8 study."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v8.artifacts import atomic_json, sha256_file
from src.glm53_user_eval.v8.proxy import (
    load_contract_tokenizer,
    proxy_messages,
    validate_codebooks,
    validate_label_tokens,
)
from src.glm53_user_eval.v8.source_locks import (
    verify_direction_dataset,
    verify_log_manifest,
    verify_parent,
)
from src.glm53_user_eval.v8.supervisor import decision_payload
from src.glm53_user_eval.v8.transcript_cache import build_cache

app = typer.Typer(no_args_is_help=True)
CONFIG = ROOT / "pipelines/glm53_user_eval/v8/configs"
DEFAULT_PREREG = CONFIG / "prereg_v8_whitebox_mechanism.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts/glm53_user_eval/v8"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@app.command("build-transcript-cache")
def build_transcript_cache(
    prereg: Path = DEFAULT_PREREG,
    output: Path = DEFAULT_ARTIFACT / "cache/v7_transcripts_25.jsonl",
) -> None:
    config = read_yaml(prereg)
    schedule = read_json(ROOT / config["selection"]["causal_schedule"])
    tasks = set().union(*schedule["tasks"].values())
    manifest = build_cache(
        raw_scores_path=ROOT
        / "artifacts/glm53_user_eval/reports/transluce_interaction_v7/raw_scores.jsonl",
        final_evidence_path=ROOT / config["parent_result"]["final_evidence_path"],
        task_ids=tasks,
        output_path=output,
    )
    missing_limit = float(config["proxy"]["source_first_turn_missing_rate_max"])
    manifest["missing_rate_limit"] = missing_limit
    manifest["passed"] = manifest["proxy_ineligible_rate"] <= missing_limit
    atomic_json(output.with_name("v7_transcripts_25_manifest.json"), manifest)
    typer.echo(json.dumps(manifest, indent=2))
    if not manifest["passed"]:
        raise typer.Exit(1)


@app.command("validate-proxy")
def validate_proxy(
    prereg: Path = DEFAULT_PREREG,
    cache: Path = DEFAULT_ARTIFACT / "cache/v7_transcripts_25.jsonl",
    output: Path = DEFAULT_ARTIFACT / "m1/proxy_contract.json",
) -> None:
    config = read_yaml(prereg)
    codebooks = read_json(ROOT / config["proxy"]["codebooks"])
    validate_codebooks(codebooks)
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    row = next(row for row in rows if row["proxy_eligible"])
    messages = proxy_messages(
        system_text=row["system_text"],
        dilemma_text=row["dilemma_text"],
        first_assistant_text=row["first_assistant_text"],
        codebook=codebooks["codebooks"]["0"],
        followup=codebooks["followup"],
        assistant_prefix=codebooks["assistant_prefix"],
    )
    processor = load_contract_tokenizer(
        config["subject"]["model_id"], config["subject"]["revision"]
    )
    token_contract = validate_label_tokens(processor, messages, codebooks["labels"])
    report = {
        "schema_version": "glm53_v8_proxy_contract_v1",
        "passed": True,
        "cache_sha256": sha256_file(cache),
        "codebook_sha256": sha256_file(ROOT / config["proxy"]["codebooks"]),
        **token_contract,
    }
    atomic_json(output, report)
    typer.echo(json.dumps(report, indent=2))


@app.command("validate-prereg")
def validate_prereg(prereg: Path = DEFAULT_PREREG) -> None:
    config = read_yaml(prereg)
    parent = verify_parent(ROOT, config)
    logs = verify_log_manifest(ROOT, ROOT / config["parent_result"]["final_evidence_path"])
    direction_split_path = ROOT / config["direction"]["split_config"]
    direction_dataset = verify_direction_dataset(ROOT, direction_split_path)
    checks = {
        "parent": True,
        "hundred_logs": len(logs) == 100,
        "causal_schedule": (ROOT / config["selection"]["causal_schedule"]).is_file(),
        "codebooks": (ROOT / config["proxy"]["codebooks"]).is_file(),
        "runtime": (CONFIG / "runtime_v8.yaml").is_file(),
        "direction_splits": direction_split_path.is_file(),
        "direction_dataset": len(direction_dataset) == 3,
    }
    payload = decision_payload(
        "M0",
        checks,
        {"prereg_sha256": sha256_file(prereg), "parent_commit": parent["tag_commit"]},
        {"recognized_v7_logs": len(logs), "direction_dataset_hashes": direction_dataset},
    )
    atomic_json(DEFAULT_ARTIFACT / "decisions/m0_decision.json", payload)
    typer.echo(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise typer.Exit(1)


@app.command("plan")
def plan(prereg: Path = DEFAULT_PREREG) -> None:
    config = read_yaml(prereg)
    schedule = read_json(ROOT / config["selection"]["causal_schedule"])
    manifest = {
        "schema_version": "glm53_v8_plan_v1",
        "project_id": config["project_id"],
        "target_pairs": len(schedule["pairs"]),
        "pilot_pairs": sum(row["pilot"] for row in schedule["pairs"]),
        "confirmation_pairs": sum(not row["pilot"] for row in schedule["pairs"]),
        "control_count_per_group": len(schedule["famous_nonai_controls"]),
        "task_counts": {key: len(value) for key, value in schedule["tasks"].items()},
        "proxy_cache_rows": 4 * 70 * sum(len(value) for value in schedule["tasks"].values()),
        "confirmation_prompt_arms": 480 * 23,
        "score_blind_layer_selection": True,
    }
    atomic_json(DEFAULT_ARTIFACT / "plan.json", manifest)
    typer.echo(json.dumps(manifest, indent=2))


@app.command("supervise")
def supervise(
    prereg: Path = DEFAULT_PREREG,
    source_root: Path = Path("/workspace/mats-glm53/reference/transluce-user-awareness"),
    artifact_root: Path = DEFAULT_ARTIFACT,
    full_rehash: bool = True,
    hourly_rate_usd: float = typer.Option(..., min=0.01),
) -> None:
    """Run M2--M8 in one model-loaded, machine-gated process."""
    from src.glm53_user_eval.v8.on_pod import run_supervisor

    summary = run_supervisor(
        repo_root=ROOT,
        source_root=source_root,
        artifact_root=artifact_root,
        prereg_path=prereg,
        full_rehash=full_rehash,
        hourly_rate_usd=hourly_rate_usd,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("verify-independent")
def verify_independent(
    artifact_root: Path = DEFAULT_ARTIFACT,
    output: Path = DEFAULT_ARTIFACT / "m8/independent_verification.json",
    reps: int = 20_000,
    seed: int = 20260903,
) -> None:
    from src.glm53_user_eval.v8.independent import verify

    report = verify(artifact_root, seed=seed, reps=reps)
    atomic_json(output, report)
    typer.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("build-audit-packet")
def build_audit_packet(
    artifact_root: Path = DEFAULT_ARTIFACT,
    output: Path = DEFAULT_ARTIFACT / "m8/manual_audit_packet.json",
    seed: int = 20260904,
) -> None:
    from src.glm53_user_eval.v8.audit import build_manual_audit_packet

    packet = build_manual_audit_packet(
        repo_root=ROOT,
        artifact_root=artifact_root,
        seed=seed,
        output_path=output,
    )
    typer.echo(json.dumps(packet["counts"], indent=2))


@app.command("finalize-m8")
def finalize_m8_command(
    artifact_root: Path = DEFAULT_ARTIFACT,
    packet: Path = DEFAULT_ARTIFACT / "m8/manual_audit_packet.json",
    manual_audit: Path = DEFAULT_ARTIFACT / "m8/manual_audit.json",
    output: Path = DEFAULT_ARTIFACT / "decisions/m8_decision.json",
) -> None:
    from src.glm53_user_eval.v8.audit import finalize_m8

    decision = finalize_m8(
        artifact_root=artifact_root,
        packet_path=packet,
        audit_path=manual_audit,
        output_path=output,
    )
    typer.echo(json.dumps(decision, indent=2))
    if not decision["passed"]:
        raise typer.Exit(1)


@app.command("build-evidence")
def build_evidence(
    prereg: Path = DEFAULT_PREREG,
    artifact_root: Path = DEFAULT_ARTIFACT,
    output: Path = DEFAULT_ARTIFACT / "final_evidence.json",
) -> None:
    from src.glm53_user_eval.v8.audit import build_final_evidence

    evidence = build_final_evidence(
        artifact_root=artifact_root,
        prereg_path=prereg,
        output_path=output,
    )
    typer.echo(json.dumps({"file_count": evidence["file_count"]}, indent=2))


@app.command("build-report")
def build_report(
    artifact_root: Path = DEFAULT_ARTIFACT,
    output: Path = DEFAULT_ARTIFACT / "final_report.md",
) -> None:
    from src.glm53_user_eval.v8.reporting import build_final_report

    report = build_final_report(artifact_root, output)
    typer.echo(report)


if __name__ == "__main__":
    app()
