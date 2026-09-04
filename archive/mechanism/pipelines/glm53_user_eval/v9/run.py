"""CLI for the paper-faithful GLM-5.3 v9 study."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v8.artifacts import atomic_json, sha256_file
from src.glm53_user_eval.v9.analysis import analyze_readout
from src.glm53_user_eval.v9.datasets import load_eval_rows

app = typer.Typer(no_args_is_help=True)
CONFIG = ROOT / "pipelines/glm53_user_eval/v9/configs"
DEFAULT_PREREG = CONFIG / "prereg_v9_paper_faithful.yaml"
DEFAULT_RUNTIME = CONFIG / "runtime_v9.yaml"
DEFAULT_ARTIFACT = ROOT / "artifacts/glm53_user_eval/v9"
DEFAULT_DATASET = ROOT / "artifacts/datasets/contrastive_prompts_v2"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _check_hash(path: Path, expected: str) -> bool:
    return path.is_file() and sha256_file(path) == expected


def _check_git_blob_hash(relative_path: str, expected: str) -> bool:
    content = subprocess.check_output(["git", "show", f"HEAD:{relative_path}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest() == expected


@app.command("validate-prereg")
def validate_prereg(prereg: Path = DEFAULT_PREREG) -> None:
    config = read_yaml(prereg)
    parent = config["parent_locks"]
    paper = config["paper_contract"]["runtime_config"]
    dataset = config["dataset"]
    rows = load_eval_rows(ROOT / dataset["root"])
    checks = {
        "schema": config["schema_version"] == "glm53_user_eval_v9_prereg_v1_3",
        "v8_final_commit": _git("rev-list", "-n", "1", parent["v8_final_tag"])
        == parent["v8_final_commit"],
        "v8_terminal_hash": _check_hash(
            ROOT / parent["v8_terminal_state"]["path"], parent["v8_terminal_state"]["sha256"]
        ),
        "v8_m2_hash": _check_hash(
            ROOT / parent["v8_m2_decision"]["path"], parent["v8_m2_decision"]["sha256"]
        ),
        "v8_m3_hash": _check_hash(
            ROOT / parent["v8_m3_decision"]["path"], parent["v8_m3_decision"]["sha256"]
        ),
        "paper_config_hash": _check_git_blob_hash(paper["path"], paper["sha256"]),
        "samples_hash": _check_git_blob_hash(
            f"{dataset['root']}/samples.csv", dataset["samples_sha256"]
        ),
        "splits_hash": _check_git_blob_hash(
            f"{dataset['root']}/splits.csv", dataset["splits_sha256"]
        ),
        "row_count": len(rows) == 448,
        "v8_results_diagnostic_only": config["scientific_correction"]["v8_results_role"]
        == "diagnostic_only_not_v9_evidence",
        "readout_steering_separate": config["analysis"]["steering_gate_is_separate"] is True,
        "paid_scope_extraction_only": "steering"
        in config["execution"]["forbidden_in_first_paid_run"],
    }
    report = {
        "schema_version": "glm53_v9_m0_prereg_validation_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "prereg_sha256": sha256_file(prereg),
        "current_head": _git("rev-parse", "HEAD"),
    }
    atomic_json(DEFAULT_ARTIFACT / "m0/prereg_validation.json", report)
    typer.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("validate-tokenizer")
def validate_tokenizer(
    tokenizer_root: Path = DEFAULT_ARTIFACT / "tokenizer_snapshot",
    output: Path = DEFAULT_ARTIFACT / "m1/local_tokenizer_contract.json",
) -> None:
    from src.glm53_user_eval.v9.tokenizer_contract import validate_rows

    rows = load_eval_rows(DEFAULT_DATASET)
    report = validate_rows(rows, tokenizer_root)
    atomic_json(output, report)
    typer.echo(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("plan")
def plan(prereg: Path = DEFAULT_PREREG) -> None:
    config = read_yaml(prereg)
    rows = load_eval_rows(ROOT / config["dataset"]["root"])
    manifest = {
        "schema_version": "glm53_v9_plan_v1",
        "project_id": config["project_id"],
        "scientific_rows": len(rows),
        "model_forwards": len(rows),
        "layers": config["subject"]["text_layers"],
        "primary_representation": config["representations"]["primary"],
        "secondary_representations": config["representations"]["secondary"],
        "token_bag_layers": 45,
        "paid_scope": config["execution"]["exact_paid_scope"],
        "forbidden_first_run": config["execution"]["forbidden_in_first_paid_run"],
        "permutation_reps_offline": config["analysis"]["permutation_reps"],
        "readout_and_steering_separate": True,
        "prereg_sha256": sha256_file(prereg),
    }
    atomic_json(DEFAULT_ARTIFACT / "plan.json", manifest)
    typer.echo(json.dumps(manifest, indent=2))


def _require_prereg_tag(config: dict[str, Any]) -> None:
    tag = config["execution"]["prereg_tag"]
    tag_commit = _git("rev-list", "-n", "1", tag)
    head = _git("rev-parse", "HEAD")
    if tag_commit != head:
        raise RuntimeError(f"HEAD {head} differs from preregistration tag {tag} at {tag_commit}")
    if _git("status", "--porcelain"):
        raise RuntimeError("paid extraction requires a clean Git tree")


@app.command("extract")
def extract(
    model_path: Annotated[Path, typer.Option()],
    prereg: Path = DEFAULT_PREREG,
    runtime_config: Path = DEFAULT_RUNTIME,
    output_root: Path = DEFAULT_ARTIFACT / "features/eval_prompt",
    full_rehash: bool = True,
) -> None:
    from src.glm53_user_eval.v8.whitebox_runtime import verify_model_snapshot
    from src.glm53_user_eval.v9.extraction import extract_eval_features
    from src.glm53_user_eval.v9.runtime import LoadedV9GLM53

    config = read_yaml(prereg)
    _require_prereg_tag(config)
    local_contract_path = DEFAULT_ARTIFACT / "m1/local_tokenizer_contract.json"
    local_contract = json.loads(local_contract_path.read_text(encoding="utf-8"))
    if not local_contract["passed"]:
        raise RuntimeError("local cue-mask contract did not pass")
    runtime_payload = read_yaml(runtime_config)
    stage_manifest = json.loads(
        (ROOT / "artifacts/glm53_user_eval/runtime/g2/model_stage.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot = verify_model_snapshot(model_path, stage_manifest, full_rehash=full_rehash)
    if not snapshot["all_shards_match"]:
        raise RuntimeError("exact model snapshot verification failed")
    rows = load_eval_rows(ROOT / config["dataset"]["root"])
    runtime = LoadedV9GLM53(model_path=model_path, config=runtime_payload)
    try:
        fp8_scale_report = runtime.fp8_scale_report()
        if not fp8_scale_report["passed"]:
            raise RuntimeError("forget-gate FP8 scale tensor check failed")
        manifest = extract_eval_features(runtime, rows, output_root=output_root)
    finally:
        runtime.close()
    report = {
        "schema_version": "glm53_v9_extraction_decision_v1",
        "passed": bool(manifest["passed"]),
        "snapshot": snapshot,
        "load_seconds": runtime.load_seconds,
        "fp8_scale_report": fp8_scale_report,
        "feature_manifest_sha256": sha256_file(output_root / "feature_manifest.json"),
        "prereg_sha256": sha256_file(prereg),
        "git_commit": _git("rev-parse", "HEAD"),
    }
    atomic_json(DEFAULT_ARTIFACT / "decisions/extraction_decision.json", report)
    typer.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("analyze-readout")
def analyze_readout_command(
    prereg: Path = DEFAULT_PREREG,
    feature_root: Path = DEFAULT_ARTIFACT / "features/eval_prompt",
    output_root: Path = DEFAULT_ARTIFACT / "readout",
) -> None:
    config = read_yaml(prereg)
    report = analyze_readout(feature_root=feature_root, output_root=output_root, config=config)
    typer.echo(
        json.dumps(
            {
                "selected_layer": report["selected_layer"],
                "primary_metrics": report["primary_metrics"],
                "readout_checks": report["readout_checks"],
                "readout_passed": report["readout_passed"],
                "steering_status": report["steering_status"],
            },
            indent=2,
        )
    )


@app.command("verify-readout")
def verify_readout_command(
    feature_root: Path = DEFAULT_ARTIFACT / "features/eval_prompt",
    readout_root: Path = DEFAULT_ARTIFACT / "readout",
    output_path: Path = DEFAULT_ARTIFACT / "readout/verification.json",
) -> None:
    from src.glm53_user_eval.v9.verification import verify_readout

    report = verify_readout(
        feature_root=feature_root,
        readout_root=readout_root,
        output_path=output_path,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("compare-v8-prompt-final")
def compare_v8_prompt_final(
    feature_root: Path = DEFAULT_ARTIFACT / "features/eval_prompt",
    v8_root: Path = ROOT.parent / "glm53_v8_terminal_v119/m3",
) -> None:
    v9_archive = __import__("numpy").load(feature_root / "fixed_features.npz")
    v8_archive = __import__("numpy").load(v8_root / "eval_features.npz")
    v9_rows = [
        json.loads(line)
        for line in (feature_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    v8_rows = [
        json.loads(line)
        for line in (v8_root / "eval_features.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [row["sample_id"] for row in v9_rows] != [row["sample_id"] for row in v8_rows]:
        raise ValueError("v8 and v9 feature row orders differ")
    layer_max_errors = {}
    for layer in range(45):
        difference = abs(
            v9_archive["prompt_final"][:, layer].astype("float32")
            - v8_archive[f"l{layer}__prompt_final"].astype("float32")
        )
        layer_max_errors[str(layer)] = float(difference.max())
    prompt_hash_match_rate = sum(
        left["input_ids_sha256"] == right["prompt_sha256"]
        for left, right in zip(v9_rows, v8_rows, strict=True)
    ) / len(v9_rows)
    report = {
        "schema_version": "glm53_v9_v8_prompt_final_equivalence_v1",
        "row_count": len(v9_rows),
        "prompt_hash_match_rate": prompt_hash_match_rate,
        "maximum_activation_absolute_error": max(layer_max_errors.values()),
        "layer_max_errors": layer_max_errors,
        "tolerance": 0.002,
        "passed": prompt_hash_match_rate == 1.0 and max(layer_max_errors.values()) <= 0.002,
        "v8_features_sha256": sha256_file(v8_root / "eval_features.npz"),
        "v9_features_sha256": sha256_file(feature_root / "fixed_features.npz"),
    }
    atomic_json(feature_root / "v8_prompt_final_equivalence.json", report)
    typer.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
