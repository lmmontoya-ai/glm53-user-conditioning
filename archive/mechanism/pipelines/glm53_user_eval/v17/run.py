"""Command line interface for the preregistered V17 Hua causal test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v11.downstream import validate_downstream_assets
from src.glm53_user_eval.v17.contract import (
    atomic_json,
    read_yaml,
    sha256_file,
    validate_v17_prereg,
)
from src.glm53_user_eval.v17.prepare import build_causal_schedule, build_positive_control_manifest

V17 = ROOT / "pipelines/glm53_user_eval/v17"
DEFAULT_PREREG = V17 / "configs/prereg_v17_hua_causal.yaml"
DEFAULT_RUNTIME = V17 / "configs/runtime_v17.yaml"
DEFAULT_DOWNSTREAM = V17 / "configs/downstream_manifest_v17.json"
DEFAULT_PC = V17 / "configs/positive_control_manifest_v1.json"
DEFAULT_SCHEDULE = V17 / "configs/causal_schedule_v17.json"
DEFAULT_OUTPUT = ROOT / os.environ.get(
    "GLM53_ARTIFACT_RELATIVE_ROOT", "artifacts/glm53_user_eval/v17"
)


def command_prepare(_args: argparse.Namespace) -> None:
    pc = build_positive_control_manifest(
        ROOT / "artifacts/datasets/contrastive_prompts_v5/samples.jsonl",
        output_path=DEFAULT_PC,
    )
    preflight, proxy_rows, _ = validate_downstream_assets(
        repo_root=ROOT, manifest_path=DEFAULT_DOWNSTREAM
    )
    schedule = build_causal_schedule(
        proxy_rows=proxy_rows,
        causal_schedule_path=ROOT / "pipelines/glm53_user_eval/v8/configs/causal_schedule_v1.json",
        output_path=DEFAULT_SCHEDULE,
    )
    report = {
        "schema_version": "glm53_v17_offline_preflight_v1",
        "passed": True,
        "positive_control_manifest_sha256": sha256_file(DEFAULT_PC),
        "causal_schedule_sha256": sha256_file(DEFAULT_SCHEDULE),
        "downstream_preflight": preflight,
        "positive_control_counts": pc["counts"],
        "causal_schedule": schedule,
    }
    atomic_json(DEFAULT_OUTPUT / "infrastructure/offline_preflight.json", report)
    print(json.dumps(report, indent=2))


def command_validate(args: argparse.Namespace) -> None:
    print(json.dumps(validate_v17_prereg(ROOT, args.prereg), indent=2))


def command_plan(args: argparse.Namespace) -> None:
    validation = validate_v17_prereg(ROOT, args.prereg)
    prereg = read_yaml(args.prereg)
    plan = {
        "schema_version": "glm53_v17_paid_plan_v1",
        "passed": True,
        "prereg_sha256": validation["prereg_sha256"],
        "hardware": {"gpu_type": "NVIDIA H200", "gpu_count": 3, "fallback": False},
        "budget": prereg["budget"],
        "stages": ["Hua direction and positive control", "local parity", "causal pilot", "causal confirmation"],
        "stop_order": "each failed machine gate stops all later model calls",
    }
    atomic_json(DEFAULT_OUTPUT / "infrastructure/paid_plan.json", plan)
    print(json.dumps(plan, indent=2))


def _require_tag() -> str:
    if os.environ.get("GLM53_SOURCE_ARCHIVE_VERIFIED") == "1":
        commit = os.environ.get("GLM53_SOURCE_COMMIT", "")
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("verified source archive lacks a literal source commit")
        return commit
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    required_tag = os.environ.get(
        "GLM53_EXECUTION_TAG", "glm53-user-eval-v17-runtime-v2"
    )
    tagged = subprocess.check_output(
        ["git", "rev-list", "-n", "1", required_tag],
        cwd=ROOT,
        text=True,
    ).strip()
    if head != tagged:
        raise ValueError("paid V17 must run from the frozen runtime-amendment tag")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    allowed_inputs = {
        "artifacts/glm53_user_eval/v11/downstream_inputs/preflight.json",
        "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl",
        "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100_manifest.json",
        "artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json",
    }
    observed = {line[3:].replace("\\", "/") for line in status if len(line) >= 4}
    if not observed.issubset(allowed_inputs):
        raise ValueError("paid V17 requires a clean Git tree")
    return head


def command_paid(args: argparse.Namespace) -> None:
    if not args.confirm_spend:
        raise ValueError("paid-supervisor requires --confirm-spend")
    validate_v17_prereg(ROOT, args.prereg)
    commit = _require_tag()
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    rate = float(os.environ.get("GLM53_V17_AGGREGATE_RATE_USD", "0"))
    balance = float(os.environ.get("GLM53_V17_LAUNCH_BALANCE_USD", "0"))
    floor = float(os.environ.get("GLM53_V17_BALANCE_FLOOR_USD", "0"))
    rate_cap = float(os.environ.get("GLM53_RATE_CAP_USD_PER_HOUR", "16.5"))
    compute_cap = float(os.environ.get("GLM53_COMPUTE_CAP_USD", "15.01"))
    if not pod_id or rate <= 0 or rate > rate_cap or balance - floor > compute_cap or floor < 8:
        raise ValueError("paid V17 environment violates hardware or budget lock")
    if any(os.environ.get(name) for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")):
        raise ValueError("science process may not inherit S3 credentials")
    from src.glm53_user_eval.v8.whitebox_runtime import verify_model_snapshot
    from src.glm53_user_eval.v17.supervisor import run_paid_ladder

    stage = json.loads(
        (ROOT / "artifacts/glm53_user_eval/runtime/g2/model_stage.json").read_text(encoding="utf-8")
    )
    snapshot = verify_model_snapshot(args.model_path, stage, full_rehash=True)
    atomic_json(DEFAULT_OUTPUT / "infrastructure/model_snapshot_verification.json", snapshot)
    result = run_paid_ladder(
        repo_root=ROOT,
        prereg_path=args.prereg,
        runtime_config_path=args.runtime,
        downstream_manifest_path=DEFAULT_DOWNSTREAM,
        model_path=args.model_path,
        output_root=DEFAULT_OUTPUT / "run",
    )
    terminal = {
        "schema_version": "glm53_v17_terminal_state_v1",
        "result": result,
        "git_commit": commit,
        "pod_id": pod_id,
        "model_snapshot": snapshot,
    }
    atomic_json(DEFAULT_OUTPUT / "infrastructure/terminal_state.json", terminal)
    print(json.dumps(terminal, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("command", choices=("prepare", "validate-prereg", "plan-paid", "paid-supervisor"))
    value.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    value.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    value.add_argument("--model-path", type=Path)
    value.add_argument("--confirm-spend", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    commands = {
        "prepare": command_prepare,
        "validate-prereg": command_validate,
        "plan-paid": command_plan,
        "paid-supervisor": command_paid,
    }
    if args.command == "paid-supervisor" and args.model_path is None:
        raise ValueError("paid-supervisor requires --model-path")
    commands[args.command](args)


if __name__ == "__main__":
    main()
