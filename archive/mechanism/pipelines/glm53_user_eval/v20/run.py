"""Command line interface for the preregistered V20 experiment."""

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

from src.glm53_user_eval.v20.contract import (
    atomic_json,
    read_json,
    read_yaml,
    sha256_file,
    validate_v20_prereg,
)

V20 = ROOT / "pipelines/glm53_user_eval/v20"
DEFAULT_PREREG = V20 / "configs/prereg_v20_direct_user_hua.yaml"
DEFAULT_RUNTIME = V20 / "configs/runtime_v20.yaml"
DEFAULT_DOWNSTREAM = ROOT / "pipelines/glm53_user_eval/v17/configs/downstream_manifest_v17.json"
DEFAULT_OUTPUT = ROOT / os.environ.get(
    "GLM53_ARTIFACT_RELATIVE_ROOT", "artifacts/glm53_user_eval/v20"
)


def command_validate(args: argparse.Namespace) -> None:
    print(json.dumps(validate_v20_prereg(ROOT, args.prereg), indent=2))


def command_plan(args: argparse.Namespace) -> None:
    validation = validate_v20_prereg(ROOT, args.prereg)
    prereg = read_yaml(args.prereg)
    design = read_json(ROOT / prereg["immutable_inputs"]["design"]["path"])
    runtime = read_yaml(args.runtime)
    plan = {
        "schema_version": "glm53_v20_paid_plan_v1",
        "passed": True,
        "prereg_sha256": validation["prereg_sha256"],
        "hardware": {
            "gpu_type": runtime["runpod"]["gpu_id"],
            "gpu_count": runtime["runpod"]["gpu_count"],
            "fallback": False,
        },
        "candidate_batch_sizes": runtime["forward"]["candidate_batch_sizes"],
        "local_parity_rows": design["full_surface"]["reconstructable_rows"],
        "full_causal_arms": design["full_surface"]["conditions"],
        "signflip_controls": design["signflip_control_surface"]["controls"],
        "signflip_rows_per_control": design["signflip_control_surface"]["rows_per_control"],
        "planned_scientific_prompt_evaluations": design["planned_scientific_prompt_evaluations"],
        "budget": runtime["runpod"],
        "stop_order": "runtime, local parity, direct intervention",
    }
    atomic_json(DEFAULT_OUTPUT / "infrastructure/paid_plan.json", plan)
    print(json.dumps(plan, indent=2))


def _require_frozen_source() -> str:
    if os.environ.get("GLM53_SOURCE_ARCHIVE_VERIFIED") == "1":
        commit = os.environ.get("GLM53_SOURCE_COMMIT", "")
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise ValueError("verified V20 source archive lacks a literal commit")
        return commit
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tag = os.environ.get("GLM53_EXECUTION_TAG", "glm53-user-eval-v20-preregistered")
    tagged = subprocess.check_output(
        ["git", "rev-list", "-n", "1", tag], cwd=ROOT, text=True
    ).strip()
    if head != tagged:
        raise ValueError("paid V20 must run from its frozen preregistration tag")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("paid V20 requires a clean Git tree")
    return head


def command_paid(args: argparse.Namespace) -> None:
    if not args.confirm_spend:
        raise ValueError("paid-supervisor requires --confirm-spend")
    validate_v20_prereg(ROOT, args.prereg, verify_git=False)
    commit = _require_frozen_source()
    runtime = read_yaml(args.runtime)
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    rate = float(os.environ.get("GLM53_V20_AGGREGATE_RATE_USD", "0"))
    balance = float(os.environ.get("GLM53_V20_LAUNCH_BALANCE_USD", "0"))
    floor = float(os.environ.get("GLM53_V20_BALANCE_FLOOR_USD", "0"))
    rate_cap = float(runtime["runpod"]["aggregate_gpu_rate_cap_usd_per_hour"])
    compute_cap = float(runtime["runpod"]["compute_hard_cap_usd"])
    minimum = float(runtime["runpod"]["minimum_uncommitted_balance_usd"])
    if (
        not pod_id
        or rate <= 0
        or rate > rate_cap
        or balance - floor > compute_cap + 1e-9
        or floor < minimum
    ):
        raise ValueError("paid V20 environment violates its hardware or budget lock")
    if any(os.environ.get(name) for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")):
        raise ValueError("the V20 science process may not inherit S3 credentials")

    from src.glm53_user_eval.v8.whitebox_runtime import verify_model_snapshot
    from src.glm53_user_eval.v20.supervisor import run_paid_ladder

    stage = read_json(ROOT / "artifacts/glm53_user_eval/runtime/g2/model_stage.json")
    snapshot = verify_model_snapshot(args.model_path, stage, full_rehash=True)
    atomic_json(DEFAULT_OUTPUT / "infrastructure/model_snapshot_verification.json", snapshot)
    result = run_paid_ladder(
        repo_root=ROOT,
        prereg_path=args.prereg,
        runtime_path=args.runtime,
        downstream_manifest_path=DEFAULT_DOWNSTREAM,
        model_path=args.model_path,
        output_root=DEFAULT_OUTPUT / "run",
    )
    terminal = {
        "schema_version": "glm53_v20_terminal_state_v1",
        "result": result,
        "git_commit": commit,
        "pod_id": pod_id,
        "model_snapshot": snapshot,
        "prereg_sha256": sha256_file(args.prereg),
    }
    atomic_json(DEFAULT_OUTPUT / "infrastructure/terminal_state.json", terminal)
    print(json.dumps(terminal, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("command", choices=("validate-prereg", "plan-paid", "paid-supervisor"))
    value.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    value.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    value.add_argument("--model-path", type=Path)
    value.add_argument("--confirm-spend", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "paid-supervisor" and args.model_path is None:
        raise ValueError("paid-supervisor requires --model-path")
    {
        "validate-prereg": command_validate,
        "plan-paid": command_plan,
        "paid-supervisor": command_paid,
    }[args.command](args)


if __name__ == "__main__":
    main()
