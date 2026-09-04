"""CLI for the post-gate V21 exploratory continuation."""

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

from src.glm53_user_eval.v20.contract import atomic_json, read_yaml
from src.glm53_user_eval.v21.contract import validate_v21_prereg

V21 = ROOT / "pipelines/glm53_user_eval/v21"
DEFAULT_PREREG = V21 / "configs/prereg_v21_exploratory_continuation.yaml"
DEFAULT_RUNTIME = V21 / "configs/runtime_v21.yaml"
DEFAULT_V20_RUNTIME = ROOT / "pipelines/glm53_user_eval/v20/configs/runtime_v20.yaml"
DEFAULT_DOWNSTREAM = ROOT / "pipelines/glm53_user_eval/v17/configs/downstream_manifest_v17.json"
DEFAULT_OUTPUT = ROOT / os.environ.get(
    "GLM53_ARTIFACT_RELATIVE_ROOT", "artifacts/glm53_user_eval/v21"
)


def command_validate(args: argparse.Namespace) -> None:
    print(json.dumps(validate_v21_prereg(ROOT, args.prereg), indent=2))


def command_plan(args: argparse.Namespace) -> None:
    validation = validate_v21_prereg(ROOT, args.prereg)
    prereg = read_yaml(args.prereg)
    runtime = read_yaml(args.runtime)
    plan = {
        "schema_version": "glm53_v21_paid_plan_v1",
        "passed": True,
        "scope": "exploratory_post_failed_local_parity_gate",
        "prereg_sha256": validation["prereg_sha256"],
        "baseline_rows_reused": prereg["execution"]["baseline_rows_reused_without_rescoring"],
        "new_prompt_evaluations": prereg["execution"]["total_new_prompt_evaluations"],
        "hardware": runtime["runpod"],
    }
    atomic_json(DEFAULT_OUTPUT / "infrastructure/paid_plan.json", plan)
    print(json.dumps(plan, indent=2))


def _require_frozen_source() -> str:
    expected = os.environ.get("GLM53_V21_PREREG_COMMIT", "")
    if len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("V21 requires its literal preregistration commit")
    if os.environ.get("GLM53_SOURCE_ARCHIVE_VERIFIED") == "1":
        return expected
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tag = "glm53-user-eval-v21-preregistered"
    tagged = subprocess.check_output(["git", "rev-list", "-n", "1", tag], cwd=ROOT, text=True).strip()
    if head != tagged or head != expected:
        raise ValueError("paid V21 must run from its frozen preregistration tag")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("paid V21 requires a clean Git tree")
    return head


def command_paid(args: argparse.Namespace) -> None:
    if not args.confirm_spend:
        raise ValueError("paid-supervisor requires --confirm-spend")
    validation = validate_v21_prereg(ROOT, args.prereg)
    commit = _require_frozen_source()
    runtime = read_yaml(args.runtime)
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    rate = float(os.environ.get("GLM53_V20_AGGREGATE_RATE_USD", "0"))
    balance = float(os.environ.get("GLM53_V20_LAUNCH_BALANCE_USD", "0"))
    floor = float(os.environ.get("GLM53_V20_BALANCE_FLOOR_USD", "0"))
    if (
        not pod_id
        or rate <= 0
        or rate > float(runtime["runpod"]["aggregate_gpu_rate_cap_usd_per_hour"])
        or balance - floor > float(runtime["runpod"]["compute_hard_cap_usd"]) + 1e-9
        or floor < float(runtime["runpod"]["minimum_uncommitted_balance_usd"])
    ):
        raise ValueError("paid V21 environment violates its hardware or budget lock")
    if any(os.environ.get(name) for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")):
        raise ValueError("the V21 science process may not inherit S3 credentials")

    from src.glm53_user_eval.v8.whitebox_runtime import verify_model_snapshot

    from src.glm53_user_eval.v21.supervisor import run_exploratory_continuation

    stage = json.loads(
        (ROOT / "artifacts/glm53_user_eval/runtime/g2/model_stage.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot = verify_model_snapshot(args.model_path, stage, full_rehash=True)
    atomic_json(DEFAULT_OUTPUT / "infrastructure/model_snapshot_verification.json", snapshot)
    result = run_exploratory_continuation(
        repo_root=ROOT,
        prereg_path=args.prereg,
        runtime_path=args.runtime,
        v20_runtime_path=DEFAULT_V20_RUNTIME,
        downstream_manifest_path=DEFAULT_DOWNSTREAM,
        model_path=args.model_path,
        output_root=DEFAULT_OUTPUT / "run",
    )
    terminal = {
        "schema_version": "glm53_v21_terminal_state_v1",
        "result": result,
        "scope": "exploratory_post_failed_local_parity_gate",
        "git_commit": commit,
        "pod_id": pod_id,
        "model_snapshot": snapshot,
        "prereg_sha256": validation["prereg_sha256"],
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
