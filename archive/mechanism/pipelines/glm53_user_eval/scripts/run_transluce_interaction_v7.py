"""Run the preregistered v7 interaction confirmation without exposing scores."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from pipelines.glm53_user_eval.scripts.audit_transluce_exact_v6 import audit_log
from pipelines.glm53_user_eval.scripts.run_transluce_exact_v6 import (
    Shard,
    atomic_json,
    build_shards,
    git_head,
    inspect_command,
    latest_eval_status,
    load_roster,
    plan_payload,
    run_shards,
    sha256_file,
    utc_now,
)

V6_PREREG = ROOT / "pipelines/glm53_user_eval/configs/prereg_v6_transluce_exact.yaml"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_and_validate(prereg_path: Path, source_root: Path) -> dict[str, Any]:
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("schema_version") != "glm53_user_eval_prereg_v7":
        raise ValueError("v7 runner requires the v7 interaction preregistration")
    parent_lock = prereg["parent_generation_contract"]
    parent_path = repo_path(parent_lock["prereg_path"])
    if parent_path.resolve() != V6_PREREG.resolve():
        raise ValueError("v7 parent must be the immutable v6 preregistration")
    if sha256_file(parent_path) != parent_lock["prereg_sha256"]:
        raise ValueError("v6 preregistration hash mismatch")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    for section in parent_lock["equal_sections"]:
        if prereg.get(section) != parent.get(section):
            raise ValueError(f"scientific contract differs from v6: {section}")
    reference = prereg["reference_contract"]
    if git_head(source_root) != reference["commit"]:
        raise ValueError("Transluce checkout commit mismatch")
    for relative, expected in reference["files"].items():
        if sha256_file(source_root / relative) != expected:
            raise ValueError(f"Transluce source hash mismatch: {relative}")
    discovery = prereg["discovery_result"]
    for field in ("analysis", "decision"):
        if sha256_file(repo_path(discovery[f"{field}_path"])) != discovery[f"{field}_sha256"]:
            raise ValueError(f"v6 discovery {field} hash mismatch")
    if prereg["subject"]["provider"] != "Novita":
        raise ValueError("v7 route must remain Novita")
    return prereg


def verify_repo_state(required_commit: str | None, *, allow_dirty: bool) -> None:
    head = git("rev-parse", "HEAD")
    if required_commit and head != required_commit:
        raise ValueError(f"running commit {head} differs from required commit {required_commit}")
    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty:
        raise ValueError("repository is dirty; commit the preregistration and code before calls")


def v7_plan(prereg_path: Path, prereg: dict[str, Any], source_root: Path) -> dict[str, Any]:
    payload = plan_payload(prereg_path, prereg, source_root, build_shards(prereg, source_root))
    payload["schema_version"] = "glm53_transluce_interaction_v7_plan_v1"
    payload["project_id"] = prereg["project_id"]
    payload["primary_estimand"] = prereg["analysis"]["primary_estimand"]["name"]
    return payload


def write_preflight(
    output_root: Path,
    prereg_path: Path,
    prereg: dict[str, Any],
    source_root: Path,
) -> None:
    try:
        import inspect_ai

        inspect_version = getattr(inspect_ai, "__version__", "unknown")
    except ImportError:  # pragma: no cover - environment diagnostic
        inspect_version = "unavailable"
    payload = {
        "schema_version": "glm53_transluce_interaction_v7_preflight_v1",
        "created_at_utc": utc_now(),
        "project_id": prereg["project_id"],
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(git("status", "--porcelain")),
        "prereg_sha256": sha256_file(prereg_path),
        "source_commit": git_head(source_root),
        "python": sys.version,
        "inspect_ai": inspect_version,
        "hostname": platform.node(),
        "credential_recorded": False,
    }
    atomic_json(output_root / "preflight.json", payload)


def completed_shards(run_root: Path, shards: list[Shard]) -> tuple[list[Shard], list[Shard]]:
    complete: list[Shard] = []
    pending: list[Shard] = []
    for shard in shards:
        _path, status = latest_eval_status(run_root / "eval_logs" / shard.shard_id)
        (complete if status == "success" else pending).append(shard)
    return complete, pending


def projected_cost(run_root: Path) -> float | None:
    audits = sorted((run_root / "audits").glob("*.json")) if (run_root / "audits").exists() else []
    candidates: list[tuple[int, float]] = []
    for path in audits:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            count = int(payload.get("audited_shards", 0))
            cost = float(payload.get("api_cost_usd", 0.0))
            if count > 0:
                candidates.append((count, cost))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    count, cost = max(candidates)
    return cost / count * 100


def smoke(
    prereg_path: Path,
    prereg: dict[str, Any],
    source_root: Path,
    output_root: Path,
    connections: int,
) -> dict[str, Any]:
    if not os.environ.get(prereg["subject"]["credential_env"]):
        raise RuntimeError("OPENROUTER_API_KEY is not present")
    os.environ["OPENROUTER_PROVIDER"] = "Novita"
    roster = load_roster(source_root)
    key = next(
        row["key"] for row in roster["famous_ai_real"] if row["key"] == "fai2r_amanda_askell"
    )
    shard = Shard("famous_ai_real", 0, 1, (str(key), "anon"))
    smoke_root = output_root / "route_smoke"
    log_dir = smoke_root / "eval_logs" / shard.shard_id
    log_dir.mkdir(parents=True, exist_ok=True)
    command = inspect_command(
        source_root=source_root,
        prereg=prereg,
        shard=shard,
        log_dir=log_dir,
        connections=connections,
    )
    transcript = smoke_root / "orchestration.log"
    with transcript.open("ab") as handle:
        completed = subprocess.run(
            command,
            cwd=source_root,
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    newest, status = latest_eval_status(log_dir)
    if completed.returncode or status != "success" or newest is None:
        raise RuntimeError(f"route smoke failed: return={completed.returncode}, status={status}")
    audit = audit_log(Path(newest), shard.expected_rows)
    audit["excluded_from_scientific_manifest"] = True
    atomic_json(smoke_root / "route_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("route smoke contract audit failed")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "smoke", "run", "status"))
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parallel-shards", type=int, default=5)
    parser.add_argument("--connections-per-shard", type=int, default=40)
    parser.add_argument("--max-new-shards", type=int)
    parser.add_argument("--require-prereg-commit")
    parser.add_argument("--budget-cap-usd", type=float, default=50.0)
    parser.add_argument("--score-output-locked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prereg_path = args.prereg.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    prereg = load_and_validate(prereg_path, source_root)
    shards = build_shards(prereg, source_root)
    complete, pending = completed_shards(output_root, shards)
    if args.command == "status":
        print(
            json.dumps(
                {
                    "completed_shards": len(complete),
                    "pending_shards": len(pending),
                    "scores_exposed": False,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "plan":
        payload = v7_plan(prereg_path, prereg, source_root)
        atomic_json(output_root / "schedule_manifest.json", payload)
        write_preflight(output_root, prereg_path, prereg, source_root)
        print(
            json.dumps(
                {
                    "shard_count": payload["shard_count"],
                    "total_expected_rows": payload["total_expected_rows"],
                    "groups": prereg["population"]["groups_in_reference_order"],
                    "scores_exposed": False,
                },
                indent=2,
            )
        )
        return 0
    verify_repo_state(args.require_prereg_commit, allow_dirty=False)
    if not args.require_prereg_commit:
        raise ValueError("scientific calls require --require-prereg-commit")
    if not args.score_output_locked:
        raise ValueError("scientific calls require --score-output-locked")
    write_preflight(output_root, prereg_path, prereg, source_root)
    if args.command == "smoke":
        result = smoke(prereg_path, prereg, source_root, output_root, args.connections_per_shard)
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "row_count": result["row_count"],
                    "scores_exposed": False,
                },
                indent=2,
            )
        )
        return 0
    cap = min(float(args.budget_cap_usd), float(prereg["budget"]["incremental_api_cap_usd"]))
    estimate = projected_cost(output_root)
    if estimate is not None and estimate > cap:
        raise RuntimeError(f"projected full-run cost ${estimate:.2f} exceeds cap ${cap:.2f}")
    selected = pending[: args.max_new_shards] if args.max_new_shards is not None else pending
    if not selected:
        print(
            json.dumps(
                {"selected_shards": 0, "completed_shards": len(complete), "scores_exposed": False},
                indent=2,
            )
        )
        return 0
    # The copied mapping is intentionally unchanged; v6 functions do not inspect schema.
    state = run_shards(
        prereg_path=prereg_path,
        prereg=copy.deepcopy(prereg),
        source_root=source_root,
        output_root=output_root,
        shards=selected,
        parallel_shards=args.parallel_shards,
        connections_per_shard=args.connections_per_shard,
    )
    atomic_json(
        output_root / "schedule_manifest.json",
        v7_plan(prereg_path, prereg, source_root),
    )
    print(json.dumps(state["summary"] | {"scores_exposed": False}, indent=2, sort_keys=True))
    return 0 if state["summary"]["failed_or_incomplete"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
