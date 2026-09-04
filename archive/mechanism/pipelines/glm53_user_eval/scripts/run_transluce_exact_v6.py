"""Run the pinned Transluce confidence lane in small, resumable Inspect shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.log import read_eval_log


GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai")


@dataclass(frozen=True)
class Shard:
    group: str
    offset: int
    limit: int
    persona_keys: tuple[str, ...]

    @property
    def shard_id(self) -> str:
        return f"{self.group}__offset{self.offset:03d}__limit{self.limit:03d}"

    @property
    def expected_rows(self) -> int:
        return len(self.persona_keys) * self.limit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_and_validate(prereg_path: Path, source_root: Path) -> dict[str, Any]:
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("schema_version") != "glm53_user_eval_prereg_v6":
        raise ValueError("strict runner requires preregistration v6")
    reference = prereg["reference_contract"]
    if git_head(source_root) != reference["commit"]:
        raise ValueError("Transluce checkout does not match the preregistered commit")
    for relative, expected in reference["files"].items():
        observed = sha256_file(source_root / relative)
        if observed != expected:
            raise ValueError(f"source hash mismatch: {relative}")
    subject = prereg["subject"]
    if subject["provider"] != "Novita" or subject["reasoning_effort"] != "high":
        raise ValueError("strict route must use Novita with high reasoning")
    if subject["replacement_model"] != "openrouter/z-ai/glm-5.3-flash":
        raise ValueError("strict subject model changed")
    return prereg


def load_roster(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads((source_root / "core/personas2.json").read_text(encoding="utf-8"))
    for group in GROUPS:
        if not isinstance(payload.get(group), list):
            raise ValueError(f"missing roster group: {group}")
    return payload


def build_shards(prereg: dict[str, Any], source_root: Path) -> list[Shard]:
    roster = load_roster(source_root)
    per_shard = int(prereg["execution"]["dilemmas_per_shard"])
    dilemma_count = int(prereg["population"]["dilemma_count"])
    shards: list[Shard] = []
    for offset in range(0, dilemma_count, per_shard):
        for group in GROUPS:
            keys = tuple(str(row["key"]) for row in roster[group]) + ("anon",)
            shards.append(
                Shard(
                    group=group,
                    offset=offset,
                    limit=min(per_shard, dilemma_count - offset),
                    persona_keys=keys,
                )
            )
    expected_rows = int(prereg["population"]["expected_scientific_rows"])
    if len(shards) != int(prereg["execution"]["shard_count"]):
        raise ValueError("generated shard count differs from preregistration")
    if sum(shard.expected_rows for shard in shards) != expected_rows:
        raise ValueError("generated scientific row count differs from preregistration")
    return shards


def latest_eval_status(log_dir: Path) -> tuple[str | None, str | None]:
    paths = sorted(log_dir.glob("*.eval"), key=lambda path: path.stat().st_mtime)
    if not paths:
        return None, None
    newest = paths[-1]
    try:
        status = str(read_eval_log(newest, header_only=True).status)
    except Exception as exc:  # noqa: BLE001 - preserve the log and rerun the shard
        return str(newest), f"unreadable:{type(exc).__name__}"
    return str(newest), status


def inspect_executable() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(sys.executable).resolve().parent / f"inspect{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Inspect executable not found beside interpreter: {path}")
    return path


def inspect_command(
    *,
    source_root: Path,
    prereg: dict[str, Any],
    shard: Shard,
    log_dir: Path,
    connections: int,
) -> list[str]:
    subject = prereg["subject"]
    return [
        str(inspect_executable()),
        "eval",
        "evals/pmisaligned/task.py@pmisaligned",
        "--model",
        str(subject["replacement_model"]),
        "-T",
        f"personas={','.join(shard.persona_keys)}",
        "-T",
        f"reasoning_effort={subject['reasoning_effort']}",
        "-T",
        "mode=plain",
        "-T",
        "dataset=dailydilemmas.json",
        "-T",
        "warmup=false",
        "-T",
        f"offset={shard.offset}",
        "-T",
        f"limit={shard.limit}",
        "-T",
        "confidence=true",
        "-T",
        "confidence_style=confidence",
        "-T",
        "seed=all",
        "--max-samples",
        str(connections),
        "--max-connections",
        str(connections),
        "--retry-on-error",
        "3",
        "--fail-on-error",
        "0.05",
        "--log-dir",
        str(log_dir),
        "--display",
        "plain",
    ]


def plan_payload(
    prereg_path: Path,
    prereg: dict[str, Any],
    source_root: Path,
    shards: list[Shard],
) -> dict[str, Any]:
    return {
        "schema_version": "glm53_transluce_exact_plan_v1",
        "created_at_utc": utc_now(),
        "prereg_path": str(prereg_path.resolve()),
        "prereg_sha256": sha256_file(prereg_path),
        "transluce_root": str(source_root.resolve()),
        "transluce_commit": git_head(source_root),
        "subject_model": prereg["subject"]["replacement_model"],
        "provider": prereg["subject"]["provider"],
        "reasoning_effort": prereg["subject"]["reasoning_effort"],
        "total_expected_rows": sum(shard.expected_rows for shard in shards),
        "shard_count": len(shards),
        "shards": [asdict(shard) | {"shard_id": shard.shard_id, "expected_rows": shard.expected_rows} for shard in shards],
    }


def run_shards(
    *,
    prereg_path: Path,
    prereg: dict[str, Any],
    source_root: Path,
    output_root: Path,
    shards: list[Shard],
    parallel_shards: int,
    connections_per_shard: int,
) -> dict[str, Any]:
    if parallel_shards * connections_per_shard > int(prereg["execution"]["total_max_connections"]):
        raise ValueError("requested concurrency exceeds the preregistered total")
    if not os.environ.get(str(prereg["subject"]["credential_env"])):
        raise RuntimeError("OPENROUTER_API_KEY is not present in the process environment")
    os.environ["OPENROUTER_PROVIDER"] = str(prereg["subject"]["provider"])
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "eval_logs"
    orchestration_root = output_root / "orchestration"
    logs_root.mkdir(exist_ok=True)
    orchestration_root.mkdir(exist_ok=True)
    atomic_json(output_root / "schedule_manifest.json", plan_payload(prereg_path, prereg, source_root, build_shards(prereg, source_root)))

    state: dict[str, Any] = {
        "schema_version": "glm53_transluce_exact_run_state_v1",
        "updated_at_utc": utc_now(),
        "parallel_shards": parallel_shards,
        "connections_per_shard": connections_per_shard,
        "total_connection_cap": parallel_shards * connections_per_shard,
        "shards": {},
    }
    state_lock = threading.Lock()

    def update_state(shard_id: str, payload: dict[str, Any]) -> None:
        with state_lock:
            state["shards"][shard_id] = payload
            state["updated_at_utc"] = utc_now()
            atomic_json(output_root / "run_state.json", state)

    def worker(shard: Shard) -> None:
        log_dir = logs_root / shard.shard_id
        log_dir.mkdir(parents=True, exist_ok=True)
        newest, status = latest_eval_status(log_dir)
        if status == "success":
            update_state(
                shard.shard_id,
                {"status": "skipped_success", "eval_log": newest, "expected_rows": shard.expected_rows},
            )
            return
        started = utc_now()
        command = inspect_command(
            source_root=source_root,
            prereg=prereg,
            shard=shard,
            log_dir=log_dir,
            connections=connections_per_shard,
        )
        transcript_path = orchestration_root / f"{shard.shard_id}.log"
        update_state(
            shard.shard_id,
            {"status": "running", "started_at_utc": started, "expected_rows": shard.expected_rows},
        )
        with transcript_path.open("ab") as transcript:
            completed = subprocess.run(
                command,
                cwd=source_root,
                env=os.environ.copy(),
                stdout=transcript,
                stderr=subprocess.STDOUT,
                check=False,
            )
        newest, status = latest_eval_status(log_dir)
        payload = {
            "status": status or "missing_eval",
            "return_code": int(completed.returncode),
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "eval_log": newest,
            "transcript": str(transcript_path),
            "expected_rows": shard.expected_rows,
        }
        update_state(shard.shard_id, payload)

    # Avoid asyncio entirely: the CPython Proactor event loop intermittently corrupts its
    # ready queue under this Windows workload. A bounded native thread pool preserves the
    # exact Inspect commands and caps live subprocesses without involving an event loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_shards) as executor:
        futures = {executor.submit(worker, shard): shard for shard in shards}
        for future in concurrent.futures.as_completed(futures):
            shard = futures[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - defensive orchestration path
                update_state(
                    shard.shard_id,
                    {
                        "status": "supervisor_exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_at_utc": utc_now(),
                        "expected_rows": shard.expected_rows,
                    },
                )
    statuses = [entry["status"] for entry in state["shards"].values()]
    state["summary"] = {
        "selected_shards": len(shards),
        "successful_or_previously_successful": sum(status in {"success", "skipped_success"} for status in statuses),
        "failed_or_incomplete": sum(status not in {"success", "skipped_success"} for status in statuses),
    }
    state["updated_at_utc"] = utc_now()
    atomic_json(output_root / "run_state.json", state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "smoke", "run"))
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parallel-shards", type=int, default=5)
    parser.add_argument("--connections-per-shard", type=int, default=40)
    parser.add_argument("--max-shards", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prereg_path = args.prereg.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    prereg = load_and_validate(prereg_path, source_root)
    shards = build_shards(prereg, source_root)
    if args.command == "plan":
        payload = plan_payload(prereg_path, prereg, source_root, shards)
        atomic_json(output_root / "schedule_manifest.json", payload)
        print(
            json.dumps(
                {
                    "schema_version": payload["schema_version"],
                    "shard_count": payload["shard_count"],
                    "total_expected_rows": payload["total_expected_rows"],
                    "provider": payload["provider"],
                    "reasoning_effort": payload["reasoning_effort"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "smoke":
        roster = load_roster(source_root)
        smoke_key = next(
            row["key"]
            for row in roster["famous_ai_real"]
            if row["key"] == "fai2r_amanda_askell"
        )
        shards = [Shard("famous_ai_real", 0, 1, (str(smoke_key), "anon"))]
    elif args.max_shards is not None:
        shards = shards[: args.max_shards]
    state = run_shards(
        prereg_path=prereg_path,
        prereg=prereg,
        source_root=source_root,
        output_root=output_root,
        shards=shards,
        parallel_shards=args.parallel_shards,
        connections_per_shard=args.connections_per_shard,
    )
    print(json.dumps(state["summary"], indent=2, sort_keys=True))
    return 0 if state["summary"]["failed_or_incomplete"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
