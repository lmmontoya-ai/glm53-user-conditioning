"""External V16 Pod watchdog.

It deletes only the exact Pod ID supplied on the command line.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _run_json(*args: str) -> Any:
    completed = subprocess.run(
        ["runpodctl", *args, "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def _balance() -> float:
    return float(_run_json("user")["clientBalance"])


def _delete(pod_id: str, evidence_root: Path, reason: str) -> None:
    evidence_root.mkdir(parents=True, exist_ok=True)
    before = _run_json("pod", "get", pod_id, "--include-machine", "--include-network-volume")
    (evidence_root / f"pod_{pod_id}_predelete.json").write_text(
        json.dumps(before, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = _run_json("pod", "delete", pod_id)
    (evidence_root / f"pod_{pod_id}_delete.json").write_text(
        json.dumps({"reason": reason, "result": result}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    after = _run_json("pod", "list", "--all")
    (evidence_root / f"pods_after_{pod_id}_delete.json").write_text(
        json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    records = after.get("pods", after) if isinstance(after, dict) else after
    if any(str(row.get("id")) == pod_id for row in records):
        raise RuntimeError(f"Pod {pod_id} still exists after deletion")


def _s3_json(bucket: str, key: str) -> dict[str, Any] | None:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    client = boto3.client("s3", endpoint_url="https://s3api-us-ks-2.runpod.io/")
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
            raise
        return None
    except BotoCoreError:
        return None
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--balance-floor", required=True, type=float)
    parser.add_argument("--heartbeat-prefix", required=True)
    parser.add_argument("--s3-bucket", default="a9diryunoj")
    parser.add_argument("--evidence-root", type=Path, default=Path("artifacts/glm53_user_eval/v16/infrastructure"))
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline_utc).astimezone(dt.UTC)
    if deadline <= dt.datetime.now(dt.UTC):
        raise ValueError("watchdog deadline has passed")
    if not args.pod_id or len(args.pod_id) < 6:
        raise ValueError("watchdog Pod ID is invalid")
    # Keep Windows awake while the deletion watchdog is responsible for a paid Pod.
    if hasattr(ctypes, "windll"):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    heartbeat_key = f"{args.heartbeat_prefix.rstrip('/')}/heartbeat.json"
    terminal_key = f"{args.heartbeat_prefix.rstrip('/')}/terminal.json"
    last_heartbeat = dt.datetime.now(dt.UTC)
    reason = "deadline"
    try:
        while dt.datetime.now(dt.UTC) < deadline:
            if _balance() <= args.balance_floor:
                reason = "balance_floor"
                break
            terminal = _s3_json(args.s3_bucket, terminal_key)
            if terminal is not None:
                if terminal.get("run_id") != args.run_id or terminal.get("pod_id") != args.pod_id:
                    reason = "invalid_terminal_binding"
                else:
                    reason = "terminal_marker"
                break
            heartbeat = _s3_json(args.s3_bucket, heartbeat_key)
            if heartbeat is not None:
                if heartbeat.get("run_id") != args.run_id or heartbeat.get("pod_id") != args.pod_id:
                    reason = "invalid_heartbeat_binding"
                    break
                created = dt.datetime.fromisoformat(str(heartbeat["created_at_utc"])).astimezone(dt.UTC)
                last_heartbeat = max(last_heartbeat, created)
            if (dt.datetime.now(dt.UTC) - last_heartbeat).total_seconds() > 600:
                reason = "missing_or_stale_heartbeat"
                break
            time.sleep(30)
        _delete(args.pod_id, args.evidence_root, reason)
    finally:
        if hasattr(ctypes, "windll"):
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
