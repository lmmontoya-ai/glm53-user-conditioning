"""Source-lock validation and immutable hashing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .schemas import SourceLocks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_source_locks(path: Path) -> SourceLocks:
    return SourceLocks.model_validate_json(path.read_text(encoding="utf-8"))


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_git_checkout(path: Path, expected: str) -> None:
    observed = git_head(path)
    if observed != expected:
        raise ValueError(f"source checkout mismatch for {path}: {observed} != {expected}")


def validate_model_metadata(
    locks: SourceLocks,
    *,
    observed_revision: str,
    safetensor_shards: int,
    safetensor_bytes: int,
) -> None:
    expected = locks.model
    if observed_revision != expected.revision:
        raise ValueError("Hugging Face model revision differs from source lock")
    if safetensor_shards != expected.safetensor_shards:
        raise ValueError("Hugging Face safetensor shard count differs from source lock")
    if safetensor_bytes != expected.safetensor_bytes:
        raise ValueError("Hugging Face safetensor byte count differs from source lock")
