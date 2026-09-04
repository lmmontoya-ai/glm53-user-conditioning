"""Reading raw scores, the roster, configs, and writing outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import CONFIGS, REPO_ROOT

GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai")
PRIMARY_GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_nonai")


def load_yaml(name: str) -> dict[str, Any]:
    """A config file from configs/ as a dict."""
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    """Absolute path for a repository-relative or absolute path string."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """All rows of a JSON-lines file."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def load_roster(task: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The pinned persona roster: group name to ordered list of persona records."""
    task = task or load_yaml("task.yaml")
    return read_json(repo_path(task["roster"]["file"]))


def load_raw_scores(run: str, task: dict[str, Any] | None = None) -> pd.DataFrame:
    """Raw score rows for one run (`discovery` or `confirmatory`).

    Columns: group, persona, stimulus, score (probability in [0, 1] or NaN), confidence_p,
    binary_answer, shard_id.
    """
    task = task or load_yaml("task.yaml")
    rows = read_jsonl(repo_path(task["runs"][run]["raw_scores"]))
    frame = pd.DataFrame(rows)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    return frame


def run_paths(run: str, task: dict[str, Any] | None = None) -> dict[str, Path]:
    task = task or load_yaml("task.yaml")
    return {
        key: repo_path(value)
        for key, value in task["runs"][run].items()
        if isinstance(value, str)
    }
