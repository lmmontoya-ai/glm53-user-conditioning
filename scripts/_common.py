"""Shared CLI plumbing for the stage scripts: paths, dry-run printing, output stamping."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from glm53.io import sha256_file, write_json  # noqa: E402


def stage_parser(stage: str, description: str) -> argparse.ArgumentParser:
    """Argument parser with the flags every stage supports."""
    parser = argparse.ArgumentParser(prog=f"{stage}", description=description)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print inputs, outputs, and (for API stages) projected rows and cost, then exit",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / stage,
        help="directory for this stage's outputs",
    )
    return parser


def print_plan(stage: str, inputs: list[Path], outputs: list[Path], extra: dict[str, Any] | None = None) -> None:
    """Dry-run report: what would run, which inputs exist, and which outputs would be written."""
    payload: dict[str, Any] = {
        "stage": stage,
        "would_run": True,
        "inputs": [
            {"path": rel(path), "exists": Path(path).exists(), "bytes": Path(path).stat().st_size if Path(path).exists() else None}
            for path in inputs
        ],
        "outputs": [rel(path) for path in outputs],
    }
    if extra:
        payload.update(extra)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def rel(path: Path) -> str:
    """Repository-relative POSIX path when possible, else the absolute path."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def provenance(inputs: list[Path], configs: list[Path]) -> dict[str, Any]:
    """Hashes of the inputs and configs a stage read, plus a timestamp and host type."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "inputs": {rel(p): sha256_file(p) for p in inputs if Path(p).is_file()},
        "configs": {rel(p): sha256_file(p) for p in configs if Path(p).is_file()},
    }


def finish(stage: str, output_root: Path, summary: dict[str, Any]) -> None:
    """Write the stage summary and print a one-line status."""
    write_json(output_root / "summary.json", summary)
    print(f"{stage}: wrote {rel(output_root / 'summary.json')}")
