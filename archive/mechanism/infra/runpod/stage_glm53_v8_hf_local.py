#!/usr/bin/env python3
"""Stage the pinned GLM-5.3 FP8 snapshot from Hugging Face to Pod-local NVMe."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from infra.runpod.glm53_vllm_local_stage.stage_and_serve import (
    stage_model_from_hub,
)

MODEL_ID = "zai-org/GLM-5.3-Flash"
REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
DEFAULT_TARGET = Path(f"/runpod-local/GLM-5.3-Flash/{REVISION}")
DEFAULT_RUNTIME_LINK = Path(f"/workspace/mats-glm53/models/GLM-5.3-Flash/{REVISION}")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ensure_runtime_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"runtime link points to {link.resolve()}, not {target}")
        return
    if link.exists():
        raise RuntimeError(f"runtime link path already exists and is not a symlink: {link}")
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--runtime-link", type=Path, default=DEFAULT_RUNTIME_LINK)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ.setdefault("HF_XET_CACHE", "/tmp/hf-xet-cache")
    started = time.perf_counter()
    stage_model_from_hub(
        MODEL_ID,
        REVISION,
        args.target,
        args.manifest,
        args.workers,
    )
    _ensure_runtime_link(args.runtime_link, args.target)
    elapsed = time.perf_counter() - started
    sentinel = json.loads((args.target / ".glm53_local_stage.json").read_text())
    report = {
        "schema_version": "glm53_v8_hf_local_stage_v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "target": str(args.target),
        "runtime_link": str(args.runtime_link),
        "elapsed_seconds": elapsed,
        "workers": args.workers,
        "sentinel": sentinel,
        "passed": True,
    }
    _atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
