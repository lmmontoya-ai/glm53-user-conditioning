"""Standalone command for the V11 semantic-stop evidence packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v11.semantic_stop_evidence import (
    DEFAULT_PREREG_TAG,
    build_semantic_stop_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic evidence for the V11 semantic stop."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=ROOT / "artifacts/glm53_user_eval/v11/offline_audit",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "artifacts/datasets/contrastive_prompts_v3/samples.jsonl",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=(
            ROOT
            / "pipelines/glm53_user_eval/v11/configs/"
            "prereg_v11_source_instrument.yaml"
        ),
    )
    parser.add_argument("--tag", default=DEFAULT_PREREG_TAG)
    parser.add_argument("--expected-row-count", type=int, default=576)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_semantic_stop_evidence(
        repo_root=args.repo_root,
        audit_root=args.audit_root,
        samples_path=args.samples,
        prereg_path=args.prereg,
        tag=args.tag,
        expected_row_count=args.expected_row_count,
        summary_path=args.summary,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
