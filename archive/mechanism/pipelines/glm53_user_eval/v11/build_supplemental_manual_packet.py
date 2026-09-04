"""Build the non-gating v11 semantic-disagreement review packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v11.supplemental_manual_packet import (
    build_supplemental_disagreement_packet,
)

DEFAULT_DATASET = ROOT / "artifacts/datasets/contrastive_prompts_v3/samples.jsonl"
DEFAULT_AUDIT_ROOT = ROOT / "artifacts/glm53_user_eval/v11/offline_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the six-row non-gating semantic-disagreement review packet."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--judgment-rows-dir",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "semantic_judge/rows",
    )
    parser.add_argument(
        "--original-packet",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "manual_packet.csv",
    )
    parser.add_argument(
        "--semantic-validation",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "semantic_validation.json",
    )
    parser.add_argument(
        "--output-packet",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "supplemental_semantic_disagreements.csv",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=(
            DEFAULT_AUDIT_ROOT / "supplemental_semantic_disagreements_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-manifest-digest",
        type=Path,
        default=(
            DEFAULT_AUDIT_ROOT / "supplemental_semantic_disagreements_manifest.sha256"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_supplemental_disagreement_packet(
        samples_path=args.samples,
        judgment_rows_dir=args.judgment_rows_dir,
        original_packet_path=args.original_packet,
        semantic_validation_path=args.semantic_validation,
        packet_path=args.output_packet,
        manifest_path=args.output_manifest,
        manifest_digest_path=args.output_manifest_digest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
