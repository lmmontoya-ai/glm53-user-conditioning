"""Validate the completed non-gating v11 supplemental manual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v11.supplemental_manual_packet import (
    validate_completed_supplemental_review,
    write_supplemental_review_report,
)

DEFAULT_DATASET = ROOT / "artifacts/datasets/contrastive_prompts_v3/samples.jsonl"
DEFAULT_AUDIT_ROOT = ROOT / "artifacts/glm53_user_eval/v11/offline_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the completed six-row supplemental diagnostic review."
    )
    parser.add_argument(
        "--completed-packet",
        type=Path,
        default=(
            DEFAULT_AUDIT_ROOT / "supplemental_semantic_disagreements_completed.csv"
        ),
    )
    parser.add_argument(
        "--frozen-packet",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "supplemental_semantic_disagreements.csv",
    )
    parser.add_argument(
        "--packet-manifest",
        type=Path,
        default=(
            DEFAULT_AUDIT_ROOT / "supplemental_semantic_disagreements_manifest.json"
        ),
    )
    parser.add_argument(
        "--packet-manifest-digest",
        type=Path,
        default=(
            DEFAULT_AUDIT_ROOT / "supplemental_semantic_disagreements_manifest.sha256"
        ),
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "supplemental_semantic_review_report.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "supplemental_semantic_review_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_completed_supplemental_review(
        completed_path=args.completed_packet,
        frozen_packet_path=args.frozen_packet,
        packet_manifest_path=args.packet_manifest,
        packet_manifest_digest_path=args.packet_manifest_digest,
        samples_path=args.samples,
    )
    manifest = write_supplemental_review_report(
        report,
        report_path=args.output_report,
        report_manifest_path=args.output_manifest,
    )
    print(
        json.dumps(
            {"report": report, "report_manifest": manifest},
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed_review_integrity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
