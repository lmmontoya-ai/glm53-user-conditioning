"""Local planning CLI for the V22 information-substitution experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v22.decision import decide
from src.glm53_user_eval.v22.power import (
    atomic_json,
    build_power_report,
    read_prereg,
    sha256_file,
    validate_parent_locks,
)

V22 = ROOT / "pipelines/glm53_user_eval/v22"
DEFAULT_PREREG = V22 / "configs/prereg_v22_information_substitution.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/glm53_user_eval/v22/power"


def validate(prereg_path: Path) -> dict[str, object]:
    prereg = read_prereg(prereg_path)
    locks = validate_parent_locks(ROOT, prereg)
    return {
        "schema_version": "glm53_v22_prereg_validation_v1",
        "passed": True,
        "project_id": prereg["project_id"],
        "parent_hashes": locks,
        "prereg_sha256": sha256_file(prereg_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-prereg", "power", "decide"))
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "validate-prereg":
        payload = validate(args.prereg)
        atomic_json(args.output_root / "prereg_validation.json", payload)
    elif args.command == "power":
        payload = build_power_report(ROOT, args.prereg)
        atomic_json(args.output_root / "power_report.json", payload)
    else:
        report_path = args.output_root / "power_report.json"
        if not report_path.exists():
            raise ValueError("run the locked power calculation first")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        payload = decide(report)
        atomic_json(args.output_root / "decision.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

