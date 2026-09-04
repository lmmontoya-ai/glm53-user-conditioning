"""Command line entry point for the frozen v10 offline audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v10.analysis import run_audit
from src.glm53_user_eval.v10.data import load_locked_data, sha256_file


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["schema_version"] != "glm53_user_eval_v10_offline_audit_prereg_v1_2":
        raise ValueError("unexpected v10 preregistration schema")
    if config["execution"]["paid_compute_allowed"]:
        raise ValueError("v10 is offline-only")
    return config


def validate(config: dict) -> dict:
    for name, item in config["locked_inputs"].items():
        if name.endswith("_availability"):
            continue
        actual = sha256_file(Path(item["path"]))
        if actual != item["sha256"]:
            raise ValueError(f"locked input {name} differs")
    return {
        "passed": True,
        "locked_input_count": sum(
            not key.endswith("_availability") for key in config["locked_inputs"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/glm53_user_eval/v10/offline_audit/analysis.json"),
    )
    args = parser.parse_args()
    config = load_config(args.prereg)
    print(json.dumps(validate(config), indent=2))
    if args.command == "run":
        data = load_locked_data(
            feature_root=Path(config["feature_root"]),
            samples_path=Path(config["locked_inputs"]["samples"]["path"]),
            expected_hashes={
                "fixed_features": config["locked_inputs"]["fixed_features"]["sha256"],
                "metadata": config["locked_inputs"]["metadata"]["sha256"],
                "feature_manifest": config["locked_inputs"]["feature_manifest"]["sha256"],
                "samples": config["locked_inputs"]["samples"]["sha256"],
            },
        )
        report = run_audit(data, config, args.output)
        print(
            json.dumps(
                {"output": str(args.output), "schema_version": report["schema_version"]}, indent=2
            )
        )


if __name__ == "__main__":
    main()
