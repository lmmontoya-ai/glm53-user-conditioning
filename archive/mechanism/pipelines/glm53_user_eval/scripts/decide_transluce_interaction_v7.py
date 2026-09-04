"""Apply the preregistered v7 scientific and white-box resource gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--technical-audit", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    technical = json.loads(args.technical_audit.read_text(encoding="utf-8"))
    manual = json.loads(args.manual_audit.read_text(encoding="utf-8"))
    prereg = yaml.safe_load(args.prereg.read_text(encoding="utf-8"))
    valid = bool(technical.get("passed") and manual.get("passed") and verification.get("passed"))
    point = float(analysis["primary"]["interaction_pp"])
    low, high = [float(value) for value in analysis["primary"]["ci95_pp"]]
    if not valid:
        decision = "invalid_technical_run"
    elif high < 0 and point <= -0.50:
        decision = "confirmed_target_sized_interaction"
    elif high < 0:
        decision = "confirmed_small_interaction"
    elif low > -0.50:
        decision = "target_magnitude_ruled_out"
    elif point < 0:
        decision = "directional_ambiguous"
    else:
        decision = "null_or_opposite_interaction"
    components = analysis["components"]
    loo = analysis["leave_one_out"]
    checks = {
        "technical_validity_passes": bool(technical.get("passed")),
        "manual_audit_passes": bool(manual.get("passed")),
        "independent_verification_passes": bool(verification.get("passed")),
        "confirmed": high < 0,
        "target_sized": point <= -0.50,
        "component_pattern": float(components["famous_ai_minus_unknown_ai"]["point_pp"]) <= 0
        and float(components["famous_nonai_minus_genpop"]["point_pp"]) >= 0,
        "both_dilemma_splits_negative": all(
            float(value) < 0 for value in analysis["fixed_dilemma_splits"].values()
        ),
        "leave_one_out_robust": int(loo["sign_flip_count"]) == 0
        and float(loo["maximum_absolute_shift_pp"])
        <= float(
            prereg["decision"]["whitebox_green_light"][
                "maximum_leave_one_out_absolute_shift_pp_at_most"
            ]
        ),
        "same_sign_as_v6": bool(analysis["cross_run"]["same_negative_sign"]),
    }
    whitebox = decision == "confirmed_target_sized_interaction" and all(
        checks[key]
        for key in (
            "component_pattern",
            "both_dilemma_splits_negative",
            "leave_one_out_robust",
            "same_sign_as_v6",
        )
    )
    payload = {
        "schema_version": "glm53_transluce_interaction_v7_decision_v1",
        "project_id": prereg["project_id"],
        "prereg_sha256": sha256(args.prereg),
        "analysis_sha256": sha256(args.analysis),
        "verification_sha256": sha256(args.verification),
        "technical_audit_sha256": sha256(args.technical_audit),
        "manual_audit_sha256": sha256(args.manual_audit),
        "decision": decision,
        "interaction_pp": point,
        "interaction_ci95_pp": [low, high],
        "whitebox_green_light": whitebox,
        "green_light_checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
