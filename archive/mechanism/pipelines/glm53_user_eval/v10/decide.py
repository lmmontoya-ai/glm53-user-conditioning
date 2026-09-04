"""Build the fail-closed v10 construct-validity decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    trainers = analysis["fixed_split_trainer_comparison"]
    trainer_names = [
        name for name, row in trainers.items() if isinstance(row, dict) and "holdout_hardneg" in row
    ]
    checks = {
        "independent_verification": bool(verification["passed"]),
        "no_trainer_passes_hard_negative_gate": all(
            trainers[name]["holdout_hardneg"]["fpr_at_80_tpr"] > 0.30 for name in trainer_names
        ),
        "historical_trainer_does_not_fix_gate": trainers["historical_full_batch"][
            "holdout_hardneg"
        ]["fpr_at_80_tpr"]
        > 0.30,
        "ordinary_family_transfer_remains_high": analysis["leave_family_out"]["median_auroc"]
        >= 0.75,
        "family_and_task_blocked_transfer_remains_high": analysis["family_and_task_blocked"][
            "median_auroc"
        ]
        >= 0.75,
        "simple_metadata_is_not_predictive": analysis["metadata_baseline"]["test"]["auroc"] <= 0.60,
        "bootstrap_has_unstable_tail": analysis["direction_stability"]["cosine_to_full"]["minimum"]
        < 0.50,
    }
    decision = {
        "schema_version": "glm53_v10_construct_decision_v1",
        "passed": all(checks.values()),
        "decision": "construct_overfit_source_instrument_invalid"
        if all(checks.values())
        else "mixed_or_unresolved_diagnosis",
        "checks": checks,
        "interpretation": "The core prompt distribution is easy across trainers, families, and blocked task identities, but every tested linear method fails the counterfactual hard-negative gate. The historical trainer does not repair the failure, and simple saved metadata is at chance. The source instrument decodes propagated explicit framing rather than a robust latent eval/deploy state.",
        "authorization": {"user_recruitment": False, "steering": False, "new_paid_compute": False},
        "recommended_next": "Design and preregister contrastive_prompts_v3 with latent factorial scenarios, counterbalanced vocabulary, disjoint generators and tasks, and a fresh untouched counterfactual holdout.",
        "inputs": {"analysis": sha256(args.analysis), "verification": sha256(args.verification)},
    }
    write(args.output, decision)
    print(json.dumps({"decision": decision["decision"], "passed": decision["passed"]}, indent=2))


if __name__ == "__main__":
    main()
