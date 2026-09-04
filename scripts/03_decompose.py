"""Stage 3: choice interaction, choice-standardized confidence, and same-choice matched estimate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, finish, print_plan, provenance, stage_parser  # noqa: E402

from glm53.decompose import (  # noqa: E402
    bootstrap_interaction,
    bootstrap_matched_same_choice,
    bootstrap_standardized,
    outcome_matrices,
    stratified,
)
from glm53.io import load_raw_scores, load_roster, load_yaml, read_json, repo_path  # noqa: E402

STAGE = "decompose"


def compare(observed: float, pinned: float, tol: float) -> dict[str, object]:
    diff = float(observed - pinned)
    return {"observed": observed, "pinned": pinned, "diff": diff, "within_tolerance": abs(diff) <= tol}


def main() -> int:
    parser = stage_parser(STAGE, __doc__)
    parser.add_argument("--reps", type=int)
    args = parser.parse_args()
    task = load_yaml("task.yaml")
    analysis = load_yaml("analysis.yaml")
    inputs = [repo_path(task["runs"]["confirmatory"]["raw_scores"]), repo_path(task["roster"]["file"])]
    committed_path = repo_path("data/judgments/reports/deterministic_analysis.json")
    if args.dry_run:
        print_plan(STAGE, inputs + [committed_path], [args.output_root / "summary.json"], {"api_calls": 0})
        return 0
    reps = args.reps or int(analysis["bootstrap"]["reps"])
    seed = int(analysis["bootstrap"]["seeds"]["decomposition"])
    tol_point = float(analysis["tolerances"]["point_estimate"])
    tol_ci = float(analysis["tolerances"]["ci_endpoint"])
    roster = load_roster(task)
    choice, conf = outcome_matrices(load_raw_scores("confirmatory", task), roster)
    choice_result = bootstrap_interaction(choice, reps=reps, seed=seed, scale=100.0)
    folded_result = bootstrap_interaction(conf, reps=reps, seed=seed + 1)
    strata = stratified(conf, choice, reps=reps, seed=seed)
    standardized = bootstrap_standardized(conf, choice, reps=reps, seed=seed + 30)
    matched = bootstrap_matched_same_choice(conf, choice, reps=reps, seed=seed + 31)
    summary = {
        "reps": reps,
        "seed": seed,
        "units": "percentage points; choice interaction is yes-rate in percentage points",
        "yes_rate_interaction": choice_result,
        "folded_confidence_interaction": folded_result,
        "folded_confidence_by_choice": strata,
        "choice_standardized_confidence_interaction": standardized,
        "same_choice_matched": matched,
        "choice_equivalence_margin_pp": 1.0,
        "choice_equivalent_to_zero": choice_result["ci95"][0] > -1.0 and choice_result["ci95"][1] < 1.0,
    }
    checks = {}
    if committed_path.exists():
        committed = read_json(committed_path)
        c = committed["first_turn_choice_interaction_pp"]
        s = committed["choice_standardized_folded_confidence"]
        m = committed["matched_same_choice_folded_confidence"]
        checks = {
            "choice_point": compare(choice_result["interaction"], c["interaction"], tol_point),
            "choice_ci_lower": compare(choice_result["ci95"][0], c["ci95"][0], tol_ci),
            "choice_ci_upper": compare(choice_result["ci95"][1], c["ci95"][1], tol_ci),
            "folded_point": compare(folded_result["interaction"], committed["folded_confidence_interaction_pp"]["interaction"], tol_point),
            "folded_ci_lower": compare(folded_result["ci95"][0], committed["folded_confidence_interaction_pp"]["ci95"][0], tol_ci),
            "folded_ci_upper": compare(folded_result["ci95"][1], committed["folded_confidence_interaction_pp"]["ci95"][1], tol_ci),
            "standardized_point": compare(standardized["interaction_pp"], s["interaction_pp"], tol_point),
            "standardized_ci_lower": compare(standardized["ci95_pp"][0], s["ci95_pp"][0], tol_ci),
            "standardized_ci_upper": compare(standardized["ci95_pp"][1], s["ci95_pp"][1], tol_ci),
            "matched_point": compare(matched["interaction_pp"], m["interaction_pp"], tol_point),
            "matched_fu_cells": compare(matched["famous_unknown_retained_cells"], m["famous_unknown_retained_cells"], 0),
            "matched_fng_cells": compare(matched["famous_nonai_genpop_retained_cells"], m["famous_nonai_genpop_retained_cells"], 0),
        }
    failures = [k for k, v in checks.items() if not v["within_tolerance"]]
    summary["checks"] = checks
    summary["all_checks_pass"] = not failures
    summary["failures"] = failures
    summary["note"] = "The same-choice matched interval is new; the committed decomposition reported that estimate without an interval."
    summary["provenance"] = provenance(inputs + [committed_path], [REPO_ROOT / "configs/analysis.yaml"])
    finish(STAGE, args.output_root, summary)
    print(f"checks: {len(checks)}, failures: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
