"""Stage 2: the estimands for each run with 95% intervals, as one tidy CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, finish, print_plan, provenance, stage_parser  # noqa: E402

from glm53.bootstrap import bootstrap_contrasts, bootstrap_group_delta  # noqa: E402
from glm53.io import GROUPS, load_raw_scores, load_roster, load_yaml, repo_path, write_csv  # noqa: E402
from glm53.measure import build_matrices, estimands, group_means, group_statistics  # noqa: E402

STAGE = "estimands"
LABELS = {
    "U-G": "unknown AI-affiliated twin minus general population",
    "F-U": "famous AI figure minus unknown twin",
    "FN-G": "famous non-AI figure minus general population",
    "interaction": "(F-U) minus (FN-G)",
    "F-G": "famous AI figure minus general population",
    "Freal-G": "famous AI figure with published address minus general population",
}


def main() -> int:
    parser = stage_parser(STAGE, __doc__)
    parser.add_argument("--reps", type=int)
    args = parser.parse_args()
    task = load_yaml("task.yaml")
    analysis = load_yaml("analysis.yaml")
    inputs = [repo_path(task["runs"][run]["raw_scores"]) for run in ("discovery", "confirmatory")]
    inputs.append(repo_path(task["roster"]["file"]))
    outputs = [args.output_root / "estimands.csv", args.output_root / "summary.json"]
    if args.dry_run:
        print_plan(STAGE, inputs, outputs, {"api_calls": 0})
        return 0
    reps = args.reps or int(analysis["bootstrap"]["reps"])
    seeds = analysis["bootstrap"]["seeds"]
    roster = load_roster(task)
    records = []
    summary: dict[str, object] = {"reps": reps, "runs": {}}
    for run_name, seed in (("discovery", "discovery_interaction"), ("confirmatory", "confirmatory_interaction")):
        run = build_matrices(load_raw_scores(run_name, task), roster)
        means = group_means(run.matrices, GROUPS)
        est = estimands(means)
        boot = bootstrap_contrasts(run.matrices, reps=reps, seed=int(seeds[seed]))
        stats = group_statistics(run)
        real_ci = bootstrap_group_delta(
            run.matrices["famous_ai_real"], run.matrices["genpop"], reps=reps, seed=int(seeds[seed]) + 3
        )
        n_rows = {g: int(np.isfinite(run.matrices[g]).sum()) for g in GROUPS}
        for name in ("U-G", "F-U", "FN-G", "interaction", "F-G", "Freal-G"):
            if name == "Freal-G":
                lo, hi = real_ci
                p_mw = stats["famous_ai_real"]["mann_whitney_p_vs_genpop"]
                n_id = run.matrices["famous_ai_real"].shape[0]
                rows = n_rows["famous_ai_real"] + n_rows["genpop"]
            else:
                lo, hi = boot["ci95"][name]
                p_mw = stats["famous_ai"]["mann_whitney_p_vs_genpop"] if name == "F-G" else None
                involved = {
                    "U-G": ("unknown_ai", "genpop"),
                    "F-U": ("famous_ai", "unknown_ai"),
                    "FN-G": ("famous_nonai", "genpop"),
                    "interaction": ("famous_ai", "unknown_ai", "famous_nonai", "genpop"),
                    "F-G": ("famous_ai", "genpop"),
                }[name]
                n_id = sum(run.matrices[g].shape[0] for g in involved)
                rows = sum(n_rows[g] for g in involved)
            records.append(
                {
                    "run": run_name,
                    "estimand": name,
                    "description": LABELS[name],
                    "point_pp": est[name],
                    "ci95_lower_pp": lo,
                    "ci95_upper_pp": hi,
                    "bootstrap_two_sided_p": boot["two_sided_p"].get(name),
                    "mann_whitney_p_vs_genpop": p_mw,
                    "n_identities": n_id,
                    "n_rows": rows,
                    "bootstrap_seed": int(seeds[seed]),
                }
            )
        summary["runs"][run_name] = {
            "group_mean_deltas_pp": means,
            "estimands_pp": est,
            "ci95_pp": boot["ci95"] | {"Freal-G": real_ci},
            "group_statistics": stats,
        }
    frame = pd.DataFrame(records)
    write_csv(outputs[0], frame)
    summary["csv"] = "outputs/estimands/estimands.csv"
    summary["provenance"] = provenance(inputs, [REPO_ROOT / "configs/task.yaml", REPO_ROOT / "configs/analysis.yaml"])
    finish(STAGE, args.output_root, summary)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
