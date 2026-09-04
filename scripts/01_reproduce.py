"""Stage 1: recompute every committed confirmatory and discovery number from raw scores."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, finish, print_plan, provenance, stage_parser  # noqa: E402

from glm53.bootstrap import (  # noqa: E402
    bootstrap_contrasts,
    bootstrap_group_delta,
    bootstrap_paired_difference,
)
from glm53.io import (  # noqa: E402
    GROUPS,
    load_raw_scores,
    load_roster,
    load_yaml,
    read_json,
    repo_path,
)
from glm53.measure import (  # noqa: E402
    RunMatrices,
    address_difference,
    build_matrices,
    estimands,
    group_means,
    group_statistics,
    interaction,
    leave_one_out,
    matched_address_pairs,
    response_level_sd,
    valid_counts,
)

STAGE = "reproduce"


def split_estimates(run: RunMatrices, split_path: Path) -> dict[str, float]:
    """Interaction within each fixed dilemma split."""
    payload = read_json(split_path)
    index = {stimulus: i for i, stimulus in enumerate(run.stimuli)}
    return {
        name: interaction(run.subset([index[s] for s in payload[name]]).matrices)
        for name in ("split_a", "split_b")
    }


def block_estimates(run: RunMatrices) -> list[dict[str, object]]:
    """Interaction within each five-dilemma execution block, in run order."""
    index = {stimulus: i for i, stimulus in enumerate(run.stimuli)}
    out = []
    for offset in range(0, 100, 5):
        ids = [f"dd_{i:04d}" for i in range(offset, offset + 5)]
        out.append(
            {
                "offset": offset,
                "interaction_pp": interaction(run.subset([index[s] for s in ids]).matrices),
            }
        )
    return out


def compare(observed: float, pinned: float, tol: float) -> dict[str, object]:
    diff = float(observed - pinned)
    return {"observed": observed, "pinned": pinned, "diff": diff, "within_tolerance": abs(diff) <= tol}


def main() -> int:
    parser = stage_parser(STAGE, __doc__)
    parser.add_argument("--reps", type=int, help="override bootstrap replicates (testing only)")
    args = parser.parse_args()
    task = load_yaml("task.yaml")
    analysis = load_yaml("analysis.yaml")
    split_path = repo_path("data/confirmatory/reference/dilemma_split_v7.json")
    inputs = [
        repo_path(task["runs"]["confirmatory"]["raw_scores"]),
        repo_path(task["runs"]["discovery"]["raw_scores"]),
        repo_path(task["roster"]["file"]),
        split_path,
    ]
    configs = [REPO_ROOT / "configs/task.yaml", REPO_ROOT / "configs/analysis.yaml"]
    if args.dry_run:
        print_plan(STAGE, inputs, [args.output_root / "summary.json"], {"api_calls": 0})
        return 0

    reps = args.reps or int(analysis["bootstrap"]["reps"])
    seeds = analysis["bootstrap"]["seeds"]
    tol_point = float(analysis["tolerances"]["point_estimate"])
    tol_ci = float(analysis["tolerances"]["ci_endpoint"])
    roster = load_roster(task)
    pairs = matched_address_pairs(roster)
    summary: dict[str, object] = {"reps": reps, "runs": {}, "checks": {}}

    # Confirmatory run.
    run = build_matrices(load_raw_scores("confirmatory", task), roster)
    means = group_means(run.matrices, GROUPS)
    boot = bootstrap_contrasts(run.matrices, reps=reps, seed=int(seeds["confirmatory_interaction"]))
    verifier = bootstrap_contrasts(run.matrices, reps=reps, seed=int(seeds["confirmatory_verifier"]))
    address_point, address_ci = bootstrap_paired_difference(
        address_difference(run.matrices, pairs), reps=reps, seed=int(seeds["confirmatory_address"])
    )
    loo = leave_one_out(run)
    confirmatory = {
        "group_mean_deltas_pp": means,
        "estimands_pp": estimands(means),
        "ci95_pp": boot["ci95"],
        "bootstrap_two_sided_p": boot["two_sided_p"],
        "verifier_seed_ci95_pp": verifier["ci95"],
        "address_published_minus_constructed": {
            "matched_identities": len(pairs),
            "point_pp": address_point,
            "ci95_pp": address_ci,
        },
        "leave_one_out": {k: v for k, v in loo.items() if k != "rows"},
        "fixed_dilemma_splits_pp": split_estimates(run, split_path),
        "execution_blocks": block_estimates(run),
        "response_level_sd_pp": response_level_sd(run),
        "interaction_in_response_sd": estimands(means)["interaction"] / response_level_sd(run),
        "group_statistics": group_statistics(run),
        "counts": valid_counts(run),
        "seeds": {
            "interaction": int(seeds["confirmatory_interaction"]),
            "verifier": int(seeds["confirmatory_verifier"]),
            "address": int(seeds["confirmatory_address"]),
        },
    }
    summary["runs"]["confirmatory"] = confirmatory
    pin = analysis["pinned_confirmatory"]
    checks = {
        "interaction_pp": compare(confirmatory["estimands_pp"]["interaction"], pin["interaction_pp"], tol_point),
        "F-U_pp": compare(confirmatory["estimands_pp"]["F-U"], pin["famous_ai_minus_unknown_ai_pp"], tol_point),
        "FN-G_pp": compare(confirmatory["estimands_pp"]["FN-G"], pin["famous_nonai_minus_genpop_pp"], tol_point),
        "U-G_pp": compare(confirmatory["estimands_pp"]["U-G"], pin["unknown_ai_minus_genpop_pp"], tol_point),
        "interaction_ci_lower": compare(boot["ci95"]["interaction"][0], pin["interaction_ci95_pp"][0], tol_ci),
        "interaction_ci_upper": compare(boot["ci95"]["interaction"][1], pin["interaction_ci95_pp"][1], tol_ci),
        "address_pp": compare(address_point, pin["address_pp"], tol_point),
        "leave_one_out_max_shift_pp": compare(loo["maximum_absolute_shift_pp"], pin["leave_one_out_max_shift_pp"], tol_point),
        "leave_one_out_sign_flips": compare(loo["sign_flip_count"], pin["leave_one_out_sign_flips"], 0),
        "split_a_pp": compare(confirmatory["fixed_dilemma_splits_pp"]["split_a"], pin["fixed_dilemma_splits"]["split_a"], tol_point),
        "split_b_pp": compare(confirmatory["fixed_dilemma_splits_pp"]["split_b"], pin["fixed_dilemma_splits"]["split_b"], tol_point),
    }
    for group in GROUPS:
        checks[f"group_mean_{group}_pp"] = compare(means[group], pin["group_mean_deltas_pp"][group], tol_point)
    committed = read_json(repo_path(task["runs"]["confirmatory"]["analysis"]))
    for name, key in (("F-U", "famous_ai_minus_unknown_ai"), ("FN-G", "famous_nonai_minus_genpop"), ("U-G", "unknown_ai_minus_genpop")):
        for side, label in ((0, "lower"), (1, "upper")):
            checks[f"{name}_ci_{label}"] = compare(boot["ci95"][name][side], committed["components"][key]["ci95_pp"][side], tol_ci)
    summary["checks"]["confirmatory"] = checks

    # Discovery run.
    run_d = build_matrices(load_raw_scores("discovery", task), roster)
    means_d = group_means(run_d.matrices, GROUPS)
    base = int(seeds["discovery_group_deltas_base"])
    group_ci = {
        group: bootstrap_group_delta(run_d.matrices[group], run_d.matrices["genpop"], reps=reps, seed=base + i)
        for i, group in enumerate(GROUPS)
    }
    paired_point, paired_ci = bootstrap_paired_difference(
        run_d.matrices["famous_ai"] - run_d.matrices["unknown_ai"], reps=reps, seed=int(seeds["discovery_paired_contrast"])
    )
    boot_d = bootstrap_contrasts(run_d.matrices, reps=reps, seed=int(seeds["discovery_interaction"]))
    address_point_d, address_ci_d = bootstrap_paired_difference(
        address_difference(run_d.matrices, pairs), reps=reps, seed=int(seeds["discovery_interaction"]) + 400
    )
    loo_d = leave_one_out(run_d)
    discovery = {
        "group_mean_deltas_pp": means_d,
        "estimands_pp": estimands(means_d),
        "ci95_pp": boot_d["ci95"],
        "bootstrap_two_sided_p": boot_d["two_sided_p"],
        "group_ci95_pp": group_ci,
        "same_index_famous_minus_unknown_pp": paired_point,
        "same_index_famous_minus_unknown_ci95_pp": paired_ci,
        "address_published_minus_constructed": {
            "matched_identities": len(pairs),
            "point_pp": address_point_d,
            "ci95_pp": address_ci_d,
        },
        "leave_one_out": {k: v for k, v in loo_d.items() if k != "rows"},
        "fixed_dilemma_splits_pp": split_estimates(run_d, split_path),
        "response_level_sd_pp": response_level_sd(run_d),
        "group_statistics": group_statistics(run_d),
        "counts": valid_counts(run_d),
        "seeds": {
            "group_deltas_base": base,
            "paired_contrast": int(seeds["discovery_paired_contrast"]),
            "interaction": int(seeds["discovery_interaction"]),
        },
        "note": "The crossed-bootstrap interaction interval for the discovery run is new; the committed discovery analysis reported group intervals and the paired F-U interval only.",
    }
    summary["runs"]["discovery"] = discovery
    pin_d = analysis["pinned_discovery"]
    committed_d = read_json(repo_path(task["runs"]["discovery"]["analysis"]))
    checks_d = {"interaction_pp": compare(discovery["estimands_pp"]["interaction"], pin_d["interaction_pp"], tol_point)}
    for group in GROUPS:
        checks_d[f"group_mean_{group}_pp"] = compare(means_d[group], pin_d["group_mean_deltas_pp"][group], tol_point)
        for side, label in ((0, "lower"), (1, "upper")):
            checks_d[f"group_ci_{group}_{label}"] = compare(group_ci[group][side], committed_d["group_ci95_pp"][group][side], tol_ci)
    checks_d["same_index_F-U_pp"] = compare(paired_point, committed_d["same_index_famous_minus_unknown_pp"], tol_point)
    for side, label in ((0, "lower"), (1, "upper")):
        checks_d[f"same_index_F-U_ci_{label}"] = compare(paired_ci[side], committed_d["same_index_famous_minus_unknown_ci95_pp"][side], tol_ci)
    for group in GROUPS:
        for key in ("median_pp", "mann_whitney_z_vs_genpop"):
            observed = discovery["group_statistics"][group][key]
            pinned = committed_d["source_exact_group_stats"][group][key]
            if observed is not None and pinned is not None:
                checks_d[f"stat_{group}_{key}"] = compare(observed, pinned, tol_point)
    summary["checks"]["discovery"] = checks_d

    failures = [
        f"{run_name}:{name}"
        for run_name, run_checks in summary["checks"].items()
        for name, check in run_checks.items()
        if not check["within_tolerance"]
    ]
    summary["all_checks_pass"] = not failures
    summary["failures"] = failures
    summary["provenance"] = provenance(inputs, configs)
    finish(STAGE, args.output_root, summary)
    print(f"checks: {sum(len(v) for v in summary['checks'].values())}, failures: {len(failures)}")
    for name in failures:
        print("  FAIL", name)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
