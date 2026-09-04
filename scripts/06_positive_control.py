"""Stage 6: session-context positive control on general-population identities.

Subcommands:
  plan     schedule and cost projection; with --dry-run prints them and stops (default)
  execute  run the shards (requires --execute, the config cost cap, and OPENROUTER_API_KEY)
  analyze  contrasts against the neutral block from completed shard logs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, print_plan, provenance, stage_parser  # noqa: E402

from glm53.bootstrap import percentile_interval  # noqa: E402
from glm53.io import load_roster, load_yaml, repo_path, write_csv, write_json  # noqa: E402
from glm53.measure import folded  # noqa: E402
from glm53.run_task import (  # noqa: E402
    extract_rows,
    manifest,
    positive_control_shards,
    run,
    source_root,
    split_key_from_persona,
    verify_source,
)

STAGE = "positive_control"


def projection(task, control, shards) -> dict[str, object]:
    rows = sum(s.expected_rows for s in shards)
    per_row = float(task["cost"]["confirmatory_total_usd"]) / float(task["cost"]["confirmatory_rows"])
    allowance = float(control["cost_projection"]["retry_allowance"])
    return {
        "rows": rows,
        "shards": len(shards),
        "conditions": list(control["conditions"]),
        "identities": int(sum(len(load_roster(task)[g]) for g in control["population"]["groups"])),
        "dilemmas": int(control["population"]["dilemmas"]),
        "per_row_usd": per_row,
        "per_row_source": "confirmatory run api_cost_usd / audited rows (subject and judge calls)",
        "projected_usd": rows * per_row,
        "projected_usd_with_retry_allowance": rows * per_row * allowance,
        "cost_cap_usd": float(control["cost_cap_usd"]),
        "within_cap": rows * per_row * allowance <= float(control["cost_cap_usd"]),
    }


def cmd_plan(args, task, control) -> int:
    roster = load_roster(task)
    root = source_root(task, args.source_root)
    shards = positive_control_shards(task, control, roster)
    proj = projection(task, control, shards)
    source = verify_source(task, root)
    plan = manifest(task, control, shards, root) | {"projection": proj, "source_check": source}
    if args.dry_run:
        print_plan(
            STAGE,
            [repo_path(control["context_blocks_file"]), repo_path(task["roster"]["file"]), root],
            [args.output_root / "schedule_manifest.json", args.output_root / "eval_logs/"],
            {
                "api_calls": proj["rows"] * 2,
                "api_calls_note": "one subject conversation (two turns) and two judge extractions per row",
                "projection": proj,
                "source_check": {k: v for k, v in source.items() if k != "checks"} | {"checks_failed": [k for k, v in source.get("checks", {}).items() if not v]},
                "execute_flag_in_config": bool(control.get("execute", False)),
                "would_execute": False,
                "example_persona_keys": list(shards[0].persona_keys[:3]),
                "example_command_task": "src/glm53/transluce_context_task.py@pmisaligned_context",
            },
        )
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "schedule_manifest.json", plan)
    print(json.dumps(proj, indent=2))
    return 0


def cmd_execute(args, task, control) -> int:
    if not args.execute or not bool(control.get("execute", False)):
        print("execution requires --execute on the command line and execute: true in configs/positive_control.yaml")
        return 2
    roster = load_roster(task)
    root = source_root(task, args.source_root)
    source = verify_source(task, root)
    if not source.get("passed"):
        print("pinned Transluce checkout failed verification; aborting before any call")
        print(json.dumps(source, indent=2))
        return 2
    shards = positive_control_shards(task, control, roster)
    proj = projection(task, control, shards)
    if not proj["within_cap"]:
        print(f"projected cost ${proj['projected_usd_with_retry_allowance']:.2f} exceeds cap ${proj['cost_cap_usd']:.2f}; aborting before any call")
        return 2
    if not os.environ.get(task["subject"]["credential_env"]):
        print(f"{task['subject']['credential_env']} is not set; aborting before any call")
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "schedule_manifest.json", manifest(task, control, shards, root) | {"projection": proj})
    selected = shards[: args.max_shards] if args.max_shards else shards
    state = run(task, control, selected, root=root, output_root=args.output_root, parallel=int(control["sharding"]["parallel_shards"]), connections=int(control["sharding"]["connections_per_shard"]))
    print(json.dumps(state["summary"], indent=2))
    return 0 if state["summary"]["failed_or_incomplete"] == 0 else 1


def cmd_analyze(args, task, control) -> int:
    if args.dry_run:
        print_plan(f"{STAGE}/analyze", [args.output_root / "schedule_manifest.json", args.output_root / "eval_logs"], [args.output_root / "analysis.json", args.output_root / "per_dilemma_model_assessment.csv"], {"api_calls": 0})
        return 0
    if not (args.output_root / "eval_logs").exists():
        print("no eval logs found; the positive control has not been executed")
        return 2
    rows = extract_rows(args.output_root)
    frame = pd.DataFrame(rows)
    frame = frame[frame["persona"] != "anon"].copy()
    parts = frame["persona"].map(split_key_from_persona)
    frame["identity"] = [p[0] for p in parts]
    frame["condition"] = [p[1] for p in parts]
    frame["family"] = [p[2] for p in parts]
    frame["folded"] = folded(pd.to_numeric(frame["score"], errors="coerce").to_numpy())
    identities = sorted(frame["identity"].unique())
    stimuli = sorted(frame["stimulus"].unique())
    cube = {}
    for condition in control["conditions"]:
        m = np.full((len(identities), len(stimuli)), np.nan)
        part = frame[frame["condition"] == condition]
        for identity, stimulus, value in zip(part["identity"], part["stimulus"], part["folded"]):
            m[identities.index(identity), stimuli.index(stimulus)] = value
        cube[condition] = m
    reps = int(control["analysis"]["bootstrap"]["reps"])
    rng = np.random.default_rng(int(control["analysis"]["bootstrap"]["seed"]))
    result = {"rows": int(len(frame)), "valid_rows": int(np.isfinite(frame["folded"]).sum()), "contrasts": {}}
    for name, (a, b) in control["analysis"]["contrasts"].items():
        diff = cube[a] - cube[b]
        with np.errstate(invalid="ignore"):
            point = float(np.nanmean(np.nanmean(diff, axis=1)))
        draws = np.empty(reps)
        for rep in range(reps):
            ii = rng.integers(0, diff.shape[0], size=diff.shape[0])
            jj = rng.integers(0, diff.shape[1], size=diff.shape[1])
            with np.errstate(invalid="ignore"):
                draws[rep] = float(np.nanmean(np.nanmean(diff[ii][:, jj], axis=1)))
        result["contrasts"][name] = {"point_pp": point, "ci95_pp": percentile_interval(draws), "unit": "folded confidence percentage points"}
    with np.errstate(invalid="ignore"):
        profile = np.nanmean(cube["model_assessment"] - cube["neutral"], axis=0)
    profile_frame = pd.DataFrame({"stimulus": stimuli, "model_assessment_minus_neutral_pp": profile})
    write_csv(args.output_root / "per_dilemma_model_assessment.csv", profile_frame)
    result["condition_means_pp"] = {c: float(np.nanmean(cube[c])) for c in control["conditions"]}
    result["valid_rows_by_condition"] = {c: int(np.isfinite(cube[c]).sum()) for c in control["conditions"]}
    # Descriptive cross-run comparison: the neutral block against the confirmatory run's general-population rows
    # (same identities and dilemmas, different run and no block). Not a within-run contrast.
    task_conf = load_yaml("task.yaml")
    try:
        from glm53.io import load_raw_scores, load_roster
        from glm53.measure import build_matrices

        conf = build_matrices(load_raw_scores("confirmatory", task_conf), load_roster(task_conf))
        genpop = conf.matrices["genpop"]
        cols = [conf.stimuli.index(s) for s in stimuli]
        rows_idx = [conf.personas["genpop"].index(i) for i in identities]
        baseline = genpop[rows_idx][:, cols]
        diff = cube["neutral"] - baseline
        with np.errstate(invalid="ignore"):
            point = float(np.nanmean(np.nanmean(diff, axis=1)))
        draws = np.empty(reps)
        for rep in range(reps):
            ii = rng.integers(0, diff.shape[0], size=diff.shape[0])
            jj = rng.integers(0, diff.shape[1], size=diff.shape[1])
            with np.errstate(invalid="ignore"):
                draws[rep] = float(np.nanmean(np.nanmean(diff[ii][:, jj], axis=1)))
        result["neutral_block_minus_confirmatory_genpop_pp"] = {"point_pp": point, "ci95_pp": percentile_interval(draws), "role": "descriptive_cross_run"}
    except Exception as exc:  # noqa: BLE001
        result["neutral_block_minus_confirmatory_genpop_pp"] = f"not computed: {type(exc).__name__}: {exc}"
    scrutiny_path = repo_path(control["analysis"]["correlate_with"])
    if scrutiny_path.exists():
        from scipy.stats import spearmanr

        scrutiny = pd.read_csv(scrutiny_path)
        joined = profile_frame.merge(scrutiny, on="stimulus")
        x = joined["model_assessment_minus_neutral_pp"].to_numpy()
        y = joined["twin_adjusted_pp"].to_numpy()
        rho = float(spearmanr(x, y).statistic)
        rng_c = np.random.default_rng(int(control["analysis"]["bootstrap"]["seed"]) + 7)
        rho_draws = np.empty(reps)
        for rep in range(reps):
            jj = rng_c.integers(0, len(x), size=len(x))
            rho_draws[rep] = spearmanr(x[jj], y[jj]).statistic
        result["correlation_with_scrutiny_profile"] = {
            "spearman_rho": rho,
            "ci95_dilemma_bootstrap": percentile_interval(rho_draws[np.isfinite(rho_draws)]),
            "n_dilemmas": int(len(joined)),
            "scrutiny_profile": str(scrutiny_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "note": "x = model_assessment minus neutral per dilemma (genpop identities); y = famous scrutiny identity minus twin per dilemma (confirmatory run)",
        }
    else:
        result["correlation_with_scrutiny_profile"] = "pending: outputs/identities/per_dilemma_scrutiny_profile.csv does not exist yet (needs merged role coding)"
    result["provenance"] = provenance([args.output_root / "schedule_manifest.json"], [REPO_ROOT / "configs/positive_control.yaml"])
    write_json(args.output_root / "analysis.json", result)
    print(json.dumps(result["contrasts"], indent=2))
    return 0


def main() -> int:
    parser = stage_parser(STAGE, __doc__)
    parser.add_argument("command", nargs="?", default="plan", choices=("plan", "execute", "analyze"))
    parser.add_argument("--execute", action="store_true", help="required together with execute: true in the config to make API calls")
    parser.add_argument("--source-root", type=Path, help="pinned Transluce checkout (default from configs/task.yaml)")
    parser.add_argument("--max-shards", type=int, help="run only the first N pending shards (smoke test); completed shards are skipped on rerun")
    parser.add_argument("--env-file", type=Path, help="untracked file providing OPENROUTER_API_KEY; only that variable is read")
    args = parser.parse_args()
    task = load_yaml("task.yaml")
    control = load_yaml("positive_control.yaml")
    if args.env_file and args.env_file.exists():
        for line in args.env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == task["subject"]["credential_env"] and not os.environ.get(key.strip()):
                os.environ[key.strip()] = value.strip().strip("'\"")
    return {"plan": cmd_plan, "execute": cmd_execute, "analyze": cmd_analyze}[args.command](args, task, control)


if __name__ == "__main__":
    raise SystemExit(main())
