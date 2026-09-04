"""Locked primary analysis for the fresh v7 AI-specific fame interaction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from pipelines.glm53_user_eval.scripts.analyze_transluce_exact_v6 import (
    atomic_json,
    atomic_text,
    exact_group_stats,
    extract_rows,
    matrices,
    reference_person_deltas,
    sha256_file,
)

PRIMARY_GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_nonai")


def person_weighted_group_mean(matrix: np.ndarray, center: np.ndarray) -> float:
    with np.errstate(invalid="ignore"):
        means = np.nanmean(matrix - center[None, :], axis=1)
    if not np.isfinite(means).all():
        raise ValueError("at least one identity has no valid dilemmas")
    return float(np.mean(means))


def interaction_point(group_matrices: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    center = np.nanmean(group_matrices["genpop"], axis=0)
    if not np.isfinite(center).all():
        raise ValueError("at least one dilemma has no valid GenPop center")
    means = {
        group: person_weighted_group_mean(group_matrices[group], center) for group in PRIMARY_GROUPS
    }
    value = means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]
    return float(value), means


def bootstrap_interaction(
    group_matrices: dict[str, np.ndarray], *, reps: int, seed: int
) -> tuple[dict[str, float], dict[str, list[float]], np.ndarray]:
    point, means = interaction_point(group_matrices)
    points = {
        "interaction_pp": point,
        "famous_ai_minus_unknown_ai_pp": means["famous_ai"] - means["unknown_ai"],
        "famous_nonai_minus_genpop_pp": means["famous_nonai"] - means["genpop"],
        "unknown_ai_minus_genpop_pp": means["unknown_ai"] - means["genpop"],
    }
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(reps, dtype=np.float64) for name in points}
    n_tasks = group_matrices["genpop"].shape[1]
    n_pairs = group_matrices["famous_ai"].shape[0]
    if group_matrices["unknown_ai"].shape != group_matrices["famous_ai"].shape:
        raise ValueError("Famous-AI and Unknown-AI matrices are not index-paired")
    for rep in range(reps):
        task_idx = rng.integers(0, n_tasks, size=n_tasks)
        pair_idx = rng.integers(0, n_pairs, size=n_pairs)
        famous_nonai_idx = rng.integers(
            0, group_matrices["famous_nonai"].shape[0], size=group_matrices["famous_nonai"].shape[0]
        )
        genpop_idx = rng.integers(
            0, group_matrices["genpop"].shape[0], size=group_matrices["genpop"].shape[0]
        )
        sampled = {
            "famous_ai": group_matrices["famous_ai"][pair_idx][:, task_idx],
            "unknown_ai": group_matrices["unknown_ai"][pair_idx][:, task_idx],
            "famous_nonai": group_matrices["famous_nonai"][famous_nonai_idx][:, task_idx],
            "genpop": group_matrices["genpop"][genpop_idx][:, task_idx],
        }
        interaction, sampled_means = interaction_point(sampled)
        draws["interaction_pp"][rep] = interaction
        draws["famous_ai_minus_unknown_ai_pp"][rep] = (
            sampled_means["famous_ai"] - sampled_means["unknown_ai"]
        )
        draws["famous_nonai_minus_genpop_pp"][rep] = (
            sampled_means["famous_nonai"] - sampled_means["genpop"]
        )
        draws["unknown_ai_minus_genpop_pp"][rep] = (
            sampled_means["unknown_ai"] - sampled_means["genpop"]
        )
    intervals = {
        name: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for name, values in draws.items()
    }
    return points, intervals, draws["interaction_pp"]


def canonical_famous_slug(key: str) -> str:
    if key.startswith("fai2r_"):
        return key.removeprefix("fai2r_")
    if key.startswith("fai2_"):
        return key.removeprefix("fai2_")
    raise ValueError(f"not a Famous-AI key: {key}")


def matched_address(
    group_matrices: dict[str, np.ndarray],
    roster: dict[str, list[dict[str, Any]]],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    constructed = {
        canonical_famous_slug(str(row["key"])): index
        for index, row in enumerate(roster["famous_ai"])
    }
    public = {
        canonical_famous_slug(str(row["key"])): index
        for index, row in enumerate(roster["famous_ai_real"])
    }
    slugs = sorted(set(constructed) & set(public))
    if len(slugs) != 59:
        raise ValueError(f"expected 59 matched public-address identities, found {len(slugs)}")
    left = group_matrices["famous_ai_real"][[public[slug] for slug in slugs]]
    right = group_matrices["famous_ai"][[constructed[slug] for slug in slugs]]
    difference = left - right
    point = float(np.mean(np.nanmean(difference, axis=1)))
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        identity_idx = rng.integers(0, len(slugs), size=len(slugs))
        task_idx = rng.integers(0, difference.shape[1], size=difference.shape[1])
        draws[rep] = float(np.mean(np.nanmean(difference[identity_idx][:, task_idx], axis=1)))
    return {
        "matched_identity_count": len(slugs),
        "point_pp": point,
        "ci95_pp": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
    }


def subset_matrices(
    group_matrices: dict[str, np.ndarray], stimulus_indices: list[int]
) -> dict[str, np.ndarray]:
    return {group: matrix[:, stimulus_indices] for group, matrix in group_matrices.items()}


def leave_one_out(group_matrices: dict[str, np.ndarray], full: float) -> dict[str, Any]:
    values: list[tuple[str, str, float]] = []
    for index in range(70):
        sampled = dict(group_matrices)
        sampled["famous_ai"] = np.delete(group_matrices["famous_ai"], index, axis=0)
        sampled["unknown_ai"] = np.delete(group_matrices["unknown_ai"], index, axis=0)
        values.append(("famous_unknown_pair", str(index), interaction_point(sampled)[0]))
    for group in ("famous_nonai", "genpop"):
        for index in range(70):
            sampled = dict(group_matrices)
            sampled[group] = np.delete(group_matrices[group], index, axis=0)
            values.append((group, str(index), interaction_point(sampled)[0]))
    shifts = [abs(value - full) for _kind, _identity, value in values]
    largest = int(np.argmax(shifts))
    return {
        "minimum_interaction_pp": min(value for _kind, _identity, value in values),
        "maximum_interaction_pp": max(value for _kind, _identity, value in values),
        "maximum_absolute_shift_pp": shifts[largest],
        "sign_flip_count": sum(value >= 0 for _kind, _identity, value in values),
        "maximum_influence": {
            "kind": values[largest][0],
            "index": values[largest][1],
            "interaction_pp": values[largest][2],
        },
        "rows": [
            {"kind": kind, "index": identity, "interaction_pp": value}
            for kind, identity, value in values
        ],
    }


def model_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return {"available": False}
    records = []
    for row in rows:
        if row["persona"] == "anon" or row["score"] is None or row["group"] not in PRIMARY_GROUPS:
            continue
        records.append(
            {
                "folded_confidence": 100.0 * max(float(row["score"]), 1.0 - float(row["score"])),
                "group": row["group"],
                "person": f"{row['group']}::{row['persona']}",
                "stimulus": row["stimulus"],
            }
        )
    frame = pd.DataFrame(records)
    fitted = smf.ols(
        "folded_confidence ~ C(group, Treatment(reference='genpop'))", data=frame
    ).fit()
    robust = fitted.get_robustcov_results(
        cov_type="cluster",
        groups=np.column_stack(
            [pd.factorize(frame["person"])[0], pd.factorize(frame["stimulus"])[0]]
        ),
    )
    names = list(fitted.params.index)
    contrast = np.zeros(len(names))
    for term, coefficient in {
        "C(group, Treatment(reference='genpop'))[T.famous_ai]": 1.0,
        "C(group, Treatment(reference='genpop'))[T.unknown_ai]": -1.0,
        "C(group, Treatment(reference='genpop'))[T.famous_nonai]": -1.0,
    }.items():
        contrast[names.index(term)] = coefficient
    test = robust.t_test(contrast)
    return {
        "available": True,
        "interaction_pp": float(test.effect),
        "standard_error_pp": float(test.sd),
        "ci95_pp": [float(value) for value in np.asarray(test.conf_int()).reshape(-1)],
    }


def statistical_state(point: float, interval: list[float]) -> str:
    low, high = interval
    if high < 0 and point <= -0.50:
        return "confirmed_target_sized_interaction"
    if high < 0:
        return "confirmed_small_interaction"
    if low > -0.50:
        return "target_magnitude_ruled_out"
    if point < 0:
        return "directional_ambiguous"
    return "null_or_opposite_interaction"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--technical-audit", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prereg = yaml.safe_load(args.prereg.read_text(encoding="utf-8"))
    technical = json.loads(args.technical_audit.read_text(encoding="utf-8"))
    manual = json.loads(args.manual_audit.read_text(encoding="utf-8"))
    if not technical.get("passed") or not technical.get("all_expected_present"):
        raise ValueError("analysis locked: full technical audit has not passed")
    if not (args.run_root / "FULL_RUN_AUDIT_PASS.json").exists():
        raise ValueError("analysis locked: FULL_RUN_AUDIT_PASS.json missing")
    if not manual.get("passed") or int(manual.get("reviewed_rows", 0)) != 40:
        raise ValueError("analysis locked: manual audit has not passed")
    rows, shards = extract_rows(args.run_root)
    if len(rows) != 34400:
        raise ValueError("analysis requires exactly 34,400 rows")
    roster = json.loads((args.source_root / "core/personas2.json").read_text(encoding="utf-8"))
    group_matrices, stimuli = matrices(rows, roster)
    people, all_group_means = reference_person_deltas(group_matrices, roster)
    uncertainty = prereg["analysis"]["uncertainty"]
    reps = int(uncertainty["bootstrap_reps"])
    seed = int(uncertainty["bootstrap_seed"])
    points, intervals, _draws = bootstrap_interaction(group_matrices, reps=reps, seed=seed)
    split_payload = json.loads(
        (Path(prereg["analysis"]["dilemma_split_path"])).read_text(encoding="utf-8")
    )
    stimulus_index = {stimulus: index for index, stimulus in enumerate(stimuli)}
    split_estimates = {
        name: interaction_point(
            subset_matrices(
                group_matrices, [stimulus_index[value] for value in split_payload[name]]
            )
        )[0]
        for name in ("split_a", "split_b")
    }
    block_estimates = []
    for offset in range(0, 100, 5):
        ids = [f"dd_{index:04d}" for index in range(offset, offset + 5)]
        block_estimates.append(
            {
                "offset": offset,
                "stimuli": ids,
                "interaction_pp": interaction_point(
                    subset_matrices(group_matrices, [stimulus_index[value] for value in ids])
                )[0],
            }
        )
    first_half = interaction_point(
        subset_matrices(group_matrices, [stimulus_index[f"dd_{index:04d}"] for index in range(50)])
    )[0]
    second_half = interaction_point(
        subset_matrices(
            group_matrices, [stimulus_index[f"dd_{index:04d}"] for index in range(50, 100)]
        )
    )[0]
    loo = leave_one_out(group_matrices, points["interaction_pp"])
    address = matched_address(group_matrices, roster, reps=reps, seed=seed + 400)
    counts = {}
    for group in ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai"):
        matrix = group_matrices[group]
        valid_by_person = np.isfinite(matrix).sum(axis=1)
        counts[group] = {
            "nonanonymous_rows": int(matrix.size),
            "valid_nonanonymous_rows": int(np.isfinite(matrix).sum()),
            "valid_rate": float(np.isfinite(matrix).mean()),
            "identities_below_90_valid_dilemmas": int((valid_by_person < 90).sum()),
        }
    v6_analysis = json.loads(
        Path(prereg["discovery_result"]["analysis_path"]).read_text(encoding="utf-8")
    )
    v6_means = v6_analysis["group_mean_deltas_pp"]
    v6_interaction = float(
        v6_means["famous_ai"]
        - v6_means["unknown_ai"]
        - v6_means["famous_nonai"]
        + v6_means["genpop"]
    )
    analysis = {
        "schema_version": "glm53_transluce_interaction_v7_analysis_v1",
        "project_id": prereg["project_id"],
        "dataset": "fresh_v7_only",
        "row_count": len(rows),
        "valid_score_count": sum(row["score"] is not None for row in rows),
        "group_counts": counts,
        "group_mean_deltas_pp": all_group_means,
        "primary": {
            "estimand": "ai_specific_fame_interaction",
            "interaction_pp": points["interaction_pp"],
            "ci95_pp": intervals["interaction_pp"],
            "bootstrap_reps": reps,
            "bootstrap_seed": seed,
        },
        "provisional_statistical_state": statistical_state(
            points["interaction_pp"], intervals["interaction_pp"]
        ),
        "components": {
            "famous_ai_minus_unknown_ai": {
                "point_pp": points["famous_ai_minus_unknown_ai_pp"],
                "ci95_pp": intervals["famous_ai_minus_unknown_ai_pp"],
            },
            "famous_nonai_minus_genpop": {
                "point_pp": points["famous_nonai_minus_genpop_pp"],
                "ci95_pp": intervals["famous_nonai_minus_genpop_pp"],
            },
            "unknown_ai_minus_genpop": {
                "point_pp": points["unknown_ai_minus_genpop_pp"],
                "ci95_pp": intervals["unknown_ai_minus_genpop_pp"],
            },
        },
        "matched_public_minus_constructed_address": address,
        "source_parity_group_statistics": exact_group_stats(people),
        "fixed_dilemma_splits": split_estimates,
        "leave_one_out": loo,
        "execution_blocks": {
            "blocks": block_estimates,
            "first_half_pp": first_half,
            "second_half_pp": second_half,
            "linear_run_order_slope_pp_per_block": float(
                np.polyfit(np.arange(20), [row["interaction_pp"] for row in block_estimates], 1)[0]
            ),
        },
        "model_based_check": model_check(rows),
        "cross_run": {
            "v6_interaction_pp": v6_interaction,
            "v7_minus_v6_pp": points["interaction_pp"] - v6_interaction,
            "same_negative_sign": v6_interaction < 0 and points["interaction_pp"] < 0,
            "role": "descriptive_only",
        },
        "input_hashes": {
            "prereg": sha256_file(args.prereg),
            "technical_audit": sha256_file(args.technical_audit),
            "manual_audit": sha256_file(args.manual_audit),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_root / "raw_scores.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    atomic_json(args.output_root / "shard_manifest.json", shards)
    atomic_json(args.output_root / "person_deltas.json", people)
    atomic_json(args.output_root / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "analysis_path": str(args.output_root / "analysis.json"),
                "interaction_pp": points["interaction_pp"],
                "ci95_pp": intervals["interaction_pp"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
