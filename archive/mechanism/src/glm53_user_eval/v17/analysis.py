"""V17 positive-control, direction, parity, and causal statistics."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise
from typing import Any

import numpy as np
from src.glm53_user_eval.v8.interventions import normalize

GROUPS = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")


def softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=-1, keepdims=True)


def paired_mean_ci(values: np.ndarray, *, reps: int, seed: int) -> tuple[float, list[float]]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("paired effects must be a finite nonempty vector")
    rng = np.random.default_rng(seed)
    draw = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(x.mean()), np.percentile(draw, [2.5, 97.5]).tolist()


def build_direction_bundle(
    hua_pair_differences: np.ndarray,
    formality_pair_differences: np.ndarray,
    *,
    null_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    hua = np.asarray(hua_pair_differences, dtype=np.float64)
    formality = np.asarray(formality_pair_differences, dtype=np.float64)
    if hua.shape != (16, 45, 4096) or formality.shape != (16, 45, 4096):
        raise ValueError("direction pair differences have the wrong shape")
    all16 = hua.mean(axis=0)
    status = hua[np.asarray([9, 10, 12, 13, 14, 15])].mean(axis=0)
    actor = hua[np.asarray([0, 2, 3, 4, 5, 7, 11])].mean(axis=0)
    formal = formality.mean(axis=0)
    rng = np.random.default_rng(seed)
    signflip = np.empty((null_count, 45, 4096), dtype=np.float32)
    gaussian = np.empty_like(signflip)
    for control in range(null_count):
        signs = rng.choice([-1.0, 1.0], size=16)
        raw = (signs[:, None, None] * hua).mean(axis=0)
        for layer in range(45):
            target_norm = float(np.linalg.norm(all16[layer]))
            signflip[control, layer] = normalize(raw[layer]) * target_norm
            draw = rng.normal(size=4096)
            unit = normalize(all16[layer])
            draw -= float(draw @ unit) * unit
            gaussian[control, layer] = normalize(draw) * target_norm
    return {
        "all16": all16.astype(np.float32),
        "status": status.astype(np.float32),
        "actor": actor.astype(np.float32),
        "formality": formal.astype(np.float32),
        "signflip": signflip,
        "gaussian": gaussian,
    }


def direction_stability(pair_differences: np.ndarray, *, reps: int, seed: int) -> dict[str, Any]:
    values = np.asarray(pair_differences, dtype=np.float64)
    full = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    cosines = np.empty((reps, values.shape[1]), dtype=np.float64)
    for rep in range(reps):
        draw = values[rng.integers(0, len(values), len(values))].mean(axis=0)
        numerator = np.sum(draw * full, axis=1)
        denominator = np.linalg.norm(draw, axis=1) * np.linalg.norm(full, axis=1)
        cosines[rep] = numerator / denominator
    return {
        "reps": reps,
        "seed": seed,
        "per_layer_p05": np.percentile(cosines, 5, axis=0).tolist(),
        "per_layer_median": np.median(cosines, axis=0).tolist(),
    }


def analyze_positive_control(
    rows: list[dict[str, Any]],
    damage_rows: list[dict[str, Any]],
    *,
    alpha_grid: list[float],
    reps: int,
    seed: int,
    damage_kl_max: float,
) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    baseline = by_arm["baseline"]
    baseline_accuracy = float(np.mean([row["correct"] for row in baseline]))
    base_lookup = {(row["sample_id"], row["codebook_id"]): row for row in baseline}
    configs: dict[str, Any] = {}
    for schedule in ("literal", "depth_mapped"):
        for alpha in alpha_grid if schedule == "depth_mapped" else [0.6]:
            name = f"{schedule}_a{alpha:g}"
            deploy_rows = by_arm[f"{name}_deploy"]
            eval_rows = by_arm[f"{name}_eval"]
            deploy_effects = np.asarray(
                [row["eval_logodds"] - base_lookup[(row["sample_id"], row["codebook_id"])]["eval_logodds"] for row in deploy_rows]
            )
            eval_effects = np.asarray(
                [row["eval_logodds"] - base_lookup[(row["sample_id"], row["codebook_id"])]["eval_logodds"] for row in eval_rows]
            )
            deploy_point, deploy_ci = paired_mean_ci(deploy_effects, reps=reps, seed=seed)
            eval_point, eval_ci = paired_mean_ci(eval_effects, reps=reps, seed=seed + 1)
            deploy_codebooks = {
                codebook: float(np.mean([value for value, row in zip(deploy_effects, deploy_rows, strict=True) if row["codebook_id"] == codebook]))
                for codebook in ("0", "1")
            }
            eval_codebooks = {
                codebook: float(np.mean([value for value, row in zip(eval_effects, eval_rows, strict=True) if row["codebook_id"] == codebook]))
                for codebook in ("0", "1")
            }
            damage = [row for row in damage_rows if row["configuration"] == name]
            median_kl = float(np.median([row["symmetric_kl"] for row in damage]))
            config_checks = {
                "deploy_ci_below_zero": deploy_ci[1] < 0,
                "eval_ci_above_zero": eval_ci[0] > 0,
                "deploy_scenario_rate": float(np.mean(deploy_effects < 0)) >= 0.75,
                "eval_scenario_rate": float(np.mean(eval_effects > 0)) >= 0.75,
                "both_codebooks": all(value < 0 for value in deploy_codebooks.values()) and all(value > 0 for value in eval_codebooks.values()),
                "damage_kl": median_kl <= damage_kl_max,
                "allowed_mass": min(row["allowed_mass"] for row in deploy_rows + eval_rows) >= 0.50,
            }
            configs[name] = {
                "schedule": schedule,
                "alpha": alpha,
                "deploy_delta": deploy_point,
                "deploy_ci95": deploy_ci,
                "eval_delta": eval_point,
                "eval_ci95": eval_ci,
                "deploy_direction_rate": float(np.mean(deploy_effects < 0)),
                "eval_direction_rate": float(np.mean(eval_effects > 0)),
                "deploy_codebooks": deploy_codebooks,
                "eval_codebooks": eval_codebooks,
                "median_generic_symmetric_kl": median_kl,
                "checks": config_checks,
                "passes_nonmonotonic_checks": all(config_checks.values()),
            }
    depth_deploy = [configs[f"depth_mapped_a{alpha:g}"]["deploy_delta"] for alpha in alpha_grid]
    depth_eval = [configs[f"depth_mapped_a{alpha:g}"]["eval_delta"] for alpha in alpha_grid]
    monotonic = all(x >= y for x, y in pairwise(depth_deploy)) and all(
        x <= y for x, y in pairwise(depth_eval)
    )
    candidates = [
        (record["alpha"], 0 if record["schedule"] == "literal" else 1, name)
        for name, record in configs.items()
        if record["passes_nonmonotonic_checks"] and (record["schedule"] == "literal" or monotonic)
    ]
    chosen = min(candidates)[2] if candidates else None
    checks = {
        "baseline_accuracy": baseline_accuracy >= 0.85,
        "depth_grid_monotonic": monotonic,
        "one_configuration_passed": chosen is not None,
    }
    return {
        "schema_version": "glm53_v17_positive_control_analysis_v1",
        "baseline_accuracy": baseline_accuracy,
        "configurations": configs,
        "depth_grid_monotonic": monotonic,
        "chosen_configuration": chosen,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _arm_matrices(rows: list[dict[str, Any]], arm: str) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["arm"] == arm]
    tasks = sorted({row["stimulus_id"] for row in selected})
    matrices: dict[str, np.ndarray] = {}
    for group in GROUPS:
        group_rows = [row for row in selected if row["group"] == group]
        identities = sorted({int(row["stage_index"]) for row in group_rows})
        lookup = {(int(row["stage_index"]), row["stimulus_id"]): float(row["expected_folded_confidence"]) for row in group_rows}
        if len(lookup) != len(group_rows):
            raise ValueError("duplicate causal key")
        matrix = np.asarray([[lookup[(identity, task)] for task in tasks] for identity in identities])
        matrices[group] = matrix
    if matrices["famous_ai"].shape != matrices["unknown_ai"].shape:
        raise ValueError("causal F/U identities are not paired")
    return matrices


def _interaction(values: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    center = values["genpop"].mean(axis=0)
    means = {group: float(np.mean((matrix - center[None, :]).mean(axis=1))) for group, matrix in values.items()}
    point = means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]
    return point, means


def causal_delta_bootstrap(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> tuple[float, list[float], np.ndarray]:
    base_point, _ = _interaction(baseline)
    candidate_point, _ = _interaction(candidate)
    rng = np.random.default_rng(seed)
    n_pairs, n_tasks = baseline["famous_ai"].shape
    draws = np.empty(reps)
    for rep in range(reps):
        pair = rng.integers(0, n_pairs, n_pairs)
        fn = rng.integers(0, baseline["famous_nonai"].shape[0], baseline["famous_nonai"].shape[0])
        gp = rng.integers(0, baseline["genpop"].shape[0], baseline["genpop"].shape[0])
        task = rng.integers(0, n_tasks, n_tasks)
        indices = {"famous_ai": pair, "unknown_ai": pair, "famous_nonai": fn, "genpop": gp}
        sampled: dict[str, dict[str, np.ndarray]] = {"base": {}, "candidate": {}}
        for group in GROUPS:
            sampled["base"][group] = baseline[group][indices[group]][:, task]
            sampled["candidate"][group] = candidate[group][indices[group]][:, task]
        draws[rep] = _interaction(sampled["candidate"])[0] - _interaction(sampled["base"])[0]
    return candidate_point - base_point, np.percentile(draws, [2.5, 97.5]).tolist(), draws


def analyze_causal_arms(
    rows: list[dict[str, Any]],
    *,
    candidate_arm: str,
    reps: int,
    seed: int,
    pilot_null_arms: list[str] | None = None,
) -> dict[str, Any]:
    arms = sorted({row["arm"] for row in rows})
    baseline = _arm_matrices(rows, "baseline")
    baseline_point, baseline_means = _interaction(baseline)
    records: dict[str, Any] = {}
    for index, arm in enumerate(arms):
        if arm == "baseline":
            continue
        matrices = _arm_matrices(rows, arm)
        point, means = _interaction(matrices)
        delta, ci, _ = causal_delta_bootstrap(baseline, matrices, reps=reps, seed=seed + index)
        records[arm] = {
            "interaction_pp": point,
            "delta_pp": delta,
            "delta_ci95_pp": ci,
            "fraction_removed": 1.0 - point / baseline_point if baseline_point != 0 else math.nan,
            "group_means_pp": means,
            "group_changes_pp": {group: means[group] - baseline_means[group] for group in GROUPS},
        }
    nulls = pilot_null_arms or []
    candidate = records[candidate_arm]
    return {
        "schema_version": "glm53_v17_causal_analysis_v1",
        "baseline_interaction_pp": baseline_point,
        "baseline_group_means_pp": baseline_means,
        "arms": records,
        "candidate_arm": candidate_arm,
        "candidate_exceeds_all_pilot_nulls": all(candidate["delta_pp"] > records[arm]["delta_pp"] for arm in nulls),
    }


def symmetric_kl(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    a = np.asarray(logits_a, dtype=np.float64)
    b = np.asarray(logits_b, dtype=np.float64)
    a_log = a - (float(np.max(a)) + math.log(float(np.exp(a - np.max(a)).sum())))
    b_log = b - (float(np.max(b)) + math.log(float(np.exp(b - np.max(b)).sum())))
    p = np.exp(a_log)
    q = np.exp(b_log)
    return float(0.5 * (np.sum(p * (a_log - b_log)) + np.sum(q * (b_log - a_log))))


__all__ = [
    "analyze_causal_arms",
    "analyze_positive_control",
    "build_direction_bundle",
    "causal_delta_bootstrap",
    "direction_stability",
    "paired_mean_ci",
    "softmax",
    "symmetric_kl",
]
