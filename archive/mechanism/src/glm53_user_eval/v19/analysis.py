"""Crossed, paired statistics for the V19 lean causal test."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

GROUPS = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")


def arm_matrices(rows: list[dict[str, Any]], arm: str) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["arm"] == arm]
    tasks = sorted({str(row["stimulus_id"]) for row in selected})
    matrices: dict[str, np.ndarray] = {}
    for group in GROUPS:
        group_rows = [row for row in selected if row["group"] == group]
        identities = sorted({int(row["stage_index"]) for row in group_rows})
        if identities != list(range(16)):
            raise ValueError(f"{group} identity indices differ from 0..15")
        lookup = {
            (int(row["stage_index"]), str(row["stimulus_id"])): float(row["expected_folded_confidence"])
            for row in group_rows
        }
        if len(lookup) != len(group_rows):
            raise ValueError("duplicate V19 causal key")
        matrices[group] = np.asarray(
            [[lookup.get((identity, task), np.nan) for task in tasks] for identity in identities],
            dtype=np.float64,
        )
        if np.any(np.sum(np.isfinite(matrices[group]), axis=1) == 0):
            raise ValueError("a V19 identity has no valid dilemmas")
    if matrices["famous_ai"].shape != matrices["unknown_ai"].shape:
        raise ValueError("V19 F/U identities are not paired")
    return matrices


def interaction(values: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    center = np.nanmean(values["genpop"], axis=0)
    if not np.isfinite(center).all():
        raise ValueError("a V19 dilemma lacks a GenPop center")
    means: dict[str, float] = {}
    for group, matrix in values.items():
        person = np.nanmean(matrix - center[None, :], axis=1)
        if not np.isfinite(person).all():
            raise ValueError("a V19 identity has no centered value")
        means[group] = float(np.mean(person))
    point = means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]
    return float(point), means


def causal_delta_bootstrap(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> tuple[float, list[float], np.ndarray]:
    if any(baseline[group].shape != candidate[group].shape for group in GROUPS):
        raise ValueError("V19 arm matrices differ")
    base_point, _ = interaction(baseline)
    candidate_point, _ = interaction(candidate)
    rng = np.random.default_rng(seed)
    n_pairs, n_tasks = baseline["famous_ai"].shape
    draws = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        pair = rng.integers(0, n_pairs, n_pairs)
        fn = rng.integers(0, n_pairs, n_pairs)
        gp = rng.integers(0, n_pairs, n_pairs)
        task = rng.integers(0, n_tasks, n_tasks)
        indices = {"famous_ai": pair, "unknown_ai": pair, "famous_nonai": fn, "genpop": gp}
        sampled_base: dict[str, np.ndarray] = {}
        sampled_candidate: dict[str, np.ndarray] = {}
        for group in GROUPS:
            sampled_base[group] = baseline[group][indices[group]][:, task]
            sampled_candidate[group] = candidate[group][indices[group]][:, task]
        draws[rep] = interaction(sampled_candidate)[0] - interaction(sampled_base)[0]
    return (
        float(candidate_point - base_point),
        [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        draws,
    )


def analyze_causal_rows(
    rows: list[dict[str, Any]], *, reps: int, seed: int
) -> dict[str, Any]:
    expected_arms = {"baseline", "all16_deploy", "all16_eval", "status_deploy", "formality"}
    arms = {str(row["arm"]) for row in rows}
    if arms != expected_arms:
        raise ValueError("V19 causal arms differ from the preregistration")
    baseline = arm_matrices(rows, "baseline")
    baseline_point, baseline_means = interaction(baseline)
    records: dict[str, Any] = {}
    for index, arm in enumerate(sorted(arms - {"baseline"})):
        matrices = arm_matrices(rows, arm)
        point, means = interaction(matrices)
        delta, ci, _ = causal_delta_bootstrap(
            baseline, matrices, reps=reps, seed=seed + index
        )
        records[arm] = {
            "interaction_pp": point,
            "delta_pp": delta,
            "delta_ci95_pp": ci,
            "fraction_removed": 1.0 - point / baseline_point if baseline_point else math.nan,
            "group_means_pp": means,
            "group_changes_pp": {
                group: means[group] - baseline_means[group] for group in GROUPS
            },
        }
    codebook_deltas: dict[str, float] = {}
    for codebook in ("0", "1"):
        subset = [row for row in rows if str(row["codebook_id"]) == codebook]
        base = interaction(arm_matrices(subset, "baseline"))[0]
        candidate = interaction(arm_matrices(subset, "all16_deploy"))[0]
        codebook_deltas[codebook] = float(candidate - base)
    quality: dict[str, Any] = {}
    for arm in sorted(arms):
        selected = [row for row in rows if row["arm"] == arm]
        quality[arm] = {
            "allowed_mass_median": float(np.median([row["allowed_mass"] for row in selected])),
            "allowed_mass_p05": float(np.percentile([row["allowed_mass"] for row in selected], 5)),
            "full_vocab_argmax_allowed_rate": float(np.mean([row["full_vocab_argmax_allowed"] for row in selected])),
            "conditional_entropy_median": float(np.median([row["conditional_entropy"] for row in selected])),
        }
    return {
        "schema_version": "glm53_v19_causal_analysis_v1",
        "baseline_interaction_pp": baseline_point,
        "baseline_group_means_pp": baseline_means,
        "arms": records,
        "candidate_codebook_delta_pp": codebook_deltas,
        "quality": quality,
    }


__all__ = ["analyze_causal_rows", "arm_matrices", "causal_delta_bootstrap", "interaction"]
