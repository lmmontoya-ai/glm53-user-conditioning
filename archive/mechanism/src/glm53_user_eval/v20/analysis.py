"""Paired statistics for the V20 direct user intervention."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

GROUPS = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
FULL_ARMS = (
    "baseline",
    "all16_deploy",
    "all16_eval",
    "status_deploy",
    "actor_deploy",
    "formality",
)


def arm_matrices(
    rows: list[dict[str, Any]], arm: str, *, identity_count: int
) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["arm"] == arm]
    if not selected:
        raise ValueError(f"V20 arm is empty: {arm}")
    tasks = sorted({str(row["stimulus_id"]) for row in selected})
    matrices: dict[str, np.ndarray] = {}
    for group in GROUPS:
        group_rows = [row for row in selected if row["group"] == group]
        identities = sorted({int(row["stage_index"]) for row in group_rows})
        if identities != list(range(identity_count)):
            raise ValueError(f"{group} identity indices differ from the expected range")
        lookup = {
            (int(row["stage_index"]), str(row["stimulus_id"])): float(
                row["expected_folded_confidence"]
            )
            for row in group_rows
        }
        if len(lookup) != len(group_rows):
            raise ValueError("duplicate V20 arm key")
        matrices[group] = np.asarray(
            [[lookup.get((identity, task), np.nan) for task in tasks] for identity in identities],
            dtype=np.float64,
        )
        if np.any(np.sum(np.isfinite(matrices[group]), axis=1) == 0):
            raise ValueError("a V20 identity has no valid dilemmas")
    if matrices["famous_ai"].shape != matrices["unknown_ai"].shape:
        raise ValueError("V20 F/U identities are not paired")
    return matrices


def interaction(values: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    center = np.nanmean(values["genpop"], axis=0)
    if not np.isfinite(center).all():
        raise ValueError("a V20 dilemma lacks a GenPop center")
    means: dict[str, float] = {}
    for group, matrix in values.items():
        counts = np.sum(np.isfinite(matrix), axis=1)
        if np.any(counts == 0):
            raise ValueError("a V20 identity has no centered value")
        person = np.nansum(matrix - center[None, :], axis=1) / counts
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
        raise ValueError("V20 paired arm matrices differ")
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
        indices = {
            "famous_ai": pair,
            "unknown_ai": pair,
            "famous_nonai": fn,
            "genpop": gp,
        }
        sampled_base = {group: baseline[group][indices[group]][:, task] for group in GROUPS}
        sampled_candidate = {group: candidate[group][indices[group]][:, task] for group in GROUPS}
        draws[rep] = interaction(sampled_candidate)[0] - interaction(sampled_base)[0]
    return (
        float(candidate_point - base_point),
        [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        draws,
    )


def _quality(rows: list[dict[str, Any]]) -> dict[str, float]:
    masses = np.asarray([float(row["allowed_mass"]) for row in rows])
    return {
        "allowed_mass_median": float(np.median(masses)),
        "allowed_mass_p05": float(np.percentile(masses, 5)),
        "full_vocab_argmax_allowed_rate": float(
            np.mean([bool(row["full_vocab_argmax_allowed"]) for row in rows])
        ),
        "conditional_entropy_median": float(
            np.median([float(row["conditional_entropy"]) for row in rows])
        ),
    }


def analyze_causal_rows(
    full_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
    *,
    pilot_task_ids: list[str],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    observed_full = {str(row["arm"]) for row in full_rows}
    if observed_full != set(FULL_ARMS):
        raise ValueError("V20 full causal arms differ from the preregistration")
    if len(full_rows) != 1404 * len(FULL_ARMS):
        raise ValueError("V20 full causal row count differs")
    null_arms = sorted({str(row["arm"]) for row in null_rows})
    if null_arms != [f"signflip_{index:02d}" for index in range(20)]:
        raise ValueError("V20 sign-flip arms differ from the preregistration")
    if len(null_rows) != 20 * 80:
        raise ValueError("V20 sign-flip row count differs")

    baseline = arm_matrices(full_rows, "baseline", identity_count=16)
    baseline_point, baseline_means = interaction(baseline)
    records: dict[str, Any] = {}
    for index, arm in enumerate(FULL_ARMS[1:]):
        matrices = arm_matrices(full_rows, arm, identity_count=16)
        point, means = interaction(matrices)
        delta, ci, _ = causal_delta_bootstrap(baseline, matrices, reps=reps, seed=seed + index)
        records[arm] = {
            "interaction_pp": point,
            "delta_pp": delta,
            "delta_ci95_pp": ci,
            "fraction_removed": 1.0 - point / baseline_point if baseline_point else math.nan,
            "group_means_pp": means,
            "group_changes_pp": {group: means[group] - baseline_means[group] for group in GROUPS},
        }

    codebook_deltas: dict[str, float] = {}
    for codebook in ("0", "1"):
        subset = [row for row in full_rows if str(row["codebook_id"]) == codebook]
        base = interaction(arm_matrices(subset, "baseline", identity_count=16))[0]
        candidate = interaction(arm_matrices(subset, "all16_deploy", identity_count=16))[0]
        codebook_deltas[codebook] = float(candidate - base)

    pilot_base_rows = [
        row
        for row in full_rows
        if row["arm"] == "baseline"
        and int(row["stage_index"]) < 4
        and str(row["stimulus_id"]) in set(pilot_task_ids)
    ]
    pilot_candidate_rows = [
        row
        for row in full_rows
        if row["arm"] == "all16_deploy"
        and int(row["stage_index"]) < 4
        and str(row["stimulus_id"]) in set(pilot_task_ids)
    ]
    if len(pilot_base_rows) != 80 or len(pilot_candidate_rows) != 80:
        raise ValueError("V20 pilot subset row count differs")
    pilot_joined = pilot_base_rows + pilot_candidate_rows
    pilot_base = arm_matrices(pilot_joined, "baseline", identity_count=4)
    pilot_candidate = arm_matrices(pilot_joined, "all16_deploy", identity_count=4)
    pilot_candidate_delta = interaction(pilot_candidate)[0] - interaction(pilot_base)[0]
    null_deltas: dict[str, float] = {}
    for arm in null_arms:
        matrices = arm_matrices(null_rows, arm, identity_count=4)
        null_deltas[arm] = float(interaction(matrices)[0] - interaction(pilot_base)[0])
    exceedances = sum(value >= pilot_candidate_delta for value in null_deltas.values())
    empirical_p = (1 + exceedances) / (1 + len(null_deltas))

    quality = {arm: _quality([row for row in full_rows if row["arm"] == arm]) for arm in FULL_ARMS}
    quality["signflip_controls_combined"] = _quality(null_rows)
    return {
        "schema_version": "glm53_v20_causal_analysis_v1",
        "baseline_interaction_pp": baseline_point,
        "baseline_group_means_pp": baseline_means,
        "arms": records,
        "candidate_codebook_delta_pp": codebook_deltas,
        "signflip_control": {
            "pilot_task_ids": pilot_task_ids,
            "pilot_candidate_delta_pp": float(pilot_candidate_delta),
            "null_delta_pp": null_deltas,
            "candidate_rank_descending": 1 + exceedances,
            "candidate_exceeds_every_null": exceedances == 0,
            "add_one_empirical_p": float(empirical_p),
        },
        "quality": quality,
    }


__all__ = [
    "FULL_ARMS",
    "GROUPS",
    "analyze_causal_rows",
    "arm_matrices",
    "causal_delta_bootstrap",
    "interaction",
]
