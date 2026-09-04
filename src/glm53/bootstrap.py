"""Crossed identity-and-dilemma percentile bootstraps.

Every draw resamples dilemmas once (shared across groups), famous-AI and unknown-twin
identities as pairs, and the other groups' identities independently. The general-population
center is recomputed inside each draw. Random draws are made in a fixed order so that seeds
reproduce the committed intervals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .io import PRIMARY_GROUPS
from .measure import genpop_center, group_means, identity_effects


def percentile_interval(draws: np.ndarray, level: float = 0.95) -> list[float]:
    """Percentile interval endpoints of finite draws."""
    finite = draws[np.isfinite(draws)]
    tail = 100.0 * (1.0 - level) / 2.0
    return [float(v) for v in np.percentile(finite, [tail, 100.0 - tail])]


def two_sided_p(draws: np.ndarray) -> float:
    """Bootstrap two-sided p-value: twice the smaller tail mass at zero."""
    finite = draws[np.isfinite(draws)]
    return min(1.0, 2.0 * min(float(np.mean(finite <= 0)), float(np.mean(finite >= 0))))


def draw_indices(
    rng: np.random.Generator, matrices: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One crossed draw: dilemma indices, then identity indices per group in the committed order."""
    n_tasks = matrices["genpop"].shape[1]
    n_pairs = matrices["famous_ai"].shape[0]
    if matrices["unknown_ai"].shape != matrices["famous_ai"].shape:
        raise ValueError("famous-AI and unknown-twin matrices are not index-paired")
    task_idx = rng.integers(0, n_tasks, size=n_tasks)
    pair_idx = rng.integers(0, n_pairs, size=n_pairs)
    n_fn = matrices["famous_nonai"].shape[0]
    n_g = matrices["genpop"].shape[0]
    fn_idx = rng.integers(0, n_fn, size=n_fn)
    g_idx = rng.integers(0, n_g, size=n_g)
    ids = {"famous_ai": pair_idx, "unknown_ai": pair_idx, "famous_nonai": fn_idx, "genpop": g_idx}
    return task_idx, ids


def resample(
    matrices: Mapping[str, np.ndarray], task_idx: np.ndarray, ids: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {group: matrices[group][ids[group]][:, task_idx] for group in PRIMARY_GROUPS}


def bootstrap_contrasts(
    matrices: Mapping[str, np.ndarray], *, reps: int, seed: int, level: float = 0.95
) -> dict[str, Any]:
    """Point estimates and intervals for U-G, F-U, FN-G, F-G, and the interaction.

    Draw order per replicate matches the committed confirmatory analysis exactly.
    """
    means = group_means(matrices)
    names = ("interaction", "F-U", "FN-G", "U-G", "F-G")
    draws = {name: np.empty(reps, dtype=np.float64) for name in names}
    rng = np.random.default_rng(seed)
    for rep in range(reps):
        task_idx, ids = draw_indices(rng, matrices)
        sampled = group_means(resample(matrices, task_idx, ids))
        fu = sampled["famous_ai"] - sampled["unknown_ai"]
        fng = sampled["famous_nonai"] - sampled["genpop"]
        draws["interaction"][rep] = fu - fng
        draws["F-U"][rep] = fu
        draws["FN-G"][rep] = fng
        draws["U-G"][rep] = sampled["unknown_ai"] - sampled["genpop"]
        draws["F-G"][rep] = sampled["famous_ai"] - sampled["genpop"]
    points = {
        "F-U": means["famous_ai"] - means["unknown_ai"],
        "FN-G": means["famous_nonai"] - means["genpop"],
        "U-G": means["unknown_ai"] - means["genpop"],
        "F-G": means["famous_ai"] - means["genpop"],
    }
    points["interaction"] = points["F-U"] - points["FN-G"]
    return {
        "group_means": means,
        "points": points,
        "ci95": {name: percentile_interval(draws[name], level) for name in names},
        "two_sided_p": {name: two_sided_p(draws[name]) for name in names},
        "reps": reps,
        "seed": seed,
        "draws": draws,
    }


def bootstrap_group_delta(
    group: np.ndarray, genpop: np.ndarray, *, reps: int, seed: int, level: float = 0.95
) -> list[float]:
    """Interval for one group's mean centered effect; identities and dilemmas resampled."""
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    tasks = group.shape[1]
    for rep in range(reps):
        task_idx = rng.integers(0, tasks, size=tasks)
        group_idx = rng.integers(0, group.shape[0], size=group.shape[0])
        genpop_idx = rng.integers(0, genpop.shape[0], size=genpop.shape[0])
        with np.errstate(invalid="ignore"):
            center = np.nanmean(genpop[genpop_idx][:, task_idx], axis=0)
            person = np.nanmean(group[group_idx][:, task_idx] - center[None, :], axis=1)
        values[rep] = np.nanmean(person)
    return percentile_interval(values, level)


def bootstrap_paired_difference(
    difference: np.ndarray, *, reps: int, seed: int, level: float = 0.95
) -> tuple[float, list[float]]:
    """Mean of an identity-by-dilemma difference matrix with identities and dilemmas resampled."""
    with np.errstate(invalid="ignore"):
        point = float(np.mean(np.nanmean(difference, axis=1)))
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        id_idx = rng.integers(0, difference.shape[0], size=difference.shape[0])
        task_idx = rng.integers(0, difference.shape[1], size=difference.shape[1])
        with np.errstate(invalid="ignore"):
            values[rep] = float(np.mean(np.nanmean(difference[id_idx][:, task_idx], axis=1)))
    return point, percentile_interval(values, level)


Statistic = Callable[[Mapping[str, np.ndarray], np.ndarray, Mapping[str, np.ndarray]], float]


def bootstrap_statistic(
    matrices: Mapping[str, np.ndarray],
    statistic: Statistic,
    *,
    reps: int,
    seed: int,
    level: float = 0.95,
) -> tuple[list[float], np.ndarray]:
    """Interval for any statistic of the resampled matrices; the statistic also sees the draw."""
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        task_idx, ids = draw_indices(rng, matrices)
        values[rep] = statistic(matrices, task_idx, ids)
    return percentile_interval(values, level), values


def identity_effect_draw(
    matrices: Mapping[str, np.ndarray], task_idx: np.ndarray, ids: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-identity centered effects under one draw, keyed by group, in resampled order."""
    sampled = resample(matrices, task_idx, ids)
    center = genpop_center(sampled)
    return {group: identity_effects(sampled[group], center) for group in PRIMARY_GROUPS}
