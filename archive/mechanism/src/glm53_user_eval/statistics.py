"""Paired estimands and crossed bootstrap utilities."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np


T = TypeVar("T")


def mean_paired_effect(positive: Sequence[float], negative: Sequence[float]) -> float:
    if len(positive) != len(negative) or not positive:
        raise ValueError("paired effects require equal non-empty inputs")
    return float(np.mean(np.asarray(positive) - np.asarray(negative)))


def two_way_cluster_bootstrap(
    pair_ids: Sequence[str],
    scenario_ids: Sequence[str],
    statistic: Callable[[list[str], list[str]], float],
    *,
    reps: int,
    seed: int,
) -> np.ndarray:
    pairs = sorted(set(pair_ids))
    scenarios = sorted(set(scenario_ids))
    if not pairs or not scenarios or reps <= 0:
        raise ValueError("crossed bootstrap requires groups and positive repetitions")
    rng = random.Random(seed)
    values = np.empty(reps, dtype=float)
    for index in range(reps):
        sampled_pairs = [rng.choice(pairs) for _ in pairs]
        sampled_scenarios = [rng.choice(scenarios) for _ in scenarios]
        values[index] = statistic(sampled_pairs, sampled_scenarios)
    return values


def percentile_interval(values: Sequence[float], level: float) -> tuple[float, float]:
    if not 0.0 < level < 1.0:
        raise ValueError("interval level must lie in (0, 1)")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=float), [tail, 1.0 - tail])
    return float(low), float(high)


def empirical_random_p(target: float, random_values: Sequence[float]) -> float:
    return (1 + sum(value >= target for value in random_values)) / (1 + len(random_values))


def reduction_fraction(baseline: float, intervention: float, *, epsilon: float = 1e-12) -> float:
    if abs(baseline) <= epsilon:
        raise ValueError("reduction fraction is undefined for a zero baseline")
    return 1.0 - intervention / baseline
