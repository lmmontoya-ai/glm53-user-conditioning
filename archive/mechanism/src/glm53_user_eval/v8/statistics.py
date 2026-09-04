"""Four-population interactions, causal effects, and crossed resampling."""

from __future__ import annotations

import numpy as np


def interaction(means: dict[str, float]) -> float:
    return float(means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"])


def causal_delta(baseline: float, candidate: float) -> float:
    return float(candidate - baseline)


def fraction_removed(baseline: float, candidate: float) -> float:
    if baseline == 0:
        raise ValueError("fraction removed is undefined at zero baseline")
    return float(1.0 - candidate / baseline)


def empirical_p(candidate: float, controls: np.ndarray) -> float:
    values = np.asarray(controls, dtype=np.float64)
    return float((1 + np.sum(values >= candidate)) / (1 + len(values)))


def control_rank(candidate: float, controls: np.ndarray) -> int:
    return int(1 + np.sum(np.asarray(controls) >= candidate))


def four_group_bootstrap(
    values: dict[str, np.ndarray], *, reps: int, seed: int
) -> tuple[float, tuple[float, float], np.ndarray]:
    for group in ("famous_ai", "unknown_ai", "famous_nonai", "genpop"):
        if values[group].ndim != 2:
            raise ValueError("group matrices must be [identities, tasks]")
    if values["famous_ai"].shape != values["unknown_ai"].shape:
        raise ValueError("F/U matrices must be paired")
    point = interaction({key: float(np.nanmean(values[key])) for key in values})
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    n_pairs, n_tasks = values["famous_ai"].shape
    for index in range(reps):
        pair = rng.integers(0, n_pairs, n_pairs)
        task = rng.integers(0, n_tasks, n_tasks)
        fn = rng.integers(0, values["famous_nonai"].shape[0], values["famous_nonai"].shape[0])
        gp = rng.integers(0, values["genpop"].shape[0], values["genpop"].shape[0])
        sampled = {
            "famous_ai": values["famous_ai"][pair][:, task],
            "unknown_ai": values["unknown_ai"][pair][:, task],
            "famous_nonai": values["famous_nonai"][fn][:, task],
            "genpop": values["genpop"][gp][:, task],
        }
        draws[index] = interaction(
            {key: float(np.nanmean(value)) for key, value in sampled.items()}
        )
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, (float(low), float(high)), draws
