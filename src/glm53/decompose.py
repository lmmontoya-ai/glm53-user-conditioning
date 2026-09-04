"""Decomposition of the interaction into the yes/no choice and confidence given the choice.

Choice is the model's first-turn yes/no answer; confidence is the folded stated confidence
from the second turn. Both come from the raw score rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .bootstrap import draw_indices, percentile_interval, resample, two_sided_p
from .io import PRIMARY_GROUPS


def identity_weighted_mean(matrix: np.ndarray) -> float:
    """Mean over identities of each identity's mean over its valid cells."""
    counts = np.isfinite(matrix).sum(axis=1)
    valid = counts > 0
    if not valid.any():
        return float("nan")
    sums = np.nansum(matrix, axis=1)
    return float(np.mean(sums[valid] / counts[valid]))


def outcome_matrices(
    frame: pd.DataFrame, roster: Mapping[str, list[dict[str, Any]]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(choice, folded confidence) matrices for the four primary groups.

    Choice is 1 for yes, 0 for no, NaN otherwise. Confidence is max(c, 100 - c) of the stated
    integer confidence, NaN when missing. The two are valid independently: a row with an answer
    but no confidence counts in the choice matrix only, and the reverse counts in the confidence
    matrix only, matching the committed decomposition.
    """
    choice = {g: np.full((70, 100), np.nan) for g in PRIMARY_GROUPS}
    conf = {g: np.full((70, 100), np.nan) for g in PRIMARY_GROUPS}
    index = {g: {row["key"]: i for i, row in enumerate(roster[g])} for g in PRIMARY_GROUPS}
    part = frame[(frame["persona"] != "anon") & frame["group"].isin(PRIMARY_GROUPS)]
    for group, persona, stimulus, answer, c in zip(
        part["group"], part["persona"], part["stimulus"], part["binary_answer"], part["confidence_p"]
    ):
        i = index[group][persona]
        j = int(str(stimulus).removeprefix("dd_"))
        if answer in {"yes", "no"}:
            choice[group][i, j] = 1.0 if answer == "yes" else 0.0
        if c is not None and not pd.isna(c):
            conf[group][i, j] = max(float(c), 100.0 - float(c))
    return choice, conf


def interaction(matrices: Mapping[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    """(F-U)-(FN-G) of identity-weighted centered means; identities with no valid cell are skipped."""
    center = np.nanmean(matrices["genpop"], axis=0)
    means = {g: identity_weighted_mean(matrices[g] - center[None, :]) for g in PRIMARY_GROUPS}
    return means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"], means


def bootstrap_interaction(
    matrices: Mapping[str, np.ndarray], *, reps: int, seed: int, scale: float = 1.0
) -> dict[str, Any]:
    """Interaction with a crossed percentile interval, scaled (100 for a 0/1 outcome)."""
    point, means = interaction(matrices)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for rep in range(reps):
        task_idx, ids = draw_indices(rng, matrices)
        draws[rep] = interaction(resample(matrices, task_idx, ids))[0] * scale
    return {
        "interaction": point * scale,
        "ci95": percentile_interval(draws),
        "bootstrap_two_sided_p": two_sided_p(draws),
        "group_means": {k: v * scale for k, v in means.items()},
        "seed": seed,
    }


def choice_standardized(
    conf: Mapping[str, np.ndarray], choice: Mapping[str, np.ndarray]
) -> float:
    """Confidence interaction with every group's yes and no strata weighted by the pooled yes-rate."""
    pooled = np.concatenate([v[np.isfinite(v)] for v in choice.values()])
    q = float(np.mean(pooled))
    center = np.nanmean(conf["genpop"], axis=0)
    means: dict[str, float] = {}
    for group in PRIMARY_GROUPS:
        centered = conf[group] - center[None, :]
        by_choice = [
            identity_weighted_mean(np.where(choice[group] == code, centered, np.nan))
            for code in (0.0, 1.0)
        ]
        means[group] = (1 - q) * by_choice[0] + q * by_choice[1]
    return means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]


def bootstrap_standardized(
    conf: Mapping[str, np.ndarray], choice: Mapping[str, np.ndarray], *, reps: int, seed: int
) -> dict[str, Any]:
    point = choice_standardized(conf, choice)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for rep in range(reps):
        task_idx, ids = draw_indices(rng, conf)
        draws[rep] = choice_standardized(resample(conf, task_idx, ids), resample(choice, task_idx, ids))
    return {"interaction_pp": point, "ci95_pp": percentile_interval(draws), "seed": seed}


def matched_same_choice_point(
    conf: Mapping[str, np.ndarray], choice: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Paired confidence differences kept only where both members of a pair gave the same answer."""
    fu = np.where(choice["famous_ai"] == choice["unknown_ai"], conf["famous_ai"] - conf["unknown_ai"], np.nan)
    fng = np.where(choice["famous_nonai"] == choice["genpop"], conf["famous_nonai"] - conf["genpop"], np.nan)
    return {
        "famous_unknown_pp": identity_weighted_mean(fu),
        "famous_nonai_genpop_pp": identity_weighted_mean(fng),
        "interaction_pp": identity_weighted_mean(fu) - identity_weighted_mean(fng),
        "famous_unknown_retained_cells": int(np.isfinite(fu).sum()),
        "famous_nonai_genpop_retained_cells": int(np.isfinite(fng).sum()),
    }


def bootstrap_matched_same_choice(
    conf: Mapping[str, np.ndarray], choice: Mapping[str, np.ndarray], *, reps: int, seed: int
) -> dict[str, Any]:
    """Same-choice matched estimate with a crossed interval; famous-nonai and genpop are paired by roster index here."""
    point = matched_same_choice_point(conf, choice)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for rep in range(reps):
        task_idx, ids = draw_indices(rng, conf)
        ids = dict(ids)
        ids["famous_nonai"] = ids["genpop"]
        draws[rep] = matched_same_choice_point(resample(conf, task_idx, ids), resample(choice, task_idx, ids))["interaction_pp"]
    return point | {"ci95_pp": percentile_interval(draws), "seed": seed, "interpretation": "conditional_descriptive_not_causal"}


def stratified(
    conf: Mapping[str, np.ndarray], choice: Mapping[str, np.ndarray], *, reps: int, seed: int
) -> dict[str, Any]:
    """Confidence interaction within the no stratum and within the yes stratum."""
    out = {}
    for label, code in (("no", 0.0), ("yes", 1.0)):
        masked = {g: np.where(choice[g] == code, conf[g], np.nan) for g in PRIMARY_GROUPS}
        out[label] = bootstrap_interaction(masked, reps=reps, seed=seed + 10 + int(code))
    return out
