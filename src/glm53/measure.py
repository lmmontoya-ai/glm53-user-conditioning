"""Measurement definitions: folded confidence, centering, identity and group summaries.

All quantities are in percentage points of folded confidence unless stated otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .io import GROUPS, PRIMARY_GROUPS


def folded(score: np.ndarray | float) -> np.ndarray | float:
    """Folded confidence in percentage points: max(p, 100 - p) for a probability p."""
    value = 100.0 * np.asarray(score, dtype=np.float64)
    return np.maximum(value, 100.0 - value)


@dataclass(frozen=True)
class RunMatrices:
    """Identity-by-dilemma folded-confidence matrices for one run, NaN where no valid score."""

    matrices: dict[str, np.ndarray]
    stimuli: list[str]
    personas: dict[str, list[str]]

    def subset(self, stimulus_indices: list[int]) -> RunMatrices:
        return RunMatrices(
            {group: matrix[:, stimulus_indices] for group, matrix in self.matrices.items()},
            [self.stimuli[index] for index in stimulus_indices],
            self.personas,
        )


def build_matrices(
    frame: pd.DataFrame, roster: Mapping[str, list[dict[str, Any]]]
) -> RunMatrices:
    """Folded-confidence matrices in roster order; anonymous rows are dropped.

    Each group matrix has one row per roster identity and one column per dilemma sorted by
    stimulus id. A cell is NaN when the row's score is missing.
    """
    stimuli = sorted(frame["stimulus"].astype(str).unique())
    if len(stimuli) != 100:
        raise ValueError(f"expected 100 dilemmas, found {len(stimuli)}")
    column = {value: index for index, value in enumerate(stimuli)}
    valid = frame[frame["persona"] != "anon"]
    matrices: dict[str, np.ndarray] = {}
    personas: dict[str, list[str]] = {}
    for group in GROUPS:
        keys = [str(row["key"]) for row in roster[group]]
        personas[group] = keys
        index = {key: position for position, key in enumerate(keys)}
        matrix = np.full((len(keys), len(stimuli)), np.nan, dtype=np.float64)
        part = valid[valid["group"] == group]
        for persona, stimulus, score in zip(part["persona"], part["stimulus"], part["score"]):
            if persona not in index:
                raise KeyError(f"{persona} not in roster group {group}")
            if not math.isnan(score):
                matrix[index[persona], column[str(stimulus)]] = folded(float(score))
        expected = len(keys) * len(stimuli)
        if len(part) != expected:
            raise ValueError(f"{group}: {len(part)} rows, expected {expected}")
        matrices[group] = matrix
    return RunMatrices(matrices, stimuli, personas)


def genpop_center(matrices: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-dilemma mean folded confidence over valid general-population identities."""
    with np.errstate(invalid="ignore"):
        center = np.nanmean(matrices["genpop"], axis=0)
    if not np.isfinite(center).all():
        raise ValueError("a dilemma has no valid general-population row")
    return center


def identity_effects(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Each identity's mean centered folded confidence across its valid dilemmas."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(matrix - center[None, :], axis=1)


def group_means(
    matrices: Mapping[str, np.ndarray], groups: tuple[str, ...] = PRIMARY_GROUPS
) -> dict[str, float]:
    """Equal-weight mean of identity effects per group, centered on the general population."""
    center = genpop_center(matrices)
    means: dict[str, float] = {}
    for group in groups:
        effects = identity_effects(matrices[group], center)
        if not np.isfinite(effects).all():
            raise ValueError(f"{group}: an identity has no valid dilemma")
        means[group] = float(np.mean(effects))
    return means


def estimands(means: Mapping[str, float]) -> dict[str, float]:
    """The preregistered contrasts and the two fame-versus-genpop differences."""
    out = {
        "U-G": means["unknown_ai"] - means["genpop"],
        "F-U": means["famous_ai"] - means["unknown_ai"],
        "FN-G": means["famous_nonai"] - means["genpop"],
    }
    out["interaction"] = out["F-U"] - out["FN-G"]
    out["F-G"] = means["famous_ai"] - means["genpop"]
    if "famous_ai_real" in means:
        out["Freal-G"] = means["famous_ai_real"] - means["genpop"]
    return out


def interaction(matrices: Mapping[str, np.ndarray]) -> float:
    """(F-U)-(FN-G) in percentage points."""
    means = group_means(matrices)
    return means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]


def matched_address_pairs(roster: Mapping[str, list[dict[str, Any]]]) -> list[tuple[int, int]]:
    """(published-address index, constructed index) for identities present in both profiles."""
    constructed = {
        str(row["key"]).removeprefix("fai2_"): index
        for index, row in enumerate(roster["famous_ai"])
    }
    public = {
        str(row["key"]).removeprefix("fai2r_"): index
        for index, row in enumerate(roster["famous_ai_real"])
    }
    slugs = sorted(set(constructed) & set(public))
    return [(public[slug], constructed[slug]) for slug in slugs]


def address_difference(
    matrices: Mapping[str, np.ndarray], pairs: list[tuple[int, int]]
) -> np.ndarray:
    """Published-address minus constructed-address folded confidence, identity by dilemma."""
    left = matrices["famous_ai_real"][[p for p, _ in pairs]]
    right = matrices["famous_ai"][[c for _, c in pairs]]
    return left - right


def mannwhitney_z(a: list[float], b: list[float]) -> float:
    """Tie-aware Mann-Whitney normal approximation, as in the pinned Transluce plotting code."""
    n1, n2 = len(a), len(b)
    ranked = sorted([(value, 0) for value in a] + [(value, 1) for value in b])
    ranks = [0.0] * len(ranked)
    index = 0
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[index][0]:
            end += 1
        midrank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[position] = midrank
        index = end + 1
    rank_sum = sum(rank for rank, (_value, source) in zip(ranks, ranked) if source == 0)
    u_value = rank_sum - n1 * (n1 + 1) / 2
    mean = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return (u_value - mean) / sd if sd else 0.0


def group_statistics(run: RunMatrices, bonferroni: int = 28) -> dict[str, dict[str, Any]]:
    """Per-group median, quartiles, range, and Mann-Whitney test against the general population."""
    center = genpop_center(run.matrices)
    effects = {group: identity_effects(run.matrices[group], center) for group in GROUPS}
    genpop = sorted(float(v) for v in effects["genpop"])
    out: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        values = sorted(float(v) for v in effects[group])
        z = None if group == "genpop" else mannwhitney_z(values, genpop)
        p = None if z is None else math.erfc(abs(z) / math.sqrt(2))
        out[group] = {
            "n": len(values),
            "median_pp": float(np.median(values)),
            "iqr_source_indexed_pp": [values[len(values) // 4], values[3 * len(values) // 4]],
            "min_pp": values[0],
            "max_pp": values[-1],
            "mann_whitney_z_vs_genpop": z,
            "mann_whitney_p_vs_genpop": p,
            "bonferroni_28_p_vs_genpop": None if p is None else min(1.0, p * bonferroni),
        }
    return out


def response_level_sd(run: RunMatrices) -> float:
    """SD of folded confidence over valid non-anonymous rows in the four primary groups."""
    values = np.concatenate([run.matrices[g].ravel() for g in PRIMARY_GROUPS])
    return float(np.nanstd(values, ddof=1))


def leave_one_out(run: RunMatrices) -> dict[str, Any]:
    """Interaction after dropping each famous/unknown pair, then each control identity."""
    full = interaction(run.matrices)
    rows: list[dict[str, Any]] = []
    n_pairs = run.matrices["famous_ai"].shape[0]
    for index in range(n_pairs):
        sampled = dict(run.matrices)
        sampled["famous_ai"] = np.delete(run.matrices["famous_ai"], index, axis=0)
        sampled["unknown_ai"] = np.delete(run.matrices["unknown_ai"], index, axis=0)
        rows.append(
            {"kind": "famous_unknown_pair", "index": index, "interaction_pp": interaction(sampled)}
        )
    for group in ("famous_nonai", "genpop"):
        for index in range(run.matrices[group].shape[0]):
            sampled = dict(run.matrices)
            sampled[group] = np.delete(run.matrices[group], index, axis=0)
            rows.append({"kind": group, "index": index, "interaction_pp": interaction(sampled)})
    shifts = [abs(row["interaction_pp"] - full) for row in rows]
    return {
        "full_pp": full,
        "maximum_absolute_shift_pp": max(shifts),
        "sign_flip_count": sum(row["interaction_pp"] >= 0 for row in rows),
        "minimum_pp": min(row["interaction_pp"] for row in rows),
        "maximum_pp": max(row["interaction_pp"] for row in rows),
        "rows": rows,
    }


def valid_counts(run: RunMatrices) -> dict[str, dict[str, float | int]]:
    """Row and validity counts per roster group."""
    out = {}
    for group in GROUPS:
        matrix = run.matrices[group]
        out[group] = {
            "nonanonymous_rows": int(matrix.size),
            "valid_rows": int(np.isfinite(matrix).sum()),
            "valid_rate": float(np.isfinite(matrix).mean()),
            "identities": int(matrix.shape[0]),
        }
    return out
