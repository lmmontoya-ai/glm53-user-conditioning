"""Per-identity effects for famous-AI figures and the blind role-coding protocol.

An identity's effect is its mean centered folded confidence; the twin-adjusted effect subtracts
the same quantity for the unknown twin at the same roster index. Role labels come only from the
merged human-and-LLM coding sheet.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import CONFIGS
from .bootstrap import draw_indices, identity_effect_draw, percentile_interval
from .io import sha256_file
from .measure import RunMatrices, genpop_center, identity_effects

CATEGORIES = ("scrutiny", "capabilities", "business", "other")
PROTOCOL_PATH = CONFIGS / "roles" / "PROTOCOL.md"


def protocol_sha256() -> str:
    return sha256_file(PROTOCOL_PATH)


def famous_identity_table(roster: Mapping[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """One row per famous-AI profile (constructed and published address) with its twin."""
    rows = []
    slug_index = {str(r["key"]).removeprefix("fai2_"): i for i, r in enumerate(roster["famous_ai"])}
    for i, row in enumerate(roster["famous_ai"]):
        rows.append(
            {
                "persona_key": row["key"],
                "profile": "constructed",
                "slug": str(row["key"]).removeprefix("fai2_"),
                "roster_index": i,
                "name": row["name"],
                "role_text": f"{row['affiliation']} ({row['org']})",
                "twin_key": roster["unknown_ai"][i]["key"],
            }
        )
    for row in roster["famous_ai_real"]:
        slug = str(row["key"]).removeprefix("fai2r_")
        i = slug_index[slug]
        rows.append(
            {
                "persona_key": row["key"],
                "profile": "published",
                "slug": slug,
                "roster_index": i,
                "name": row["name"],
                "role_text": f"{row['affiliation']} ({row['org']})",
                "twin_key": roster["unknown_ai"][i]["key"],
            }
        )
    return pd.DataFrame(rows)


def per_identity_effects(run: RunMatrices) -> pd.DataFrame:
    """Effect and twin-adjusted effect for every famous-AI profile in one run."""
    center = genpop_center(run.matrices)
    famous = identity_effects(run.matrices["famous_ai"], center)
    unknown = identity_effects(run.matrices["unknown_ai"], center)
    real = identity_effects(run.matrices["famous_ai_real"], center)
    slug_index = {k.removeprefix("fai2_"): i for i, k in enumerate(run.personas["famous_ai"])}
    rows = []
    for i, key in enumerate(run.personas["famous_ai"]):
        rows.append(
            {
                "persona_key": key,
                "effect_pp": float(famous[i]),
                "twin_effect_pp": float(unknown[i]),
                "twin_adjusted_pp": float(famous[i] - unknown[i]),
                "valid_dilemmas": int(np.isfinite(run.matrices["famous_ai"][i]).sum()),
            }
        )
    for j, key in enumerate(run.personas["famous_ai_real"]):
        i = slug_index[key.removeprefix("fai2r_")]
        rows.append(
            {
                "persona_key": key,
                "effect_pp": float(real[j]),
                "twin_effect_pp": float(unknown[i]),
                "twin_adjusted_pp": float(real[j] - unknown[i]),
                "valid_dilemmas": int(np.isfinite(run.matrices["famous_ai_real"][j]).sum()),
            }
        )
    return pd.DataFrame(rows)


def per_dilemma_profile(run: RunMatrices, famous_indices: list[int] | None = None) -> pd.DataFrame:
    """Per-dilemma mean of (famous - twin) folded confidence over the chosen famous-AI identities."""
    idx = list(range(run.matrices["famous_ai"].shape[0])) if famous_indices is None else famous_indices
    diff = run.matrices["famous_ai"][idx] - run.matrices["unknown_ai"][idx]
    with np.errstate(invalid="ignore"):
        profile = np.nanmean(diff, axis=0)
    return pd.DataFrame({"stimulus": run.stimuli, "twin_adjusted_pp": profile, "n_identities": np.isfinite(diff).sum(axis=0)})


def _twin_adjusted_draw(run: RunMatrices, task_idx: np.ndarray, ids: Mapping[str, np.ndarray]) -> np.ndarray:
    effects = identity_effect_draw(run.matrices, task_idx, ids)
    return effects["famous_ai"] - effects["unknown_ai"]


def spearman_between_runs(
    left: RunMatrices, right: RunMatrices, *, reps: int, seed: int
) -> dict[str, Any]:
    """Spearman rho of twin-adjusted effects across runs, with a crossed bootstrap interval.

    Both runs are resampled with the same dilemma and identity draw so the pairing is kept.
    """
    a = per_identity_effects(left)
    b = per_identity_effects(right)
    a = a[a.persona_key.str.startswith("fai2_")].twin_adjusted_pp.to_numpy()
    b = b[b.persona_key.str.startswith("fai2_")].twin_adjusted_pp.to_numpy()
    rho = float(spearmanr(a, b).statistic)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for rep in range(reps):
        task_idx, ids = draw_indices(rng, left.matrices)
        x = _twin_adjusted_draw(left, task_idx, ids)
        y = _twin_adjusted_draw(right, task_idx, ids)
        draws[rep] = spearmanr(x, y).statistic
    return {"rho": rho, "ci95": percentile_interval(draws), "n_identities": int(len(a)), "seed": seed}


def spearman_between_profiles(run: RunMatrices, *, reps: int, seed: int) -> dict[str, Any]:
    """Spearman rho of twin-adjusted effects for constructed versus published profiles, one run."""
    slug_index = {k.removeprefix("fai2_"): i for i, k in enumerate(run.personas["famous_ai"])}
    real_idx = [slug_index[k.removeprefix("fai2r_")] for k in run.personas["famous_ai_real"]]
    center = genpop_center(run.matrices)
    unknown = identity_effects(run.matrices["unknown_ai"], center)
    constructed = identity_effects(run.matrices["famous_ai"], center)[real_idx] - unknown[real_idx]
    published = identity_effects(run.matrices["famous_ai_real"], center) - unknown[real_idx]
    rho = float(spearmanr(constructed, published).statistic)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    n_real = len(real_idx)
    for rep in range(reps):
        task_idx, ids = draw_indices(rng, run.matrices)
        pick = rng.integers(0, n_real, size=n_real)
        base = {g: run.matrices[g][ids[g]][:, task_idx] for g in ("genpop",)}
        c = genpop_center(base)
        fa = run.matrices["famous_ai"][[real_idx[p] for p in pick]][:, task_idx]
        fr = run.matrices["famous_ai_real"][pick][:, task_idx]
        un = run.matrices["unknown_ai"][[real_idx[p] for p in pick]][:, task_idx]
        x = identity_effects(fa, c) - identity_effects(un, c)
        y = identity_effects(fr, c) - identity_effects(un, c)
        draws[rep] = spearmanr(x, y).statistic
    return {"rho": rho, "ci95": percentile_interval(draws), "n_identities": n_real, "seed": seed}


def merge_codings(llm: pd.DataFrame, human: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join the two coding sheets; final_category is set only where both agree or a resolution exists."""
    for frame, label in ((llm, "llm"), (human, "human")):
        if not set(frame["category"].dropna()).issubset(CATEGORIES):
            raise ValueError(f"{label} sheet has categories outside the frozen taxonomy")
    merged = llm[["persona_key", "name", "role_text", "category", "ambiguous", "alternative_category"]].rename(
        columns={"category": "llm_category", "ambiguous": "llm_ambiguous", "alternative_category": "llm_alternative"}
    ).merge(
        human[["persona_key", "category", "ambiguous", "alternative_category"]].rename(
            columns={"category": "human_category", "ambiguous": "human_ambiguous", "alternative_category": "human_alternative"}
        ),
        on="persona_key",
        how="outer",
        validate="one_to_one",
    )
    merged["agree"] = merged["llm_category"] == merged["human_category"]
    merged["final_category"] = np.where(merged["agree"], merged["llm_category"], "")
    merged["resolution"] = ""
    if existing is not None:
        prior = existing.set_index("persona_key")
        for i, key in enumerate(merged["persona_key"]):
            if key in prior.index and not merged.at[i, "agree"]:
                merged.at[i, "final_category"] = str(prior.at[key, "final_category"] or "")
                merged.at[i, "resolution"] = str(prior.at[key, "resolution"] or "")
    merged["ambiguous"] = merged["llm_ambiguous"].fillna(False).astype(bool) | merged["human_ambiguous"].fillna(False).astype(bool)
    merged["alternative_category"] = merged["human_alternative"].where(merged["human_alternative"].notna() & (merged["human_alternative"] != ""), merged["llm_alternative"])
    return merged


def unresolved(merged: pd.DataFrame) -> pd.DataFrame:
    """Rows whose coders disagree and no resolution has been recorded."""
    bad = (~merged["agree"]) & (
        (merged["final_category"].fillna("") == "") | (merged["resolution"].fillna("") == "")
    )
    return merged[bad]


def category_contrast(
    run: RunMatrices,
    labels: Mapping[str, str],
    *,
    reps: int,
    seed: int,
    positive: str = "scrutiny",
    negative: str = "business",
) -> dict[str, Any]:
    """Mean twin-adjusted effect for one category minus another, crossed bootstrap interval."""
    keys = run.personas["famous_ai"]
    cats = np.array([labels.get(k, "") for k in keys])
    pos = np.where(cats == positive)[0]
    neg = np.where(cats == negative)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"empty category: {positive}={len(pos)}, {negative}={len(neg)}")
    center = genpop_center(run.matrices)
    adjusted = identity_effects(run.matrices["famous_ai"], center) - identity_effects(run.matrices["unknown_ai"], center)
    point = float(adjusted[pos].mean() - adjusted[neg].mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for rep in range(reps):
        task_idx, ids = draw_indices(rng, run.matrices)
        # Resample identities within category so both categories stay populated.
        p = pos[rng.integers(0, len(pos), size=len(pos))]
        n = neg[rng.integers(0, len(neg), size=len(neg))]
        c = genpop_center({"genpop": run.matrices["genpop"][ids["genpop"]][:, task_idx]})
        def adj(sel: np.ndarray) -> np.ndarray:
            fa = run.matrices["famous_ai"][sel][:, task_idx]
            un = run.matrices["unknown_ai"][sel][:, task_idx]
            return identity_effects(fa, c) - identity_effects(un, c)
        draws[rep] = adj(p).mean() - adj(n).mean()
    return {
        "contrast": f"{positive}_minus_{negative}",
        "point_pp": point,
        "ci95_pp": percentile_interval(draws),
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "category_means_pp": {c: float(adjusted[cats == c].mean()) for c in CATEGORIES if (cats == c).any()},
        "category_counts": {c: int((cats == c).sum()) for c in CATEGORIES},
        "seed": seed,
    }


def swapped_labels(merged: pd.DataFrame) -> dict[str, str]:
    """Labels with every ambiguous identity moved to its alternative category."""
    out = {}
    for _, row in merged.iterrows():
        cat = str(row["final_category"])
        alt = str(row.get("alternative_category") or "")
        if bool(row.get("ambiguous")) and alt in CATEGORIES:
            cat = alt
        out[row["persona_key"]] = cat
    return out


def mechanical_labels(roster: Mapping[str, list[dict[str, Any]]], rules: Mapping[str, Any]) -> dict[str, str]:
    """Role labels from fixed keyword rules on the affiliation string; first matching category in precedence order wins."""
    import re

    field = str(rules.get("source_field", "affiliation"))
    compiled = {cat: [re.compile(p, re.IGNORECASE) for p in rules["patterns"][cat]] for cat in rules["precedence"]}
    out: dict[str, str] = {}
    for row in roster["famous_ai"]:
        text = str(row.get(field) or "")
        label = str(rules.get("default", "other"))
        for cat in rules["precedence"]:
            if any(p.search(text) for p in compiled[cat]):
                label = cat
                break
        out[str(row["key"])] = label
    return out


def label_hash(labels: Mapping[str, str]) -> str:
    return hashlib.sha256("\n".join(f"{k},{v}" for k, v in sorted(labels.items())).encode()).hexdigest()
