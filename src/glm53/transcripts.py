"""Transcript-level measures: response length, judge annotations, and matched reading samples.

Length units: tokens are the provider-reported output tokens of the confidence turn minus its
reasoning tokens (visible tokens); sentences are counted by splitting the visible confidence-turn
text at sentence-ending punctuation. Judge annotations are the committed blinded ratings from two
independent model judges on 2,000 transcripts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .decompose import bootstrap_interaction
from .io import PRIMARY_GROUPS, read_jsonl

JUDGES = ("luna_max", "terra_high")
CONTENT_FIELDS = (
    "explicit_evaluation_or_scrutiny_inference",
    "deference_to_user_expertise",
    "caution_or_risk_sensitivity",
    "hedging_or_uncertainty",
    "social_desirability_or_impression_management",
    "reasoning_supports_reported_confidence",
)
FIELD_UNITS = {
    "explicit_evaluation_or_scrutiny_inference": "fraction of transcripts (0 to 1)",
    "deference_to_user_expertise": "points on a 0 to 2 scale",
    "caution_or_risk_sensitivity": "points on a 0 to 2 scale",
    "hedging_or_uncertainty": "points on a 0 to 2 scale",
    "social_desirability_or_impression_management": "points on a 0 to 2 scale",
    "reasoning_supports_reported_confidence": "points on a 0 to 2 scale",
}


def sentence_count(text: str) -> int:
    """Number of sentences in visible text, split at ., !, or ? followed by whitespace or the end."""
    clean = text.strip()
    if not clean:
        return 0
    return len([s for s in re.split(r"(?<=[.!?])(?:\s+|$)", clean) if s.strip()])


def load_transcripts(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 28000:
        raise ValueError(f"expected 28,000 transcript rows, found {len(rows)}")
    return rows


def matrices(rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]) -> dict[str, np.ndarray]:
    """70-by-100 matrices of a row-level value for the four primary groups, NaN where absent."""
    out = {g: np.full((70, 100), np.nan) for g in PRIMARY_GROUPS}
    for row in rows:
        if row["group"] not in out:
            continue
        j = int(str(row["stimulus_id"]).removeprefix("dd_"))
        out[row["group"]][int(row["identity_index"]), j] = value(row)
    return out


def _length(row: dict[str, Any], key: str) -> float:
    if key == "confidence_turn_visible_tokens":
        value = row["confidence_turn_usage"]["visible_tokens"]
    elif key == "confidence_turn_reasoning_tokens":
        value = row["confidence_turn_usage"]["reasoning_tokens"]
    elif key == "first_turn_visible_tokens":
        value = row["first_turn_usage"]["visible_tokens"]
    elif key == "first_turn_reasoning_tokens":
        value = row["first_turn_usage"]["reasoning_tokens"]
    elif key == "confidence_visible_sentence_count":
        value = row["confidence_visible_sentence_count"]
    elif key == "first_visible_sentence_count":
        value = row["first_visible_sentence_count"]
    elif key == "confidence_sentences_recomputed":
        value = sentence_count(str(row["confidence_turn_answer"]))
    else:
        raise KeyError(key)
    return float("nan") if value is None else float(value)


LENGTH_SEED_OFFSETS = {
    "first_turn_reasoning_tokens": 100,
    "first_turn_visible_tokens": 101,
    "confidence_turn_reasoning_tokens": 102,
    "confidence_turn_visible_tokens": 103,
    "first_visible_sentence_count": 104,
    "confidence_visible_sentence_count": 105,
}


def length_analysis(rows: list[dict[str, Any]], *, reps: int, seed: int) -> dict[str, Any]:
    """Group means and the four-group interaction for each length measure."""
    out: dict[str, Any] = {}
    for key, offset in LENGTH_SEED_OFFSETS.items():
        mats = matrices(rows, lambda r, k=key: _length(r, k))
        result = bootstrap_interaction(mats, reps=reps, seed=seed + offset)
        result["raw_group_means"] = {g: float(np.nanmean(mats[g])) for g in PRIMARY_GROUPS}
        result["unit"] = "tokens" if "tokens" in key else "sentences"
        out[key] = result
    recomputed = matrices(rows, lambda r: _length(r, "confidence_sentences_recomputed"))
    stored = matrices(rows, lambda r: _length(r, "confidence_visible_sentence_count"))
    out["sentence_count_recomputation_matches_stored"] = bool(
        np.array_equal(np.nan_to_num(recomputed["genpop"]), np.nan_to_num(stored["genpop"]))
    )
    return out


def load_judgments(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """sample_id -> judge -> parsed ratings, from the committed per-row judge files."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for judge in JUDGES:
        for path in sorted((root / judge / "rows").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            out.setdefault(str(record["sample_id"]), {})[judge] = dict(record["parsed"])
    return out


def annotation_analysis(
    rows: list[dict[str, Any]], judged: Mapping[str, Mapping[str, Mapping[str, Any]]], *, reps: int, seed: int
) -> dict[str, Any]:
    """Inter-judge agreement and the four-group interaction of each rated dimension."""
    source = {row["sample_id"]: row for row in rows}
    joined = []
    for sample_id, values in judged.items():
        if set(values) != set(JUDGES):
            raise ValueError(f"{sample_id} lacks a rating from both judges")
        joined.append(source[sample_id] | {"luna": values["luna_max"], "terra": values["terra_high"]})
    if len(joined) != 2000:
        raise ValueError(f"expected 2,000 judged transcripts, found {len(joined)}")
    agreement: dict[str, Any] = {}
    dimensions: dict[str, Any] = {}
    for k, field in enumerate(CONTENT_FIELDS):
        left = [int(r["luna"][field]) for r in joined]
        right = [int(r["terra"][field]) for r in joined]
        binary = field == "explicit_evaluation_or_scrutiny_inference"
        kappa = float(cohen_kappa_score(left, right, weights=None if binary else "quadratic"))
        exact = float(np.mean(np.asarray(left) == np.asarray(right)))
        prevalence_alt = exact >= 0.90 and len(set(left)) >= 2 and len(set(right)) >= 2
        agreement[field] = {
            "kappa": kappa,
            "kind": "cohen" if binary else "quadratic_weighted",
            "exact_agreement": exact,
            "adequate": kappa >= 0.60 or prevalence_alt,
            "prevalence_aware_alternative_passed": prevalence_alt,
        }
        per_judge = {}
        for j, judge in enumerate(("luna", "terra")):
            mats = matrices(joined, lambda r, f=field, jj=judge: float(r[jj][f]))
            per_judge[judge] = bootstrap_interaction(mats, reps=reps, seed=seed + (k + 1) * 10 + j)
        combined = matrices(joined, lambda r, f=field: 0.5 * (float(r["luna"][f]) + float(r["terra"][f])))
        dimensions[field] = {
            "unit": FIELD_UNITS[field],
            "combined": bootstrap_interaction(combined, reps=reps, seed=seed + 100 + (k + 1)),
            "per_judge": per_judge,
            "combined_interpretable": agreement[field]["adequate"],
            "group_means_combined": {g: float(np.nanmean(combined[g])) for g in PRIMARY_GROUPS},
        }
    return {"judged_rows": len(joined), "agreement": agreement, "dimensions": dimensions, "seed": seed}


def redact(text: str, row: Mapping[str, Any]) -> str:
    """Replace the profile's name, affiliation, organization, and email in text with placeholders."""
    values = [
        (str(row.get("email", "")), "<USER_AFFILIATION>"),
        (str(row.get("affiliation", "")), "<USER_AFFILIATION>"),
        (str(row.get("name", "")), "<USER_NAME>"),
        (str(row.get("organization", "")), "<USER_AFFILIATION>"),
    ]
    result = text
    for literal, replacement in sorted(values, key=lambda item: -len(item[0])):
        if literal.strip():
            result = re.sub(re.escape(literal), replacement, result, flags=re.IGNORECASE)
    return result


def matched_sets(rows: list[dict[str, Any]], *, n_sets: int, seed: int) -> list[dict[str, Any]]:
    """Seeded random sets: a famous-AI identity, its twin, a famous non-AI identity, and a genpop identity on one dilemma."""
    by_key = {(r["group"], int(r["identity_index"]), r["stimulus_id"]): r for r in rows}
    rng = np.random.default_rng(seed)
    sets = []
    attempts = 0
    while len(sets) < n_sets and attempts < 50 * n_sets:
        attempts += 1
        i = int(rng.integers(0, 70))
        d = f"dd_{int(rng.integers(0, 100)):04d}"
        j = int(rng.integers(0, 70))
        k = int(rng.integers(0, 70))
        members = [("famous_ai", i), ("unknown_ai", i), ("famous_nonai", j), ("genpop", k)]
        picked = [by_key.get((g, idx, d)) for g, idx in members]
        if any(p is None or not p["parse_valid"] for p in picked):
            continue
        sets.append({"set_id": f"set_{len(sets) + 1:02d}", "stimulus_id": d, "members": picked})
    return sets


def blinded_copy(item: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """The same set with profile fields removed, identity strings redacted, and member order shuffled."""
    order = rng.permutation(len(item["members"]))
    members = []
    for n, position in enumerate(order):
        row = item["members"][int(position)]
        members.append(
            {
                "reader_label": f"transcript_{chr(65 + n)}",
                "dilemma_text": row["dilemma_text"],
                "first_turn_reasoning": redact(row["first_turn_reasoning"], row),
                "first_turn_answer": redact(row["first_turn_answer"], row),
                "confidence_turn_reasoning": redact(row["confidence_turn_reasoning"], row),
                "confidence_turn_answer": redact(row["confidence_turn_answer"], row),
            }
        )
    return {"set_id": item["set_id"], "stimulus_id": item["stimulus_id"], "members": members}


def sets_frame(sets: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for item in sets:
        for row in item["members"]:
            records.append(
                {
                    "set_id": item["set_id"],
                    "stimulus_id": item["stimulus_id"],
                    "group": row["group"],
                    "persona_key": row["persona_key"],
                    "first_turn_choice": row["first_turn_choice"],
                    "folded_confidence": row["folded_confidence"],
                    "confidence_visible_tokens": row["confidence_turn_usage"]["visible_tokens"],
                }
            )
    return pd.DataFrame(records)
