"""Independent v7 recomputation; deliberately imports no primary-analysis code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT.parent / "reference/transluce-user-awareness"
GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_nonai")
ALL_GROUPS = GROUPS + ("famous_ai_real",)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def make_matrices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    roster = json.loads((SOURCE / "core/personas2.json").read_text(encoding="utf-8"))
    stimuli = sorted({str(row["stimulus"]) for row in rows})
    lookup = {(str(row["group"]), str(row["persona"]), str(row["stimulus"])): row for row in rows}
    output = {}
    for group in ALL_GROUPS:
        personas = [str(row["key"]) for row in roster[group]]
        matrix = np.full((len(personas), len(stimuli)), np.nan)
        for i, persona in enumerate(personas):
            for j, stimulus in enumerate(stimuli):
                row = lookup[(group, persona, stimulus)]
                if row["score"] is not None:
                    score = float(row["score"])
                    if not 0.0 <= score <= 1.0:
                        raise ValueError("score outside [0,1]")
                    matrix[i, j] = 100.0 * max(score, 1.0 - score)
        output[group] = matrix
    return output


def calculate(matrices: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    center = np.nanmean(matrices["genpop"], axis=0)
    means = {}
    for group in GROUPS:
        person = np.nanmean(matrices[group] - center[None, :], axis=1)
        if not np.isfinite(person).all():
            raise ValueError("identity lacks all valid scores")
        means[group] = float(np.mean(person))
    value = means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]
    return float(value), means


def classify(point: float, interval: list[float]) -> str:
    low, high = interval
    if high < 0 and point <= -0.50:
        return "confirmed_target_sized_interaction"
    if high < 0:
        return "confirmed_small_interaction"
    if low > -0.50:
        return "target_magnitude_ruled_out"
    if point < 0:
        return "directional_ambiguous"
    return "null_or_opposite_interaction"


def address_effect(matrices: dict[str, np.ndarray]) -> float:
    roster = json.loads((SOURCE / "core/personas2.json").read_text(encoding="utf-8"))
    constructed = {
        str(row["key"]).removeprefix("fai2_"): index
        for index, row in enumerate(roster["famous_ai"])
    }
    public = {
        str(row["key"]).removeprefix("fai2r_"): index
        for index, row in enumerate(roster["famous_ai_real"])
    }
    slugs = sorted(set(constructed) & set(public))
    if len(slugs) != 59:
        raise ValueError("independent address map does not contain 59 identities")
    difference = (
        matrices["famous_ai_real"][[public[slug] for slug in slugs]]
        - matrices["famous_ai"][[constructed[slug] for slug in slugs]]
    )
    return float(np.mean(np.nanmean(difference, axis=1)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scores", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--primary-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = yaml.safe_load(args.prereg.read_text(encoding="utf-8"))
    primary = json.loads(args.primary_analysis.read_text(encoding="utf-8"))
    rows = read_rows(args.raw_scores)
    if len(rows) != 34400:
        raise ValueError("verifier requires exactly 34,400 rows")
    matrices = make_matrices(rows)
    point, means = calculate(matrices)
    components = {
        "famous_ai_minus_unknown_ai_pp": means["famous_ai"] - means["unknown_ai"],
        "famous_nonai_minus_genpop_pp": means["famous_nonai"] - means["genpop"],
        "unknown_ai_minus_genpop_pp": means["unknown_ai"] - means["genpop"],
        "matched_public_minus_constructed_address_pp": address_effect(matrices),
    }
    reps = int(prereg["analysis"]["uncertainty"]["bootstrap_reps"])
    seed = int(prereg["analysis"]["uncertainty"]["verifier_seed"])
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for rep in range(reps):
        task_idx = rng.integers(0, 100, size=100)
        pair_idx = rng.integers(0, 70, size=70)
        fn_idx = rng.integers(0, 70, size=70)
        gp_idx = rng.integers(0, 70, size=70)
        sampled = {
            "famous_ai": matrices["famous_ai"][pair_idx][:, task_idx],
            "unknown_ai": matrices["unknown_ai"][pair_idx][:, task_idx],
            "famous_nonai": matrices["famous_nonai"][fn_idx][:, task_idx],
            "genpop": matrices["genpop"][gp_idx][:, task_idx],
        }
        draws[rep] = calculate(sampled)[0]
    interval = [float(value) for value in np.percentile(draws, [2.5, 97.5])]
    primary_point = float(primary["primary"]["interaction_pp"])
    primary_interval = [float(value) for value in primary["primary"]["ci95_pp"]]
    checks = {
        "point_agreement": abs(point - primary_point) <= 1e-10,
        "group_mean_agreement": all(
            abs(means[group] - float(primary["group_mean_deltas_pp"][group])) <= 1e-10
            for group in GROUPS
        ),
        "bootstrap_endpoint_agreement": max(
            abs(left - right) for left, right in zip(interval, primary_interval, strict=True)
        )
        <= 0.05,
        "decision_classification_agreement": classify(point, interval)
        == primary["provisional_statistical_state"],
        "component_agreement": (
            abs(
                components["famous_ai_minus_unknown_ai_pp"]
                - float(primary["components"]["famous_ai_minus_unknown_ai"]["point_pp"])
            )
            <= 1e-10
            and abs(
                components["famous_nonai_minus_genpop_pp"]
                - float(primary["components"]["famous_nonai_minus_genpop"]["point_pp"])
            )
            <= 1e-10
            and abs(
                components["unknown_ai_minus_genpop_pp"]
                - float(primary["components"]["unknown_ai_minus_genpop"]["point_pp"])
            )
            <= 1e-10
            and abs(
                components["matched_public_minus_constructed_address_pp"]
                - float(primary["matched_public_minus_constructed_address"]["point_pp"])
            )
            <= 1e-10
        ),
    }
    payload = {
        "schema_version": "glm53_transluce_interaction_v7_verification_v1",
        "point_pp": point,
        "group_mean_deltas_pp": means,
        "components": components,
        "ci95_pp": interval,
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
        "classification": classify(point, interval),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
