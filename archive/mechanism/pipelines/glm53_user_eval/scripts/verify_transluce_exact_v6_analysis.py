"""Independently recompute the exact v6 headline statistics from immutable rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mann_whitney_z(left: list[float], right: list[float]) -> float:
    tagged = sorted([(value, 0) for value in left] + [(value, 1) for value in right])
    rank_sum = 0.0
    index = 0
    while index < len(tagged):
        end = index + 1
        while end < len(tagged) and tagged[end][0] == tagged[index][0]:
            end += 1
        midrank = ((index + 1) + end) / 2
        rank_sum += midrank * sum(source == 0 for _value, source in tagged[index:end])
        index = end
    n_left, n_right = len(left), len(right)
    u_value = rank_sum - n_left * (n_left + 1) / 2
    expected = n_left * n_right / 2
    scale = math.sqrt(n_left * n_right * (n_left + n_right + 1) / 12)
    return (u_value - expected) / scale


def close(actual: float, expected: float, *, label: str, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scores", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.raw_scores.read_text(encoding="utf-8").splitlines()]
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    primary = json.loads(args.analysis.read_text(encoding="utf-8"))
    values: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row["score"] is None:
            continue
        score = 100.0 * float(row["score"])
        values[(row["group"], row["persona"], row["stimulus"])] = max(score, 100.0 - score)

    stimuli = sorted({row["stimulus"] for row in rows})
    genpop_keys = [row["key"] for row in roster["genpop"]]
    centers = {
        stimulus: statistics.fmean(
            values[("genpop", persona, stimulus)]
            for persona in genpop_keys
            if ("genpop", persona, stimulus) in values
        )
        for stimulus in stimuli
    }
    person_deltas: dict[str, dict[str, float]] = defaultdict(dict)
    stats: dict[str, dict[str, float | int | list[float] | None]] = {}
    for group in GROUPS:
        for persona in [row["key"] for row in roster[group]]:
            deltas = [
                values[(group, persona, stimulus)] - centers[stimulus]
                for stimulus in stimuli
                if (group, persona, stimulus) in values
            ]
            person_deltas[group][persona] = statistics.fmean(deltas)
        ordered = sorted(person_deltas[group].values())
        z_value = None
        adjusted_p = None
        if group != "genpop":
            z_value = mann_whitney_z(ordered, sorted(person_deltas["genpop"].values()))
            adjusted_p = min(1.0, math.erfc(abs(z_value) / math.sqrt(2)) * 28)
        stats[group] = {
            "n": len(ordered),
            "mean_pp": statistics.fmean(ordered),
            "median_pp": statistics.median(ordered),
            "iqr_source_indexed_pp": [ordered[len(ordered) // 4], ordered[3 * len(ordered) // 4]],
            "mann_whitney_z_vs_genpop": z_value,
            "bonferroni_28_p_vs_genpop": adjusted_p,
        }

    paired_people: list[float] = []
    for index in range(70):
        famous = roster["famous_ai"][index]["key"]
        unknown = roster["unknown_ai"][index]["key"]
        paired_cells = [
            values[("famous_ai", famous, stimulus)]
            - values[("unknown_ai", unknown, stimulus)]
            for stimulus in stimuli
            if ("famous_ai", famous, stimulus) in values
            and ("unknown_ai", unknown, stimulus) in values
        ]
        paired_people.append(statistics.fmean(paired_cells))
    paired = statistics.fmean(paired_people)

    for group in GROUPS:
        expected = primary["source_exact_group_stats"][group]
        close(float(stats[group]["median_pp"]), float(expected["median_pp"]), label=f"{group} median")
        close(float(stats[group]["mean_pp"]), float(primary["group_mean_deltas_pp"][group]), label=f"{group} mean")
        if group != "genpop":
            close(
                float(stats[group]["bonferroni_28_p_vs_genpop"]),
                float(expected["bonferroni_28_p_vs_genpop"]),
                label=f"{group} adjusted p",
                tolerance=1e-3,
            )
    close(paired, float(primary["same_index_famous_minus_unknown_pp"]), label="paired F-U")

    payload: dict[str, Any] = {
        "schema_version": "glm53_transluce_exact_independent_verification_v1",
        "passed": True,
        "raw_scores_sha256": sha256_file(args.raw_scores),
        "primary_analysis_sha256": sha256_file(args.analysis),
        "roster_sha256": sha256_file(args.roster),
        "row_count": len(rows),
        "recomputed_group_stats": stats,
        "recomputed_same_index_famous_minus_unknown_pp": paired,
        "point_estimate_tolerance": 1e-10,
        "adjusted_p_tolerance": 1e-3,
        "adjusted_p_tolerance_reason": (
            "Independent standard-library summation can perturb near-ties relative to "
            "NumPy before the source's rank calculation; threshold conclusions must agree."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
