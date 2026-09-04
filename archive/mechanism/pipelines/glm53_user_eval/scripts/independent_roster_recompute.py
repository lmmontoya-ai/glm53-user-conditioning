"""Recompute roster headline estimates without importing the main analysis module."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CONTRASTS = {
    "name": ("famous_coherent", "unknown_same_org"),
    "affiliation": ("unknown_same_org", "unknown_general"),
    "generic_fame": ("famous_nonai_control", "unknown_general"),
}


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(
    results: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    positive: str,
    negative: str,
) -> np.ndarray:
    schedule_by_id = {str(row["sample_id"]): row for row in schedule}
    values: dict[tuple[int, str, str], float] = {}
    scenarios = sorted({str(row["scenario_id"]) for row in schedule})
    for result in results:
        if not result.get("parse_valid"):
            continue
        sample_id = str(result["sample_id"])
        item = schedule_by_id[sample_id]
        values[(int(item["pair_index"]), str(item["scenario_id"]), str(item["condition"]))] = float(
            result["confidence_p"]
        )
    output = np.full((70, len(scenarios)), np.nan, dtype=float)
    for pair_index in range(70):
        for column, scenario_id in enumerate(scenarios):
            first = values.get((pair_index, scenario_id, positive))
            second = values.get((pair_index, scenario_id, negative))
            if first is not None and second is not None:
                output[pair_index, column] = first - second
    return output


def _bootstrap(matrix: np.ndarray, *, reps: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=float)
    for index in range(reps):
        rows = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        columns = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        values[index] = float(np.nanmean(matrix[np.ix_(rows, columns)]))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def recompute(
    results: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    result_ids = [str(row["sample_id"]) for row in results]
    schedule_ids = [str(row["sample_id"]) for row in schedule]
    if len(result_ids) != len(set(result_ids)) or len(schedule_ids) != len(set(schedule_ids)):
        raise ValueError("duplicate IDs in independent recomputation inputs")
    if set(result_ids) != set(schedule_ids):
        raise ValueError("independent recomputation requires complete matched inputs")
    matrices = {
        concept: _matrix(results, schedule, positive, negative)
        for concept, (positive, negative) in CONTRASTS.items()
    }
    return {
        "sample_count": len(results),
        "parse_rate": float(np.mean([bool(row["parse_valid"]) for row in results])),
        "name_effect_pp": float(np.nanmean(matrices["name"])),
        "name_ci95_pp": _bootstrap(matrices["name"], reps=reps, seed=seed),
        "negative_name_identity_count": int(
            np.count_nonzero(np.nanmean(matrices["name"], axis=1) < 0)
        ),
        "affiliation_effect_pp": float(np.nanmean(matrices["affiliation"])),
        "affiliation_ci95_pp": _bootstrap(
            matrices["affiliation"], reps=reps, seed=seed + 1
        ),
        "generic_fame_effect_pp": float(np.nanmean(matrices["generic_fame"])),
        "generic_fame_ci95_pp": _bootstrap(
            matrices["generic_fame"], reps=reps, seed=seed + 2
        ),
    }


def _compare(primary: dict[str, Any], independent: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "sample_count",
        "parse_rate",
        "name_effect_pp",
        "name_ci95_pp",
        "negative_name_identity_count",
        "affiliation_effect_pp",
        "affiliation_ci95_pp",
        "generic_fame_effect_pp",
        "generic_fame_ci95_pp",
    )
    differences: dict[str, float] = {}
    for key in keys:
        expected = np.asarray(primary[key], dtype=float)
        observed = np.asarray(independent[key], dtype=float)
        differences[key] = float(np.max(np.abs(expected - observed)))
    return {
        "maximum_absolute_differences": differences,
        "all_headline_estimates_match": all(value <= 1e-12 for value in differences.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--schedules", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-analysis", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = _read_jsonl(args.results)
    schedule = _read_jsonl(args.schedules)
    independent = recompute(
        results,
        schedule,
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    primary_payload = json.loads(args.primary_analysis.read_text(encoding="utf-8"))
    comparison = _compare(primary_payload["estimates"], independent)
    payload = {
        "schema_version": "glm53_roster_independent_recompute_v1",
        "input_hashes": {
            str(path): _sha256(path)
            for path in [*args.results, *args.schedules, args.primary_analysis]
        },
        "independent_estimates": independent,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not comparison["all_headline_estimates_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
