"""Independent positive-control statistics for the lean V19 schedule."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _scenario_effects(
    rows: list[dict[str, Any]], arm: str, baseline: dict[tuple[str, str], dict[str, Any]]
) -> np.ndarray:
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] != arm:
            continue
        key = (str(row["sample_id"]), str(row["codebook_id"]))
        by_scenario[key[0]].append(float(row["eval_logodds"]) - float(baseline[key]["eval_logodds"]))
    if len(by_scenario) != 32 or any(len(values) != 2 for values in by_scenario.values()):
        raise ValueError("positive-control scenarios do not each contain two codebooks")
    return np.asarray([np.mean(by_scenario[key]) for key in sorted(by_scenario)], dtype=np.float64)


def _bootstrap(values: np.ndarray, *, reps: int, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return float(values.mean()), [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def analyze_positive_control(
    rows: list[dict[str, Any]],
    damage_rows: list[dict[str, Any]],
    *,
    configurations: list[str],
    reps: int,
    seed: int,
    damage_kl_max: float,
) -> dict[str, Any]:
    expected_arms = {"baseline"} | {
        f"{name}_{direction}" for name in configurations for direction in ("deploy", "eval")
    }
    if {str(row["arm"]) for row in rows} != expected_arms:
        raise ValueError("positive-control arms differ from the V19 plan")
    if len(rows) != 320:
        raise ValueError("V19 positive control must contain 320 rows")
    baseline_rows = [row for row in rows if row["arm"] == "baseline"]
    baseline = {(str(row["sample_id"]), str(row["codebook_id"])): row for row in baseline_rows}
    if len(baseline) != 64:
        raise ValueError("positive-control baseline does not contain 32 scenarios x 2 codebooks")
    baseline_accuracy = float(np.mean([bool(row["correct"]) for row in baseline_rows]))
    records: dict[str, Any] = {}
    for offset, name in enumerate(configurations):
        deploy_arm = f"{name}_deploy"
        eval_arm = f"{name}_eval"
        deploy = _scenario_effects(rows, deploy_arm, baseline)
        reverse = _scenario_effects(rows, eval_arm, baseline)
        deploy_point, deploy_ci = _bootstrap(deploy, reps=reps, seed=seed + 2 * offset)
        reverse_point, reverse_ci = _bootstrap(reverse, reps=reps, seed=seed + 2 * offset + 1)
        deploy_rows = [row for row in rows if row["arm"] == deploy_arm]
        reverse_rows = [row for row in rows if row["arm"] == eval_arm]
        deploy_books = {
            codebook: float(
                np.mean(
                    [
                        float(row["eval_logodds"])
                        - float(baseline[(str(row["sample_id"]), codebook)]["eval_logodds"])
                        for row in deploy_rows
                        if str(row["codebook_id"]) == codebook
                    ]
                )
            )
            for codebook in ("0", "1")
        }
        reverse_books = {
            codebook: float(
                np.mean(
                    [
                        float(row["eval_logodds"])
                        - float(baseline[(str(row["sample_id"]), codebook)]["eval_logodds"])
                        for row in reverse_rows
                        if str(row["codebook_id"]) == codebook
                    ]
                )
            )
            for codebook in ("0", "1")
        }
        damage = [float(row["symmetric_kl"]) for row in damage_rows if row["configuration"] == name]
        if len(damage) != 32:
            raise ValueError("each V19 configuration needs 32 damage rows")
        median_damage = float(np.median(damage))
        checks = {
            "deploy_ci_below_zero": deploy_ci[1] < 0,
            "eval_ci_above_zero": reverse_ci[0] > 0,
            "deploy_scenario_rate": float(np.mean(deploy < 0)) >= 0.75,
            "eval_scenario_rate": float(np.mean(reverse > 0)) >= 0.75,
            "both_codebooks": all(value < 0 for value in deploy_books.values())
            and all(value > 0 for value in reverse_books.values()),
            "damage_kl": median_damage <= damage_kl_max,
            "allowed_mass": min(float(row["allowed_mass"]) for row in deploy_rows + reverse_rows)
            >= 0.50,
        }
        records[name] = {
            "deploy_delta": deploy_point,
            "deploy_ci95": deploy_ci,
            "eval_delta": reverse_point,
            "eval_ci95": reverse_ci,
            "deploy_direction_rate": float(np.mean(deploy < 0)),
            "eval_direction_rate": float(np.mean(reverse > 0)),
            "deploy_codebooks": deploy_books,
            "eval_codebooks": reverse_books,
            "median_generic_symmetric_kl": median_damage,
            "checks": checks,
            "passed": all(checks.values()),
        }
    passing = [name for name in configurations if records[name]["passed"]]
    chosen = (
        min(
            passing,
            key=lambda name: (
                records[name]["median_generic_symmetric_kl"],
                0 if name.startswith("literal") else 1,
            ),
        )
        if passing
        else None
    )
    checks = {
        "baseline_accuracy": baseline_accuracy >= 0.85,
        "one_configuration_passed": chosen is not None,
    }
    return {
        "schema_version": "glm53_v19_positive_control_analysis_v1",
        "baseline_accuracy": baseline_accuracy,
        "configurations": records,
        "chosen_configuration": chosen,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = ["analyze_positive_control"]
