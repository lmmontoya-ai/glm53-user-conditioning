"""Independent recomputation for V20 result tables.

This module imports no V20 analysis or supervisor code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

GROUPS = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
FULL_ARMS = (
    "baseline",
    "all16_deploy",
    "all16_eval",
    "status_deploy",
    "actor_deploy",
    "formality",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _matrices(
    rows: list[dict[str, Any]], *, arm: str, identity_count: int
) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["arm"] == arm]
    tasks = sorted({str(row["stimulus_id"]) for row in selected})
    matrices: dict[str, np.ndarray] = {}
    for group in GROUPS:
        group_rows = [row for row in selected if row["group"] == group]
        lookup = {
            (int(row["stage_index"]), str(row["stimulus_id"])): float(
                row["expected_folded_confidence"]
            )
            for row in group_rows
        }
        matrices[group] = np.asarray(
            [
                [lookup.get((identity, task), np.nan) for task in tasks]
                for identity in range(identity_count)
            ],
            dtype=np.float64,
        )
    return matrices


def _point(matrices: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    center = np.nanmean(matrices["genpop"], axis=0)
    means = {
        group: float(np.mean(np.nanmean(matrix - center[None, :], axis=1)))
        for group, matrix in matrices.items()
    }
    value = means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]
    return float(value), means


def _candidate_ci(rows: list[dict[str, Any]], *, reps: int, seed: int) -> list[float]:
    baseline = _matrices(rows, arm="baseline", identity_count=16)
    candidate = _matrices(rows, arm="all16_deploy", identity_count=16)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    n_pairs, n_tasks = baseline["famous_ai"].shape
    for rep in range(reps):
        pair = rng.integers(0, n_pairs, n_pairs)
        fn = rng.integers(0, n_pairs, n_pairs)
        gp = rng.integers(0, n_pairs, n_pairs)
        task = rng.integers(0, n_tasks, n_tasks)
        indices = {
            "famous_ai": pair,
            "unknown_ai": pair,
            "famous_nonai": fn,
            "genpop": gp,
        }
        left = {group: baseline[group][indices[group]][:, task] for group in GROUPS}
        right = {group: candidate[group][indices[group]][:, task] for group in GROUPS}
        draws[rep] = _point(right)[0] - _point(left)[0]
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def verify_local_parity(
    root: Path,
    *,
    parent_interaction_pp: float = -0.39025945842352083,
    expected_rows_per_group: int = 352,
) -> dict[str, Any]:
    rows = _jsonl(root / "raw_scores.jsonl")
    analysis = _json(root / "analysis.json")
    converted = [dict(row, arm="baseline", stage_index=row["analysis_index"]) for row in rows]
    point, means = _point(_matrices(converted, arm="baseline", identity_count=16))
    codebook_points = {
        codebook: _point(
            _matrices(
                [row for row in converted if str(row["codebook_id"]) == codebook],
                arm="baseline",
                identity_count=16,
            )
        )[0]
        for codebook in ("0", "1")
    }
    matched = [row for row in converted if row["original_folded_confidence"] is not None]
    matched_point, _ = _point(_matrices(matched, arm="baseline", identity_count=16))
    masses = np.asarray([float(row["allowed_mass"]) for row in rows], dtype=np.float64)
    rates = [
        sum(row["group"] == group for row in rows) / expected_rows_per_group
        for group in GROUPS
    ]
    derived_checks = {
        "negative": point < 0,
        "codebooks": all(value < 0 for value in codebook_points.values()),
        "api_matched_negative": matched_point < 0,
        "magnitude_or_ci": abs(matched_point) / abs(parent_interaction_pp) >= 0.40
        or float(analysis["ci90_pp"][1]) < 0,
        "components": means["famous_ai"] - means["unknown_ai"] <= 0
        and means["famous_nonai"] - means["genpop"] >= 0,
        "mass_median": float(np.median(masses)) >= 0.80,
        "mass_p05": float(np.percentile(masses, 5)) >= 0.50,
        "argmax": float(np.mean([bool(row["full_vocab_argmax_allowed"]) for row in rows]))
        >= 0.95,
        "codebook_artifact": max(codebook_points.values()) < 0,
        "missingness": max(rates) - min(rates) <= 0.005,
    }
    checks = {
        "point": abs(point - analysis["interaction_pp"]) <= 1e-10,
        "group_means": all(
            abs(means[group] - analysis["group_means_pp"][group]) <= 1e-10
            for group in GROUPS
        ),
        "codebook_points": all(
            abs(codebook_points[key] - analysis["codebook_interactions_pp"][key]) <= 1e-10
            for key in ("0", "1")
        ),
        "api_matched_point": abs(matched_point - analysis["api_matched_interaction_pp"])
        <= 1e-10,
        "gate_checks": derived_checks == analysis["checks"],
        "gate_decision": bool(all(derived_checks.values())) == bool(analysis["passed"]),
        "rows": len(rows) == 1404,
        "api_matched_rows": len(matched) == 1401,
        "keys_unique": len(rows)
        == len({(row["group"], row["persona_key"], row["stimulus_id"]) for row in rows}),
    }
    return {
        "schema_version": "glm53_v20_parity_verification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "interaction_pp": point,
        "api_matched_interaction_pp": matched_point,
        "codebook_interactions_pp": codebook_points,
        "recomputed_gate_checks": derived_checks,
        "recomputed_gate_passed": all(derived_checks.values()),
    }


def _classify(
    analysis: dict[str, Any], *, verification_passed: bool
) -> tuple[str, dict[str, bool]]:
    candidate = analysis["arms"]["all16_deploy"]
    reverse = analysis["arms"]["all16_eval"]
    status = analysis["arms"]["status_deploy"]
    actor = analysis["arms"]["actor_deploy"]
    formality = analysis["arms"]["formality"]
    changes = candidate["group_changes_pp"]
    target_change = changes["famous_ai"] - changes["unknown_ai"]
    fame_change = changes["famous_nonai"] - changes["genpop"]
    quality = analysis["quality"]
    checks = {
        "baseline_negative": analysis["baseline_interaction_pp"] < 0,
        "deployment_ci_above_zero": candidate["delta_ci95_pp"][0] > 0,
        "meaningful_attenuation": candidate["fraction_removed"] >= 0.30
        or candidate["delta_pp"] >= 0.20,
        "reverse_direction": reverse["delta_pp"] < 0,
        "status_support": status["delta_pp"] > 0
        and status["delta_pp"] >= 0.5 * candidate["delta_pp"],
        "not_actor_only": actor["delta_pp"] < candidate["delta_pp"],
        "not_formality": abs(formality["delta_pp"]) < 0.5 * abs(candidate["delta_pp"]),
        "signflip_specificity": analysis["signflip_control"]["candidate_exceeds_every_null"]
        and analysis["signflip_control"]["add_one_empirical_p"] <= 1.0 / 21.0,
        "both_codebooks": all(
            value > 0 for value in analysis["candidate_codebook_delta_pp"].values()
        ),
        "target_pair_change_dominates": target_change > 0
        and abs(target_change) >= abs(fame_change),
        "output_integrity": all(
            record["allowed_mass_median"] >= 0.80
            and record["allowed_mass_p05"] >= 0.50
            and record["full_vocab_argmax_allowed_rate"] >= 0.95
            for record in quality.values()
        ),
        "independent_verification": verification_passed,
    }
    if not verification_passed:
        state = "invalid_independent_verification"
    elif all(checks.values()):
        state = "fixed_hua_intervention_selectively_attenuates_user_conditioning"
    elif not checks["deployment_ci_above_zero"] or not checks["meaningful_attenuation"]:
        state = "no_resolved_deployment_attenuation"
    elif not checks["signflip_specificity"]:
        state = "nonspecific_activation_perturbation"
    elif not checks["not_formality"]:
        state = "formality_confounded_attenuation"
    elif not checks["status_support"] and not checks["not_actor_only"]:
        state = "actor_semantics_better_explain_attenuation"
    elif not checks["reverse_direction"]:
        state = "asymmetric_hua_sensitivity"
    else:
        state = "attenuation_without_full_specificity"
    return state, checks


def verify_causal(
    root: Path,
    *,
    pilot_task_ids: list[str],
    primary_bootstrap_ci: list[float],
    reps: int,
    seed: int,
    expected_decision: str | None = None,
) -> dict[str, Any]:
    full_rows = _jsonl(root / "full_rows.jsonl")
    null_rows = _jsonl(root / "null_rows.jsonl")
    analysis = _json(root / "analysis.json")
    points = {arm: _point(_matrices(full_rows, arm=arm, identity_count=16))[0] for arm in FULL_ARMS}
    baseline = points["baseline"]
    pilot_base_rows = [
        row
        for row in full_rows
        if row["arm"] == "baseline"
        and int(row["stage_index"]) < 4
        and str(row["stimulus_id"]) in set(pilot_task_ids)
    ]
    pilot_candidate_rows = [
        row
        for row in full_rows
        if row["arm"] == "all16_deploy"
        and int(row["stage_index"]) < 4
        and str(row["stimulus_id"]) in set(pilot_task_ids)
    ]
    pilot_joined = pilot_base_rows + pilot_candidate_rows
    pilot_base = _point(_matrices(pilot_joined, arm="baseline", identity_count=4))[0]
    pilot_candidate = _point(_matrices(pilot_joined, arm="all16_deploy", identity_count=4))[0]
    null_arms = sorted({str(row["arm"]) for row in null_rows})
    null_deltas = {
        arm: _point(_matrices(null_rows, arm=arm, identity_count=4))[0] - pilot_base
        for arm in null_arms
    }
    exceedances = sum(value >= pilot_candidate - pilot_base for value in null_deltas.values())
    independent_ci = _candidate_ci(full_rows, reps=reps, seed=seed)
    checks = {
        "full_rows": len(full_rows) == 8424,
        "null_rows": len(null_rows) == 1600,
        "full_arms": sorted({row["arm"] for row in full_rows}) == sorted(FULL_ARMS),
        "null_arms": null_arms == [f"signflip_{index:02d}" for index in range(20)],
        "full_keys_unique": len(full_rows)
        == len({(row["sample_id"], row["arm"]) for row in full_rows}),
        "null_keys_unique": len(null_rows)
        == len({(row["sample_id"], row["arm"]) for row in null_rows}),
        "baseline": abs(baseline - analysis["baseline_interaction_pp"]) <= 1e-10,
        "points": all(
            abs(points[arm] - analysis["arms"][arm]["interaction_pp"]) <= 1e-10
            and abs(points[arm] - baseline - analysis["arms"][arm]["delta_pp"]) <= 1e-10
            for arm in FULL_ARMS[1:]
        ),
        "pilot_candidate": abs(
            pilot_candidate - pilot_base - analysis["signflip_control"]["pilot_candidate_delta_pp"]
        )
        <= 1e-10,
        "null_points": all(
            abs(null_deltas[arm] - analysis["signflip_control"]["null_delta_pp"][arm]) <= 1e-10
            for arm in null_arms
        ),
        "empirical_p": abs(
            (1 + exceedances) / 21.0 - analysis["signflip_control"]["add_one_empirical_p"]
        )
        <= 1e-12,
        "bootstrap_ci": max(
            abs(left - right)
            for left, right in zip(independent_ci, primary_bootstrap_ci, strict=True)
        )
        <= 0.05,
    }
    preliminary_pass = all(checks.values())
    state, decision_checks = _classify(analysis, verification_passed=preliminary_pass)
    if expected_decision is not None:
        checks["decision"] = state == expected_decision
    passed = all(checks.values())
    return {
        "schema_version": "glm53_v20_causal_verification_v1",
        "passed": passed,
        "checks": checks,
        "independent_candidate_ci95_pp": independent_ci,
        "recomputed_decision": state,
        "recomputed_decision_checks": decision_checks,
    }


__all__ = ["verify_causal", "verify_local_parity"]
