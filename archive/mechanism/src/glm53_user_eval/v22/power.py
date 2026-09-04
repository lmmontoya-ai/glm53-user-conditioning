"""Empirical power planning for the V22 paired context experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_nonai")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_prereg(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "glm53_user_eval_v22_prereg_v1":
        raise ValueError("V22 preregistration schema mismatch")
    return payload


def repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_parent_locks(root: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    checked: dict[str, str] = {}
    for parent_name in ("v6", "v7"):
        parent = prereg["parent_evidence"][parent_name]
        tag_commit = __import__("subprocess").check_output(
            ["git", "rev-list", "-n", "1", parent["tag"]],
            cwd=root,
            text=True,
        ).strip()
        if tag_commit != parent["commit"]:
            raise ValueError(f"{parent_name} tag does not resolve to its locked commit")
        for item_name, item in parent["files"].items():
            path = repo_path(root, item["path"])
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise ValueError(f"{parent_name} {item_name} hash mismatch")
            checked[f"{parent_name}.{item_name}"] = actual
    roster = prereg["inputs"]["roster"]
    roster_path = repo_path(root, roster["path"])
    actual_roster = sha256_file(roster_path)
    if actual_roster != roster["sha256"]:
        raise ValueError("roster hash mismatch")
    checked["roster"] = actual_roster
    return checked


def load_folded_matrices(
    raw_path: Path, roster_path: Path
) -> tuple[dict[str, np.ndarray], list[str]]:
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lookup = {
        (str(row["group"]), str(row["persona"]), str(row["stimulus"])): row
        for row in rows
    }
    stimuli = sorted({str(row["stimulus"]) for row in rows})
    if stimuli != [f"dd_{index:04d}" for index in range(100)]:
        raise ValueError("parent rows do not contain the locked 100 dilemmas")
    matrices: dict[str, np.ndarray] = {}
    for group in GROUPS:
        people = [str(row["key"]) for row in roster[group]]
        if len(people) != 70:
            raise ValueError(f"{group} does not contain 70 locked identities")
        matrix = np.full((70, 100), np.nan, dtype=np.float64)
        for person_index, persona in enumerate(people):
            for task_index, stimulus in enumerate(stimuli):
                row = lookup.get((group, persona, stimulus))
                if row is None:
                    raise ValueError(f"missing parent key: {group}/{persona}/{stimulus}")
                if row["score"] is not None:
                    score = float(row["score"])
                    matrix[person_index, task_index] = 100.0 * max(score, 1.0 - score)
        matrices[group] = matrix
    return matrices, stimuli


def person_weighted_mean(matrix: np.ndarray) -> float:
    with np.errstate(invalid="ignore"):
        per_person = np.nanmean(matrix, axis=1)
    if not np.isfinite(per_person).all():
        raise ValueError("an identity has no valid matched dilemmas")
    return float(np.mean(per_person))


def interaction(matrices: dict[str, np.ndarray]) -> float:
    center = np.nanmean(matrices["genpop"], axis=0)
    if not np.isfinite(center).all():
        raise ValueError("a dilemma has no valid general-population center")
    means = {
        group: person_weighted_mean(matrices[group] - center[None, :])
        for group in GROUPS
    }
    return float(
        means["famous_ai"]
        - means["unknown_ai"]
        - means["famous_nonai"]
        + means["genpop"]
    )


def task_order(stimuli: list[str], salt: str) -> list[str]:
    return sorted(
        stimuli,
        key=lambda stimulus: hashlib.sha256(f"{salt}|{stimulus}".encode()).hexdigest(),
    )


def paired_noise(
    v6: dict[str, np.ndarray], v7: dict[str, np.ndarray], task_indices: np.ndarray
) -> dict[str, np.ndarray]:
    result = {
        group: (v7[group] - v6[group])[:, task_indices].copy() for group in GROUPS
    }
    # Remove observed cross-run group shifts. The simulation is about sampling noise
    # under a null context effect, not the realized V6-to-V7 drift.
    for group in GROUPS:
        result[group] -= person_weighted_mean(result[group])
    # Small differences in missing cells can leave a residual after dilemma centering.
    # Remove that residual from F so the null interaction is exactly zero.
    result["famous_ai"] -= interaction(result)
    if abs(interaction(result)) > 1e-10:
        raise AssertionError("failed to center the empirical null interaction")
    return result


def crossed_bootstrap_draws(
    residuals: dict[str, np.ndarray], *, reps: int, seed: int
) -> np.ndarray:
    if reps < 100:
        raise ValueError("power calculation needs at least 100 bootstrap draws")
    task_count = residuals["genpop"].shape[1]
    if any(matrix.shape != (70, task_count) for matrix in residuals.values()):
        raise ValueError("power matrices must be 70 identities by a common task count")
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        tasks = rng.integers(0, task_count, size=task_count)
        paired = rng.integers(0, 70, size=70)
        famous_nonai = rng.integers(0, 70, size=70)
        genpop = rng.integers(0, 70, size=70)
        sampled = {
            "famous_ai": residuals["famous_ai"][paired][:, tasks],
            "unknown_ai": residuals["unknown_ai"][paired][:, tasks],
            "famous_nonai": residuals["famous_nonai"][famous_nonai][:, tasks],
            "genpop": residuals["genpop"][genpop][:, tasks],
        }
        draws[rep] = interaction(sampled)
    return draws


def empirical_power(draws: np.ndarray, effect_pp: float) -> float:
    critical = float(np.percentile(draws, 97.5))
    return float(np.mean(draws + effect_pp > critical))


def mde_for_power(draws: np.ndarray, target_power: float) -> float:
    if not 0.0 < target_power < 1.0:
        raise ValueError("target power must lie between zero and one")
    critical = float(np.percentile(draws, 97.5))
    lower_quantile = float(np.percentile(draws, 100.0 * (1.0 - target_power)))
    return critical - lower_quantile


def repetitions_for_power(
    draws: np.ndarray, effect_pp: float, target_power: float, *, maximum: int = 64
) -> int | None:
    critical = float(np.percentile(draws, 97.5))
    for repetitions in range(1, maximum + 1):
        power = float(
            np.mean(draws + effect_pp * math.sqrt(float(repetitions)) > critical)
        )
        if power >= target_power:
            return repetitions
    return None


def build_power_report(root: Path, prereg_path: Path) -> dict[str, Any]:
    prereg = read_prereg(prereg_path)
    locks = validate_parent_locks(root, prereg)
    roster_path = repo_path(root, prereg["inputs"]["roster"]["path"])
    raw_paths = {
        name: repo_path(root, prereg["parent_evidence"][name]["files"]["raw_scores"]["path"])
        for name in ("v6", "v7")
    }
    v6, stimuli = load_folded_matrices(raw_paths["v6"], roster_path)
    v7, stimuli_v7 = load_folded_matrices(raw_paths["v7"], roster_path)
    if stimuli_v7 != stimuli:
        raise ValueError("V6 and V7 dilemma orders differ")
    power_spec = prereg["power"]
    effects = [float(value) for value in power_spec["simulated_effects_pp"]]
    target = float(power_spec["target_power"])
    smallest = float(power_spec["smallest_meaningful_effect_pp"])
    candidate_counts = [int(value) for value in power_spec["candidate_dilemma_counts"]]
    ordered = task_order(stimuli, str(power_spec["dilemma_hash_salt"]))
    stimulus_index = {stimulus: index for index, stimulus in enumerate(stimuli)}
    results: list[dict[str, Any]] = []
    cost_per_row = float(prereg["cost_reference"]["v7_total_cost_usd"]) / float(
        prereg["cost_reference"]["v7_scientific_rows"]
    )
    for offset, count in enumerate(candidate_counts):
        selected = ordered[:count]
        indices = np.asarray([stimulus_index[value] for value in selected], dtype=np.int64)
        residuals = paired_noise(v6, v7, indices)
        draws = crossed_bootstrap_draws(
            residuals,
            reps=int(power_spec["bootstrap_reps"]),
            seed=int(power_spec["bootstrap_seed"]) + offset,
        )
        critical = float(np.percentile(draws, 97.5))
        powers = {f"{effect:.3f}": empirical_power(draws, effect) for effect in effects}
        repetition_rows: dict[str, Any] = {}
        for effect in effects:
            repetitions = repetitions_for_power(draws, effect, target)
            rows_bd = None if repetitions is None else 2 * 4 * 70 * count * repetitions
            repetition_rows[f"{effect:.3f}"] = {
                "equal_repetitions_per_arm": repetitions,
                "b_plus_d_rows": rows_bd,
                "estimated_b_plus_d_cost_usd": (
                    None if rows_bd is None else rows_bd * cost_per_row
                ),
            }
        results.append(
            {
                "dilemma_count": count,
                "selected_stimuli": selected,
                "matched_parent_cells": {
                    group: int(np.isfinite(residuals[group]).sum()) for group in GROUPS
                },
                "null_point_pp": interaction(residuals),
                "null_standard_deviation_pp": float(np.std(draws, ddof=1)),
                "null_ci95_pp": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
                "positive_critical_value_pp": critical,
                "power_by_effect": powers,
                "mde_for_80pct_power_pp": mde_for_power(draws, target),
                "single_replicate_rows": {
                    "b_plus_d": 2 * 4 * 70 * count,
                    "b_plus_n_plus_d": 3 * 4 * 70 * count,
                },
                "single_replicate_estimated_cost_usd": {
                    "b_plus_d": 2 * 4 * 70 * count * cost_per_row,
                    "b_plus_n_plus_d": 3 * 4 * 70 * count * cost_per_row,
                },
                "repetitions_needed": repetition_rows,
            }
        )
    selected_design = next(
        (
            row
            for row in results
            if row["power_by_effect"][f"{smallest:.3f}"] >= target
        ),
        None,
    )
    decision = (
        "power_gate_passed_design_selected"
        if selected_design is not None
        else "stop_before_scientific_calls_insufficient_power"
    )
    report = {
        "schema_version": "glm53_v22_power_report_v1",
        "project_id": prereg["project_id"],
        "method": power_spec["method"],
        "interpretation": (
            "The V6-to-V7 matched cell differences estimate the noise of two independent "
            "condition calls. Group-level cross-run shifts are removed before a crossed "
            "bootstrap resamples shared dilemmas, paired F/U identities, and independent "
            "FN/G identities. Effects are added to the target F/U component."
        ),
        "parent_hashes": locks,
        "parent_interactions_pp": {
            "v6": interaction(v6),
            "v7": interaction(v7),
            "v7_minus_v6": interaction(v7) - interaction(v6),
        },
        "bootstrap_reps": int(power_spec["bootstrap_reps"]),
        "bootstrap_seed": int(power_spec["bootstrap_seed"]),
        "target_power": target,
        "smallest_meaningful_effect_pp": smallest,
        "candidate_results": results,
        "selected_dilemma_count": (
            None if selected_design is None else selected_design["dilemma_count"]
        ),
        "decision": decision,
        "authorization": {"fresh_subject_or_judge_calls": selected_design is not None},
        "prereg_sha256": sha256_file(prereg_path),
    }
    return report

