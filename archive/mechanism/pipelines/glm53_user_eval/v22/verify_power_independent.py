"""Independent recomputation of the V22 power gate.

This file intentionally does not import the primary V22 power or decision modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_nonai")


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_matrix(raw: Path, roster: dict[str, Any]) -> dict[str, np.ndarray]:
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    lookup = {
        (row["group"], row["persona"], row["stimulus"]): row for row in records
    }
    output: dict[str, np.ndarray] = {}
    for group in GROUPS:
        values = np.full((70, 100), np.nan, dtype=np.float64)
        for person, roster_row in enumerate(roster[group]):
            for task in range(100):
                record = lookup[(group, roster_row["key"], f"dd_{task:04d}")]
                if record["score"] is not None:
                    score = 100.0 * float(record["score"])
                    values[person, task] = max(score, 100.0 - score)
        output[group] = values
    return output


def identity_mean(values: np.ndarray) -> float:
    return float(np.mean(np.nanmean(values, axis=1)))


def contrast(values: dict[str, np.ndarray]) -> float:
    general_center = np.nanmean(values["genpop"], axis=0)
    centered = {
        group: identity_mean(values[group] - general_center[None, :]) for group in GROUPS
    }
    return float(
        centered["famous_ai"]
        - centered["unknown_ai"]
        - centered["famous_nonai"]
        + centered["genpop"]
    )


def null_draws(
    v6: dict[str, np.ndarray],
    v7: dict[str, np.ndarray],
    task_indices: list[int],
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    residual = {
        group: (v7[group] - v6[group])[:, task_indices].copy() for group in GROUPS
    }
    for group in GROUPS:
        residual[group] -= identity_mean(residual[group])
    residual["famous_ai"] -= contrast(residual)
    if abs(contrast(residual)) > 1e-10:
        raise AssertionError("independent null did not center")
    rng = np.random.default_rng(seed)
    answer = np.empty(draws, dtype=np.float64)
    task_count = len(task_indices)
    for index in range(draws):
        task_sample = rng.integers(task_count, size=task_count)
        pair_sample = rng.integers(70, size=70)
        famous_sample = rng.integers(70, size=70)
        general_sample = rng.integers(70, size=70)
        sample = {
            "famous_ai": residual["famous_ai"][pair_sample][:, task_sample],
            "unknown_ai": residual["unknown_ai"][pair_sample][:, task_sample],
            "famous_nonai": residual["famous_nonai"][famous_sample][:, task_sample],
            "genpop": residual["genpop"][general_sample][:, task_sample],
        }
        answer[index] = contrast(sample)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prereg",
        type=Path,
        default=ROOT
        / "pipelines/glm53_user_eval/v22/configs/prereg_v22_information_substitution.yaml",
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "artifacts/glm53_user_eval/v22/power/power_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/glm53_user_eval/v22/power/verification.json",
    )
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    prereg = yaml.safe_load(args.prereg.read_text(encoding="utf-8"))
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    roster_spec = prereg["inputs"]["roster"]
    roster_path = ROOT / roster_spec["path"]
    if file_hash(roster_path) != roster_spec["sha256"]:
        raise ValueError("independent roster hash mismatch")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    parents = prereg["parent_evidence"]
    for name in ("v6", "v7"):
        item = parents[name]["files"]["raw_scores"]
        if file_hash(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"independent {name} hash mismatch")
    v6 = load_matrix(ROOT / parents["v6"]["files"]["raw_scores"]["path"], roster)
    v7 = load_matrix(ROOT / parents["v7"]["files"]["raw_scores"]["path"], roster)
    effect_values = [float(value) for value in prereg["power"]["simulated_effects_pp"]]
    target = float(prereg["power"]["target_power"])
    smallest_key = f"{float(prereg['power']['smallest_meaningful_effect_pp']):.3f}"
    checked = []
    for offset, primary_row in enumerate(primary["candidate_results"]):
        tasks = [int(value.removeprefix("dd_")) for value in primary_row["selected_stimuli"]]
        values = null_draws(v6, v7, tasks, draws=args.draws, seed=20261903 + offset)
        critical = float(np.percentile(values, 97.5))
        powers = {
            f"{effect:.3f}": float(np.mean(values + effect > critical))
            for effect in effect_values
        }
        mde = critical - float(np.percentile(values, 20.0))
        checked.append(
            {
                "dilemma_count": int(primary_row["dilemma_count"]),
                "power_by_effect": powers,
                "mde_for_80pct_power_pp": mde,
                "maximum_power_difference": max(
                    abs(powers[key] - float(primary_row["power_by_effect"][key]))
                    for key in powers
                ),
                "mde_difference_pp": abs(
                    mde - float(primary_row["mde_for_80pct_power_pp"])
                ),
            }
        )
    independent_pass_candidates = [
        row for row in checked if row["power_by_effect"][smallest_key] >= target
    ]
    independent_decision = (
        "power_gate_passed_design_selected"
        if independent_pass_candidates
        else "stop_before_scientific_calls_insufficient_power"
    )
    checks = {
        "all_power_differences_at_most_0_02": all(
            row["maximum_power_difference"] <= 0.02 for row in checked
        ),
        "all_mde_differences_at_most_0_08pp": all(
            row["mde_difference_pp"] <= 0.08 for row in checked
        ),
        "decision_matches": independent_decision == primary["decision"],
        "independent_seed_differs": 20261903 != int(primary["bootstrap_seed"]),
        "primary_hash_matches": file_hash(args.primary)
        == file_hash(ROOT / "artifacts/glm53_user_eval/v22/power/power_report.json"),
    }
    output = {
        "schema_version": "glm53_v22_independent_power_verification_v1",
        "passed": all(checks.values()),
        "independent_draws": args.draws,
        "independent_seed_base": 20261903,
        "checks": checks,
        "candidate_results": checked,
        "independent_decision": independent_decision,
        "primary_report_sha256": file_hash(args.primary),
        "prereg_sha256": file_hash(args.prereg),
        "normal_approximation_note": (
            "Equal independent repetitions per condition scale the single-call difference "
            f"noise by 1/sqrt(k); sqrt(4)={math.sqrt(4):.1f}."
        ),
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

