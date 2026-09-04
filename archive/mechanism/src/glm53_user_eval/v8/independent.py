"""Independent v8 recomputation; deliberately imports no primary analysis code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interaction(rows: list[dict[str, Any]], value: str) -> float:
    means = {
        group: float(np.mean([row[value] for row in rows if row["group"] == group]))
        for group in ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
    }
    return means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]


def _matrix(rows: list[dict[str, Any]], group: str, value: str) -> np.ndarray:
    selected = [row for row in rows if row["group"] == group]
    identities = sorted(
        {row["persona_key"] for row in selected},
        key=lambda key: next(
            int(row["pair_index"]) for row in selected if row["persona_key"] == key
        ),
    )
    tasks = sorted({row["stimulus_id"] for row in selected})
    lookup = {(row["persona_key"], row["stimulus_id"]): row[value] for row in selected}
    return np.asarray(
        [[lookup.get((identity, task), np.nan) for task in tasks] for identity in identities]
    )


def _bootstrap_delta(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    seed: int,
    reps: int,
) -> tuple[float, list[float]]:
    groups = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
    difference = {
        group: _matrix(candidate_rows, group, "expected_folded_confidence")
        - _matrix(baseline_rows, group, "expected_folded_confidence")
        for group in groups
    }
    point = (
        float(np.nanmean(difference["famous_ai"]))
        - float(np.nanmean(difference["unknown_ai"]))
        - float(np.nanmean(difference["famous_nonai"]))
        + float(np.nanmean(difference["genpop"]))
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    n_pairs, n_tasks = difference["famous_ai"].shape
    for index in range(reps):
        pair = rng.integers(0, n_pairs, n_pairs)
        task = rng.integers(0, n_tasks, n_tasks)
        fn = rng.integers(
            0, difference["famous_nonai"].shape[0], difference["famous_nonai"].shape[0]
        )
        gp = rng.integers(0, difference["genpop"].shape[0], difference["genpop"].shape[0])
        draws[index] = (
            float(np.nanmean(difference["famous_ai"][pair][:, task]))
            - float(np.nanmean(difference["unknown_ai"][pair][:, task]))
            - float(np.nanmean(difference["famous_nonai"][fn][:, task]))
            + float(np.nanmean(difference["genpop"][gp][:, task]))
        )
    return point, np.percentile(draws, [2.5, 97.5]).tolist()


def _bootstrap_interaction(
    rows: list[dict[str, Any]], value: str, *, seed: int, reps: int
) -> tuple[float, list[float]]:
    matrices = {
        group: _matrix(rows, group, value)
        for group in ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
    }
    point = (
        float(np.nanmean(matrices["famous_ai"]))
        - float(np.nanmean(matrices["unknown_ai"]))
        - float(np.nanmean(matrices["famous_nonai"]))
        + float(np.nanmean(matrices["genpop"]))
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    n_pairs, n_tasks = matrices["famous_ai"].shape
    for index in range(reps):
        pair = rng.integers(0, n_pairs, n_pairs)
        task = rng.integers(0, n_tasks, n_tasks)
        fn = rng.integers(0, matrices["famous_nonai"].shape[0], matrices["famous_nonai"].shape[0])
        gp = rng.integers(0, matrices["genpop"].shape[0], matrices["genpop"].shape[0])
        draws[index] = (
            float(np.nanmean(matrices["famous_ai"][pair][:, task]))
            - float(np.nanmean(matrices["unknown_ai"][pair][:, task]))
            - float(np.nanmean(matrices["famous_nonai"][fn][:, task]))
            + float(np.nanmean(matrices["genpop"][gp][:, task]))
        )
    return point, np.percentile(draws, [2.5, 97.5]).tolist()


def _verify_decision_lineage(artifact_root: Path) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for gate in ("m2", "m3", "m4", "m5", "m6", "m7"):
        path = artifact_root / f"decisions/{gate}_decision.json"
        if not path.is_file():
            continue
        decision = json.loads(path.read_text(encoding="utf-8"))
        output[f"{gate}_classification"] = decision.get("passed") is (
            bool(decision.get("checks")) and all(decision.get("checks", {}).values())
        )
        inputs = decision.get("inputs", {})
        output[f"{gate}_inputs_present"] = bool(inputs)
        for name, record in inputs.items():
            source = Path(record["path"])
            output[f"{gate}_input_{name}"] = (
                source.is_file()
                and source.stat().st_size == int(record["size_bytes"])
                and _sha256(source) == record["sha256"]
            )
    return output


def verify(artifact_root: Path, *, seed: int = 20260903, reps: int = 20_000) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    estimates: dict[str, Any] = {}
    m3_path = artifact_root / "m3/eval_direction_report.json"
    if m3_path.is_file():
        report = json.loads(m3_path.read_text(encoding="utf-8"))
        layer = int(report["selected_layer"])
        features = np.load(artifact_root / "m3/eval_features.npz")[
            f"l{layer}__prompt_final"
        ].astype(np.float64)
        metadata = _rows(artifact_root / "m3/eval_features.jsonl")
        vectors = np.load(artifact_root / "m3/eval_directions.npz")
        score = (
            (features - vectors[f"l{layer}__scaler_mean"]) / vectors[f"l{layer}__scaler_scale"]
        ) @ vectors[f"l{layer}__coefficient_standardized"] + float(
            report["layers"][str(layer)]["intercept"]
        )
        labels = np.asarray([row["label"] for row in metadata])
        splits = np.asarray([row["split"] for row in metadata])
        test_auc = float(roc_auc_score(labels[splits == "test"], score[splits == "test"]))
        realism_auc = float(
            roc_auc_score(labels[splits == "holdout_realism"], score[splits == "holdout_realism"])
        )
        fpr, tpr, _ = roc_curve(
            labels[splits == "holdout_hardneg"], score[splits == "holdout_hardneg"]
        )
        hard_fpr = float(fpr[tpr >= 0.8].min())
        estimates["m3"] = {
            "test_auroc": test_auc,
            "realism_auroc": realism_auc,
            "hard_fpr": hard_fpr,
        }
        checks["m3_test"] = abs(test_auc - report["core_test_auroc"]) <= 1e-10
        checks["m3_realism"] = abs(realism_auc - report["realism_auroc"]) <= 1e-10
        checks["m3_hard"] = abs(hard_fpr - report["hard_negative_fpr_at_80_tpr"]) <= 1e-10
    m4_path = artifact_root / "m4/recruitment_report.json"
    if m4_path.is_file():
        report = json.loads(m4_path.read_text(encoding="utf-8"))
        direction_report = json.loads(m3_path.read_text(encoding="utf-8"))
        layer = int(direction_report["selected_layer"])
        features = np.load(artifact_root / "m4/user_features.npz")[
            f"l{layer}__prompt_final"
        ].astype(np.float64)
        metadata = _rows(artifact_root / "m4/user_features.jsonl")
        vectors = np.load(artifact_root / "m3/eval_directions.npz")
        scores = (
            (features - vectors[f"l{layer}__scaler_mean"])
            / vectors[f"l{layer}__scaler_scale"]
        ) @ vectors[f"l{layer}__coefficient_standardized"] + float(
            direction_report["layers"][str(layer)]["intercept"]
        )
        scores /= float(vectors[f"l{layer}__source_score_sd"][0])
        group_map = {"F": "famous_ai", "U": "unknown_ai", "FN": "famous_nonai", "G": "genpop"}
        scored = [
            row | {"group": group_map[row["group"]], "eval_score_sd": float(value)}
            for row, value in zip(metadata, scores, strict=True)
        ]
        point, interval = _bootstrap_interaction(
            scored, "eval_score_sd", seed=seed, reps=reps
        )
        estimates["m4_interaction_sd"] = point
        estimates["m4_ci95_sd"] = interval
        checks["m4_point"] = abs(point - report["interaction_sd"]) <= 1e-10
        checks["m4_interval"] = (
            max(abs(a - b) for a, b in zip(interval, report["ci95_sd"], strict=True))
            <= 0.05
        )
    m5_path = artifact_root / "m5/proxy_report.json"
    if m5_path.is_file():
        report = json.loads(m5_path.read_text(encoding="utf-8"))
        rows = _rows(artifact_root / "m5/rows/baseline.jsonl")
        point = _interaction(rows, "expected_folded_confidence")
        estimates["m5_interaction_pp"] = point
        checks["m5_point"] = abs(point - report["interaction_pp"]) <= 1e-10
    m6_path = artifact_root / "m6/pilot_report.json"
    if m6_path.is_file():
        report = json.loads(m6_path.read_text(encoding="utf-8"))
        positive = report["positive_control"]["passed_negative_alphas"]
        baseline_mass = float(report["baseline"]["allowed_mass_median"])
        eligible = []
        for key, alpha_report in report["alphas"].items():
            alpha = float(key)
            if (
                alpha < 0
                and positive.get(key) is True
                and float(alpha_report["delta_pp"]) > 0
                and float(alpha_report["allowed_mass_median"]) >= baseline_mass - 0.05
                and float(alpha_report["allowed_mass_p05"]) >= 0.40
            ):
                eligible.append((float(alpha_report["delta_pp"]), -abs(alpha), alpha))
        selected = max(eligible)[2] if eligible else None
        estimates["m6_selected_alpha"] = selected
        checks["m6_alpha_selection"] = selected == report.get("chosen_alpha")
        if selected is not None:
            checks["m6_random_median"] = float(
                report["alphas"][str(selected)]["delta_pp"]
            ) > float(np.median(report["control_deltas_pp"]))
    m7_path = artifact_root / "m7/confirmation_report.json"
    if m7_path.is_file():
        report = json.loads(m7_path.read_text(encoding="utf-8"))
        baseline = _rows(artifact_root / "m7/rows/baseline.jsonl")
        candidate = _rows(artifact_root / "m7/rows/candidate.jsonl")
        point, interval = _bootstrap_delta(baseline, candidate, seed=seed, reps=reps)
        estimates["m7_delta_pp"] = point
        estimates["m7_ci95_pp"] = interval
        checks["m7_point"] = abs(point - report["delta_pp"]) <= 1e-10
        checks["m7_interval"] = (
            max(abs(a - b) for a, b in zip(interval, report["delta_ci95_pp"], strict=True)) <= 0.05
        )
        baseline_point = _interaction(baseline, "expected_folded_confidence")
        control_deltas = []
        for path in sorted((artifact_root / "m7/rows").glob("random_*.jsonl")):
            control_deltas.append(
                _interaction(_rows(path), "expected_folded_confidence") - baseline_point
            )
        rank = 1 + sum(value >= point for value in control_deltas)
        empirical = (1 + sum(value >= point for value in control_deltas)) / (
            1 + len(control_deltas)
        )
        estimates["m7_control_rank"] = rank
        estimates["m7_empirical_p"] = empirical
        checks["m7_control_deltas"] = np.allclose(
            sorted(control_deltas), sorted(report["control_deltas_pp"]), atol=1e-10
        )
        checks["m7_control_rank"] = rank == report["candidate_control_rank"]
        checks["m7_empirical_p"] = abs(empirical - report["empirical_p"]) <= 1e-10
        selection = json.loads(
            (artifact_root / "m6/frozen_selection.json").read_text(encoding="utf-8")
        )
        positive_report_path = next(
            path
            for path in sorted((artifact_root / "m6").glob("*_positive_control_report.json"))
            if _sha256(path) == selection["positive_control_report_sha256"]
        )
        positive_report = json.loads(positive_report_path.read_text(encoding="utf-8"))
        checks["m7_positive_control"] = (
            report["positive_control_passed"]
            is positive_report["passed_negative_alphas"][str(selection["alpha"])]
        )
    m8_path = artifact_root / "m8/hardening_report.json"
    if m8_path.is_file():
        report = json.loads(m8_path.read_text(encoding="utf-8"))
        baseline = _rows(artifact_root / "m8/rows/baseline.jsonl")
        candidate = _rows(artifact_root / "m8/rows/candidate.jsonl")
        point = _interaction(candidate, "expected_folded_confidence") - _interaction(
            baseline, "expected_folded_confidence"
        )
        estimates["m8_delta_pp"] = point
        checks["m8_point"] = abs(point - report["delta_pp"]) <= 1e-10
    checks.update(_verify_decision_lineage(artifact_root))
    return {
        "schema_version": "glm53_v8_independent_verification_v1",
        "seed": seed,
        "bootstrap_reps": reps,
        "checks": checks,
        "estimates": estimates,
        "passed": bool(checks) and all(checks.values()),
    }
