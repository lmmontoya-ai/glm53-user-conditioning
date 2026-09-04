"""Independent recomputation for V11 downstream results.

This module intentionally does not import ``v11.downstream`` or any V8
scientific analysis or decision module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

GROUPS = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_source_binding(
    binding: dict[str, str],
    *,
    source_decision_path: Path,
    source_root: Path,
    source_feature_root: Path,
    downstream_manifest_path: Path,
    downstream_preflight_path: Path,
) -> dict[str, str]:
    expected_paths = {
        "downstream_manifest": downstream_manifest_path,
        "downstream_preflight": downstream_preflight_path,
        "source_decision": source_decision_path,
        "source_readout_lock": source_root / "source_readout_lock.json",
        "source_readout_arrays": source_root / "source_readout_arrays.npz",
        "source_feature_manifest": source_feature_root / "feature_manifest.json",
    }
    if set(binding) != set(expected_paths) | {"paid_process_nonce"}:
        raise ValueError("independent downstream source-binding fields differ")
    observed = {name: _sha256(path) for name, path in expected_paths.items()}
    if any(binding[name] != digest for name, digest in observed.items()):
        raise ValueError("independent downstream source-binding hash differs")
    feature_manifest = json.loads(
        (source_feature_root / "feature_manifest.json").read_text(encoding="utf-8")
    )
    nonce = str(feature_manifest["source_hashes"]["paid_process_nonce"])
    if len(nonce) != 64 or binding["paid_process_nonce"] != nonce:
        raise ValueError("independent downstream paid-process nonce differs")
    source_decision = json.loads(source_decision_path.read_text(encoding="utf-8"))
    inputs = source_decision.get("inputs", {})
    if (
        inputs.get("readout_lock") != observed["source_readout_lock"]
        or inputs.get("feature_manifest") != observed["source_feature_manifest"]
    ):
        raise ValueError("source decision does not bind the verified readout and features")
    return observed | {"paid_process_nonce": nonce}


def _interaction(means: dict[str, float]) -> float:
    return means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]


def _person_mean(matrix: np.ndarray, *, require_all: bool) -> float:
    counts = np.sum(np.isfinite(matrix), axis=1)
    eligible = counts > 0
    if require_all and not eligible.all():
        raise ValueError("an identity lacks all downstream observations")
    if not eligible.any():
        raise ValueError("no identity has downstream observations")
    return float(np.mean(np.nansum(matrix[eligible], axis=1) / counts[eligible]))


def _compact_proxy(row: dict[str, Any], label_ids: list[int]) -> dict[str, Any]:
    logits = np.asarray(row["allowed_logits"], dtype=np.float64)
    values = np.asarray(row["codebook_values"], dtype=np.float64)
    shifted = logits - logits.max()
    probability = np.exp(shifted) / np.exp(shifted).sum()
    log_allowed = float(logits.max() + math.log(float(np.exp(shifted).sum())))
    folded = np.maximum(values, 100.0 - values)
    return {
        "expected_folded_confidence": float(probability @ folded),
        "allowed_mass": float(math.exp(log_allowed - float(row["full_logsumexp"]))),
        "argmax_label_position": int(np.argmax(logits)),
        "full_vocab_argmax_allowed": int(row["full_argmax_token_id"]) in set(label_ids),
    }


def _matrices(
    rows: list[dict[str, Any]],
    value: str,
    task: str,
    group_map: dict[str, str],
    *,
    identity_key: str = "pair_index",
    identity_count: int = 70,
) -> dict[str, np.ndarray]:
    tasks = sorted({str(row[task]) for row in rows})
    result: dict[str, np.ndarray] = {}
    for source, target in group_map.items():
        selected = [row for row in rows if row["group"] == source]
        observed = {int(row[identity_key]) for row in selected}
        if not observed <= set(range(identity_count)):
            raise ValueError("downstream identity index is out of range")
        lookup = {
            (int(row[identity_key]), str(row[task])): float(row[value]) for row in selected
        }
        if len(lookup) != len(selected):
            raise ValueError("downstream verification found duplicate identity/task keys")
        result[target] = np.asarray(
            [
                [lookup.get((identity, item), np.nan) for item in tasks]
                for identity in range(identity_count)
            ],
            dtype=np.float64,
        )
    return result


def _bootstrap(matrices: dict[str, np.ndarray], *, reps: int, seed: int) -> np.ndarray:
    if matrices["famous_ai"].shape != matrices["unknown_ai"].shape:
        raise ValueError("F/U bootstrap matrices are not aligned")
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    n_pairs, n_tasks = matrices["famous_ai"].shape
    for rep in range(reps):
        pair = rng.integers(n_pairs, size=n_pairs)
        task = rng.integers(n_tasks, size=n_tasks)
        fn_count = matrices["famous_nonai"].shape[0]
        gp_count = matrices["genpop"].shape[0]
        fn = rng.integers(fn_count, size=fn_count)
        gp = rng.integers(gp_count, size=gp_count)
        sampled = {
            "famous_ai": matrices["famous_ai"][pair][:, task],
            "unknown_ai": matrices["unknown_ai"][pair][:, task],
            "famous_nonai": matrices["famous_nonai"][fn][:, task],
            "genpop": matrices["genpop"][gp][:, task],
        }
        means = _centered_means(sampled, require_all=True)
        draws[rep] = _interaction(means)
    return draws


def _centered_means(
    matrices: dict[str, np.ndarray], *, require_all: bool
) -> dict[str, float]:
    center = np.nanmean(matrices["genpop"], axis=0)
    if not np.isfinite(center).all():
        raise ValueError("a downstream task lacks a GenPop center")
    return {
        group: _person_mean(matrix - center[None, :], require_all=require_all)
        for group, matrix in matrices.items()
    }


def verify_proxy(
    *,
    raw_scores_path: Path,
    analysis_path: Path,
    manifest: dict[str, Any],
    label_ids: list[int],
    source_binding: dict[str, str],
    source_decision_path: Path,
    source_root: Path,
    source_feature_root: Path,
    downstream_manifest_path: Path,
    downstream_preflight_path: Path,
) -> dict[str, Any]:
    verified_binding = _verify_source_binding(
        source_binding,
        source_decision_path=source_decision_path,
        source_root=source_root,
        source_feature_root=source_feature_root,
        downstream_manifest_path=downstream_manifest_path,
        downstream_preflight_path=downstream_preflight_path,
    )
    rows = _read_jsonl(raw_scores_path)
    primary = json.loads(analysis_path.read_text(encoding="utf-8"))
    max_derived_error = 0.0
    for row in rows:
        derived = _compact_proxy(row, label_ids)
        max_derived_error = max(
            max_derived_error,
            abs(derived["expected_folded_confidence"] - row["expected_folded_confidence"]),
            abs(derived["allowed_mass"] - row["allowed_mass"]),
        )
        if derived["argmax_label_position"] != row["argmax_label_position"]:
            raise ValueError("independent proxy argmax differs")
        if derived["full_vocab_argmax_allowed"] != row["full_vocab_argmax_allowed"]:
            raise ValueError("independent full-vocabulary argmax classification differs")

    config = manifest["local_proxy"]
    matrices = _matrices(
        rows,
        "expected_folded_confidence",
        "stimulus_id",
        {g: g for g in GROUPS},
        identity_key="analysis_index",
        identity_count=int(config["identities_per_group"]),
    )
    means = _centered_means(matrices, require_all=True)
    point = _interaction(means)
    draws = _bootstrap(
        matrices,
        reps=int(config["bootstrap_reps"]),
        seed=int(config["bootstrap_seed"]) + 1,
    )
    ci90 = np.percentile(draws, [5, 95]).tolist()
    codebooks: dict[str, float] = {}
    for codebook in sorted({str(row["codebook_id"]) for row in rows}):
        subset = [row for row in rows if str(row["codebook_id"]) == codebook]
        values = _matrices(
            subset,
            "expected_folded_confidence",
            "stimulus_id",
            {group: group for group in GROUPS},
            identity_key="analysis_index",
            identity_count=int(config["identities_per_group"]),
        )
        codebooks[codebook] = _interaction(_centered_means(values, require_all=False))
    allowed = np.asarray([float(row["allowed_mass"]) for row in rows])
    matched_rows = [row for row in rows if row.get("original_folded_confidence") is not None]
    matched_matrices = _matrices(
        matched_rows,
        "expected_folded_confidence",
        "stimulus_id",
        {group: group for group in GROUPS},
        identity_key="analysis_index",
        identity_count=int(config["identities_per_group"]),
    )
    matched_means = _centered_means(matched_matrices, require_all=True)
    matched_point = _interaction(matched_means)
    counts = {group: sum(row["group"] == group for row in rows) for group in GROUPS}
    rates = [
        counts[group] / int(config["expected_pre_missing_rows_per_group"])
        for group in GROUPS
    ]
    retained = abs(matched_point) / abs(float(config["parent_interaction_pp"]))
    f_u = means["famous_ai"] - means["unknown_ai"]
    fn_g = means["famous_nonai"] - means["genpop"]
    gate_checks = {
        "negative": point < 0,
        "codebooks": all(value < 0 for value in codebooks.values()),
        "api_matched_negative": matched_point < 0,
        "magnitude_or_ci": retained >= 0.40 or ci90[1] < 0,
        "components": f_u <= 0 and fn_g >= 0,
        "mass_median": float(np.median(allowed)) >= 0.80,
        "mass_p05": float(np.percentile(allowed, 5)) >= 0.50,
        "argmax": float(np.mean([row["full_vocab_argmax_allowed"] for row in rows])) >= 0.95,
        "codebook_artifact": max(codebooks.values()) < 0,
        "missingness": max(rates) - min(rates) <= 0.005,
    }
    scientific_pass = all(gate_checks.values())
    checks = {
        "derived_values": max_derived_error <= 1e-10,
        "point": abs(point - float(primary["interaction_pp"])) <= 1e-10,
        "api_matched_point": abs(
            matched_point - float(primary["api_matched_interaction_pp"])
        )
        <= 1e-10,
        "api_matched_n": len(matched_rows) == int(primary["parent_confidence_comparison_n"]),
        "key_hashes": (
            _canonical_sha256(
                sorted(
                    (row["group"], row["persona_key"], row["stimulus_id"])
                    for row in rows
                )
            )
            == primary["local_reconstructable_key_sha256"]
            and _canonical_sha256(
                sorted(
                    (row["group"], row["persona_key"], row["stimulus_id"])
                    for row in matched_rows
                )
            )
            == primary["api_matched_key_sha256"]
        ),
        "ci": max(abs(ci90[index] - primary["ci90_pp"][index]) for index in (0, 1)) <= 0.05,
        "gate_checks": gate_checks == primary["checks"],
        "classification": scientific_pass is bool(primary["passed"]),
    }
    return {
        "schema_version": "glm53_v11_proxy_independent_verification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "interaction_pp": point,
        "api_matched_interaction_pp": matched_point,
        "ci90_pp": ci90,
        "max_derived_error": max_derived_error,
        "scientific_gate_would_pass": scientific_pass,
        "inputs": {
            "raw_scores": _sha256(raw_scores_path),
            "primary_analysis": _sha256(analysis_path),
            "source_binding": _canonical_sha256(verified_binding),
        },
    }


def _load_probe(source_root: Path, feature_root: Path) -> dict[str, Any]:
    lock = json.loads((source_root / "source_readout_lock.json").read_text(encoding="utf-8"))
    arrays_path = source_root / "source_readout_arrays.npz"
    if _sha256(arrays_path) != lock["arrays_sha256"]:
        raise ValueError("independent verifier found changed source readout arrays")
    with np.load(arrays_path) as arrays:
        probe = {
            "mean": arrays["logistic_mean"].astype(np.float64),
            "scale": arrays["logistic_scale"].astype(np.float64),
            "weight": arrays["logistic_weight"].astype(np.float64),
            "bias": float(lock["logistic"]["bias"]),
        }
    manifest = json.loads((feature_root / "feature_manifest.json").read_text(encoding="utf-8"))
    record = manifest["partitions"]["development"]
    features_path = feature_root / record["features"]
    metadata_path = feature_root / record["metadata"]
    if _sha256(features_path) != record["features_sha256"] or _sha256(metadata_path) != record[
        "metadata_sha256"
    ]:
        raise ValueError("independent verifier found changed source development features")
    with np.load(features_path) as archive:
        source = archive["shared_task_suffix_mean"].astype(np.float64)
    metadata = _read_jsonl(metadata_path)
    layer = int(lock["selected_layer"])
    train = np.asarray([row["split"] == "train" for row in metadata])
    scores = ((source[train, layer] - probe["mean"]) / probe["scale"]) @ probe["weight"] + probe[
        "bias"
    ]
    probe["sd"] = float(np.std(scores, ddof=1))
    return probe


def verify_recruitment(
    *,
    feature_path: Path,
    metadata_path: Path,
    analysis_path: Path,
    source_root: Path,
    source_feature_root: Path,
    schedule_path: Path,
    manifest: dict[str, Any],
    source_binding: dict[str, str],
    source_decision_path: Path,
    downstream_manifest_path: Path,
    downstream_preflight_path: Path,
) -> dict[str, Any]:
    verified_binding = _verify_source_binding(
        source_binding,
        source_decision_path=source_decision_path,
        source_root=source_root,
        source_feature_root=source_feature_root,
        downstream_manifest_path=downstream_manifest_path,
        downstream_preflight_path=downstream_preflight_path,
    )
    primary = json.loads(analysis_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(metadata_path)
    probe = _load_probe(source_root, source_feature_root)
    with np.load(feature_path) as archive:
        features = archive["neutral_task_mean"].astype(np.float64)
    score = (
        ((features - probe["mean"]) / probe["scale"]) @ probe["weight"] + probe["bias"]
    ) / probe["sd"]
    scored = [row | {"score": float(value)} for row, value in zip(rows, score, strict=True)]
    matrices = _matrices(
        scored,
        "score",
        "template_id",
        {"F": "famous_ai", "U": "unknown_ai", "FN": "famous_nonai", "G": "genpop"},
    )
    means = _centered_means(matrices, require_all=True)
    point = _interaction(means)
    config = manifest["recruitment"]
    draws = _bootstrap(
        matrices,
        reps=int(config["bootstrap_reps"]),
        seed=int(config["bootstrap_seed"]) + 1,
    )
    interval = np.percentile(draws, [2.5, 97.5]).tolist()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    target_sets: dict[str, list[int]] = {"enriched": [], "prospective": []}
    for pair in schedule["pairs"]:
        target_sets[str(pair["set"])].append(int(pair["twin_index"]))
    subsets = {
        name: _person_mean(
            matrices["famous_ai"][indices] - matrices["unknown_ai"][indices],
            require_all=True,
        )
        for name, indices in target_sets.items()
    }
    loo = []
    for index in range(70):
        keep = np.arange(70) != index
        loo.append(
            _person_mean(matrices["famous_ai"][keep], require_all=True)
            - _person_mean(matrices["unknown_ai"][keep], require_all=True)
            - _person_mean(matrices["famous_nonai"], require_all=True)
            + _person_mean(matrices["genpop"], require_all=True)
        )
    f_u = means["famous_ai"] - means["unknown_ai"]
    fn_g = means["famous_nonai"] - means["genpop"]
    gate_checks = {
        "point": point >= 0.20,
        "interval": interval[0] > 0,
        "f_u_positive": f_u > 0,
        "fame_control": fn_g <= 0,
        "subset_signs": subsets["enriched"] > 0 and subsets["prospective"] > 0,
        "loo": min(loo) > 0,
    }
    scientific_pass = all(gate_checks.values())
    checks = {
        "point": abs(point - float(primary["interaction_sd"])) <= 1e-10,
        "source_sd": abs(probe["sd"] - float(primary["source_train_sd"])) <= 1e-10,
        "ci": max(abs(interval[index] - primary["ci95_sd"][index]) for index in (0, 1)) <= 0.05,
        "gate_checks": gate_checks == primary["checks"],
        "classification": scientific_pass is bool(primary["passed"]),
    }
    return {
        "schema_version": "glm53_v11_recruitment_independent_verification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "interaction_sd": point,
        "ci95_sd": interval,
        "scientific_gate_would_pass": scientific_pass,
        "inputs": {
            "features": _sha256(feature_path),
            "metadata": _sha256(metadata_path),
            "primary_analysis": _sha256(analysis_path),
            "source_readout_lock": _sha256(source_root / "source_readout_lock.json"),
            "source_readout_arrays": _sha256(source_root / "source_readout_arrays.npz"),
            "source_feature_manifest": _sha256(source_feature_root / "feature_manifest.json"),
            "schedule": _sha256(schedule_path),
            "source_binding": _canonical_sha256(verified_binding),
        },
    }


__all__ = ["verify_proxy", "verify_recruitment"]
