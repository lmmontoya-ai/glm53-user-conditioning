"""Reproduce Transluce's folded-confidence population aggregation over v6 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def newest_success(log_dir: Path) -> Path:
    from inspect_ai.log import read_eval_log

    paths = sorted(log_dir.glob("*.eval"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise ValueError(f"missing eval log: {log_dir}")
    path = paths[-1]
    if str(read_eval_log(path, header_only=True).status) != "success":
        raise ValueError(f"newest eval is not successful: {path}")
    return path


def extract_rows(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from inspect_ai.log import read_eval_log_sample_summaries

    manifest = json.loads((run_root / "schedule_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for spec in manifest["shards"]:
        shard_id = str(spec["shard_id"])
        group = str(spec["group"])
        path = newest_success(run_root / "eval_logs" / shard_id)
        shard_total = 0
        shard_valid = 0
        shard_errors = 0
        for sample in read_eval_log_sample_summaries(path):
            shard_total += 1
            metadata = sample.metadata or {}
            persona = str(metadata.get("persona") or "")
            stimulus = str(metadata.get("scenario_id") or "")
            key = (group, persona, stimulus)
            if not persona or not stimulus or key in seen:
                raise ValueError(f"invalid or duplicate sample key: {key}")
            seen.add(key)
            scores = list((sample.scores or {}).values())
            score_value: float | None = None
            score_metadata: dict[str, Any] = {}
            if sample.error is not None:
                shard_errors += 1
            elif scores:
                score_metadata = dict(scores[0].metadata or {})
                try:
                    candidate = float(scores[0].value)
                    if not math.isnan(candidate):
                        score_value = candidate
                        shard_valid += 1
                except (TypeError, ValueError):
                    pass
            rows.append(
                {
                    "shard_id": shard_id,
                    "group": group,
                    "persona": persona,
                    "stimulus": stimulus,
                    "score": score_value,
                    "confidence_p": score_metadata.get("confidence_p"),
                    "binary_answer": score_metadata.get("binary_answer"),
                    "refused_estimate": score_metadata.get("refused_estimate"),
                    "binary_refused": score_metadata.get("binary_refused"),
                    "sample_error": None if sample.error is None else str(sample.error),
                }
            )
        expected = int(spec["expected_rows"])
        if shard_total != expected:
            raise ValueError(f"shard row count differs: {shard_id}: {shard_total} != {expected}")
        shards.append(
            {
                "shard_id": shard_id,
                "eval_log": str(path),
                "eval_sha256": sha256_file(path),
                "expected_rows": expected,
                "total_rows": shard_total,
                "valid_rows": shard_valid,
                "error_rows": shard_errors,
            }
        )
    return rows, shards


def folded(score: float) -> float:
    value = 100.0 * score
    return max(value, 100.0 - value)


def matrices(
    rows: list[dict[str, Any]], roster: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, np.ndarray], list[str]]:
    stimuli = sorted({str(row["stimulus"]) for row in rows})
    if len(stimuli) != 100:
        raise ValueError("analysis requires exactly 100 dilemmas")
    stimulus_index = {value: index for index, value in enumerate(stimuli)}
    result: dict[str, np.ndarray] = {}
    lookup = {(row["group"], row["persona"], row["stimulus"]): row for row in rows}
    for group in GROUPS:
        personas = [str(row["key"]) for row in roster[group]]
        matrix = np.full((len(personas), len(stimuli)), np.nan, dtype=np.float64)
        for person_index, persona in enumerate(personas):
            for stimulus in stimuli:
                row = lookup[(group, persona, stimulus)]
                if row["score"] is not None:
                    matrix[person_index, stimulus_index[stimulus]] = folded(float(row["score"]))
        result[group] = matrix
    return result, stimuli


def reference_person_deltas(
    group_matrices: dict[str, np.ndarray], roster: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    genpop_center = np.nanmean(group_matrices["genpop"], axis=0)
    people: list[dict[str, Any]] = []
    group_means: dict[str, float] = {}
    for group in GROUPS:
        per_person = np.nanmean(group_matrices[group] - genpop_center[None, :], axis=1)
        for person_index, (roster_row, value) in enumerate(
            zip(roster[group], per_person, strict=True)
        ):
            people.append(
                {
                    "persona": roster_row["key"],
                    "group": group,
                    "mean_delta_pp": float(value),
                    "valid_dilemmas": int(
                        np.isfinite(group_matrices[group][person_index]).sum()
                    ),
                }
            )
        group_means[group] = float(np.nanmean(per_person))
    return people, group_means


def mannwhitney_z(a: list[float], b: list[float]) -> float:
    """Pinned Transluce tie-aware Mann-Whitney normal approximation."""
    n1, n2 = len(a), len(b)
    ranked = sorted([(value, 0) for value in a] + [(value, 1) for value in b])
    ranks: list[float] = [0.0] * len(ranked)
    index = 0
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[index][0]:
            end += 1
        midrank = (index + end) / 2 + 1
        for rank_index in range(index, end + 1):
            ranks[rank_index] = midrank
        index = end + 1
    rank_sum = sum(
        rank for rank, (_value, source) in zip(ranks, ranked, strict=True) if source == 0
    )
    u_value = rank_sum - n1 * (n1 + 1) / 2
    mean = n1 * n2 / 2
    standard_deviation = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return (u_value - mean) / standard_deviation if standard_deviation else 0.0


def exact_group_stats(people: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in people:
        grouped[str(row["group"])].append(float(row["mean_delta_pp"]))
    genpop = sorted(grouped["genpop"])
    result: dict[str, dict[str, float | int | None]] = {}
    for group in GROUPS:
        values = sorted(grouped[group])
        median = float(np.median(values))
        z_value = None if group == "genpop" else mannwhitney_z(values, genpop)
        p_value = None if z_value is None else math.erfc(abs(z_value) / math.sqrt(2))
        result[group] = {
            "n": len(values),
            "median_pp": median,
            "iqr_source_indexed_pp": [
                float(values[len(values) // 4]),
                float(values[3 * len(values) // 4]),
            ],
            "min_pp": float(values[0]),
            "max_pp": float(values[-1]),
            "mann_whitney_z_vs_genpop": z_value,
            "mann_whitney_p_vs_genpop": p_value,
            "bonferroni_28_p_vs_genpop": (
                None if p_value is None else min(1.0, p_value * 28)
            ),
        }
    return result


def bootstrap_group_delta(
    group: np.ndarray,
    genpop: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    tasks = group.shape[1]
    for rep in range(reps):
        task_indices = rng.integers(0, tasks, size=tasks)
        group_indices = rng.integers(0, group.shape[0], size=group.shape[0])
        genpop_indices = rng.integers(0, genpop.shape[0], size=genpop.shape[0])
        sampled_center = np.nanmean(genpop[genpop_indices][:, task_indices], axis=0)
        person_means = np.nanmean(
            group[group_indices][:, task_indices] - sampled_center[None, :], axis=1
        )
        values[rep] = np.nanmean(person_means)
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def bootstrap_paired_contrast(
    left: np.ndarray,
    right: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[float, list[float]]:
    if left.shape != right.shape:
        raise ValueError("paired contrast matrices differ in shape")
    difference = left - right
    point = float(np.nanmean(np.nanmean(difference, axis=1)))
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        pair_indices = rng.integers(0, difference.shape[0], size=difference.shape[0])
        task_indices = rng.integers(0, difference.shape[1], size=difference.shape[1])
        values[rep] = np.nanmean(
            np.nanmean(difference[pair_indices][:, task_indices], axis=1)
        )
    return point, [float(value) for value in np.percentile(values, [2.5, 97.5])]


def released_glm52(
    source_root: Path,
) -> tuple[dict[str, float], float, dict[str, dict[str, float | int | None]]]:
    payload = json.loads(
        (source_root / "cache/aggregates/s2glm52_deltas_conf.json").read_text(encoding="utf-8")
    )
    by_group: dict[str, list[float]] = defaultdict(list)
    for _persona, value in payload.items():
        group, mean, _deltas = value
        by_group[str(group)].append(float(mean))
    means = {group: float(np.mean(values)) for group, values in by_group.items()}
    roster = json.loads((source_root / "core/personas2.json").read_text(encoding="utf-8"))
    paired = []
    for index in range(70):
        famous = roster["famous_ai"][index]["key"]
        unknown = roster["unknown_ai"][index]["key"]
        if famous in payload and unknown in payload:
            paired.append(float(payload[famous][1]) - float(payload[unknown][1]))
    people = [
        {"persona": persona, "group": value[0], "mean_delta_pp": float(value[1])}
        for persona, value in payload.items()
    ]
    return means, float(np.mean(paired)), exact_group_stats(people)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--analysis-amendment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prereg = yaml.safe_load(args.prereg.read_text(encoding="utf-8"))
    amendment = yaml.safe_load(args.analysis_amendment.read_text(encoding="utf-8"))
    if amendment.get("schema_version") != "glm53_user_eval_analysis_amendment_v6_2":
        raise ValueError("analysis requires the v6.2 parity amendment")
    if amendment.get("parent_prereg_sha256") != sha256_file(args.prereg):
        raise ValueError("analysis amendment does not match the parent preregistration")
    reps = int(prereg["analysis"]["bootstrap_reps"])
    seed = int(prereg["analysis"]["bootstrap_seed"])
    rows, shards = extract_rows(args.run_root)
    expected_rows = int(prereg["population"]["expected_scientific_rows"])
    if len(rows) != expected_rows:
        raise ValueError(f"scientific row count differs: {len(rows)} != {expected_rows}")
    roster = json.loads((args.source_root / "core/personas2.json").read_text(encoding="utf-8"))
    group_matrices, stimuli = matrices(rows, roster)
    people, group_means = reference_person_deltas(group_matrices, roster)
    source_group_stats = exact_group_stats(people)
    intervals = {
        group: bootstrap_group_delta(
            group_matrices[group],
            group_matrices["genpop"],
            reps=reps,
            seed=seed + index,
        )
        for index, group in enumerate(GROUPS)
    }
    paired_effect, paired_ci = bootstrap_paired_contrast(
        group_matrices["famous_ai"],
        group_matrices["unknown_ai"],
        reps=reps,
        seed=seed + 100,
    )
    released_means, released_pair, released_group_stats = released_glm52(args.source_root)
    counts: dict[str, dict[str, float | int]] = {}
    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        nonanon = [row for row in group_rows if row["persona"] != "anon"]
        valid = sum(row["score"] is not None for row in nonanon)
        counts[group] = {
            "expected_nonanonymous_rows": len(nonanon),
            "valid_nonanonymous_rows": valid,
            "completion_rate": valid / len(nonanon),
        }
    missing_rates = [1.0 - float(counts[group]["completion_rate"]) for group in GROUPS]
    completion_rate = sum(row["score"] is not None for row in rows) / len(rows)
    missingness_spread = max(missing_rates) - min(missing_rates)
    primary_ci = intervals["famous_ai"]
    primary_source_stats = source_group_stats["famous_ai"]
    replication_positive = (
        completion_rate >= 0.95
        and missingness_spread <= 0.02
        and float(primary_source_stats["median_pp"]) < 0
        and float(primary_source_stats["bonferroni_28_p_vs_genpop"]) < 0.05
    )
    clean_negative = (
        completion_rate >= 0.95
        and not (primary_ci[0] <= released_means["famous_ai"] <= primary_ci[1])
        and not (paired_ci[0] <= released_pair <= paired_ci[1])
    )
    decision = (
        "replication_positive"
        if replication_positive
        else "clean_negative"
        if clean_negative
        else "ambiguous_complete_replication"
    )
    analysis = {
        "schema_version": "glm53_transluce_exact_analysis_v1",
        "row_count": len(rows),
        "valid_score_count": sum(row["score"] is not None for row in rows),
        "completion_rate": completion_rate,
        "condition_missingness_spread": missingness_spread,
        "dilemma_count": len(stimuli),
        "group_counts": counts,
        "group_mean_deltas_pp": group_means,
        "group_ci95_pp": intervals,
        "source_exact_group_stats": source_group_stats,
        "same_index_famous_minus_unknown_pp": paired_effect,
        "same_index_famous_minus_unknown_ci95_pp": paired_ci,
        "released_glm52_group_mean_deltas_pp": released_means,
        "released_glm52_same_index_famous_minus_unknown_pp": released_pair,
        "released_glm52_source_exact_group_stats": released_group_stats,
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }
    decision_payload = {
        "schema_version": "glm53_transluce_exact_decision_v1",
        "project_id": prereg["project_id"],
        "prereg_sha256": sha256_file(args.prereg),
        "analysis_amendment_sha256": sha256_file(args.analysis_amendment),
        "analysis": analysis,
        "checks": {
            "replication_positive": replication_positive,
            "clean_negative": clean_negative,
            "completion_rate_at_least_95pct": completion_rate >= 0.95,
            "missingness_spread_at_most_2pct": missingness_spread <= 0.02,
            "famous_ai_median_delta_negative": float(primary_source_stats["median_pp"]) < 0,
            "famous_ai_source_adjusted_p_below_0_05": (
                float(primary_source_stats["bonferroni_28_p_vs_genpop"]) < 0.05
            ),
        },
        "decision": decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_root / "raw_scores.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    atomic_json(args.output_root / "shard_manifest.json", shards)
    atomic_json(args.output_root / "person_deltas.json", people)
    atomic_json(args.output_root / "analysis.json", analysis)
    atomic_json(args.output_root / "decision.json", decision_payload)
    print(json.dumps(decision_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
