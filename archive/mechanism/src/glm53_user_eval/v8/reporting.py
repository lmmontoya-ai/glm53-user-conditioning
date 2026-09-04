"""Post-decision direction geometry and terminal report generation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .artifacts import atomic_json
from .interventions import normalize
from .probes import paired_mean_direction


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(normalize(left) @ normalize(right))


def _user_direction(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    positive_group: str,
    negative_group: str,
) -> np.ndarray:
    positive: dict[tuple[int, str], np.ndarray] = {}
    negative: dict[tuple[int, str], np.ndarray] = {}
    for feature, row in zip(features, rows, strict=True):
        if row["target_identity"] or row["target_organization"]:
            continue
        key = (int(row["pair_index"]), str(row["template_id"]))
        if row["group"] == positive_group:
            positive[key] = feature
        elif row["group"] == negative_group:
            negative[key] = feature
    keys = sorted(set(positive) & set(negative))
    if not keys:
        raise ValueError("no paired non-target rows for geometry")
    return paired_mean_direction(
        np.asarray([positive[key] for key in keys]),
        np.asarray([negative[key] for key in keys]),
    )


def build_geometry_report(artifact_root: Path, output_path: Path) -> dict[str, Any]:
    eval_report = json.loads(
        (artifact_root / "m3/eval_direction_report.json").read_text(encoding="utf-8")
    )
    construction = str(eval_report["selected_construction"])
    selected_layer = int(eval_report["selected_layer"])
    eval_vectors = np.load(artifact_root / "m3/eval_directions.npz")
    user_features = np.load(artifact_root / "m4/user_features.npz")
    rows = _rows(artifact_root / "m4/user_features.jsonl")
    concepts = {
        "name": ("F", "U"),
        "affiliation": ("U", "G"),
        "fame": ("FN", "G"),
    }
    layerwise: dict[str, list[dict[str, float]]] = defaultdict(list)
    selected_directions: dict[str, np.ndarray] = {
        "eval": normalize(eval_vectors[f"l{selected_layer}__{construction}"])
    }
    for layer in range(45):
        features = user_features[f"l{layer}__prompt_final"].astype(np.float64)
        eval_direction = normalize(eval_vectors[f"l{layer}__{construction}"])
        for concept, groups in concepts.items():
            direction = _user_direction(features, rows, *groups)
            layerwise[concept].append(
                {"layer": layer, "raw_cosine_with_eval": _cosine(eval_direction, direction)}
            )
            if layer == selected_layer:
                selected_directions[concept] = direction

    selected_features = user_features[f"l{selected_layer}__prompt_final"].astype(np.float64)
    centered = selected_features - selected_features.mean(axis=0)
    names = tuple(selected_directions)
    raw_cosines: dict[str, float] = {}
    whitened_cosines: dict[str, float] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}__{right}"
            raw_cosines[key] = _cosine(selected_directions[left], selected_directions[right])
            left_scores = centered @ normalize(selected_directions[left])
            right_scores = centered @ normalize(selected_directions[right])
            whitened_cosines[key] = float(np.corrcoef(left_scores, right_scores)[0, 1])

    contrasts = {
        "name": ("F", "U"),
        "affiliation": ("U", "G"),
        "fame": ("FN", "G"),
    }
    cross_decoding: dict[str, dict[str, float]] = {}
    for direction_name, direction in selected_directions.items():
        scores = selected_features @ normalize(direction)
        cross_decoding[direction_name] = {}
        for contrast, (positive, negative) in contrasts.items():
            indices = [
                index for index, row in enumerate(rows) if row["group"] in {positive, negative}
            ]
            labels = np.asarray([int(rows[index]["group"] == positive) for index in indices])
            cross_decoding[direction_name][contrast] = float(
                roc_auc_score(labels, scores[indices])
            )

    residual_norms: dict[str, float] = {}
    for name, direction in selected_directions.items():
        others = [normalize(value) for key, value in selected_directions.items() if key != name]
        basis, _ = np.linalg.qr(np.column_stack(others))
        residual = normalize(direction) - basis @ (basis.T @ normalize(direction))
        residual_norms[name] = float(np.linalg.norm(residual))
    report = {
        "schema_version": "glm53_v8_geometry_report_v1",
        "selected_layer": selected_layer,
        "eval_construction": construction,
        "raw_cosines": raw_cosines,
        "whitened_projection_cosines": whitened_cosines,
        "cross_decoding_auroc": cross_decoding,
        "residual_direction_norms": residual_norms,
        "layerwise_eval_overlap": dict(layerwise),
        "causal_claim_from_cosine_alone": False,
    }
    atomic_json(output_path, report)
    return report


def terminal_state(artifact_root: Path) -> dict[str, Any]:
    final_m8 = artifact_root / "decisions/m8_decision.json"
    if final_m8.is_file():
        decision = json.loads(final_m8.read_text(encoding="utf-8"))
        return {"terminal": True, "gate": "M8", "decision": decision}
    summary_path = artifact_root / "supervisor_summary.json"
    if not summary_path.is_file():
        return {"terminal": False, "reason": "supervisor_summary_missing"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stopped = summary.get("stopped_at")
    if not stopped or stopped == "M8_manual_audit_pending":
        return {"terminal": False, "reason": stopped or "supervisor_not_stopped"}
    gate = "M4" if stopped == "M4_distinct_path_unavailable" else stopped.split("_")[0]
    decision_path = artifact_root / f"decisions/{gate.lower()}_decision.json"
    decision = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file()
        else None
    )
    return {"terminal": True, "gate": gate, "decision": decision, "stopped_at": stopped}


def build_final_report(artifact_root: Path, output_path: Path) -> str:
    state = terminal_state(artifact_root)
    if not state["terminal"]:
        raise ValueError(f"study is not terminal: {state['reason']}")
    m4_path = artifact_root / "m4/recruitment_report.json"
    m5_path = artifact_root / "m5/proxy_report.json"
    m7_path = artifact_root / "m7/confirmation_report.json"
    m4 = json.loads(m4_path.read_text(encoding="utf-8")) if m4_path.is_file() else None
    m5 = json.loads(m5_path.read_text(encoding="utf-8")) if m5_path.is_file() else None
    m7 = json.loads(m7_path.read_text(encoding="utf-8")) if m7_path.is_file() else None
    if state["gate"] == "M8" and state["decision"]["passed"]:
        outcome = "shared_path" if m7["pathway"] == "shared_eval" else "distinct_path"
    elif m5 is not None and not m5.get("passed", False):
        outcome = "local_proxy_mismatch"
    elif m4 is not None and not m4.get("passed", False):
        outcome = "no_eval_recruitment"
    elif m7 is not None and not m7.get("passed", False):
        outcome = "readable_but_not_causally_supported"
    else:
        outcome = f"stopped_at_{state['gate'].lower()}"
    lines = [
        "# GLM-5.3 v8 mechanism study",
        "",
        f"Terminal outcome: `{outcome}`.",
        f"Highest terminal gate: `{state['gate']}`.",
        "",
    ]
    if m4 is not None:
        lines.append(f"Frozen eval-readout interaction: {m4['interaction_sd']:.6f} source SD.")
    if m5 is not None:
        lines.append(f"Local one-token proxy interaction: {m5['interaction_pp']:.6f} pp.")
    if m7 is not None:
        lines.extend(
            [
                f"Causal candidate path: `{m7['pathway']}` (`{m7['concept']}`).",
                f"Candidate interaction change: {m7['delta_pp']:.6f} pp.",
                f"Fraction removed: {m7['fraction_removed']:.6f}.",
                f"Matched-control rank: {m7['candidate_control_rank']} of {m7['control_count'] + 1}.",
            ]
        )
    lines.extend(
        [
            "",
            "The tested readouts are linear. Steering establishes dependence only under the specified intervention. It does not establish a literal belief or complete mediation.",
            "",
        ]
    )
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text
