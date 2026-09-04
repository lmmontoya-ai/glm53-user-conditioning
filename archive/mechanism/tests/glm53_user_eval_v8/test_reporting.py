from __future__ import annotations

from pathlib import Path

import numpy as np
from src.glm53_user_eval.v8.artifacts import atomic_json, atomic_jsonl
from src.glm53_user_eval.v8.reporting import (
    build_final_report,
    build_geometry_report,
    terminal_state,
)


def test_terminal_state_waits_for_manual_audit(tmp_path) -> None:
    atomic_json(
        tmp_path / "supervisor_summary.json",
        {"highest_passed_gate": "M7", "stopped_at": "M8_manual_audit_pending"},
    )
    result = terminal_state(tmp_path)
    assert result == {"terminal": False, "reason": "M8_manual_audit_pending"}


def test_terminal_state_accepts_honest_failed_gate(tmp_path) -> None:
    atomic_json(tmp_path / "decisions/m5_decision.json", {"gate": "M5", "passed": False})
    atomic_json(
        tmp_path / "supervisor_summary.json",
        {"highest_passed_gate": "M4", "stopped_at": "M5"},
    )
    result = terminal_state(tmp_path)
    assert result["terminal"] is True
    assert result["gate"] == "M5"
    assert result["decision"]["passed"] is False


def test_final_report_names_local_proxy_mismatch(tmp_path) -> None:
    atomic_json(tmp_path / "decisions/m5_decision.json", {"gate": "M5", "passed": False})
    atomic_json(
        tmp_path / "supervisor_summary.json",
        {"highest_passed_gate": "M4", "stopped_at": "M5"},
    )
    atomic_json(
        tmp_path / "m5/proxy_report.json",
        {"interaction_pp": 0.1, "passed": False},
    )
    output = tmp_path / "final_report.md"
    text = build_final_report(tmp_path, output)
    assert "local_proxy_mismatch" in text
    assert output.read_text(encoding="utf-8") == text


def _geometry_artifacts(root: Path) -> None:
    rows = []
    base_features = []
    group_values = {
        "F": np.asarray([2.0, 0.0, 0.0]),
        "U": np.asarray([1.0, 1.0, 0.0]),
        "G": np.asarray([0.0, 1.0, 1.0]),
        "FN": np.asarray([0.0, 0.0, 2.0]),
    }
    for group, value in group_values.items():
        for pair in range(2):
            rows.append(
                {
                    "sample_id": f"{group}-{pair}",
                    "group": group,
                    "persona_key": f"{group}-{pair}",
                    "pair_index": pair,
                    "template_id": "neutral",
                    "target_identity": False,
                    "target_organization": False,
                }
            )
            base_features.append(value + pair * 0.01)
    atomic_jsonl(root / "m4/user_features.jsonl", rows)
    user_arrays = {
        f"l{layer}__prompt_final": np.asarray(base_features, dtype=np.float32)
        for layer in range(45)
    }
    (root / "m4").mkdir(parents=True, exist_ok=True)
    np.savez(root / "m4/user_features.npz", **user_arrays)
    eval_arrays = {
        f"l{layer}__paired_mean": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        for layer in range(45)
    }
    (root / "m3").mkdir(parents=True, exist_ok=True)
    np.savez(root / "m3/eval_directions.npz", **eval_arrays)
    atomic_json(
        root / "m3/eval_direction_report.json",
        {"selected_layer": 20, "selected_construction": "paired_mean"},
    )


def test_geometry_report_has_layerwise_and_cross_decoding_outputs(tmp_path) -> None:
    _geometry_artifacts(tmp_path)
    report = build_geometry_report(tmp_path, tmp_path / "m8/geometry_report.json")
    assert report["selected_layer"] == 20
    assert len(report["layerwise_eval_overlap"]["name"]) == 45
    assert set(report["cross_decoding_auroc"]) == {"eval", "name", "affiliation", "fame"}
    assert report["causal_claim_from_cosine_alone"] is False
