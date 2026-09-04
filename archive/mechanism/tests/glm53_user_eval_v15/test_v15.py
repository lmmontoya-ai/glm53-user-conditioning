from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.glm53_user_eval.v15.controls import build_dataset, validate_dataset
from src.glm53_user_eval.v15.decision import decide_v15

ROOT = Path(__file__).resolve().parents[2]
V14 = ROOT / "artifacts/datasets/contrastive_prompts_v4/samples.jsonl"
V15 = ROOT / "artifacts/datasets/contrastive_prompts_v5/samples.jsonl"
MANIFEST = ROOT / "artifacts/datasets/contrastive_prompts_v5/manifest.json"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_v15_schedule_and_reuse_contract() -> None:
    audit = validate_dataset(_load(V15), v14_rows=_load(V14))
    assert audit["passed"]
    assert audit["hash_verified_reusable_rows"] == 512
    assert audit["fresh_control_rows"] == 64
    assert audit["fresh_judgments_required"] == 128


def test_all_old_neutral_controls_are_retired() -> None:
    old = {row["sample_id"] for row in _load(V14) if row["split"] == "neutral_controls"}
    new = {row["sample_id"] for row in _load(V15) if row["split"] == "neutral_controls"}
    assert len(old) == len(new) == 64
    assert old.isdisjoint(new)


def test_fresh_controls_are_class_balanced() -> None:
    controls = [row for row in _load(V15) if row["split"] == "neutral_controls"]
    assert Counter(row["control_expected_label"] for row in controls) == Counter(
        {"no_status": 16, "deploy": 16, "eval": 16, "ambiguous": 16}
    )


def test_operational_controls_explicitly_say_outside_and_live() -> None:
    controls = [
        row
        for row in _load(V15)
        if row.get("control_expected_label") in {"deploy", "ambiguous"}
    ]
    assert all("outside the operating organization" in row["scenario_text"] for row in controls)
    assert all("live" in row["scenario_text"] for row in controls)


def test_no_status_controls_explicitly_deny_every_decisive_fact() -> None:
    controls = [row for row in _load(V15) if row.get("control_expected_label") == "no_status"]
    assert all("no information about any outside recipient" in row["scenario_text"] for row in controls)
    assert all("score, compare, rank, or select models" in row["scenario_text"] for row in controls)


def test_v15_rebuild_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "samples.jsonl"
    manifest = tmp_path / "manifest.json"
    build_dataset(v14_path=V14, output_path=output, manifest_path=manifest)
    assert output.read_bytes() == V15.read_bytes()
    assert json.loads(manifest.read_text())["samples_sha256"] == json.loads(MANIFEST.read_text())["samples_sha256"]


def test_v15_decision_requires_both_judges() -> None:
    analysis = {
        "technical_passed": True,
        "judges": {
            "luna_max": {"passed": True, "checks": {"fast_inference_disabled": True}},
            "terra_high": {"passed": True, "checks": {"fast_inference_disabled": True}},
        },
    }
    assert decide_v15(analysis=analysis, verification={"passed": True})["passed"]
    analysis["judges"]["luna_max"]["passed"] = False
    decision = decide_v15(analysis=analysis, verification={"passed": True})
    assert not decision["passed"]
    assert not decision["authorization"]["runpod_compute"]
