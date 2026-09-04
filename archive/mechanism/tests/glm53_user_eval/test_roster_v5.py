import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipelines.glm53_user_eval.scripts.build_roster_audit_packet import select_sample
from pipelines.glm53_user_eval.scripts.independent_roster_recompute import recompute
from src.glm53_user_eval.analysis import _bh_adjust, decide_roster_result


ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/configs/prereg_v5_roster.yaml"
RUNNER = ROOT / "pipelines/glm53_user_eval/run.py"


def test_v5_preregistration_validates_and_preserves_v4() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "validate-prereg", "--prereg", str(PREREG)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "glm53_user_eval_prereg_v5"


def _build(tmp_path: Path, split: str, phase: str) -> Path:
    output = tmp_path / split
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "build-roster-v5-schedule",
            "--prereg",
            str(PREREG),
            "--task-split",
            split,
            "--phase",
            phase,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def test_v5_schedules_cover_all_roster_indices_and_untouched_splits(tmp_path: Path) -> None:
    discovery = _build(tmp_path, "behavior_hardening_25", "roster_discovery_v5")
    confirmation = _build(tmp_path, "behavior_causal_25", "roster_confirmation_v5")
    discovery_rows = [
        json.loads(line)
        for line in (discovery / "schedule.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    confirmation_rows = [
        json.loads(line)
        for line in (confirmation / "schedule.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(discovery_rows) == len(confirmation_rows) == 7000
    assert {row["pair_index"] for row in discovery_rows} == set(range(70))
    assert {row["condition"] for row in discovery_rows} == {
        "famous_coherent",
        "unknown_same_org",
        "unknown_general",
        "famous_nonai_control",
    }
    assert {row["scenario_id"] for row in discovery_rows}.isdisjoint(
        {row["scenario_id"] for row in confirmation_rows}
    )
    assert {row["sample_id"] for row in discovery_rows}.isdisjoint(
        {row["sample_id"] for row in confirmation_rows}
    )


def test_bh_adjustment_is_monotone_in_rank() -> None:
    adjusted = _bh_adjust({0: 0.001, 1: 0.02, 2: 0.2, 3: 0.9})
    assert adjusted[0] <= adjusted[1] <= adjusted[2] <= adjusted[3]
    assert adjusted[0] == 0.004


def _stage(
    *,
    name: float,
    name_ci: list[float],
    negative: int,
    affiliation: float = 0.0,
    affiliation_ci: list[float] | None = None,
    fame: float = 0.0,
    candidate: bool = False,
    confirm_candidate: bool = False,
) -> dict:
    identity_rows = []
    for index in range(70):
        is_candidate = candidate and index == 7
        effect = -2.0 if (is_candidate or (confirm_candidate and index == 7)) else name
        identity_rows.append(
            {
                "pair_index": index,
                "effect_pp": effect,
                "bh_q": 0.01 if is_candidate else 1.0,
                "task_bootstrap_ci95_pp": [-3.0, -1.0]
                if confirm_candidate and index == 7
                else [-1.0, 1.0],
            }
        )
    return {
        "name_effect_pp": name,
        "name_ci95_pp": name_ci,
        "negative_name_identity_count": negative,
        "affiliation_effect_pp": affiliation,
        "affiliation_ci95_pp": affiliation_ci or [-0.5, 0.5],
        "generic_fame_effect_pp": fame,
        "identity_effects": identity_rows,
    }


def test_roster_decision_requires_held_out_replication() -> None:
    discovery = _stage(name=-1.2, name_ci=[-1.6, -0.8], negative=50, fame=-0.2)
    confirmation = _stage(name=-0.8, name_ci=[-1.2, -0.4], negative=48, fame=-0.1)
    combined = _stage(name=-1.0, name_ci=[-1.2, -0.8], negative=50)
    checks, decision = decide_roster_result(discovery, confirmation, combined)
    assert checks["roster_effect_positive"] is True
    assert decision == "roster_effect_positive_unlock_exact_checkpoint_decision"


def test_identity_candidate_must_confirm_on_second_split() -> None:
    discovery = _stage(
        name=0.0,
        name_ci=[-0.3, 0.3],
        negative=35,
        candidate=True,
    )
    confirmation = _stage(
        name=0.0,
        name_ci=[-0.3, 0.3],
        negative=35,
        confirm_candidate=True,
    )
    combined = _stage(name=0.0, name_ci=[-0.2, 0.2], negative=35)
    checks, decision = decide_roster_result(discovery, confirmation, combined)
    assert checks["identity_specific_effect_positive"] is True
    assert decision == "identity_specific_effect_positive_unlock_targeted_exact_checkpoint_decision"


def test_clean_null_uses_combined_lower_bound() -> None:
    discovery = _stage(name=0.0, name_ci=[-0.4, 0.4], negative=35)
    confirmation = _stage(name=0.1, name_ci=[-0.3, 0.5], negative=32)
    combined = _stage(name=0.05, name_ci=[-0.2, 0.3], negative=34)
    checks, decision = decide_roster_result(discovery, confirmation, combined)
    assert checks["clean_null_established"] is True
    assert decision == "clean_roster_null_stop_glm53_user_awareness_project"


def test_manual_audit_selection_balances_conditions_and_blocks() -> None:
    schedule = []
    for condition in (
        "famous_coherent",
        "unknown_same_org",
        "unknown_general",
        "famous_nonai_control",
    ):
        for block in range(5):
            for offset in range(3):
                schedule.append(
                    {
                        "sample_id": f"{condition}-{block}-{offset}",
                        "condition": condition,
                        "analysis_block": block,
                    }
                )
    selected = select_sample(schedule, seed=7, sample_size=40)
    assert len(selected) == 40
    for condition in {row["condition"] for row in schedule}:
        assert sum(row["condition"] == condition for row in selected) == 10
    for block in range(5):
        assert sum(row["analysis_block"] == block for row in selected) == 8
    assert selected == select_sample(schedule, seed=7, sample_size=40)


def test_independent_roster_recompute_uses_matched_cells() -> None:
    schedule = []
    results = []
    values = {
        "famous_coherent": 70.0,
        "unknown_same_org": 75.0,
        "unknown_general": 76.0,
        "famous_nonai_control": 77.0,
    }
    for pair_index in range(70):
        for scenario_index in range(2):
            for condition, confidence in values.items():
                sample_id = f"{pair_index}-{scenario_index}-{condition}"
                schedule.append(
                    {
                        "sample_id": sample_id,
                        "pair_index": pair_index,
                        "scenario_id": f"task-{scenario_index}",
                        "condition": condition,
                    }
                )
                results.append(
                    {
                        "sample_id": sample_id,
                        "parse_valid": True,
                        "confidence_p": confidence,
                    }
                )
    estimates = recompute(results, schedule, reps=100, seed=3)
    assert estimates["sample_count"] == 560
    assert estimates["name_effect_pp"] == pytest.approx(-5.0)
    assert estimates["affiliation_effect_pp"] == pytest.approx(-1.0)
    assert estimates["generic_fame_effect_pp"] == pytest.approx(1.0)
    assert estimates["negative_name_identity_count"] == 70
