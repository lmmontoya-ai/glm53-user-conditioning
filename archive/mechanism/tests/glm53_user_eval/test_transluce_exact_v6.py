import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pipelines.glm53_user_eval.scripts.analyze_transluce_exact_v6 import (
    bootstrap_paired_contrast,
    exact_group_stats,
    folded,
    reference_person_deltas,
)
from pipelines.glm53_user_eval.scripts.build_transluce_v6_audit_packet import (
    select_candidates,
)

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/configs/prereg_v6_transluce_exact.yaml"
RUNNER = ROOT / "pipelines/glm53_user_eval/run.py"
SHARD_RUNNER = ROOT / "pipelines/glm53_user_eval/scripts/run_transluce_exact_v6.py"
TRANSLUCE = ROOT.parent / "reference/transluce-user-awareness"


def test_v6_preregistration_validates() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "validate-prereg", "--prereg", str(PREREG)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "glm53_user_eval_prereg_v6"
    assert payload["api_provider"] == "Novita"
    assert payload["api_model_id"] == "z-ai/glm-5.3-flash"


def test_v6_plan_is_the_complete_reference_population(tmp_path: Path) -> None:
    source_python = TRANSLUCE / ".venv/Scripts/python.exe"
    if not source_python.exists():
        return
    subprocess.run(
        [
            str(source_python),
            str(SHARD_RUNNER),
            "plan",
            "--prereg",
            str(PREREG),
            "--source-root",
            str(TRANSLUCE),
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads((tmp_path / "schedule_manifest.json").read_text(encoding="utf-8"))
    assert payload["shard_count"] == 100
    assert payload["total_expected_rows"] == 34400
    assert {row["group"] for row in payload["shards"]} == {
        "genpop",
        "unknown_ai",
        "famous_ai",
        "famous_ai_real",
        "famous_nonai",
    }
    assert all(row["limit"] == 5 for row in payload["shards"])
    assert [row["group"] for row in payload["shards"][:5]] == [
        "genpop",
        "unknown_ai",
        "famous_ai",
        "famous_ai_real",
        "famous_nonai",
    ]


def test_v6_runtime_command_keeps_original_sampling_defaults() -> None:
    text = SHARD_RUNNER.read_text(encoding="utf-8")
    assert '"reasoning_effort={subject[\'reasoning_effort\']}"' in text
    assert '"confidence_style=confidence"' in text
    assert '"seed=all"' in text
    assert '"temperature"' not in text
    assert '"top_p"' not in text
    assert '"assistant_name"' not in text


def test_v6_folded_confidence_matches_reference_transform() -> None:
    assert folded(0.9) == 90.0
    assert folded(0.1) == 90.0
    assert folded(0.5) == 50.0


def test_v6_person_delta_centers_on_genpop_per_dilemma() -> None:
    matrices = {
        "genpop": np.array([[60.0, 80.0], [80.0, 60.0]]),
        "unknown_ai": np.array([[60.0, 60.0]]),
        "famous_ai": np.array([[50.0, 50.0]]),
        "famous_ai_real": np.array([[70.0, 70.0]]),
        "famous_nonai": np.array([[80.0, 80.0]]),
    }
    roster = {
        "genpop": [{"key": "g0"}, {"key": "g1"}],
        "unknown_ai": [{"key": "u0"}],
        "famous_ai": [{"key": "f0"}],
        "famous_ai_real": [{"key": "r0"}],
        "famous_nonai": [{"key": "n0"}],
    }
    people, means = reference_person_deltas(matrices, roster)
    assert means["genpop"] == 0.0
    assert means["famous_ai"] == -20.0
    assert next(row for row in people if row["persona"] == "f0")["valid_dilemmas"] == 2


def test_v6_paired_bootstrap_point_uses_same_index_pairs() -> None:
    left = np.array([[80.0, 70.0], [60.0, 50.0]])
    right = np.array([[70.0, 60.0], [50.0, 40.0]])
    point, interval = bootstrap_paired_contrast(left, right, reps=100, seed=7)
    assert point == 10.0
    assert interval == [10.0, 10.0]


def test_v6_exact_group_stats_match_source_indexing_and_direction() -> None:
    people = []
    for group, values in {
        "genpop": [-1.0, 0.0, 1.0, 2.0],
        "unknown_ai": [-1.0, 0.0, 1.0, 2.0],
        "famous_ai": [-4.0, -3.0, -2.0, -1.0],
        "famous_ai_real": [-2.0, -1.0, 0.0, 1.0],
        "famous_nonai": [-1.0, 0.0, 1.0, 2.0],
    }.items():
        people.extend(
            {"persona": f"{group}-{index}", "group": group, "mean_delta_pp": value}
            for index, value in enumerate(values)
        )
    stats = exact_group_stats(people)
    assert stats["famous_ai"]["median_pp"] == -2.5
    assert stats["famous_ai"]["iqr_source_indexed_pp"] == [-3.0, -1.0]
    assert float(stats["famous_ai"]["mann_whitney_z_vs_genpop"]) < 0


def test_v6_manual_audit_selection_is_group_and_block_balanced() -> None:
    candidates = []
    groups = ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai")
    for group in groups:
        for block in range(5):
            for index in range(3):
                candidates.append(
                    {
                        "sample_id": f"{group}-{block}-{index}",
                        "shard_id": f"{group}-{block}",
                        "group": group,
                        "persona": f"person-{index}",
                        "stimulus": f"dd_{block * 20 + index:04d}",
                        "task_block": block,
                    }
                )
        candidates.append(
            {
                "sample_id": f"{group}-anon",
                "shard_id": f"{group}-0",
                "group": group,
                "persona": "anon",
                "stimulus": "dd_0000",
                "task_block": 0,
            }
        )
    selected = select_candidates(candidates, seed=11)
    assert len(selected) == 40
    for group in groups:
        group_rows = [row for row in selected if row["group"] == group]
        assert len(group_rows) == 8
        assert sum(row["persona"] == "anon" for row in group_rows) == 1
        assert {row["task_block"] for row in group_rows} == set(range(5))
