import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_task_splits_are_disjoint_and_complete() -> None:
    payload = json.loads(
        (ROOT / "pipelines/glm53_user_eval/configs/task_splits_v1.json").read_text(encoding="utf-8")
    )
    values = [set(items) for items in payload["splits"].values()]
    assert len(payload["splits"]["behavior_main_50"]) == 50
    assert len(payload["splits"]["behavior_hardening_25"]) == 25
    assert len(payload["splits"]["behavior_causal_25"]) == 25
    assert not values[0] & values[1]
    assert not values[0] & values[2]
    assert not values[1] & values[2]
    assert len(set.union(*values)) == 100


def test_main_schedule_has_frozen_grid(tmp_path: Path) -> None:
    output = tmp_path / "schedule"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipelines/glm53_user_eval/run.py"),
            "build-behavior-schedule",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["persona_condition_count"] == 65
    assert manifest["sample_count"] == 3250


def test_local_schedule_has_frozen_matched_grid(tmp_path: Path) -> None:
    output = tmp_path / "local-schedule"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipelines/glm53_user_eval/run.py"),
            "build-local-behavior-schedule",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output / "schedule.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["identity_pair_count"] == 4
    assert manifest["persona_condition_count"] == 12
    assert manifest["sample_count"] == 600
    assert [row["condition"] for row in rows[:3]] == [
        "famous_coherent",
        "unknown_same_org",
        "unknown_general",
    ]
    assert len({row["analysis_block"] for row in rows}) == 5
