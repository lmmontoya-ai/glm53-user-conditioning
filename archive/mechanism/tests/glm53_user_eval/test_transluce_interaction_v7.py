import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from pipelines.glm53_user_eval.scripts.analyze_transluce_exact_v6 import matrices
from pipelines.glm53_user_eval.scripts.analyze_transluce_interaction_v7 import (
    bootstrap_interaction,
    interaction_point,
    statistical_state,
)
from pipelines.glm53_user_eval.scripts.run_transluce_exact_v6 import (
    Shard,
    inspect_command,
)
from pipelines.glm53_user_eval.scripts.run_transluce_interaction_v7 import (
    load_and_validate,
    v7_plan,
)

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/configs/prereg_v7_interaction_confirmatory.yaml"
V6_PREREG = ROOT / "pipelines/glm53_user_eval/configs/prereg_v6_transluce_exact.yaml"
RUNNER = ROOT / "pipelines/glm53_user_eval/run.py"
V7_RUNNER = ROOT / "pipelines/glm53_user_eval/scripts/run_transluce_interaction_v7.py"
TRANSLUCE = ROOT.parent / "reference/transluce-user-awareness"


def test_v7_preregistration_validates_and_locks_discovery() -> None:
    completed = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(RUNNER),
            "validate-prereg",
            "--prereg",
            str(PREREG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "glm53_user_eval_prereg_v7"
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    assert prereg["discovery_result"]["interaction_pp"] == -0.8314710848350346
    assert prereg["discovery_result"]["role"] == "exploratory_hypothesis_generation_only"


def test_v7_generation_contract_sections_equal_v6() -> None:
    v7 = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    v6 = yaml.safe_load(V6_PREREG.read_text(encoding="utf-8"))
    for section in v7["parent_generation_contract"]["equal_sections"]:
        assert v7[section] == v6[section]


def test_v7_plan_has_100_shards_and_34400_rows() -> None:
    prereg = load_and_validate(PREREG, TRANSLUCE)
    payload = v7_plan(PREREG, prereg, TRANSLUCE)
    assert payload["shard_count"] == 100
    assert payload["total_expected_rows"] == 34400
    assert [row["group"] for row in payload["shards"][:5]] == [
        "genpop",
        "unknown_ai",
        "famous_ai",
        "famous_ai_real",
        "famous_nonai",
    ]
    assert [row["expected_rows"] for row in payload["shards"][:5]] == [355, 355, 355, 300, 355]


def test_v7_command_is_identical_to_v6_except_output_path(tmp_path: Path) -> None:
    v7 = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    v6 = yaml.safe_load(V6_PREREG.read_text(encoding="utf-8"))
    shard = Shard("famous_ai", 10, 5, ("fai2_neel_nanda", "anon"))
    left = inspect_command(
        source_root=TRANSLUCE, prereg=v6, shard=shard, log_dir=tmp_path / "v6", connections=40
    )
    right = inspect_command(
        source_root=TRANSLUCE, prereg=v7, shard=shard, log_dir=tmp_path / "v7", connections=40
    )
    left[left.index("--log-dir") + 1] = "<LOG_DIR>"
    right[right.index("--log-dir") + 1] = "<LOG_DIR>"
    assert left == right
    command = " ".join(right)
    assert "temperature" not in command
    assert "top_p" not in command
    assert "generation_seed" not in command


def test_fixed_dilemma_split_is_hash_reproducible() -> None:
    path = ROOT / "pipelines/glm53_user_eval/reference/dilemma_split_v7.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [f"dd_{index:04d}" for index in range(100)]
    expected = sorted(
        ids,
        key=lambda value: hashlib.sha256(
            f"glm53_user_eval_interaction_v7|{value}".encode()
        ).hexdigest(),
    )
    assert payload["split_a"] == expected[:50]
    assert payload["split_b"] == expected[50:]


def test_interaction_point_and_group_centering() -> None:
    matrices = {
        "genpop": np.array([[60.0, 80.0], [80.0, 60.0]]),
        "unknown_ai": np.array([[70.0, 70.0], [70.0, 70.0]]),
        "famous_ai": np.array([[68.0, 68.0], [68.0, 68.0]]),
        "famous_nonai": np.array([[73.0, 73.0], [73.0, 73.0]]),
    }
    point, means = interaction_point(matrices)
    assert means["genpop"] == 0.0
    assert point == -5.0


def test_crossed_bootstrap_preserves_constant_known_interaction() -> None:
    matrices = {
        "genpop": np.full((70, 100), 70.0),
        "unknown_ai": np.full((70, 100), 70.0),
        "famous_ai": np.full((70, 100), 68.0),
        "famous_nonai": np.full((70, 100), 73.0),
    }
    points, intervals, draws = bootstrap_interaction(matrices, reps=100, seed=7)
    assert points["interaction_pp"] == -5.0
    assert intervals["interaction_pp"] == [-5.0, -5.0]
    assert np.all(draws == -5.0)


def test_decision_boundaries() -> None:
    assert statistical_state(-0.7, [-1.0, -0.2]) == "confirmed_target_sized_interaction"
    assert statistical_state(-0.2, [-0.4, -0.1]) == "confirmed_small_interaction"
    assert statistical_state(-0.1, [-0.3, 0.1]) == "target_magnitude_ruled_out"
    assert statistical_state(-0.4, [-0.8, 0.1]) == "directional_ambiguous"
    assert statistical_state(0.1, [-0.8, 0.3]) == "null_or_opposite_interaction"


def test_golden_v6_discovery_interaction() -> None:
    raw = ROOT / "artifacts/glm53_user_eval/reports/transluce_exact_v6/raw_scores.jsonl"
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    roster = json.loads((TRANSLUCE / "core/personas2.json").read_text(encoding="utf-8"))
    group_matrices, _stimuli = matrices(rows, roster)
    interaction, _means = interaction_point(group_matrices)
    assert abs(interaction - (-0.8314710848350346)) <= 1e-12


def test_mechanistic_anchors_do_not_use_glm53_rankings() -> None:
    payload = json.loads(
        (
            ROOT / "pipelines/glm53_user_eval/reference/mechanistic_anchor_manifest_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["selection_source"] == "pinned_released_glm52_cache_only"
    assert len(payload["famous_unknown_pairs"]) == 8
    assert len(payload["famous_nonai_controls"]) == 8
    assert len(payload["dilemmas"]) == 25
