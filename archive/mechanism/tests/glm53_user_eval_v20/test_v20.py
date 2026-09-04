from __future__ import annotations

import inspect
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from src.glm53_user_eval.v20 import supervisor, verification
from src.glm53_user_eval.v20.analysis import (
    FULL_ARMS,
    analyze_causal_rows,
    arm_matrices,
    causal_delta_bootstrap,
    interaction,
)
from src.glm53_user_eval.v20.contract import expected_signs, validate_v20_prereg

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/v20/configs/prereg_v20_direct_user_hua.yaml"


def test_preregistration_and_immutable_hashes_pass() -> None:
    report = validate_v20_prereg(ROOT, PREREG)
    assert report["passed"] is True
    assert report["planned_scientific_prompt_evaluations"] == 10024
    assert report["signflip_controls"] == 20


def test_v19_parent_is_immutable_and_contains_no_target_rows() -> None:
    prereg = __import__("yaml").safe_load(PREREG.read_text())
    decision = json.loads(
        (ROOT / prereg["immutable_inputs"]["v19_positive_control_decision"]["path"]).read_text()
    )
    assert decision["decision"] == "stop_hua_direction_did_not_transfer"
    assert decision["authorization"] == {
        "causal_user_test": False,
        "local_proxy_parity": False,
    }


def test_design_counts_are_exact() -> None:
    design = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v20/configs/design_v20.json").read_text()
    )
    assert design["full_surface"]["reconstructable_rows"] == 1404
    assert design["full_surface"]["prompt_evaluations"] == 8424
    assert design["signflip_control_surface"]["prompt_evaluations"] == 1600
    assert design["planned_scientific_prompt_evaluations"] == 10024


def test_selected_real_rows_and_null_subset_counts() -> None:
    from src.glm53_user_eval.v11.downstream import validate_downstream_assets

    parent = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v19/configs/design_v19.json").read_text()
    )
    _, rows, _ = validate_downstream_assets(
        repo_root=ROOT,
        manifest_path=ROOT / "pipelines/glm53_user_eval/v17/configs/downstream_manifest_v17.json",
    )
    selected = supervisor._selected_proxy_rows(rows, parent)
    assert len(selected) == 1404
    pilot = [
        row
        for row in selected
        if row["stage_index"] < 4 and row["stimulus_id"] in set(parent["tasks"][:5])
    ]
    assert len(pilot) == 80
    assert {
        group: sum(row["group"] == group for row in pilot) for group in parent["identities"]
    } == {group: 20 for group in parent["identities"]}


def test_signflip_derivation_is_literal_and_balanced() -> None:
    payload = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v20/configs/null_signs_v20.json").read_text()
    )
    assert len(payload["controls"]) == 20
    for row in payload["controls"]:
        assert row["signs"] == expected_signs(row["derivation_index"])
        assert row["signs"].count(1) == row["signs"].count(-1) == 8


def test_direction_bundle_shapes_norms_and_actor_definition() -> None:
    prereg = __import__("yaml").safe_load(PREREG.read_text())
    bundle = supervisor._direction_bundle(
        ROOT / prereg["immutable_inputs"]["v19_directions"]["path"],
        ROOT / prereg["immutable_inputs"]["null_signs"]["path"],
    )
    assert set(bundle) == {
        "all16",
        "status",
        "actor",
        "formality",
        *{f"signflip_{index:02d}" for index in range(20)},
    }
    assert all(value.shape == (45, 4096) for value in bundle.values())
    candidate_norm = np.linalg.norm(bundle["all16"].astype(np.float64), axis=1)
    for index in range(20):
        null_norm = np.linalg.norm(bundle[f"signflip_{index:02d}"].astype(np.float64), axis=1)
        assert np.allclose(null_norm, candidate_norm, rtol=2e-6, atol=2e-6)


def _full_rows() -> list[dict[str, object]]:
    offsets = {
        "baseline": {"famous_ai": -0.6, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "all16_deploy": {"famous_ai": -0.1, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "all16_eval": {"famous_ai": -0.9, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "status_deploy": {"famous_ai": -0.2, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "actor_deploy": {"famous_ai": -0.4, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "formality": {"famous_ai": -0.55, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
    }
    rows: list[dict[str, object]] = []
    for arm, by_group in offsets.items():
        for group, offset in by_group.items():
            for identity in range(16):
                for task in range(22):
                    if identity == 15 and task == 21:
                        continue
                    rows.append(
                        {
                            "sample_id": f"{group}-{identity}-{task}",
                            "arm": arm,
                            "group": group,
                            "stage_index": identity,
                            "stimulus_id": f"task-{task:02d}",
                            "expected_folded_confidence": 75.0 + offset,
                            "codebook_id": str((identity + task) % 2),
                            "allowed_mass": 0.9,
                            "full_vocab_argmax_allowed": True,
                            "conditional_entropy": 0.5,
                        }
                    )
    return rows


def _null_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for null_index in range(20):
        arm = f"signflip_{null_index:02d}"
        shift = 0.01 * (null_index + 1)
        for group in ("famous_ai", "unknown_ai", "famous_nonai", "genpop"):
            offset = (
                -0.6 + shift if group == "famous_ai" else (0.2 if group == "famous_nonai" else 0.0)
            )
            for identity in range(4):
                for task in range(5):
                    rows.append(
                        {
                            "sample_id": f"{group}-{identity}-{task}",
                            "arm": arm,
                            "group": group,
                            "stage_index": identity,
                            "stimulus_id": f"task-{task:02d}",
                            "expected_folded_confidence": 75.0 + offset,
                            "codebook_id": str((identity + task) % 2),
                            "allowed_mass": 0.9,
                            "full_vocab_argmax_allowed": True,
                            "conditional_entropy": 0.5,
                        }
                    )
    return rows


def test_known_interaction_and_paired_bootstrap() -> None:
    rows = _full_rows()
    baseline = arm_matrices(rows, "baseline", identity_count=16)
    candidate = arm_matrices(rows, "all16_deploy", identity_count=16)
    assert interaction(baseline)[0] == pytest.approx(-0.8)
    point, interval, draws = causal_delta_bootstrap(baseline, candidate, reps=100, seed=20)
    assert point == pytest.approx(0.5)
    assert interval == pytest.approx([0.5, 0.5])
    assert draws.shape == (100,)


def test_causal_analysis_includes_actor_and_twenty_nulls() -> None:
    report = analyze_causal_rows(
        _full_rows(),
        _null_rows(),
        pilot_task_ids=[f"task-{index:02d}" for index in range(5)],
        reps=100,
        seed=21,
    )
    assert set(report["arms"]) == set(FULL_ARMS[1:])
    assert report["arms"]["all16_deploy"]["delta_pp"] == pytest.approx(0.5)
    assert report["arms"]["all16_eval"]["delta_pp"] == pytest.approx(-0.3)
    assert report["signflip_control"]["candidate_exceeds_every_null"] is True
    assert report["signflip_control"]["add_one_empirical_p"] == pytest.approx(1 / 21)


def test_causal_analysis_is_row_order_invariant() -> None:
    kwargs = {
        "pilot_task_ids": [f"task-{index:02d}" for index in range(5)],
        "reps": 50,
        "seed": 22,
    }
    left = analyze_causal_rows(_full_rows(), _null_rows(), **kwargs)
    right = analyze_causal_rows(
        list(reversed(_full_rows())), list(reversed(_null_rows())), **kwargs
    )
    assert left["baseline_interaction_pp"] == right["baseline_interaction_pp"]
    assert left["signflip_control"] == right["signflip_control"]


def test_duplicate_full_key_fails_closed() -> None:
    rows = _full_rows()
    rows[-1] = dict(rows[0], arm=rows[-1]["arm"])
    with pytest.raises(ValueError, match="duplicate"):
        analyze_causal_rows(
            rows,
            _null_rows(),
            pilot_task_ids=[f"task-{index:02d}" for index in range(5)],
            reps=10,
            seed=23,
        )


def test_missing_actor_arm_fails_closed() -> None:
    rows = [row for row in _full_rows() if row["arm"] != "actor_deploy"]
    with pytest.raises(ValueError, match="arms"):
        analyze_causal_rows(
            rows,
            _null_rows(),
            pilot_task_ids=[f"task-{index:02d}" for index in range(5)],
            reps=10,
            seed=24,
        )


def test_independent_verifier_import_isolation() -> None:
    source = inspect.getsource(verification)
    assert "v20.analysis" not in source
    assert "v20.supervisor" not in source


def test_v20_has_no_explicit_status_gate_or_depth_mapped_schedule() -> None:
    source = inspect.getsource(supervisor.run_paid_ladder)
    assert "run_positive_control" not in source
    assert "depth_mapped" not in source
    assert '"local_proxy"' in source


def test_batch_selection_is_technical_only() -> None:
    source = inspect.getsource(supervisor.calibrate_batches)
    assert "expected_folded_confidence" not in source
    assert "group" not in source
    assert "prompts_per_second" in source


def test_runtime_freezes_literal_hua_schedule_and_budget() -> None:
    import yaml

    runtime = yaml.safe_load(
        (ROOT / "pipelines/glm53_user_eval/v20/configs/runtime_v20.yaml").read_text()
    )
    prereg = yaml.safe_load(PREREG.read_text())
    assert runtime["runpod"]["gpu_count"] == 2
    assert runtime["runpod"]["gpu_id"] == "NVIDIA B300 SXM6 AC"
    assert runtime["runpod"]["compute_hard_cap_usd"] == 55.0
    assert runtime["forward"]["candidate_batch_sizes"] == [1, 2, 4]
    assert prereg["intervention"]["layers"] == [10, 14, 18, 22, 26, 30]
    assert prereg["intervention"]["deployment_alpha"] == 0.6
    assert prereg["intervention"]["evaluation_alpha"] == -0.6


def test_loader_adapter_supplies_every_inherited_runtime_key_without_mutation() -> None:
    import copy

    import yaml

    runtime = yaml.safe_load(
        (ROOT / "pipelines/glm53_user_eval/v20/configs/runtime_v20.yaml").read_text()
    )
    original = copy.deepcopy(runtime)
    compatible = supervisor._loader_compatible_runtime_config(runtime)
    assert runtime == original
    assert compatible["runtime_checks"] == {
        "expected_cuda_devices": 2,
        "expected_gpu_name": "NVIDIA B300 SXM6 AC",
    }
    assert compatible["architecture"]["linear_attention_layers"] == 34
    assert compatible["architecture"]["sparse_attention_layers"] == 11
    assert (
        compatible["architecture"]["linear_attention_layers"]
        + compatible["architecture"]["sparse_attention_layers"]
        == compatible["architecture"]["text_layers"]
    )
    assert (
        2 * compatible["architecture"]["linear_attention_layers"]
        == compatible["architecture"]["forget_gate_scale_inv_tensors"]
    )
    for section, keys in {
        "runtime_checks": ("expected_cuda_devices", "expected_gpu_name"),
        "software": (
            "torch",
            "cuda",
            "transformers_commit",
            "transformers_source_sha256",
            "transformers_source_filename",
        ),
        "model": ("revision", "initialization_seed", "device_map", "low_cpu_mem_usage"),
        "architecture": (
            "text_layers",
            "linear_attention_layers",
            "sparse_attention_layers",
            "forget_gate_scale_inv_tensors",
        ),
        "rendering": ("reasoning_effort", "clear_thinking"),
    }.items():
        assert section in compatible
        assert all(key in compatible[section] for key in keys)


def test_baseline_is_reused_not_rescored_for_causal_stage() -> None:
    source = inspect.getsource(supervisor.run_paid_ladder)
    assert 'dict(row, arm="baseline"' in source
    assert "5 * len(selected) + 20 * 80" in source


def test_local_parity_precedes_all_target_interventions() -> None:
    source = inspect.getsource(supervisor.run_paid_ladder)
    parity_position = source.index("analyze_local_proxy")
    intervention_position = source.index("full_specs = {")
    assert parity_position < intervention_position
    assert "if not parity_passed:" in source[parity_position:intervention_position]


def test_paid_transport_scripts_share_the_exact_v20_contract() -> None:
    # V20 is immutable at its final runtime tag. The working-tree transport
    # entry points now carry V21's explicitly exploratory continuation.
    launcher = subprocess.check_output(
        [
            "git",
            "show",
            "glm53-user-eval-v20-runtime-r3:infra/runpod/new_glm53_v20_hua_pod.ps1",
        ],
        cwd=ROOT,
        text=True,
    )
    bootstrap = subprocess.check_output(
        [
            "git",
            "show",
            "glm53-user-eval-v20-runtime-r3:infra/runpod/bootstrap_glm53_v20.sh",
        ],
        cwd=ROOT,
        text=True,
    )
    for stale in (
        "$V15Decision",
        "glm53-user-eval-v20-preregistered-r8",
        "prereg_v20_lean_hua.yaml",
        "exceeds 150 minutes",
        'Decimal("32.00")',
    ):
        assert stale not in launcher
        assert stale not in bootstrap
    assert '"glm53-user-eval-v20-preregistered"' in launcher
    assert '"glm53-user-eval-v20-preregistered"' in bootstrap
    assert '"55.00"' in bootstrap
    assert 'Decimal("8.00")' in bootstrap
    assert "local_parity_then_direct_intervention" in bootstrap
    assert "run_positive_control" not in bootstrap

    launcher_bundle = launcher[
        launcher.index("function New-SignedInputBundle") : launcher.index(
            "$files = [ordered]@{}", launcher.index("function New-SignedInputBundle")
        )
    ]
    launcher_names = set(
        re.findall(r'^\s*"([^"]+)" = \[ordered\]@\{', launcher_bundle, re.MULTILINE)
    )
    bootstrap_targets = bootstrap[
        bootstrap.index("expected_targets = {") : bootstrap.index(
            "files = manifest.get", bootstrap.index("expected_targets = {")
        )
    ]
    bootstrap_names = set(
        re.findall(r'^\s*"([^"]+)":', bootstrap_targets, re.MULTILINE)
    )
    assert launcher_names == bootstrap_names
    assert len(launcher_names) == 25
    assert "runtime_supervisor_patch.py" in launcher_names
    assert "eb9dd825cdbba24217b5806e53cb8774c1023ce2" in launcher
    assert "59d3fba85ed10c02859246da40d1545866b6ff21bdd794306b700f26236a28e6" in launcher
    assert "eb9dd825cdbba24217b5806e53cb8774c1023ce2" in bootstrap
    assert "59d3fba85ed10c02859246da40d1545866b6ff21bdd794306b700f26236a28e6" in bootstrap


def test_transport_placeholders_are_freezable() -> None:
    for relative in (
        "infra/runpod/new_glm53_v20_hua_pod.ps1",
        "infra/runpod/bootstrap_glm53_v20.sh",
    ):
        text = (ROOT / relative).read_text()
        commit = re.search(r'(?:GLM53_PROJECT_COMMIT=|scientificCommit -ne )["\']([^"\']+)', text)
        archive = re.search(r'(?:GLM53_SOURCE_ARCHIVE_SHA256=|SourceArchiveSha256 = )["\']([^"\']+)', text)
        assert commit and (
            commit.group(1) == "__V20_PREREG_COMMIT__"
            or re.fullmatch(r"[0-9a-f]{40}", commit.group(1))
        )
        assert archive and (
            archive.group(1) == "__V20_SOURCE_ARCHIVE_SHA256__"
            or re.fullmatch(r"[0-9a-f]{64}", archive.group(1))
        )
