from __future__ import annotations

import json

import yaml
from src.glm53_user_eval.v8.artifacts import sha256_file
from src.glm53_user_eval.v8.source_locks import verify_direction_dataset, verify_hashes

from .conftest import ROOT


def test_v7_tag_commit() -> None:
    import subprocess

    assert (
        subprocess.check_output(
            ["git", "rev-list", "-n", "1", "glm53-user-eval-v7-final"], cwd=ROOT, text=True
        ).strip()
        == "2b21609c67f9921fb36426cd95c2d0c2faec9c60"
    )


def test_parent_hashes() -> None:
    prereg = yaml.safe_load(
        (
            ROOT / "pipelines/glm53_user_eval/v8/configs/prereg_v8_whitebox_mechanism.yaml"
        ).read_text()
    )
    assert len(verify_hashes(ROOT, prereg["parent_result"]["artifact_hashes"])) == 9


def test_v7_decision_green_light() -> None:
    decision = json.loads(
        (
            ROOT / "artifacts/glm53_user_eval/reports/transluce_interaction_v7/decision.json"
        ).read_text()
    )
    assert decision["decision"] == "confirmed_target_sized_interaction"
    assert decision["whitebox_green_light"] is True


def test_model_stage_hash() -> None:
    assert sha256_file(ROOT / "artifacts/glm53_user_eval/runtime/g2/model_stage.json") == (
        "28d8a5842d8be519fb23d031ddd9b91d0a814b7e679b748f0ea7fd46f03434ca"
    )


def test_raw_scores_hash() -> None:
    assert (
        sha256_file(
            ROOT / "artifacts/glm53_user_eval/reports/transluce_interaction_v7/raw_scores.jsonl"
        )
        == "0694a3eda67de4f0bdb27e556877a38f7c742fcfb52dd42a811d6b1d9aa051ee"
    )


def test_direction_dataset_bytes_match_frozen_manifest() -> None:
    split_config = (
        ROOT / "pipelines/glm53_user_eval/v8/configs/direction_splits_v1.json"
    )
    observed = verify_direction_dataset(ROOT, split_config)
    assert set(observed) == {
        "artifacts/datasets/contrastive_prompts_v2/samples.csv",
        "artifacts/datasets/contrastive_prompts_v2/splits.csv",
        "artifacts/datasets/contrastive_prompts_v2/summary.json",
    }


def test_two_b300_v119_amendment_is_pre_launch_and_frozen() -> None:
    prereg = yaml.safe_load(
        (
            ROOT / "pipelines/glm53_user_eval/v8/configs/prereg_v8_whitebox_mechanism.yaml"
        ).read_text()
    )
    assert prereg["amendment"]["version"] == "two_b300_cuda13_text_runtime_v1_19"
    assert prereg["amendment"]["scientific_contract_changed"] is False
    assert prereg["amendment"]["completed_m2_reports_before_amendment"] == 0
    assert prereg["amendment"]["v8_scientific_rows_before_amendment"] == 0
    assert prereg["execution"]["prereg_tag"] == "glm53-user-eval-v8-preregistered-v1.19"
    assert prereg["amendment"]["resume_authorization"] == (
        "explicit_user_request_2026_08_31_four_h200_or_more"
    )
    idle_attempt = prereg["infrastructure"]["serverless"]["idle_probe_attempt"]
    assert sha256_file(ROOT / idle_attempt["path"]) == idle_attempt["sha256"]
    assert prereg["infrastructure"]["allowed_topologies"] == [
        {
            "gpu_id": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "gpu_count": 4,
            "aggregate_hourly_rate_usd": 8.36,
            "maximum_runtime_hours": 8.75,
        },
        {
            "gpu_id": "NVIDIA H100 NVL",
            "gpu_count": 4,
            "aggregate_hourly_rate_usd": 12.76,
            "maximum_runtime_hours": 7.1,
        },
        {
            "gpu_id": "NVIDIA H100 PCIe",
            "gpu_count": 5,
            "aggregate_hourly_rate_usd": 14.45,
            "maximum_runtime_hours": 6.0,
            "rationale": "native_h100_fp8_and_400_gb_aggregate_vram_in_same_volume_datacenter",
        },
        {
            "gpu_id": "NVIDIA H200",
            "gpu_count": 3,
            "aggregate_hourly_rate_usd": 13.77,
            "maximum_runtime_hours": 5.5,
            "network_volume_attached": False,
            "model_storage": "local_container_nvme",
            "device_map": "balanced_low_0",
            "max_memory_gib_per_gpu": [130, 134, 134],
            "rationale": "measured_caps_avoid_disk_offload_and_finalization_oom",
        },
        {
            "gpu_id": "NVIDIA H200",
            "gpu_count": 4,
            "aggregate_hourly_rate_usd": 18.36,
            "maximum_runtime_hours": 3.25,
            "network_volume_attached": False,
            "model_storage": "local_container_nvme",
            "model_transport": "huggingface_xet_public_exact_revision",
            "device_map": "balanced_low_0",
            "max_memory_gib_per_gpu": "auto_detected",
            "aggregate_vram_gb": 564,
            "rationale": "four_way_distribution_preserves_weight_and_forward_memory",
        },
        {
            "gpu_id": "NVIDIA B300 SXM6 AC",
            "gpu_count": 2,
            "aggregate_hourly_rate_usd": 15.78,
            "maximum_runtime_hours": 3.25,
            "network_volume_attached": False,
            "model_storage": "local_container_nvme",
            "model_transport": "huggingface_xet_public_exact_revision",
            "device_map": "balanced",
            "max_memory_gib_per_gpu": "auto_detected",
            "aggregate_vram_gb": 576,
            "rationale": "more_vram_than_four_h200_with_lower_live_aggregate_rate",
        },
    ]
    throttled_attempt = prereg["infrastructure"]["serverless"]["throttled_probe_attempt"]
    assert sha256_file(ROOT / throttled_attempt["path"]) == throttled_attempt["sha256"]
    serverless = prereg["infrastructure"]["serverless"]
    assert serverless["enabled_after_ordinary_pod_capacity_exhaustion"] is True
    assert serverless["worker_mode"] == "active_unmounted_pinned_worker"
    assert serverless["network_volume_attached"] is False
    assert serverless["artifact_transport"] == "runpod_s3_api"
    assert serverless["preflight_workers_min"] == 1
    assert serverless["execution_workers_min"] == 1
    assert serverless["workers_max"] == 1
    assert serverless["keep_same_worker_through_scientific_job"] is True
    assert serverless["scale_to_zero_between_probe_and_science"] is False
    assert serverless["endpoint_attempts_allowed"] == 1
    assert serverless["scientific_jobs_allowed"] == 1
    assert serverless["max_ready_wait_minutes"] == 90
    assert serverless["primary_topology"] == {
        "gpu_id": "NVIDIA H200",
        "gpu_count": 3,
        "planned_active_hourly_rate_usd": 13.392,
        "active_hourly_rate_cap_usd": 14.50,
        "maximum_runtime_hours": 6.0,
        "aggregate_vram_gb": 423,
        "live_rate_tolerance_usd": 0.05,
    }
    assert serverless["active_rate_probe"] == {
        "command": "rate_probe",
        "hold_seconds": 90,
        "model_forward_allowed": False,
        "scientific_row_allowed": False,
        "observation_timeout_seconds": 60,
        "completion_timeout_seconds": 180,
        "maximum_spend_usd": 0.50,
        "required_worker_identity_fields": ["pod_id", "gpu_names", "gpu_count"],
    }
    assert serverless["transport"] == {
        "protocol": "s3",
        "endpoint_url": "https://s3api-us-ks-2.runpod.io/",
        "bucket": "a9diryunoj",
        "input_prefix": "glm53-v8-input/v1.7",
        "result_prefix": "glm53-v8-results/v1.7",
        "credentials_from_environment_only": True,
        "preferred_template_id": "create_after_v1_7_tag",
        "bootstrap_image": (
            "runpod/pytorch@sha256:"
            "f40e33a190d6823439541d1dde52003fbed66539a7af998f38e29f499ca5bdd6"
        ),
        "avoid_ghcr_template_id": "a9fwi6am3u",
        "reason": "docker_hub_bootstrap_avoided_observed_ghcr_layer_throttling",
    }
    assert serverless["server_side_watchdog"] == {
        "execution_timeout_seconds": 21600,
        "hard_wall_clock_seconds": 21600,
        "execution_workers_min": 1,
    }


def test_v115_preserves_audit_enforcement_without_changing_primary_thresholds() -> None:
    prereg = yaml.safe_load(
        (
            ROOT / "pipelines/glm53_user_eval/v8/configs/prereg_v8_whitebox_mechanism.yaml"
        ).read_text()
    )
    assert prereg["intervention"]["pilot_alphas"] == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert prereg["intervention"]["confirmation_fraction_removed_min"] == 0.30
    assert prereg["intervention"]["confirmation_controls"] == 20
    assert prereg["manual_audit"]["required_before_final_m8_decision"] is True
    assert prereg["final_evidence"]["requires_terminal_machine_decision"] is True
    assert prereg["final_evidence"]["requires_passing_m8_decision"] is False


def test_v115_runtime_uses_two_b300s_with_detected_capacity() -> None:
    runtime = yaml.safe_load(
        (
            ROOT / "pipelines/glm53_user_eval/v8/configs/runtime_v8.yaml"
        ).read_text()
    )
    assert runtime["device_map"] == "balanced"
    assert runtime["max_memory_gib_per_gpu"] == "auto_detected"
    assert runtime["expected_cuda_devices"] == 2
    assert runtime["cuda_allocator_config"] == "expandable_segments:True"
    assert runtime["minimum_free_vram_fraction"] == 0.10
    assert runtime["batch_candidates"] == [1, 2, 4, 8]


def test_active_ordinary_pod_paths_require_the_v119_tag() -> None:
    tag = "glm53-user-eval-v8-preregistered-v1.19"
    for relative in (
        "infra/runpod/bootstrap_glm53_v8.sh",
        "infra/runpod/new_glm53_v8_pod.ps1",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert tag in text


def test_retired_serverless_paths_preserve_the_v17_tag() -> None:
    tag = "glm53-user-eval-v8-preregistered-v1.7"
    for relative in (
        ".github/workflows/build-glm53-v8-serverless.yml",
        "infra/runpod/new_glm53_v8_serverless.ps1",
        "infra/runpod/glm53_v8_serverless/Dockerfile",
        "infra/runpod/glm53_v8_serverless/handler.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert tag in text


def test_serverless_v17_has_no_volume_mount_and_keeps_the_worker() -> None:
    launcher = (ROOT / "infra/runpod/new_glm53_v8_serverless.ps1").read_text(
        encoding="utf-8"
    )
    handler = (ROOT / "infra/runpod/glm53_v8_serverless/handler.py").read_text(
        encoding="utf-8"
    )
    assert "--network-volume-id" not in launcher
    assert "--workers-min 1" in launcher
    assert "--workers-max 1" in launcher
    assert "--workers-min 0" not in launcher
    assert "expected_worker_id" in launcher
    assert "expected_worker_id" in handler
    assert 'EXPECTED_GPU_COUNTS = {3}' in handler
    assert 'EXPECTED_GPU_NAME = "NVIDIA H200"' in handler
    assert 'S3_INPUT_PREFIX = "glm53-v8-input/v1.7"' in handler
    assert 'S3_RESULT_PREFIX = "glm53-v8-results/v1.7"' in handler
