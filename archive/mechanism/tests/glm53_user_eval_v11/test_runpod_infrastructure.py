from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "infra/runpod/new_glm53_v11_source_pod.ps1"
BOOTSTRAP = ROOT / "infra/runpod/bootstrap_glm53_v11.sh"


def test_launcher_is_exact_bounded_no_volume_topology() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"NVIDIA B300 SXM6 AC"' in source
    assert "$ExpectedGpuCount = 2" in source
    assert 'cloudType = "SECURE"' in source
    assert '"--container-disk-in-gb", "450"' in source
    assert "$ComputeHardCapUsd = [decimal]29.50" in source
    assert "$MinimumReserveUsd = [decimal]15.00" in source
    assert "$StorageAllowanceUsd = [decimal]0.10" in source
    assert "$WallClockMinutes = 110" in source
    assert (
        "$availableComputeCap = $liveBalance - $MinimumReserveUsd - $StorageAllowanceUsd" in source
    )
    assert "GLM53_V11_DEADLINE_UTC" in source
    assert "GLM53_V11_AGGREGATE_RATE_USD" in source
    assert "GLM53_V11_BALANCE_FLOOR_USD" in source
    assert "--network-volume-id" not in source
    assert "DataCenterId" not in source


def test_launcher_fails_closed_before_paid_create() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    create_position = source.index("$podCreateBody = [ordered]@{")
    prefix = source[:create_position]
    assert "if (-not $ConfirmSpend)" in prefix
    assert "source_text_instrument_valid_for_activation_test" in prefix
    assert "glm53-user-eval-v11-preregistered" in prefix
    assert "git status --porcelain" in prefix
    assert "builder" in prefix and "spec" in prefix
    assert '@("pod", "list", "--all", "-o", "json")' in prefix
    assert '@("serverless", "list", "-o", "json")' in prefix
    assert '@("gpu", "list", "-o", "json")' in prefix and "clientBalance" in prefix
    assert "RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC" in prefix


def test_launcher_roundtrip_verifies_signed_input_bundle_before_create() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    create_position = source.index("$podCreateBody = [ordered]@{")
    bundle_position = source.index("$inputBundle = New-SignedInputBundle")
    assert bundle_position < create_position
    assert "glm53_v11_signed_input_manifest_v1" in source
    assert "HMACSHA256" in source
    assert "glm53-v11-input-manifest-v1" in source
    for name in (
        "text_decision.json",
        "samples.jsonl",
        "dataset_manifest.json",
        "tokenizer_audit.json",
        "structural_audit.json",
        "development_analysis.json",
        "final_text_analysis.json",
        "FINAL_TEXT_HOLDOUT_OPENED.json",
        "lexical_decision.json",
        "semantic_validation.json",
        "manual_audit.json",
        "offline_verification.json",
    ):
        assert name in source
    assert "round-trip hash mismatch" in source
    assert "GLM53_V11_INPUT_PREFIX" in source
    assert "s3_credential_read_write_probe_passed = $true" in source


def test_bootstrap_reconstructs_and_rehashes_all_text_decision_inputs() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    for field in (
        "prereg",
        "samples",
        "dataset_manifest",
        "tokenizer_audit",
        "builder",
        "spec",
        "runtime_config",
        "structural",
        "development",
        "final_text",
        "final_text_marker",
        "lexical_decision",
        "semantic",
        "manual",
        "verification",
    ):
        assert f'"{field}"' in source
    assert 'input_manifest.get("decision_inputs") != decision_inputs' in source
    assert "v11 reconstructed input differs from decision hash" in source


def test_s3_credential_lifecycle_is_attested_then_rotated_after_project() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    runtime = (ROOT / "pipelines/glm53_user_eval/v11/configs/runtime_v11.yaml").read_text(
        encoding="utf-8"
    )
    assert "CREDENTIAL_ISSUED" not in launcher + bootstrap
    assert "RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC" in launcher
    assert "RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC" in bootstrap
    assert "outside the allowed 24-hour window" in launcher
    assert "outside the 24-hour window" in bootstrap
    assert "s3_credential_rotation_required_after_project = $true" in launcher
    assert "credential_session_attestation_required: true" in runtime
    assert "credential_read_write_probe_required: true" in runtime
    assert "credential_rotation_required_after_terminal_cleanup: true" in runtime


def test_all_s3_curl_calls_supply_sigv4_credentials_without_literal_secrets() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    launcher_user = '--user "$($env:AWS_ACCESS_KEY_ID):$($env:AWS_SECRET_ACCESS_KEY)"'
    assert launcher.count("--aws-sigv4") == launcher.count(launcher_user)
    assert 'user = "${s3_access_key}:${s3_secret_key}"' in bootstrap
    assert 'aws-sigv4 = "aws:amz:${AWS_DEFAULT_REGION}:s3"' in bootstrap
    assert '--user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}"' not in bootstrap
    for source in (launcher, bootstrap):
        assert "AWS_ACCESS_KEY_ID=" not in source
        assert "AWS_SECRET_ACCESS_KEY=" not in source
        assert "Start-Transcript" not in source
        assert "set -x" not in source
        assert "Get-CimInstance Win32_Process" not in source
        assert "Get-Process" not in source
        assert "ps aux" not in source


def test_launcher_passes_transient_pod_environment_without_serializing_secrets() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    environment = source[
        source.index("$podEnvironment = [ordered]@{") : source.index("$podCreateBody = [ordered]@{")
    ]
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC",
        "GLM53_V11_RUN_SCIENCE",
    ):
        assert name in environment
    launch_start = source.index("$launchRecord = [ordered]@{")
    launch_record = source[launch_start : source.index("Write-AtomicJson -Path", launch_start)]
    assert "$podEnvironment" not in launch_record
    assert "AWS_ACCESS_KEY_ID" not in launch_record
    assert "AWS_SECRET_ACCESS_KEY" not in launch_record
    assert "Write-Output $podEnvironment" not in source
    assert '$podEnvironment["AWS_ACCESS_KEY_ID"] = ""' in source
    assert '$podEnvironment["AWS_SECRET_ACCESS_KEY"] = ""' in source
    assert "Invoke-RestMethod" in source
    assert '"--env"' not in source


def test_launcher_starts_exact_hash_verified_bootstrap_automatically() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    create_position = source.index("$podCreateBody = [ordered]@{")
    prefix = source[:create_position]
    assert '$BootstrapRelativePath = "infra/runpod/bootstrap_glm53_v11.sh"' in prefix
    assert "$bootstrapBase64" in prefix
    assert "$bootstrapSha256" in prefix
    assert "sha256sum --check --strict" in prefix
    assert '"--docker-entrypoint", "/bin/bash,-lc"' in source
    assert '"--docker-start-cmd", $dockerStartCommand' in source
    assert "templateId = $templateId" in source
    assert '"template", "delete", $templateId' in source
    assert 'bootstrap_start = "observed_bound_$bootstrapSignal"' in source
    assert "bootstrap_template_deleted = $true" in source
    template_block = source[
        source.index("$templateCreate = Invoke-RunPodRaw -Arguments @(") : create_position
    ]
    assert "AWS_ACCESS_KEY_ID" not in template_block
    assert "AWS_SECRET_ACCESS_KEY" not in template_block
    assert "RunPod injects RUNPOD_POD_ID" in source
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    requirement_position = bootstrap.index("RUNPOD_POD_ID \\")
    staging_position = bootstrap.index("stage_glm53_v8_hf_local.py")
    assert requirement_position < staging_position


def test_launcher_starts_external_heartbeat_watchdog() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Start-Process powershell.exe -WindowStyle Hidden" in source
    assert "$HeartbeatDeleteAfterSeconds = 600" in source
    assert '"missing_or_stale_heartbeat"' in source
    assert '"balance_floor"' in source
    assert '"terminal_marker"' in source
    assert 'Invoke-RunPodRaw -Arguments @("pod", "delete", $TargetPodId' in source
    assert "pod_absent_after_delete" in source
    assert "SetThreadExecutionState" in source
    assert "could not prevent workstation sleep" in source
    assert "invalid_terminal_binding" in source
    assert "heartbeat.pod_id -ne $PodId" in source
    assert "-PassThru" in source
    assert "watchdog did not produce a bound start handshake" in source
    assert "Pod did not emit a bound heartbeat or terminal marker" in source
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    assert '"pod_id": os.environ["RUNPOD_POD_ID"]' in bootstrap


def test_launcher_cleanup_is_verified_after_every_create_failure() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "function Get-OptionalProperty" in source
    assert "if ($null -eq $Pod) { return $null }" in source
    assert "Always issue DELETE by exact ID" in source
    assert '"pod", "list", "--all", "-o", "json"' in source
    assert "successful_list_checks" in source
    assert "Remove-PodsByNameVerified -ExactName $podName" in source
    assert "Remove-TemplatesByNameVerified -ExactName $templateName" in source
    assert '$podName = "mats-glm53-v11-$RunId"' in source
    assert "if (-not $launchComplete)" in source


def test_on_pod_deadline_and_terminal_paths_delete_the_exact_pod() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "start_deadline_guards" in source
    assert "hard_deadline_imminent" in source
    assert "delete_current_pod()" in source
    assert 'url = "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"' in source
    assert 'header = "Authorization: Bearer ${pod_api_key}"' in source
    assert "delete_current_pod_with_retry 120" in source
    assert "write_delete_failure_marker" in source
    assert 'wait "${deadline_delete_pid}"' in source
    assert "RUNPOD_API_KEY=" not in source
    on_exit = source[source.index("on_exit()") : source.index("trap on_exit EXIT")]
    assert on_exit.index("delete_current_pod_with_retry") < on_exit.index("stop_deadline_guards")


def test_bootstrap_limits_secret_inheritance_and_network_waits() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    capture = source.index('s3_access_key="${AWS_ACCESS_KEY_ID}"')
    unset = source.index("unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY RUNPOD_API_KEY")
    apt = source.index("apt-get update")
    assert capture < unset < apt
    assert 'user = "${s3_access_key}:${s3_secret_key}"' in source
    assert 'header = "Authorization: Bearer ${pod_api_key}"' in source
    assert "connect-timeout = 20" in source
    assert "max-time = ${s3_operation_max_seconds}" in source
    assert 'os.read(3, 1024 * 1024).rstrip(b"\\n")' in source


def test_bootstrap_uses_exact_xet_runtime_and_machine_gates() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'GLM53_PROJECT_REF="glm53-user-eval-v11-preregistered"' in source
    assert "HF_XET_HIGH_PERFORMANCE=1" in source
    assert "stage_glm53_v8_hf_local.py" in source
    assert "--workers 8" in source
    assert "torch==2.13.0 torchvision==0.28.0" in source
    assert "https://astral.sh/uv/0.9.26/install.sh" in source
    assert 'test "$(uv --version)" = "uv 0.9.26 (ee4f00362 2026-01-15)"' in source
    assert "expected two GPUs" in source
    assert '"B300" not in name' in source
    assert "vllm" not in source.lower()


def test_bootstrap_requires_gate_and_persistent_heartbeat() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "source_text_instrument_valid_for_activation_test" in source
    assert "RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC" in source
    assert "AWS_SECRET_ACCESS_KEY=" not in source
    assert "RUNPOD_API_KEY" in source
    assert 'url = "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"' in source
    assert 'header = "Authorization: Bearer ${pod_api_key}"' in source
    assert "delete_current_pod_with_retry 120" in source
    assert "heartbeat.json" in source
    assert "terminal.json" in source
    assert "sleep 60" in source
    assert "RUNPOD_POD_ID" in source
    assert "paid-ladder" in source
    assert "--confirm-spend" in source
    assert "--permutation-reps 1000 --confirm-spend" in source
    assert "source_evidence_manifest.json" in source
    assert "GLM53_V11_INPUT_PREFIX" in source
    assert "v11 S3 input-manifest HMAC verification failed" in source
    assert "v11 S3 input hash mismatch" in source
    assert source.index("input_manifest.hmac-sha256") < source.index("paid-ladder")
    runner = (ROOT / "pipelines/glm53_user_eval/v11/run.py").read_text(encoding="utf-8")
    assert "worker_candidates != [16, 32]" in runner
    assert '"benchmarks_count_toward_frozen_1000"' in runner
    assert '"full_ladder_fits_with_headroom"' in runner


def test_infrastructure_scripts_parse_without_execution() -> None:
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"[void][scriptblock]::Create((Get-Content -Raw -LiteralPath '{LAUNCHER}'))",
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        ["bash", "-n"],
        input=BOOTSTRAP.read_bytes(),
        check=True,
        cwd=ROOT,
    )
