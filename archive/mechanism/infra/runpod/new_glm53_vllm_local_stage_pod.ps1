[CmdletBinding()]
param(
    [decimal]$CommittedProjectSpendUsd = 26.24,
    [decimal]$AggregateHourlyRateUsd = 12.76,
    [decimal]$MaximumStartupHours = 0.5833334,
    [decimal]$ProjectHardCapUsd = 125,
    [string]$RunPodCtl = "$env:LOCALAPPDATA\Programs\runpodctl\runpodctl.exe",
    [string]$VolumeId = "a9diryunoj",
    [string]$DataCenterId = "US-KS-2",
    [switch]$ConfirmSpend
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$image = "ghcr.io/lmmontoya-ai/glm53-vllm-local-stage@sha256:8b17364b275452a42d2439d492c26194338dfd4690befcab04aaec35a5b03706"
$revision = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
$manifest = "/workspace/mats-glm53/artifacts/glm53_user_eval/runtime/g2/model_stage.json"
$local = "/runpod-local/GLM-5.3-Flash/$revision"
$projectedIncrement = $AggregateHourlyRateUsd * $MaximumStartupHours
$projectedTotal = $CommittedProjectSpendUsd + $projectedIncrement

if (-not $ConfirmSpend) {
    throw "Refusing paid launch without -ConfirmSpend."
}
if (-not (Test-Path -LiteralPath $RunPodCtl -PathType Leaf)) {
    throw "runpodctl executable not found: $RunPodCtl"
}
if ($projectedTotal -gt $ProjectHardCapUsd) {
    throw "Projected spend $projectedTotal USD exceeds the $ProjectHardCapUsd USD hard cap."
}

$volume = (& $RunPodCtl network-volume get $VolumeId -o json | ConvertFrom-Json)
if ($volume.dataCenterId -ne $DataCenterId -or [int]$volume.size -lt 500) {
    throw "The expected 500 GB US-KS-2 network volume is unavailable."
}

$environment = [ordered]@{
    MODEL_ID = "zai-org/GLM-5.3-Flash"
    MODEL_REVISION = $revision
    MODEL_STAGE_MANIFEST = $manifest
    MODEL_LOCAL = $local
    STAGE_WORKERS = "4"
} | ConvertTo-Json -Compress

$dockerArgs = @(
    "--served-model-name zai-org/GLM-5.3-Flash"
    "--tensor-parallel-size 4"
    "--max-model-len 16384"
    "--gpu-memory-utilization 0.90"
    "--reasoning-parser glm45"
    "--tool-call-parser glm47"
    "--enable-auto-tool-choice"
    "--host 0.0.0.0"
    "--port 8000"
) -join " "

[ordered]@{
    schema_version = "glm53_local_stage_launch_preflight_v1"
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    image = $image
    model_revision = $revision
    volume_id = $VolumeId
    data_center_id = $DataCenterId
    gpu_type = "NVIDIA H100 NVL"
    gpu_count = 4
    container_disk_gb = 450
    aggregate_hourly_rate_usd = $AggregateHourlyRateUsd
    maximum_startup_hours = $MaximumStartupHours
    committed_project_spend_usd = $CommittedProjectSpendUsd
    projected_increment_usd = $projectedIncrement
    projected_total_usd = $projectedTotal
    project_hard_cap_usd = $ProjectHardCapUsd
} | ConvertTo-Json -Depth 4 | Write-Output

& $RunPodCtl pod create `
    --name mats-glm53-vllm-local-stage `
    --gpu-id "NVIDIA H100 NVL" `
    --gpu-count 4 `
    --image $image `
    --cloud-type SECURE `
    --data-center-ids $DataCenterId `
    --network-volume-id $VolumeId `
    --volume-mount-path /workspace `
    --container-disk-in-gb 450 `
    --ports "8000/http" `
    --env $environment `
    --docker-args $dockerArgs `
    -o json

if ($LASTEXITCODE -ne 0) {
    throw "runpodctl pod create failed with exit code $LASTEXITCODE."
}
