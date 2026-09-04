[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$G0Decision,

    [Parameter(Mandatory = $true)]
    [string]$GpuId,

    [Parameter(Mandatory = $true)]
    [string]$Image,

    [Parameter(Mandatory = $true)]
    [decimal]$AggregateHourlyRateUsd,

    [decimal]$CommittedProjectSpendUsd = 0,
    [decimal]$MaximumRuntimeHours = 2,
    [string]$RunPodCtl = "$env:LOCALAPPDATA\Programs\runpodctl\runpodctl.exe",
    [string]$VolumeId = "a9diryunoj",
    [string]$DataCenterId = "US-KS-2",
    [int]$RequiredVolumeSizeGb = 500,
    [int]$GpuCount = 4,
    [decimal]$ProjectHardCapUsd = 125,
    [switch]$ConfirmSpend
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ConfirmSpend) {
    throw "Refusing paid launch without -ConfirmSpend."
}
if (-not (Test-Path -LiteralPath $RunPodCtl -PathType Leaf)) {
    throw "runpodctl executable not found: $RunPodCtl"
}
if (-not (Test-Path -LiteralPath $G0Decision -PathType Leaf)) {
    throw "G0 decision file not found: $G0Decision"
}

$decision = Get-Content -LiteralPath $G0Decision -Raw | ConvertFrom-Json
if ($decision.gate -ne "G0" -or $decision.passed -ne $true) {
    throw "Local-first G0 has not passed. Paid white-box compute is locked."
}

$projectedIncrement = $AggregateHourlyRateUsd * $MaximumRuntimeHours
$projectedTotal = $CommittedProjectSpendUsd + $projectedIncrement
if ($projectedTotal -gt $ProjectHardCapUsd) {
    throw "Projected spend $projectedTotal USD exceeds the $ProjectHardCapUsd USD hard cap."
}

$volume = (& $RunPodCtl network-volume get $VolumeId -o json | ConvertFrom-Json)
if ($volume.dataCenterId -ne $DataCenterId) {
    throw "Volume datacenter $($volume.dataCenterId) differs from $DataCenterId."
}
if ([int]$volume.size -lt $RequiredVolumeSizeGb) {
    throw "Volume is $($volume.size) GB. Expand it to $RequiredVolumeSizeGb GB after local-first G0."
}

$event = [ordered]@{
    schema_version = "glm53_runpod_launch_preflight_v1"
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    g0_decision = (Resolve-Path -LiteralPath $G0Decision).Path
    g0_passed = $true
    volume_id = $VolumeId
    data_center_id = $DataCenterId
    volume_size_gb = [int]$volume.size
    gpu_id = $GpuId
    gpu_count = $GpuCount
    image = $Image
    aggregate_hourly_rate_usd = $AggregateHourlyRateUsd
    maximum_runtime_hours = $MaximumRuntimeHours
    committed_project_spend_usd = $CommittedProjectSpendUsd
    projected_increment_usd = $projectedIncrement
    projected_total_usd = $projectedTotal
    project_hard_cap_usd = $ProjectHardCapUsd
}
$event | ConvertTo-Json -Depth 4 | Write-Output

& $RunPodCtl pod create `
    --name mats-glm53-whitebox `
    --gpu-id $GpuId `
    --gpu-count $GpuCount `
    --image $Image `
    --cloud-type SECURE `
    --data-center-ids $DataCenterId `
    --network-volume-id $VolumeId `
    --volume-mount-path /workspace `
    --container-disk-in-gb 80 `
    --ports "8888/http,22/tcp" `
    --wait `
    --wait-timeout 15m `
    -o json

if ($LASTEXITCODE -ne 0) {
    throw "runpodctl pod create failed with exit code $LASTEXITCODE."
}
