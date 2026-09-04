[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$M0Decision,
    [Parameter(Mandatory = $true)][string]$M1Contract,
    [Parameter(Mandatory = $true)][string]$GpuId,
    [Parameter(Mandatory = $true)][string]$Image,
    [Parameter(Mandatory = $true)][decimal]$AggregateHourlyRateUsd,
    [decimal]$CommittedProjectSpendUsd = 0,
    [decimal]$MaximumRuntimeHours = 3.25,
    [decimal]$ProjectHardCapUsd = 60,
    [decimal]$RequiredReserveUsd = 15,
    [decimal]$StorageAllowanceUsd = 2,
    [string]$RunPodCtl = "$env:LOCALAPPDATA\Programs\runpodctl\runpodctl.exe",
    [string]$DataCenterId = "",
    [int]$GpuCount = 2,
    [switch]$ConfirmSpend
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $ConfirmSpend) { throw "Refusing paid launch without -ConfirmSpend." }
foreach ($path in @($M0Decision, $M1Contract, $RunPodCtl)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file missing: $path" }
}
$m0 = Get-Content -LiteralPath $M0Decision -Raw | ConvertFrom-Json
$m1 = Get-Content -LiteralPath $M1Contract -Raw | ConvertFrom-Json
if ($m0.gate -ne "M0" -or $m0.passed -ne $true) { throw "M0 has not passed." }
if ($m1.passed -ne $true) { throw "M1 has not passed." }

$tagCommit = git rev-list -n 1 glm53-user-eval-v8-preregistered-v1.19
$headCommit = git rev-parse HEAD
if ($tagCommit -ne $headCommit) { throw "HEAD is not the v8 preregistration tag." }
if (git status --porcelain) { throw "Repository is dirty." }

$allowedTopologies = @(
    @{ GpuId = "NVIDIA B300 SXM6 AC"; GpuCount = 2; MaximumRate = [decimal]15.78; MaximumHours = [decimal]3.25 }
)
$matchedTopology = $allowedTopologies | Where-Object {
    $_.GpuId -eq $GpuId -and $_.GpuCount -eq $GpuCount -and
    $AggregateHourlyRateUsd -le $_.MaximumRate -and
    $MaximumRuntimeHours -le $_.MaximumHours
}
if ($null -eq $matchedTopology) {
    throw "Requested GPU topology is not preregistered."
}

$projected = $CommittedProjectSpendUsd + $AggregateHourlyRateUsd * $MaximumRuntimeHours
if ($projected -gt $ProjectHardCapUsd) { throw "Projected spend $projected exceeds cap $ProjectHardCapUsd." }
$account = (& $RunPodCtl user -o json | ConvertFrom-Json)
if (($projected + $RequiredReserveUsd + $StorageAllowanceUsd) -gt [decimal]$account.clientBalance) {
    throw "Projected spend, reserve, and storage exceed the live RunPod balance."
}
$manifest = [ordered]@{
    schema_version = "glm53_v8_runpod_launch_v1"
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    git_commit = $headCommit
    prereg_tag = "glm53-user-eval-v8-preregistered-v1.19"
    network_volume_attached = $false
    model_storage = "local_container_nvme"
    data_center_id = $DataCenterId
    gpu_id = $GpuId
    gpu_count = $GpuCount
    image = $Image
    aggregate_hourly_rate_usd = $AggregateHourlyRateUsd
    maximum_runtime_hours = $MaximumRuntimeHours
    projected_total_usd = $projected
    live_balance_usd = [decimal]$account.clientBalance
    required_reserve_usd = $RequiredReserveUsd
    storage_allowance_usd = $StorageAllowanceUsd
    terminate_after_hours = 3.25
}
$manifest | ConvertTo-Json -Depth 4 | Write-Output

$createArguments = @(
    "pod", "create",
    "--name", "mats-glm53-v8-whitebox-2xb300",
    "--gpu-id", $GpuId,
    "--gpu-count", [string]$GpuCount,
    "--image", $Image,
    "--cloud-type", "SECURE",
    "--container-disk-in-gb", "450",
    "--ports", "22/tcp",
    "-o", "json"
)
if ($DataCenterId) {
    $createArguments += @("--data-center-ids", $DataCenterId)
}
$created = (& $RunPodCtl @createArguments | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw "runpodctl pod create failed: $LASTEXITCODE" }
$podId = [string]$created.id
if (-not $podId) { throw "Pod creation returned no Pod ID." }
if ([decimal]$created.costPerHr -gt $AggregateHourlyRateUsd) {
    & $RunPodCtl pod delete $podId | Out-Null
    throw "Created Pod rate $($created.costPerHr) exceeds $AggregateHourlyRateUsd."
}
$deadline = (Get-Date).ToUniversalTime().AddHours(3.25).ToString("o")
$evidence = Join-Path (Get-Location) "artifacts\glm53_user_eval\v8\infrastructure"
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
    (Resolve-Path "infra\runpod\watchdog_glm53_v8.ps1").Path,
    "-PodId", $podId, "-DeadlineUtc", $deadline,
    "-EvidenceDirectory", (Resolve-Path $evidence).Path,
    "-RunPodCtl", $RunPodCtl,
    "-BalanceFloorUsd", [string]$RequiredReserveUsd
)
$created | ConvertTo-Json -Depth 8 | Write-Output
