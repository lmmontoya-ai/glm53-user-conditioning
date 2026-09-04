[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$M0Decision,
    [Parameter(Mandatory = $true)][string]$M1Contract,
    [Parameter(Mandatory = $true)][string]$Image,
    [decimal]$AggregateHourlyRateUsd = 15.78,
    [decimal]$MaximumRuntimeHours = 1.75,
    [decimal]$ProjectHardCapUsd = 30,
    [decimal]$RequiredReserveUsd = 25,
    [decimal]$StorageAllowanceUsd = 1,
    [string]$RunPodCtl = "$env:LOCALAPPDATA\Programs\runpodctl\runpodctl.exe",
    [string]$DataCenterId = "",
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
if ($m0.passed -ne $true) { throw "V9 preregistration validation has not passed." }
if ($m1.passed -ne $true) { throw "V9 tokenizer contract has not passed." }

$tag = "glm53-user-eval-v9-preregistered-v1.3"
$tagCommit = git rev-list -n 1 $tag
$headCommit = git rev-parse HEAD
if ($tagCommit -ne $headCommit) { throw "HEAD is not the v9 preregistration tag." }
if (git status --porcelain) { throw "Repository is dirty." }
if ($AggregateHourlyRateUsd -gt 15.78 -or $MaximumRuntimeHours -gt 1.75) {
    throw "Requested two-B300 rate or duration exceeds the preregistration."
}
$projected = $AggregateHourlyRateUsd * $MaximumRuntimeHours
if ($projected -gt $ProjectHardCapUsd) { throw "Projected spend exceeds the v9 cap." }
$account = (& $RunPodCtl user -o json | ConvertFrom-Json)
if (($projected + $RequiredReserveUsd + $StorageAllowanceUsd) -gt [decimal]$account.clientBalance) {
    throw "Projected spend, reserve, and storage exceed the live balance."
}

$createArguments = @(
    "pod", "create",
    "--name", "mats-glm53-v9-paper-probe-2xb300",
    "--gpu-id", "NVIDIA B300 SXM6 AC",
    "--gpu-count", "2",
    "--image", $Image,
    "--cloud-type", "SECURE",
    "--container-disk-in-gb", "450",
    "--ports", "22/tcp",
    "-o", "json"
)
if ($DataCenterId) { $createArguments += @("--data-center-ids", $DataCenterId) }
$created = (& $RunPodCtl @createArguments | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0 -or -not $created.id) { throw "RunPod Pod creation failed." }
$podId = [string]$created.id
if ([decimal]$created.costPerHr -gt $AggregateHourlyRateUsd) {
    & $RunPodCtl pod delete $podId | Out-Null
    throw "Created Pod rate exceeds the preregistered cap."
}
$evidence = Join-Path (Get-Location) "artifacts\glm53_user_eval\v9\infrastructure"
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
$manifest = [ordered]@{
    schema_version = "glm53_v9_runpod_launch_v1"
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    pod = $created
    git_commit = $headCommit
    prereg_tag = $tag
    live_balance_usd = [decimal]$account.clientBalance
    projected_compute_usd = $projected
    required_reserve_usd = $RequiredReserveUsd
    maximum_runtime_hours = $MaximumRuntimeHours
    network_volume_attached = $false
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $evidence "launch.json") -Encoding utf8
$deadline = (Get-Date).ToUniversalTime().AddHours([double]$MaximumRuntimeHours)
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
    (Resolve-Path "infra\runpod\watchdog_glm53_v9.ps1").Path,
    "-PodId", $podId,
    "-DeadlineUtc", $deadline.ToString("o"),
    "-EvidenceDirectory", (Resolve-Path $evidence).Path,
    "-RunPodCtl", $RunPodCtl,
    "-BalanceFloorUsd", [string]$RequiredReserveUsd
)
$created | ConvertTo-Json -Depth 8 | Write-Output
