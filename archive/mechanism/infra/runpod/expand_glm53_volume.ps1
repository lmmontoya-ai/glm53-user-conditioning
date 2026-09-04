[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$G0Decision,
    [string]$RunPodCtl = "$env:LOCALAPPDATA\Programs\runpodctl\runpodctl.exe",
    [string]$VolumeId = "a9diryunoj",
    [string]$DataCenterId = "US-KS-2",
    [int]$RequiredSizeGb = 500,
    [switch]$ConfirmIrreversibleExpansion
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ConfirmIrreversibleExpansion) {
    throw "Volume expansion cannot be reversed. Pass -ConfirmIrreversibleExpansion."
}
if (-not (Test-Path -LiteralPath $RunPodCtl -PathType Leaf)) {
    throw "runpodctl executable not found: $RunPodCtl"
}
if (-not (Test-Path -LiteralPath $G0Decision -PathType Leaf)) {
    throw "G0 decision file not found: $G0Decision"
}
$decision = Get-Content -LiteralPath $G0Decision -Raw | ConvertFrom-Json
if ($decision.gate -ne "G0" -or $decision.passed -ne $true) {
    throw "A passing local-first G0 decision is required."
}
$before = (& $RunPodCtl network-volume get $VolumeId -o json | ConvertFrom-Json)
if ($before.dataCenterId -ne $DataCenterId) {
    throw "Volume datacenter $($before.dataCenterId) differs from $DataCenterId."
}
if ([int]$before.size -gt $RequiredSizeGb) {
    throw "Volume is already larger than the preregistered size: $($before.size) GB."
}
if ([int]$before.size -lt $RequiredSizeGb) {
    & $RunPodCtl network-volume update $VolumeId --size $RequiredSizeGb -o json | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "runpodctl network-volume update failed with exit code $LASTEXITCODE."
    }
}
$after = (& $RunPodCtl network-volume get $VolumeId -o json | ConvertFrom-Json)
if ([int]$after.size -ne $RequiredSizeGb) {
    throw "Volume size is $($after.size) GB after update, expected $RequiredSizeGb GB."
}
[ordered]@{
    schema_version = "glm53_volume_expansion_v1"
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    g0_decision = (Resolve-Path -LiteralPath $G0Decision).Path
    volume_id = $VolumeId
    data_center_id = $DataCenterId
    size_before_gb = [int]$before.size
    size_after_gb = [int]$after.size
} | ConvertTo-Json -Depth 4
