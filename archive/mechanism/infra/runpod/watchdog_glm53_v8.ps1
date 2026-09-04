[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PodId,
    [Parameter(Mandatory = $true)][datetime]$DeadlineUtc,
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory,
    [Parameter(Mandatory = $true)][string]$RunPodCtl,
    [decimal]$BalanceFloorUsd = 15
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$evidence = [System.IO.Path]::GetFullPath($EvidenceDirectory)
if (-not (Test-Path -LiteralPath $evidence -PathType Container)) {
    throw "Evidence directory does not exist: $evidence"
}
$start = [ordered]@{
    schema_version = "glm53_v8_watchdog_v1"
    pod_id = $PodId
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    deadline_utc = $DeadlineUtc.ToUniversalTime().ToString("o")
    balance_floor_usd = $BalanceFloorUsd
}
$start | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence "watchdog_start.json") -Encoding utf8

$deleteReason = "deadline"
while ((Get-Date).ToUniversalTime() -lt $DeadlineUtc.ToUniversalTime()) {
    Start-Sleep -Seconds 60
    try {
        $account = (& $RunPodCtl user -o json | ConvertFrom-Json)
        if ($null -ne $account.clientBalance -and [decimal]$account.clientBalance -le $BalanceFloorUsd) {
            $deleteReason = "balance_floor"
            break
        }
    } catch {
        # A transient account query must not trigger resource deletion.
    }
}

$preDelete = & $RunPodCtl pod get $PodId -o json 2>&1
$preDelete | Set-Content -LiteralPath (Join-Path $evidence "watchdog_predelete.json") -Encoding utf8
& $RunPodCtl pod delete $PodId | Out-File -LiteralPath (Join-Path $evidence "watchdog_delete.txt") -Encoding utf8
$result = [ordered]@{
    schema_version = "glm53_v8_watchdog_result_v1"
    pod_id = $PodId
    deleted_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    delete_exit_code = $LASTEXITCODE
    delete_reason = $deleteReason
}
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence "watchdog_result.json") -Encoding utf8
