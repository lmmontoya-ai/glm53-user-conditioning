[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PodId,
    [Parameter(Mandatory = $true)][datetime]$DeadlineUtc,
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory,
    [Parameter(Mandatory = $true)][string]$RunPodCtl,
    [decimal]$BalanceFloorUsd = 25
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$evidence = [System.IO.Path]::GetFullPath($EvidenceDirectory)
if (-not (Test-Path -LiteralPath $evidence -PathType Container)) {
    throw "Evidence directory does not exist: $evidence"
}
@{
    schema_version = "glm53_v9_watchdog_v1"
    pod_id = $PodId
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    deadline_utc = $DeadlineUtc.ToUniversalTime().ToString("o")
    balance_floor_usd = $BalanceFloorUsd
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence "watchdog_start.json") -Encoding utf8

$reason = "deadline"
while ((Get-Date).ToUniversalTime() -lt $DeadlineUtc.ToUniversalTime()) {
    Start-Sleep -Seconds 30
    try {
        $account = (& $RunPodCtl user -o json | ConvertFrom-Json)
        if ([decimal]$account.clientBalance -le $BalanceFloorUsd) {
            $reason = "balance_floor"
            break
        }
    } catch {
        # A transient account query does not trigger early deletion.
    }
}
& $RunPodCtl pod get $PodId -o json 2>&1 |
    Set-Content -LiteralPath (Join-Path $evidence "watchdog_predelete.json") -Encoding utf8
& $RunPodCtl pod delete $PodId -o json |
    Set-Content -LiteralPath (Join-Path $evidence "watchdog_delete.json") -Encoding utf8
@{
    schema_version = "glm53_v9_watchdog_result_v1"
    pod_id = $PodId
    deleted_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    reason = $reason
    delete_exit_code = $LASTEXITCODE
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence "watchdog_result.json") -Encoding utf8

