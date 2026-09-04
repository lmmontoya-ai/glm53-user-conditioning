param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,

    [Parameter(Mandatory = $true)]
    [string]$CredentialFile,

    [int]$ExistingDiscoveryProcessId = 0,
    [ValidateRange(1, 96)]
    [int]$Concurrency = 96
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Import-OpenRouterCredential {
    param([string]$Path)
    $line = Get-Content -LiteralPath $Path |
        Where-Object { $_ -match '^OPENROUTER_API_KEY\s*=' } |
        Select-Object -First 1
    if (-not $line) {
        throw "OPENROUTER_API_KEY is absent from the supplied credential file"
    }
    $value = (($line -split '=', 2)[1]).Trim().Trim('"').Trim("'")
    if ($value.Length -lt 20) {
        throw "OPENROUTER_API_KEY is invalid"
    }
    $env:OPENROUTER_API_KEY = $value
}

function Get-ResultCount {
    param([string]$RunRoot)
    $path = Join-Path $RunRoot "results.jsonl"
    if (-not (Test-Path -LiteralPath $path)) {
        return 0
    }
    return (Get-Content -LiteralPath $path).Count
}

function Invoke-RosterStage {
    param(
        [string]$ScheduleRoot,
        [string]$RunRoot,
        [string]$RunId
    )
    $attempt = 0
    while ((Get-ResultCount -RunRoot $RunRoot) -lt 7000) {
        $attempt += 1
        if ($attempt -gt 3) {
            throw "$RunId did not complete after three resumable process attempts"
        }
        & uv run python pipelines/glm53_user_eval/run.py behavior-api `
            --prereg pipelines/glm53_user_eval/configs/prereg_v5_roster.yaml `
            --behavior-config pipelines/glm53_user_eval/configs/behavior_v4.yaml `
            --schedule-root $ScheduleRoot `
            --output $RunRoot `
            --run-id $RunId `
            --concurrency $Concurrency `
            --max-samples 7000
        if ($LASTEXITCODE -ne 0 -and (Get-ResultCount -RunRoot $RunRoot) -ge 7000) {
            throw "$RunId returned a failure after all rows were present"
        }
    }
}

function Assert-Route {
    param(
        [string]$RunRoot,
        [string]$OutputName
    )
    & uv run python pipelines/glm53_user_eval/run.py validate-api-route `
        --prereg pipelines/glm53_user_eval/configs/prereg_v5_roster.yaml `
        --results (Join-Path $RunRoot "results.jsonl") `
        --output (Join-Path $RunRoot $OutputName)
    if ($LASTEXITCODE -ne 0) {
        throw "provider route audit failed for $RunRoot"
    }
}

Set-Location -LiteralPath $RepositoryRoot
Import-OpenRouterCredential -Path $CredentialFile

$discoveryRoot = Join-Path $RepositoryRoot "artifacts/glm53_user_eval/behavior_api/roster_v5_discovery"
$confirmationRoot = Join-Path $RepositoryRoot "artifacts/glm53_user_eval/behavior_api/roster_v5_confirmation"
$discoverySchedule = Join-Path $RepositoryRoot "artifacts/glm53_user_eval/behavior_api/roster_v5_discovery_schedule"
$confirmationSchedule = Join-Path $RepositoryRoot "artifacts/glm53_user_eval/behavior_api/roster_v5_confirmation_schedule"

if ($ExistingDiscoveryProcessId -gt 0) {
    while (Get-Process -Id $ExistingDiscoveryProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
}

Invoke-RosterStage `
    -ScheduleRoot $discoverySchedule `
    -RunRoot $discoveryRoot `
    -RunId "glm53-g3-roster-v5-discovery"
Assert-Route -RunRoot $discoveryRoot -OutputName "route_validation_7000.json"

Invoke-RosterStage `
    -ScheduleRoot $confirmationSchedule `
    -RunRoot $confirmationRoot `
    -RunId "glm53-g3-roster-v5-confirmation"
Assert-Route -RunRoot $confirmationRoot -OutputName "route_validation_7000.json"

$summary = [ordered]@{
    schema_version = "glm53_roster_unattended_summary_v1"
    discovery_rows = Get-ResultCount -RunRoot $discoveryRoot
    confirmation_rows = Get-ResultCount -RunRoot $confirmationRoot
    discovery_route_audit = "route_validation_7000.json"
    confirmation_route_audit = "route_validation_7000.json"
    effects_inspected = $false
    completed_at_utc = [DateTime]::UtcNow.ToString("o")
}
$summary |
    ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $RepositoryRoot "artifacts/glm53_user_eval/behavior_api/roster_v5_unattended_summary.json") -Encoding utf8
