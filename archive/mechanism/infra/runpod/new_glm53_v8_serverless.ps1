[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$TemplateId,
    [Parameter(Mandatory = $true)][string]$Image,
    [string]$RunPodCtl = "$env:LOCALAPPDATA\Programs\runpodctl\runpodctl.exe",
    [string]$GpuId = "NVIDIA H200",
    [int]$GpuCount = 3,
    [decimal]$PlannedHourlyRateUsd = 13.392,
    [decimal]$HourlyRateCapUsd = 14.50,
    [decimal]$MaximumRuntimeHours = 6.0,
    [decimal]$ComputeHardCapUsd = 90,
    [decimal]$RequiredReserveUsd = 15,
    [decimal]$StorageAllowanceUsd = 2,
    [int]$ReadyWaitMinutes = 90,
    [switch]$ConfirmSpend
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $ConfirmSpend) { throw "Refusing Serverless deployment without -ConfirmSpend." }
if (-not (Test-Path -LiteralPath $RunPodCtl -PathType Leaf)) { throw "runpodctl missing" }
$tag = "glm53-user-eval-v8-preregistered-v1.7"
if ((git rev-list -n 1 $tag) -ne (git rev-parse HEAD)) { throw "HEAD is not $tag." }
if (git status --porcelain) { throw "Repository is dirty." }
if ($GpuId -ne "NVIDIA H200" -or $GpuCount -ne 3) {
    throw "Requested Serverless topology is not preregistered."
}
if ($PlannedHourlyRateUsd -ne [decimal]13.392 -or $HourlyRateCapUsd -ne [decimal]14.50) {
    throw "Requested Serverless rates are not preregistered."
}
if ($MaximumRuntimeHours -ne [decimal]6.0 -or $ComputeHardCapUsd -ne [decimal]90) {
    throw "Requested Serverless budget is not preregistered."
}
foreach ($name in @("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")) {
    if (-not [Environment]::GetEnvironmentVariable($name)) {
        throw "Required S3 credential is absent: $name"
    }
}

$account = (& $RunPodCtl user -o json | ConvertFrom-Json)
$baselineSpendRate = [decimal]$account.currentSpendPerHr
$projected = [math]::Min([double]$ComputeHardCapUsd, [double]($HourlyRateCapUsd * $MaximumRuntimeHours))
if (($projected + $RequiredReserveUsd + $StorageAllowanceUsd) -gt [decimal]$account.clientBalance) {
    throw "Projected Serverless spend, storage allowance, and reserve exceed the live balance."
}
$volume = (& $RunPodCtl network-volume get a9diryunoj -o json | ConvertFrom-Json)
if ($volume.dataCenterId -ne "US-KS-2" -or [int]$volume.size -lt 500) {
    throw "S3 result-volume contract failed."
}

$modelReference = "https://huggingface.co/zai-org/GLM-5.3-Flash:04c4e9e95c5da8862dced7e5056455116f83a7e0"
$created = (& $RunPodCtl serverless create `
    --template-id $TemplateId `
    --name mats-glm53-v8-h200-pinned `
    --gpu-id $GpuId `
    --gpu-count $GpuCount `
    --model-reference $modelReference `
    --workers-min 1 `
    --workers-max 1 `
    --idle-timeout 60 `
    --execution-timeout 21600 `
    --env "AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID" `
    --env "AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY" `
    -o json | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0 -or -not $created.id) { throw "Serverless endpoint creation failed." }
$endpointId = [string]$created.id

try {
    $readyDeadline = (Get-Date).ToUniversalTime().AddMinutes($ReadyWaitMinutes)
    $readyCount = 0
    do {
        $health = (& $RunPodCtl serverless health $endpointId -o json | ConvertFrom-Json)
        $workers = $health.workers
        $readyCount = [int]$workers.ready + [int]$workers.running
        if ($readyCount -gt 0) { break }
        Start-Sleep -Seconds 30
    } while ((Get-Date).ToUniversalTime() -lt $readyDeadline)
    if ($readyCount -le 0) {
        throw "The one permitted unmounted Serverless endpoint did not become ready in 90 minutes."
    }

    $probeInput = [ordered]@{
        command = "rate_probe"
        revision = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
        hold_seconds = 90
    } | ConvertTo-Json -Compress
    $probe = (& $RunPodCtl serverless run $endpointId --input $probeInput --no-wait -o json | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0 -or -not $probe.id) {
        throw "Non-scientific active-rate probe could not be submitted."
    }

    $rateDeadline = (Get-Date).ToUniversalTime().AddSeconds(90)
    $observedGpuRate = [decimal]0
    do {
        Start-Sleep -Seconds 5
        $activeAccount = (& $RunPodCtl user -o json | ConvertFrom-Json)
        $observedGpuRate = [decimal]$activeAccount.currentSpendPerHr - $baselineSpendRate
        $probeStatus = (& $RunPodCtl serverless status $endpointId ([string]$probe.id) -o json | ConvertFrom-Json)
        if ($probeStatus.status -in @("FAILED", "CANCELLED", "TIMED_OUT")) {
            throw "Non-scientific active-rate probe failed."
        }
    } while ($observedGpuRate -le 0 -and (Get-Date).ToUniversalTime() -lt $rateDeadline)
    if ($observedGpuRate -le 0 -or $observedGpuRate -gt $HourlyRateCapUsd) {
        throw "Observed Serverless GPU rate $observedGpuRate is absent or exceeds the preregistered cap."
    }

    $probeDeadline = (Get-Date).ToUniversalTime().AddSeconds(180)
    do {
        $probeStatus = (& $RunPodCtl serverless status $endpointId ([string]$probe.id) -o json | ConvertFrom-Json)
        if ($probeStatus.status -eq "COMPLETED") { break }
        if ($probeStatus.status -in @("FAILED", "CANCELLED", "TIMED_OUT")) {
            throw "Non-scientific active-rate probe failed."
        }
        Start-Sleep -Seconds 5
    } while ((Get-Date).ToUniversalTime() -lt $probeDeadline)
    if ($probeStatus.status -ne "COMPLETED") {
        throw "Non-scientific active-rate probe did not complete in time."
    }
    $workerId = [string]$probeStatus.output.worker_id
    if (-not $workerId) { throw "Rate probe did not return a worker identity." }

    $readyAccount = (& $RunPodCtl user -o json | ConvertFrom-Json)
    $remainingProjected = [math]::Min([double]$ComputeHardCapUsd, [double]($observedGpuRate * $MaximumRuntimeHours))
    if (($remainingProjected + $RequiredReserveUsd + $StorageAllowanceUsd) -gt [decimal]$readyAccount.clientBalance) {
        throw "The rate-probed balance no longer covers the run and reserve."
    }

    $scienceInput = [ordered]@{
        command = "supervise_v8"
        revision = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
        expected_worker_id = $workerId
        aggregate_hourly_rate_usd = [double]$observedGpuRate
        maximum_runtime_hours = [double]$MaximumRuntimeHours
    } | ConvertTo-Json -Compress
    $science = (& $RunPodCtl serverless run $endpointId --input $scienceInput --no-wait -o json | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0 -or -not $science.id) {
        throw "The one permitted scientific job could not be submitted."
    }

    $evidence = Join-Path (Get-Location) "artifacts\glm53_user_eval\v8\infrastructure"
    New-Item -ItemType Directory -Force -Path $evidence | Out-Null
    $manifest = [ordered]@{
        schema_version = "glm53_v8_serverless_endpoint_v2"
        created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        git_commit = git rev-parse HEAD
        prereg_tag = $tag
        endpoint_id = $endpointId
        template_id = $TemplateId
        image = $Image
        model_reference = $modelReference
        network_volume_attached = $false
        s3_result_volume_id = "a9diryunoj"
        gpu_id = $GpuId
        gpu_count = $GpuCount
        planned_hourly_rate_usd = $PlannedHourlyRateUsd
        hourly_rate_cap_usd = $HourlyRateCapUsd
        observed_gpu_hourly_rate_usd = $observedGpuRate
        maximum_runtime_hours = $MaximumRuntimeHours
        compute_hard_cap_usd = $ComputeHardCapUsd
        live_balance_usd = [decimal]$account.clientBalance
        ready_balance_usd = [decimal]$readyAccount.clientBalance
        baseline_spend_rate_usd = $baselineSpendRate
        rate_probe_job_id = [string]$probe.id
        rate_probe_status = [string]$probeStatus.status
        worker_id = $workerId
        scientific_job_id = [string]$science.id
        workers_min = 1
        workers_max = 1
        execution_timeout_seconds = 21600
        required_reserve_usd = $RequiredReserveUsd
        endpoint_created = $created
    }
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $evidence "serverless_endpoint.json") -Encoding utf8
    $manifest | ConvertTo-Json -Depth 12 | Write-Output
}
catch {
    & $RunPodCtl serverless delete $endpointId -o json | Out-Null
    throw
}
