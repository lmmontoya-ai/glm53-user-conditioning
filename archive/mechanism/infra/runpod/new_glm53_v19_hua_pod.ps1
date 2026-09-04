[CmdletBinding()]
param(
    [ValidateSet("Launch", "Watchdog")][string]$Mode = "Launch",
    [string]$V15Decision = "artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json",
    [string]$TranslucePersonas = "..\reference\transluce-user-awareness\core\personas2.json",
    [string]$RuntimeConfig = "pipelines/glm53_user_eval/v19/configs/runtime_v19.yaml",
    [string]$Image = "runpod/pytorch@sha256:f40e33a190d6823439541d1dde52003fbed66539a7af998f38e29f499ca5bdd6",
    [string]$RunPodCtl = "$env:LOCALAPPDATA\Programs\runpodctl\runpodctl.exe",
    [string]$EvidenceDirectory = "artifacts/glm53_user_eval/v19/infrastructure",
    [switch]$ConfirmSpend,
    [string]$PodId = "",
    [string]$RunId = "",
    [datetime]$DeadlineUtc,
    [decimal]$BalanceFloorUsd = 0,
    [string]$S3Endpoint = "https://s3api-us-ks-2.runpod.io/",
    [Parameter(Mandatory)][string]$S3Bucket,
    [string]$S3Prefix = "glm53-v19-results",
    [string]$S3InputPrefix = "glm53-v19-inputs"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PreregTag = "glm53-user-eval-v19-preregistered-r8"
$InfrastructureAmendmentTag = "glm53-user-eval-v19-runtime-v16"
$ExpectedDecision = "fresh_control_bank_validated_by_both_codex_judges"
$ExpectedGpuId = "NVIDIA B300 SXM6 AC"
$ExpectedGpuCount = 2
$ExpectedImage = "runpod/pytorch@sha256:f40e33a190d6823439541d1dde52003fbed66539a7af998f38e29f499ca5bdd6"
$RateCapUsdPerHour = [decimal]16.00
$ComputeHardCapUsd = [decimal]39.50
$MinimumReserveUsd = [decimal]32.00
$StorageAllowanceUsd = [decimal]0.10
$WallClockMinutes = 150
$HeartbeatDeleteAfterSeconds = 600
$S3Region = "US-KS-2"
$BootstrapRelativePath = "infra/runpod/bootstrap_glm53_v19.sh"
$SourceArchive = "artifacts/glm53_user_eval/v19/infrastructure/source_transport/v19_science_repo_r8.tar.gz"
$SourceArchiveSha256 = "f60cc5b3b7ef4ae91a927445dd6730a5ec3bf1287f182951fdf550b8a97d105b"

function Write-AtomicJson {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)]$Value)
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $temporary = "$Path.partial"
    $Value | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-FileSha256 {
    param([Parameter(Mandatory)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-LiteralSha256 {
    param($Value)
    return $Value -is [string] -and $Value -match '^[0-9a-f]{64}$'
}

function Get-OptionalProperty {
    param(
        $Value,
        [Parameter(Mandatory)][string]$Name,
        $Default = $null
    )
    if ($null -eq $Value) { return $Default }
    $property = $Value.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) { return $Default }
    return $property.Value
}

function Invoke-RunPodRaw {
    param([Parameter(Mandatory)][string[]]$Arguments)
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $RunPodCtl
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [Text.Encoding]::UTF8
    foreach ($argument in $Arguments) {
        $startInfo.ArgumentList.Add($argument)
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "runpodctl failed to start."
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $combined = @($stdout.Trim(), $stderr.Trim()) | Where-Object { $_ }
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Text = ($combined -join [Environment]::NewLine).Trim()
    }
}

function Invoke-RunPodJson {
    param([Parameter(Mandatory)][string[]]$Arguments)
    $result = Invoke-RunPodRaw -Arguments $Arguments
    if ($result.ExitCode -ne 0) {
        throw "runpodctl failed with exit code $($result.ExitCode): $($Arguments -join ' ')"
    }
    if (-not $result.Text) {
        throw "runpodctl returned no JSON: $($Arguments -join ' ')"
    }
    try {
        return $result.Text | ConvertFrom-Json
    } catch {
        throw "runpodctl returned malformed JSON: $($Arguments -join ' ')"
    }
}

function Get-RunPodRecords {
    param(
        $Value,
        [string[]]$ContainerNames = @()
    )
    if ($null -eq $Value) { return @() }
    foreach ($name in $ContainerNames) {
        $property = $Value.PSObject.Properties[$name]
        if ($null -ne $property) { return @($property.Value) }
    }
    return @($Value)
}

function Get-LocalRunPodApiKey {
    $fromEnvironment = [Environment]::GetEnvironmentVariable("RUNPOD_API_KEY")
    if ($fromEnvironment) { return $fromEnvironment }
    $configPath = Join-Path $HOME ".runpod\config.toml"
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "RunPod API credentials are not configured."
    }
    $configText = Get-Content -LiteralPath $configPath -Raw
    $matched = [regex]::Match(
        $configText,
        '(?m)^\s*apikey\s*=\s*["''](?<key>[^"'']+)["'']\s*$'
    )
    if (-not $matched.Success -or -not $matched.Groups["key"].Value) {
        throw "RunPod API credentials could not be read from the local CLI config."
    }
    return $matched.Groups["key"].Value
}

function Assert-S3CredentialSession {
    foreach ($name in @("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC")) {
        if (-not [Environment]::GetEnvironmentVariable($name)) {
            throw "Required dedicated S3 credential metadata is absent: $name"
        }
    }
    $attestedText = [Environment]::GetEnvironmentVariable("RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC")
    $attested = [datetime]::MinValue
    if (-not [datetime]::TryParse(
        $attestedText,
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::AssumeUniversal,
        [ref]$attested
    )) {
        throw "RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC is not a valid UTC timestamp."
    }
    $age = (Get-Date).ToUniversalTime() - $attested.ToUniversalTime()
    if ($age.TotalMinutes -lt -5 -or $age.TotalHours -gt 24) {
        throw "The S3 credential session attestation is outside the allowed 24-hour window."
    }
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
        throw "curl.exe is required for a credential-only S3 read probe."
    }
    $probeUrl = "$($S3Endpoint.TrimEnd('/'))/${S3Bucket}?list-type=2&max-keys=0"
    & curl.exe --silent --show-error --fail `
        --connect-timeout 20 --max-time 120 --retry 3 --retry-all-errors `
        --user "$($env:AWS_ACCESS_KEY_ID):$($env:AWS_SECRET_ACCESS_KEY)" `
        --aws-sigv4 "aws:amz:${S3Region}:s3" `
        --output NUL $probeUrl
    if ($LASTEXITCODE -ne 0) {
        throw "The dedicated S3 credentials failed the read-only bucket probe."
    }
    return $attested.ToUniversalTime()
}

function Get-S3Json {
    param([Parameter(Mandatory)][string]$ObjectKey)
    $temporary = Join-Path $EvidenceDirectory ".s3-watchdog-$PID-$([guid]::NewGuid().ToString('N')).json"
    $url = "$($S3Endpoint.TrimEnd('/'))/$S3Bucket/$ObjectKey"
    try {
        & curl.exe --silent --show-error --fail `
            --connect-timeout 10 --max-time 30 --retry 2 --retry-all-errors `
            --user "$($env:AWS_ACCESS_KEY_ID):$($env:AWS_SECRET_ACCESS_KEY)" `
            --aws-sigv4 "aws:amz:${S3Region}:s3" `
            --output $temporary $url 2>$null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $temporary -PathType Leaf)) {
            return $null
        }
        return Get-Content -LiteralPath $temporary -Raw | ConvertFrom-Json
    } catch {
        return $null
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Send-S3File {
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$ObjectKey
    )
    $url = "$($S3Endpoint.TrimEnd('/'))/$S3Bucket/$ObjectKey"
    & curl.exe --silent --show-error --fail `
        --connect-timeout 20 --max-time 900 --retry 3 --retry-all-errors `
        --user "$($env:AWS_ACCESS_KEY_ID):$($env:AWS_SECRET_ACCESS_KEY)" `
        --aws-sigv4 "aws:amz:${S3Region}:s3" `
        --upload-file $Source $url
    if ($LASTEXITCODE -ne 0) { throw "S3 upload failed for $ObjectKey." }
}

function Receive-S3File {
    param(
        [Parameter(Mandatory)][string]$ObjectKey,
        [Parameter(Mandatory)][string]$Destination
    )
    $url = "$($S3Endpoint.TrimEnd('/'))/$S3Bucket/$ObjectKey"
    & curl.exe --silent --show-error --fail `
        --connect-timeout 20 --max-time 900 --retry 3 --retry-all-errors `
        --user "$($env:AWS_ACCESS_KEY_ID):$($env:AWS_SECRET_ACCESS_KEY)" `
        --aws-sigv4 "aws:amz:${S3Region}:s3" `
        --output $Destination $url
    if ($LASTEXITCODE -ne 0) { throw "S3 download failed for $ObjectKey." }
}

function New-SignedInputBundle {
    param(
        [Parameter(Mandatory)][string]$BundleDirectory,
        [Parameter(Mandatory)][string]$ObjectPrefix,
        [Parameter(Mandatory)][string]$GitCommit,
        [Parameter(Mandatory)][string]$CurrentRunId,
        [Parameter(Mandatory)]$Decision
    )
    New-Item -ItemType Directory -Force -Path $BundleDirectory | Out-Null
    $inputs = [ordered]@{
        "downstream_preflight.json" = [ordered]@{
            source = "artifacts/glm53_user_eval/v11/downstream_inputs/preflight.json"
            target = "artifacts/glm53_user_eval/v11/downstream_inputs/preflight.json"
        }
        "v7_transcripts_all100.jsonl" = [ordered]@{
            source = "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl"
            target = "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl"
        }
        "v7_transcripts_all100_manifest.json" = [ordered]@{
            source = "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100_manifest.json"
            target = "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100_manifest.json"
        }
        "transluce_personas2.json" = [ordered]@{
            source = $TranslucePersonas
            target = "artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json"
        }
        "v15_parent_decision.json" = [ordered]@{
            source = "artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json"
            target = "artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json"
        }
        "v15_parent_verification.json" = [ordered]@{
            source = "artifacts/glm53_user_eval/v15/reports/codex_cohort/verification.json"
            target = "artifacts/glm53_user_eval/v15/reports/codex_cohort/verification.json"
        }
        "immutable_v7_decision.json" = [ordered]@{
            source = "artifacts/glm53_user_eval/reports/transluce_interaction_v7/decision.json"
            target = "artifacts/glm53_user_eval/reports/transluce_interaction_v7/decision.json"
        }
        "immutable_v7_analysis.json" = [ordered]@{
            source = "artifacts/glm53_user_eval/reports/transluce_interaction_v7/analysis.json"
            target = "artifacts/glm53_user_eval/reports/transluce_interaction_v7/analysis.json"
        }
        "immutable_v19_prereg.yaml" = [ordered]@{
            source = "pipelines/glm53_user_eval/v19/configs/prereg_v19_lean_hua.yaml"
            target = "pipelines/glm53_user_eval/v19/configs/prereg_v19_lean_hua.yaml"
        }
        "immutable_v18_decision.json" = [ordered]@{
            source = "artifacts/glm53_user_eval/v18/reports/decision.json"
            target = "artifacts/glm53_user_eval/v18/reports/decision.json"
        }
        "immutable_hua16.yaml" = [ordered]@{
            source = "manifests/task_sources/contrastive_prompts_v2/hua16_exact_raw_v1.yaml"
            target = "manifests/task_sources/contrastive_prompts_v2/hua16_exact_raw_v1.yaml"
        }
        "immutable_v15_dataset.jsonl" = [ordered]@{
            source = "artifacts/datasets/contrastive_prompts_v5/samples.jsonl"
            target = "artifacts/datasets/contrastive_prompts_v5/samples.jsonl"
        }
        "immutable_proxy_token_contract.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v11/configs/proxy_token_contract_v2.json"
            target = "pipelines/glm53_user_eval/v11/configs/proxy_token_contract_v2.json"
        }
        "immutable_proxy_codebooks.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v11/configs/proxy_codebooks_v2.json"
            target = "pipelines/glm53_user_eval/v11/configs/proxy_codebooks_v2.json"
        }
        "immutable_downstream_manifest.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v17/configs/downstream_manifest_v17.json"
            target = "pipelines/glm53_user_eval/v17/configs/downstream_manifest_v17.json"
        }
        "immutable_formality_pairs.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v17/configs/formality_pairs_v1.json"
            target = "pipelines/glm53_user_eval/v17/configs/formality_pairs_v1.json"
        }
        "immutable_neutral_damage_prompts.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v17/configs/neutral_damage_prompts_v1.json"
            target = "pipelines/glm53_user_eval/v17/configs/neutral_damage_prompts_v1.json"
        }
        "immutable_positive_control_manifest.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v17/configs/positive_control_manifest_v1.json"
            target = "pipelines/glm53_user_eval/v17/configs/positive_control_manifest_v1.json"
        }
        "immutable_positive_control_selection.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v19/configs/positive_control_selection_v19.json"
            target = "pipelines/glm53_user_eval/v19/configs/positive_control_selection_v19.json"
        }
        "immutable_causal_schedule.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v8/configs/causal_schedule_v1.json"
            target = "pipelines/glm53_user_eval/v8/configs/causal_schedule_v1.json"
        }
        # These two text assets are hash-bound to their Windows worktree bytes.
        # `git archive` stores the normalized LF blobs, so restore the frozen
        # byte-exact copies through the signed input bundle before validation.
        "immutable_user_prompt_templates.jsonl" = [ordered]@{
            source = "pipelines/glm53_user_eval/v8/configs/user_prompt_templates_v1.jsonl"
            target = "pipelines/glm53_user_eval/v8/configs/user_prompt_templates_v1.jsonl"
        }
        "immutable_identity_selection.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/configs/identity_selection_v1.json"
            target = "pipelines/glm53_user_eval/configs/identity_selection_v1.json"
        }
        "immutable_v19_design.json" = [ordered]@{
            source = "pipelines/glm53_user_eval/v19/configs/design_v19.json"
            target = "pipelines/glm53_user_eval/v19/configs/design_v19.json"
        }
        "immutable_v19_runtime.yaml" = [ordered]@{
            source = "pipelines/glm53_user_eval/v19/configs/runtime_v19.yaml"
            target = "pipelines/glm53_user_eval/v19/configs/runtime_v19.yaml"
        }
        "immutable_model_stage.json" = [ordered]@{
            source = "artifacts/glm53_user_eval/runtime/g2/model_stage.json"
            target = "artifacts/glm53_user_eval/runtime/g2/model_stage.json"
        }
        "immutable_transformers_source.tar.gz" = [ordered]@{
            source = "artifacts/glm53_user_eval/v17/infrastructure/transformers_805a9e939fa8c1bff8d8ffdf041c051b71a914aa.tar.gz"
            target = "artifacts/glm53_user_eval/v17/infrastructure/transformers_805a9e939fa8c1bff8d8ffdf041c051b71a914aa.tar.gz"
        }
    }
    $files = [ordered]@{}
    foreach ($entry in $inputs.GetEnumerator()) {
        $source = [System.IO.Path]::GetFullPath([string]$entry.Value.source)
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Input-bundle source is absent: $source"
        }
        $files[$entry.Key] = [ordered]@{
            object_name = $entry.Key
            target_relative_path = [string]$entry.Value.target
            sha256 = Get-FileSha256 -Path $source
            bytes = (Get-Item -LiteralPath $source).Length
        }
    }
    $manifestPath = Join-Path $BundleDirectory "input_manifest.json"
    $manifest = [ordered]@{
        schema_version = "glm53_v19_signed_input_manifest_v1"
        project_id = "glm53_user_eval_hua_causal_v19"
        run_id = $CurrentRunId
        created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        prereg_tag = $PreregTag
        git_commit = $GitCommit
        object_prefix = $ObjectPrefix
        parent_decision = [ordered]@{
            decision = $Decision.decision
            sha256 = Get-FileSha256 -Path $V15Decision
        }
        files = $files
    }
    Write-AtomicJson -Path $manifestPath -Value $manifest

    $secret = [Environment]::GetEnvironmentVariable("AWS_SECRET_ACCESS_KEY")
    $secretBytes = [Text.Encoding]::UTF8.GetBytes($secret)
    $domain = [Text.Encoding]::UTF8.GetBytes("glm53-v19-input-manifest-v1`0")
    $manifestBytes = [IO.File]::ReadAllBytes($manifestPath)
    $message = New-Object byte[] ($domain.Length + $manifestBytes.Length)
    [Array]::Copy($domain, 0, $message, 0, $domain.Length)
    [Array]::Copy($manifestBytes, 0, $message, $domain.Length, $manifestBytes.Length)
    try {
        $hmac = [Security.Cryptography.HMACSHA256]::new($secretBytes)
        try {
            $signature = ([BitConverter]::ToString($hmac.ComputeHash($message))).Replace("-", "").ToLowerInvariant()
        } finally {
            $hmac.Dispose()
        }
    } finally {
        [Array]::Clear($secretBytes, 0, $secretBytes.Length)
        $secret = $null
    }
    $signaturePath = Join-Path $BundleDirectory "input_manifest.hmac-sha256"
    [IO.File]::WriteAllText($signaturePath, "$signature`n", [Text.UTF8Encoding]::new($false))

    foreach ($entry in $inputs.GetEnumerator()) {
        Send-S3File -Source ([System.IO.Path]::GetFullPath([string]$entry.Value.source)) `
            -ObjectKey "$ObjectPrefix/$($entry.Key)"
    }
    Send-S3File -Source $manifestPath -ObjectKey "$ObjectPrefix/input_manifest.json"
    Send-S3File -Source $signaturePath -ObjectKey "$ObjectPrefix/input_manifest.hmac-sha256"

    $roundtrip = Join-Path $BundleDirectory "roundtrip"
    New-Item -ItemType Directory -Force -Path $roundtrip | Out-Null
    foreach ($entry in $inputs.GetEnumerator()) {
        $download = Join-Path $roundtrip $entry.Key
        Receive-S3File -ObjectKey "$ObjectPrefix/$($entry.Key)" -Destination $download
        if ((Get-FileSha256 -Path $download) -ne $files[$entry.Key].sha256) {
            throw "S3 input-bundle round-trip hash mismatch: $($entry.Key)"
        }
    }
    foreach ($name in @("input_manifest.json", "input_manifest.hmac-sha256")) {
        $download = Join-Path $roundtrip $name
        Receive-S3File -ObjectKey "$ObjectPrefix/$name" -Destination $download
        if ((Get-FileSha256 -Path $download) -ne (Get-FileSha256 -Path (Join-Path $BundleDirectory $name))) {
            throw "S3 input-bundle round-trip hash mismatch: $name"
        }
    }
    Remove-Item -LiteralPath $roundtrip -Recurse -Force
    return [ordered]@{
        object_prefix = $ObjectPrefix
        manifest_sha256 = Get-FileSha256 -Path $manifestPath
        signature_sha256 = Get-FileSha256 -Path $signaturePath
        file_count = $files.Count
        roundtrip_verified = $true
    }
}

function Get-SanitizedPodState {
    param($Pod)
    if ($null -eq $Pod) { return $null }
    $machine = Get-OptionalProperty -Value $Pod -Name "machine"
    return [ordered]@{
        id = [string](Get-OptionalProperty -Value $Pod -Name "id" -Default "")
        name = [string](Get-OptionalProperty -Value $Pod -Name "name" -Default "")
        desired_status = [string](Get-OptionalProperty -Value $Pod -Name "desiredStatus" -Default "unknown")
        runtime_status = [string](Get-OptionalProperty -Value $Pod -Name "runtimeStatus" -Default "unknown")
        gpu_count = [int](Get-OptionalProperty -Value $Pod -Name "gpuCount" -Default 0)
        gpu_id = [string](Get-OptionalProperty -Value $machine -Name "gpuId" -Default "")
        secure_cloud = [bool](Get-OptionalProperty -Value $machine -Name "secureCloud" -Default $false)
        data_center_id = [string](Get-OptionalProperty -Value $machine -Name "dataCenterId" -Default "")
        cost_per_hour_usd = [decimal](Get-OptionalProperty -Value $Pod -Name "costPerHr" -Default 0)
        container_disk_gb = [int](Get-OptionalProperty -Value $Pod -Name "containerDiskInGb" -Default 0)
        persistent_volume_gb = [int](Get-OptionalProperty -Value $Pod -Name "volumeInGb" -Default 0)
        image = [string](Get-OptionalProperty -Value $Pod -Name "imageName" -Default "")
        ports = @(Get-OptionalProperty -Value $Pod -Name "ports" -Default @())
        uptime_seconds = [int](Get-OptionalProperty -Value $Pod -Name "uptimeSeconds" -Default 0)
    }
}

function Remove-BoundedPod {
    param(
        [Parameter(Mandatory)][string]$Reason,
        [string]$TargetPodId = $PodId
    )
    if (-not $TargetPodId) { throw "A Pod ID is required for verified deletion." }
    $predelete = $null
    $predeleteError = ""
    try {
        $predelete = Invoke-RunPodJson -Arguments @(
            "pod", "get", $TargetPodId,
            "--include-machine", "--include-network-volume", "-o", "json"
        )
    } catch {
        $predeleteError = $_.Exception.Message
    }
    Write-AtomicJson -Path (Join-Path $EvidenceDirectory "watchdog_predelete.json") -Value ([ordered]@{
        schema_version = "glm53_v19_watchdog_predelete_v1"
        captured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        reason = $Reason
        pod = Get-SanitizedPodState -Pod $predelete
        pod_get_error = $predeleteError
    })

    # Always issue DELETE by exact ID. A failed GET is not evidence of absence.
    $delete = Invoke-RunPodRaw -Arguments @("pod", "delete", $TargetPodId, "-o", "json")
    $stillPresent = $true
    $listChecks = 0
    $listError = ""
    for ($attempt = 0; $attempt -lt 12; $attempt++) {
        try {
            $listed = @(
                Get-RunPodRecords `
                    -Value (Invoke-RunPodJson -Arguments @("pod", "list", "--all", "-o", "json")) `
                    -ContainerNames @("pods", "items")
            )
            $listChecks += 1
            $stillPresent = @($listed | Where-Object {
                [string](Get-OptionalProperty -Value $_ -Name "id" -Default "") -eq $TargetPodId
            }).Count -ne 0
            if (-not $stillPresent) { break }
            $retryDelete = Invoke-RunPodRaw -Arguments @("pod", "delete", $TargetPodId, "-o", "json")
            if ($retryDelete.ExitCode -eq 0) { $delete = $retryDelete }
        } catch {
            $listError = $_.Exception.Message
        }
        Start-Sleep -Seconds 5
    }
    Write-AtomicJson -Path (Join-Path $EvidenceDirectory "watchdog_result.json") -Value ([ordered]@{
        schema_version = "glm53_v19_watchdog_result_v1"
        pod_id = $TargetPodId
        deleted_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        reason = $Reason
        delete_exit_code = $delete.ExitCode
        pod_absent_after_delete = ($listChecks -gt 0 -and -not $stillPresent)
        successful_list_checks = $listChecks
        final_list_error = $listError
        delete_response_present = [bool]$delete.Text
        s3_credential_rotation_required_after_project = $true
    })
    if ($listChecks -eq 0 -or $stillPresent) {
        throw "The v19 watchdog could not verify deletion of Pod $TargetPodId."
    }
}

function Remove-PodsByNameVerified {
    param(
        [Parameter(Mandatory)][string]$ExactName,
        [Parameter(Mandatory)][string]$Reason
    )
    $absentStreak = 0
    for ($attempt = 0; $attempt -lt 12; $attempt++) {
        $pods = @(
            Get-RunPodRecords `
                -Value (Invoke-RunPodJson -Arguments @("pod", "list", "--all", "-o", "json")) `
                -ContainerNames @("pods", "items")
        )
        $matching = @($pods | Where-Object {
            [string](Get-OptionalProperty -Value $_ -Name "name" -Default "") -eq $ExactName
        })
        if ($matching.Count -eq 0) {
            $absentStreak += 1
            if ($absentStreak -ge 3) { return }
        } else {
            $absentStreak = 0
            foreach ($pod in $matching) {
                $id = [string](Get-OptionalProperty -Value $pod -Name "id" -Default "")
                if (-not $id) { throw "A run-scoped Pod lacks an ID during cleanup." }
                Remove-BoundedPod -Reason $Reason -TargetPodId $id
            }
        }
        Start-Sleep -Seconds 5
    }
    $remaining = @(
        Get-RunPodRecords `
            -Value (Invoke-RunPodJson -Arguments @("pod", "list", "--all", "-o", "json")) `
            -ContainerNames @("pods", "items")
    )
    if (@($remaining | Where-Object {
        [string](Get-OptionalProperty -Value $_ -Name "name" -Default "") -eq $ExactName
    }).Count -ne 0) {
        throw "Run-scoped Pod cleanup by name did not reach verified absence."
    }
}

function Remove-TemplatesByNameVerified {
    param([Parameter(Mandatory)][string]$ExactName)
    $absentStreak = 0
    for ($attempt = 0; $attempt -lt 12; $attempt++) {
        $templates = @(
            Get-RunPodRecords `
                -Value (Invoke-RunPodJson -Arguments @(
                    "template", "list", "--type", "user", "--limit", "1000", "-o", "json"
                )) `
                -ContainerNames @("templates", "items")
        )
        $matching = @($templates | Where-Object {
            [string](Get-OptionalProperty -Value $_ -Name "name" -Default "") -eq $ExactName
        })
        if ($matching.Count -eq 0) {
            $absentStreak += 1
            if ($absentStreak -ge 3) { return }
        } else {
            $absentStreak = 0
        }
        foreach ($template in $matching) {
            $id = [string](Get-OptionalProperty -Value $template -Name "id" -Default "")
            if (-not $id) { throw "A run-scoped template lacks an ID during cleanup." }
            $delete = Invoke-RunPodRaw -Arguments @("template", "delete", $id, "-o", "json")
            if ($delete.ExitCode -ne 0) {
                throw "RunPod template deletion failed for $id."
            }
        }
        Start-Sleep -Seconds 3
    }
    throw "Run-scoped template cleanup did not reach verified absence."
}

if ($Mode -eq "Watchdog") {
    foreach ($value in @($PodId, $RunId, $RunPodCtl, $EvidenceDirectory)) {
        if (-not $value) { throw "Watchdog mode is missing a required argument." }
    }
    if ($DeadlineUtc -eq [datetime]::MinValue -or $BalanceFloorUsd -le 0) {
        throw "Watchdog mode requires a deadline and a positive balance floor."
    }
    if (-not (Test-Path -LiteralPath $RunPodCtl -PathType Leaf)) {
        throw "runpodctl is absent: $RunPodCtl"
    }
    Assert-S3CredentialSession | Out-Null
    Add-Type @"
using System.Runtime.InteropServices;
public static class Glm53V19SleepGuard {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    $continuous = [uint32]2147483648
    $systemRequired = [uint32]0x00000001
    $sleepState = [Glm53V19SleepGuard]::SetThreadExecutionState(
        $continuous -bor $systemRequired
    )
    if ($sleepState -eq 0) {
        throw "The v19 watchdog could not prevent workstation sleep."
    }
    New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null
    $started = (Get-Date).ToUniversalTime()
    Write-AtomicJson -Path (Join-Path $EvidenceDirectory "watchdog_start.json") -Value ([ordered]@{
        schema_version = "glm53_v19_watchdog_v1"
        pod_id = $PodId
        run_id = $RunId
        started_at_utc = $started.ToString("o")
        deadline_utc = $DeadlineUtc.ToUniversalTime().ToString("o")
        balance_floor_usd = $BalanceFloorUsd
        heartbeat_delete_after_seconds = $HeartbeatDeleteAfterSeconds
    })
    $heartbeatKey = "$S3Prefix/$RunId/heartbeat.json"
    $terminalKey = "$S3Prefix/$RunId/terminal.json"
    $lastFreshHeartbeat = $started
    $reason = "deadline"
    while ((Get-Date).ToUniversalTime() -lt $DeadlineUtc.ToUniversalTime()) {
        Start-Sleep -Seconds 30
        try {
            $account = Invoke-RunPodJson -Arguments @("user", "-o", "json")
            if ([decimal]$account.clientBalance -le $BalanceFloorUsd) {
                $reason = "balance_floor"
                break
            }
        } catch {
            # A transient balance query does not defeat the deadline or heartbeat limits.
        }
        $terminal = Get-S3Json -ObjectKey $terminalKey
        if ($null -ne $terminal) {
            if ([string]$terminal.run_id -ne $RunId -or [string]$terminal.pod_id -ne $PodId) {
                $reason = "invalid_terminal_binding"
                break
            }
            $reason = "terminal_marker"
            break
        }
        $heartbeat = Get-S3Json -ObjectKey $heartbeatKey
        if ($null -ne $heartbeat -and $heartbeat.created_at_utc) {
            try {
                if ([string]$heartbeat.run_id -ne $RunId -or [string]$heartbeat.pod_id -ne $PodId) {
                    throw "Heartbeat names another run or Pod."
                }
                $observed = ([datetime]$heartbeat.created_at_utc).ToUniversalTime()
                if ($observed -gt $lastFreshHeartbeat) { $lastFreshHeartbeat = $observed }
            } catch {
                # Malformed heartbeats are treated as absent.
            }
        }
        if (((Get-Date).ToUniversalTime() - $lastFreshHeartbeat).TotalSeconds -gt $HeartbeatDeleteAfterSeconds) {
            $reason = "missing_or_stale_heartbeat"
            break
        }
    }
    try {
        Remove-BoundedPod -Reason $reason
    } finally {
        [void][Glm53V19SleepGuard]::SetThreadExecutionState($continuous)
    }
    exit 0
}

if (-not $ConfirmSpend) { throw "Refusing paid launch without -ConfirmSpend." }
foreach ($path in @($V15Decision, $RuntimeConfig, $RunPodCtl)) {
    if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required launch input is absent: $path"
    }
}
if ($Image -ne $ExpectedImage) { throw "The requested image differs from the v19 runtime lock." }

$decision = Get-Content -LiteralPath $V15Decision -Raw | ConvertFrom-Json
if ($decision.passed -ne $true -or $decision.decision -ne $ExpectedDecision) {
    throw "The V15 parent has not authorized source activation extraction."
}
foreach ($field in @("exact_fp8_source_extraction", "runpod_compute")) {
    if ($decision.authorization.$field -ne $true) {
        throw "The V15 decision does not authorize $field."
    }
}
if ((Get-FileSha256 -Path $V15Decision) -ne "6546853ea1be45f179a4a396c621be375f15ba0ff5412d9775fbadbb4725b9fc") {
    throw "The V15 decision differs from the preregistered parent."
}
$v15Verification = "artifacts/glm53_user_eval/v15/reports/codex_cohort/verification.json"
if ((Get-FileSha256 -Path $v15Verification) -ne "da4f4d7d3e6fd5d33a34f14f99eae33f05d43f9ca6af18dea062646daf25509e") {
    throw "The V15 verification differs from the preregistered parent."
}

$tagCommit = (git rev-list -n 1 $InfrastructureAmendmentTag).Trim()
$headCommit = (git rev-parse HEAD).Trim()
if (-not $tagCommit -or $tagCommit -ne $headCommit) {
    throw "HEAD is not the immutable v19 infrastructure-amendment tag."
}
$scientificCommit = (git rev-list -n 1 $PreregTag).Trim()
if (-not $scientificCommit -or $scientificCommit -ne "24f358173193c97103dc5d3fff3c0a14ac7c88b1") {
    throw "The v19 frozen runtime-amendment tag differs from its recorded commit."
}
git merge-base --is-ancestor glm53-user-eval-v19-preregistered-r8 HEAD
if ($LASTEXITCODE -ne 0) {
    throw "The infrastructure amendment does not descend from the scientific preregistration."
}
if (git status --porcelain) { throw "Paid v19 launch requires a clean worktree." }
$bootstrapPath = Join-Path (Get-Location) $BootstrapRelativePath
if (-not (Test-Path -LiteralPath $bootstrapPath -PathType Leaf)) {
    throw "The hash-bound v19 bootstrap script is absent."
}
if (-not (Test-Path -LiteralPath $SourceArchive -PathType Leaf)) {
    throw "The frozen v19 source-transport archive is absent."
}
if ((Get-FileSha256 -Path $SourceArchive) -ne $SourceArchiveSha256) {
    throw "The v19 source-transport archive differs from the infrastructure amendment."
}
$archiveEntries = @(tar -tzf $SourceArchive)
if ($LASTEXITCODE -ne 0) {
    throw "The v19 source-transport archive cannot be listed."
}
foreach ($requiredArchiveEntry in @(
    "Non-verbal-Eval-Awareness/",
    "Non-verbal-Eval-Awareness/artifacts/datasets/contrastive_prompts_v5/manifest.json",
    "Non-verbal-Eval-Awareness/pipelines/glm53_user_eval/v19/configs/prereg_v19_lean_hua.yaml"
)) {
    if ($requiredArchiveEntry -notin $archiveEntries) {
        throw "The v19 source-transport archive has the wrong extraction layout: $requiredArchiveEntry"
    }
}
$bootstrapSha256 = Get-FileSha256 -Path $bootstrapPath
$bootstrapBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($bootstrapPath))
$dockerStartCommand = "set -euo pipefail; printf '%s' '$bootstrapBase64' | base64 --decode > /tmp/bootstrap_glm53_v19.sh.partial; printf '%s  %s\n' '$bootstrapSha256' '/tmp/bootstrap_glm53_v19.sh.partial' | sha256sum --check --strict -; mv /tmp/bootstrap_glm53_v19.sh.partial /tmp/bootstrap_glm53_v19.sh; chmod 700 /tmp/bootstrap_glm53_v19.sh; exec bash /tmp/bootstrap_glm53_v19.sh"

$credentialAttestedAt = Assert-S3CredentialSession
$activePods = @(
    Get-RunPodRecords `
        -Value (Invoke-RunPodJson -Arguments @("pod", "list", "--all", "-o", "json")) `
        -ContainerNames @("pods", "items")
)
if ($activePods.Count -ne 0) { throw "An active RunPod Pod already exists." }
$activeEndpoints = @(
    Get-RunPodRecords `
        -Value (Invoke-RunPodJson -Arguments @("serverless", "list", "-o", "json")) `
        -ContainerNames @("endpoints", "items")
)
if ($activeEndpoints.Count -ne 0) { throw "A RunPod Serverless endpoint already exists." }

$gpuRows = @(
    Get-RunPodRecords `
        -Value (Invoke-RunPodJson -Arguments @("gpu", "list", "-o", "json")) `
        -ContainerNames @("gpus", "items")
)
$gpu = @($gpuRows | Where-Object { $_.gpuId -eq $ExpectedGpuId })
if ($gpu.Count -ne 1 -or $gpu[0].secureCloud -ne $true -or $gpu[0].available -ne $true) {
    throw "The exact two-B300 Secure Cloud topology is not currently available."
}
$liveAggregateRate = [decimal]$gpu[0].securePricePerHr * $ExpectedGpuCount
if ($liveAggregateRate -le 0 -or $liveAggregateRate -gt $RateCapUsdPerHour) {
    throw "The live two-B300 rate exceeds the preregistered cap."
}
$account = Invoke-RunPodJson -Arguments @("user", "-o", "json")
$liveBalance = [decimal]$account.clientBalance
$availableComputeCap = $liveBalance - $MinimumReserveUsd - $StorageAllowanceUsd
if ($availableComputeCap -le 0) {
    throw "The live balance cannot fund the hard cap while preserving the v19 reserve."
}
$effectiveComputeCap = if ($availableComputeCap -lt $ComputeHardCapUsd) {
    $availableComputeCap
} else {
    $ComputeHardCapUsd
}
$projectedCompute = $liveAggregateRate * ([decimal]$WallClockMinutes / 60)
if ($projectedCompute -gt $effectiveComputeCap) {
    throw "The live 150-minute projection exceeds the effective V19 compute cap."
}
$balanceFloor = $liveBalance - $effectiveComputeCap
if ($balanceFloor -lt ($MinimumReserveUsd + $StorageAllowanceUsd)) {
    throw "The live balance cannot preserve both the v19 reserve and storage allowance."
}

New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null
$EvidenceDirectory = (Resolve-Path $EvidenceDirectory).Path
$RunId = "glm53-v19-hua-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))-$([guid]::NewGuid().ToString('N').Substring(0,8))"
$inputObjectPrefix = "$S3InputPrefix/$RunId"
$inputBundle = New-SignedInputBundle `
    -BundleDirectory (Join-Path $EvidenceDirectory "input_bundle\$RunId") `
    -ObjectPrefix $inputObjectPrefix `
    -GitCommit $scientificCommit `
    -CurrentRunId $RunId `
    -Decision $decision
    $sourceArchiveKey = "$inputObjectPrefix/v19_science_repo.tar.gz"
Send-S3File -Source ([System.IO.Path]::GetFullPath($SourceArchive)) -ObjectKey $sourceArchiveKey
$sourceArchiveRoundtrip = Join-Path $EvidenceDirectory ".v19-source-archive-$PID.partial"
try {
    Receive-S3File -ObjectKey $sourceArchiveKey -Destination $sourceArchiveRoundtrip
    if ((Get-FileSha256 -Path $sourceArchiveRoundtrip) -ne $SourceArchiveSha256) {
        throw "S3 source-transport archive round-trip hash mismatch."
    }
} finally {
    Remove-Item -LiteralPath $sourceArchiveRoundtrip -Force -ErrorAction SilentlyContinue
}
$inputBundle.source_transport = [ordered]@{
    object_key = $sourceArchiveKey
    sha256 = $SourceArchiveSha256
    scientific_commit = $scientificCommit
    roundtrip_verified = $true
}
$deadline = (Get-Date).ToUniversalTime().AddMinutes($WallClockMinutes)
$podEnvironment = [ordered]@{
    AWS_ACCESS_KEY_ID = [Environment]::GetEnvironmentVariable("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = [Environment]::GetEnvironmentVariable("AWS_SECRET_ACCESS_KEY")
    RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC = $credentialAttestedAt.ToString("o")
    GLM53_V19_RUN_ID = $RunId
    GLM53_V19_DEADLINE_UTC = $deadline.ToString("o")
    GLM53_V19_AGGREGATE_RATE_USD = [string]$liveAggregateRate
    GLM53_V19_LAUNCH_BALANCE_USD = [string]$liveBalance
    GLM53_V19_BALANCE_FLOOR_USD = [string]$balanceFloor
    GLM53_V19_INPUT_PREFIX = $inputObjectPrefix
    GLM53_V19_S3_ENDPOINT = $S3Endpoint.TrimEnd('/')
    GLM53_V19_S3_BUCKET = $S3Bucket
    GLM53_V19_RUN_SCIENCE = "1"
}
# RunPod injects RUNPOD_POD_ID. The bootstrap requires it and exits before
# model staging or science if the platform-provided value is absent.
$podName = "mats-glm53-v19-$RunId"
$PodId = ""
$details = $null
$watchdogProcess = $null
$launchComplete = $false
try {
    $localRunPodApiKey = Get-LocalRunPodApiKey
    $podCreateHeaders = @{ Authorization = "Bearer $localRunPodApiKey" }
    $podCreateBody = [ordered]@{
        name = $podName
        cloudType = "SECURE"
        computeType = "GPU"
        gpuTypeIds = @($ExpectedGpuId)
        gpuCount = $ExpectedGpuCount
        dataCenterIds = @()
        containerDiskInGb = 450
        volumeInGb = 0
        imageName = $ExpectedImage
        dockerEntrypoint = @("/bin/bash", "-lc")
        dockerStartCmd = @($dockerStartCommand)
        ports = @("22/tcp")
        env = $podEnvironment
    } | ConvertTo-Json -Depth 5 -Compress
    try {
        $created = Invoke-RestMethod `
            -Method Post `
            -Uri "https://rest.runpod.io/v1/pods" `
            -Headers $podCreateHeaders `
            -ContentType "application/json" `
            -Body $podCreateBody `
            -TimeoutSec 120
        $PodId = [string](Get-OptionalProperty -Value $created -Name "id" -Default "")
    } catch {
        throw "RunPod Pod creation failed or returned a malformed response."
    } finally {
        $podEnvironment["AWS_ACCESS_KEY_ID"] = ""
        $podEnvironment["AWS_SECRET_ACCESS_KEY"] = ""
        $podCreateHeaders["Authorization"] = ""
        $podCreateBody = ""
        $localRunPodApiKey = ""
    }
    if (-not $PodId) { throw "RunPod Pod creation response lacks an ID." }

    $details = Invoke-RunPodJson -Arguments @(
        "pod", "get", $PodId,
        "--include-machine", "--include-network-volume", "-o", "json"
    )
    $machine = Get-OptionalProperty -Value $details -Name "machine"
    $topologyValid = (
        [int](Get-OptionalProperty -Value $details -Name "gpuCount" -Default 0) -eq $ExpectedGpuCount -and
        [string](Get-OptionalProperty -Value $machine -Name "gpuId" -Default "") -eq $ExpectedGpuId -and
        [bool](Get-OptionalProperty -Value $machine -Name "secureCloud" -Default $false) -eq $true -and
        [int](Get-OptionalProperty -Value $details -Name "containerDiskInGb" -Default 0) -eq 450 -and
        [int](Get-OptionalProperty -Value $details -Name "volumeInGb" -Default 0) -eq 0 -and
        [string](Get-OptionalProperty -Value $details -Name "imageName" -Default "") -eq $ExpectedImage -and
        [decimal](Get-OptionalProperty -Value $details -Name "costPerHr" -Default 0) -le $RateCapUsdPerHour
    )
    if (-not $topologyValid) { throw "Created Pod violates the exact v19 topology contract." }

    $scriptPath = $MyInvocation.MyCommand.Path
    $watchdogStdout = Join-Path $EvidenceDirectory "watchdog_stdout.log"
    $watchdogStderr = Join-Path $EvidenceDirectory "watchdog_stderr.log"
    $watchdogProcess = Start-Process (Get-Process -Id $PID).Path `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $watchdogStdout `
        -RedirectStandardError $watchdogStderr `
        -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptPath,
        "-Mode", "Watchdog",
        "-PodId", $PodId,
        "-RunId", $RunId,
        "-DeadlineUtc", $deadline.ToString("o"),
        "-BalanceFloorUsd", [string]$balanceFloor,
        "-RunPodCtl", $RunPodCtl,
        "-EvidenceDirectory", $EvidenceDirectory,
        "-S3Endpoint", $S3Endpoint,
        "-S3Bucket", $S3Bucket,
        "-S3Prefix", $S3Prefix
        )
    $watchdogStartPath = Join-Path $EvidenceDirectory "watchdog_start.json"
    $watchdogStarted = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($watchdogProcess.HasExited) {
            throw "The external v19 watchdog exited before its start handshake."
        }
        if (Test-Path -LiteralPath $watchdogStartPath -PathType Leaf) {
            try {
                $watchdogStart = Get-Content -LiteralPath $watchdogStartPath -Raw | ConvertFrom-Json
                if (
                    [string]$watchdogStart.run_id -eq $RunId -and
                    [string]$watchdogStart.pod_id -eq $PodId
                ) {
                    $watchdogStarted = $true
                    break
                }
            } catch {
                # An incomplete atomic handoff is retried until the bounded timeout.
            }
        }
        Start-Sleep -Seconds 2
    }
    if (-not $watchdogStarted) {
        throw "The external v19 watchdog did not produce a bound start handshake."
    }

    $heartbeatKey = "$S3Prefix/$RunId/heartbeat.json"
    $terminalKey = "$S3Prefix/$RunId/terminal.json"
    $bootstrapSignal = ""
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($watchdogProcess.HasExited) {
            throw "The external v19 watchdog exited before bootstrap was observed."
        }
        foreach ($candidate in @(
            [ordered]@{ name = "terminal"; value = (Get-S3Json -ObjectKey $terminalKey) },
            [ordered]@{ name = "heartbeat"; value = (Get-S3Json -ObjectKey $heartbeatKey) }
        )) {
            if ($null -eq $candidate.value) { continue }
            if (
                [string]$candidate.value.run_id -ne $RunId -or
                [string]$candidate.value.pod_id -ne $PodId
            ) {
                throw "The first Pod signal is not bound to this v19 run and Pod."
            }
            $bootstrapSignal = [string]$candidate.name
            break
        }
        if ($bootstrapSignal) { break }
        Start-Sleep -Seconds 5
    }
    if (-not $bootstrapSignal) {
        throw "The Pod did not emit a bound heartbeat or terminal marker within five minutes."
    }

    $launchRecord = [ordered]@{
        schema_version = "glm53_v19_runpod_launch_v1"
        created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        run_id = $RunId
        git_commit = $headCommit
        prereg_tag = $PreregTag
        infrastructure_amendment_tag = $InfrastructureAmendmentTag
        scientific_commit = $scientificCommit
        v15_decision_sha256 = Get-FileSha256 -Path $V15Decision
        runtime_config_sha256 = Get-FileSha256 -Path $RuntimeConfig
        bootstrap_sha256 = $bootstrapSha256
        bootstrap_source = "$BootstrapRelativePath@<preregistered-commit>"
        bootstrap_start = "observed_bound_$bootstrapSignal"
        bootstrap_delivery = "direct_pod_rest_body_without_template"
        external_watchdog_start_observed = $true
        external_watchdog_process_id = $watchdogProcess.Id
        parent_decision = [ordered]@{
            decision = $decision.decision
            sha256 = Get-FileSha256 -Path $V15Decision
        }
        input_bundle = $inputBundle
        s3_credentials_validated = $true
        s3_credential_session_attested_at_utc = $credentialAttestedAt.ToString("o")
        s3_credential_read_write_probe_passed = $true
        s3_credential_rotation_required_after_project = $true
        pod_scoped_api_deadline_delete_required = $true
        pod = Get-SanitizedPodState -Pod $details
        network_volume_attached = $false
        local_model_staging = "huggingface_xet_to_container_disk"
        live_balance_usd = $liveBalance
        live_aggregate_rate_usd_per_hour = $liveAggregateRate
        projected_compute_usd = $projectedCompute
        configured_compute_hard_cap_usd = $ComputeHardCapUsd
        effective_compute_hard_cap_usd = $effectiveComputeCap
        storage_allowance_usd = $StorageAllowanceUsd
        balance_floor_usd = $balanceFloor
        minimum_reserve_usd = $MinimumReserveUsd
        deadline_utc = $deadline.ToString("o")
        heartbeat_key = $heartbeatKey
        terminal_key = $terminalKey
    }
    Write-AtomicJson -Path (Join-Path $EvidenceDirectory "launch.json") -Value $launchRecord
    $launchComplete = $true
} finally {
    $podEnvironment["AWS_ACCESS_KEY_ID"] = ""
    $podEnvironment["AWS_SECRET_ACCESS_KEY"] = ""
    if (-not $launchComplete) {
        $cleanupFailures = @()
        if ($PodId) {
            try {
                Remove-BoundedPod -Reason "launch_failed" -TargetPodId $PodId
            } catch {
                $cleanupFailures += $_.Exception.Message
            }
        }
        try {
            Remove-PodsByNameVerified -ExactName $podName -Reason "launch_reconciliation"
        } catch {
            $cleanupFailures += $_.Exception.Message
        }
        if ($null -ne $watchdogProcess -and -not $watchdogProcess.HasExited) {
            Stop-Process -Id $watchdogProcess.Id -Force -ErrorAction SilentlyContinue
        }
        if ($cleanupFailures.Count -ne 0) {
            throw "The v19 launch failed and cleanup was not fully verified: $($cleanupFailures -join '; ')"
        }
    }
}

$launchRecord | ConvertTo-Json -Depth 10 | Write-Output
