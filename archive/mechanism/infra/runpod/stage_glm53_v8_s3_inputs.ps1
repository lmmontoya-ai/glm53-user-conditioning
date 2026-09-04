[CmdletBinding()]
param(
    [string]$Aws = "aws",
    [string]$EndpointUrl = "https://s3api-us-ks-2.runpod.io/",
    [string]$Bucket = "a9diryunoj",
    [string]$Prefix = "glm53-v8-input/v1.7"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$tag = "glm53-user-eval-v8-preregistered-v1.7"
if ((git rev-list -n 1 $tag) -ne (git rev-parse HEAD)) { throw "HEAD is not $tag." }
if (git status --porcelain) { throw "Repository is dirty." }
foreach ($name in @("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")) {
    if (-not [Environment]::GetEnvironmentVariable($name)) {
        throw "Required S3 credential is absent: $name"
    }
}

$root = (Get-Location).Path
$bundle = Join-Path $root "artifacts\glm53_user_eval\v8\infrastructure\s3_input_bundle_v1_7"
New-Item -ItemType Directory -Force -Path $bundle | Out-Null
$inputs = [ordered]@{
    "m0_decision.json" = "artifacts\glm53_user_eval\v8\decisions\m0_decision.json"
    "m1_decision.json" = "artifacts\glm53_user_eval\v8\m1\proxy_contract.json"
    "transcript_cache.jsonl" = "artifacts\glm53_user_eval\v8\cache\v7_transcripts_25.jsonl"
    "transcript_cache.manifest.json" = "artifacts\glm53_user_eval\v8\cache\v7_transcripts_25_manifest.json"
    "preregistration.yaml" = "pipelines\glm53_user_eval\v8\configs\prereg_v8_whitebox_mechanism.yaml"
    "direction_splits_v1.json" = "pipelines\glm53_user_eval\v8\configs\direction_splits_v1.json"
}
$hashes = [ordered]@{}
foreach ($entry in $inputs.GetEnumerator()) {
    $source = Join-Path $root $entry.Value
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing frozen input: $($entry.Value)"
    }
    $destination = Join-Path $bundle $entry.Key
    Copy-Item -LiteralPath $source -Destination $destination -Force
    $hashes[$entry.Key] = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
}
$manifest = [ordered]@{
    schema_version = "glm53_v8_s3_input_manifest_v1"
    prereg_tag = $tag
    git_commit = git rev-parse HEAD
    endpoint_url = $EndpointUrl
    bucket = $Bucket
    prefix = $Prefix
    files = $hashes
}
$manifestPath = Join-Path $bundle "expected_hashes.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8

& $Aws s3 cp $bundle "s3://$Bucket/$Prefix/" `
    --recursive `
    --region US-KS-2 `
    --endpoint-url $EndpointUrl `
    --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "S3 input upload failed." }
& $Aws s3 cp "s3://$Bucket/$Prefix/expected_hashes.json" "$manifestPath.roundtrip" `
    --region US-KS-2 `
    --endpoint-url $EndpointUrl `
    --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "S3 manifest round-trip failed." }
$expectedHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash
$roundtripHash = (Get-FileHash -LiteralPath "$manifestPath.roundtrip" -Algorithm SHA256).Hash
Remove-Item -LiteralPath "$manifestPath.roundtrip" -Force
if ($expectedHash -ne $roundtripHash) { throw "S3 manifest round-trip hash mismatch." }
$manifest | ConvertTo-Json -Depth 5 | Write-Output
