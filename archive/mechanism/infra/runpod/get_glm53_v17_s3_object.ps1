param(
    [Parameter(Mandatory)]
    [string]$ObjectKey,

    [Parameter(Mandatory)]
    [string]$Destination,

    [string]$Bucket = "aehhoaoldv",
    [string]$Endpoint = "https://s3api-us-ks-2.runpod.io",
    [string]$Region = "us-ks-2",
    [string]$CredentialDirectory = "$env:LOCALAPPDATA\glm53-v16"
)

$ErrorActionPreference = "Stop"

function Read-DpapiSecret {
    param([Parameter(Mandatory)][string]$Path)

    $protected = (Get-Content -LiteralPath $Path -Raw).Trim()
    $secure = ConvertTo-SecureString -String $protected
    return [Net.NetworkCredential]::new("", $secure).Password
}

$access = Read-DpapiSecret -Path (Join-Path $CredentialDirectory "s3_access.dpapi")
$secret = Read-DpapiSecret -Path (Join-Path $CredentialDirectory "s3_secret.dpapi")
$parent = Split-Path -Parent $Destination
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

$partial = "$Destination.partial"
$url = "$($Endpoint.TrimEnd('/'))/$Bucket/$ObjectKey"
try {
    & curl.exe --silent --show-error --fail `
        --connect-timeout 20 --max-time 900 --retry 3 --retry-all-errors `
        --user "${access}:${secret}" `
        --aws-sigv4 "aws:amz:${Region}:s3" `
        --output $partial $url
    if ($LASTEXITCODE -ne 0) {
        throw "S3 download failed for $ObjectKey."
    }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
} finally {
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Remove-Variable access, secret -ErrorAction SilentlyContinue
}

[ordered]@{
    object_key = $ObjectKey
    destination = (Resolve-Path -LiteralPath $Destination).Path
    bytes = (Get-Item -LiteralPath $Destination).Length
    sha256 = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json
