param(
    [string]$Prefix = "glm53-v17-results/",
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
$temporary = Join-Path $env:TEMP "glm53-v17-s3-list-$([guid]::NewGuid().ToString('N')).xml"
$escapedPrefix = [Uri]::EscapeDataString($Prefix)
$url = "$($Endpoint.TrimEnd('/'))/$Bucket`?list-type=2&prefix=$escapedPrefix"
try {
    & curl.exe --silent --show-error --fail `
        --connect-timeout 20 --max-time 120 --retry 3 --retry-all-errors `
        --user "${access}:${secret}" `
        --aws-sigv4 "aws:amz:${Region}:s3" `
        --output $temporary $url
    if ($LASTEXITCODE -ne 0) {
        throw "S3 object listing failed for prefix $Prefix."
    }
    [xml]$document = Get-Content -LiteralPath $temporary -Raw
    @($document.ListBucketResult.Contents) |
        ForEach-Object {
            [ordered]@{
                key = [string]$_.Key
                bytes = [long]$_.Size
            }
        } |
        ConvertTo-Json -Depth 3
} finally {
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    Remove-Variable access, secret -ErrorAction SilentlyContinue
}
