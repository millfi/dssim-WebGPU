[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$toolRoot = Join-Path $repositoryRoot 'third_party/slang-2026.14.1'
$compiler = Join-Path $toolRoot 'bin/slangc.exe'
if (Test-Path -LiteralPath $compiler) {
    $version = & $compiler -version 2>&1
    if ($LASTEXITCODE -eq 0 -and "$version" -match '^2026\.14\.1') {
        Write-Host "Using Slang $version from $toolRoot"
        return
    }
}
$archive = Join-Path $repositoryRoot 'third_party/slang-2026.14.1.zip'
$expectedHash = '5ED0A59D650A0AF0ACA45D5DB4E083B3D8FB5CEA05748747DD95DFBE9C580658'
if (-not (Test-Path -LiteralPath $archive) -or
    (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
    New-Item -ItemType Directory -Force -Path (Split-Path $archive) | Out-Null
    Invoke-WebRequest -Uri 'https://github.com/shader-slang/slang/releases/download/v2026.14.1/slang-2026.14.1-windows-x86_64.zip' -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Slang archive SHA-256 verification failed.'
}
Expand-Archive -LiteralPath $archive -DestinationPath $toolRoot -Force
& $compiler -version
if ($LASTEXITCODE -ne 0) { throw 'Slang compiler did not run successfully.' }
