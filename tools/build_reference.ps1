[CmdletBinding()]
param([switch]$SkipFfmpegBuild)
$ErrorActionPreference = 'Stop'
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$FfmpegRoot = Join-Path $RepositoryRoot 'third_party\ffmpeg-reference-shared'
if (-not $SkipFfmpegBuild) {
    & (Join-Path $PSScriptRoot 'build_ffmpeg_minimal.ps1') -Variant Reference
}
$Marker = Join-Path $FfmpegRoot 'dssim-ffmpeg-variant.txt'
if (-not (Test-Path -LiteralPath $Marker) -or
    (Get-Content -LiteralPath $Marker -Raw).Trim() -ne 'reference-shared-v1') {
    throw 'Build the unpatched Reference FFmpeg DLLs first.'
}
$PreviousFfmpegDir = $env:FFMPEG_DIR
$PreviousTargetDir = $env:CARGO_TARGET_DIR
try {
    $env:FFMPEG_DIR = $FfmpegRoot
    $env:CARGO_TARGET_DIR = Join-Path $RepositoryRoot 'src_reference\target'
    & cargo build --manifest-path (Join-Path $RepositoryRoot 'src_reference\Cargo.toml') --release --features video
    if ($LASTEXITCODE -ne 0) { throw 'Reference cargo build failed.' }
    # App-local DLLs take precedence over PATH, including GPU FFmpeg on PATH.
    Get-ChildItem -LiteralPath (Join-Path $FfmpegRoot 'bin') -Filter '*.dll' -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $env:CARGO_TARGET_DIR 'release') -Force
    }
} finally {
    $env:FFMPEG_DIR = $PreviousFfmpegDir
    $env:CARGO_TARGET_DIR = $PreviousTargetDir
}
