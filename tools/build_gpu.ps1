[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Configuration = 'Release',
    [string]$BuildDirectory = 'build'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$FfmpegRoot = Join-Path $RepositoryRoot 'third_party\ffmpeg-8.1.2-shared'
$RequiredFfmpegFiles = @(
    'include\libavcodec\avcodec.h',
    'include\libavformat\avformat.h',
    'include\libavutil\avutil.h',
    'include\libswscale\swscale.h',
    'lib\avcodec.lib',
    'lib\avformat.lib',
    'lib\avutil.lib',
    'lib\swscale.lib',
    'bin\avcodec-62.dll',
    'bin\avformat-62.dll',
    'bin\avutil-60.dll',
    'bin\swscale-9.dll'
)

$MissingFfmpegFiles = @(
    $RequiredFfmpegFiles | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $FfmpegRoot $_) -PathType Leaf)
    }
)

Push-Location -LiteralPath $RepositoryRoot
try {
    if ($MissingFfmpegFiles.Count -gt 0) {
        Write-Warning 'The bundled minimal FFmpeg files are incomplete; rebuilding the dependency.'
        & (Join-Path $PSScriptRoot 'build_ffmpeg_minimal.ps1') -Linkage Dynamic
        if ($LASTEXITCODE -ne 0) {
            throw "Minimal FFmpeg build failed with exit code $LASTEXITCODE."
        }
    } else {
        Write-Host "Using bundled minimal FFmpeg from $FfmpegRoot"
    }

    & cmake -S . -B $BuildDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE."
    }

    & cmake --build $BuildDirectory --config $Configuration --target dssim_webgpu
    if ($LASTEXITCODE -ne 0) {
        throw "dssim_webgpu build failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "Built src_gpu configuration $Configuration in $BuildDirectory"
