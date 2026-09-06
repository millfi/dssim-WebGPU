[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Configuration = 'Release',
    [string]$BuildDirectory = 'build'
)
$ErrorActionPreference = 'Stop'
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location -LiteralPath $RepositoryRoot
try {
    & .\tools\build_ffmpeg_minimal.ps1 -Variant Gpu
    & cmake -S . -B $BuildDirectory
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
    & cmake --build $BuildDirectory --config $Configuration --target dssim_vulkan
    if ($LASTEXITCODE -ne 0) { throw 'GPU build failed.' }
} finally {
    Pop-Location
}
