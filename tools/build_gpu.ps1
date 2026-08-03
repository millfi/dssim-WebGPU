[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Configuration = 'Release',
    [string]$BuildDirectory = 'build',
    [string]$FfmpegArchiveUrl = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$FfmpegRoot = Join-Path $RepositoryRoot 'third_party\ffmpeg-8.1.2-shared'
$FfmpegArchive = Join-Path $RepositoryRoot 'third_party\ffmpeg-8.1.2-shared.zip'
$FfmpegArchiveSha256 = '408D2FCCF2C4B0973B75AE2375D2DF174BD334DFFCBA7AA01AE80DED4AED6092'
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

function Test-FfmpegArchive([string]$ArchivePath) {
    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        return $false
    }
    return (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash -eq
        $FfmpegArchiveSha256
}

function Get-MissingFfmpegFiles {
    return @($RequiredFfmpegFiles | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $FfmpegRoot $_) -PathType Leaf)
    })
}

function Get-DefaultFfmpegArchiveUrl {
    $git = Get-Command 'git.exe' -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $git) {
        $git = Get-Command 'git' -CommandType Application -ErrorAction SilentlyContinue
    }
    if ($null -eq $git) {
        throw 'git is required to determine the FFmpeg archive download URL. Pass -FfmpegArchiveUrl explicitly.'
    }

    $originUrl = (& $git.Source -C $RepositoryRoot remote get-url origin).Trim()
    $commit = (& $git.Source -C $RepositoryRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($commit)) {
        throw 'Unable to determine the current Git commit for the FFmpeg archive download.'
    }
    if ($originUrl -notmatch 'github\.com[/:](?<repository>[^/]+/[^/]+?)(?:\.git)?$') {
        throw 'The origin remote is not a GitHub repository. Pass -FfmpegArchiveUrl explicitly.'
    }

    $repository = $Matches.repository
    return "https://raw.githubusercontent.com/$repository/$commit/third_party/ffmpeg-8.1.2-shared.zip"
}

Push-Location -LiteralPath $RepositoryRoot
try {
    $ArchiveWasDownloaded = $false
    if (-not (Test-FfmpegArchive $FfmpegArchive)) {
        if ([string]::IsNullOrWhiteSpace($FfmpegArchiveUrl)) {
            $FfmpegArchiveUrl = Get-DefaultFfmpegArchiveUrl
        }
        $TemporaryArchive = "$FfmpegArchive.download"
        Write-Warning "The bundled FFmpeg archive is missing or has an invalid SHA-256; downloading $FfmpegArchiveUrl"
        try {
            Invoke-WebRequest -Uri $FfmpegArchiveUrl -OutFile $TemporaryArchive
            if (-not (Test-FfmpegArchive $TemporaryArchive)) {
                throw "Downloaded FFmpeg archive failed SHA-256 verification: $FfmpegArchiveUrl"
            }
            Move-Item -LiteralPath $TemporaryArchive -Destination $FfmpegArchive -Force
            $ArchiveWasDownloaded = $true
        } finally {
            Remove-Item -LiteralPath $TemporaryArchive -Force -ErrorAction SilentlyContinue
        }
    }

    $MissingFfmpegFiles = @(Get-MissingFfmpegFiles)
    if ($ArchiveWasDownloaded -or $MissingFfmpegFiles.Count -gt 0) {
        if (Test-Path -LiteralPath $FfmpegRoot) {
            Remove-Item -LiteralPath $FfmpegRoot -Recurse -Force
        }
        Write-Host "Extracting verified minimal FFmpeg archive to $FfmpegRoot"
        Expand-Archive -LiteralPath $FfmpegArchive `
            -DestinationPath (Join-Path $RepositoryRoot 'third_party') -Force
    }

    $MissingFfmpegFiles = @(Get-MissingFfmpegFiles)
    if ($MissingFfmpegFiles.Count -gt 0) {
        throw "The verified FFmpeg archive is incomplete: $($MissingFfmpegFiles -join ', ')"
    }
    Write-Host "Using verified minimal FFmpeg from $FfmpegRoot"

    & cmake -S . -B $BuildDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE."
    }

    & cmake --build $BuildDirectory --config $Configuration --target dssim_vulkan
    if ($LASTEXITCODE -ne 0) {
        throw "dssim_vulkan build failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "Built src_gpu configuration $Configuration in $BuildDirectory"
