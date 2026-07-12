[CmdletBinding()]
param(
    [ValidateSet('Dynamic', 'Static')]
    [string]$Linkage = 'Dynamic',
    [string]$Prefix = (Join-Path $PSScriptRoot "..\third_party\ffmpeg-8.1.2-shared"),
    [string]$MsysRoot = (Join-Path $env:USERPROFILE 'msys64'),
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

# Update this together with the matching ffmpeg-next dependency in
# src_reference/Cargo.toml. 8.1.2 is the latest stable release as of 2026-07-10.
$Version = '8.1.2'
$SourceUrl = "https://ffmpeg.org/releases/ffmpeg-$Version.tar.xz"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$BuildRoot = Join-Path $RepositoryRoot 'third_party\ffmpeg-build'
$SourceRoot = Join-Path $BuildRoot "ffmpeg-$Version"
$Prefix = [System.IO.Path]::GetFullPath($Prefix)

function Get-RequiredCommand([string]$Name, [string]$Hint) {
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "'$Name' was not found. $Hint"
    }
    return $command.Source
}

function ConvertTo-BashPath([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if ($fullPath -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Cannot convert path to a Bash path: $fullPath"
    }
    return '/' + $Matches[1].ToLowerInvariant() + '/' + ($Matches[2] -replace '\\', '/')
}

$VsDevShell = @(
    'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1',
    'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($null -eq (Get-Command 'cl.exe' -CommandType Application -ErrorAction SilentlyContinue)) {
    if ([string]::IsNullOrWhiteSpace($VsDevShell)) {
        throw 'cl.exe is not on PATH and a Visual Studio Developer PowerShell launcher was not found.'
    }
    . $VsDevShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
}
foreach ($tool in @('cl.exe', 'link.exe', 'lib.exe')) {
    Get-RequiredCommand $tool 'Install Visual Studio C++ build tools for x64.' | Out-Null
}

$BashPath = $null
foreach ($root in @($MsysRoot, 'C:\msys64')) {
    $candidate = Join-Path $root 'usr\bin\bash.exe'
    if (Test-Path -LiteralPath $candidate) {
        $BashPath = $candidate
        break
    }
}
if ($null -eq $BashPath) {
    throw "MSYS2 bash.exe was not found. Install MSYS2 or pass -MsysRoot <path-to-msys64>."
}
Get-RequiredCommand 'tar.exe' 'Install bsdtar or use the tar.exe bundled with Windows.' | Out-Null
$MsysPrelude = 'export PATH=/usr/bin:/bin:$PATH; '
& $BashPath --noprofile --norc -lc "${MsysPrelude}command -v make >/dev/null"
if ($LASTEXITCODE -ne 0) {
    throw "GNU make was not found in MSYS2. Install it with: pacman -S --needed make"
}

if ($Clean -and (Test-Path -LiteralPath $BuildRoot)) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
$Archive = Join-Path $BuildRoot "ffmpeg-$Version.tar.xz"
if (-not (Test-Path -LiteralPath $SourceRoot)) {
    if (-not (Test-Path -LiteralPath $Archive)) {
        Write-Host "Downloading FFmpeg $Version from ffmpeg.org"
        Invoke-WebRequest -Uri $SourceUrl -OutFile $Archive
    }
    & tar.exe -xf $Archive -C $BuildRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to unpack $Archive"
    }
}

$SourceBashPath = ConvertTo-BashPath $SourceRoot
$PrefixBashPath = ConvertTo-BashPath $Prefix
$VulkanIncludeBashPath = ConvertTo-BashPath (Join-Path $env:VULKAN_SDK 'Include')
$Parallelism = [Math]::Max(1, [Environment]::ProcessorCount)
$LinkageArguments = if ($Linkage -eq 'Dynamic') {
    @('--enable-shared', '--disable-static')
} else {
    @('--enable-static', '--disable-shared')
}

# Keep only the containers, parsers, codecs, and Vulkan Video hwaccels needed
# by dssim-WebGPU. The application consumes AV_PIX_FMT_VULKAN frames directly;
# there is deliberately no swscale/libavfilter path in this development build.
$ConfigureArguments = @(
    '--toolchain=msvc', '--arch=x86_64',
    "--prefix=$PrefixBashPath"
) + $LinkageArguments + @(
    '--disable-programs', '--disable-doc', '--disable-debug',
    '--disable-autodetect', '--disable-network', '--disable-avdevice', '--disable-avfilter',
    '--disable-swresample', '--disable-swscale', '--disable-encoders', '--disable-muxers',
    '--disable-filters', '--disable-bsfs', '--disable-protocols', '--disable-indevs', '--disable-outdevs',
    '--disable-decoders', '--disable-demuxers', '--disable-parsers', '--disable-hwaccels', '--disable-asm',
    '--enable-avutil', '--enable-avcodec', '--enable-avformat',
    '--enable-protocol=file', '--enable-demuxer=mov,matroska',
    '--enable-parser=h264,hevc,av1,vp9', '--enable-decoder=h264,hevc,av1,vp9',
    '--enable-vulkan',
    '--enable-hwaccel=h264_vulkan,hevc_vulkan,av1_vulkan,vp9_vulkan',
    # Rust's MSVC target uses the dynamic CRT. Match it in FFmpeg so the
    # FFmpeg objects do not introduce LIBCMT and cause LNK4098 at the final
    # executable link.
    # `-MD` is accepted by cl.exe and, unlike `/MD`, is not rewritten as an
    # MSYS2 path while it passes through bash.exe.
    '--extra-cflags=-MD',
    "--extra-cflags=-I$VulkanIncludeBashPath"
)

$ConfigureLine = './configure ' + ($ConfigureArguments -join ' ')
if (Test-Path -LiteralPath (Join-Path $SourceRoot 'ffbuild\config.mak')) {
    & $BashPath --noprofile --norc -lc "${MsysPrelude}cd '$SourceBashPath' && make distclean"
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to clean the previous FFmpeg configuration.'
    }
}
Write-Host "Configuring FFmpeg $Version"
& $BashPath --noprofile --norc -lc "${MsysPrelude}cd '$SourceBashPath' && $ConfigureLine"
if ($LASTEXITCODE -ne 0) {
    throw 'FFmpeg configure failed.'
}

Write-Host "Building FFmpeg $Version"
& $BashPath --noprofile --norc -lc "${MsysPrelude}cd '$SourceBashPath' && make -j$Parallelism && make install"
if ($LASTEXITCODE -ne 0) {
    throw 'FFmpeg build failed.'
}

if ($Linkage -eq 'Dynamic') {
    # FFmpeg's MSVC shared build installs DLLs and their import libraries in
    # bin/. ffmpeg-sys-next discovers prebuilt libraries exclusively under
    # FFMPEG_DIR/lib, so mirror only the four import libraries there.
    foreach ($library in @('avcodec.lib', 'avformat.lib', 'avutil.lib')) {
        $source = Join-Path $Prefix "bin\$library"
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Shared FFmpeg import library was not installed: $source"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $Prefix "lib\$library") -Force
    }
}

Write-Host "Minimal $Linkage FFmpeg installed at $Prefix"
Write-Host 'Build the reference CLI with:'
Write-Host "  `$env:FFMPEG_DIR = '$Prefix'"
Write-Host '  & cargo build --manifest-path .\src_reference\Cargo.toml --release --features video'
if ($Linkage -eq 'Dynamic') {
    Write-Host 'Run with the FFmpeg DLL directory on PATH:'
    Write-Host "  `$env:PATH = '$Prefix\bin;' + `$env:PATH"
}
