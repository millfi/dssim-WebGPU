[CmdletBinding()]
param(
    [ValidateSet('Dynamic')]
    [string]$Linkage = 'Dynamic',
    [ValidateSet('Gpu', 'Reference')]
    [string]$Variant = 'Gpu',
    [string]$MsysRoot = (Join-Path $env:USERPROFILE 'msys64'),
    [string]$VcpkgRoot = '',
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

# Check local tool prerequisites before downloading/building dependencies.
if (-not (Test-Path -LiteralPath (Join-Path $MsysRoot 'usr\bin\bash.exe')) -and
    -not (Test-Path -LiteralPath 'C:\msys64\usr\bin\bash.exe')) {
    throw 'MSYS2 is required (make and diffutils). Install it or pass -MsysRoot <directory>.'
}

# Update this together with the matching ffmpeg-next dependency in
# src_reference/Cargo.toml. 8.1.2 is the latest stable release as of 2026-07-10.
$Version = '8.1.2'
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$UpstreamRoot = Join-Path $RepositoryRoot "third_party\ffmpeg-$Version"
$VariantName = $Variant.ToLowerInvariant()
$BuildRoot = Join-Path $RepositoryRoot "third_party\ffmpeg-build\$VariantName"
$SourceRoot = Join-Path $BuildRoot "source"
$Prefix = Join-Path $RepositoryRoot "third_party\ffmpeg-$VariantName-shared"
if (-not (Test-Path -LiteralPath (Join-Path $UpstreamRoot 'configure'))) {
    throw "Vendored FFmpeg source is missing: $UpstreamRoot"
}
if ((Get-Content -LiteralPath (Join-Path $UpstreamRoot 'libavcodec\vulkan_av1.c') -Raw).Contains('VK_DRIVER_ID_AMD_PROPRIETARY ?')) {
    throw 'The vendored upstream source must remain unpatched; only the GPU working copy may be patched.'
}
$ZlibVersion = '1.3.1'
$ZlibSourceRoot = Join-Path $BuildRoot "zlib-$ZlibVersion"
$ZlibBuildRoot = Join-Path $BuildRoot "zlib-$ZlibVersion-build"
$ZlibPrefix = Join-Path $BuildRoot "zlib-$ZlibVersion-prefix"
$VcpkgTriplet = 'x64-windows'
$RepositoryVcpkgRoot = Join-Path $RepositoryRoot 'third_party\vcpkg'

function Test-VcpkgRoot([string]$Root) {
    return -not [string]::IsNullOrWhiteSpace($Root) -and
        (Test-Path -LiteralPath (Join-Path $Root 'vcpkg.exe') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Root 'ports') -PathType Container) -and
        (Test-Path -LiteralPath (Join-Path $Root 'scripts\buildsystems\vcpkg.cmake') -PathType Leaf)
}

function Get-VcpkgRoot([string]$RequestedRoot, [string]$LocalRoot) {
    # An explicit parameter is intentional. Otherwise favour the vcpkg on PATH,
    # then VCPKG_ROOT, before using a repository-local checkout.
    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        if (-not (Test-VcpkgRoot $RequestedRoot)) {
            throw "-VcpkgRoot does not contain vcpkg.exe: $RequestedRoot"
        }
        return [System.IO.Path]::GetFullPath($RequestedRoot)
    }

    $PathVcpkg = Get-Command 'vcpkg.exe' -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $PathVcpkg) {
        $PathVcpkg = Get-Command 'vcpkg' -CommandType Application -ErrorAction SilentlyContinue
    }
    if ($null -ne $PathVcpkg) {
        $PathRoot = Split-Path -Parent $PathVcpkg.Source
        if (Test-VcpkgRoot $PathRoot) {
            return [System.IO.Path]::GetFullPath($PathRoot)
        }
    }

    if (Test-VcpkgRoot $env:VCPKG_ROOT) {
        return [System.IO.Path]::GetFullPath($env:VCPKG_ROOT)
    }

    if (-not (Test-VcpkgRoot $LocalRoot)) {
        if (-not (Test-Path -LiteralPath $LocalRoot)) {
            $git = Get-Command 'git.exe' -CommandType Application -ErrorAction SilentlyContinue
            if ($null -eq $git) {
                $git = Get-Command 'git' -CommandType Application -ErrorAction SilentlyContinue
            }
            if ($null -eq $git) {
                throw 'vcpkg was not found and git.exe is required to create the repository-local vcpkg checkout.'
            }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LocalRoot) | Out-Null
            Write-Host "Cloning vcpkg into $LocalRoot"
            & $git.Source clone --depth 1 https://github.com/microsoft/vcpkg.git $LocalRoot
            if ($LASTEXITCODE -ne 0) {
                throw 'Unable to clone the repository-local vcpkg checkout.'
            }
        }

        $Bootstrap = Join-Path $LocalRoot 'bootstrap-vcpkg.bat'
        if (-not (Test-Path -LiteralPath $Bootstrap)) {
            throw "vcpkg was not found and the repository-local checkout is incomplete: $LocalRoot"
        }
        Write-Host "Bootstrapping repository-local vcpkg at $LocalRoot"
        & $Bootstrap -disableMetrics | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-VcpkgRoot $LocalRoot)) {
            throw 'Unable to bootstrap the repository-local vcpkg checkout.'
        }
    }

    return [System.IO.Path]::GetFullPath($LocalRoot)
}

function Test-FfmpegVcpkgDependencies([string]$Root, [string]$Triplet) {
    $installed = Join-Path $Root "installed\\$Triplet"
    $pkgconf = @(
        (Join-Path $installed 'bin\pkgconf.exe'),
        (Join-Path $installed 'bin\pkg-config.exe'),
        (Join-Path $installed 'tools\pkgconf\pkgconf.exe'),
        (Join-Path $installed 'tools\pkgconf\pkg-config.exe')
    ) | Where-Object { Test-Path -LiteralPath $_ }
    return (Test-Path -LiteralPath (Join-Path $installed 'lib\dav1d.lib')) -and
        (Test-Path -LiteralPath (Join-Path $installed 'lib\jxl.lib')) -and
        (@($pkgconf).Count -gt 0)
}

$VcpkgRoot = Get-VcpkgRoot $VcpkgRoot $RepositoryVcpkgRoot
$VcpkgExecutable = Join-Path $VcpkgRoot 'vcpkg.exe'
$VcpkgInstallArguments = @(
    'install',
    "dav1d:$VcpkgTriplet",
    "libjxl:$VcpkgTriplet",
    "pkgconf:$VcpkgTriplet"
)
$ResolvedRepositoryVcpkgRoot = [System.IO.Path]::GetFullPath($RepositoryVcpkgRoot).TrimEnd('\')
$ResolvedVcpkgRoot = [System.IO.Path]::GetFullPath($VcpkgRoot).TrimEnd('\')
if ($ResolvedVcpkgRoot.Equals(
        $ResolvedRepositoryVcpkgRoot,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    $VcpkgDownloads = Join-Path $VcpkgRoot 'downloads'
    $VcpkgBinaryCache = Join-Path $VcpkgRoot 'binary-cache'
    New-Item -ItemType Directory -Force -Path $VcpkgDownloads, $VcpkgBinaryCache | Out-Null
    $VcpkgInstallArguments += @(
        "--vcpkg-root=$VcpkgRoot",
        "--downloads-root=$VcpkgDownloads",
        "--binarysource=clear;files,$VcpkgBinaryCache,readwrite"
    )
}
if (-not (Test-FfmpegVcpkgDependencies $VcpkgRoot $VcpkgTriplet)) {
    Write-Host "Installing FFmpeg vcpkg dependencies for $VcpkgTriplet"
    Push-Location -LiteralPath $VcpkgRoot
    try {
        & $VcpkgExecutable @VcpkgInstallArguments
        $VcpkgExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($VcpkgExitCode -ne 0 -or -not (Test-FfmpegVcpkgDependencies $VcpkgRoot $VcpkgTriplet)) {
        throw "Unable to install the required vcpkg dependencies for $VcpkgTriplet."
    }
}

$VcpkgInstalled = Join-Path $VcpkgRoot "installed\\$VcpkgTriplet"
$VcpkgPkgConfig = Join-Path $VcpkgInstalled 'lib\pkgconfig'
$VcpkgInclude = Join-Path $VcpkgInstalled 'include'
$VcpkgLib = Join-Path $VcpkgInstalled 'lib'
$VcpkgPkgConfCandidates = @(
    (Join-Path $VcpkgInstalled 'bin\pkgconf.exe'),
    (Join-Path $VcpkgInstalled 'bin\pkg-config.exe'),
    (Join-Path $VcpkgInstalled 'tools\pkgconf\pkgconf.exe'),
    (Join-Path $VcpkgInstalled 'tools\pkgconf\pkg-config.exe')
)
$VcpkgPkgConf = $VcpkgPkgConfCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($null -eq $VcpkgPkgConf) {
    throw 'pkgconf is required to detect vcpkg libdav1d/libjxl. Install pkgconf:x64-windows first.'
}

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
    'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\Launch-VsDevShell.ps1',
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
$VcpkgPkgConfBashPath = ConvertTo-BashPath $VcpkgPkgConf
$VcpkgPkgConfigWindowsPath = $VcpkgPkgConfig -replace '\\', '/'
$MsysPrelude = "export PATH=/usr/bin:/bin:`$PATH; export PKG_CONFIG='$VcpkgPkgConfBashPath'; export PKG_CONFIG_PATH='$VcpkgPkgConfigWindowsPath'; "
foreach ($MsysCommand in @('make', 'cmp')) {
    & $BashPath --noprofile --norc -lc "${MsysPrelude}command -v $MsysCommand >/dev/null"
    if ($LASTEXITCODE -ne 0) {
        throw "'$MsysCommand' was not found in MSYS2. Install the FFmpeg build tools with: pacman -S --needed make diffutils"
    }
}

# Both variants build private working copies; never modify the vendored source.
$ExpectedBuildRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $RepositoryRoot "third_party\ffmpeg-build\$VariantName"))
if ([System.IO.Path]::GetFullPath($BuildRoot) -ne $ExpectedBuildRoot) {
    throw 'Unexpected FFmpeg build directory.'
}
if ($Clean -and (Test-Path -LiteralPath $BuildRoot)) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
if (Test-Path -LiteralPath $SourceRoot) {
    if ([System.IO.Path]::GetFullPath($SourceRoot) -ne (Join-Path $ExpectedBuildRoot 'source')) {
        throw 'Unexpected FFmpeg source working directory.'
    }
    Remove-Item -LiteralPath $SourceRoot -Recurse -Force
}
Copy-Item -LiteralPath $UpstreamRoot -Destination $SourceRoot -Recurse
# Remove the success marker before any build, so a failed rebuild cannot be used.
$MarkerPath = Join-Path $Prefix 'dssim-ffmpeg-variant.txt'
Remove-Item -LiteralPath $MarkerPath -Force -ErrorAction SilentlyContinue

if ($Variant -eq 'Gpu') {
# FFmpeg commit b637624046a0 changed AV1 tile starts from superblock units to
# the mode-info units required by the Vulkan Video specification. The AMD
# proprietary Windows driver still interprets these values as superblock units,
# corrupting every tile after the first. Keep the upstream behavior everywhere
# else, and remove this compatibility patch once the AMD driver accepts MI units.
$VulkanAv1Source = Join-Path $SourceRoot 'libavcodec\vulkan_av1.c'
$VulkanAv1Original = '    uint16_t sb_shift = seq->use_128x128_superblock ? 5 : 4;'
$VulkanAv1Patched = @'
    const uint16_t sb_shift =
        dec->shared_ctx->s.driver_props.driverID == VK_DRIVER_ID_AMD_PROPRIETARY ?
        0 : (seq->use_128x128_superblock ? 5 : 4);
'@
$VulkanAv1PatchMarker =
    'dec->shared_ctx->s.driver_props.driverID == VK_DRIVER_ID_AMD_PROPRIETARY ?'
$VulkanAv1Text = Get-Content -LiteralPath $VulkanAv1Source -Raw
if ($VulkanAv1Text.Contains($VulkanAv1Original)) {
    $VulkanAv1Text = $VulkanAv1Text.Replace($VulkanAv1Original, $VulkanAv1Patched)
    Set-Content -LiteralPath $VulkanAv1Source -Value $VulkanAv1Text -NoNewline -Encoding utf8
} elseif (-not $VulkanAv1Text.Contains($VulkanAv1PatchMarker)) {
    throw 'FFmpeg vulkan_av1.c changed; the AMD AV1 tile-unit compatibility patch must be reviewed.'
}

}

# FFmpeg's native PNG decoder requires zlib. Keep it as an FFmpeg build
# dependency, but do not expose or link zlib/libpng from the application.
$ZlibArchive = Join-Path $BuildRoot "zlib-$ZlibVersion.zip"
if (-not (Test-Path -LiteralPath $ZlibSourceRoot)) {
    if (-not (Test-Path -LiteralPath $ZlibArchive)) {
        Write-Host "Downloading zlib $ZlibVersion from github.com"
        Invoke-WebRequest -Uri "https://github.com/madler/zlib/archive/refs/tags/v$ZlibVersion.zip" -OutFile $ZlibArchive
    }
    Expand-Archive -LiteralPath $ZlibArchive -DestinationPath $BuildRoot -Force
}

$ZlibLibrary = Join-Path $ZlibPrefix 'lib\zlibstatic.lib'
if (-not (Test-Path -LiteralPath $ZlibLibrary)) {
    Write-Host "Building zlib $ZlibVersion for FFmpeg"
    & cmake -S $ZlibSourceRoot -B $ZlibBuildRoot `
        -DBUILD_SHARED_LIBS=OFF -DZLIB_BUILD_SHARED=OFF -DZLIB_BUILD_TESTING=OFF `
        "-DCMAKE_INSTALL_PREFIX=$ZlibPrefix" `
        "-DINSTALL_LIB_DIR=$ZlibPrefix\lib" "-DINSTALL_BIN_DIR=$ZlibPrefix\bin" `
        "-DINSTALL_INC_DIR=$ZlibPrefix\include" "-DINSTALL_MAN_DIR=$ZlibPrefix\share\man" `
        "-DINSTALL_PKGCONFIG_DIR=$ZlibPrefix\share\pkgconfig"
    if ($LASTEXITCODE -ne 0) {
        throw 'zlib configure failed.'
    }
    & cmake --build $ZlibBuildRoot --config Release --target install
    if ($LASTEXITCODE -ne 0) {
        throw 'zlib build failed.'
    }
}

# FFmpeg's generated config.h defines HAVE_UNISTD_H while building on MSVC.
# zlib's zconf.h uses #ifdef for that macro, which would incorrectly include
# the POSIX-only unistd.h on Windows. Keep the generated zlib header usable by
# both the standalone zlib build and FFmpeg's MSVC compilation.
$ZconfPath = Join-Path $ZlibPrefix 'include\zconf.h'
if (Test-Path -LiteralPath $ZconfPath) {
    $zconfText = Get-Content -LiteralPath $ZconfPath -Raw
    $patchedZconfText = [regex]::Replace(
        $zconfText,
        '(?m)^#ifdef HAVE_UNISTD_H.*$',
        '#if defined(HAVE_UNISTD_H) && !defined(_WIN32)')
    if ($patchedZconfText -ne $zconfText) {
        Set-Content -LiteralPath $ZconfPath -Value $patchedZconfText -NoNewline -Encoding ascii
    }
}

$SourceBashPath = ConvertTo-BashPath $SourceRoot
$PrefixBashPath = ConvertTo-BashPath $Prefix
$ZlibIncludeBashPath = ConvertTo-BashPath (Join-Path $ZlibPrefix 'include')
$ZlibLibBashPath = ConvertTo-BashPath (Join-Path $ZlibPrefix 'lib')
$VcpkgIncludeBashPath = ConvertTo-BashPath $VcpkgInclude
$VcpkgLibBashPath = ConvertTo-BashPath $VcpkgLib
$HardwareArguments = if ($Variant -eq 'Gpu') {
    $VulkanIncludeBashPath = ConvertTo-BashPath (Join-Path $env:VULKAN_SDK 'Include')
    @('--enable-vulkan', '--disable-d3d11va',
      '--enable-hwaccel=h264_vulkan,hevc_vulkan,av1_vulkan,vp9_vulkan',
      "--extra-cflags=-I$VulkanIncludeBashPath")
} else {
    # Both D3D11VA switch forms are required for FFmpeg's shared dxva2 objects.
    @('--disable-vulkan', '--enable-d3d11va',
      '--enable-hwaccel=h264_d3d11va,hevc_d3d11va,av1_d3d11va,vp9_d3d11va,h264_d3d11va2,hevc_d3d11va2,av1_d3d11va2,vp9_d3d11va2')
}
$Parallelism = [Math]::Max(1, [Environment]::ProcessorCount)
$LinkageArguments = @('--enable-shared', '--disable-static')

$ConfigureArguments = @(
    '--toolchain=msvc', '--arch=x86_64',
    "--prefix=$PrefixBashPath"
) + $LinkageArguments + @(
    '--disable-programs', '--disable-doc', '--disable-debug',
    '--disable-autodetect', '--disable-network', '--disable-avdevice', '--disable-avfilter',
    '--disable-swresample', '--disable-encoders', '--disable-muxers',
    '--disable-filters', '--disable-bsfs', '--disable-protocols', '--disable-indevs', '--disable-outdevs',
    '--disable-decoders', '--disable-demuxers', '--disable-parsers', '--disable-hwaccels', '--disable-asm',
    '--enable-avutil', '--enable-avcodec', '--enable-avformat', '--enable-swscale', '--enable-zlib',
    "--pkg-config=$VcpkgPkgConfBashPath",
    '--enable-protocol=file', '--enable-demuxer=image2,mov,matroska',
    '--enable-parser=h264,hevc,av1,vp9,jpegxl',
    '--enable-decoder=h264,hevc,av1,vp9,libdav1d,png,mjpeg,libjxl,jpeg2000,webp',
    '--enable-libdav1d', '--enable-libjxl',
    # Rust's MSVC target uses the dynamic CRT. Match it in FFmpeg so the
    # FFmpeg objects do not introduce LIBCMT and cause LNK4098 at the final
    # executable link.
    # `-MD` is accepted by cl.exe and, unlike `/MD`, is not rewritten as an
    # MSYS2 path while it passes through bash.exe.
    '--extra-cflags=-MD',
    "--extra-cflags=-I$ZlibIncludeBashPath",
    "--extra-cflags=-I$VcpkgIncludeBashPath",
    "--extra-ldflags=-LIBPATH:$ZlibLibBashPath",
    "--extra-ldflags=-LIBPATH:$VcpkgLibBashPath",
    '--extra-libs=zlibstatic.lib'
) + $HardwareArguments

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
    # bin/. CMake discovers the import libraries under the prefix's lib/.
    foreach ($library in @('avcodec.lib', 'avformat.lib', 'avutil.lib', 'swscale.lib')) {
        $source = Join-Path $Prefix "bin\$library"
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Shared FFmpeg import library was not installed: $source"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $Prefix "lib\$library") -Force
    }
    $ZlibDll = Join-Path $ZlibPrefix 'bin\zlib.dll'
    if (Test-Path -LiteralPath $ZlibDll) {
        Copy-Item -LiteralPath $ZlibDll -Destination (Join-Path $Prefix 'bin\zlib.dll') -Force
    }
    $VcpkgBin = Join-Path $VcpkgInstalled 'bin'
    if (Test-Path -LiteralPath $VcpkgBin) {
        Get-ChildItem -LiteralPath $VcpkgBin -Filter '*.dll' -File | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Prefix 'bin\' $_.Name) -Force
        }
    }
}

Set-Content -LiteralPath $MarkerPath -Value "$VariantName-shared-v1" -Encoding ascii
Write-Host "Minimal $Variant $Linkage FFmpeg installed at $Prefix"
Write-Host 'Configure and build the Vulkan GPU executable with:'
Write-Host '  & cmake -S . -B build'
Write-Host '  & cmake --build build --config Release --target dssim_vulkan'
