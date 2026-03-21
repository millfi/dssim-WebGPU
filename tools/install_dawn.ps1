param(
    [Parameter(Mandatory = $true)]
    [string]$DawnRoot,

    [Parameter(Mandatory = $true)]
    [string]$DawnOutDir,

    [Parameter(Mandatory = $true)]
    [string]$DepotToolsDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Resolve-FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    try {
        return (Resolve-Path -LiteralPath $PathValue -ErrorAction Stop).Path
    } catch {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter()]
        [string[]]$Arguments = @(),

        [string]$WorkingDirectory
    )

    if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        Push-Location $WorkingDirectory
    }

    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
        }
    } finally {
        if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
            Pop-Location
        }
    }
}

function Get-CommandPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $cmd) {
        throw "Required command not found on PATH: $Name"
    }
    return $cmd.Source
}

$DawnRoot = Resolve-FullPath $DawnRoot
$DawnOutDir = Resolve-FullPath $DawnOutDir
$DepotToolsDir = Resolve-FullPath $DepotToolsDir
$workspaceDir = Split-Path -Parent $DawnRoot
$dawnDirName = Split-Path -Leaf $DawnRoot

New-Item -ItemType Directory -Path $workspaceDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $DepotToolsDir)) {
    $gitPath = Get-CommandPath "git"
    Invoke-Native -FilePath $gitPath -Arguments @(
        "clone",
        "https://chromium.googlesource.com/chromium/tools/depot_tools.git",
        $DepotToolsDir
    ) -WorkingDirectory $workspaceDir
}

$env:PATH = "$DepotToolsDir;$env:PATH"
if (-not $env:DEPOT_TOOLS_WIN_TOOLCHAIN) {
    $env:DEPOT_TOOLS_WIN_TOOLCHAIN = "0"
}

$gclientPath = Get-CommandPath "gclient"
$gnPath = Get-CommandPath "gn"
$ninjaPath = Get-CommandPath "ninja"
$gclientConfigPath = Join-Path $workspaceDir ".gclient"
$dawnLooksLikeCheckout = (Test-Path -LiteralPath (Join-Path $DawnRoot "DEPS")) -or `
    (Test-Path -LiteralPath (Join-Path $DawnRoot ".git"))

if ((Test-Path -LiteralPath $DawnRoot) -and -not $dawnLooksLikeCheckout) {
    throw "Dawn root exists but is not a full Dawn checkout: $DawnRoot"
}

if (-not (Test-Path -LiteralPath $gclientConfigPath)) {
    Invoke-Native -FilePath $gclientPath -Arguments @(
        "config",
        "--name",
        $dawnDirName,
        "https://dawn.googlesource.com/dawn"
    ) -WorkingDirectory $workspaceDir
}

Invoke-Native -FilePath $gclientPath -Arguments @("sync") -WorkingDirectory $workspaceDir

if (-not (Test-Path -LiteralPath $DawnRoot)) {
    throw "Dawn root was not created by gclient sync: $DawnRoot"
}

$gnArgs = "is_debug=false dcheck_always_on=false dawn_build_tests=false dawn_enable_opengl=false"
Invoke-Native -FilePath $gnPath -Arguments @("gen", $DawnOutDir, "--args=$gnArgs") -WorkingDirectory $DawnRoot
Invoke-Native -FilePath $ninjaPath -Arguments @(
    "-C",
    $DawnOutDir,
    "dawn_native.dll",
    "dawn_proc.dll",
    "webgpu_dawn.dll"
) -WorkingDirectory $DawnRoot

$requiredOutputs = @(
    (Join-Path $DawnOutDir "dawn_native.dll"),
    (Join-Path $DawnOutDir "dawn_proc.dll"),
    (Join-Path $DawnOutDir "webgpu_dawn.dll")
)

foreach ($path in $requiredOutputs) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Expected Dawn runtime was not produced: $path"
    }
}

Write-Host ("[install_dawn] ready: {0}" -f $DawnOutDir)
