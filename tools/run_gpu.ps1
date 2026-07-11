param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$Image1,

    [Parameter(Position = 1, Mandatory = $true)]
    [string]$Image2,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-OptionalArgs {
    param(
        [string[]]$ArgsList
    )

    $options = @{
        Out = $null
        Exe = $null
        DebugDumpDir = $null
    }

    $argItems = @()
    if ($null -ne $ArgsList) {
        $argItems = @($ArgsList)
    }

    for ($i = 0; $i -lt $argItems.Count; $i++) {
        $arg = $argItems[$i]

        if ($arg -eq "--out" -or $arg -eq "-out" -or $arg -eq "-o") {
            if ($i + 1 -ge $argItems.Count) {
                throw "Missing value for $arg"
            }
            $options.Out = $argItems[$i + 1]
            $i++
            continue
        }
        if ($arg.StartsWith("--out=")) {
            $options.Out = $arg.Substring("--out=".Length)
            continue
        }

        if ($arg -eq "--exe") {
            if ($i + 1 -ge $argItems.Count) {
                throw "Missing value for --exe"
            }
            $options.Exe = $argItems[$i + 1]
            $i++
            continue
        }
        if ($arg.StartsWith("--exe=")) {
            $options.Exe = $arg.Substring("--exe=".Length)
            continue
        }

        if ($arg -eq "--debug-dump-dir") {
            if ($i + 1 -ge $argItems.Count) {
                throw "Missing value for --debug-dump-dir"
            }
            $options.DebugDumpDir = $argItems[$i + 1]
            $i++
            continue
        }
        if ($arg.StartsWith("--debug-dump-dir=")) {
            $options.DebugDumpDir = $arg.Substring("--debug-dump-dir=".Length)
            continue
        }

        throw "Unknown argument: $arg"
    }

    if ([string]::IsNullOrWhiteSpace($options.Out)) {
        throw "Missing required argument: --out <path>"
    }

    return $options
}

function Resolve-PathSafe {
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

function Get-FirstExistingPath {
    param(
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return $null
}

$options = Resolve-OptionalArgs -ArgsList $RemainingArgs
$outPath = [System.IO.Path]::GetFullPath($options.Out)
$image1Path = Resolve-PathSafe $Image1
$image2Path = Resolve-PathSafe $Image2
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$exePath = $null
if (-not [string]::IsNullOrWhiteSpace($options.Exe)) {
    $exePath = Resolve-PathSafe $options.Exe
} else {
    $candidates = @(
        (Join-Path $repoRoot "build/src_gpu/Release/dssim-WebGPU.exe"),
        (Join-Path $repoRoot "build/src_gpu/Debug/dssim-WebGPU.exe"),
        (Join-Path $repoRoot "build/src_gpu/dssim-WebGPU.exe")
    )
    $exePath = Get-FirstExistingPath -Candidates $candidates
}

if ([string]::IsNullOrWhiteSpace($exePath) -or
    -not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
    throw "Vulkan GPU executable not found. Build target dssim_webgpu or pass --exe <path>."
}

$exeArgs = @($image1Path, $image2Path, "--out", $outPath)
if (-not [string]::IsNullOrWhiteSpace($options.DebugDumpDir)) {
    $exeArgs += @(
        "--debug-dump-dir",
        ([System.IO.Path]::GetFullPath($options.DebugDumpDir))
    )
}
& $exePath @exeArgs
if ($LASTEXITCODE -ne 0) {
    throw "GPU executable failed with code ${LASTEXITCODE}: $exePath"
}

Write-Host ("[run_gpu] wrote {0} via {1}" -f $outPath, $exePath)
