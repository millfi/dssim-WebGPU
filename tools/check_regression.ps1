[CmdletBinding()]
param(
    [string]$PairList = ".\tests\test_pairs.txt",
    [string]$GpuExecutable = ".\build\src_gpu\Release\dssim-WebGPU.exe",
    [double]$RelativeTolerance = 0.01,
    [string]$IdentityImagePath = ".\tests\gradation.png",
    [string]$IdentityVideoPath = ".\benchmark\3s.webm"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$invariant = [Globalization.CultureInfo]::InvariantCulture
$scorePattern = '^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+'

function Resolve-ExistingPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
}

function Parse-Score {
    param(
        [Parameter(Mandatory = $true)][string[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Source
    )

    foreach ($line in $Lines) {
        $match = [regex]::Match($line, $scorePattern)
        if ($match.Success) {
            return [ordered]@{
                Text = $match.Groups[1].Value
                Value = [double]::Parse($match.Groups[1].Value, $invariant)
            }
        }
    }
    throw "Could not parse a score from $Source output."
}

function Assert-FiniteZero {
    param(
        [Parameter(Mandatory = $true)][System.Collections.IDictionary]$Score,
        [Parameter(Mandatory = $true)][string]$Source
    )

    if ([double]::IsNaN($Score.Value) -or [double]::IsInfinity($Score.Value)) {
        throw "$Source produced a non-finite score: $($Score.Text)."
    }
    if ($Score.Value -ne 0.0) {
        throw "$Source expected a zero score, got $($Score.Text)."
    }
}

function Invoke-IdentityImageCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string]$ImagePath
    )

    $lines = @(
        & $Executable $ImagePath $ImagePath 2>&1 |
            ForEach-Object { $_.ToString() }
    )
    if ($LASTEXITCODE -ne 0) {
        throw "GPU identical-image check exited with code $LASTEXITCODE.`n$($lines -join [Environment]::NewLine)"
    }
    $score = Parse-Score -Lines $lines -Source "GPU identical-image check"
    Assert-FiniteZero -Score $score -Source "GPU identical-image check"
    return $score
}

function Invoke-IdentityVideoCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string]$VideoPath
    )

    $lines = @(
        & $Executable $VideoPath $VideoPath 2>&1 |
            ForEach-Object { $_.ToString() }
    )
    if ($LASTEXITCODE -ne 0) {
        throw "GPU identical-video check exited with code $LASTEXITCODE.`n$($lines -join [Environment]::NewLine)"
    }

    $summaryPattern = '^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+.*frames=(\d+)\s*$'
    $summary = $null
    foreach ($line in $lines) {
        $match = [regex]::Match($line, $summaryPattern)
        if ($match.Success) {
            $summary = $match
        }
    }
    if ($null -eq $summary) {
        throw "Could not parse the final identical-video summary."
    }

    $score = [ordered]@{
        Text = $summary.Groups[1].Value
        Value = [double]::Parse($summary.Groups[1].Value, $invariant)
    }
    $frames = [int]::Parse($summary.Groups[2].Value, $invariant)
    if ($frames -le 0) {
        throw "GPU identical-video check produced no decoded frames."
    }
    Assert-FiniteZero -Score $score -Source "GPU identical-video check"
    return [ordered]@{
        Score = $score
        Frames = $frames
    }
}

$repositoryRoot = Resolve-ExistingPath (Join-Path $PSScriptRoot "..")
$pairListPath = Resolve-ExistingPath $PairList
$gpuPath = Resolve-ExistingPath $GpuExecutable
$referencePath = Join-Path $repositoryRoot "src_reference\target\release\dssim.exe"
$referenceManifest = Join-Path $repositoryRoot "src_reference\Cargo.toml"
$ffmpegRoot = Join-Path $repositoryRoot "third_party\ffmpeg-8.1.2-shared"

if (-not (Test-Path -LiteralPath $referencePath -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $referenceManifest -PathType Leaf)) {
        throw "Reference Cargo manifest was not found: $referenceManifest"
    }
    if (-not (Test-Path -LiteralPath $ffmpegRoot -PathType Container)) {
        throw "FFmpeg prefix was not found: $ffmpegRoot. Build it with tools/build_ffmpeg_minimal.ps1 first."
    }
    $cargoCommand = Get-Command cargo.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $cargoCommand) {
        throw "cargo.exe was not found; it is required to build the local reference executable."
    }

    $previousFfmpegDir = $env:FFMPEG_DIR
    try {
        $env:FFMPEG_DIR = (Resolve-ExistingPath $ffmpegRoot)
        Write-Host "Building local reference executable: $referencePath"
        & $cargoCommand.Source build --manifest-path $referenceManifest --release --features video
        if ($LASTEXITCODE -ne 0) {
            throw "Reference cargo build exited with code $LASTEXITCODE."
        }
    } finally {
        if ($null -eq $previousFfmpegDir) {
            Remove-Item Env:FFMPEG_DIR -ErrorAction SilentlyContinue
        } else {
            $env:FFMPEG_DIR = $previousFfmpegDir
        }
    }
}

$referencePath = Resolve-ExistingPath $referencePath
$ffmpegBin = Join-Path $ffmpegRoot "bin"
if (Test-Path -LiteralPath $ffmpegBin -PathType Container) {
    $env:PATH = "$(Resolve-ExistingPath $ffmpegBin);$env:PATH"
}

$pairs = @()
$lineNumber = 0
foreach ($line in Get-Content -LiteralPath $pairListPath) {
    $lineNumber++
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }

    $columns = $line -split "`t"
    if ($columns.Count -ne 2) {
        throw "${pairListPath}:${lineNumber}: expected exactly two tab-delimited paths."
    }

    $pairs += [pscustomobject]@{
        Line = $line
        Image1 = Resolve-ExistingPath $columns[0]
        Image2 = Resolve-ExistingPath $columns[1]
    }
}

if ($pairs.Count -eq 0) {
    throw "No image pairs found in $pairListPath."
}

$gpuLines = @(
    Get-Content -LiteralPath $pairListPath |
        & $gpuPath --stdin-pairs 2>&1 |
        ForEach-Object { $_.ToString() }
)
if ($LASTEXITCODE -ne 0) {
    throw "GPU implementation exited with code $LASTEXITCODE.`n$($gpuLines -join [Environment]::NewLine)"
}

$gpuScores = @()
foreach ($line in $gpuLines) {
    if ([regex]::IsMatch($line, $scorePattern)) {
        $gpuScores += Parse-Score -Lines @($line) -Source "GPU"
    }
}
if ($gpuScores.Count -ne $pairs.Count) {
    throw "GPU returned $($gpuScores.Count) scores for $($pairs.Count) pairs."
}

$results = @()
$failed = $false
for ($i = 0; $i -lt $pairs.Count; $i++) {
    $pair = $pairs[$i]
    $referenceLines = @(
        & $referencePath $pair.Image1 $pair.Image2 2>&1 |
            ForEach-Object { $_.ToString() }
    )
    if ($LASTEXITCODE -ne 0) {
        throw "dssim.exe exited with code $LASTEXITCODE for pair $($i + 1)."
    }

    $reference = Parse-Score -Lines $referenceLines -Source "dssim.exe"
    $gpu = $gpuScores[$i]
    if ($reference.Value -eq 0.0) {
        $relativeError = [Math]::Abs($gpu.Value - $reference.Value)
        $passed = $relativeError -eq 0.0
    } else {
        $relativeError = [Math]::Abs($gpu.Value - $reference.Value) / [Math]::Abs($reference.Value)
        $passed = $relativeError -lt $RelativeTolerance
    }

    if (-not $passed) {
        $failed = $true
    }

    $results += [pscustomobject]@{
        Pair = $i + 1
        Reference = $reference.Text
        GPU = $gpu.Text
        RelativeErrorPercent = "{0:F4}" -f ($relativeError * 100.0)
        Result = if ($passed) { "PASS" } else { "FAIL" }
        Image2 = $pair.Image2
    }
}

$results | Format-Table -AutoSize
Write-Host ("Reference executable: {0}" -f $referencePath)
Write-Host ("Tolerance: relative error < {0:P2}" -f $RelativeTolerance)

$identityImagePath = Resolve-ExistingPath $IdentityImagePath
$identityVideoPath = Resolve-ExistingPath $IdentityVideoPath
$identityImageScore = Invoke-IdentityImageCheck -Executable $gpuPath -ImagePath $identityImagePath
$identityVideoResult = Invoke-IdentityVideoCheck -Executable $gpuPath -VideoPath $identityVideoPath
Write-Host ("Identity image: PASS score={0} path={1}" -f $identityImageScore.Text, $identityImagePath)
Write-Host ("Identity video: PASS score={0} frames={1} path={2}" -f $identityVideoResult.Score.Text, $identityVideoResult.Frames, $identityVideoPath)

if ($failed) {
    Write-Error "Score regression check failed."
    exit 1
}

Write-Host "All score regression checks passed."
