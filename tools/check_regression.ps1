[CmdletBinding()]
param(
    [string]$PairList = ".\tests\test_pairs.txt",
    [string]$GpuExecutable = ".\build\src_gpu\Release\dssim-WebGPU.exe",
    [double]$RelativeTolerance = 0.01
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

$pairListPath = Resolve-ExistingPath $PairList
$gpuPath = Resolve-ExistingPath $GpuExecutable
$referenceCommand = Get-Command dssim.exe -CommandType Application -ErrorAction Stop |
    Select-Object -First 1

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
        & $referenceCommand.Source $pair.Image1 $pair.Image2 2>&1 |
            ForEach-Object { $_.ToString() }
    )
    if ($LASTEXITCODE -ne 0) {
        throw "dssim.exe exited with code $LASTEXITCODE for pair $($i + 1)."
    }

    $reference = Parse-Score -Lines $referenceLines -Source "dssim.exe"
    $gpu = $gpuScores[$i]
    $sameImage = $pair.Image1 -eq $pair.Image2

    if ($sameImage) {
        $relativeError = 0.0
        $passed = $gpu.Text -eq "0.00000000"
    } elseif ($reference.Value -eq 0.0) {
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
Write-Host ("Reference executable: {0}" -f $referenceCommand.Source)
Write-Host ("Tolerance: relative error < {0:P2}" -f $RelativeTolerance)

if ($failed) {
    Write-Error "Score regression check failed."
    exit 1
}

Write-Host "All score regression checks passed."
