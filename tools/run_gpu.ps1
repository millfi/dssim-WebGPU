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

function Ensure-ParentDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath
    )

    $parent = Split-Path -Parent $FilePath
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function To-HexU64 {
    param(
        [Parameter(Mandatory = $true)]
        [double]$Value
    )

    $bits = [System.BitConverter]::DoubleToInt64Bits($Value)
    return ('0x{0:X16}' -f ([UInt64]$bits))
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

function Write-AbsDiffU32LeDebugDump {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Bytes1,
        [Parameter(Mandatory = $true)]
        [byte[]]$Bytes2,
        [Parameter(Mandatory = $true)]
        [string]$DumpDir,
        [Parameter(Mandatory = $true)]
        [string]$Suffix
    )

    $absDir = [System.IO.Path]::GetFullPath($DumpDir)
    New-Item -ItemType Directory -Path $absDir -Force | Out-Null

    $len = [Math]::Max($Bytes1.Length, $Bytes2.Length)
    $buf = New-Object byte[] ($len * 4)
    for ($i = 0; $i -lt $len; $i++) {
        $a = 0
        if ($i -lt $Bytes1.Length) { $a = [int]$Bytes1[$i] }
        $b = 0
        if ($i -lt $Bytes2.Length) { $b = [int]$Bytes2[$i] }
        $diff = [uint32][Math]::Abs($a - $b)
        [System.BitConverter]::GetBytes($diff).CopyTo($buf, $i * 4)
    }

    $path = Join-Path $absDir ("stage0_absdiff_u32le.{0}.bin" -f $Suffix)
    [System.IO.File]::WriteAllBytes($path, $buf)

    return [ordered]@{
        path = [System.IO.Path]::GetFullPath($path)
        elem_type = "u32_le"
        elem_count = $len
    }
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
        (Join-Path $repoRoot "build/src_gpu/Release/dssim_gpu_dawn_checksum.exe"),
        (Join-Path $repoRoot "build/src_gpu/Debug/dssim_gpu_dawn_checksum.exe"),
        (Join-Path $repoRoot "build/src_gpu/dssim_gpu_dawn_checksum.exe"),
        (Join-Path $repoRoot "build/src_gpu/Release/dssim_gpu_dummy.exe"),
        (Join-Path $repoRoot "build/src_gpu/Debug/dssim_gpu_dummy.exe"),
        (Join-Path $repoRoot "build/src_gpu/dssim_gpu_dummy.exe"),
        (Join-Path $repoRoot "build/Release/dssim_gpu_dummy.exe"),
        (Join-Path $repoRoot "build/Debug/dssim_gpu_dummy.exe"),
        (Join-Path $repoRoot "build/dssim_gpu_dummy.exe")
    )
    $exePath = Get-FirstExistingPath -Candidates $candidates
}

if (-not [string]::IsNullOrWhiteSpace($exePath)) {
    $exeLeaf = [System.IO.Path]::GetFileName($exePath)
    $supportsDebugArg = $exeLeaf -ieq "dssim_gpu_dawn_checksum.exe"
    $exeArgs = @($image1Path, $image2Path, "--out", $outPath)
    if ($supportsDebugArg -and -not [string]::IsNullOrWhiteSpace($options.DebugDumpDir)) {
        $exeArgs += @("--debug-dump-dir", ([System.IO.Path]::GetFullPath($options.DebugDumpDir)))
    }
    & $exePath @exeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "GPU executable failed with code ${LASTEXITCODE}: $exePath"
    }

    if ((-not $supportsDebugArg) -and -not [string]::IsNullOrWhiteSpace($options.DebugDumpDir)) {
        $bytes1 = [System.IO.File]::ReadAllBytes($image1Path)
        $bytes2 = [System.IO.File]::ReadAllBytes($image2Path)
        $obj = Get-Content -LiteralPath $outPath -Raw | ConvertFrom-Json -AsHashtable
        $obj["debug_dumps"] = [ordered]@{
            stage0_absdiff_u32le = Write-AbsDiffU32LeDebugDump `
                -Bytes1 $bytes1 `
                -Bytes2 $bytes2 `
                -DumpDir $options.DebugDumpDir `
                -Suffix "gpu"
        }
        $obj | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outPath -Encoding UTF8
    }

    Write-Host ("[run_gpu] wrote {0} via {1}" -f $outPath, $exePath)
    exit 0
}

$bytes1 = [System.IO.File]::ReadAllBytes($image1Path)
$bytes2 = [System.IO.File]::ReadAllBytes($image2Path)

$sum1 = [UInt64]0
foreach ($value in $bytes1) { $sum1 += [UInt64]$value }
$sum2 = [UInt64]0
foreach ($value in $bytes2) { $sum2 += [UInt64]$value }

$lenMax = [Math]::Max($bytes1.Length, $bytes2.Length)
$denominator = [Math]::Max(1.0, [double]$lenMax * 255.0)
$score = [Math]::Abs([double]$sum1 - [double]$sum2) / $denominator
$scoreText = $score.ToString("F8", [Globalization.CultureInfo]::InvariantCulture)

$payload = [ordered]@{
    schema_version = 1
    engine = "gpu-placeholder-powershell"
    status = "ok"
    input = [ordered]@{
        image1 = $image1Path
        image2 = $image2Path
    }
    command = "tools/run_gpu.ps1 `"$image1Path`" `"$image2Path`" --out `"$outPath`""
    version = "placeholder-0"
    result = [ordered]@{
        score_text = $scoreText
        score_f64 = $score
        score_bits_u64 = To-HexU64 -Value $score
        compared_path = $image2Path
    }
    note = "No GPU executable found. Emitted deterministic placeholder score."
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
}

if (-not [string]::IsNullOrWhiteSpace($options.DebugDumpDir)) {
    $payload.debug_dumps = [ordered]@{
        stage0_absdiff_u32le = Write-AbsDiffU32LeDebugDump `
            -Bytes1 $bytes1 `
            -Bytes2 $bytes2 `
            -DumpDir $options.DebugDumpDir `
            -Suffix "gpu"
    }
}

Ensure-ParentDirectory -FilePath $outPath
$payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outPath -Encoding UTF8
Write-Host ("[run_gpu] wrote placeholder JSON {0}" -f $outPath)
