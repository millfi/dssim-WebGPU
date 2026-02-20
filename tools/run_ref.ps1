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
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Resolve-OptionalArgs {
    param(
        [string[]]$ArgsList
    )

    $options = @{
        Out = $null
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

function To-HexU64 {
    param(
        [Parameter(Mandatory = $true)]
        [double]$Value
    )

    $bits = [System.BitConverter]::DoubleToInt64Bits($Value)
    return ('0x{0:X16}' -f ([UInt64]$bits))
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
$commandString = 'dssim "{0}" "{1}"' -f $image1Path, $image2Path

$payload = [ordered]@{
    schema_version = 1
    engine = "ref-dssim-cli"
    status = "error"
    input = [ordered]@{
        image1 = $image1Path
        image2 = $image2Path
    }
    command = $commandString
    version = $null
    result = $null
    stdout = $null
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
}

try {
    $null = Get-Command dssim -ErrorAction Stop

    $helpLines = @(& dssim -h 2>&1 | ForEach-Object { $_.ToString() })
    $versionLine = $helpLines | Where-Object { $_ -match '^Version\s+' } | Select-Object -First 1
    if ($null -eq $versionLine -or [string]::IsNullOrWhiteSpace($versionLine)) {
        $versionLine = "Version:unknown"
    }
    $payload.version = $versionLine

    $runLines = @(& dssim $image1Path $image2Path 2>&1 | ForEach-Object { $_.ToString() })
    $exitCode = $LASTEXITCODE
    $stdoutText = ($runLines -join [Environment]::NewLine).TrimEnd()
    $payload.stdout = $stdoutText

    if ($exitCode -ne 0) {
        throw "dssim exited with code $exitCode"
    }

    $scoreLine = $runLines | Where-Object {
        $_ -match '^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s+'
    } | Select-Object -First 1

    if ($null -eq $scoreLine) {
        throw "Could not find score line in dssim output."
    }

    $match = [regex]::Match(
        $scoreLine,
        '^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+(.+?)\s*$'
    )
    if (-not $match.Success) {
        throw "Could not parse score line: $scoreLine"
    }

    $scoreText = $match.Groups[1].Value
    $comparedPath = $match.Groups[2].Value
    $scoreValue = [double]::Parse($scoreText, [Globalization.CultureInfo]::InvariantCulture)
    $scoreBits = To-HexU64 -Value $scoreValue

    $payload.status = "ok"
    $payload.result = [ordered]@{
        score_text = $scoreText
        score_f64 = $scoreValue
        score_bits_u64 = $scoreBits
        compared_path = $comparedPath
    }

    if (-not [string]::IsNullOrWhiteSpace($options.DebugDumpDir)) {
        $bytes1 = [System.IO.File]::ReadAllBytes($image1Path)
        $bytes2 = [System.IO.File]::ReadAllBytes($image2Path)
        $payload.debug_dumps = [ordered]@{
            stage0_absdiff_u32le = Write-AbsDiffU32LeDebugDump `
                -Bytes1 $bytes1 `
                -Bytes2 $bytes2 `
                -DumpDir $options.DebugDumpDir `
                -Suffix "ref"
        }
    }
} catch {
    $payload.error = $_.Exception.Message
}

Ensure-ParentDirectory -FilePath $outPath
$payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outPath -Encoding UTF8

if ($payload.status -ne "ok") {
    Write-Error ("run_ref failed: {0}" -f $payload.error)
    exit 1
}

Write-Host ("[run_ref] wrote {0}" -f $outPath)
