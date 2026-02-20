param(
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
        OutDir = (Join-Path $PSScriptRoot "golden")
    }

    $argItems = @()
    if ($null -ne $ArgsList) {
        $argItems = @($ArgsList)
    }

    for ($i = 0; $i -lt $argItems.Count; $i++) {
        $arg = $argItems[$i]
        if ($arg -eq "--out-dir" -or $arg -eq "-out-dir") {
            if ($i + 1 -ge $argItems.Count) {
                throw "Missing value for $arg"
            }
            $options.OutDir = $argItems[$i + 1]
            $i++
            continue
        }
        if ($arg.StartsWith("--out-dir=")) {
            $options.OutDir = $arg.Substring("--out-dir=".Length)
            continue
        }
        throw "Unknown argument: $arg"
    }

    return $options
}

$options = Resolve-OptionalArgs -ArgsList $RemainingArgs
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outDir = [System.IO.Path]::GetFullPath($options.OutDir)
$runRef = Join-Path $PSScriptRoot "run_ref.ps1"

$pairs = @(
    @{ Name = "gray-profile__gray-profile2"; Image1 = "tests/gray-profile.png"; Image2 = "tests/gray-profile2.png" },
    @{ Name = "profile__profile-stripped"; Image1 = "tests/profile.png"; Image2 = "tests/profile-stripped.png" },
    @{ Name = "test1-sm__test2-sm"; Image1 = "tests/test1-sm.png"; Image2 = "tests/test2-sm.png" }
)

New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$records = @()

foreach ($pair in $pairs) {
    $image1Path = Join-Path $repoRoot $pair.Image1
    $image2Path = Join-Path $repoRoot $pair.Image2
    $jsonPath = Join-Path $outDir ("{0}.ref.json" -f $pair.Name)

    & $runRef $image1Path $image2Path --out $jsonPath
    if ($LASTEXITCODE -ne 0) {
        throw "run_ref failed for pair: $($pair.Image1) <-> $($pair.Image2)"
    }

    $records += [ordered]@{
        name = $pair.Name
        image1 = $pair.Image1
        image2 = $pair.Image2
        output_json = $jsonPath
    }
}

$manifest = [ordered]@{
    schema_version = 1
    generator = "tools/make_golden.ps1"
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    reference_engine = "ref-dssim-cli"
    pairs = $records
}

$manifestPath = Join-Path $outDir "golden_manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host ("[make_golden] wrote {0}" -f $manifestPath)
