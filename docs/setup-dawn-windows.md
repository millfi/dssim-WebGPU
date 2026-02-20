# Dawn native setup on Windows (PowerShell 7)

This document sets up Dawn in `third_party/dawn` and builds the C++ GPU sample targets.

## 1) Prerequisites

- Visual Studio 2022 with C++ workload
- CMake 3.24+
- Python 3.10+
- Git
- PowerShell 7

## 2) Get `depot_tools`

```powershell
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git third_party/depot_tools
$env:PATH = "$(Resolve-Path .\third_party\depot_tools);$env:PATH"
```

Persist PATH for future shells as needed.

## 3) Fetch Dawn source

```powershell
New-Item -ItemType Directory -Path third_party -Force | Out-Null
Push-Location third_party
fetch --nohooks dawn
Pop-Location
```

Expected source path: `third_party/dawn`.

## 4) Sync and build Dawn (Release)

```powershell
Push-Location .\third_party\dawn
gclient sync
gn gen out/Release --args="is_debug=false dcheck_always_on=false dawn_build_tests=false dawn_enable_opengl=false"
ninja -C out/Release dawn_native dawn_proc webgpu_dawn
Pop-Location
```

## 5) Build this repository GPU targets

Dummy target only (always available):

```powershell
cmake -S . -B build
cmake --build build --config Release --target dssim_gpu_dummy
```

Enable Dawn scaffold target:

```powershell
cmake -S . -B build `
  -DDSSIM_ENABLE_DAWN_SAMPLE=ON `
  -DDSSIM_DAWN_ROOT="$PWD/third_party/dawn" `
  -DDSSIM_DAWN_OUT_DIR="$PWD/third_party/dawn/out/Release"

cmake --build build --config Release --target dssim_gpu_dawn_checksum
```

## 6) Run end-to-end scripts

Reference JSON:

```powershell
.\tools\run_ref.ps1 .\tests\gray-profile.png .\tests\gray-profile2.png --out .\out\ref.json
```

GPU JSON (auto-detect Dawn or dummy executable):

```powershell
.\tools\run_gpu.ps1 .\tests\gray-profile.png .\tests\gray-profile2.png --out .\out\gpu.json
```

If `python` is not on PATH in the current shell, use:

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\Python\Python312;$env:PATH"
```

Exact compare:

```powershell
python .\tools\compare.py .\out\ref.json .\out\gpu.json
```

## Notes

- Current `dssim_gpu_dawn_checksum` is scaffolding: it creates Dawn adapter/device and writes placeholder score JSON.
- Replace placeholder path with deterministic WGSL compute + readback stages and compare against reference after each step.
- Keep strict determinism for reductions and float handling to satisfy exact-match policy.
