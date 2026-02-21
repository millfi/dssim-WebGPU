## WebGPU (Dawn) status in this fork

This repository also contains an experimental WebGPU implementation in `src_gpu/`.

- Binary: `dssim_gpu_dawn_checksum`
- Input format today: PNG only (decoded with `libpng`)
- Runtime dependency inside GPU binary: no `dssim` CLI dependency
- Current state: working end-to-end, but not bit-exact with the reference yet on every image pair

As of February 21, 2026, for `tests/1440p.png` vs `tests/1440p.jxl.png`:

- Reference (`dssim` CLI): `0.00044658`
- WebGPU (`dssim_gpu_dawn_checksum`): `0.00044680`

### Build and run (PowerShell)

```powershell
cmake -S . -B build -DDSSIM_ENABLE_DAWN_SAMPLE=ON `
  -DDSSIM_DAWN_ROOT="$PWD/third_party/dawn" `
  -DDSSIM_DAWN_OUT_DIR="$PWD/third_party/dawn/out/Release"

cmake --build build --config Release --target dssim_gpu_dawn_checksum

$env:PATH = "$(Resolve-Path .\third_party\dawn\out\Release);$env:PATH"

.\build\src_gpu\Release\dssim_gpu_dawn_checksum.exe `
  .\tests\gray-profile.png .\tests\gray-profile2.png `
  --out .\out\gpu.json `
  --debug-dump-dir .\out\debug
```

If `--out` is omitted, the score is printed to stdout.

### Notes

- Images must have the same width/height.
- `--debug-dump-dir` emits intermediate GPU buffers for mismatch analysis.
- The default backend on Windows is D3D12.
