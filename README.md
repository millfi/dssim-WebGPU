## WebGPU (Dawn) status in this fork

This repository also contains an experimental WebGPU implementation in `src_gpu/`.

- Binary: `dssim_gpu_dawn_checksum`
- Input format today: PNG only (decoded with `libpng`)
- Runtime dependency inside GPU binary: no `dssim` CLI dependency
- Language standard: C++20 (set by CMake defaults; no extra build flag required)
- Current state: working end-to-end, but not bit-exact with the reference yet on every image pair

As of February 21, 2026, for `tests/1440p.png` vs `tests/1440p.jxl.png`:

- Reference (`dssim` CLI): `0.00044658`
- WebGPU (`dssim_gpu_dawn_checksum`): `0.00044680`

### Build and run (PowerShell)

No explicit C++ standard option is needed. `CMakeLists.txt` enforces C++20 globally.

```powershell
cmake -S . -B build `
  -DDSSIM_DAWN_ROOT="<path-to-dawn-src>" `
  -DDSSIM_DAWN_OUT_DIR="$(Resolve-Path .\third_party\dawn\out\Release)"

cmake --build build --config Release --target dssim_gpu_dawn_checksum

$env:PATH = "$(Resolve-Path .\third_party\dawn\out\Release);$env:PATH"

.\build\src_gpu\Release\dssim_gpu_dawn_checksum.exe `
  .\tests\gray-profile.png .\tests\gray-profile2.png `
  --out .\out\gpu.json `
  --debug-dump-dir .\out\debug
```

If `--out` is omitted, the score is printed to stdout.

If Dawn is not available (for example after deleting `third_party/dawn`), CMake tries to auto-install it by default.  
You can explicitly disable the sample with `-DDSSIM_ENABLE_DAWN_SAMPLE=OFF`.

### Profiling output

The executable prints aggregated profiling metrics in milliseconds:

- `decode_done_to_score_ms`
- `CreateShaderModule processing time`
- `CreatePSO processing time`
- `CreateBuffer processing time`
- `WriteInputBuffer processing time`
- `CreatePipelineLayout processing time`
- `CreateBindGroup processing time`
- `DispatchAndSubmit processing time`
- `Readback processing time`
- `PostProcess processing time`

Interpretation notes:

- `DispatchAndSubmit` is CPU-side command encoding/submission overhead, not pure WGSL kernel execution time.
- `Readback` includes waiting for GPU completion and map/readback overhead.

When `--out <json>` is specified, the same aggregated values are written to the top-level `profiling` object:

- `decode_done_to_score_ms`
- `create_shader_module_ms`
- `create_pso_ms`
- `create_buffer_ms`
- `write_input_buffer_ms`
- `create_pipeline_layout_ms`
- `create_bind_group_ms`
- `dispatch_and_submit_ms`
- `readback_ms`
- `post_process_ms`
### Auto-install Dawn (Windows)

By default, CMake fetches/builds Dawn automatically when it is missing:

```powershell
cmake -S . -B build
cmake --build build --config Release --target dssim_gpu_dawn_checksum
```

This invokes `tools/install_dawn.ps1`, which installs `third_party/depot_tools`, fetches `third_party/dawn`, and builds `dawn_native`, `dawn_proc`, and `webgpu_dawn`.

To disable this behavior, pass `-DDSSIM_AUTO_INSTALL_DAWN=OFF`.

### Notes

- Images must have the same width/height.
- `--debug-dump-dir` emits intermediate GPU buffers for mismatch analysis.
- The default backend on Windows is D3D12.


