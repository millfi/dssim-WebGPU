## WebGPU (Dawn) status in this fork

This repository also contains an experimental WebGPU implementation in `src_gpu/`.

- Binary: `dssim-WebGPU`
- Input format today: PNG only (decoded with `libpng`)
- Runtime dependency inside GPU binary: no `dssim` CLI dependency
- Language standard: C++20 (set by CMake defaults; no extra build flag required)
- Current state: working end-to-end, but not bit-exact with the reference yet on every image pair

As of February 21, 2026, for `tests/1440p.png` vs `tests/1440p.jxl.png`:

- Reference (dssim v3.4.0): `0.00044658`
- WebGPU : `0.00044330`
`.\tests\0.02s.png` vs `.\tests\0s.png`:
- Reference (dssim v3.4.0): `0.00113154`
- WebGPU : `0.00112473`
### Build and run (PowerShell)

No explicit C++ standard option is needed. `CMakeLists.txt` enforces C++20 globally.

```powershell
cmake -S . -B build

cmake --build build --config Release --target dssim_webgpu

.\build\src_gpu\Release\dssim-WebGPU.exe `
  .\tests\gray-profile.png .\tests\gray-profile2.png `
  --profiling `
  --out .\out\gpu.json `
  --debug-dump-dir .\out\debug
```

If `--out` is omitted, the score is printed to stdout.

If `--profiling` is omitted, profiling is not printed to stdout.

To reuse the same WebGPU device and pipelines across multiple comparisons in one process, use
`--stdin-pairs` and provide one tab-delimited pair per line on stdin:

```powershell
$tab = [char]9
@(
  ".\tests\gradation.png${tab}.\tests\gradation-fs8.png"
  ".\tests\gradation.png${tab}.\tests\gradation-256.png"
) | .\build\src_gpu\Release\dssim-WebGPU.exe --stdin-pairs --profiling
```

If Dawn is not available (for example after deleting `third_party/dawn`), CMake tries to auto-install it by default.  
You can explicitly disable the sample with `-DDSSIM_ENABLE_DAWN_SAMPLE=OFF`.

### Profiling output

When `--profiling` is specified, the executable prints MECE profiling buckets in milliseconds:

- `session_init_total_ms`
- `session_init_pipeline_setup_ms`
- `session_init_resource_prep_ms`
- `session_init_gpu_execution_ms`
- `session_init_cpu_postprocess_ms`
- `session_init_other_ms`
- `total_ms`
- `pipeline_setup_ms`
- `resource_prep_ms`
- `gpu_execution_ms`
- `cpu_postprocess_ms`
- `other_ms`

Interpretation notes:

- `pipeline_setup_ms` is shader module creation, pipeline layout creation, and PSO creation.
- `resource_prep_ms` is buffer creation, buffer upload, and bind group creation.
- `gpu_execution_ms` is dispatch/submit plus readback/map wait.

When `--out <json>` is specified, the raw timing breakdown is written to the top-level `profiling` object:

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
cmake --build build --config Release --target dssim_webgpu
```

This invokes `tools/install_dawn.ps1`, which installs `third_party/depot_tools`, fetches `third_party/dawn`, and builds `dawn_native`, `dawn_proc`, and `webgpu_dawn`.

To disable this behavior, pass `-DDSSIM_AUTO_INSTALL_DAWN=OFF`.

### Notes

- Images must have the same width/height.
- `--debug-dump-dir` emits intermediate GPU buffers for mismatch analysis.
- The default backend on Windows is D3D12.



### Score-matching workflow

For score-matching work, use `tests/test_list.csv` as the main regression list.
The intended loop is:

1. Build `dssim-WebGPU`.
2. Run every image pair listed in `tests/test_list.csv`.
3. Update the `dssim-WebGPU` column in `tests/test_list.csv` with the newly measured scores.
4. Compare the updated GPU scores against `reference_score(dssim v3.4.0)` and focus on shrinking the gap.

Current priority order:

- identical-image comparisons must reach `0.00000000`
- `gradation.png` vs `gradation-fs8.png` must stop showing a very large relative error
- only after those are fixed should optimization work resume

### Reference implementation

For reproducible score-matching work, prefer keeping an upstream `dssim` checkout under `src_reference/`.
If a locally built reference binary is available from that checkout, use it for validation in preference to a `dssim.exe` found on `PATH`.
Using the `PATH` binary is acceptable only when it is known to match the checked out source/version.

