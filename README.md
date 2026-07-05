# dssim-WebGPU

An optimized C++20/WebGPU implementation of the DSSIM image comparison
algorithm. The native GPU executable uses Dawn and WGSL compute shaders.

## Requirements

- Windows with a D3D12-capable GPU
- PowerShell
- CMake 3.24 or newer
- A C++20 compiler

The top-level CMake configuration selects C++20 automatically. No additional
C++ standard flag is required.

Both input images must be PNG files with identical dimensions.

## Build

Run all repository commands from PowerShell:

```powershell
& cmake -S . -B build
& cmake --build build --config Release --target dssim_webgpu
```

The executable is written to:

```text
build\src_gpu\Release\dssim-WebGPU.exe
```

## Compare one pair

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png
```

The score and compared path are printed to stdout:

```text
0.00328379    .\tests\laptop.q24.jpegli.jpg.png
```

Add `--profiling` to print timing information:

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png `
    --profiling
```

Add `--out <json>` for per-scale results and detailed timings:

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png `
    --profiling `
    --out .\out\gpu.json
```

Add `--debug-dump-dir <directory>` to emit intermediate buffers used for
score-matching investigations.

## Fixed multi-pair benchmark

Use `tests/test_pairs.txt` as the executable benchmark list. Each non-empty
line contains two tab-delimited paths.

Run the benchmark with this command:

```powershell
Get-Content .\tests\test_pairs.txt |
    & .\build\src_gpu\Release\dssim-WebGPU.exe --stdin-pairs --profiling
```

`--stdin-pairs` creates the WebGPU device, shader modules, layouts, and PSOs
once, then reuses them for every pair. Image resolution is supplied through
uniform buffers and dispatch dimensions, so different resolutions do not
require different PSOs.

`--stdin-pairs` cannot be combined with `--out` or `--debug-dump-dir`.

## Mechanical score regression check

The regression checker compares the WebGPU implementation against the original
`dssim.exe` resolved from `PATH`:

```powershell
& .\tools\check_regression.ps1
```

To inspect the selected reference executable:

```powershell
(Get-Command dssim.exe -CommandType Application).Source
```

The checker:

- reads every pair from `tests/test_pairs.txt`
- runs all WebGPU comparisons in one `--stdin-pairs` session
- runs the original `dssim.exe` for each pair
- requires `0.00000000` for identical-image comparisons
- requires relative error below 1% for other comparisons
- prints a result table and exits nonzero if any comparison fails

Useful overrides:

```powershell
& .\tools\check_regression.ps1 `
    -PairList .\tests\test_pairs.txt `
    -GpuExecutable .\build\src_gpu\Release\dssim-WebGPU.exe `
    -RelativeTolerance 0.01
```

Run this check after every score-affecting or performance change and before
committing an optimization.

## Profiling output

`--profiling` prints mutually exclusive wall-clock timing buckets in
milliseconds, plus an independent WebGPU Timestamp Query result.

Session initialization:

- `session_init_pipeline_setup_ms`: shader modules, pipeline layouts, and PSOs
- `session_init_resource_prep_ms`: session-level resource preparation
- `session_init_gpu_submit_wait_ms`: session-level GPU submission/waiting
- `session_init_gpu_timestamp_ms`: session-level GPU Timestamp Query duration
- `session_init_cpu_postprocess_ms`: session-level CPU post-processing
- `session_init_other_ms`: uncategorized session initialization work

Each comparison:

- `pipeline_setup_ms`: per-comparison shader and pipeline setup
- `resource_prep_ms`: buffer creation, uploads, and Bind Group creation
- `gpu_submit_wait_ms`: CPU wall time for dispatch/submission plus readback/map
  waiting
- `gpu_timestamp_ms`: actual GPU execution duration measured by WebGPU
  Timestamp Query
- `cpu_postprocess_ms`: CPU-side score aggregation
- `other_ms`: color conversion, pyramid construction, and other uncategorized
  work after decoding

When `--out <json>` is used, the `profiling` object contains the finer-grained
fields:

- `decode_done_to_score_ms`
- `create_shader_module_ms`
- `create_pso_ms`
- `create_buffer_ms`
- `write_input_buffer_ms`
- `create_pipeline_layout_ms`
- `create_bind_group_ms`
- `dispatch_and_submit_ms`
- `readback_ms`
- `gpu_submit_wait_ms`
- `gpu_timestamp_ms`
- `post_process_base_scale_ms`
- `post_process_remaining_scales_ms`
- `post_process_ms`

`dispatch_and_submit_ms` measures CPU command encoding/submission, not pure
shader execution. `readback_ms` includes GPU completion and mapping wait time.
`gpu_timestamp_ms` may overlap the wall-clock buckets because CPU and GPU work
asynchronously, so it is not included in the mutually exclusive total.
The two per-scale post-process fields are independent durations that overlap
when the parallel aggregation path is active.
Profiling requires an adapter that supports the WebGPU `TimestampQuery`
feature.

## Current performance design

- PSOs are created once per process and reused across all resolutions.
- Stage buffers and Bind Groups grow to the largest encountered image and are
  reused across scale levels and subsequent pairs.
- Debug-statistics resources use a separate cache from the normal benchmark
  path.
- sRGB-to-linear conversion uses a 256-entry lookup table.
- The two input images are converted and downsampled in parallel.
- CPU pixel conversion and pyramid construction use the same parallel path for
  all image sizes; there is no separate small-image fallback.
- Only the SSIM map is read back during normal execution.

## Optimization and score policy

The current priority is reducing end-to-end latency while preserving scores.

- Identical images must produce `0.00000000`.
- Other pairs must remain below 1% relative error versus `dssim.exe`.
- Floating-point precision and algebraic transformations may change only when
  the regression check remains within tolerance.
- Blur weights and SSIM constants must not change.
- Keep all shader dispatches two-dimensional with
  `@workgroup_size(16, 16, 1)` and dispatch dimensions
  `(ceil(width / 16), ceil(height / 16), 1)`.

## Dawn setup

If Dawn is missing, CMake automatically invokes `tools/install_dawn.ps1` to
prepare `third_party/depot_tools`, fetch Dawn into `third_party/dawn`, and build
the required Dawn libraries.

Disable automatic installation with:

```powershell
& cmake -S . -B build -DDSSIM_AUTO_INSTALL_DAWN=OFF
```

Disable the WebGPU target entirely with:

```powershell
& cmake -S . -B build -DDSSIM_ENABLE_DAWN_SAMPLE=OFF
```

The reference source under `src_reference/` remains available for studying
algorithm details. Automated regression validation intentionally uses the
`dssim.exe` selected from `PATH`.
