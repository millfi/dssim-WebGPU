# dssim-WebGPU

An optimized C++20/Vulkan implementation of the DSSIM image comparison
algorithm. The executable and CMake target retain their existing names for
command-line and build compatibility.

## Requirements

- Windows with a Vulkan-capable GPU and driver exposing:
  - Vulkan 1.3
  - `VK_EXT_shader_object`
  - `VK_KHR_push_descriptor`
  - `synchronization2` and `dynamicRendering` Vulkan 1.3 features
- The Vulkan SDK, including the Vulkan loader library and headers, and `glslc`
- PowerShell
- CMake 3.24 or newer
- A C++20 compiler

The top-level CMake configuration selects C++20 automatically. No additional
C++ standard flag is required.

CMake locates the SDK with `find_package(Vulkan REQUIRED COMPONENTS glslc)`.
The standard Vulkan SDK installation sets `VULKAN_SDK`, which CMake can use to
find these components. There is no automatic SDK download.

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

The root configuration keeps the existing command line and delegates the GPU
target to `src_gpu/CMakeLists.txt`. During the build, `glslc` compiles the GLSL
compute shaders in `src_gpu/shaders` to SPIR-V beside the executable:

```text
build\src_gpu\Release\shaders\*.spv
```

## Compare one pair

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png
```

The score and compared path are printed to stdout:

```text
0.00328441    .\tests\laptop.q24.jpegli.jpg.png
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

`--stdin-pairs` creates the Vulkan instance and device, loads the SPIR-V
modules, and creates the shader objects once, then reuses them for every pair.
Image resolution is supplied through push constants and dispatch dimensions,
so different resolutions do not require different shader objects.

`--stdin-pairs` cannot be combined with `--out` or `--debug-dump-dir`.

## Mechanical score regression check

The regression checker compares the Vulkan implementation against the original
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
- runs all Vulkan comparisons in one `--stdin-pairs` session
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
milliseconds, plus an independent Vulkan timestamp-query result when the
selected compute queue supports timestamp queries.

Session initialization:

- `session_init_pipeline_setup_ms`: pipeline layout and shader object creation
- `session_init_resource_prep_ms`: session-level resource preparation
- `session_init_gpu_submit_wait_ms`: session-level GPU submission/waiting
- `session_init_gpu_timestamp_ms`: session-level Vulkan timestamp-query duration
- `session_init_cpu_postprocess_ms`: session-level CPU post-processing
- `session_init_other_ms`: uncategorized session initialization work

Each comparison:

- `pipeline_setup_ms`: per-comparison shader setup
- `resource_prep_ms`: buffer creation, uploads, and resource binding preparation
- `gpu_submit_wait_ms`: CPU wall time for command recording/submission plus readback
  waiting
- `gpu_timestamp_ms`: actual GPU execution duration measured by Vulkan timestamp
  queries, when supported
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
shader execution. `readback_ms` includes GPU completion and host readback time.
`gpu_timestamp_ms` may overlap the wall-clock buckets because CPU and GPU work
asynchronously, so it is not included in the mutually exclusive total.
The two per-scale post-process fields are independent durations that overlap
when the parallel aggregation path is active.
The JSON field names are retained for compatibility. With shader objects,
`create_pso_ms` and `create_bind_group_ms` are expected to be zero; shader
object and pipeline-layout work is recorded in the corresponding existing
buckets. Timestamp-query support is optional: comparisons and wall-clock
profiling still work when the selected queue does not support it.

## Current performance design

- Shader objects are created once per process and reused across all resolutions.
- Stage buffers grow to the largest encountered image and are reused across
  scale levels and subsequent pairs.
- Push descriptors avoid descriptor-pool and descriptor-set allocation.
- Debug-statistics resources use a separate cache from the normal benchmark
  path.
- sRGB-to-linear conversion uses a 256-entry lookup table.
- The normal path converts both inputs and constructs every pyramid level on
  the GPU without an intermediate CPU round-trip.
- The debug path uses parallel CPU conversion and pyramid construction so it
  can retain and write intermediate scale data.
- Only the SSIM map is read back during normal execution.

## Optimization and score policy

The current priority is reducing end-to-end latency while preserving scores.

- Identical images must produce `0.00000000`.
- Other pairs must remain below 1% relative error versus `dssim.exe`.
- Floating-point precision and algebraic transformations may change only when
  the regression check remains within tolerance.
- Blur weights and SSIM constants must not change.
- Keep all shader dispatches two-dimensional with GLSL
  `layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;` and
  dispatch dimensions
  `(ceil(width / 16), ceil(height / 16), 1)`.

## Vulkan SDK and shaders

Install the Vulkan SDK before configuring. The build requires its loader
library, headers, and `glslc`; configuration fails with a clear error when any
required component is unavailable. You can verify the shader compiler from
PowerShell with:

```powershell
& glslc --version
```

The build compiles these GLSL compute shaders to SPIR-V; shaders are not
compiled at application startup:

- `rgba8_to_linear.comp`
- `downsample_2x2.comp`
- `lab_preprocess.comp`
- `stage0_absdiff.comp`
- `stage0_score.comp`

The reference source under `src_reference/` remains available for studying
algorithm details. Automated regression validation intentionally uses the
`dssim.exe` selected from `PATH`.
