# dssim-Vulkan


This is an accelerated implementation of the DSSIM image similarity evaluation algorithm using C++20 and Vulkan, with added video support.


## Requirements


- Windows with a Vulkan GPU and driver supporting the following features:
  - Vulkan 1.3
  - `VK_EXT_shader_object`
  - `VK_KHR_push_descriptor`
  - Vulkan 1.3 `synchronization2` and `dynamicRendering` features
- Vulkan SDK including the Vulkan loader library and headers, and `glslc`
- PowerShell
- CMake 3.24 or later
- C++20 compatible compiler
- For video comparison, a Vulkan Video decode queue and `VK_KHR_video_decode_h264`, `VK_KHR_video_decode_h265`, `VK_KHR_video_decode_vp9`, or `VK_KHR_video_decode_av1` corresponding to each input codec


The top-level CMake configuration automatically selects C++20, so you do not need to specify additional C++ standard flags during the build.


CMake detects the SDK using `find_package(Vulkan REQUIRED COMPONENTS glslc)`. CMake can use `VULKAN_SDK`, which is set by standard Vulkan SDK installations. There is no automated SDK retrieval process.


Inputs are either PNG images of the same width and height, or two videos. Video containers are determined by the extensions `.mp4`, `.m4v`, `.mov`, `.mkv`, and `.webm`. `AV_PIX_FMT_VULKAN` frames are received from the FFmpeg Vulkan Video decoder (H.264/HEVC/VP9/AV1), and NV12 or P010 Vulkan images are converted to RGBA8 on the GPU. There is no CPU readback of decoded frames.


## build
```powershell
& .\tools\build_gpu.ps1
```


or


```powershell
& cmake -S . -B build
& cmake --build build --config Release --target dssim_vulkan
```


`build_gpu.ps1` verifies the archive with a fixed SHA-256 before extraction. If the archive is missing or modified, the same archive is downloaded from the current commit on GitHub `origin`, verified, and then replaced. `build_ffmpeg_minimal.ps1` can be used to regenerate the archive contents. Both downloaded vcpkg download files and binary caches are stored under `third_party/vcpkg` within the repository, and the user-common AppData cache is not used.


The executable is generated at the following location:


```text
build\src_gpu\Release\dssim-Vulkan.exe
```


The root CMake configuration maintains compatibility with conventional configure commands, delegating GPU target definitions to `src_gpu/CMakeLists.txt`. During the build, `glslc` compiles the GLSL compute shaders in `src_gpu/shaders` to SPIR-V and places them next to the executable:


```text
build\src_gpu\Release\shaders\*.spv
```


## Comparing a single pair


```powershell
& .\build\src_gpu\Release\dssim-Vulkan.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png
```


The standard output displays the score and the paths of the comparison targets.


```text
0.00328441    .\tests\laptop.q24.jpegli.jpg.png
```


To display timing, add `--profiling`.


```powershell
& .\build\src_gpu\Release\dssim-Vulkan.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png `
    --profiling
```


To output scale-by-scale results and detailed timing to JSON, use `--out <json>`.


```powershell
& .\build\src_gpu\Release\dssim-Vulkan.exe `
    .\tests\laptop.png `
    .\tests\laptop.q24.jpegli.jpg.png `
    --profiling `
    --out .\out\gpu.json
```


To output intermediate buffers for score investigation, specify `--debug-dump-dir <directory>`. This option is for PNG comparison.


## Comparing videos


Similar to image pairs, specify two video paths.


```powershell
& .\build\src_gpu\Release\dssim-Vulkan.exe `
    .\benchmark\x264_medium_g40_fastdecode_crf40.mp4 `
    .\benchmark\3s.webm `
    --profiling `
    --csv .\out\video_scores.csv
```


Both inputs must be videos. An error will occur if the width and height of the corresponding decoded frames, as well as the number of decoded frames in both videos, do not match. Comparisons are performed in zero-based decode order; timestamp alignment, retime, and resample are not performed.


During processing, stderr displays the FPS, current in-flight pipeline depth, configured pipeline capacity, number of processed frames, elapsed time, DSSIM of the immediately preceding frame, and cumulative average DSSIM. Upon completion, stdout displays the average DSSIM for all frame pairs and `frames=<N>`.


```text
0.06837245    .\benchmark\3s.webm    frames=180
```


`--csv <path>` outputs `time_seconds,frame_number,dssim`, one row per frame pair. `time_seconds` is a timestamp where the first decoded frame of the first video is 0 seconds. `--out <json>` outputs the aggregated results and detailed profiling values summing up the values for each frame.


Each of the two videos is processed by its dedicated FFmpeg decode thread and passed to the comparison pipeline. The number of frame pairs processed simultaneously can be specified with `--pipeline-depth <N>`, which takes a positive integer and defaults to 3. Upon startup, each codec and the selected Vulkan Video queue family are displayed. If possible, the two are assigned to separate compatible queue families, but if both are AV1, they share the primary queue family.


## Fixed multi-pair benchmark


Use `tests/test_pairs.txt` as a runnable benchmark list. Each line, excluding empty lines, describes two image paths separated by a tab.


Benchmarks are run with the following fixed command:


```powershell
Get-Content .\tests\test_pairs.txt |
    & .\build\src_gpu\Release\dssim-Vulkan.exe --stdin-pairs --profiling
```


With `--stdin-pairs`, a Vulkan instance and device are created, SPIR-V is loaded, and shader objects are created once within the process and reused for all pairs. Because image resolutions are specified by push constants and dispatch counts, there is no need to recreate shader objects even if resolutions differ.


`--stdin-pairs` cannot be used simultaneously with `--out`, `--csv`, `--pipeline-depth`, or `--debug-dump-dir`.


## Automated Testing

The test script `check_regression.ps1` compares the Vulkan version scores against the original `dssim.exe` resolved from the PATH.

```powershell
& .\tools\check_regression.ps1
```

You can verify the reference executable actually selected using the following command:

```powershell
(Get-Command dssim.exe -CommandType Application).Source
```

The test script performs the following process:

- Reads all pairs in `tests/test_pairs.txt`
- Runs the Vulkan version in a single `--stdin-pairs` session
- Runs the original `dssim.exe` for each pair
- Requires `0.00000000` for identical image comparisons
- Requires a relative error of less than 1% for all other comparisons
- Displays results in a tabular format and exits with a non-zero code if there are any violations

To override arguments, run the script as follows:

```powershell
& .\tools\check_regression.ps1 `
    -PairList .\tests\test_pairs.txt `
    -GpuExecutable .\build\src_gpu\Release\dssim-Vulkan.exe `
    -RelativeTolerance 0.01
```


## Profiling Output

When `--profiling` is specified, mutually exclusive wall-clock time intervals and independent Vulkan timestamp query results are displayed in milliseconds. Timestamp query results are obtained only if supported by the selected compute queue.

Session initialization:

- `session_init_pipeline_setup_ms`: Creation of pipeline layout and shader objects
- `session_init_resource_prep_ms`: Session-level resource preparation
- `session_init_gpu_submit_wait_ms`: Session-level GPU submission and waiting
- `session_init_gpu_timestamp_ms`: Vulkan timestamp query time per session
- `session_init_cpu_postprocess_ms`: Session-level CPU post-processing
- `session_init_other_ms`: Session initialization other than the above

Each comparison:

- `pipeline_setup_ms`: Shader preparation per comparison
- `resource_prep_ms`: Buffer creation, uploading, and resource binding preparation
- `gpu_submit_wait_ms`: CPU wall time for command recording, submission, and readback waiting
- `gpu_timestamp_ms`: Actual GPU execution time measured via Vulkan timestamp queries (obtained only on supported queues)
- `cpu_postprocess_ms`: CPU-side score aggregation
- `other_ms`: Unclassified processing after decode completion, such as color conversion and image pyramid generation

Using `--out <json>` outputs the following detailed items to the `profiling` object:

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

`dispatch_and_submit_ms` is the CPU-side command construction and submission time, not the pure shader execution time. `readback_ms` includes GPU completion waiting and host readback time. Because the CPU and GPU overlap asynchronously, `gpu_timestamp_ms` is not included in the sum of wall-clock time intervals. The two scale-specific post-process items are also independent times that overlap with each other during parallel aggregation. JSON field names are maintained for compatibility. Since shader objects are used, `create_pso_ms` and `create_bind_group_ms` are expected to be 0. Shader object and pipeline layout processing are accounted for in the corresponding existing buckets. Timestamp queries are an optional feature; comparison processing and wall-clock profiling will work even if the queue does not support them.

## Current Optimization Design

- Shader objects are created only once per process and reused across all resolutions
- Staging buffers are expanded up to the largest image processed so far and reused for each scale and subsequent pairs
- Push descriptors eliminate descriptor pool and descriptor set allocations
- Debug statistics resources are cached separately from the regular benchmark path
- A 256-element lookup table is used for sRGB-to-linear conversion
- In the normal path, two inputs are converted on the GPU, and image pyramids for all scales are built without any intermediate round-trip to the CPU
- In the debug path, pixel conversion and image pyramid generation are performed in parallel on the CPU so that intermediate scale data can be retained and output
- Normal execution reads back only the SSIM map

## Implementation Constraints

- Identical images/videos must output `0.00000000` without running the identical check
- Other pairs must maintain a relative error of less than 1% compared to `dssim.exe`
- As long as the above conditions are met, floating-point additions and multiplications may be transformed assuming the associative law holds
- Blur weights and SSIM constants must not be changed

## Vulkan SDK and Shaders

Install the Vulkan SDK before configuring. The build requires the SDK's loader library, headers, and `glslc`; configure will terminate with a clear error if any of these are missing. You can check the shader compiler from PowerShell:

```powershell
& glslc --version
```

The following GLSL compute shaders are compiled to SPIR-V during the build. Shader compilation is not performed at application startup.

- `rgba8_to_linear.comp`
- `vulkan_yuv_to_rgba8.comp`
- `downsample_2x2.comp`
- `lab_preprocess.comp`
- `stage0_absdiff.comp`
- `stage0_score.comp`

To facilitate implementation while comparing with the original dssim algorithm, the source code of the original implementation is retained in `src_reference/`.
