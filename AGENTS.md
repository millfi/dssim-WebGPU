# AGENTS

## Build policy

- This repository must build as C++20 by default.
- Do not require users to pass an extra C++ standard flag in build commands.
- CMake sets this at the top level (`CMAKE_CXX_STANDARD=20`, `CMAKE_CXX_STANDARD_REQUIRED=ON`, `CMAKE_CXX_EXTENSIONS=OFF`).
- Use PowerShell for every repository command. Do not provide Command Prompt, batch, Bash, or POSIX-shell commands.
- Invoke executables and scripts with PowerShell's call operator (`&`) when appropriate.
- `dssim_webgpu` always links the minimal FFmpeg Vulkan Video build under
  `third_party/ffmpeg-8.1.2-shared`, including for PNG-only work. If it is absent,
  create it with `& .\tools\build_ffmpeg_minimal.ps1 -Linkage Dynamic` before configuring.

## Verification workflow

1. Configure with normal command:
   - `& cmake -S . -B build`
2. Build target:
   - `& cmake --build build --config Release --target dssim_webgpu`
3. Run the fixed multi-pair benchmark command:
   - `Get-Content .\tests\test_pairs.txt | & .\build\src_gpu\Release\dssim-WebGPU.exe --stdin-pairs --profiling`
4. Mechanically compare scores with the original `dssim.exe` found on `PATH`:
   - `& .\tools\check_regression.ps1`

The fixed regression list currently covers PNG comparisons. Changes to video
decoding, Vulkan YUV conversion, frame scheduling, or video output also require
the video verification workflow below; video checks do not replace the PNG
regression check.

## Video support and verification

- Video inputs are recognized by `.mp4`, `.m4v`, `.mov`, `.mkv`, and `.webm`.
- Supported codecs are H.264, HEVC, VP9, and AV1 through FFmpeg Vulkan Video.
- Video comparison requires two video inputs with the same decoded frame count
  and matching dimensions for every corresponding frame. Frames are paired by
  zero-based decode order; do not describe the implementation as timestamp
  alignment or frame-rate conversion.
- The accepted decoded Vulkan formats are NV12 and P010. Conversion to RGBA8 is
  performed by `vulkan_yuv_to_rgba8.comp` without CPU frame readback.
- `--csv` is supported only for video comparisons, and `--pipeline-depth`
  affects only the video pipeline. `--stdin-pairs` cannot be combined with
  `--out`, `--csv`, `--pipeline-depth`, or `--debug-dump-dir`.

For every video-affecting change:

1. Build `dssim_webgpu` and run `& .\tools\check_regression.ps1`.
2. Run the repository video pair with profiling and CSV output:
   - `& .\build\src_gpu\Release\dssim-WebGPU.exe .\benchmark\x264_medium_g40_fastdecode_crf40.mp4 .\benchmark\3s.webm --profiling --csv .\out\video_scores.csv`
3. Require a successful exit, a finite average DSSIM, and a nonzero
   `frames=<N>` value. Confirm that the CSV header is
   `time_seconds,frame_number,dssim`, frame numbers are contiguous from 0, and
   the number of data rows equals `N`.
4. When changing pipeline scheduling, repeat the video comparison with at least
   `--pipeline-depth 1` and the default depth 3. Scores and frame counts must be
   identical; compare timings only after correctness is established.

Do not add hand-authored video reference scores. If a stable video score
regression is needed, implement a mechanical comparison against an appropriate
reference executable or a reproducibly generated same-input result.

## Current priority: Performance optimization

- Score-matching is done. All test pairs in `tests/test_pairs.txt` are within 0.2% relative error of the reference.
- Current priority is **reducing end-to-end latency** while keeping scores within the regression tolerance.

### Regression tolerance

- Same-image comparison must still produce `0.00000000`.
- Do not implement special-case identical-input detection for images or videos
  (including path equality, encoded/decoded byte comparison, or hashes), and do
  not override the resulting score based on such a check. An identical-input
  score of zero must emerge from the normal DSSIM computation path.
- All other test pairs in `tests/test_pairs.txt`: **relative error < 1%** against the `dssim.exe` resolved from `PATH`.
- After every optimization, re-run all pairs and confirm the tolerance holds before committing.
- Do not replace the mechanical `dssim.exe` comparison with hand-edited reference scores.

### Permitted precision trade-offs

- Replacing f64 with f32 in CPU-side aggregation is allowed if it stays within tolerance.
- Algebraically equivalent transformations that differ under floating-point arithmetic (e.g. `a*b + a*c` → `a*(b+c)`, reordering FMA, strength reduction) are allowed.
- Reducing intermediate precision (e.g. f32 → f16 for non-critical paths) is allowed if it stays within tolerance.
- Changing blur kernel weights or SSIM constants is **not** allowed.

### Optimization workflow

1. Identify the bottleneck using `--profiling` or `--out <json>` (see Profiling output section).
2. Hypothesize an optimization.
3. Implement the change.
4. Build: `& cmake --build build --config Release --target dssim_webgpu`
5. Run `& .\tools\check_regression.ps1` and confirm every pair passes.
6. Measure timing improvement with `--profiling`.
7. If improved without regression: commit with a clear message describing the optimization and measured speedup.
8. If regression: revert and try a different approach.
9. Repeat.

### Known bottlenecks and opportunities

- `lab_preprocess.comp` cooperatively converts a 20×20 LAB tile for each 16×16 workgroup, so neighboring output pixels already reuse LAB conversion work.
- The normal batch path keeps all scale levels on-GPU in one command buffer, one queue submission, and one readback. The debug path intentionally uses per-scale readback for intermediate statistics.
- Upload, device-local workspace, and readback arenas are reused across `--stdin-pairs`; a larger comparison can still trigger arena growth.
- Shader objects and layouts are created once per session. The preprocess and 5×5 SSIM dispatches remain the main GPU execution cost.
- Video decoding uses one FFmpeg thread per input and overlaps decode with GPU
  comparison through a bounded frame-pair queue (default depth 3). Preserve
  frame ordering, bounded in-flight ownership, and decode-thread shutdown on
  success and failure when optimizing this path.

## Score-matching workflow (reference — complete)

- Treat `tests/test_pairs.txt` as the executable regression list.
- For each validation pass:
  1. Build `dssim-WebGPU`.
  2. Run the fixed benchmark command.
  3. Run `& .\tools\check_regression.ps1`.
  4. Require every pair to pass before committing.
- `tools/check_regression.ps1` resolves `dssim.exe` from `PATH`, runs both implementations, and exits nonzero on a tolerance violation.
- Do not update reference scores by hand.
- Use `--out <json>` for per-scale inspection and `--debug-dump-dir` for buffer-level investigation.

## Reference implementation

- Use the original `dssim.exe` resolved from `PATH` for mechanical regression validation.
- Confirm the resolved executable when needed with `(Get-Command dssim.exe -CommandType Application).Source`.
- The reference source under `src_reference/` can be read to understand algorithmic details when needed.

## Profiling output

- When `--profiling` is specified, wall-clock MECE profiling buckets are printed in milliseconds:
  - `pipeline_setup_ms`: shader object + descriptor/pipeline layout creation
  - `resource_prep_ms`: arena buffer creation + mapped input upload (push descriptors require no bind-group allocation)
  - `gpu_submit_wait_ms`: CPU wall time for dispatch/submit + readback/map wait
  - `cpu_postprocess_ms`: CPU-side score aggregation
  - `other_ms`: uncategorized overhead
- `gpu_timestamp_ms` is the independent GPU execution duration measured with Vulkan
  timestamp queries. It can overlap CPU wall-clock buckets and is not added to the MECE total.
- When `--out <json>` is specified, finer-grained timing is in the `profiling` object:
  - `create_shader_module_ms`, `create_pso_ms`, `create_buffer_ms`, `write_input_buffer_ms`, `create_pipeline_layout_ms`, `create_bind_group_ms`, `dispatch_and_submit_ms`, `readback_ms`, `gpu_submit_wait_ms`, `gpu_timestamp_ms`, `post_process_base_scale_ms`, `post_process_remaining_scales_ms`, `post_process_ms`
- `dispatch_and_submit_ms` is CPU-side Vulkan command encoding/submission cost, not pure shader execution time.
- `readback_ms` includes waiting for GPU work completion plus readback/map overhead.
- Video profiling prefixes aggregate comparison buckets with `video_`; these are
  sums across decoded frame pairs, while the reported video score is their
  arithmetic mean.

## GPU dispatch constraints

- All shaders use 2D dispatch with `layout(local_size_x=16, local_size_y=16, local_size_z=1)`.
- Dispatch dimensions are `(ceil(width/16), ceil(height/16), 1)`.
- This avoids exceeding `maxComputeWorkgroupsPerDimension` (65535) for large images (e.g. 3200×2400 required 120,000 1D workgroups, which silently failed).

## C++20 proof point

- Keep at least one designated initializer in `src_gpu/dssim-WebGPU.cpp` (for example `ParamsData` / `DecodedInputInfo`) so non-C++20 builds fail early.


