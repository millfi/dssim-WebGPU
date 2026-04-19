# AGENTS

## Build policy

- This repository must build as C++20 by default.
- Do not require users to pass an extra C++ standard flag in build commands.
- CMake sets this at the top level (`CMAKE_CXX_STANDARD=20`, `CMAKE_CXX_STANDARD_REQUIRED=ON`, `CMAKE_CXX_EXTENSIONS=OFF`).

## Verification workflow

1. Configure with normal command:
   - `cmake -S . -B build`
2. Build target:
   - `cmake --build build --config Release --target dssim_webgpu`
3. Run executable:
   - `build\src_gpu\Release\dssim-WebGPU.exe original.png modified.png`

## Current priority: Performance optimization

- Score-matching is done. All test pairs in `tests/test_list.csv` are within 0.2% relative error of the reference.
- Current priority is **reducing end-to-end latency** while keeping scores within the regression tolerance.

### Regression tolerance

- Same-image comparison must still produce `0.00000000`.
- All other test pairs in `tests/test_list.csv`: **relative error < 1%** against the reference.
- After every optimization, re-run all pairs and confirm the tolerance holds before committing.
- Do not edit the `reference_score(dssim v3.4.0)` column by hand.

### Permitted precision trade-offs

- Replacing f64 with f32 in CPU-side aggregation is allowed if it stays within tolerance.
- Algebraically equivalent transformations that differ under floating-point arithmetic (e.g. `a*b + a*c` → `a*(b+c)`, reordering FMA, strength reduction) are allowed.
- Reducing intermediate precision (e.g. f32 → f16 for non-critical paths) is allowed if it stays within tolerance.
- Changing blur kernel weights or SSIM constants is **not** allowed.

### Optimization workflow

1. Identify the bottleneck using `--profiling` or `--out <json>` (see Profiling output section).
2. Hypothesize an optimization.
3. Implement the change.
4. Build: `cmake --build build --config Release --target dssim_webgpu`
5. Re-run all `tests/test_list.csv` pairs and confirm regression tolerance.
6. Measure timing improvement with `--profiling`.
7. If improved without regression: commit with a clear message describing the optimization and measured speedup.
8. If regression: revert and try a different approach.
9. Repeat.

### Known bottlenecks and opportunities

- The preprocess shader (`lab_preprocess.wgsl`) runs the full LAB conversion (including `cbrt_poly`) for **every neighbor** in the 5×5 chroma blur — 25 LAB conversions per pixel per image. Consider converting all pixels to LAB first, then blurring in a separate pass.
- Each scale level currently round-trips buffers through CPU (GPU → readback → re-upload). Keeping data on-GPU across scales would eliminate this overhead.
- The downsample, preprocess, and stage0 are separate dispatches with separate buffer allocations. Fusing passes or reusing buffers could reduce overhead.
- Pipeline/PSO creation is done once per session and reused across `--stdin-pairs`, but buffer creation/upload happens per comparison.

## Score-matching workflow (reference — complete)

- Treat `tests/test_list.csv` as the regression list.
- For each validation pass:
  1. Build `dssim-WebGPU`.
  2. Run the image pairs listed in `tests/test_list.csv`.
  3. Update the `dssim-WebGPU` score column with the newly measured scores.
- Use `--out <json>` for per-scale inspection and `--debug-dump-dir` for buffer-level investigation.

## Reference implementation

- Prefer using `src_reference/target/release/dssim.exe` for validation over `dssim.exe` on PATH.
- Falling back to the PATH binary is acceptable only when it is known to match the checked out source/version.
- The reference source under `src_reference/` can be read to understand algorithmic details when needed.

## Profiling output

- When `--profiling` is specified, MECE profiling buckets are printed in milliseconds:
  - `pipeline_setup_ms`: shader module + pipeline layout + PSO creation
  - `resource_prep_ms`: buffer creation + buffer upload + bind group creation
  - `gpu_execution_ms`: dispatch/submit + readback/map wait
  - `cpu_postprocess_ms`: CPU-side score aggregation
  - `other_ms`: uncategorized overhead
- When `--out <json>` is specified, finer-grained timing is in the `profiling` object:
  - `create_shader_module_ms`, `create_pso_ms`, `create_buffer_ms`, `write_input_buffer_ms`, `create_pipeline_layout_ms`, `create_bind_group_ms`, `dispatch_and_submit_ms`, `readback_ms`, `post_process_ms`
- `dispatch_and_submit_ms` is CPU-side command encoding/submission cost, not pure WGSL kernel time.
- `readback_ms` includes waiting for GPU work completion plus readback/map overhead.

## GPU dispatch constraints

- All shaders use 2D dispatch with `@workgroup_size(16, 16, 1)`.
- Dispatch dimensions are `(ceil(width/16), ceil(height/16), 1)`.
- This avoids exceeding `maxComputeWorkgroupsPerDimension` (65535) for large images (e.g. 3200×2400 required 120,000 1D workgroups, which silently failed).

## C++20 proof point

- Keep at least one designated initializer in `src_gpu/dawn_checksum.cpp` (for example `ParamsData` / `DecodedInputInfo`) so non-C++20 builds fail early.


