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

## Score-matching workflow

- Current priority is score agreement with the reference `dssim` implementation, not GPU-side performance.
- Treat `tests/test_list.csv` as the source of truth for the current mismatch set and the main verification loop.
- For each validation pass:
  1. Build `dssim-WebGPU`.
  2. Run the image pairs listed in `tests/test_list.csv`.
  3. Update the `dssim-WebGPU` score column in `tests/test_list.csv` with the newly measured scores.
- During score-matching work, prefer using `--out <json>` so the agent can inspect decoded input metadata, per-scale results, and profiling in a stable machine-readable format.
- `--debug-dump-dir` and the JSON `debug_dumps` section are useful when score mismatches need buffer-level investigation; keep them available for correctness work even if they are not needed in normal runs.
- The following cases are highest priority and must be fixed before performance work:
  - comparing the same image must produce `0.00000000`
  - `gradation.png` vs `gradation-fs8.png` must no longer have a large relative error
- Do not edit the `reference_score(dssim v3.4.0)` column by hand unless the reference implementation/version is intentionally changed.

## Reference implementation

- Prefer keeping a local `src_reference/` workspace for the upstream `dssim` source used during score-matching work.
- If both source and a locally built reference binary are available under `src_reference/`, prefer using that binary for validation so the agent can inspect and rebuild the exact implementation it is comparing against.
- Falling back to the `dssim.exe` found on `PATH` is acceptable only when it is known to match the checked out reference source/version.
- When score differences are being investigated, read the reference source first before changing the GPU math.

## Profiling output

- `DispatchAndSubmit processing time` is CPU-side command encoding/submission cost, not pure WGSL kernel time.
- `Readback processing time` includes waiting for GPU work completion plus readback/map overhead.
- When `--out <json>` is specified, the same aggregated profiling values are emitted under top-level `profiling`.

## C++20 proof point

- Keep at least one designated initializer in `src_gpu/dawn_checksum.cpp` (for example `ParamsData` / `DecodedInputInfo`) so non-C++20 builds fail early.


