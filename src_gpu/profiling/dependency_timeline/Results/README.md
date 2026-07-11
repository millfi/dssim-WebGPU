# Dependency timeline result

`gradation-256.timeline.txt` follows
`../Specification/dependency-timeline.ebnf`. Its durations were measured with:

```powershell
& .\build\src_gpu\Release\dssim-WebGPU.exe `
    .\tests\gradation.png `
    .\tests\gradation-256.png `
    --out .\build\dependency_timeline_profile.json `
    --profiling
```

`gradation-256.timeline.svg` renders the measured dependencies and highlights
two overlaps:

- Vulkan timestamp-query execution overlaps CPU-side submit/readback waiting.
- Scale 1-4 CPU aggregation runs on a worker while scale 0 is aggregated on
  the main thread.

The timestamp-query duration is exact, but this profiler does not calibrate the
GPU timestamp clock against the CPU clock. Its horizontal placement inside the
submit/wait window is therefore schematic. CPU wall-clock spans and all
displayed durations are the measured values.

Validate the timeline source with the bundled parser:

```powershell
& node --experimental-strip-types `
    .\src_gpu\profiling\dependency_timeline\Test-program\runnner.ts `
    .\src_gpu\profiling\dependency_timeline\Results\gradation-256.timeline.txt
```
