# FFmpeg sources and builds

`ffmpeg-8.1.2/` is the unmodified upstream release source, vendored as ordinary
files (including its licenses), not a binary distribution or a nested Git checkout.
Source: https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz
Release archive SHA-256: `464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c`.

`tools/build_ffmpeg_minimal.ps1 -Variant Gpu` copies this source into
`ffmpeg-build/gpu/source`, applies the AMD AV1 tile-unit compatibility patch there,
and builds Vulkan Video DLLs in `ffmpeg-gpu-shared` (D3D11VA disabled).

`-Variant Reference` copies the same pristine source into
`ffmpeg-build/reference/source` without that patch and builds D3D11VA DLLs in
`ffmpeg-reference-shared` (Vulkan disabled). Both variants disable static FFmpeg
libraries. Generated source copies, object files, and installed DLLs are ignored.

Never interchange the two sets of DLLs, even though their filenames match.
The application build helpers copy the correct set beside each executable.
The success marker is written only after a successful FFmpeg build/install.
