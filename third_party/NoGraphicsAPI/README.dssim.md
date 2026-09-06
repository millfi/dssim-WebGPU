# Vendored NoGraphicsAPI

Upstream: https://github.com/sebbbi/NoGraphicsAPI
Revision: e4af4da63bb82907bed0fd6cf28a8608cc9882a8
License: MIT (see LICENSE).

Only the core header, implementation, license, and shader contract are vendored.
The parent project builds the library as C++20; optional utilities are not needed.

Local changes declared in NoGraphicsAPI_vulkan.hpp:
- An optional device-creation callback adds FFmpeg video extensions, features,
  and decode queues to the device selected and owned by NoGraphicsAPI.
- Borrowed Vulkan device/command handles support imported decoded images and
  timestamp queries.
- Native timeline waits/signals join the same NoGraphicsAPI submission.
  NoGraphicsAPI owns submission and command-buffer retirement.
- Failed headless recordings can be discarded without executing them.
  This requires no pending newly created textures.

Compute PSOs, GPU memory, copies, barriers, descriptor heaps, dispatch, and
timeline completion use NoGraphicsAPI. Native Vulkan is limited to FFmpeg
interoperability, device capability queries, and GPU timestamp queries.
Imported video descriptor slots are reused only after comparison completion.
