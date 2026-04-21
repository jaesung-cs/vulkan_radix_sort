# Changelog

## v0.3.1
- Added Fuchsia radix sort benchmark.
- Volk integration simplified: include `volk.h` before `vk_radix_sort.h` instead of defining `VRDX_USE_VOLK`.

## v0.3.0
- Benchmark sweeps N linearly from 2^18 to 2^25 (128 steps), measuring both keys-only and key-value sort.
- Benchmark outputs CSV with GPU and CPU wall-clock throughput per (N, sort type).
- Added `tools/plot.py` for plotting benchmark results with matplotlib.
- Header generation automated via `src/vk_radix_sort.h.in` template and `tools/generate_header.py`.

## v0.2.2
- Use `minStorageBufferOffsetAlignment` instead of fixed `16` (@hypengw)

## v0.2.1
- Volk is optional. Control which Vulkan header to use with the `VRDX_USE_VOLK` definition.

## v0.2.0
- Single-file header-only library.
- Requires VulkanSDK >= 1.4 for push descriptor.
- Requires Volk.

## v0.1.0
- Use `VK_KHR_push_descriptor` instead of `VK_KHR_buffer_device_address`.
  - Buffers no longer need `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`.
  - Buffer offsets must be multiples of `minStorageBufferOffsetAlignment` (usually `16`).
  - Users of previous versions need to update device creation accordingly.
- Migrate from GLSL to Slang shader language for extensibility to other graphics APIs.

## v0.0.0
- First publish.
