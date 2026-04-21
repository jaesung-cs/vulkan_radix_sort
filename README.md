# vulkan_radix_sort

Reduce-then-scan GPU radix sort, implemented as a single-file header-only Vulkan library.
Easily integrates into any Vulkan project without additional dependencies, making it suitable for applications such as 3D Gaussian Splatting rendering.

> **Note:** As of January 2025, this library was competitive with CUB Reduce-then-Scan radix sort.
> Benchmarking in April 2026 against CUDA 13.2 and CUB v3.2.0 (which now defaults to Onesweep) shows CUB is faster by 2.1× on keys-only and 1.3× on key-value at N = 2^25.
> Nevertheless, it remains a practical choice for Vulkan-based applications such as 3D Gaussian Splatting.


## Requirements

- `VulkanSDK >= 1.4.328.1` — download from https://vulkan.lunarg.com/
  - `slangc` is included in VulkanSDK >= 1.3.296.0
  - Push descriptor requires VulkanSDK >= 1.4 (>= 1.4.328.1 for macOS)
- `cmake >= 3.24`


## Benchmark

### Build

```bash
$ cmake . -B build
$ cmake --build build --config Release -j
```

### Run

```bash
$ ./build/Release/bench.exe <type> [output.csv]  # Windows
$ ./build/bench <type> [output.csv]              # Linux
```

- `type`: `cpu`, `vulkan`, `cuda`, `fuchsia`
- Sweeps N from 2^18 to 2^25 (128 steps), 1 warmup + 10 timed runs each
- Outputs median GPU and CPU throughput to CSV

Plot results:
```bash
$ python tools/plot.py vulkan.csv cuda.csv results.png
```

### Results

Test environment: Windows, NVIDIA GeForce RTX 5080, CUDA 13.2, CUB v3.2.0 (Onesweep default).

Median throughput at N = 2^25 (33,554,432 elements). Ratios are relative to this library (> 1× means the competitor is faster).

| Sort type | This library (Vulkan) | Fuchsia (Vulkan) | CUB Onesweep (CUDA) |
|---|---|---|---|
| 32-bit keys only | 10.66 GItems/s | 13.58 GItems/s (1.27×) | 22.40 GItems/s (2.10×) |
| 32-bit key-value | 9.04 GItems/s | 5.02 GItems/s (0.56×) | 11.68 GItems/s (1.29×) |

[Fuchsia radix sort](https://github.com/juliusikkala/fuchsia_radix_sort) is faster on keys-only, but 1.80× slower on key-value. Fuchsia sorts key-value pairs as a single 64-bit key, doubling memory traffic per pass, while this library sorts the two buffers independently.

![Benchmark Result](media/results.png)


## Integration

This is a single header-only library. You can integrate it via CMake or by copying `include/vk_radix_sort.h` directly into your project.

### CMake

1. Add subdirectory:
    ```cmake
    add_subdirectory(path/to/vulkan_radix_sort)
    ```

1. Link to `vk_radix_sort`:
    ```cmake
    # With Volk
    target_link_libraries(my_project PRIVATE volk::volk_headers vk_radix_sort)

    # With standard Vulkan loader
    target_link_libraries(my_project PRIVATE Vulkan::Vulkan vk_radix_sort)
    ```

### Manual

Copy `include/vk_radix_sort.h` into your project and include it directly.


## Usage

1. In exactly one source file, define `VRDX_IMPLEMENTATION` before including `vk_radix_sort.h`.

    If you are using Volk, include `volk.h` first so the library uses Volk's function pointer dispatch. Otherwise it falls back to the standard `<vulkan/vulkan.h>` prototypes.

    ```c++
    // With Volk
    #include "volk.h"
    #define VRDX_IMPLEMENTATION
    #include "vk_radix_sort.h"

    // With standard Vulkan headers
    #define VRDX_IMPLEMENTATION
    #include "vk_radix_sort.h"
    ```

1. Create `VkBuffer` for keys and values with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.

1. Create `VrdxSorter`:

    ```c++
    VrdxSorter sorter = VK_NULL_HANDLE;
    VrdxSorterCreateInfo sorterInfo = {};
    sorterInfo.physicalDevice = physicalDevice;
    sorterInfo.device = device;
    sorterInfo.pipelineCache = pipelineCache;
    vrdxCreateSorter(&sorterInfo, &sorter);
    ```

1. Allocate a temporary storage buffer:

    ```c++
    VrdxSorterStorageRequirements requirements;
    vrdxGetSorterStorageRequirements(sorter, elementCount, &requirements);         // keys only
    vrdxGetSorterKeyValueStorageRequirements(sorter, elementCount, &requirements); // key-value

    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = requirements.size;
    bufferInfo.usage = requirements.usage;
    // ...
    ```

1. Record sort commands.

    Buffer offsets must be multiples of `minStorageBufferOffsetAlignment` (usually `16`).

    The sort command binds its own pipeline, pipeline layout, and push constants — previously bound state is not preserved after the call.

    Add **execution barriers** around the sort. Use global memory barriers rather than per-resource barriers ([Vulkan synchronization examples](https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples#three-dispatches-first-dispatch-writes-to-one-storage-buffer-second-dispatch-writes-to-a-different-storage-buffer-third-dispatch-reads-both)):

    - **Before** the sort: `COMPUTE_SHADER` stage (+ `TRANSFER` for indirect), `SHADER_READ` access (+ `TRANSFER_READ` for indirect).
    - **After** the sort: `COMPUTE_SHADER` stage, `SHADER_WRITE` access.

    ```c++
    VkQueryPool queryPool;  // VK_NULL_HANDLE, or a timestamp query pool with at least 15 entries.

    // Sort keys only
    vrdxCmdSort(commandBuffer, sorter, elementCount,
                keysBuffer, 0,
                storageBuffer, 0,
                queryPool, 0);

    // Sort keys with values
    vrdxCmdSortKeyValue(commandBuffer, sorter, elementCount,
                        keysBuffer, 0,
                        valuesBuffer, 0,
                        storageBuffer, 0,
                        queryPool, 0);

    // Sort with indirect element count (read from GPU buffer)
    // Actual count in indirectBuffer must not exceed maxElementCount.
    vrdxCmdSortKeyValueIndirect(commandBuffer, sorter, maxElementCount,
                                indirectBuffer, 0,
                                keysBuffer, 0,
                                valuesBuffer, 0,
                                storageBuffer, 0,
                                queryPool, 0);
    ```


## Development Guide

After modifying shaders, run the cmake build. It runs `slangc`, generates `src/generated/*.h`, then regenerates `include/vk_radix_sort.h` from `src/vk_radix_sort.h.in` via `tools/generate_header.py`.


## TODO

- [ ] Compare with VkRadixSort.
- [ ] Find optimal `WORKGROUP_SIZE` and `PARTITION_DIVISION` for different devices.


## References

- https://github.com/b0nes164/GPUSorting — CUDA kernel references for understanding the algorithm.


## Troubleshooting

**NVIDIA GPU (Windows): slow performance after a few seconds.**
- Cause: NVIDIA driver downgrades GPU/memory clock under sustained load. Check with the Performance Overlay (Alt+R).
- Fix: set performance mode to maximum in NVIDIA Control Panel.

![](media/performance_mode.jpg)
