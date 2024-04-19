# vulkan_radix_sort
Vulkan implementation of radix sort.

State-of-the-art GPU radix sort algorithm, [Onesweep (Link to NVidia Research)](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus), is implemented.


## Requirements
- `VulkanSDK>=1.3`
  - Download from https://vulkan.lunarg.com/ and follow install instruction.
  - Requires several features available in `1.3`.
- `cmake>=3.24`
  - `Vulkan::shaderc_combined` new in version `3.24`.


## Usage
1. When creating `VkDevice`, enable `VK_KHR_maintenance4` and `VK_KHR_synchronization2` device features.

1. Create `VxSorterLayout`

    It creates shared resources: descriptor layouts, pipeline layouts, pipelines, etc.

    ```c++
    VxSorterLayout sorterLayout = VK_NULL_HANDLE;
    VxSorterLayoutCreateInfo sorterLayoutInfo = {};
    sorterLayoutInfo.device = device_;
    sorterLayoutInfo.histogramWorkgroupSize = 1024;
    vxCreateSorterLayout(&sorterLayoutInfo, &sorterLayout);
    ```

1. Create `VxSorter` from `VxSorterLayout`.

    `VxSorter` owns a temporary storage buffer. The size of temporary storage is `2N` for key/value output, plus histogram.

    It also create its own descriptor pool and descriptor sets of size equal to `maxCommandsInFlight`.

    ```c++
    VxSorterCreateInfo sorterInfo = {};
    sorterInfo.allocator = allocator_;
    sorterInfo.sorterLayout = sorterLayout_;
    sorterInfo.maxElementCount = 10000000;
    sorterInfo.maxCommandsInFlight = 2;
    vxCreateSorter(&sorterInfo, &sorter_);
    ```

1. Record sort commands.

    This command binds pipeline, pipeline layout, descriptors, and push constants internally.

    So, users must not expect previously bound targets retain after the sort command.

    User must add proper barriers for key/value buffers.

    The second synchronization scope **before** sort command for **key/value buffer** must include `COMPUTE_SHADER` stage and `SHADER_READ` access.

    The second synchronization scope **before** sort command for **indirect buffer** must include `TRANSFER` stage and `TRANSFER_READ` access.

    The first synchronization scope **after** sort command for **key/value buffer** must include `COMPUTE_SHADER` stage and `SHADER_WRITE` access.

    The first synchronization scope **after** sort command for **indirect buffer** must include `TRANSFER` stage and `TRANSFER_READ` access.

    ```c++
    VkQueryPool queryPool;  // VK_NULL_HANDLE, or a valid timestamp query pool with size at least 8.
    vxCmdRadixSort(commandBuffer, sorter, elementCount, keysBuffer, 0, queryPool, 0);
    vxCmdRadixSortKeyValue(commandBuffer, sorter, elementCount, keysBuffer, 0, valuesBuffer, 0, queryPool, 0);

    // indirectBuffer contains elementCount, a single uint entry in GPU buffer.
    vxCmdRadixSortKeyValueIndirect(commandBuffer, sorter, indirectBuffer, 0, keysBuffer, 0, valuesBuffer, 0, queryPool, 0);
    ```


## Current Limitations
- `maxElementCount` $\le$ 10000000.


## TODO
- [ ] Use `VkPhysicalDeviceLimits` to get compute shader-related limits, such as `maxComputeWorkGroupSize` or `maxComputeSharedMemorySize`.
- [ ] Increase allowed `maxElementCount` by allocating buffers properly.
- [ ] Compare with CUB radix sort
- [ ] Compare with VkRadixSort


## References
- https://github.com/b0nes164/GPUSorting : their CUDA kernel codes were very helpful when trying to catch the idea of how the algorithm works.


## Troubleshooting
- (NVIDIA GPU, Windows) Slow runtime after a few seconds.
  - Reason: NVidia driver adjust GPU/Memory clock.
    Open Performance Overlay (Alt+R), then you will see GPU/Memory Clock gets down.
  - Solution: change performance mode in control panel.
    ![](media/performance_mode.jpg)
