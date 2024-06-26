# vulkan_radix_sort

Vulkan implementation of radix sort.

Reduce-then-scan GPU radix sort algorithm is implemented (Onesweep is abandoned.)


## Requirements
- `VulkanSDK>=1.2`
  - Download from https://vulkan.lunarg.com/ and follow install instruction.
  - Requires several features available in `1.2`.
  - Must support `VK_KHR_buffer_device_address`:
    - Run `vulkaninfo` and check if `VK_KHR_buffer_device_address` device extension is available.
- `cmake>=3.15`


## Dependencies
- `VulkanMemoryAllocator`
  - To avoid conflict with parent project which also depends on a specific version of `VulkanMemoryAllocator`, this library only contains forward declaration.
  - The parent must contains a cpp file with `#define VMA_IMPLEMENTATION`.


## Build and Test
```bash
$ cmake . -B build
$ cmake --build build --config Release -j
$ ./build/Release/bench.exe  # Windows
$ ./build/bench  # Linux
```


### Test Environment
- Windows, NVIDIA GeForce RTX 4090.


## Use as a Library with CMake
- Add `VulkanMemoryAllocator` before addigng `vulkan_radix_sort`
    ```cmake
    add_subdirectory(path/to/VulkanMemoryAllocator)
    add_subdirectory(path/to/vulkan_radix_sort)
    ```

- Link to `vk_radix_sort` in your project (library, binary)
    ```cmake
    target_link_libraries(my_project PRIVATE Vulkan::Vulkan VulkanMemoryAllocator vk_radix_sort)
    ```

## Usage
1. When creating `VkDevice`, enable `VkPhysicalDeviceBufferAddressFeatures`.

1. When creating `VmaAllocator`, enable `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT` flag.

1. Create `VkBuffer` for keys and values, with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` and `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`.

1. Create `VrdxSorterLayout`

    It creates shared resources: pipeline layouts, pipelines, etc.

    ```c++
    VrdxSorterLayout sorterLayout = VK_NULL_HANDLE;
    VrdxSorterLayoutCreateInfo sorterLayoutInfo = {};
    sorterLayoutInfo.physicalDevice = physicalDevice;
    sorterLayoutInfo.device = device;
    vrdxCreateSorterLayout(&sorterLayoutInfo, &sorterLayout);
    ```

1. Create `VrdxSorter` from `VrdxSorterLayout`.

    `VrdxSorter` owns a temporary storage buffer. The size of temporary storage is `2N` for key/value output, plus histogram.

    ```c++
    VrdxSorter sorter = VK_NULL_HANDLE;
    VrdxSorterCreateInfo sorterInfo = {};
    sorterInfo.allocator = allocator;  // VmaAllocator
    sorterInfo.sorterLayout = sorterLayout;
    sorterInfo.maxElementCount = 10000000;
    vrdxCreateSorter(&sorterInfo, &sorter);
    ```

1. Record sort commands.

    This command binds pipeline, pipeline layout, and push constants internally.

    So, users must not expect previously bound targets retain after the sort command.

    Users must add proper **execution barriers**.

    One can use buffer memory barrier, but in general, global barriers are more efficient than per-resource, according to [official synchronization examples](https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples#three-dispatches-first-dispatch-writes-to-one-storage-buffer-second-dispatch-writes-to-a-different-storage-buffer-third-dispatch-reads-both):

    > ... global memory barrier covers all resources. Generally considered more efficient to do a global memory barrier than per-resource barriers, per-resource barriers should usually be used for queue ownership transfers and image layout transitions - otherwise use global barriers.

    The sort command will read from key/value buffers (and elementCount buffer for indirect sort) in compute shader stage, and write to output key/value buffers in later compute shader stage.

    The second synchronization scope **before** sort command must include `COMPUTE_SHADER` stage (and `TRANSFER` for indirect sort) and `SHADER_READ` access (and `TRANSFER_READ` for indirect sort).

    The first synchronization scope **after** sort command must include `COMPUTE_SHADER` stage and `SHADER_WRITE` access.

    ```c++
    VkQueryPool queryPool;  // VK_NULL_HANDLE, or a valid timestamp query pool with size at least 8.

    // sort keys
    vrdxCmdSort(commandBuffer, sorter, elementCount, keysBuffer, 0, queryPool, 0);

    // sort keys with values
    vrdxCmdSortKeyValue(commandBuffer, sorter, elementCount, keysBuffer, 0, valuesBuffer, 0, queryPool, 0);

    // indirectBuffer contains elementCount, a single uint entry in GPU buffer.
    vrdxCmdSortKeyValueIndirect(commandBuffer, sorter, indirectBuffer, 0, keysBuffer, 0, valuesBuffer, 0, queryPool, 0);
    ```


## Current Limitations
- `maxElementCount` $\le$ 10000000.


## TODO
- [x] Use `VkPhysicalDeviceLimits` to get compute shader-related limits, such as `maxComputeWorkGroupSize` or `maxComputeSharedMemorySize`.
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
