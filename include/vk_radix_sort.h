#ifndef VK_RADIX_SORT_H
#define VK_RADIX_SORT_H

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

struct VxSorterLayout_T;
struct VxSorter_T;

/**
 * VxSorterLayout creates shared resources, such as pipelines.
 */
VK_DEFINE_HANDLE(VxSorterLayout)

/**
 * VxSorter creates resources per command, such as buffer and descriptor sets.
 */
VK_DEFINE_HANDLE(VxSorter)

struct VxSorterLayoutCreateInfo {
  VkDevice device;

  /**
   * leave it 0 for default=1024.
   * try 256, 512, or 1024.
   * 1024 gives best result, but may depend on device properties.
   */
  uint32_t histogramWorkgroupSize;
};

struct VxSorterCreateInfo {
  VxSorterLayout sorterLayout;
  VmaAllocator allocator;
  uint32_t maxElementCount;
  uint32_t maxCommandsInFlight;
};

void vxCreateSorterLayout(const VxSorterLayoutCreateInfo* pCreateInfo,
                          VxSorterLayout* pSorterLayout);

void vxDestroySorterLayout(VxSorterLayout sorterLayout);

void vxCreateSorter(const VxSorterCreateInfo* pCreateInfo, VxSorter* pSorter);

void vxDestroySorter(VxSorter sorter);

/**
 * if queryPool is not VK_NULL_HANDLE, it writes timestamps to 8 entries
 * [query..query+7].
 * query + 0: start timestamp (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
 * query + 1: histogram end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 2: scan end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 3: binning0 end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 4: binning1 end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 5: binning2 end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 6: binning3 end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 7: sort end timestamp (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
 */
void vxCmdRadixSort(VkCommandBuffer commandBuffer, VxSorter sorter,
                    uint32_t elementCount, VkBuffer buffer, VkDeviceSize offset,
                    VkQueryPool queryPool, uint32_t query);

#endif  // VK_RADIX_SORT_H
