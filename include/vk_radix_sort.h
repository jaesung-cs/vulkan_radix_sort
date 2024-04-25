#ifndef VK_RADIX_SORT_H
#define VK_RADIX_SORT_H

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

struct VrdxSorterLayout_T;
struct VrdxSorter_T;

/**
 * VrdxSorterLayout creates shared resources, such as pipelines.
 */
VK_DEFINE_HANDLE(VrdxSorterLayout)

/**
 * VrdxSorter creates resources per command, such as buffer and descriptor sets.
 */
VK_DEFINE_HANDLE(VrdxSorter)

typedef enum VrdxSortMethod {
  /**
   * Recommended.
   */
  VRDX_SORT_METHOD_REDUCE_THEN_SCAN = 0,

  /**
   * It seems works only on NVidia GPUs.
   *
   * Not recommended. You can set this on other devices, but may lead to crash.
   */
  VRDX_SORT_METHOD_ONESWEEP = 1,
} VrdxSortMethod;

struct VrdxSorterLayoutCreateInfo {
  VkPhysicalDevice physicalDevice;
  VkDevice device;
};

struct VrdxSorterCreateInfo {
  VrdxSorterLayout sorterLayout;
  VmaAllocator allocator;
  uint32_t maxElementCount;
  uint32_t maxCommandsInFlight;
};

void vrdxCreateSorterLayout(const VrdxSorterLayoutCreateInfo* pCreateInfo,
                            VrdxSorterLayout* pSorterLayout);

void vrdxDestroySorterLayout(VrdxSorterLayout sorterLayout);

void vrdxCreateSorter(const VrdxSorterCreateInfo* pCreateInfo,
                      VrdxSorter* pSorter);

void vrdxDestroySorter(VrdxSorter sorter);

/**
 * if queryPool is not VK_NULL_HANDLE, it writes timestamps to N entries
 * [query..query+N-1].
 *
 * Onesweep: N=8
 * query + 0: start timestamp (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
 * query + 1: histogram end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 2: scan end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 3: binning0 end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 4: binning1 end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 5: binning2 end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 6: binning3 end timestamp (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 7: sort end timestamp (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
 *
 * Reduce-then-scan: N=15
 * query + 0: start timestamp (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
 * query + 1: transfer timestamp (VK_PIPELINE_STAGE_2_TRANSFER_BIT)
 * query + 2 + (3 * i) + 0: upsweep (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 2 + (3 * i) + 1: spine (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 2 + (3 * i) + 2: downsweep (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
 * query + 14: sort end timestamp (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
 */
void vrdxCmdSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                 VrdxSortMethod sortMethod, uint32_t elementCount,
                 VkBuffer buffer, VkDeviceSize offset, VkQueryPool queryPool,
                 uint32_t query);

void vrdxCmdSortIndirect(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         VrdxSortMethod sortMethod, VkBuffer indirectBuffer,
                         VkDeviceSize indirectOffset, VkBuffer buffer,
                         VkDeviceSize offset, VkQueryPool queryPool,
                         uint32_t query);

void vrdxCmdSortKeyValue(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         VrdxSortMethod sortMethod, uint32_t elementCount,
                         VkBuffer buffer, VkDeviceSize offset,
                         VkBuffer valueBuffer, VkDeviceSize valueOffset,
                         VkQueryPool queryPool, uint32_t query);

/**
 * indirectBuffer contains elementCount.
 *
 * The sort command reads a uint32_t value from indirectBuffer at
 * indirectOffset.
 *
 * User must add barrier with second synchronization scope
 * COMPUTE_SHADER stage and SHADER_READ access.
 *
 * indirectBuffer requires TRANSFER_SRC buffer usage flag.
 */
void vrdxCmdSortKeyValueIndirect(VkCommandBuffer commandBuffer,
                                 VrdxSorter sorter, VrdxSortMethod sortMethod,
                                 VkBuffer indirectBuffer,
                                 VkDeviceSize indirectOffset, VkBuffer buffer,
                                 VkDeviceSize offset, VkBuffer valueBuffer,
                                 VkDeviceSize valueOffset,
                                 VkQueryPool queryPool, uint32_t query);

#endif  // VK_RADIX_SORT_H
