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
 * VrdxSorter creates internal resources.
 */
VK_DEFINE_HANDLE(VrdxSorter)

struct VrdxSorterLayoutCreateInfo {
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkPipelineCache pipelineCache;
};

struct VrdxSorterCreateInfo {
  VrdxSorterLayout sorterLayout;
  VmaAllocator allocator;
  uint32_t maxElementCount;
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
 * N=15
 * query + 0: start timestamp (VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
 * query + 1: transfer timestamp (VK_PIPELINE_STAGE_TRANSFER_BIT)
 * query + 2 + (3 * i) + 0: upsweep (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
 * query + 2 + (3 * i) + 1: spine (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
 * query + 2 + (3 * i) + 2: downsweep (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
 * query + 14: sort end timestamp (VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
 */
void vrdxCmdSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                 uint32_t elementCount, VkBuffer keysBuffer,
                 VkDeviceSize keysOffset, VkQueryPool queryPool,
                 uint32_t query);

void vrdxCmdSortIndirect(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         VkBuffer indirectBuffer, VkDeviceSize indirectOffset,
                         VkBuffer keysBuffer, VkDeviceSize keysOffset,
                         VkQueryPool queryPool, uint32_t query);

void vrdxCmdSortKeyValue(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         uint32_t elementCount, VkBuffer keysBuffer,
                         VkDeviceSize keysOffset, VkBuffer valuesBuffer,
                         VkDeviceSize valuesOffset, VkQueryPool queryPool,
                         uint32_t query);

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
                                 VrdxSorter sorter, VkBuffer indirectBuffer,
                                 VkDeviceSize indirectOffset,
                                 VkBuffer keysBuffer, VkDeviceSize keysOffset,
                                 VkBuffer valuesBuffer,
                                 VkDeviceSize valuesOffset,
                                 VkQueryPool queryPool, uint32_t query);

#endif  // VK_RADIX_SORT_H
