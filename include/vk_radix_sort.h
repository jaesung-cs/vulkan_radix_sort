#ifndef VK_RADIX_SORT_H
#define VK_RADIX_SORT_H

#include <vulkan/vulkan.h>

struct VxSorter_T;

VK_DEFINE_HANDLE(VxSorter)

struct VxSorterCreateInfo {
  VkDevice device;
  uint32_t maxCommandsInFlight;
};

void vxCreateSorter(const VxSorterCreateInfo* pCreateInfo, VxSorter* pSorter);

void vxDestroySorter(VxSorter sorter);

void vxGetSorterBufferSize(uint32_t maxElementCount, VkDeviceSize* size);

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
                    VkBuffer histogramBuffer, VkDeviceSize histogramOffset,
                    VkBuffer lookbackBuffer, VkDeviceSize lookbackOffset,
                    VkBuffer outBuffer, VkDeviceSize outOffset,
                    VkQueryPool queryPool, uint32_t query);

void vxCmdRadixSortGlobalHistogram(VkCommandBuffer commandBuffer,
                                   VxSorter sorter, uint32_t elementCount,
                                   VkBuffer buffer, VkDeviceSize offset,
                                   VkBuffer histogramBuffer,
                                   VkDeviceSize histogramOffset);

void vxCmdRadixSortGlobalHistogramScan(VkCommandBuffer commandBuffer,
                                       VxSorter sorter,
                                       VkBuffer histogramBuffer,
                                       VkDeviceSize histogramOffset);

/**
 * pass: 0, 1, 2, or 3.
 */
void vxCmdRadixSortBinning(VkCommandBuffer commandBuffer, VxSorter sorter,
                           uint32_t elementCount, uint32_t pass,
                           VkBuffer buffer, VkDeviceSize offset,
                           VkBuffer histogramBuffer,
                           VkDeviceSize histogramOffset,
                           VkBuffer lookbackBuffer, VkDeviceSize lookbackOffset,
                           VkBuffer outBuffer, VkDeviceSize outOffset);

#endif  // VK_RADIX_SORT_H
